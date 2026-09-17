"""Provider-facing retrieval services with safe, typed outcomes."""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Generic, Sequence, TypeVar, Union


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class RetrievalErrorCode(str, Enum):
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    MISSING_INDEX = "missing_index"
    DIMENSION = "dimension"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER = "provider"


@dataclass(frozen=True)
class RetrievalSuccess(Generic[T]):
    value: T


@dataclass(frozen=True)
class RetrievalError:
    code: RetrievalErrorCode
    user_message: str


RetrievalResult = Union[RetrievalSuccess[T], RetrievalError]


class _MalformedResponseError(Exception):
    pass


_MESSAGES = {
    RetrievalErrorCode.AUTHENTICATION: "The medical knowledge service is not configured correctly. Please contact support.",
    RetrievalErrorCode.TIMEOUT: "The medical knowledge service took too long to respond. Please try again.",
    RetrievalErrorCode.RATE_LIMIT: "The medical knowledge service is busy. Please wait a moment and try again.",
    RetrievalErrorCode.MISSING_INDEX: "The medical knowledge source is unavailable. Please contact support.",
    RetrievalErrorCode.DIMENSION: "The medical knowledge source has an incompatible configuration. Please contact support.",
    RetrievalErrorCode.MALFORMED_RESPONSE: "The medical knowledge service returned an unusable response. Please try again.",
    RetrievalErrorCode.PROVIDER: "The medical knowledge service is temporarily unavailable. Please try again later.",
}


def _classify_exception(exc: Exception) -> RetrievalErrorCode:
    """Classify without importing provider-specific exception packages."""
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    status = getattr(exc, "status_code", getattr(exc, "status", None))
    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in message:
        return RetrievalErrorCode.TIMEOUT
    if ("authentication" in name or "unauthorized" in name or "forbidden" in name
            or "invalid api key" in message or status in (401, 403)):
        return RetrievalErrorCode.AUTHENTICATION
    if ("ratelimit" in name or "rate limit" in message or "too many requests" in message
            or status == 429):
        return RetrievalErrorCode.RATE_LIMIT
    if ("notfound" in name or "index not found" in message or "does not exist" in message
            or status == 404):
        return RetrievalErrorCode.MISSING_INDEX
    if "dimension" in name or "dimension" in message:
        return RetrievalErrorCode.DIMENSION
    return RetrievalErrorCode.PROVIDER


def _failure(operation: str, exc: Exception) -> RetrievalError:
    code = _classify_exception(exc)
    # Deliberately exclude exception messages, queries, vectors, credentials, and responses.
    LOGGER.warning(
        "retrieval_failed operation=%s category=%s exception_type=%s",
        operation,
        code.value,
        type(exc).__name__,
    )
    return RetrievalError(code, _MESSAGES[code])


def retrieve_text(query: str, vectorstore: object, *, top_k: int = 4) -> RetrievalResult[Sequence[object]]:
    """Retrieve text documents while containing provider errors at this boundary."""
    try:
        documents = vectorstore.similarity_search(query, k=top_k)  # type: ignore[attr-defined]
        if not isinstance(documents, (list, tuple)):
            raise _MalformedResponseError
        return RetrievalSuccess(documents)
    except Exception as exc:
        if isinstance(exc, _MalformedResponseError):
            LOGGER.warning("retrieval_failed operation=text category=malformed_response exception_type=%s", type(exc).__name__)
            return RetrievalError(RetrievalErrorCode.MALFORMED_RESPONSE, _MESSAGES[RetrievalErrorCode.MALFORMED_RESPONSE])
        return _failure("text", exc)


def retrieve_image(vector: Sequence[float], index: object, *, expected_dimension: int) -> RetrievalResult[str]:
    """Return the closest disease label for an image embedding."""
    if len(vector) != expected_dimension:
        LOGGER.warning("retrieval_failed operation=image category=dimension actual=%d expected=%d", len(vector), expected_dimension)
        return RetrievalError(RetrievalErrorCode.DIMENSION, _MESSAGES[RetrievalErrorCode.DIMENSION])
    try:
        response = index.query(vector=list(vector), top_k=1, include_metadata=True)  # type: ignore[attr-defined]
        matches = response.get("matches") if hasattr(response, "get") else getattr(response, "matches", None)
        if not matches:
            raise _MalformedResponseError
        match = matches[0]
        metadata = match.get("metadata") if hasattr(match, "get") else getattr(match, "metadata", None)
        disease = metadata.get("Disease") if hasattr(metadata, "get") else None
        if not isinstance(disease, str) or not disease.strip():
            raise _MalformedResponseError
        return RetrievalSuccess(disease.strip())
    except Exception as exc:
        if isinstance(exc, _MalformedResponseError):
            LOGGER.warning("retrieval_failed operation=image category=malformed_response exception_type=%s", type(exc).__name__)
            return RetrievalError(RetrievalErrorCode.MALFORMED_RESPONSE, _MESSAGES[RetrievalErrorCode.MALFORMED_RESPONSE])
        return _failure("image", exc)
