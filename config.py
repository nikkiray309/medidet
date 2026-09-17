"""Validated runtime configuration for MediDet."""

from dataclasses import dataclass
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when deployment configuration is absent or invalid."""


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    pinecone_api_key: str
    text_index_name: str
    image_index_name: str
    text_embedding_dimension: int
    image_embedding_dimension: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AppConfig":
        required = (
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
            "TEXT_INDEX_NAME",
            "IMAGE_INDEX_NAME",
            "TEXT_EMBEDDING_DIMENSION",
            "IMAGE_EMBEDDING_DIMENSION",
        )
        missing = [name for name in required if not str(values.get(name, "")).strip()]
        if missing:
            raise ConfigurationError(
                "Missing required configuration: " + ", ".join(missing)
            )

        try:
            text_dimension = int(values["TEXT_EMBEDDING_DIMENSION"])
            image_dimension = int(values["IMAGE_EMBEDDING_DIMENSION"])
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("Embedding dimensions must be integers.") from exc
        if text_dimension <= 0 or image_dimension <= 0:
            raise ConfigurationError("Embedding dimensions must be positive.")

        return cls(
            openai_api_key=str(values["OPENAI_API_KEY"]).strip(),
            pinecone_api_key=str(values["PINECONE_API_KEY"]).strip(),
            text_index_name=str(values["TEXT_INDEX_NAME"]).strip(),
            image_index_name=str(values["IMAGE_INDEX_NAME"]).strip(),
            text_embedding_dimension=text_dimension,
            image_embedding_dimension=image_dimension,
        )
