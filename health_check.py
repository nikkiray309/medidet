"""Deployment health check; only index metadata is inspected."""

import os
from typing import Any, Mapping

from config import AppConfig, ConfigurationError


def check_indexes(config: AppConfig, pinecone_client: object) -> tuple[bool, list[str]]:
    checks: list[str] = []
    healthy = True
    for purpose, name, expected in (
        ("text", config.text_index_name, config.text_embedding_dimension),
        ("image", config.image_index_name, config.image_embedding_dimension),
    ):
        try:
            description = pinecone_client.describe_index(name)  # type: ignore[attr-defined]
            dimension = description.get("dimension") if hasattr(description, "get") else getattr(description, "dimension", None)
            if dimension != expected:
                healthy = False
                checks.append(f"{purpose} index dimension mismatch (expected {expected}, found {dimension})")
            else:
                checks.append(f"{purpose} index is configured with dimension {expected}")
        except Exception as exc:
            healthy = False
            checks.append(f"{purpose} index check failed ({type(exc).__name__})")
    return healthy, checks


def main(values: Mapping[str, Any] = os.environ) -> int:
    try:
        config = AppConfig.from_mapping(values)
    except ConfigurationError as exc:
        print(f"UNHEALTHY: {exc}")
        return 1
    from pinecone import Pinecone

    healthy, checks = check_indexes(config, Pinecone(api_key=config.pinecone_api_key))
    for check in checks:
        print(("OK: " if "failed" not in check and "mismatch" not in check else "ERROR: ") + check)
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
