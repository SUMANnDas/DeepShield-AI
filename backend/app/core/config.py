"""Environment-backed settings for the API foundation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "DeepShield AI"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    docs_enabled: bool = True

    @classmethod
    def from_environment(cls) -> "Settings":
        # Do not override values explicitly supplied by the host process.
        load_dotenv(override=False)

        environment = os.getenv("DEEPSHIELD_ENVIRONMENT", "development").strip().lower()
        default_docs_enabled = environment == "development"
        api_prefix = os.getenv("DEEPSHIELD_API_PREFIX", "/api/v1").strip()

        if not api_prefix.startswith("/") or api_prefix.endswith("/"):
            raise ValueError("DEEPSHIELD_API_PREFIX must start with '/' and omit a trailing '/'")

        log_level = os.getenv("DEEPSHIELD_LOG_LEVEL", "INFO").strip().upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ValueError("DEEPSHIELD_LOG_LEVEL must be a standard logging level")

        return cls(
            app_name=os.getenv("DEEPSHIELD_APP_NAME", "DeepShield AI").strip(),
            environment=environment,
            log_level=log_level,
            api_prefix=api_prefix,
            docs_enabled=_read_bool("DEEPSHIELD_DOCS_ENABLED", default_docs_enabled),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide application settings."""
    return Settings.from_environment()
