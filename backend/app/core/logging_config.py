"""Safe, minimal process logging configuration."""

import logging


def configure_logging(level_name: str) -> None:
    """Configure standard logging without recording media or secrets."""
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {level_name}")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger().setLevel(level)
