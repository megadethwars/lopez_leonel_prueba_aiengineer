"""Chat en terminal:  python -m tiendahogar_agent  (con PYTHONPATH=src)."""

import os

from .logging_config import setup_logging

# En la terminal los logs ensucian la conversación: por defecto solo WARNING+ (LOG_LEVEL=INFO para verlos).
setup_logging(os.getenv("LOG_LEVEL") or "WARNING")

from .adapters.inbound.cli import run_cli  # noqa: E402
from .bootstrap import build_support_agent  # noqa: E402

if __name__ == "__main__":
    run_cli(build_support_agent())
