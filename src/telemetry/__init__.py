"""Telemetry module with graceful degradation."""
from src.telemetry.tracing import (
    init_telemetry,
    get_tracer,
    traced,
    shutdown_telemetry
)

__all__ = [
    "init_telemetry",
    "get_tracer",
    "traced",
    "shutdown_telemetry"
]
