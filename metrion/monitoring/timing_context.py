from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import ClassVar


@dataclass
class TimingContext:
    """Immutable timing info from an energy data source."""

    source_name: str  # e.g., "intel_rapl", "nvidia_smi"
    start_time_ms: int  # Epoch milliseconds when readings started
    interval_seconds: float  # Collection interval (e.g., 1.0, 3.0)
    start_datetime: datetime  # Human-readable start time


class TimingContextRegistry:
    """Thread-safe singleton registry for timing contexts."""

    _instance: ClassVar["TimingContextRegistry | None"] = None
    _lock: ClassVar[Lock] = Lock()

    @classmethod
    def instance(cls) -> "TimingContextRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self._contexts: dict[str, TimingContext] = {}
        self._ctx_lock = Lock()

    def register(self, context: TimingContext) -> None:
        """Register a timing context from an energy data source."""
        with self._ctx_lock:
            self._contexts[context.source_name] = context

    def get(self, source_name: str) -> TimingContext | None:
        """Get timing context by source name."""
        with self._ctx_lock:
            return self._contexts.get(source_name)

    def get_all(self) -> dict[str, TimingContext]:
        """Get all registered timing contexts."""
        with self._ctx_lock:
            return dict(self._contexts)
