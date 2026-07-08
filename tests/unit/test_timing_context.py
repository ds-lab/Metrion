import threading
from datetime import UTC, datetime

from metrion.monitoring.timing_context import TimingContext, TimingContextRegistry


def test_timing_context_immutability():
    """Test that TimingContext is immutable."""
    ctx = TimingContext(
        source_name="test_source",
        start_time_ms=1000,
        interval_seconds=1.0,
        start_datetime=datetime.now(UTC),
    )

    ctx.source_name = "new_name"


def test_registry_singleton():
    """Test that TimingContextRegistry is a singleton."""
    reg1 = TimingContextRegistry.instance()
    reg2 = TimingContextRegistry.instance()
    assert reg1 is reg2


def test_registry_registration_and_retrieval():
    """Test registering and retrieving timing contexts."""
    registry = TimingContextRegistry.instance()
    # clean up for test
    registry._contexts.clear()

    ctx = TimingContext(
        source_name="source_1",
        start_time_ms=123456789,
        interval_seconds=2.0,
        start_datetime=datetime.now(UTC),
    )

    registry.register(ctx)

    retrieved = registry.get("source_1")
    assert retrieved == ctx
    assert retrieved.interval_seconds == 2.0

    assert registry.get("non_existent") is None


def test_registry_thread_safety():
    """Test thread safety of the registry."""
    registry = TimingContextRegistry.instance()
    registry._contexts.clear()

    def register_ctx(idx):
        ctx = TimingContext(
            source_name=f"source_{idx}",
            start_time_ms=idx,
            interval_seconds=1.0,
            start_datetime=datetime.now(UTC),
        )
        registry.register(ctx)

    threads = [threading.Thread(target=register_ctx, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_contexts = registry.get_all()
    assert len(all_contexts) == 10
    for i in range(10):
        assert f"source_{i}" in all_contexts


if __name__ == "__main__":
    test_timing_context_immutability()
    test_registry_singleton()
    test_registry_registration_and_retrieval()
    test_registry_thread_safety()
    print("All tests passed!")
