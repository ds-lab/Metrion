import threading
import time
from queue import Empty
from typing import Any

from loguru import logger


def drain_until_sentinel(
    queue: Any,
    sentinel: Any,
    deadline_s: float,
    poll_timeout: float = 0.1,
    name: str = "queue",
) -> tuple[list[Any], bool]:
    """
    Drain a queue until the sentinel arrives or the deadline expires.

    Parameters
    ----------
    queue : queue.Queue | multiprocessing.Queue
        The queue to drain.
    sentinel : Any
        Value marking the end of the stream. Compared with ``==``, not ``is``,
        because multiprocessing serialises it and identity does not survive.
    deadline_s : float
        Maximum total time to wait, in seconds.
    poll_timeout : float
        Blocking time of a single ``get``, in seconds.
    name : str
        Label for the warning logged when the deadline is hit.

    Returns
    -------
    tuple[list[Any], bool]
        The items read (sentinel excluded) and whether the sentinel was seen.
    """
    items: list[Any] = []
    end = time.monotonic() + deadline_s

    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            logger.warning(
                f"drain_until_sentinel({name}): deadline of {deadline_s}s reached, "
                f"{len(items)} items read, sentinel NOT seen"
            )
            return items, False

        try:
            item = queue.get(timeout=min(poll_timeout, remaining))
        except Empty:
            continue

        if item == sentinel:
            return items, True
        items.append(item)


def drain_remaining(
    queue: Any,
    idle_timeout: float = 0.5,
    deadline_s: float = 5.0,
    name: str = "queue",
) -> list[Any]:
    """
    Drain whatever is still in a queue, without waiting for a sentinel.

    Parameters
    ----------
    queue : queue.Queue | multiprocessing.Queue
        The queue to drain.
    idle_timeout : float
        Return once nothing has arrived for this many seconds.
    deadline_s : float
        Hard upper bound on the total time spent, in seconds.
    name : str
        Label for the warning logged when the deadline is hit.

    Returns
    -------
    list[Any]
        The items read.
    """
    items: list[Any] = []
    end = time.monotonic() + deadline_s

    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            logger.warning(f"drain_remaining({name}): deadline of {deadline_s}s reached, {len(items)} items read")
            return items

        try:
            items.append(queue.get(timeout=min(idle_timeout, remaining)))
        except Empty:
            return items


def close_and_join_feeder(queue: Any, timeout_s: float = 10.0, name: str = "queue") -> bool:
    """
    Close a multiprocessing queue and wait for its feeder thread to flush.

    Falls back to ``cancel_join_thread`` when the timeout expires, which drops whatever
    is still buffered. That only happens when nothing is reading the other end.

    Parameters
    ----------
    queue : multiprocessing.Queue
        The queue to close.
    timeout_s : float
        Maximum time to wait for the feeder thread, in seconds.
    name : str
        Label for the warning logged when the timeout is hit.

    Returns
    -------
    bool
        Whether the buffered data was flushed.
    """
    queue.close()

    joiner = threading.Thread(target=queue.join_thread, daemon=True)
    joiner.start()
    joiner.join(timeout_s)

    if joiner.is_alive():
        logger.warning(f"close_and_join_feeder({name}): feeder did not flush within {timeout_s}s, dropping buffer")
        queue.cancel_join_thread()
        return False
    return True
