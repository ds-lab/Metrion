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
