import time

import pytest

from metrion.util import boot_epoch_s

pytestmark = pytest.mark.skipif(not hasattr(time, "CLOCK_BOOTTIME"), reason="CLOCK_BOOTTIME is Linux only")


def test_boot_epoch_plus_boottime_lands_on_the_wall_clock():
    """BPF stamps CLOCK_BOOTTIME; converting it has to hit the wall clock, not a fraction of a second off."""
    wall = time.time()
    since_boot = time.clock_gettime(time.CLOCK_BOOTTIME)

    assert abs(boot_epoch_s() + since_boot - wall) < 0.010
