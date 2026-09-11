import hashlib
import math
import os
import sys
from datetime import UTC, datetime
from typing import override

import polars as pl
import psutil

from metrion.entity.entity import Entity


class LogicalEntity(Entity):
    """
    Represents a task/thread accross different operating systems.

    Attributes
    ----------
    id : str
        Unique task/thread identifier (alias for thread_id).
    thread_id : str
        Hash of (native_thread_id, start_time, host_id).
    pid_id : str
        Hash of (native_pid, pid_start_time, host_id).
    parent_pid_id : str
        Hash of (native_parent_pid, parent_pid_start_time, host_id).
    parent_tid_id: str
        Hash of (native_parent_thread_id, parent_tid_start_time, host_id).
    """

    BOOT_TIME: datetime = datetime.fromtimestamp(psutil.boot_time(), tz=UTC)
    HZ: int = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    DEFAULT_HOST_ID: str = os.environ.get("METRION_HOST_ID", "")

    def __init__(
        self,
        native_thread_id: int,
        native_pid: int,
        native_parent_pid: int,
        native_parent_thread_id: int,
        executable_name: str,
        cmdline: str,
        start_time: datetime,
        pid_start_time: datetime,
        parent_tid_start_time: datetime,
        parent_pid_start_time: datetime,
        resource_group_id: int | None = None,
        resource_group_name: str | None = None,
        user_id: int | None = None,
        group_id: int | None = None,
    ):
        """
        Initialize a Process instance.

        Parameters
        ----------
        native_thread_id : int
            OS-native thread id.
        native_pid : int
            OS-native process id (Linux tgid, Windows PID, macOS pid).
        native_parent_pid : int
            OS-native parent PID.
        native_parent_thread_id : int
            OS-native parent thread id.
        executable_name : str
            Executable name (Linux comm, Windows image name).
        cmdline : str
            Full command line.
        start_time : datetime
            Timestamp for this task/thread.
        pid_start_time : datetime
            Start time of the process (thread group leader).
        parent_tid_start_time : datetime
            Start time of the parent thread.
        parent_pid_start_time : datetime
            Start time of the parent process.
        resource_group_id : int, optional
            Resource group id (cgroup id, job object id, container id).
        resource_group_name : str, optional
            Resource group name (cgroup path, job object name, container name).
        user_id : int, optional
            User id (uid / security identifier).
        group_id : int, optional
            Primary group id (gid).
        """
        # Host identification (from parameter or environment variable)
        self._host_id = self.DEFAULT_HOST_ID

        # Identity & hierarchy (native OS values)
        self._native_thread_id = native_thread_id
        self._native_pid = native_pid
        self._native_parent_thread_id = native_parent_thread_id
        self._native_parent_pid = native_parent_pid

        # Names & image. The threads of a process share these strings; interned, they hold one copy.
        self._executable_name = sys.intern(executable_name.strip()) if executable_name else ""
        self._cmdline = sys.intern(cmdline.strip()) if cmdline else ""

        # Resource grouping / "container"
        self._resource_group_id = resource_group_id
        self._resource_group_name = sys.intern(resource_group_name.strip()) if resource_group_name else None

        # Timing
        self._start_time = start_time
        self._process_start_time = pid_start_time
        self._parent_start_time = parent_tid_start_time
        self._parent_process_start_time = parent_pid_start_time

        # Security / user info
        self._user_id = user_id
        self._group_id = group_id

        # Generate derived IDs
        self._thread_id = self._generate_thread_id()
        self._pid_id = self._generate_pid_id()
        self._parent_tid_id = self._generate_parent_tid_id()
        self._parent_pid_id = self._generate_parent_pid_id()

        super().__init__(id=self._thread_id, parent_id=self._parent_pid_id)

    def _to_df(self) -> pl.DataFrame:
        """
        Convert the Process to a Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame representation of the Process.
        """
        schema = {
            "id": pl.String,
            "host_id": pl.String,
            # native ids
            "native_pid": pl.Int64,
            "native_thread_id": pl.Int64,
            "native_parent_pid": pl.Int64,
            "native_parent_thread_id": pl.Int64,
            # derived ids
            "thread_id": pl.String,
            "pid_id": pl.String,
            "parent_tid_id": pl.String,
            "parent_pid_id": pl.String,
            # timing
            "start_time": pl.Datetime("ns"),
            "pid_start_time": pl.Datetime("ns"),
            "parent_tid_start_time": pl.Datetime("ns"),
            "parent_pid_start_time": pl.Datetime("ns"),
            # names
            "executable_name": pl.String,
            "cmdline": pl.String,
            # resource grouping
            "resource_group_id": pl.Int64,
            "resource_group_name": pl.String,
            # security
            "user_id": pl.Int64,
            "group_id": pl.Int64,
        }
        data = {
            "id": [self.id],
            "host_id": [self.host_id],
            # derived ids
            "thread_id": [self.thread_id],
            "pid_id": [self.pid_id],
            "parent_tid_id": [self.parent_tid_id],
            "parent_pid_id": [self.parent_pid_id],
            # native ids
            "native_thread_id": [self.native_thread_id],
            "native_pid": [self.native_pid],
            "native_parent_pid": [self.native_parent_pid],
            "native_parent_thread_id": [self.native_parent_thread_id],
            # names
            "executable_name": [self.executable_name],
            "cmdline": [self.cmdline],
            # resource grouping
            "resource_group_id": [self.resource_group_id],
            "resource_group_name": [self.resource_group_name],
            # timing
            "start_time": [self.start_time],
            "pid_start_time": [self.pid_start_time],
            "parent_tid_start_time": [self.parent_tid_start_time],
            "parent_pid_start_time": [self.parent_pid_start_time],
            # security
            "user_id": [self.user_id],
            "group_id": [self.group_id],
        }
        return pl.DataFrame(data, schema=schema)

    # Host identification
    @property
    def host_id(self) -> str:
        """Unique identifier of the machine."""
        return self._host_id

    # Identity & hierarchy (derived hashed IDs)
    @property
    def thread_id(self) -> str:
        """Thread identifier: hash(native_thread_id, start_time, host_id)."""
        return self._thread_id

    @property
    def pid_id(self) -> str:
        """Process identifier: hash(native_pid, pid_start_time, host_id)."""
        return self._pid_id

    @property
    def parent_tid_id(self) -> str | None:
        """Parent thread identifier: hash(native_parent_thread_id, parent_tid_start_time, host_id)."""
        return self._parent_tid_id

    @property
    def parent_pid_id(self) -> str | None:
        """Parent process identifier: hash(native_parent_pid, parent_tid_start_time, host_id)."""
        return self._parent_pid_id

    # Identity & hierarchy (native OS values)
    @property
    def native_thread_id(self) -> int:
        """OS-native thread id."""
        return self._native_thread_id

    @property
    def native_pid(self) -> int:
        """OS-native process id (Linux tgid, Windows PID, macOS pid)."""
        return self._native_pid

    @property
    def native_parent_thread_id(self) -> int:
        """OS-native parent thread id."""
        return self._native_parent_thread_id

    @property
    def native_parent_pid(self) -> int:
        """OS-native parent PID."""
        return self._native_parent_pid

    # Names & image
    @property
    def executable_name(self) -> str:
        """Executable name (Linux comm, Windows image name)."""
        return self._executable_name

    @executable_name.setter
    def executable_name(self, value: str):
        self._executable_name = sys.intern(value) if value else value

    @property
    def cmdline(self) -> str | None:
        """Full command line."""
        return self._cmdline

    @cmdline.setter
    def cmdline(self, value: str | None):
        self._cmdline = sys.intern(value) if value else value

    # Resource grouping
    @property
    def resource_group_id(self) -> int | None:
        """Resource group id (cgroup id, job object id, container id)."""
        return self._resource_group_id

    @property
    def resource_group_name(self) -> str | None:
        """Resource group name (cgroup path, job object name, container name)."""
        return self._resource_group_name

    # Timing
    @property
    def start_time(self) -> datetime:
        """Timestamp for this task/thread."""
        return self._start_time

    @property
    def pid_start_time(self) -> datetime:
        """Start time of the process (thread group leader)."""
        return self._process_start_time

    @property
    def parent_tid_start_time(self) -> datetime:
        """Start time of the parent thread."""
        return self._parent_start_time

    @property
    def parent_pid_start_time(self) -> datetime:
        """Start time of the parent process."""
        return self._parent_process_start_time

    # Security / user info
    @property
    def user_id(self) -> int | None:
        """User id (uid / security identifier)."""
        return self._user_id

    @property
    def group_id(self) -> int | None:
        """Primary group id (gid)."""
        return self._group_id

    @override
    def _generate_id(self) -> str:
        """Generate unique id (same as thread_id)."""
        return self._generate_thread_id()

    def _generate_thread_id(self) -> str:
        """Generate thread id: hash(native_thread_id, start_time, host_id)."""
        unique_string = f"{self._native_thread_id}{self._normalize_timestamp(self._start_time)}{self._host_id}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _generate_pid_id(self) -> str:
        """Generate process id: hash(native_pid, pid_start_time, host_id)."""
        unique_string = f"{self._native_pid}{self._normalize_timestamp(self._process_start_time)}{self._host_id}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _generate_parent_tid_id(self) -> str | None:
        """Generate parent thread id: hash(native_parent_thread_id, parent_tid_start_time, host_id)."""
        unique_string = (
            f"{self._native_parent_thread_id}{self._normalize_timestamp(self._parent_start_time)}{self._host_id}"
        )
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _generate_parent_pid_id(self) -> str | None:
        """Generate parent process id: hash(native_parent_pid, parent_tid_start_time, host_id)."""
        unique_string = (
            f"{self._native_parent_pid}{self._normalize_timestamp(self._parent_process_start_time)}{self._host_id}"
        )
        return hashlib.md5(unique_string.encode()).hexdigest()

    @override
    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.id}, native_pid={self.native_pid}, "
            f"executable_name={self.executable_name})"
        )

    def _normalize_timestamp(self, ts: datetime, round_mode: str = "trunc") -> int:
        """
        Convert a datetime timestamp into an integer tick count since boot.

        Parameters
        ----------
        ts : datetime
            Timestamp to normalize.
        round_mode : {'trunc', 'round'}, optional
            Rounding mode: 'trunc' (default) or 'round'.

        Returns
        -------
        int
            Number of ticks elapsed since boot.

        Raises
        ------
        ValueError
            If round_mode is not 'trunc' or 'round'.
        """
        if ts is None:
            ts = self.BOOT_TIME

        # Align naïve timestamps to UTC to match BOOT_TIME tz-awareness.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        since_boot = (ts - self.BOOT_TIME).total_seconds()
        raw_ticks = since_boot * self.HZ

        if round_mode == "round":
            return int(math.floor(raw_ticks + 0.5))
        elif round_mode == "trunc":
            return int(raw_ticks)
        else:
            raise ValueError("round_mode must be 'round' or 'trunc'")
