"""Backend-measured system and pipeline facts for the web page.

Everything here is read from the running process and ``/proc``; the page formats it and
never guesses. Pure stdlib; every reader degrades to ``None`` when a source is missing.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

CACHE_SECONDS = 1.0
BUSY_WINDOW_SECONDS = 3.0
_THERMAL = Path("/sys/class/thermal")


class BusyMeter:
    """Fraction of wall time a resource was busy, over a short sliding window."""

    def __init__(self, window: float = BUSY_WINDOW_SECONDS) -> None:
        self._window = window
        self._lock = threading.Lock()
        self._spans: deque[tuple[float, float]] = deque()  # (end, duration)

    def add(self, seconds: float, now: float | None = None) -> None:
        end = time.monotonic() if now is None else now
        with self._lock:
            self._spans.append((end, max(0.0, seconds)))
            self._trim(end)

    def percent(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._trim(now)
            busy = sum(duration for _, duration in self._spans)
        return min(100.0, 100.0 * busy / self._window)

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        while self._spans and self._spans[0][0] < cutoff:
            self._spans.popleft()


def parse_cpu_times(text: str) -> tuple[float, float] | None:
    """(busy, total) jiffies from the aggregate ``cpu`` line of ``/proc/stat``."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            fields = [float(value) for value in line.split()[1:]]
            if len(fields) < 4:
                return None
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
            total = sum(fields[:8])
            return total - idle, total
    return None


def parse_meminfo(text: str) -> tuple[int, int] | None:
    """(total_kb, available_kb) from ``/proc/meminfo``."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(rest.split()[0])
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    return values["MemTotal"], values["MemAvailable"]


def parse_process_jiffies(text: str) -> float | None:
    """utime + stime of a process from ``/proc/<pid>/stat``."""
    _, _, tail = text.rpartition(")")
    fields = tail.split()
    if len(fields) < 13:
        return None
    return float(fields[11]) + float(fields[12])


def parse_vmrss_kb(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


# PMIC zones report a fixed trip value and Wi-Fi is not the SoC; neither says how hot the chip is.
_NOT_SOC = ("pf", "nxp-wifi", "wifi")


def _soc_temperature() -> float | None:
    """Hottest SoC thermal zone in degrees C, ignoring PMIC and Wi-Fi sensors."""
    best: float | None = None
    try:
        zones = sorted(_THERMAL.glob("thermal_zone*"))
    except OSError:
        return None
    for zone in zones:
        try:
            kind = (zone / "type").read_text().strip().lower()
            value = int((zone / "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        if kind.startswith(_NOT_SOC):
            continue
        best = value if best is None else max(best, value)
    return best


class SystemMonitor:
    """Samples CPU, memory, temperature and NPU load; snapshots are cached for a second."""

    def __init__(self, npu_meter: BusyMeter | None = None) -> None:
        self._npu = npu_meter
        self._lock = threading.Lock()
        self._cores = os.cpu_count() or 1
        self._clock_hz = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0
        self._last_cpu: tuple[float, float] | None = None
        self._last_proc: tuple[float, float] | None = None  # (jiffies, monotonic)
        self._cached: dict[str, object] = {}
        self._cached_at = 0.0

    def snapshot(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            if self._cached and now - self._cached_at < CACHE_SECONDS:
                return self._cached
            self._cached = self._sample(now)
            self._cached_at = now
            return self._cached

    def _sample(self, now: float) -> dict[str, object]:
        out: dict[str, object] = {"cores": self._cores}
        stat = _read("/proc/stat")
        cpu = parse_cpu_times(stat) if stat else None
        if cpu is not None:
            if self._last_cpu is not None and cpu[1] > self._last_cpu[1]:
                busy = cpu[0] - self._last_cpu[0]
                out["cpu_percent"] = round(100.0 * busy / (cpu[1] - self._last_cpu[1]), 1)
            self._last_cpu = cpu
        proc = _read("/proc/self/stat")
        jiffies = parse_process_jiffies(proc) if proc else None
        if jiffies is not None:
            if self._last_proc is not None and now > self._last_proc[1]:
                seconds = (jiffies - self._last_proc[0]) / self._clock_hz
                span = now - self._last_proc[1]
                # Share of the whole machine, like the system figure.
                out["app_cpu_percent"] = round(100.0 * seconds / span / self._cores, 1)
            self._last_proc = (jiffies, now)
        meminfo = _read("/proc/meminfo")
        mem = parse_meminfo(meminfo) if meminfo else None
        if mem is not None:
            out["mem_total_mb"] = round(mem[0] / 1024)
            out["mem_used_mb"] = round((mem[0] - mem[1]) / 1024)
        status = _read("/proc/self/status")
        rss = parse_vmrss_kb(status) if status else None
        if rss is not None:
            out["app_mem_mb"] = round(rss / 1024)
        temperature = _soc_temperature()
        if temperature is not None:
            out["temperature_c"] = round(temperature, 1)
        if self._npu is not None:
            out["npu_busy_percent"] = round(self._npu.percent(now), 1)
        return out
