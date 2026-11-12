from anpr.sysmon import (
    BusyMeter,
    SystemMonitor,
    parse_cpu_times,
    parse_meminfo,
    parse_process_jiffies,
    parse_vmrss_kb,
)


def test_cpu_times_split_busy_from_idle() -> None:
    text = "cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 1 1 1 1 1 1 1 1 1\n"
    busy, total = parse_cpu_times(text)
    assert total == 1000 and busy == 150
    assert parse_cpu_times("intr 5\n") is None


def test_meminfo_and_process_files() -> None:
    assert parse_meminfo("MemTotal: 8000 kB\nMemFree: 1 kB\nMemAvailable: 6000 kB\n") == (
        8000,
        6000,
    )
    assert parse_meminfo("MemTotal: 8000 kB\n") is None
    stat = "42 (python3 x) S 1 2 3 4 5 6 7 8 9 10 120 30 0 0 20 0 4 0 0"
    assert parse_process_jiffies(stat) == 150.0
    assert parse_vmrss_kb("Name: x\nVmRSS:\t  2048 kB\n") == 2048


def test_busy_meter_reports_share_of_the_window_and_forgets_old_spans() -> None:
    meter = BusyMeter(window=2.0)
    meter.add(0.5, now=10.0)
    meter.add(0.5, now=11.0)
    assert meter.percent(now=11.0) == 50.0
    assert meter.percent(now=14.0) == 0.0


def test_monitor_reports_local_facts_without_an_npu() -> None:
    monitor = SystemMonitor()
    first = monitor.snapshot()
    assert first["cores"] >= 1
    assert "npu_busy_percent" not in first
    assert monitor.snapshot() is first  # cached within a second
    meter = BusyMeter()
    meter.add(0.3)
    assert "npu_busy_percent" in SystemMonitor(meter).snapshot()


def test_soc_temperature_ignores_pmic_and_wifi_zones(tmp_path, monkeypatch) -> None:
    import anpr.sysmon as sysmon

    for index, (kind, milli) in enumerate(
        [
            ("a55-thermal", 62340),
            ("ana-thermal", 60100),
            ("pf53_soc", 105000),
            ("nxp-wifi-tz", 90000),
        ]
    ):
        zone = tmp_path / f"thermal_zone{index}"
        zone.mkdir()
        (zone / "type").write_text(kind + "\n")
        (zone / "temp").write_text(f"{milli}\n")
    monkeypatch.setattr(sysmon, "_THERMAL", tmp_path)
    assert sysmon._soc_temperature() == 62.34
    monkeypatch.setattr(sysmon, "_THERMAL", tmp_path / "missing")
    assert sysmon._soc_temperature() is None
