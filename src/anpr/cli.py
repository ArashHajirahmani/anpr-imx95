"""Command line entry points for the ANPR service."""

import argparse
import logging
import signal
import threading
from pathlib import Path

from anpr.config import ConfigError, load_config
from anpr.infer import probe_delegates
from anpr.pipeline import run_streams

logger = logging.getLogger("anpr")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anpr")
    subcommands = parser.add_subparsers(dest="command", required=True)

    check_config = subcommands.add_parser("check-config", help="validate a JSON config")
    check_config.add_argument("--config", type=Path, required=True)

    run = subcommands.add_parser("run", help="capture configured streams until stopped")
    run.add_argument("--config", type=Path, required=True)

    subcommands.add_parser("probe-npu", help="search for the TFLite Neutron delegate")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "check-config":
        return _check_config(args.config)
    if args.command == "run":
        return _run(args.config)
    if args.command == "probe-npu":
        return _probe_npu()
    parser.error(f"unknown command: {args.command}")
    return 2


def _check_config(path: Path) -> int:
    try:
        config = load_config(path)
    except (OSError, ConfigError) as exc:
        logger.error("%s", exc)
        return 1
    enabled = ", ".join(stream.name for stream in config.enabled_streams()) or "(none)"
    logger.info("config ok: %s", path)
    logger.info("enabled streams: %s", enabled)
    logger.info("detector: %s", config.detector.path)
    logger.info("recognizer enabled: %s", config.recognizer.enabled)
    return 0


def _run(path: Path) -> int:
    try:
        config = load_config(path)
    except (OSError, ConfigError) as exc:
        logger.error("%s", exc)
        return 1
    stop = threading.Event()

    def _stop(signum: int, _frame: object) -> None:
        logger.info("received signal %s", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    wired = run_streams(config, stop)
    for name, pipeline in wired.pipelines.items():
        logger.info("stream %s stopped: %s", name, pipeline)
    return 0


def _probe_npu() -> int:
    probe = probe_delegates()
    if probe.candidates:
        for candidate in probe.candidates:
            logger.info("delegate candidate: %s", candidate)
    else:
        logger.info("no delegate candidates found")
    if probe.device_present:
        logger.info("device node present: %s", probe.device_node)
    else:
        logger.info("device node absent: %s", probe.device_node)
    if probe.selected is None:
        logger.error("TFLite Neutron delegate was not found")
        return 1
    logger.info("selected TFLite delegate: %s", probe.selected)
    return 0
