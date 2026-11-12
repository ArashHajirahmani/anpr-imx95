"""Detector -> OCR workers -> ordered publisher.

```text
latest-frame slots -> detector thread --job queue--> N OCR workers
                                |                         |
                                +--- empty frames --------+--> sequencer -> publisher
```

* The detector thread is the only owner of the TFLite interpreter. It runs the detector
  and nothing else CPU-heavy.
* OCR workers run ROI refine, enhance, quality gate, and OCR on CPU, in parallel.
* Every frame the detector takes gets a sequence number. The publisher hands frames to
  `ReadPublisher` strictly in that order, so the temporal vote, history, and the UI see
  the same stream they saw with the single-threaded loop.
* The job queue is bounded. When all workers are busy the detector waits, and capture
  keeps only the newest frame in each slot, so stale frames are dropped, not queued.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass

from anpr.pipeline import LatestFrameSlot
from anpr.publisher import ReadPublisher
from anpr.recognize import DetectedFrame, InferenceStats, OcrOutcome, _rss_mb

logger = logging.getLogger("anpr.pipeline")
_GAP_SECONDS = 5.0


@dataclass(frozen=True)
class _Job:
    seq: int
    detected: DetectedFrame


@dataclass(frozen=True)
class _Done:
    detected: DetectedFrame
    outcome: OcrOutcome | None
    failed: bool = False


class StagedRecognizer:
    def __init__(
        self,
        slots: dict[str, LatestFrameSlot],
        stop: threading.Event,
        reader: object,
        fps_limit: int,
        workers: int,
        publisher: ReadPublisher,
    ) -> None:
        self._slots = slots
        self._stop = stop
        self._reader = reader
        self._interval = 1.0 / fps_limit
        self._workers = workers
        self._publisher = publisher
        self._jobs: queue.Queue[_Job] = queue.Queue(maxsize=workers)
        self._cond = threading.Condition()
        self._done: dict[int, _Done] = {}
        self._next = 1

    def threads(self) -> list[threading.Thread]:
        threads = [
            threading.Thread(target=self._detect_loop, name="plate-detector", daemon=True),
            threading.Thread(target=self._publish_loop, name="plate-publisher", daemon=True),
        ]
        threads.extend(
            threading.Thread(target=self._ocr_loop, name=f"plate-ocr-{index}", daemon=True)
            for index in range(1, self._workers + 1)
        )
        return threads

    def _finish(self, seq: int, done: _Done) -> None:
        with self._cond:
            self._done[seq] = done
            self._cond.notify_all()

    def _detect_loop(self) -> None:
        seq = 0
        next_tick = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_tick:
                if self._stop.wait(next_tick - now):
                    return
                continue
            next_tick = now + self._interval
            for name, slot in self._slots.items():
                frame = slot.take()
                if frame is None or frame.width <= 0 or frame.height <= 0:
                    continue
                seq += 1
                try:
                    detected = self._reader.detect(frame)
                except Exception as exc:
                    logger.warning("plate detect failed for %s: %s", name, exc)
                    self._finish(
                        seq, _Done(DetectedFrame(frame, None, (), "cpu", 0.0, 0.0), None, True)
                    )
                    continue
                if not detected.boxes:
                    self._finish(seq, _Done(detected, OcrOutcome([], 0.0, 0.0)))
                    continue
                if not self._submit(_Job(seq, detected)):
                    return

    def _submit(self, job: _Job) -> bool:
        while not self._stop.is_set():
            try:
                self._jobs.put(job, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _ocr_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                outcome = self._reader.recognize(job.detected)
                done = _Done(job.detected, outcome)
            except Exception as exc:
                logger.warning(
                    "plate recognize failed for %s: %s", job.detected.frame.stream_name, exc
                )
                done = _Done(job.detected, None, True)
            self._finish(job.seq, done)

    def _publish_loop(self) -> None:
        waiting_since: float | None = None
        while not self._stop.is_set():
            with self._cond:
                done = self._done.pop(self._next, None)
                if done is None:
                    self._cond.wait(0.2)
                    if self._done and min(self._done) > self._next:
                        waiting_since = waiting_since or time.monotonic()
                        if time.monotonic() - waiting_since > _GAP_SECONDS:
                            logger.warning("frame %d never finished; skipping it", self._next)
                            self._next += 1
                            waiting_since = None
                    continue
                self._next += 1
                waiting_since = None
            self._publish(done)

    def _publish(self, done: _Done) -> None:
        if done.failed or done.outcome is None:
            return
        detected = done.detected
        outcome = done.outcome
        wall = max((detected.det_ms + outcome.ocr_ms) / 1000, 1e-6)
        stats = InferenceStats(
            runtime=detected.runtime,
            det_ms=detected.det_ms,
            ocr_ms=outcome.ocr_ms,
            cpu_percent=(detected.cpu_s + outcome.cpu_s) / wall * 100,
            rss_mb=_rss_mb(),
        )
        frame = detected.frame
        self._publisher.handle(frame.stream_name, frame, outcome.reads, stats)
