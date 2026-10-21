"""
Worker pipeline that coordinates audio capture and recognition.

Runs recognition on a dedicated worker thread and sends structured events
to a thread-safe GUI queue.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from .audio import AudioCapture
from .config import Config
from .logging_setup import get_logger
from .recognizer import Recognizer

logger = get_logger("speech2text.pipeline")


class EventType(Enum):
    """Types of events sent from the worker to the GUI."""

    PARTIAL = "partial"
    FINAL = "final"
    STATUS = "status"
    WARNING = "warning"
    ERROR = "error"
    DIAGNOSTIC = "diagnostic"


@dataclass
class PipelineEvent:
    """A structured event from the recognition pipeline."""

    type: EventType
    text: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0


class RecognitionWorker:
    """Consumes audio blocks and produces recognition events."""

    def __init__(
        self,
        config: Config,
        gui_queue: queue.Queue,
        on_event: Optional[Callable[[PipelineEvent], None]] = None,
    ) -> None:
        self.config = config
        self.gui_queue = gui_queue
        self.on_event = on_event

        self.audio = AudioCapture(config.audio)
        self.recognizer = Recognizer(config)

        self._running = False
        self._stop_requested = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_audio_time = time.monotonic()
        self._last_no_audio_status = ""
        self._no_audio_error_emitted = False

        # Diagnostics
        self.dropped_blocks = 0
        self.total_blocks = 0
        self.finalization_deadline: float = 0.0

    def _emit(self, event: PipelineEvent) -> None:
        """Send an event to the GUI queue and callback."""
        try:
            self.gui_queue.put(event, block=False)
        except queue.Full:
            logger.warning("GUI queue full, dropping event")
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                logger.exception("on_event callback failed")

    def _run_recognition(self) -> None:
        """Main worker loop: consume audio, run recognition, emit events."""
        self.dropped_blocks = 0
        self.total_blocks = 0

        self._emit(PipelineEvent(type=EventType.STATUS, message="Loading model..."))
        try:
            self.recognizer.load()
        except Exception as e:
            self._running = False
            self._emit(PipelineEvent(type=EventType.ERROR, message=f"Model load failed: {e}"))
            return

        self._emit(PipelineEvent(type=EventType.STATUS, message="Listening"))

        while self._running:
            block = self.audio.get_block(timeout=0.1)
            if block is None:
                if time.monotonic() - self._last_audio_time > 3.0:
                    status = "Listening - waiting for audio"
                    if status != self._last_no_audio_status:
                        self._emit(PipelineEvent(type=EventType.STATUS, message=status))
                        self._last_no_audio_status = status
                if time.monotonic() - self._last_audio_time > 3.0 and not self._no_audio_error_emitted:
                    self._emit(PipelineEvent(type=EventType.ERROR, message="No audio from microphone. Check device selection and Windows microphone permissions (Settings > Privacy & security > Microphone)."))
                    self._no_audio_error_emitted = True
                continue

            self._last_audio_time = time.monotonic()
            self._last_no_audio_status = ""
            self._no_audio_error_emitted = False
            self.total_blocks += 1
            self.dropped_blocks = self.audio.dropped_blocks

            try:
                result = self.recognizer.feed(block)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._emit(PipelineEvent(type=EventType.ERROR, message=f"Recognition error: {e}"))
                continue

            if result.partial_text:
                self._emit(PipelineEvent(
                    type=EventType.PARTIAL,
                    text=result.partial_text,
                    processing_time_ms=result.processing_time_ms,
                ))

            if result.is_endpoint and result.final_text:
                self._emit(PipelineEvent(
                    type=EventType.FINAL,
                    text=result.final_text,
                    processing_time_ms=result.processing_time_ms,
                    timestamp=result.timestamp,
                ))

        # Shutdown: flush final utterance
        self._shutdown()

    def _shutdown(self) -> None:
        """Stop accepting audio, close stream, flush remaining audio."""
        self._emit(PipelineEvent(type=EventType.STATUS, message="Stopping"))

        # Stop the audio stream first
        self.audio.stop()

        # Allow a short bounded period for queued audio to finish
        drain_deadline = time.time() + 2.0  # 2 second drain window
        while time.time() < drain_deadline:
            block = self.audio.get_block(timeout=0.05)
            if block is None:
                break
            try:
                self.recognizer.feed(block)
            except Exception:
                break

        # Flush final utterance
        try:
            result = self.recognizer.flush()
            if result.final_text:
                self._emit(PipelineEvent(
                    type=EventType.FINAL,
                    text=result.final_text,
                ))
        except Exception as e:
            logger.warning("Flush failed: %s", e)

        # Reset for next session
        self.recognizer.reset()
        self.recognizer.recreate_stream()
        self.audio.clear_queue()

        self._emit(PipelineEvent(type=EventType.STATUS, message="Idle"))

    def start(self) -> None:
        """Start the recognition worker."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_requested = False
            self._thread = threading.Thread(target=self._run_recognition, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the recognition worker and shut down cleanly."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_requested = True

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Worker thread did not stop cleanly")
            self._thread = None

    def start_audio(self, device_index: Optional[int] = None) -> None:
        """Start microphone capture."""
        self.audio.start(device_index=device_index)

    def stop_audio(self) -> None:
        """Stop microphone capture."""
        self.audio.stop()

    def cleanup(self) -> None:
        """Release all resources."""
        self.stop()
        self.audio.stop()
        self.recognizer.cleanup()
        self.audio.clear_queue()