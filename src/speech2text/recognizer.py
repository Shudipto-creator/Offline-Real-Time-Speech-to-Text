"""
Streaming speech recognition using sherpa-onnx.

Wraps the OnlineRecognizer API for streaming transducer recognition
with endpoint detection.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .logging_setup import get_logger

logger = get_logger("speech2text.recognizer")


@dataclass
class RecognitionResult:
    """Result from a recognition step."""

    partial_text: str = ""
    final_text: str = ""
    is_endpoint: bool = False
    timestamp: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0


class Recognizer:
    """Streaming speech recognizer using sherpa-onnx OnlineRecognizer."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._recognizer = None
        self._stream = None
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        """Load the recognition model. Called on a background thread."""
        from sherpa_onnx import online_recognizer as or_mod

        model_paths = self.config.model.resolved_paths()
        endpoint_cfg = self.config.endpoint
        pipeline_cfg = self.config.pipeline

        logger.info("Loading model from: %s", model_paths["encoder"].parent)

        decoding_method = pipeline_cfg.decoding_method
        if decoding_method == "greedy":
            decoding_method = "greedy_search"
        elif decoding_method == "modified_beam_search":
            decoding_method = "modified_beam_search"

        self._recognizer = or_mod.OnlineRecognizer.from_transducer(
            tokens=str(model_paths["tokens"]),
            encoder=str(model_paths["encoder"]),
            decoder=str(model_paths["decoder"]),
            joiner=str(model_paths["joiner"]),
            num_threads=pipeline_cfg.num_threads,
            sample_rate=self.config.audio.sample_rate,
            feature_dim=80,
            enable_endpoint_detection=pipeline_cfg.enable_endpoint,
            rule1_min_trailing_silence=endpoint_cfg.trailing_silence_seconds,
            rule2_min_trailing_silence=endpoint_cfg.short_trailing_silence_seconds,
            rule3_min_utterance_length=endpoint_cfg.max_utterance_length_seconds,
            decoding_method=decoding_method,
            max_active_paths=pipeline_cfg.max_active_paths,
            provider=pipeline_cfg.provider,
        )
        self._stream = self._recognizer.create_stream()
        self._loaded = True
        logger.info("Model loaded successfully")

    def is_loaded(self) -> bool:
        return self._loaded

    def feed(self, samples: np.ndarray) -> RecognitionResult:
        """Feed audio samples to the recognizer.

        Args:
            samples: float32 audio at the configured sample rate (16000 Hz).

        Returns:
            RecognitionResult with partial/final text and endpoint status.
        """
        if not self._loaded or self._stream is None:
            return RecognitionResult()

        start = time.time()
        with self._lock:
            samples = np.asarray(samples, dtype=np.float32)
            if samples.ndim == 2:
                if samples.shape[1] == 1:
                    samples = samples[:, 0]
                else:
                    samples = samples.mean(axis=1)
            elif samples.ndim != 1:
                samples = samples.reshape(-1)
            samples = np.ascontiguousarray(samples)

            self._stream.accept_waveform(self.config.audio.sample_rate, samples)
            
            # CRITICAL: Must run decode loop after accept_waveform
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            
            is_endpoint = self._recognizer.is_endpoint(self._stream)
            result_text = self._recognizer.get_result(self._stream)

            final_text = result_text if is_endpoint else ""
            if is_endpoint:
                self._recognizer.reset(self._stream)

        elapsed_ms = (time.time() - start) * 1000
        return RecognitionResult(
            partial_text=result_text,
            final_text=final_text,
            is_endpoint=is_endpoint,
            processing_time_ms=elapsed_ms,
        )

    def flush(self) -> RecognitionResult:
        """Flush the final utterance on Stop."""
        if not self._loaded or self._stream is None:
            return RecognitionResult()

        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        result_text = self._recognizer.get_result(self._stream)
        self._recognizer.reset(self._stream)

        return RecognitionResult(
            final_text=result_text,
            is_endpoint=True,
        )

    def reset(self) -> None:
        """Reset the recognition stream."""
        if self._loaded and self._stream is not None:
            with self._lock:
                self._recognizer.reset(self._stream)

    def recreate_stream(self) -> None:
        """Create a fresh recognition stream."""
        if self._recognizer is not None:
            with self._lock:
                self._stream = self._recognizer.create_stream()

    def cleanup(self) -> None:
        """Release recognizer resources."""
        with self._lock:
            self._stream = None
            self._recognizer = None
            self._loaded = False