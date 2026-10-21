"""
Microphone audio capture using sounddevice.

Uses sounddevice.InputStream with a callback that copies samples into a
bounded queue. Recognition and GUI work happen off the audio callback thread.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from .config import AudioConfig

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class AudioTestResult:
    """Result of a short microphone test capture."""

    sample_rate: int
    channels: int
    peak_level: float
    is_silent: bool
    duration_seconds: float
    device_name: str = ""


class AudioCapture:
    """Captures microphone audio and feeds it into a bounded queue."""

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.config = config or AudioConfig()
        self._queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=self.config.max_queue_blocks
        )
        self._stream: Optional[sd.InputStream] = None
        self._dropped_blocks: int = 0
        self._running: bool = False
        self._lock = threading.Lock()
        self._native_sample_rate: int = self.config.sample_rate
        self._resample_ratio: float = 1.0
        self._resample_state = None  # For scipy.signal.resample_poly state

    @property
    def dropped_blocks(self) -> int:
        return self._dropped_blocks

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def get_block(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Retrieve one audio block from the queue."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        """Discard all pending audio blocks."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def preferred_device_index(self) -> Optional[int]:
        """Prefer a real hardware microphone over the generic Windows mapper."""
        try:
            devices = sd.query_devices()
        except Exception:
            return self.config.device_index

        preferred: list[int] = []
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) <= 0:
                continue
            name = str(device.get("name", "")).lower()
            if "sound mapper" in name or "primary sound capture driver" in name:
                continue
            preferred.append(index)

        if preferred:
            return preferred[0]

        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                return index

        return self.config.device_index

    def _resample(self, data: np.ndarray) -> np.ndarray:
        """Resample from native sample rate to target sample rate (16000 Hz)."""
        if self._resample_ratio == 1.0:
            return data
        if not HAS_SCIPY:
            # Fallback: simple linear interpolation (not ideal but works)
            target_len = int(len(data) * self._resample_ratio)
            return np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)),
                data
            ).astype(np.float32)
        # Use scipy for quality resampling
        return scipy.signal.resample_poly(data, up=16000, down=self._native_sample_rate).astype(np.float32)

    def _to_mono(self, data: np.ndarray) -> np.ndarray:
        """Convert to mono float32."""
        if data.ndim == 2:
            if data.shape[1] == 1:
                return data[:, 0].astype(np.float32)
            return data.mean(axis=1).astype(np.float32)
        return data.astype(np.float32)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback - resample to 16kHz mono and enqueue."""
        if status:
            if status.input_overflow:
                self._dropped_blocks += 1
        
        # Convert to mono float32 at native rate
        mono = self._to_mono(indata)
        
        # Resample to target sample rate (16000 Hz)
        resampled = self._resample(mono)
        
        try:
            self._queue.put(resampled, block=False)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._dropped_blocks += 1
            self._queue.put(resampled, block=False)

    def start(self, device_index: Optional[int] = None) -> None:
        """Start microphone capture at device native rate, resample to 16kHz."""
        with self._lock:
            if self._running:
                return

            device = device_index if device_index is not None else self.config.device_index
            if device is None:
                device = self.preferred_device_index()
            self.config.device_index = device

            # Query device native sample rate and channels
            try:
                device_info = sd.query_devices(device)
                if isinstance(device_info, dict):
                    self._native_sample_rate = int(device_info.get("default_sample_rate", self.config.sample_rate))
                    native_channels = device_info.get("max_input_channels", self.config.channels)
                else:
                    self._native_sample_rate = self.config.sample_rate
                    native_channels = self.config.channels
            except Exception:
                self._native_sample_rate = self.config.sample_rate
                native_channels = self.config.channels

            # Calculate resample ratio
            self._resample_ratio = 16000.0 / self._native_sample_rate
            
            # Block size at native rate
            block_size = int(self._native_sample_rate * self.config.block_duration_ms / 1000)

            self._stream = sd.InputStream(
                samplerate=self._native_sample_rate,
                blocksize=block_size,
                channels=native_channels,
                dtype="float32",
                device=device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True

    def stop(self) -> None:
        """Stop microphone capture and release the stream safely."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def test_microphone(self, duration_seconds: float = 5.0,
                         device_index: Optional[int] = None) -> AudioTestResult:
        """Capture a short test clip and report basic metrics.

        Does NOT save audio to disk.
        """
        device_name = ""
        try:
            dev = sd.query_devices(device_index)
            if isinstance(dev, dict):
                device_name = dev.get("name", "")
        except Exception:
            pass

        sample_rate = self.config.sample_rate
        channels = self.config.channels
        block_size = int(sample_rate * 0.1)
        num_blocks = int(duration_seconds / 0.1)

        peak = 0.0
        total_samples = 0

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                blocksize=block_size,
                channels=channels,
                dtype="float32",
                device=device_index,
            ) as stream:
                for _ in range(num_blocks):
                    data, _ = stream.read(block_size)
                    if data is not None and len(data) > 0:
                        peak = max(peak, float(np.max(np.abs(data))))
                        total_samples += len(data)
        except Exception:
            pass

        is_silent = peak < 0.01
        return AudioTestResult(
            sample_rate=sample_rate,
            channels=channels,
            peak_level=peak,
            is_silent=is_silent,
            duration_seconds=duration_seconds,
            device_name=device_name,
        )

    def list_devices(self) -> list[dict]:
        """Return input-capable audio devices."""
        devices = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": i,
                        "name": d.get("name", f"Device {i}"),
                        "channels": d.get("max_input_channels", 0),
                    })
        except Exception:
            pass
        return devices