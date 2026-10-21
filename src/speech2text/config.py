"""
Configuration for the offline speech-to-text application.

All user-adjustable settings live here. Paths are resolved relative to
the project root so the app works from any working directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# Project root: two levels up from this file (src/speech2text/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelConfig:
    """Paths to the sherpa-onnx model files."""

    encoder: str = "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx"
    decoder: str = "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.int8.onnx"
    joiner: str = "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx"
    tokens: str = "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"

    def resolved_paths(self) -> dict[str, Path]:
        """Return paths resolved against the project root."""
        return {
            "encoder": PROJECT_ROOT / self.encoder,
            "decoder": PROJECT_ROOT / self.decoder,
            "joiner": PROJECT_ROOT / self.joiner,
            "tokens": PROJECT_ROOT / self.tokens,
        }


@dataclass
class AudioConfig:
    """Microphone capture settings."""

    sample_rate: int = 16000
    channels: int = 1
    block_duration_ms: int = 100  # ~100 ms blocks
    device_index: int | None = None  # None = default input device
    max_queue_blocks: int = 100


@dataclass
class EndpointConfig:
    """Endpoint detection tuning."""

    # Trailing silence in seconds for the "long" rule
    trailing_silence_seconds: float = 2.4
    # Trailing silence in seconds for the "short" rule
    short_trailing_silence_seconds: float = 1.2
    # Maximum utterance length in seconds
    max_utterance_length_seconds: float = 20.0


@dataclass
class PipelineConfig:
    """Recognition pipeline settings."""

    num_threads: int = 2
    provider: str = "cpu"  # cpu, cuda
    max_active_paths: int = 4
    decoding_method: str = "greedy"  # greedy, modified_beam_search
    enable_endpoint: bool = True
    language_mode: str = "auto"  # auto, chinese, english


@dataclass
class GuiConfig:
    """Tkinter UI settings."""

    poll_interval_ms: int = 50
    max_transcript_lines: int = 200
    window_title: str = "Offline Speech-to-Text"
    window_width: int = 700
    window_height: int = 600


@dataclass
class RefinementConfig:
    """Optional post-processing refinement settings."""

    enabled: bool = False
    model_size: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"
    min_text_length: int = 4


@dataclass
class Config:
    """Top-level application configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a Config from a plain dict, ignoring unknown keys."""
        known = {f.name for f in __import__("dataclasses").fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**{k: _build_nested(v) for k, v in filtered.items()})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load config from a JSON file. Returns defaults if file missing."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> list[str]:
        """Validate config and return a list of error messages."""
        errors: list[str] = []
        model_paths = self.model.resolved_paths()
        for name, path in model_paths.items():
            if not path.exists():
                errors.append(f"Model file not found: {path}")
            elif path.stat().st_size == 0:
                errors.append(f"Model file is empty: {path}")
        if self.audio.sample_rate <= 0:
            errors.append(f"Invalid sample_rate: {self.audio.sample_rate}")
        if self.audio.channels < 1:
            errors.append(f"Invalid channels: {self.audio.channels}")
        if self.pipeline.num_threads < 1:
            errors.append(f"Invalid num_threads: {self.pipeline.num_threads}")
        return errors


def _build_nested(value: Any) -> Any:
    """Recursively build nested dataclass instances from dicts."""
    if isinstance(value, dict):
        # Try to match known nested config types
        for cls in (ModelConfig, AudioConfig, EndpointConfig, PipelineConfig, GuiConfig, RefinementConfig):
            try:
                return cls(**{k: v for k, v in value.items() if k in {f.name for f in __import__("dataclasses").fields(cls)}})
            except TypeError:
                continue
    return value


# Default config file location
CONFIG_FILE = PROJECT_ROOT / "config.json"


def load_config() -> Config:
    """Load configuration from config.json, falling back to defaults."""
    return Config.load(CONFIG_FILE)