"""Shared pytest configuration for the speech2text test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
)

REQUIRED_MODEL_FILES = (
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.int8.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
    "tokens.txt",
)

MODEL_AVAILABLE = all((MODEL_DIR / name).exists() for name in REQUIRED_MODEL_FILES)


def pytest_collection_modifyitems(config, items):
    """Skip model-dependent tests when the model has not been downloaded.

    The model files are intentionally not committed to the repository, so a
    fresh clone must still be able to run the test suite. Run
    `python scripts/download_models.py` to enable the skipped tests.
    """
    if MODEL_AVAILABLE:
        return

    skip = pytest.mark.skip(
        reason="Model files missing - run `python scripts/download_models.py` first"
    )
    for item in items:
        if "requires_model" in item.keywords:
            item.add_marker(skip)
