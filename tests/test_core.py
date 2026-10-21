import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech2text.config import Config
from speech2text.language import label_utterance
from speech2text.recognizer import Recognizer


@pytest.mark.requires_model
def test_config_defaults_are_valid():
    cfg = Config()
    errors = cfg.validate()
    assert not errors, errors


def test_language_labels_are_coherent():
    assert label_utterance("你好，世界") == "Chinese"
    assert label_utterance("hello world") == "English"
    assert label_utterance("hello 你好") == "Mixed"


@pytest.mark.requires_model
def test_recognizer_loads_model_and_accepts_audio():
    cfg = Config()
    rec = Recognizer(cfg)
    rec.load()
    try:
        assert rec.is_loaded() is True
        result = rec.feed(np.zeros(1600, dtype=np.float32))
        assert isinstance(result.partial_text, str)
        assert isinstance(result.final_text, str)
    finally:
        rec.cleanup()


@pytest.mark.requires_model
def test_recognizer_accepts_sounddevice_channel_shape():
    cfg = Config()
    rec = Recognizer(cfg)
    rec.load()
    try:
        result = rec.feed(np.zeros((1600, 1), dtype=np.float32))
        assert isinstance(result.partial_text, str)
        assert isinstance(result.final_text, str)
    finally:
        rec.cleanup()
