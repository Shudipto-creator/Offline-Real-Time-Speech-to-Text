"""
Language label detection and text handling utilities.

Provides heuristics for labeling utterances as Chinese, English, or mixed.
This is display-only and does NOT force the recognizer's language mode.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class LanguageLabel(Enum):
    """Display language labels."""

    AUTO = "auto"
    CHINESE = "chinese"
    ENGLISH = "english"
    MIXED = "mixed"


# Unicode ranges for CJK characters
CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0xFF00, 0xFFEF),    # Halfwidth/Fullwidth forms
]


def _has_cjk(text: str) -> bool:
    """Check if text contains any CJK characters."""
    for ch in text:
        code = ord(ch)
        for start, end in CJK_RANGES:
            if start <= code <= end:
                return True
    return False


def _has_latin(text: str) -> bool:
    """Check if text contains Latin letters."""
    return bool(re.search(r"[A-Za-z]", text))


def _has_chinese_punctuation(text: str) -> bool:
    """Check for Chinese-specific punctuation."""
    chinese_punct = "，。！？：；""''（）【】《》、"
    return any(ch in text for ch in chinese_punct)


def detect_language(text: str) -> LanguageLabel:
    """Detect the language composition of recognized text.

    Uses a simple heuristic based on character sets:
    - Only CJK characters -> Chinese
    - Only Latin letters -> English
    - Both CJK and Latin -> Mixed

    Args:
        text: The recognized text.

    Returns:
        A LanguageLabel indicating the detected language.
    """
    if not text or not text.strip():
        return LanguageLabel.ENGLISH  # Default for empty

    has_cjk = _has_cjk(text)
    has_latin = _has_latin(text)

    if has_cjk and has_latin:
        return LanguageLabel.MIXED
    elif has_cjk:
        return LanguageLabel.CHINESE
    else:
        return LanguageLabel.ENGLISH


def label_utterance(text: str, preference: str = "auto") -> str:
    """Return a human-readable language label for an utterance.

    Args:
        text: The recognized text.
        preference: One of "auto", "chinese", "english".

    Returns:
        A display label string.
    """
    preference = preference.lower()
    detected = detect_language(text)

    if preference == "chinese":
        return "Chinese"
    elif preference == "english":
        return "English"
    else:  # auto
        if detected == LanguageLabel.CHINESE:
            return "Chinese"
        elif detected == LanguageLabel.ENGLISH:
            return "English"
        else:
            return "Mixed"


def trim_text(text: str) -> str:
    """Trim leading/trailing whitespace without changing recognized words."""
    return text.strip() if text else ""


def is_mostly_silent(text: str, threshold: int = 2) -> bool:
    """Check if text appears to be noise or silence."""
    return len(text.strip()) < threshold


__all__ = [
    "LanguageLabel",
    "detect_language",
    "label_utterance",
    "trim_text",
    "is_mostly_silent",
]