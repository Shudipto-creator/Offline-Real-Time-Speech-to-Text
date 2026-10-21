"""Optional final-sentence refinement using a local faster-whisper model.

This stage is intentionally disabled by default and only runs when the user
explicitly opts in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RefinementResult:
    """Result of a final-sentence refinement pass."""

    original_text: str = ""
    refined_text: str = ""
    accepted: bool = False
    reason: str = ""


class FinalSentenceRefiner:
    """Local-only optional refinement placeholder for Stage 10."""

    def __init__(self, config: Optional[object] = None) -> None:
        self.config = config
        self.enabled = bool(getattr(config, "enabled", False))

    def is_available(self) -> bool:
        try:
            import faster_whisper  # type: ignore
            return True
        except Exception:
            return False

    def refine(self, text: str) -> RefinementResult:
        """Return the original text unless the optional dependency is installed and enabled."""
        if not text or not self.enabled:
            return RefinementResult(original_text=text, refined_text=text, accepted=False, reason="disabled")
        if not self.is_available():
            return RefinementResult(original_text=text, refined_text=text, accepted=False, reason="dependency unavailable")
        return RefinementResult(original_text=text, refined_text=text, accepted=False, reason="not executed")
