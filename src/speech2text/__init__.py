"""Offline speech-to-text package."""

from .config import Config, load_config
from .recognizer import Recognizer

__all__ = ["Config", "Recognizer", "load_config"]
__version__ = "0.1.0"
