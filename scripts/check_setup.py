#!/usr/bin/env python3
"""Verify the local environment and basic recognizer setup."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _use_utf8_output() -> None:
    """Print device names with non-ASCII characters without crashing.

    Device names such as "麦克风阵列 (Realtek(R) Audio)" cannot be encoded with
    the default Windows console code page, which would abort the check.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def check_python() -> None:
    print(f"Python: {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Platform: {platform.platform()}")
    if sys.version_info[:2] != (3, 11):
        print("WARNING: expected Python 3.11 for this project.")


def check_imports() -> None:
    missing: list[str] = []
    for name in ["numpy", "sounddevice", "sherpa_onnx"]:
        try:
            __import__(name)
            print(f"OK import: {name}")
        except Exception as exc:  # pragma: no cover - reporting only
            missing.append(f"{name}: {exc}")
    if missing:
        print("MISSING IMPORTS:")
        for item in missing:
            print("  -", item)


def check_model() -> None:
    model_dir = ROOT / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
    required = [
        "encoder-epoch-99-avg-1.int8.onnx",
        "decoder-epoch-99-avg-1.int8.onnx",
        "joiner-epoch-99-avg-1.int8.onnx",
        "tokens.txt",
    ]
    missing = []
    for file_name in required:
        path = model_dir / file_name
        if not path.exists():
            missing.append(file_name)
        else:
            print(f"OK model file: {file_name} ({path.stat().st_size} bytes)")
    if missing:
        print("MISSING MODEL FILES:")
        for item in missing:
            print("  -", item)


def check_microphone() -> None:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if not inputs:
            print("No microphone input devices detected.")
            return
        print("Input devices:")
        for idx, device in enumerate(inputs):
            print(f"  [{idx}] {device.get('name', 'Unnamed device')} ({device.get('max_input_channels', 0)} in channels)")
    except Exception as exc:  # pragma: no cover - reporting only
        print(f"Unable to query microphones: {exc}")


def check_offline_usage() -> None:
    app_files = [
        ROOT / "src" / "speech2text" / name
        for name in [
            "audio.py",
            "config.py",
            "gui.py",
            "language.py",
            "logging_setup.py",
            "main.py",
            "pipeline.py",
            "recognizer.py",
            "refinement.py",
            "__main__.py",
        ]
    ]
    banned = [
        "urllib",
        "requests",
        "httpx",
        "socket",
        "websocket",
        "http.client",
        "urlopen",
        "socket.create_connection",
    ]
    offenders = []
    for path in app_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if any(token in text for token in banned):
            offenders.append(path.name)
    if offenders:
        print("WARNING: app modules reference network APIs:")
        for name in offenders:
            print(f"  - {name}")
    else:
        print("OK app modules: no network clients found in runtime code")


def smoke_test() -> None:
    try:
        from speech2text.recognizer import Recognizer
        from speech2text.config import Config

        cfg = Config()
        rec = Recognizer(cfg)
        rec.load()
        print("OK recognizer load")
        rec.cleanup()
    except Exception as exc:  # pragma: no cover - reporting only
        print(f"Recognizer smoke test failed: {exc}")


def main() -> None:
    _use_utf8_output()
    print("=== Speech-to-text environment check ===")
    check_python()
    check_imports()
    check_model()
    check_microphone()
    check_offline_usage()
    smoke_test()


if __name__ == "__main__":
    main()
