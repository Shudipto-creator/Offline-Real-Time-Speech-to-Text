#!/usr/bin/env python3
"""
Download the sherpa-onnx streaming zipformer bilingual model for speech-to-text.

Model: sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
Source: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

This script downloads the model only when explicitly run by the user.
Verified file names from official documentation:
  - encoder-epoch-99-avg-1.int8.onnx  (174 MB)
  - encoder-epoch-99-avg-1.onnx       (315 MB, full precision)
  - decoder-epoch-99-avg-1.int8.onnx  (13 MB)
  - decoder-epoch-99-avg-1.onnx       (14 MB)
  - joiner-epoch-99-avg-1.int8.onnx   (3.1 MB)
  - joiner-epoch-99-avg-1.onnx        (13 MB)
  - tokens.txt
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

MODEL_NAME = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_NAME}.tar.bz2"
)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Verified required files (using int8 variants per spec preference)
REQUIRED_FILES = [
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.int8.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
    "tokens.txt",
]

# Full precision variants (optional, for comparison)
OPTIONAL_FILES = [
    "encoder-epoch-99-avg-1.onnx",
    "decoder-epoch-99-avg-1.onnx",
    "joiner-epoch-99-avg-1.onnx",
]


def download_file(url: str, dest: Path, timeout: int = 600) -> None:
    """Download a file with a progress indicator."""
    print(f"Source URL: {url}")
    print(f"Destination: {dest}")
    print(f"Estimated size: ~487 MB. Please be patient...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1 << 16  # 64 KB

        with open(dest, "wb") as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                downloaded += len(buffer)
                if total_size > 0:
                    pct = downloaded * 100 // total_size
                    filled = min(40, downloaded * 40 // total_size)
                    bar = "=" * filled + " " * (40 - filled)
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(
                        f"\r  [{bar}] {pct}%  {mb:.1f}/{total_mb:.1f} MB",
                        end="", flush=True,
                    )
        print("\nDownload complete.")


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract a .tar.bz2 archive safely."""
    print(f"Extracting: {archive_path.name} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:bz2") as tf:
        tf.extractall(extract_dir, filter="data")
    print("Extraction complete.")


def verify_model_files(model_dir: Path) -> dict:
    """Verify required model files exist and are non-empty."""
    found = {}
    for fname in REQUIRED_FILES + OPTIONAL_FILES:
        fpath = model_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            if size > 0:
                found[fname] = {"size": size, "path": str(fpath.relative_to(model_dir.parent))}
                status = "REQUIRED" if fname in REQUIRED_FILES else "OPTIONAL"
                print(f"  [{status}] {fname}: {size:,} bytes")
            else:
                print(f"  [EMPTY] {fname}: file is empty!")
        else:
            if fname in REQUIRED_FILES:
                print(f"  [MISSING] {fname}")
    return found


def main() -> None:
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = MODELS_DIR / MODEL_NAME

    # Check if already downloaded
    if model_dir.exists() and (model_dir / "tokens.txt").exists():
        print("Model already present. Verifying...")
        found = verify_model_files(model_dir)
        missing = [f for f in REQUIRED_FILES if f not in found]
        if not missing:
            print("\nAll required files present. No download needed.")
            return
        else:
            print(f"\nMissing files: {missing}. Re-downloading...")

    # Download
    archive_path = MODELS_DIR.parent / f"{MODEL_NAME}.tar.bz2"
    print()

    try:
        download_file(MODEL_URL, archive_path)
        extract_archive(archive_path, MODELS_DIR)

        # Remove archive to save disk space
        archive_path.unlink()
        print(f"Removed archive: {archive_path}")

        # Verify
        print("\nVerifying model files...")
        found = verify_model_files(model_dir)

        missing = [f for f in REQUIRED_FILES if f not in found]
        if missing:
            print(f"\nERROR: Missing required files: {missing}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Model download and verification COMPLETE")
        print("=" * 60)
        print(f"Model directory: {model_dir}")
        print("\nFiles on disk:")
        for f in sorted(model_dir.iterdir()):
            if f.is_file():
                print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    except urllib.error.URLError as e:
        print(f"\nERROR: Network error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()