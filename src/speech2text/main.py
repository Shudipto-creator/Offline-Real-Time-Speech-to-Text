"""Application entry point."""

from __future__ import annotations

import argparse

from .gui import SpeechToTextApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline speech-to-text application")
    parser.add_argument("--device-index", type=int, default=None, help="Microphone device index")
    parser.add_argument("--skip-gui", action="store_true", help="Run a non-GUI smoke check")
    args = parser.parse_args()

    if args.skip_gui:
        print("GUI disabled; use the Tkinter app with python -m speech2text.main")
        return

    app = SpeechToTextApp(device_index=args.device_index)
    app.mainloop()


if __name__ == "__main__":
    main()
