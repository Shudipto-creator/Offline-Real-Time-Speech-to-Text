"""Tkinter GUI for the offline speech-to-text app."""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from .config import Config
from .language import label_utterance
from .pipeline import EventType, PipelineEvent, RecognitionWorker
from .refinement import FinalSentenceRefiner


class SpeechToTextApp(tk.Tk):
    """Minimal but functional desktop speech-to-text interface."""

    def __init__(self, config: Optional[Config] = None, device_index: Optional[int] = None) -> None:
        super().__init__()
        self.config = config or Config()
        self.device_index = device_index
        self.gui_queue: queue.Queue[PipelineEvent] = queue.Queue(maxsize=200)
        self.worker = RecognitionWorker(self.config, self.gui_queue)
        self.refiner = FinalSentenceRefiner(self.config.refinement)
        self.title(self.config.gui.window_title)
        self.geometry(f"{self.config.gui.window_width}x{self.config.gui.window_height}")
        self.minsize(560, 420)

        self.status_var = tk.StringVar(value="Idle")
        self.partial_var = tk.StringVar(value="")
        self.device_var = tk.StringVar()
        self.preference_var = tk.StringVar(value="auto")
        self.transcript_text = None

        self._build_ui()
        self._populate_devices()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(self.config.gui.poll_interval_ms, self._poll_queue)

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 8))

        ttk.Button(top, text="Start", command=self.start_listening).pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Stop", command=self.stop_listening).pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Copy", command=self.copy_transcript).pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Clear", command=self.clear_transcript).pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Refresh", command=self._populate_devices).pack(side="left")

        ttk.Label(top, text="Microphone:").pack(side="left", padx=(16, 4))
        self.device_combo = ttk.Combobox(top, textvariable=self.device_var, width=30, state="readonly")
        self.device_combo.pack(side="left")

        ttk.Label(top, text="Label:").pack(side="left", padx=(12, 4))
        self.label_combo = ttk.Combobox(top, textvariable=self.preference_var, width=12, state="readonly")
        self.label_combo["values"] = ("auto", "Chinese", "English")
        self.label_combo.set("auto")
        self.label_combo.pack(side="left")

        status = ttk.Frame(main)
        status.pack(fill="x", pady=(0, 8))
        ttk.Label(status, text="Status:").pack(side="left")
        ttk.Label(status, textvariable=self.status_var, foreground="darkblue").pack(side="left", padx=(8, 0))

        ttk.Label(main, text="Live partial:").pack(anchor="w")
        ttk.Label(main, textvariable=self.partial_var, wraplength=620, justify="left").pack(anchor="w", pady=(0, 8))

        ttk.Label(main, text="Transcript:").pack(anchor="w")
        transcript_frame = ttk.Frame(main)
        transcript_frame.pack(fill="both", expand=True)
        self.transcript_text = tk.Text(transcript_frame, height=18, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(transcript_frame, orient="vertical", command=self.transcript_text.yview)
        self.transcript_text.configure(yscrollcommand=scrollbar.set)
        self.transcript_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _populate_devices(self) -> None:
        devices = self.worker.audio.list_devices()
        names = [d["name"] for d in devices]
        if not names:
            self.device_combo["values"] = ["No microphone detected"]
            self.device_combo.set("No microphone detected")
            return

        preferred_index = self.worker.audio.preferred_device_index()
        preferred_name = None
        if preferred_index is not None:
            for device in devices:
                if device["index"] == preferred_index:
                    preferred_name = device["name"]
                    break

        self.device_combo["values"] = names
        if preferred_name is not None:
            self.device_combo.set(preferred_name)
        elif self.device_index is not None and self.device_index < len(devices):
            self.device_combo.set(devices[self.device_index]["name"])
        else:
            self.device_combo.set(names[0])

    def _selected_device_index(self) -> Optional[int]:
        selected = self.device_combo.get()
        devices = self.worker.audio.list_devices()
        for index, device in enumerate(devices):
            if device["name"] == selected:
                return index
        return None

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def append_transcript(self, text: str) -> None:
        if not text:
            return
        self.transcript_text.configure(state="normal")
        self.transcript_text.insert("end", text + "\n")
        self.transcript_text.configure(state="disabled")
        self.transcript_text.see("end")

    def clear_transcript(self) -> None:
        self.transcript_text.configure(state="normal")
        self.transcript_text.delete("1.0", "end")
        self.transcript_text.configure(state="disabled")
        self.partial_var.set("")

    def copy_transcript(self) -> None:
        text = self.transcript_text.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._set_status("Transcript copied")

    def start_listening(self) -> None:
        try:
            self._set_status("Loading model...")
            print("[DBG] GUI: Starting worker thread")
            self.worker.start()
            print("[DBG] GUI: Starting audio capture")
            self.worker.start_audio(device_index=self._selected_device_index())
            # NOTE: "Listening" status will be set by worker AFTER model loads
            print("[DBG] GUI: start_listening done, waiting for worker to emit Listening")
        except Exception as exc:  # pragma: no cover - UI path
            self._set_status("Error")
            messagebox.showerror("Speech-to-text", f"Unable to start: {exc}")

    def stop_listening(self) -> None:
        try:
            self._set_status("Stopping")
            self.worker.stop_audio()
            self.worker.stop()
            self._set_status("Idle")
        except Exception as exc:  # pragma: no cover - UI path
            self._set_status("Error")
            messagebox.showerror("Speech-to-text", f"Unable to stop: {exc}")

    def _poll_queue(self) -> None:
        try:
            while True:
                event = self.gui_queue.get_nowait()
                if event.type == EventType.PARTIAL:
                    self.partial_var.set(event.text)
                elif event.type == EventType.FINAL:
                    final_text = event.text.strip()
                    if self.refiner.enabled:
                        refined = self.refiner.refine(final_text)
                        if refined.accepted and refined.refined_text:
                            final_text = refined.refined_text
                    label = label_utterance(final_text, self.preference_var.get().lower())
                    self.append_transcript(f"[{label}] {final_text}")
                    self.partial_var.set("")
                elif event.type == EventType.STATUS:
                    self._set_status(event.message)
                elif event.type == EventType.ERROR:
                    self._set_status("Error")
                    messagebox.showerror("Speech-to-text", event.message)
        except queue.Empty:
            pass
        self.after(self.config.gui.poll_interval_ms, self._poll_queue)

    def _on_close(self) -> None:
        try:
            self.worker.stop_audio()
            self.worker.stop()
            self.worker.cleanup()
        except Exception:
            pass
        self.destroy()


def main() -> None:
    app = SpeechToTextApp()
    app.mainloop()
