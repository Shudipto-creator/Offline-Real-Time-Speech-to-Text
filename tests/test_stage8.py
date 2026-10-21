import queue
from unittest.mock import MagicMock

import numpy as np

from speech2text.audio import AudioCapture, AudioConfig
from speech2text.config import Config
from speech2text.pipeline import EventType, RecognitionWorker
from speech2text.recognizer import RecognitionResult


def test_audio_capture_queue_overload_counts_drops():
    capture = AudioCapture(AudioConfig(max_queue_blocks=2))

    for _ in range(5):
        capture._audio_callback(np.ones(1600, dtype=np.float32), 1600, None, None)

    assert capture.queue_size <= 2
    assert capture.dropped_blocks >= 1


def test_recognition_worker_emits_partial_and_final_events():
    q = queue.Queue(maxsize=20)
    worker = RecognitionWorker(Config(), q)

    worker.audio = MagicMock()
    worker.audio.dropped_blocks = 0
    worker.audio.stop = MagicMock()
    worker.audio.clear_queue = MagicMock()

    def get_block(timeout=0.1):
        if not hasattr(get_block, "called"):
            get_block.called = True
            return np.ones(1600, dtype=np.float32)
        worker._running = False
        return None

    worker.audio.get_block.side_effect = get_block

    worker.recognizer = MagicMock()
    worker.recognizer.load = MagicMock()
    worker.recognizer.feed.return_value = RecognitionResult(
        partial_text="hello there",
        final_text="hello there",
        is_endpoint=True,
    )
    worker.recognizer.flush.return_value = RecognitionResult(final_text="hello there", is_endpoint=True)
    worker.recognizer.reset = MagicMock()
    worker.recognizer.recreate_stream = MagicMock()
    worker.recognizer.cleanup = MagicMock()

    worker._running = True
    worker._run_recognition()

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    assert any(event.type == EventType.PARTIAL for event in events)
    assert any(event.type == EventType.FINAL for event in events)
    assert any(event.type == EventType.STATUS for event in events)
    status_messages = [
        event.message for event in events if event.type == EventType.STATUS
    ]
    assert status_messages[:2] == ["Loading model...", "Listening"]


def test_worker_stop_is_safe_when_not_running():
    q = queue.Queue(maxsize=20)
    worker = RecognitionWorker(Config(), q)

    worker.stop()

    assert worker._running is False
    assert worker._thread is None


def test_audio_prefers_real_microphone_over_sound_mapper(monkeypatch):
    capture = AudioCapture(AudioConfig())

    monkeypatch.setattr(
        "speech2text.audio.sd.query_devices",
        lambda: [
            {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2},
            {"name": "Microphone Array (Realtek)", "max_input_channels": 2},
        ],
    )

    assert capture.preferred_device_index() == 1
