"""
Muse headband EEG streaming via Lab Streaming Layer (LSL).

Setup
-----
1. Pair your Muse over Bluetooth.
2. Start the Muse LSL bridge in a terminal:

       muse-lsl stream

   (or use the BlueMuse app on Windows)

3. Then run the live demo:

       python -m EEGClassifier.live_demo

This module connects to the LSL 'EEG' stream published by muse-lsl and
buffers samples into fixed-length windows for the classifier.

Muse channel order (muselsl default): TP9, AF7, AF8, TP10  (4 channels)
Sampling rate: 256 Hz
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Callable

import numpy as np

try:
    from pylsl import StreamInlet, resolve_byprop, LostError
    _HAS_PYLSL = True
except ImportError:
    _HAS_PYLSL = False

from .features import MUSE_FS, N_CHANNELS

# ── constants ────────────────────────────────────────────────────────────────

MUSE_CHANNEL_COUNT = 4          # TP9, AF7, AF8, TP10
LSL_STREAM_TYPE    = "EEG"
LSL_RESOLVE_TIMEOUT = 10.0      # seconds to search for stream


# ── MuseStream ───────────────────────────────────────────────────────────────

class MuseStream:
    """Buffers live EEG from the Muse LSL stream into overlapping windows.

    Parameters
    ----------
    window_sec  : window length in seconds (default 1.0)
    step_sec    : step between windows in seconds (default 0.5 → 50% overlap)
    fs          : expected sampling rate (default 256 Hz for Muse)
    on_window   : optional callback  (window: np.ndarray) → None
                  called every time a new window is ready.
                  window shape: (4, window_samples)
    """

    def __init__(
        self,
        window_sec: float = 1.0,
        step_sec: float = 0.5,
        fs: int = MUSE_FS,
        on_window: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        if not _HAS_PYLSL:
            raise ImportError(
                "pylsl is required for Muse streaming.\n"
                "Install with:  pip install pylsl"
            )

        self.fs = fs
        self.window_samples = int(window_sec * fs)
        self.step_samples   = int(step_sec * fs)
        self.on_window      = on_window

        self._buffer: deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._inlet: StreamInlet | None = None

    # ── public API ───────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Resolve the LSL EEG stream (blocks until found or timeout)."""
        print(f"Searching for LSL stream of type '{LSL_STREAM_TYPE}' …")
        streams = resolve_byprop("type", LSL_STREAM_TYPE, timeout=LSL_RESOLVE_TIMEOUT)
        if not streams:
            raise RuntimeError(
                "No LSL EEG stream found.\n"
                "Make sure the Muse is paired and run:  muse-lsl stream"
            )
        self._inlet = StreamInlet(streams[0])
        info = self._inlet.info()
        actual_fs = info.nominal_srate()
        actual_ch = info.channel_count()
        print(f"Connected to '{info.name()}'  —  {actual_ch}ch @ {actual_fs:.0f} Hz")
        if actual_ch < MUSE_CHANNEL_COUNT:
            raise RuntimeError(
                f"Expected {MUSE_CHANNEL_COUNT} EEG channels, got {actual_ch}."
            )

    def start(self) -> None:
        """Start background acquisition thread."""
        if self._inlet is None:
            self.connect()
        self._running = True
        self._thread = threading.Thread(target=self._acquire, daemon=True)
        self._thread.start()
        print("Muse stream started.")

    def stop(self) -> None:
        """Stop background acquisition thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("Muse stream stopped.")

    def get_window(self) -> np.ndarray | None:
        """Return the most recent complete window, or None if not ready.

        Returns
        -------
        window : (4, window_samples)  or  None
        """
        with self._buffer_lock:
            if len(self._buffer) < self.window_samples:
                return None
            samples = list(self._buffer)[-self.window_samples:]
        return np.stack(samples, axis=1)   # (4, window_samples)

    def get_windows_blocking(self, n_windows: int = 5) -> np.ndarray:
        """Block until *n_windows* consecutive windows are available.

        Returns
        -------
        windows : (n_windows, 4, window_samples)
        """
        collected: list[np.ndarray] = []
        last_trigger = 0
        while len(collected) < n_windows:
            with self._buffer_lock:
                buf_len = len(self._buffer)
            total_needed = self.window_samples + last_trigger * self.step_samples
            if buf_len >= total_needed + self.window_samples:
                w = self.get_window()
                if w is not None:
                    collected.append(w)
                    last_trigger += 1
            else:
                time.sleep(0.05)
        return np.stack(collected, axis=0)  # (n_windows, 4, window_samples)

    # ── internal ─────────────────────────────────────────────────────────────

    def _acquire(self) -> None:
        """Continuously pull samples from LSL and append to the ring buffer."""
        step_counter = 0
        try:
            while self._running:
                sample, _ = self._inlet.pull_sample(timeout=1.0)
                if sample is None:
                    continue
                ch_data = np.array(sample[:MUSE_CHANNEL_COUNT], dtype=np.float32)
                with self._buffer_lock:
                    self._buffer.append(ch_data)
                    # Keep only as many samples as needed for one window
                    max_buf = self.window_samples * 4
                    while len(self._buffer) > max_buf:
                        self._buffer.popleft()
                    step_counter += 1

                # Fire callback when a new step's worth of data has arrived
                if (
                    self.on_window is not None
                    and len(self._buffer) >= self.window_samples
                    and step_counter >= self.step_samples
                ):
                    step_counter = 0
                    w = self.get_window()
                    if w is not None:
                        self.on_window(w)

        except LostError:
            print("LSL stream lost. Stopping acquisition.")
            self._running = False


# ── convenience: simulate Muse data (for testing without hardware) ───────────

def simulate_muse_window(
    state: str = "calm",
    fs: int = MUSE_FS,
    window_sec: float = 1.0,
    noise_scale: float = 5.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a synthetic Muse EEG window for testing.

    Injects sinusoidal components at frequencies characteristic of each
    emotional state on top of white noise.

    Parameters
    ----------
    state  : 'calm' | 'focused' | 'stressed'
    fs     : sampling rate (Hz)
    window_sec : length of the window
    noise_scale : amplitude of background white noise
    seed   : optional random seed for reproducibility

    Returns
    -------
    window : (4, window_samples) float32
    """
    rng = np.random.default_rng(seed)
    n = int(window_sec * fs)
    t = np.arange(n) / fs
    window = rng.standard_normal((4, n)).astype(np.float32) * noise_scale

    # Inject band-specific sinusoids per state
    if state == "calm":
        # Prominent alpha (10 Hz)
        window += 15.0 * np.sin(2 * np.pi * 10 * t)
    elif state == "focused":
        # Prominent beta (20 Hz) + moderate alpha
        window += 10.0 * np.sin(2 * np.pi * 20 * t)
        window += 8.0  * np.sin(2 * np.pi * 10 * t)
    elif state == "stressed":
        # Prominent beta (25 Hz) + gamma (35 Hz)
        window += 12.0 * np.sin(2 * np.pi * 25 * t)
        window += 8.0  * np.sin(2 * np.pi * 35 * t)

    return window
