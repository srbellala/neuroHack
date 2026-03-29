"""
BaselineSession — the core of the NeuroTune pipeline.

Wires together:
  - MuseStream (EEG input)
  - EEGClassifier (state detection)
  - TrackLibrary (song selection)
  - SpotifyPlayer (playback control)

Session loop
------------
1. User sets a baseline state at start (calm / focused / stressed).
2. EEG windows stream in via MuseStream's on_window callback.
3. Every CLASSIFY_EVERY windows, a majority-voted state is produced.
4. If the state has differed from baseline for DRIFT_THRESHOLD consecutive
   classifications, a drift is declared.
5. The top steering candidate from TrackLibrary is played.
6. A SWITCH_COOLDOWN prevents switching again too soon.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from song_emotion_profiling import TrackLibrary
from song_emotion_profiling.models import EmotionalState

from .spotify_player import SpotifyPlayer

# ── tuning constants ──────────────────────────────────────────────────────────

CLASSIFY_EVERY  = 5    # classify after every N new windows
DRIFT_THRESHOLD = 3    # N consecutive off-baseline classifications = drift
SWITCH_COOLDOWN = 30   # seconds before another track switch is allowed
TOP_N_CANDIDATES = 10  # how many steering candidates to rank


class BaselineSession:
    """
    Orchestrates real-time EEG → track-switching pipeline.

    Parameters
    ----------
    baseline  : the emotional state the user wants to maintain
    library   : TrackLibrary loaded with profiled songs
    player    : SpotifyPlayer for playback control
    classifier: EEGClassifier instance (import kept lazy to avoid hard dep)
    on_state_change : optional callback(current: EmotionalState, baseline: EmotionalState)
                      fired whenever a new state is classified
    """

    def __init__(
        self,
        baseline: EmotionalState,
        library: TrackLibrary,
        player: SpotifyPlayer,
        classifier,
        on_state_change: Callable[[EmotionalState, EmotionalState], None] | None = None,
    ) -> None:
        self.baseline = baseline
        self._library = library
        self._player = player
        self._classifier = classifier
        self._on_state_change = on_state_change

        self._window_buf: deque[np.ndarray] = deque(maxlen=CLASSIFY_EVERY * 4)
        self._windows_since_classify = 0
        self._drift_streak = 0
        self._last_switch_time: float = 0.0
        self._current_state: EmotionalState = baseline
        self._lock = threading.Lock()
        self._history: list[tuple[float, EmotionalState]] = []

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def current_state(self) -> EmotionalState:
        return self._current_state

    @property
    def history(self) -> list[tuple[float, EmotionalState]]:
        """List of (timestamp, state) pairs recorded during the session."""
        return list(self._history)

    def on_window(self, window: np.ndarray) -> None:
        """
        Called by MuseStream each time a new EEG window is ready.
        Thread-safe — designed to be used as MuseStream's on_window callback.
        """
        with self._lock:
            self._window_buf.append(window)
            self._windows_since_classify += 1

            if self._windows_since_classify < CLASSIFY_EVERY:
                return

            self._windows_since_classify = 0

        windows = np.stack(list(self._window_buf), axis=0)
        self._classify_and_act(windows)

    def process_windows(self, windows: np.ndarray) -> None:
        """
        Alternative to on_window — feed a batch of windows directly.
        Useful for simulation or when not using MuseStream's callback.

        windows : (N, 4, window_samples)
        """
        self._classify_and_act(windows)

    # ── internal ─────────────────────────────────────────────────────────────

    def _classify_and_act_with_state(self, state: EmotionalState) -> None:
        """Skip EEG classification and act directly on a known state (for simulation)."""
        self._classify_and_act_core(state)

    def _classify_and_act(self, windows: np.ndarray) -> None:
        state: EmotionalState = self._classifier.detect_emotional_state(windows)
        self._classify_and_act_core(state)

    def _classify_and_act_core(self, state: EmotionalState) -> None:
        self._current_state = state
        self._history.append((time.time(), state))

        if self._on_state_change:
            self._on_state_change(state, self.baseline)

        if state == self.baseline:
            self._drift_streak = 0
            return

        self._drift_streak += 1

        if self._drift_streak < DRIFT_THRESHOLD:
            return

        # Drift confirmed — check cooldown
        now = time.time()
        if now - self._last_switch_time < SWITCH_COOLDOWN:
            return

        self._switch_track(from_state=state, to_state=self.baseline)
        self._last_switch_time = now
        self._drift_streak = 0

    def _switch_track(
        self, from_state: EmotionalState, to_state: EmotionalState
    ) -> None:
        candidates = self._library.get_steering_candidates(
            from_state=from_state,
            to_state=to_state,
            top_n=TOP_N_CANDIDATES,
        )

        if not candidates:
            print(f"[NeuroTune] No steering candidates for {from_state.value} → {to_state.value}")
            return

        # Avoid replaying the currently-playing track if possible
        current = self._player.get_current_track()
        current_id = current["track_id"] if current else None

        track = next(
            (c for c in candidates if c.track_id != current_id),
            candidates[0],  # fall back to top if all match
        )

        key = f"{from_state.value}_to_{to_state.value}"
        score = track.steering_score.get(key, 0.0)

        print(
            f"[NeuroTune] Drift detected: {from_state.value} → switching to steer toward {to_state.value}\n"
            f"            Playing: \"{track.name}\" — {track.artist}  "
            f"(steering score: {score:.2f})"
        )

        self._player.play_track(track.track_id)
