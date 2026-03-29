"""
EEGClassifier — the public inference interface for the NeuroTune pipeline.

Loads the trained MLP + StandardScaler from disk and exposes two methods:

    classify_window(eeg_window) → EmotionalState
        Classify a single 1-second EEG window.

    detect_state(eeg_windows, n_vote=5) → EmotionalState
        Classify multiple windows and return the majority-vote state.
        Use this for more stable real-time predictions.

Both methods accept a (4, n_samples) NumPy array where the 4 channels
are ordered to match the Muse headband: [TP9, AF7, AF8, TP10].
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from collections import Counter

from .features import (
    DREAMER_FS,
    MUSE_FS,
    N_FEATURES,
    extract_de_features,
    resample_to_128,
)
from .model import EmotionMLP, IDX_TO_STATE

# Lazy import to avoid hard dependency when song_emotion_profiling is not on path
try:
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from song_emotion_profiling.models import EmotionalState
    _HAS_EMOTIONAL_STATE = True
except ImportError:
    _HAS_EMOTIONAL_STATE = False

_DEFAULT_MODEL_DIR = Path(__file__).parent / "models"


class EEGClassifier:
    """Real-time EEG emotion classifier.

    Parameters
    ----------
    model_path  : path to the saved MLP state-dict (.pt)
    scaler_path : path to the saved StandardScaler (.pkl)
    fs          : sampling rate of the incoming EEG (Hz)
                  — 256 for Muse, 128 for DREAMER.
                  If 256, windows are automatically downsampled to 128 Hz.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        fs: int = MUSE_FS,
    ) -> None:
        model_path  = Path(model_path)  if model_path  else _DEFAULT_MODEL_DIR / "emotion_mlp_best.pt"
        scaler_path = Path(scaler_path) if scaler_path else _DEFAULT_MODEL_DIR / "scaler.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run training first:\n"
                "  python -m EEGClassifier.train --mat DREAMER.mat --out EEGClassifier/models"
            )
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        self.fs = fs
        self.device = torch.device("cpu")   # always CPU for inference

        # Load feature metadata if available (written by train_csv.py)
        meta_path = model_path.parent / "feature_meta.pkl"
        input_dim = N_FEATURES
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            input_dim = meta.get("n_features", N_FEATURES)

        self.model = EmotionMLP(input_dim=input_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    # ── core methods ─────────────────────────────────────────────────────────

    def classify_window(self, eeg_window: np.ndarray) -> str:
        """Classify a single EEG window.

        Parameters
        ----------
        eeg_window : (4, n_samples)
            4-channel EEG in the order [TP9, AF7, AF8, TP10].
            Must be at least 1 second of data at the configured `fs`.

        Returns
        -------
        str — one of 'calm', 'focused', 'stressed'
        """
        window = self._preprocess(eeg_window)
        feat = extract_de_features(window, fs=DREAMER_FS).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat).astype(np.float32)
        x = torch.from_numpy(feat_scaled)
        with torch.no_grad():
            pred_idx = int(self.model(x).argmax(dim=1).item())
        return IDX_TO_STATE[pred_idx]

    def detect_state(
        self,
        eeg_windows: np.ndarray,
        n_vote: int = 5,
    ) -> str:
        """Classify multiple windows and return the majority-vote state.

        Parameters
        ----------
        eeg_windows : (n_windows, 4, n_samples)  or  (4, n_samples) for single
        n_vote      : how many most-recent windows to include in the vote.
                      Ignored when fewer windows are provided.

        Returns
        -------
        str — 'calm', 'focused', or 'stressed'
        """
        if eeg_windows.ndim == 2:
            # Single window passed directly
            return self.classify_window(eeg_windows)

        windows = eeg_windows[-n_vote:]  # take the most recent n_vote windows
        predictions = [self.classify_window(w) for w in windows]
        majority, _ = Counter(predictions).most_common(1)[0]
        return majority

    # ── EmotionalState integration ───────────────────────────────────────────

    def detect_emotional_state(
        self,
        eeg_windows: np.ndarray,
        n_vote: int = 5,
    ):
        """Same as detect_state but returns an EmotionalState enum.

        Requires song_emotion_profiling to be importable.
        """
        if not _HAS_EMOTIONAL_STATE:
            raise ImportError(
                "song_emotion_profiling is not importable. "
                "Use detect_state() for the raw string label."
            )
        state_str = self.detect_state(eeg_windows, n_vote=n_vote)
        return EmotionalState(state_str)

    # ── internal ─────────────────────────────────────────────────────────────

    def _preprocess(self, window: np.ndarray) -> np.ndarray:
        """Downsample Muse 256 Hz → 128 Hz if needed."""
        if self.fs != DREAMER_FS:
            window = resample_to_128(window, fs_in=self.fs)
        return window
