"""
Feature extraction from EEG signals.

Computes differential entropy (DE) and band power in standard frequency
bands for each EEG channel. These features are used both at training time
(from DREAMER) and at inference time (from the Muse headband).

Frequency bands
---------------
theta  : 4–8  Hz
alpha  : 8–13 Hz
beta   : 13–30 Hz
gamma  : 30–45 Hz

Channel mapping
---------------
DREAMER (Emotiv EPOC, 14 ch):
    AF3=0, F7=1, F3=2, FC5=3, T7=4, P7=5, O1=6,
    O2=7, P8=8, T8=9, FC6=10, F4=11, F8=12, AF4=13

Muse headband (4 ch): TP9, AF7, AF8, TP10
Best DREAMER match   : T7(4), F7(1), F8(12), T8(9)
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

# ── constants ───────────────────────────────────────────────────────────────

DREAMER_FS: int = 128   # Hz
MUSE_FS: int    = 256   # Hz
WINDOW_SEC: float = 1.0  # seconds

# DREAMER channel indices that best match Muse electrode positions
# Order: T7, F7, F8, T8  (maps to Muse: TP9, AF7, AF8, TP10)
DREAMER_MUSE_CHANNEL_INDICES: list[int] = [4, 1, 12, 9]
MUSE_CHANNEL_NAMES: list[str] = ["TP9", "AF7", "AF8", "TP10"]

BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES: list[str] = list(BANDS.keys())
N_BANDS: int = len(BANDS)          # 4
N_CHANNELS: int = 4                # channels used (matches Muse)
N_FEATURES: int = N_BANDS * N_CHANNELS  # 16


# ── filtering ───────────────────────────────────────────────────────────────

def _bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def bandpass_filter(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal : (n_samples,) or (n_channels, n_samples)
    low, high : frequency limits in Hz
    fs : sampling rate in Hz
    """
    sos = _bandpass_sos(low, high, fs)
    return sosfiltfilt(sos, signal, axis=-1)


# ── resampling ───────────────────────────────────────────────────────────────

def resample_to_128(signal: np.ndarray, fs_in: int) -> np.ndarray:
    """Resample signal from *fs_in* Hz to 128 Hz using polyphase filter.

    Parameters
    ----------
    signal : (n_channels, n_samples)
    fs_in  : source sampling rate
    """
    if fs_in == DREAMER_FS:
        return signal
    from math import gcd
    g = gcd(DREAMER_FS, fs_in)
    up, down = DREAMER_FS // g, fs_in // g
    return resample_poly(signal, up, down, axis=-1)


# ── differential entropy ────────────────────────────────────────────────────

def differential_entropy(signal: np.ndarray) -> float:
    """DE of a 1-D signal: 0.5 * log(2πe * var(x)).

    Assumes the signal is approximately Gaussian in each frequency band,
    which is a standard assumption in EEG affective computing.
    """
    var = np.var(signal)
    # Clamp to avoid log(0)
    var = max(var, 1e-10)
    return 0.5 * np.log(2 * np.pi * np.e * var)


# ── main feature extraction ─────────────────────────────────────────────────

def extract_de_features(
    window: np.ndarray,
    fs: float = DREAMER_FS,
) -> np.ndarray:
    """Compute DE features for a single EEG window.

    Parameters
    ----------
    window : (n_channels, n_samples)  — already baseline-removed & filtered
    fs     : sampling rate of *window*

    Returns
    -------
    features : (N_FEATURES,) = (n_channels * n_bands,)
        Order: [ch0_theta, ch0_alpha, ch0_beta, ch0_gamma,
                ch1_theta, ..., ch3_gamma]
    """
    n_channels = window.shape[0]
    features = np.zeros(n_channels * N_BANDS)
    for ch in range(n_channels):
        for b_idx, (band_name, (low, high)) in enumerate(BANDS.items()):
            filtered = bandpass_filter(window[ch], low, high, fs)
            features[ch * N_BANDS + b_idx] = differential_entropy(filtered)
    return features


def extract_features_from_trial(
    eeg: np.ndarray,
    baseline: np.ndarray,
    fs: float = DREAMER_FS,
    channel_indices: list[int] = DREAMER_MUSE_CHANNEL_INDICES,
    window_samples: int = 128,
) -> np.ndarray:
    """Extract DE features from a full DREAMER trial.

    1. Select the 4 channels that match Muse positions.
    2. Compute per-channel baseline mean and subtract it.
    3. Segment into non-overlapping 1-second windows.
    4. Compute DE features per window.

    Parameters
    ----------
    eeg      : (14, n_samples) — full stimulus EEG for one trial
    baseline : (14, n_samples) — baseline EEG for one trial (already averaged
                                 to (14, window_samples) or full length)
    fs       : sampling rate
    channel_indices : which DREAMER channels to use
    window_samples  : samples per window (128 at 128 Hz = 1 s)

    Returns
    -------
    features : (n_windows, N_FEATURES)
    """
    eeg_sel = eeg[channel_indices, :]          # (4, n_samples)

    # Baseline removal — subtract per-channel mean of the baseline
    if baseline.ndim == 2 and baseline.shape[1] > 0:
        baseline_sel = baseline[channel_indices, :]
        baseline_mean = baseline_sel.mean(axis=-1, keepdims=True)
        eeg_sel = eeg_sel - baseline_mean

    # Segment into windows
    n_samples = eeg_sel.shape[1]
    n_windows = n_samples // window_samples
    all_features = []
    for w in range(n_windows):
        start = w * window_samples
        chunk = eeg_sel[:, start: start + window_samples]
        feat = extract_de_features(chunk, fs=fs)
        all_features.append(feat)

    return np.array(all_features) if all_features else np.empty((0, N_FEATURES))
