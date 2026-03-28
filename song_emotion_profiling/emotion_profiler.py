"""
Core emotion profiling algorithm.

Each track is classified into one of three emotional states (calm, focused,
stressed) by computing its Euclidean distance to hand-tuned centroids in a
normalized 7-dimensional audio feature space.

Features come from the inference_client module (Claude-inferred per-song values
equivalent to Spotify's valence, energy, tempo, danceability etc.).

A steering_score dict is also computed per track, indicating how well the
track would move a listener FROM one state TOWARD another — the primary
input to the EEG baseline-matching algorithm.
"""
import numpy as np

from .models import EmotionalState, FeatureScores, TrackProfile

# Fixed feature ordering — must stay consistent across all arrays in this module.
FEATURE_ORDER = (
    "valence",
    "energy",
    "tempo",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
)

# Hand-tuned centroids in normalized [0, 1] feature space.
# Tempo is normalized by dividing by 250 BPM (physical upper bound).
# Loudness is stored in FeatureScores but excluded from distance math
# because it is highly correlated with energy (double-weighting risk).
CENTROIDS: dict[EmotionalState, np.ndarray] = {
    EmotionalState.CALM: np.array([
        0.65,  # valence: positive affect
        0.25,  # energy: low
        0.30,  # tempo: slow (~75 BPM)
        0.35,  # danceability: low-moderate
        0.80,  # acousticness: high
        0.50,  # instrumentalness: moderate (ambient/acoustic)
        0.05,  # speechiness: minimal
    ]),
    EmotionalState.FOCUSED: np.array([
        0.50,  # valence: neutral
        0.50,  # energy: moderate
        0.50,  # tempo: moderate (~125 BPM)
        0.40,  # danceability: moderate
        0.30,  # acousticness: low-moderate
        0.70,  # instrumentalness: high (lyrics compete with verbal working memory)
        0.04,  # speechiness: minimal
    ]),
    EmotionalState.STRESSED: np.array([
        0.40,  # valence: lower (negative/tense affect)
        0.85,  # energy: very high
        0.75,  # tempo: fast (~188 BPM)
        0.75,  # danceability: high
        0.10,  # acousticness: low
        0.20,  # instrumentalness: low (vocal-driven)
        0.08,  # speechiness: moderate
    ]),
}


_MAX_DIST = float(np.sqrt(len(FEATURE_ORDER)))


def _compute_distances(feature_vec: np.ndarray) -> dict[EmotionalState, float]:
    return {
        state: float(np.linalg.norm(feature_vec - centroid))
        for state, centroid in CENTROIDS.items()
    }


def _classify(
    distances: dict[EmotionalState, float],
) -> tuple[EmotionalState, float]:
    """
    Return the closest state and a confidence score in [0, 1].

    Confidence = 1 - (d_best / avg_d_others)
      - 1.0 → track sits exactly at the winning centroid
      - 0.0 → track is equidistant from best and all others
    """
    best_state = min(distances, key=distances.get)
    d_best = distances[best_state]
    other_dists = [d for s, d in distances.items() if s != best_state]
    avg_other = sum(other_dists) / len(other_dists)

    if avg_other == 0:
        confidence = 1.0
    else:
        confidence = 1.0 - (d_best / avg_other)

    confidence = max(0.0, min(1.0, confidence))
    return best_state, confidence


def _compute_steering_scores(
    distances: dict[EmotionalState, float],
) -> dict[str, float]:
    """
    For each (source, target) pair, compute how well this track steers a
    listener away from source and toward target.

    Formula: raw = (d_source - d_target) / MAX_DIST  ∈ [-1, 1]
    Normalized to [0, 1]: score = (raw + 1) / 2

    score = 1.0  → maximally effective steering (close to target, far from source)
    score = 0.5  → neutral (equidistant)
    score = 0.0  → counterproductive
    """
    scores = {}
    for source in EmotionalState:
        for target in EmotionalState:
            if source == target:
                continue
            key = f"{source.value}_to_{target.value}"
            raw = (distances[source] - distances[target]) / _MAX_DIST
            scores[key] = (raw + 1.0) / 2.0
    return scores


def normalize_features(raw: dict) -> np.ndarray:
    """Convert raw Spotify audio feature dict to normalized [0,1] feature vector."""
    return np.array([
        float(raw["valence"]),
        float(raw["energy"]),
        min(float(raw["tempo"]) / 250.0, 1.0),
        float(raw["danceability"]),
        float(raw["acousticness"]),
        float(raw["instrumentalness"]),
        float(raw["speechiness"]),
    ])


def profile_track(track_meta: dict, raw_features: dict) -> TrackProfile:
    """
    Build a TrackProfile from track metadata and Spotify audio features.

    Args:
        track_meta: dict with keys track_id, name, artist
        raw_features: raw dict from Spotify audio_features API
    """
    feature_vec = normalize_features(raw_features)
    distances = _compute_distances(feature_vec)
    state, confidence = _classify(distances)
    steering = _compute_steering_scores(distances)

    scores = FeatureScores(
        valence=float(raw_features["valence"]),
        energy=float(raw_features["energy"]),
        tempo=float(raw_features["tempo"]),
        danceability=float(raw_features["danceability"]),
        acousticness=float(raw_features["acousticness"]),
        instrumentalness=float(raw_features["instrumentalness"]),
        loudness=float(raw_features["loudness"]),
        speechiness=float(raw_features["speechiness"]),
    )

    return TrackProfile(
        track_id=track_meta["track_id"],
        name=track_meta["name"],
        artist=track_meta["artist"],
        emotional_state=state,
        feature_scores=scores,
        confidence=confidence,
        steering_score=steering,
    )
