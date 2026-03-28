from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum


class EmotionalState(str, Enum):
    CALM = "calm"
    FOCUSED = "focused"
    STRESSED = "stressed"


@dataclass
class FeatureScores:
    valence: float
    energy: float
    tempo: float
    danceability: float
    acousticness: float
    instrumentalness: float
    loudness: float
    speechiness: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FeatureScores:
        return cls(**d)


@dataclass
class TrackProfile:
    track_id: str
    name: str
    artist: str
    emotional_state: EmotionalState
    feature_scores: FeatureScores
    confidence: float
    # Keys like "calm_to_focused", "stressed_to_calm", etc.
    steering_score: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "artist": self.artist,
            "emotional_state": self.emotional_state.value,
            "feature_scores": self.feature_scores.to_dict(),
            "confidence": self.confidence,
            "steering_score": self.steering_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrackProfile:
        return cls(
            track_id=d["track_id"],
            name=d["name"],
            artist=d["artist"],
            emotional_state=EmotionalState(d["emotional_state"]),
            feature_scores=FeatureScores.from_dict(d["feature_scores"]),
            confidence=d["confidence"],
            steering_score=d["steering_score"],
        )
