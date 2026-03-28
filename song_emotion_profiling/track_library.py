"""
Persistence and query layer for TrackProfiles.

The cache is a single JSON file keyed by track_id. Profiles are loaded on
init and written after any mutation. The get_steering_candidates method is
the primary integration surface for the EEG baseline-matching algorithm.
"""
import json
import os

from .models import EmotionalState, TrackProfile


class TrackLibrary:
    def __init__(self, cache_path: str = ".track_cache.json"):
        self._cache_path = cache_path
        self._profiles: dict[str, TrackProfile] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        with open(self._cache_path, "r") as f:
            data = json.load(f)
        for track_id, raw in data.items():
            try:
                self._profiles[track_id] = TrackProfile.from_dict(raw)
            except (KeyError, ValueError):
                # Skip malformed entries from an old schema version
                continue

    def _save_cache(self) -> None:
        with open(self._cache_path, "w") as f:
            json.dump(
                {tid: p.to_dict() for tid, p in self._profiles.items()},
                f,
                indent=2,
            )

    def add_profiles(self, profiles: list[TrackProfile]) -> None:
        """Merge new profiles into the library and persist."""
        for profile in profiles:
            self._profiles[profile.track_id] = profile
        self._save_cache()

    def get_cached_ids(self) -> set[str]:
        """Return track IDs already profiled — skip these when fetching features."""
        return set(self._profiles.keys())

    def get_profile(self, track_id: str) -> TrackProfile | None:
        return self._profiles.get(track_id)

    def query_by_state(self, state: EmotionalState) -> list[TrackProfile]:
        """All profiles matching a state, sorted by confidence descending."""
        matches = [p for p in self._profiles.values() if p.emotional_state == state]
        matches.sort(key=lambda p: p.confidence, reverse=True)
        return matches

    def get_steering_candidates(
        self,
        from_state: EmotionalState,
        to_state: EmotionalState,
        top_n: int = 10,
    ) -> list[TrackProfile]:
        """
        Return tracks ranked by how well they steer a listener from
        from_state toward to_state. Primary API for baseline matching.
        """
        key = f"{from_state.value}_to_{to_state.value}"
        profiles = list(self._profiles.values())
        profiles.sort(key=lambda p: p.steering_score.get(key, 0.0), reverse=True)
        return profiles[:top_n]

    def all_profiles(self) -> list[TrackProfile]:
        return list(self._profiles.values())

    def summary(self) -> dict:
        counts = {state: 0 for state in EmotionalState}
        for p in self._profiles.values():
            counts[p.emotional_state] += 1
        return {
            "total": len(self._profiles),
            "by_state": {state.value: count for state, count in counts.items()},
        }
