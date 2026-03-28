"""
Claude-powered audio feature inference.

Since Spotify's audio-features API is restricted for new apps, we use Claude's
music knowledge to estimate per-song feature values (valence, energy, tempo, etc.)
that are equivalent to what the Spotify API would have returned.

Tracks are batched (20 per request) and structured outputs guarantee valid JSON.
"""
import os
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

_BATCH_SIZE = 20


class _TrackFeatures(BaseModel):
    track_id: str
    valence: float = Field(ge=0.0, le=1.0, description="Musical positiveness (0=sad/dark, 1=happy/euphoric)")
    energy: float = Field(ge=0.0, le=1.0, description="Intensity and activity (0=slow/quiet, 1=loud/fast)")
    tempo: float = Field(ge=40.0, le=220.0, description="Beats per minute")
    danceability: float = Field(ge=0.0, le=1.0, description="How suitable for dancing")
    acousticness: float = Field(ge=0.0, le=1.0, description="Acoustic vs electronic (1=fully acoustic)")
    instrumentalness: float = Field(ge=0.0, le=1.0, description="Absence of vocals (1=no vocals)")
    speechiness: float = Field(ge=0.0, le=1.0, description="Spoken words (>0.33 = rap/podcast)")
    loudness: float = Field(ge=-60.0, le=0.0, description="Overall loudness in dBFS")


class _BatchResponse(BaseModel):
    tracks: list[_TrackFeatures]


_SYSTEM_PROMPT = """\
You are a music expert estimating Spotify-style audio features for songs.
Be precise and calibrated against known reference points:

Calm/ambient reference — "Weightless" by Marconi Union:
  valence=0.28, energy=0.05, tempo=65, danceability=0.28, acousticness=0.82,
  instrumentalness=0.93, speechiness=0.03, loudness=-24

Focused/instrumental reference — "Experience" by Ludovico Einaudi:
  valence=0.45, energy=0.18, tempo=72, danceability=0.30, acousticness=0.95,
  instrumentalness=0.97, speechiness=0.03, loudness=-20

Stressed/high-energy reference — "SICKO MODE" by Travis Scott:
  valence=0.42, energy=0.82, tempo=155, danceability=0.78, acousticness=0.03,
  instrumentalness=0.0, speechiness=0.31, loudness=-6

Always return the exact track_id provided — do not modify it."""


def infer_audio_features(
    tracks: list[dict],
    api_key: Optional[str] = None,
) -> dict[str, dict]:
    """
    Infer Spotify-style audio features for each track using Claude.

    Args:
        tracks: list of dicts with keys track_id, name, artist
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)

    Returns:
        dict of track_id -> feature dict (same shape as Spotify audio_features response)
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    results: dict[str, dict] = {}

    for batch_start in range(0, len(tracks), _BATCH_SIZE):
        batch = tracks[batch_start : batch_start + _BATCH_SIZE]

        track_list = "\n".join(
            f'{i + 1}. track_id="{t["track_id"]}" | "{t["name"]}" by {t["artist"]}'
            for i, t in enumerate(batch)
        )

        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    "Estimate Spotify audio features for each song below. "
                    "Return the exact track_id string for each entry.\n\n"
                    f"{track_list}"
                ),
            }],
            output_format=_BatchResponse,
        )

        for tf in response.parsed_output.tracks:
            results[tf.track_id] = {
                "id": tf.track_id,
                "valence": tf.valence,
                "energy": tf.energy,
                "tempo": tf.tempo,
                "danceability": tf.danceability,
                "acousticness": tf.acousticness,
                "instrumentalness": tf.instrumentalness,
                "speechiness": tf.speechiness,
                "loudness": tf.loudness,
            }

    return results
