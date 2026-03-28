# NeuroTune

Brainwave-controlled music system. Monitors EEG brain state in real time and automatically switches tracks from your Spotify library to steer you back toward a baseline mood.

## Architecture

```
eeg_detection/          → detects current brain state (calm / focused / stressed)
song_emotion_profiling/ → profiles your Spotify library by emotional state  ← THIS MODULE
baseline_matching/      → picks a track to steer you back toward baseline
```

Data flows in one direction:

```
EEG headset → eeg_detection → baseline_matching → song_emotion_profiling → Spotify playback
```

---

## Song Emotion Profiling

Builds an emotional profile of a user's Spotify listening history. Each track is classified as **calm**, **focused**, or **stressed** and assigned a **steering score** indicating how effectively it moves a listener from one state toward another.

### How It Works

1. **Fetch** — pulls saved tracks and recently played from Spotify
2. **Infer** — uses Claude to estimate audio features per song (valence, energy, tempo, danceability, acousticness, instrumentalness, speechiness)
3. **Classify** — computes Euclidean distance from each track's feature vector to 3 hand-tuned state centroids
4. **Score** — calculates a `steering_score` for every source→target state pair (e.g. `stressed_to_calm`)
5. **Cache** — saves all profiles to `.track_cache.json` so Claude is only called once per track

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your API keys (see below)
```

**Required API keys in `.env`:**

| Key | Where to get it |
|-----|----------------|
| `SPOTIPY_CLIENT_ID` | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → Create app |
| `SPOTIPY_CLIENT_SECRET` | Same Spotify app → Settings |
| `SPOTIPY_REDIRECT_URI` | Set to `http://127.0.0.1:8888/callback` in both `.env` and Spotify dashboard |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API Keys |

### CLI Usage

Profile your library and show steering candidates:

```bash
# Profile library (runs once, results cached)
python -m song_emotion_profiling.main

# Show which songs steer you from stressed → calm
python -m song_emotion_profiling.main --from stressed --to calm --top-n 5

# Skip saved tracks, only use recently played
python -m song_emotion_profiling.main --no-saved --from focused --to calm
```

**Example output:**
```
Library: 51 tracks  (calm: 14 | focused: 10 | stressed: 27)

Top 5 steering candidates: STRESSED -> CALM
   1. [0.89] "Falling for a Friend" — grentperez
       valence=0.72  energy=0.21  tempo=84bpm  instrumentalness=0.12
   2. [0.84] "Add Up My Love" — Clairo
       valence=0.68  energy=0.24  tempo=91bpm  instrumentalness=0.08
```

---

## Integration Guide for Other Algorithms

### For EEG Emotion Detection (algo 1)

Import the `EmotionalState` enum so your output type is compatible with the rest of the system:

```python
from song_emotion_profiling import EmotionalState

# Your EEG algo should output one of these three values:
detected_state = EmotionalState.STRESSED   # or CALM, FOCUSED
```

That's all — algo 1 doesn't need to read the track library.

---

### For Baseline Matching (algo 3)

This is the primary consumer. Load the library and call `get_steering_candidates()` with the current EEG state and the user's baseline:

```python
from song_emotion_profiling import TrackLibrary, EmotionalState

# Load once at startup (reads from .track_cache.json)
library = TrackLibrary()

# User sets their baseline at session start
baseline = EmotionalState.FOCUSED

# EEG algo detects the user has drifted to stressed
current_state = EmotionalState.STRESSED

# Get the top tracks that will steer them back
candidates = library.get_steering_candidates(
    from_state=current_state,
    to_state=baseline,
    top_n=5
)

# Each candidate has a track_id ready for Spotify playback
for track in candidates:
    print(track.track_id, track.name, track.steering_score[f"{current_state.value}_to_{baseline.value}"])

# Pass the top candidate's track_id to the Spotify playback API
best_track_id = candidates[0].track_id
```

#### Available TrackProfile fields

```python
track.track_id          # Spotify track ID → use with playback API
track.name              # "Falling for a Friend"
track.artist            # "grentperez"
track.emotional_state   # EmotionalState.CALM
track.confidence        # 0.84 — how strongly it fits that state
track.steering_score    # dict — e.g. {"stressed_to_calm": 0.89, "calm_to_stressed": 0.11, ...}
track.feature_scores    # valence, energy, tempo, danceability, acousticness, instrumentalness, loudness, speechiness
```

#### Other useful queries

```python
# All calm songs sorted by confidence
calm_tracks = library.query_by_state(EmotionalState.CALM)

# Summary of library
print(library.summary())
# {"total": 51, "by_state": {"calm": 14, "focused": 10, "stressed": 27}}

# Look up a specific track
profile = library.get_profile(track_id="5xAVweJkMzCTib1YRDuZJi")
```

---

## State Definitions

| State | Characteristics | Example tracks |
|-------|----------------|----------------|
| **calm** | Low energy, slow tempo, acoustic, positive valence | Ambient, acoustic, lo-fi |
| **focused** | Moderate energy, often instrumental, minimal lyrics | Post-rock, jazz, film scores |
| **stressed** | High energy, fast tempo, intense, vocal-driven | Hip-hop, pop, EDM, rock |
