import os
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

_API_BASE = "https://api.spotify.com/v1"

SCOPES = "user-read-recently-played user-library-read"
_BATCH_SIZE = 100


def build_client() -> spotipy.Spotify:
    auth_manager = SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ["SPOTIPY_REDIRECT_URI"],
        scope=SCOPES,
        cache_path=".spotify_token_cache",
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def _extract_track_meta(item: dict) -> dict | None:
    track = item.get("track") or item
    if not track or not track.get("id"):
        return None
    artists = track.get("artists") or []
    return {
        "track_id": track["id"],
        "name": track.get("name", "Unknown"),
        "artist": artists[0]["name"] if artists else "Unknown",
        "artist_id": artists[0]["id"] if artists else None,
    }


def fetch_recent_tracks(client: spotipy.Spotify, limit: int = 50) -> list[dict]:
    results = client.current_user_recently_played(limit=limit)
    tracks = []
    for item in results.get("items", []):
        meta = _extract_track_meta(item)
        if meta:
            tracks.append(meta)
    return tracks


def fetch_saved_tracks(client: spotipy.Spotify, max_tracks: int = 500) -> list[dict]:
    tracks = []
    offset = 0
    page_size = 50

    while len(tracks) < max_tracks:
        results = client.current_user_saved_tracks(limit=page_size, offset=offset)
        items = results.get("items", [])
        if not items:
            break
        for item in items:
            meta = _extract_track_meta(item)
            if meta:
                tracks.append(meta)
        if not results.get("next"):
            break
        offset += page_size

    return tracks[:max_tracks]


def _chunk(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _auth_headers(client: spotipy.Spotify) -> dict:
    """Get a valid Bearer token from spotipy's auth manager."""
    token = client.auth_manager.get_access_token(as_dict=False)
    return {"Authorization": f"Bearer {token}"}


def fetch_audio_features(
    client: spotipy.Spotify, track_ids: list[str]
) -> dict[str, dict]:
    """
    Batch fetch audio features (valence, energy, tempo, etc.) for up to 100 tracks at a time.

    Uses requests directly instead of spotipy because spotipy uses the old
    /audio-features/?ids= URL format which returns 403 on apps created after Feb 2026.
    """
    headers = _auth_headers(client)
    features = {}
    for batch in _chunk(track_ids, _BATCH_SIZE):
        resp = requests.get(
            f"{_API_BASE}/audio-features?ids={','.join(batch)}",
            headers=headers,
        )
        resp.raise_for_status()
        for f in resp.json().get("audio_features", []):
            if f is not None:
                features[f["id"]] = f
    return features


def fetch_all_tracks(
    client: spotipy.Spotify,
    include_saved: bool = True,
    include_recent: bool = True,
) -> list[dict]:
    tracks: list[dict] = []
    seen: set[str] = set()

    if include_saved:
        for t in fetch_saved_tracks(client):
            if t["track_id"] not in seen:
                seen.add(t["track_id"])
                tracks.append(t)

    if include_recent:
        for t in fetch_recent_tracks(client):
            if t["track_id"] not in seen:
                seen.add(t["track_id"])
                tracks.append(t)

    return tracks
