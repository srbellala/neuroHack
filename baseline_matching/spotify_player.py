"""
Spotify playback control.

Uses direct requests calls (not spotipy) to avoid deprecated endpoint issues.
Requires the user-modify-playback-state and user-read-playback-state scopes.
"""
import os

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

_API_BASE = "https://api.spotify.com/v1"

PLAYBACK_SCOPES = (
    "user-read-recently-played "
    "user-library-read "
    "user-modify-playback-state "
    "user-read-playback-state"
)


def build_playback_client() -> spotipy.Spotify:
    """Build a Spotify client with playback scopes."""
    auth_manager = SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ["SPOTIPY_REDIRECT_URI"],
        scope=PLAYBACK_SCOPES,
        cache_path=".spotify_token_cache",
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def _auth_headers(client: spotipy.Spotify) -> dict:
    token = client.auth_manager.get_access_token(as_dict=False)
    return {"Authorization": f"Bearer {token}"}


class SpotifyPlayer:
    """Controls Spotify playback on the user's active device."""

    def __init__(self, client: spotipy.Spotify | None = None):
        self._client = client or build_playback_client()

    def play_track(self, track_id: str) -> bool:
        """
        Start playing a track on the user's active Spotify device.

        Returns True on success, False if no active device is found.
        """
        headers = _auth_headers(self._client)
        headers["Content-Type"] = "application/json"

        resp = requests.put(
            f"{_API_BASE}/me/player/play",
            headers=headers,
            json={"uris": [f"spotify:track:{track_id}"]},
        )

        if resp.status_code == 404:
            print("No active Spotify device found. Open Spotify on any device first.")
            return False

        resp.raise_for_status()
        return True

    def get_current_track(self) -> dict | None:
        """Return currently playing track metadata, or None if nothing is playing."""
        headers = _auth_headers(self._client)
        resp = requests.get(f"{_API_BASE}/me/player/currently-playing", headers=headers)

        if resp.status_code == 204 or not resp.content:
            return None

        resp.raise_for_status()
        data = resp.json()
        item = data.get("item")
        if not item:
            return None

        artists = item.get("artists") or []
        return {
            "track_id": item["id"],
            "name": item["name"],
            "artist": artists[0]["name"] if artists else "Unknown",
            "is_playing": data.get("is_playing", False),
        }
