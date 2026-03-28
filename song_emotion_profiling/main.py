"""
CLI entry point for Song Emotion Profiling.

Usage:
    python -m song_emotion_profiling.main [options]

Options:
    --saved          Include saved/liked tracks (default: True)
    --recent         Include recently played tracks (default: True)
    --no-saved       Skip saved tracks
    --no-recent      Skip recently played tracks
    --from STATE     Show steering candidates departing from this state
    --to STATE       Show steering candidates arriving at this state
    --top-n N        Number of steering candidates to show (default: 10)

Example — profile library and show what songs steer stressed→calm:
    python -m song_emotion_profiling.main --from stressed --to calm --top-n 5
"""
import argparse
import os
import sys

from dotenv import load_dotenv

# load_dotenv must run before any import that reads env vars
load_dotenv()

from .emotion_profiler import profile_track
from .inference_client import infer_audio_features
from .models import EmotionalState
from .spotify_client import build_client, fetch_all_tracks
from .track_library import TrackLibrary


def _parse_state(value: str) -> EmotionalState:
    try:
        return EmotionalState(value.lower())
    except ValueError:
        valid = [s.value for s in EmotionalState]
        raise argparse.ArgumentTypeError(
            f"Invalid state '{value}'. Choose from: {', '.join(valid)}"
        )


def _print_summary(summary: dict) -> None:
    total = summary["total"]
    by_state = summary["by_state"]
    state_str = " | ".join(f"{s}: {c}" for s, c in by_state.items())
    print(f"\nLibrary: {total} tracks  ({state_str})")


def _print_steering(
    library: TrackLibrary,
    from_state: EmotionalState,
    to_state: EmotionalState,
    top_n: int,
) -> None:
    candidates = library.get_steering_candidates(from_state, to_state, top_n)
    key = f"{from_state.value}_to_{to_state.value}"
    print(f"\nTop {top_n} steering candidates: {from_state.value.upper()} -> {to_state.value.upper()}")
    for i, profile in enumerate(candidates, 1):
        score = profile.steering_score.get(key, 0.0)
        print(f"  {i:2}. [{score:.2f}] \"{profile.name}\" — {profile.artist}")
        f = profile.feature_scores
        print(
            f"       valence={f.valence:.2f}  energy={f.energy:.2f}  "
            f"tempo={f.tempo:.0f}bpm  instrumentalness={f.instrumentalness:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile your Spotify library by emotional state.")
    parser.add_argument("--saved", dest="saved", action="store_true", default=True)
    parser.add_argument("--no-saved", dest="saved", action="store_false")
    parser.add_argument("--recent", dest="recent", action="store_true", default=True)
    parser.add_argument("--no-recent", dest="recent", action="store_false")
    parser.add_argument("--from", dest="from_state", type=_parse_state, default=None)
    parser.add_argument("--to", dest="to_state", type=_parse_state, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--debug-tags", action="store_true",
                        help="Print raw Last.fm tags for each artist (useful for tuning)")
    args = parser.parse_args()

    cache_path = os.environ.get("CACHE_FILE", ".track_cache.json")
    library = TrackLibrary(cache_path=cache_path)

    print("Connecting to Spotify...")
    try:
        client = build_client()
    except Exception as e:
        print(f"Error authenticating with Spotify: {e}", file=sys.stderr)
        sys.exit(1)

    print("Fetching tracks...")
    all_tracks = fetch_all_tracks(
        client,
        include_saved=args.saved,
        include_recent=args.recent,
    )

    cached_ids = library.get_cached_ids()
    new_tracks = [t for t in all_tracks if t["track_id"] not in cached_ids]

    print(f"Found {len(all_tracks)} tracks ({len(cached_ids)} cached, {len(new_tracks)} new)")

    if new_tracks:
        print(f"Inferring audio features for {len(new_tracks)} tracks via Claude...")
        features = infer_audio_features(new_tracks)

        profiles = []
        skipped = 0
        for track in new_tracks:
            tid = track["track_id"]
            if tid not in features:
                skipped += 1
                continue
            profiles.append(profile_track(track, features[tid]))

        if skipped:
            print(f"Skipped {skipped} tracks (inference returned no data)")

        library.add_profiles(profiles)
        print(f"Profiled and cached {len(profiles)} new tracks")

    _print_summary(library.summary())

    if args.from_state and args.to_state:
        _print_steering(library, args.from_state, args.to_state, args.top_n)
    elif args.from_state or args.to_state:
        print("\nNote: provide both --from and --to to see steering candidates")


if __name__ == "__main__":
    main()
