"""
Baseline Matching — CLI entry point.

Usage:
    # Live session with Muse headband
    python -m baseline_matching.main --baseline calm

    # Simulated session (no hardware needed)
    python -m baseline_matching.main --baseline calm --simulate

    # Simulate a specific drift sequence
    python -m baseline_matching.main --baseline focused --simulate --sim-states calm stressed stressed stressed calm
"""
import argparse
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from song_emotion_profiling import TrackLibrary
from song_emotion_profiling.models import EmotionalState

from .session import BaselineSession
from .spotify_player import SpotifyPlayer

_STATE_LABELS = {
    EmotionalState.CALM: "CALM    ",
    EmotionalState.FOCUSED: "FOCUSED ",
    EmotionalState.STRESSED: "STRESSED",
}


def _parse_state(value: str) -> EmotionalState:
    try:
        return EmotionalState(value.lower())
    except ValueError:
        valid = [s.value for s in EmotionalState]
        raise argparse.ArgumentTypeError(
            f"Invalid state '{value}'. Choose from: {', '.join(valid)}"
        )


def _on_state_change(current: EmotionalState, baseline: EmotionalState) -> None:
    drift = "" if current == baseline else f"  ⚠ drift from {baseline.value}"
    print(f"  State: {_STATE_LABELS[current]}{drift}")


def _run_simulate(session: BaselineSession, sim_states: list[str]) -> None:
    """Replay a list of state strings through the session to test switching logic."""
    print("\n[Simulate] Replaying state sequence:", " → ".join(sim_states))
    print("-" * 50)

    for state_str in sim_states:
        # Inject the state directly — bypasses EEG → feature extraction → model,
        # which avoids training/inference feature-space mismatches in demos.
        state = _parse_state(state_str)
        session._classify_and_act_with_state(state)
        time.sleep(1.0)

    print("-" * 50)
    print("[Simulate] Done.")


def _run_live(session: BaselineSession) -> None:
    """Live session using Muse headband via LSL."""
    from EEGClassifier.muse_stream import MuseStream

    stream = MuseStream(
        window_sec=1.0,
        step_sec=0.5,
        on_window=session.on_window,
    )

    try:
        stream.start()
        print(f"\nListening for EEG... Press Ctrl+C to stop.\n")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping session.")
    finally:
        stream.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroTune baseline matching session.")
    parser.add_argument(
        "--baseline", type=_parse_state, required=True,
        help="Your target emotional state: calm, focused, or stressed"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run without hardware using synthetic EEG windows"
    )
    parser.add_argument(
        "--sim-states", nargs="+", default=["calm", "stressed", "stressed", "stressed", "calm"],
        metavar="STATE",
        help="Sequence of states to simulate (default: calm stressed stressed stressed calm)"
    )
    parser.add_argument(
        "--cache", default=os.environ.get("CACHE_FILE", ".track_cache.json"),
        help="Path to track cache file"
    )
    args = parser.parse_args()

    # Load track library
    library = TrackLibrary(cache_path=args.cache)
    summary = library.summary()
    if summary["total"] == 0:
        print(
            "No tracks found in cache. Run song emotion profiling first:\n"
            "  python -m song_emotion_profiling.main"
        )
        sys.exit(1)

    by_state = summary["by_state"]
    print(f"Loaded {summary['total']} tracks  "
          f"(calm: {by_state['calm']} | focused: {by_state['focused']} | stressed: {by_state['stressed']})")

    # Check steering candidates exist
    other_states = [s for s in EmotionalState if s != args.baseline]
    for drift_state in other_states:
        n = len(library.get_steering_candidates(drift_state, args.baseline, top_n=100))
        if n == 0:
            print(f"Warning: no steering candidates for {drift_state.value} → {args.baseline.value}")

    # Load classifier
    try:
        from EEGClassifier import EEGClassifier
        classifier = EEGClassifier()
    except Exception as e:
        print(f"Failed to load EEGClassifier: {e}")
        sys.exit(1)

    # Set up Spotify player
    try:
        player = SpotifyPlayer()
    except Exception as e:
        print(f"Failed to connect to Spotify: {e}")
        sys.exit(1)

    # Create session
    session = BaselineSession(
        baseline=args.baseline,
        library=library,
        player=player,
        classifier=classifier,
        on_state_change=_on_state_change,
    )

    print(f"\nBaseline set to: {args.baseline.value.upper()}")
    print(f"Will switch tracks after {3} consecutive off-baseline classifications "
          f"(cooldown: 30s)\n")

    if args.simulate:
        _run_simulate(session, args.sim_states)
    else:
        _run_live(session)


if __name__ == "__main__":
    main()
