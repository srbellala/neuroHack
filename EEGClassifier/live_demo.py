"""
Live demo: Muse headband → EEG emotion classifier → terminal display.

Usage
-----
# With a real Muse headband (pair first, then start the LSL bridge):
    muse-lsl stream &
    python -m EEGClassifier.live_demo

# Simulate without hardware (useful for testing):
    python -m EEGClassifier.live_demo --simulate

# Use a specific trained model:
    python -m EEGClassifier.live_demo --model models/emotion_mlp_best.pt \\
                                       --scaler models/scaler.pkl

Options
-------
--simulate          Use synthetic EEG instead of a real Muse stream.
--sim-state STATE   State to simulate: calm | focused | stressed (default: cycles).
--model PATH        Path to trained model (.pt).
--scaler PATH       Path to saved scaler (.pkl).
--window SEC        Classification window in seconds (default: 1.0).
--step SEC          Step between windows in seconds (default: 0.5).
--vote N            Number of windows for majority vote (default: 5).
--interval SEC      Seconds between printed predictions (default: 2.0).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

from .classifier import EEGClassifier
from .muse_stream import MuseStream, simulate_muse_window
from .features import MUSE_FS

# ── display helpers ──────────────────────────────────────────────────────────

_STATE_ICON = {
    "calm":     "🧘  CALM",
    "focused":  "🎯  FOCUSED",
    "stressed": "⚡  STRESSED",
}

_STATE_COLOR = {
    "calm":     "\033[94m",   # blue
    "focused":  "\033[92m",   # green
    "stressed": "\033[91m",   # red
}
_RESET = "\033[0m"


def _render_bar(history: list[str], width: int = 40) -> str:
    """Render a rolling history bar of state predictions."""
    chars = {"calm": "▓", "focused": "▪", "stressed": "█"}
    bar = "".join(chars.get(s, "?") for s in history[-width:])
    return f"[{bar:<{width}}]"


def _print_state(state: str, confidence_str: str = "", history: list[str] | None = None) -> None:
    color = _STATE_COLOR.get(state, "")
    label = _STATE_ICON.get(state, state.upper())
    bar   = _render_bar(history) if history else ""
    print(f"\r  {color}{label}{_RESET}  {confidence_str}  {bar}      ", end="", flush=True)


# ── live loop (real Muse) ────────────────────────────────────────────────────

def run_live(
    classifier: EEGClassifier,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
    vote: int = 5,
    interval: float = 2.0,
) -> None:
    """Stream from Muse, classify, and print predictions."""
    window_buf: list[np.ndarray] = []

    def on_window(w: np.ndarray) -> None:
        window_buf.append(w)
        # Keep last `vote` windows
        if len(window_buf) > vote:
            window_buf.pop(0)

    stream = MuseStream(
        window_sec=window_sec,
        step_sec=step_sec,
        on_window=on_window,
    )
    stream.connect()
    stream.start()

    history: list[str] = []
    print("\nClassifying EEG state. Press Ctrl-C to stop.\n")

    try:
        while True:
            time.sleep(interval)
            if len(window_buf) < 1:
                print("\r  Waiting for EEG data …", end="", flush=True)
                continue

            windows = np.stack(window_buf, axis=0)
            state = classifier.detect_state(windows, n_vote=vote)
            history.append(state)
            _print_state(state, history=history)
    except KeyboardInterrupt:
        print("\n\nStopping …")
    finally:
        stream.stop()


# ── simulation loop ──────────────────────────────────────────────────────────

def run_simulate(
    classifier: EEGClassifier,
    sim_state: str | None = None,
    vote: int = 5,
    interval: float = 2.0,
    window_sec: float = 1.0,
) -> None:
    """Classify synthetic EEG and print predictions (no hardware needed)."""
    states = ["calm", "focused", "stressed"]
    state_cycle = itertools.cycle(states)

    history: list[str] = []
    window_buf: list[np.ndarray] = []

    print("\nSimulation mode — no Muse required. Press Ctrl-C to stop.\n")
    print(f"  Cycling through states: {states}\n")

    tick = 0
    try:
        while True:
            if sim_state:
                current_sim = sim_state
            else:
                # Change simulated state every 6 ticks
                if tick % 6 == 0:
                    current_sim = next(state_cycle)

            w = simulate_muse_window(
                state=current_sim,
                fs=MUSE_FS,
                window_sec=window_sec,
                seed=tick,
            )
            window_buf.append(w)
            if len(window_buf) > vote:
                window_buf.pop(0)

            windows = np.stack(window_buf, axis=0)
            predicted = classifier.detect_state(windows, n_vote=vote)
            history.append(predicted)

            match = "✓" if predicted == current_sim else "✗"
            print(
                f"\r  sim={current_sim:<8s}  pred={predicted:<8s}  {match}  "
                f"{_render_bar(history)}   ",
                end="",
                flush=True,
            )

            tick += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nStopped.")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuroTune EEG live demo — Muse headband or simulation"
    )
    parser.add_argument("--simulate",  action="store_true",
                        help="Run with synthetic EEG (no Muse required)")
    parser.add_argument("--sim-state", choices=["calm", "focused", "stressed"],
                        default=None,
                        help="Fix the simulated state (default: cycles automatically)")
    parser.add_argument("--model",  type=str, default=None,
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--scaler", type=str, default=None,
                        help="Path to trained scaler (.pkl)")
    parser.add_argument("--window", type=float, default=1.0,
                        help="Classification window length in seconds (default: 1.0)")
    parser.add_argument("--step",   type=float, default=0.5,
                        help="Step between windows in seconds (default: 0.5)")
    parser.add_argument("--vote",   type=int,   default=5,
                        help="Majority-vote window count (default: 5)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between printed predictions (default: 2.0)")
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroTune — EEG Emotion Classifier  (DREAMER / Muse)")
    print("=" * 60)

    # Load model
    try:
        clf = EEGClassifier(
            model_path=args.model,
            scaler_path=args.scaler,
            fs=MUSE_FS,
        )
        print(f"  Model loaded successfully.\n")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n", file=sys.stderr)
        sys.exit(1)

    if args.simulate:
        run_simulate(
            classifier=clf,
            sim_state=args.sim_state,
            vote=args.vote,
            interval=args.interval,
            window_sec=args.window,
        )
    else:
        run_live(
            classifier=clf,
            window_sec=args.window,
            step_sec=args.step,
            vote=args.vote,
            interval=args.interval,
        )


if __name__ == "__main__":
    main()
