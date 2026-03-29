"""
EEGClassifier — real-time EEG emotion detection for NeuroTune.

Quick-start
-----------
1. Train on DREAMER::

       python -m EEGClassifier.train --mat DREAMER.mat --out EEGClassifier/models

2. Live demo with Muse headband::

       muse-lsl stream &               # start the LSL bridge
       python -m EEGClassifier.live_demo

3. Simulate (no hardware needed)::

       python -m EEGClassifier.live_demo --simulate

Integration with NeuroTune pipeline
------------------------------------
    from EEGClassifier import EEGClassifier
    from song_emotion_profiling.models import EmotionalState

    clf = EEGClassifier()
    state: EmotionalState = clf.detect_emotional_state(eeg_windows)
"""

from .classifier import EEGClassifier

__all__ = ["EEGClassifier"]
