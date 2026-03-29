"""
Train the emotion MLP on the DREAMER dataset.

Usage
-----
python -m EEGClassifier.train --mat path/to/DREAMER.mat --out models/

The DREAMER .mat file is available on Kaggle:
  https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions
  or from the original DREAMER paper authors.

Label mapping (valence × arousal → EmotionalState)
---------------------------------------------------
valence > 3 AND arousal > 3  →  FOCUSED   (positive, activated)
valence > 3 AND arousal ≤ 3  →  CALM      (positive, relaxed)
valence ≤ 3 AND arousal > 3  →  STRESSED  (negative, activated)
valence ≤ 3 AND arousal ≤ 3  →  CALM      (low-energy / subdued)
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .features import (
    DREAMER_FS,
    DREAMER_MUSE_CHANNEL_INDICES,
    N_FEATURES,
    extract_features_from_trial,
)
from .model import EmotionMLP, STATE_TO_IDX

# ── label helpers ────────────────────────────────────────────────────────────

VALENCE_THRESHOLD = 3.0
AROUSAL_THRESHOLD = 3.0


def _map_label(valence: float, arousal: float) -> int:
    if valence > VALENCE_THRESHOLD and arousal > AROUSAL_THRESHOLD:
        return STATE_TO_IDX["focused"]
    if valence > VALENCE_THRESHOLD and arousal <= AROUSAL_THRESHOLD:
        return STATE_TO_IDX["calm"]
    if valence <= VALENCE_THRESHOLD and arousal > AROUSAL_THRESHOLD:
        return STATE_TO_IDX["stressed"]
    return STATE_TO_IDX["calm"]   # low valence + low arousal → subdued/calm


# ── DREAMER loading ──────────────────────────────────────────────────────────

def load_dreamer(mat_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load DREAMER.mat and extract DE features + labels.

    Returns
    -------
    X : (N, N_FEATURES)  float32
    y : (N,)             int64
    """
    print(f"Loading {mat_path} …")
    mat = scio.loadmat(mat_path, verify_compressed_data_integrity=False)
    dreamer = mat["DREAMER"][0, 0]
    data = dreamer["Data"][0]

    n_subjects = len(data)
    n_trials = len(data[0]["EEG"][0, 0]["stimuli"][0, 0])
    print(f"  {n_subjects} subjects × {n_trials} trials")

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for subj in range(n_subjects):
        subj_data = data[subj]
        eeg_data = subj_data["EEG"][0, 0]

        for trial in range(n_trials):
            # ── stimulus EEG (n_samples, 14) → (14, n_samples) ──
            stim_raw = eeg_data["stimuli"][0, 0][trial, 0]
            if stim_raw.ndim == 1:
                continue   # malformed entry
            stim = stim_raw[:, :14].T.astype(np.float64)  # (14, n_samples)

            # ── baseline EEG ──
            base_raw = eeg_data["baseline"][0, 0][trial, 0]
            base = base_raw[:, :14].T.astype(np.float64)   # (14, n_baseline_samples)

            # ── labels ──
            valence = float(subj_data["ScoreValence"][0, 0][trial, 0])
            arousal = float(subj_data["ScoreArousal"][0, 0][trial, 0])
            label = _map_label(valence, arousal)

            # ── feature extraction ──
            feats = extract_features_from_trial(
                eeg=stim,
                baseline=base,
                fs=DREAMER_FS,
                channel_indices=DREAMER_MUSE_CHANNEL_INDICES,
                window_samples=DREAMER_FS,   # 1-second windows
            )
            if feats.shape[0] == 0:
                continue

            all_X.append(feats.astype(np.float32))
            all_y.append(np.full(len(feats), label, dtype=np.int64))

        print(f"  Subject {subj + 1:02d}/{n_subjects} done", end="\r")

    print()
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"  Total windows: {len(X)}  |  class counts: {np.bincount(y)}")
    return X, y


# ── training loop ────────────────────────────────────────────────────────────

def train(
    mat_path: str,
    out_dir: str = "models",
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── data ──
    X, y = load_dreamer(mat_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler_path = out_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {scaler_path}")

    # ── class-weighted loss to handle imbalance ──
    counts = np.bincount(y_train, minlength=3).astype(float)
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = EmotionMLP(input_dim=N_FEATURES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_model_path = out_path / "emotion_mlp_best.pt"

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        # ── eval ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_ds)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={acc:.4f}  (best={best_acc:.4f})")

    # ── also save final model ──
    final_path = out_path / "emotion_mlp_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nBest val accuracy : {best_acc:.4f}")
    print(f"Model saved       → {best_model_path}")
    print(f"Final model saved → {final_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG emotion classifier on DREAMER")
    parser.add_argument("--mat",    required=True,          help="Path to DREAMER.mat")
    parser.add_argument("--out",    default="models",       help="Output directory for model + scaler")
    parser.add_argument("--epochs", type=int, default=60,   help="Training epochs")
    parser.add_argument("--batch",  type=int, default=256,  help="Batch size")
    parser.add_argument("--lr",     type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed",   type=int, default=42,   help="Random seed")
    args = parser.parse_args()

    train(
        mat_path=args.mat,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
    )
