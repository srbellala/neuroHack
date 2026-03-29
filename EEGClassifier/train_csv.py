"""
Train the emotion MLP on the Kaggle EEG Brainwave Dataset (CSV format).

Dataset: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions
Download the dataset and unzip it into a folder, then pass that folder with --data.

Usage
-----
# First, inspect the CSV format:
python -m EEGClassifier.train_csv --data path/to/csv/folder --peek

# Then train:
python -m EEGClassifier.train_csv --data path/to/csv/folder --out EEGClassifier/models

Label mapping
-------------
positive  →  FOCUSED  (high valence, high arousal)
neutral   →  CALM     (baseline resting state)
negative  →  STRESSED (low valence, high arousal)
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .features import N_FEATURES, extract_de_features, DREAMER_FS, MUSE_CHANNEL_NAMES
from .model import EmotionMLP, STATE_TO_IDX

# ── label mapping ─────────────────────────────────────────────────────────────

# Maps CSV label values → our 3 states
# Handles string labels, integer labels, and common variations
_LABEL_MAP: dict[str, int] = {
    # positive → focused
    "positive": STATE_TO_IDX["focused"],
    "pos":      STATE_TO_IDX["focused"],
    "1":        STATE_TO_IDX["focused"],
    "2":        STATE_TO_IDX["focused"],
    # neutral → calm
    "neutral":  STATE_TO_IDX["calm"],
    "neu":      STATE_TO_IDX["calm"],
    "0":        STATE_TO_IDX["calm"],
    # negative → stressed
    "negative": STATE_TO_IDX["stressed"],
    "neg":      STATE_TO_IDX["stressed"],
    "-1":       STATE_TO_IDX["stressed"],
    "3":        STATE_TO_IDX["stressed"],
}


def _map_label(raw: str) -> int | None:
    return _LABEL_MAP.get(str(raw).strip().lower())


# ── CSV loading ───────────────────────────────────────────────────────────────

def _find_label_column(df: pd.DataFrame) -> str | None:
    """Heuristically find the label column."""
    candidates = ["label", "emotion", "class", "state", "target", "y"]
    for col in df.columns:
        if col.strip().lower() in candidates:
            return col
    # Fall back to last column if it looks categorical
    last = df.columns[-1]
    if df[last].dtype == object or df[last].nunique() <= 10:
        return last
    return None


def _find_eeg_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    """Return all feature columns (everything except the label column)."""
    return [c for c in df.columns if c != label_col]


def _extract_de_compatible(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, list[int]]:
    """
    Derive 16 DE-equivalent features from the CSV's per-band stddev columns.

    The Kaggle emotions.csv has stddev_{0..3}_{a,b} (std per frequency band for
    two EEG recordings) and entropy{0..3}_{a,b} (spectral entropy per band).
    DE = 0.5 * log(2πe * σ²), so log(stddev) is proportional to DE.

    We select 16 features:
      [log_stddev_0..3_a, log_stddev_0..3_b, entropy0..3_a, entropy0..3_b]

    These are semantically equivalent to what extract_de_features() computes
    on raw Muse data (4 channels × 4 bands).
    """
    # Find available stddev and entropy columns
    std_a  = [f"stddev_{i}_a" for i in range(4)]
    std_b  = [f"stddev_{i}_b" for i in range(4)]
    ent_a  = [f"entropy{i}_a" for i in range(4)]
    ent_b  = [f"entropy{i}_b" for i in range(4)]
    needed = std_a + std_b + ent_a + ent_b

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"--de-compatible requires columns {needed}.\n"
            f"Missing: {missing}"
        )

    X_list, y_list = [], []
    for _, row in df.iterrows():
        label = _map_label(row[label_col])
        if label is None:
            continue

        std_vals_a = df[std_a].loc[row.name].values.astype(np.float64)
        std_vals_b = df[std_b].loc[row.name].values.astype(np.float64)
        ent_vals_a = df[ent_a].loc[row.name].values.astype(np.float64)
        ent_vals_b = df[ent_b].loc[row.name].values.astype(np.float64)

        # Convert stddev → DE: 0.5 * log(2πe * σ²)
        _k = 0.5 * np.log(2 * np.pi * np.e)
        de_a = _k + np.log(np.maximum(np.abs(std_vals_a), 1e-10))
        de_b = _k + np.log(np.maximum(np.abs(std_vals_b), 1e-10))

        feat = np.concatenate([de_a, de_b, ent_vals_a, ent_vals_b]).astype(np.float32)
        if np.isnan(feat).any():
            continue
        X_list.append(feat)
        y_list.append(label)

    n = len(needed)
    return (np.array(X_list) if X_list else np.empty((0, n))), y_list


def _load_csv_folder(data_dir: str, de_compatible: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all CSV files from data_dir and return (X, y, feature_names).

    Handles two data formats automatically:
    - Pre-extracted features: each row is one sample with numeric feature columns
    - Raw time-series: each row is one EEG sample; groups rows into 1-second windows
      and computes DE features (requires a 'time' or sequential ordering)
    """
    path = Path(data_dir)
    csv_files = sorted(path.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  {f}")

    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    feature_names: list[str] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        print(f"\n{csv_path.name}: {len(df)} rows × {len(df.columns)} columns")

        label_col = _find_label_column(df)
        if label_col is None:
            print(f"  Warning: could not find label column — skipping {csv_path.name}")
            continue

        feat_cols = _find_eeg_columns(df, label_col)
        print(f"  Label column : '{label_col}'  (values: {df[label_col].unique()[:5]})")
        print(f"  Feature cols : {len(feat_cols)}  ({feat_cols[:5]}{'...' if len(feat_cols) > 5 else ''})")

        if de_compatible:
            X, y = _extract_de_compatible(df, label_col)
            if not feature_names:
                feature_names = (
                    [f"de_stddev_{i}_a" for i in range(4)]
                    + [f"de_stddev_{i}_b" for i in range(4)]
                    + [f"entropy{i}_a" for i in range(4)]
                    + [f"entropy{i}_b" for i in range(4)]
                )
        else:
            # Detect if this is a raw time-series file (by checking if channel names appear)
            muse_channels_present = any(
                any(ch.lower() in c.lower() for ch in MUSE_CHANNEL_NAMES)
                for c in feat_cols
            )
            n_numeric_cols = df[feat_cols].select_dtypes(include=np.number).shape[1]

            is_raw_timeseries = muse_channels_present and n_numeric_cols == len(MUSE_CHANNEL_NAMES)

            if is_raw_timeseries:
                X, y = _load_raw_timeseries(df, feat_cols, label_col)
                if not feature_names:
                    feature_names = [f"de_feature_{i}" for i in range(N_FEATURES)]
            else:
                X, y = _load_preextracted(df, feat_cols, label_col)
                if not feature_names:
                    feature_names = feat_cols

        if len(X) > 0:
            all_X.append(X)
            all_y.extend(y)
            print(f"  Loaded {len(X)} samples  |  labels: {np.bincount(y, minlength=3)}")

    if not all_X:
        raise ValueError("No valid samples loaded. Check --peek output for format issues.")

    X_all = np.concatenate(all_X, axis=0).astype(np.float32)
    y_all = np.array(all_y, dtype=np.int64)
    print(f"\nTotal: {len(X_all)} samples  |  class counts: {np.bincount(y_all, minlength=3)}")
    return X_all, y_all, feature_names


def _load_preextracted(
    df: pd.DataFrame, feat_cols: list[str], label_col: str
) -> tuple[np.ndarray, list[int]]:
    """Load a CSV where each row is a pre-extracted feature vector."""
    X_list, y_list = [], []
    feat_df = df[feat_cols].select_dtypes(include=np.number)

    for _, row in df.iterrows():
        label = _map_label(row[label_col])
        if label is None:
            continue
        x = feat_df.loc[row.name].values.astype(np.float32)
        if np.isnan(x).any():
            continue
        X_list.append(x)
        y_list.append(label)

    return (np.array(X_list) if X_list else np.empty((0, len(feat_cols))), y_list)


def _load_raw_timeseries(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    fs: int = DREAMER_FS,
    window_samples: int = 128,
) -> tuple[np.ndarray, list[int]]:
    """
    Load a CSV where each row is one raw EEG sample.
    Groups rows into 1-second windows and computes DE features.
    """
    X_list, y_list = [], []

    for label_val, group in df.groupby(label_col):
        label = _map_label(label_val)
        if label is None:
            continue

        eeg = group[feat_cols].values.T.astype(np.float64)  # (n_channels, n_samples)
        n_samples = eeg.shape[1]
        n_windows = n_samples // window_samples

        for w in range(n_windows):
            chunk = eeg[:, w * window_samples: (w + 1) * window_samples]
            feat = extract_de_features(chunk, fs=float(fs))
            X_list.append(feat.astype(np.float32))
            y_list.append(label)

    return (np.array(X_list) if X_list else np.empty((0, N_FEATURES)), y_list)


# ── peek ──────────────────────────────────────────────────────────────────────

def peek(data_dir: str) -> None:
    """Print CSV structure without training — use this first to verify format."""
    path = Path(data_dir)
    csv_files = sorted(path.glob("**/*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        print(f"\n{'='*60}")
        print(f"File: {csv_path.name}")
        print(f"Shape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            dtype = df[col].dtype
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.isna().all():
                vals = df[col].unique()[:5]
                print(f"  {col:40s}  {str(dtype):10s}  values: {vals}")
            else:
                print(f"  {col:40s}  {str(dtype):10s}  range: [{numeric.min():.3f}, {numeric.max():.3f}]")
        print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

    print(f"\n{'='*60}")
    print("Label mapping that will be applied:")
    print("  positive / 1 / 2  →  FOCUSED")
    print("  neutral  / 0      →  CALM")
    print("  negative / -1 / 3 →  STRESSED")


# ── training loop (same as train.py) ─────────────────────────────────────────

def train(
    data_dir: str,
    out_dir: str = "EEGClassifier/models",
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    test_size: float = 0.2,
    seed: int = 42,
    de_compatible: bool = False,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = _load_csv_folder(data_dir, de_compatible=de_compatible)

    n_features = X.shape[1]
    print(f"\nFeature dimensions: {n_features}")
    if n_features != N_FEATURES:
        print(
            f"Note: CSV has {n_features} features, model expects {N_FEATURES}. "
            f"Rebuilding model with {n_features} input dimensions."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler_path = out_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Also save feature metadata so inference knows what format was used
    meta_path = out_path / "feature_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({"n_features": n_features, "feature_names": feature_names}, f)

    print(f"Scaler saved → {scaler_path}")

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

    model = EmotionMLP(input_dim=n_features).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_model_path = out_path / "emotion_mlp_best.pt"

    for epoch in range(1, epochs + 1):
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

    final_path = out_path / "emotion_mlp_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nBest val accuracy : {best_acc:.4f}")
    print(f"Model saved       → {best_model_path}")
    print(f"Final model saved → {final_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EEG emotion classifier on the Kaggle brainwave CSV dataset"
    )
    parser.add_argument("--data",   required=True,          help="Folder containing the downloaded CSV files")
    parser.add_argument("--out",    default="EEGClassifier/models", help="Output directory for model + scaler")
    parser.add_argument("--epochs", type=int, default=60,   help="Training epochs")
    parser.add_argument("--batch",  type=int, default=256,  help="Batch size")
    parser.add_argument("--lr",     type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed",   type=int, default=42,   help="Random seed")
    parser.add_argument("--peek",          action="store_true", help="Print CSV structure and exit without training")
    parser.add_argument("--de-compatible", action="store_true",
                        help="Train on 16 DE-equivalent features (stddev+entropy per band). "
                             "Required for compatibility with the live Muse inference pipeline.")
    args = parser.parse_args()

    if args.peek:
        peek(args.data)
    else:
        train(
            data_dir=args.data,
            out_dir=args.out,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            seed=args.seed,
            de_compatible=args.de_compatible,
        )
