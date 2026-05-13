"""
Step 2: Preprocess Real Sleep-EDF EDF Files
Only needed if you downloaded real data from PhysioNet.
If you used --synthetic in Step 1, skip this file — data is already preprocessed.

For real data: loads PSG + hypnogram EDF files, extracts single-channel EEG
(Fpz-Cz), segments into 30-second epochs, maps annotations to 5-class labels,
normalizes, and saves .pkl files.
"""

import os
import pickle
import numpy as np
import mne
from pathlib import Path

# Constants
CHANNEL       = "EEG Fpz-Cz"     # single-channel EEG used by DeepSleepNet
SFREQ_TARGET  = 100               # resample to 100 Hz
EPOCH_SEC     = 30                # standard PSG epoch length
SAMPLES       = SFREQ_TARGET * EPOCH_SEC  # 3000

DATA_DIR      = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# PhysioNet annotation → our 5-class label
ANNOTATION_MAP = {
    "Sleep stage W":    0,
    "Sleep stage 1":    1,
    "Sleep stage 2":    2,
    "Sleep stage 3":    3,
    "Sleep stage 4":    3,   # merge S3+S4 → N3 (modern AASM standard)
    "Sleep stage R":    4,
    "Movement time":    0,   # treat as Wake
}
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def load_psg_hypnogram(psg_path: str, hyp_path: str):
    """Load PSG recording + hypnogram annotation file."""
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot, emit_warning=False)
    return raw


def extract_epochs(raw: mne.io.BaseRaw):
    """Extract single-channel EEG epochs with sleep stage labels."""
    # Pick only the Fpz-Cz channel
    if CHANNEL in raw.ch_names:
        raw.pick_channels([CHANNEL])
    else:
        # Fallback: use first EEG channel
        eeg_chs = mne.pick_types(raw.info, eeg=True)
        raw.pick(eeg_chs[:1])

    # Resample
    raw.resample(SFREQ_TARGET, verbose=False)

    # Build events from annotations
    events, event_id = mne.events_from_annotations(
        raw, event_id=ANNOTATION_MAP, verbose=False
    )

    # Remove unknown annotations
    valid_ids = list(set(ANNOTATION_MAP.values()))
    mask = np.isin(events[:, 2], valid_ids)
    events = events[mask]

    if len(events) == 0:
        return None, None

    # Extract fixed-length epochs
    epochs = mne.Epochs(
        raw, events,
        event_id={str(v): v for v in valid_ids},
        tmin=0.0,
        tmax=EPOCH_SEC - 1 / SFREQ_TARGET,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data().astype(np.float32)    # (N, 1, 3000)
    y = epochs.events[:, -1].astype(np.int64)   # (N,)

    # Z-score normalize per epoch
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-8
    X    = (X - mean) / std

    return X, y


def preprocess_all():
    """Process all downloaded PSG+hypnogram pairs."""
    # Find PSG files (end in E0-PSG.edf) and match hypnograms (end in -Hypnogram.edf)
    raw_dir = Path(DATA_DIR)
    psg_files = sorted(raw_dir.rglob("*PSG.edf"))

    if not psg_files:
        print("No EDF files found in data/raw/")
        print("Run: python src/download_data.py  (needs PhysioNet account)")
        print("Or:  python src/download_data.py --synthetic")
        return

    print(f"Found {len(psg_files)} PSG files")
    all_X, all_y, all_subj = [], [], []

    for subj_idx, psg_path in enumerate(psg_files):
        # Match hypnogram: SC4001E0-PSG.edf → SC4001EC-Hypnogram.edf
        stem = psg_path.stem.replace("E0-PSG", "EC-Hypnogram")
        hyp_path = psg_path.parent / f"{stem}.edf"

        if not hyp_path.exists():
            print(f"  Subject {subj_idx:02d}: no hypnogram found, skipping")
            continue

        print(f"  Subject {subj_idx:02d}: {psg_path.name}...", end=" ")
        try:
            raw      = load_psg_hypnogram(str(psg_path), str(hyp_path))
            X, y     = extract_epochs(raw)

            if X is None:
                print("no valid epochs")
                continue

            # Save per-subject
            out = os.path.join(PROCESSED_DIR, f"subject_{subj_idx:02d}.pkl")
            with open(out, "wb") as f:
                pickle.dump({"X": X, "y": y, "subject": subj_idx}, f)

            all_X.append(X)
            all_y.append(y)
            all_subj.extend([subj_idx] * len(y))

            dist = {STAGE_NAMES[k]: int(v)
                    for k, v in zip(*np.unique(y, return_counts=True))}
            print(f"OK — {len(y)} epochs | {dist}")

        except Exception as e:
            print(f"FAILED — {e}")

    if all_X:
        X_all = np.concatenate(all_X)
        y_all = np.concatenate(all_y)
        s_all = np.array(all_subj)

        out = os.path.join(PROCESSED_DIR, "all_subjects.pkl")
        with open(out, "wb") as f:
            pickle.dump({
                "X": X_all, "y": y_all, "subjects": s_all,
                "synthetic": False, "sfreq": SFREQ_TARGET, "epoch_sec": EPOCH_SEC,
            }, f)

        print(f"\nSaved combined: {X_all.shape} → {out}")


if __name__ == "__main__":
    preprocess_all()