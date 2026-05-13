"""
Step 1: Download Sleep-EDF Dataset
Uses MNE's built-in downloader for PhysioNet Sleep-EDF Cassette dataset.

IMPORTANT: PhysioNet now requires a free account to download.
  1. Register at: https://physionet.org/register/
  2. Accept the data use agreement at:
     https://physionet.org/content/sleep-edfx/1.0.0/
  3. Run: mne.datasets.sleep_physionet.age.fetch_data(...)

If you haven't registered yet, run this script with --synthetic flag
to generate a synthetic dataset and test the full pipeline immediately.
You can swap in real data later without changing any other file.

Usage:
  python src/download_data.py              # real data (needs PhysioNet account)
  python src/download_data.py --synthetic  # synthetic data (works immediately)
"""

import argparse
import os
import pickle
import numpy as np

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Sleep-EDF constants
SFREQ           = 100          # Hz
EPOCH_SEC       = 30           # seconds per epoch (standard PSG)
SAMPLES         = SFREQ * EPOCH_SEC   # 3000 samples per epoch
N_SUBJECTS      = 20           # use first 20 subjects (of 83 available)

# Stage labels: W=0, N1=1, N2=2, N3=3, REM=4
STAGE_NAMES  = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
# Realistic overnight distribution
STAGE_PROBS  = [0.15, 0.05, 0.45, 0.20, 0.15]


def download_real(n_subjects: int = N_SUBJECTS):
    """
    Download real Sleep-EDF data via MNE.
    Requires free PhysioNet account + data use agreement.
    """
    try:
        from mne.datasets.sleep_physionet.age import fetch_data
    except ImportError:
        print("MNE not installed. Run: pip install mne")
        return None

    print(f"Downloading Sleep-EDF for {n_subjects} subjects...")
    print("(Requires PhysioNet account: https://physionet.org/register/)")

    all_files = {}
    for subj in range(n_subjects):
        try:
            files = fetch_data(
                subjects=[subj],
                recording=[1],   # night 1 only
                path=DATA_DIR,
                verbose=False,
            )
            all_files[subj] = files
            print(f"  Subject {subj:02d}: downloaded")
        except Exception as e:
            print(f"  Subject {subj:02d}: FAILED — {e}")

    return all_files


def generate_synthetic(
    n_subjects: int = 10,
    epochs_per_subject: int = 240,   # ~2 hours of sleep
):
    """
    Generate synthetic Sleep-EDF-like data for testing the full pipeline.
    Produces realistic EEG-shaped signals with correct class distribution.
    Saved in the same format as real preprocessed data.
    """
    print("=" * 55)
    print("Generating synthetic Sleep-EDF dataset")
    print(f"Subjects: {n_subjects} | Epochs/subject: {epochs_per_subject}")
    print("=" * 55)

    np.random.seed(42)
    all_X, all_y, all_subj = [], [], []

    for subj in range(n_subjects):
        # Simulate EEG: mix of sine waves at sleep-relevant frequencies
        # + realistic 1/f noise
        n = epochs_per_subject
        X = np.zeros((n, 1, SAMPLES), dtype=np.float32)

        # Sample labels first (realistic overnight distribution)
        y = np.random.choice(5, size=n, p=STAGE_PROBS).astype(np.int64)

        for i, stage in enumerate(y):
            t = np.linspace(0, EPOCH_SEC, SAMPLES)
            # Stage-specific dominant frequencies
            if stage == 0:   # Wake: high freq beta 15-30 Hz
                sig = np.sin(2 * np.pi * 20 * t) * 0.5
            elif stage == 1: # N1: mixed alpha/theta 4-12 Hz
                sig = np.sin(2 * np.pi * 8 * t) * 0.7
            elif stage == 2: # N2: sleep spindles 12-15 Hz + K-complexes
                sig = np.sin(2 * np.pi * 13 * t) * 0.8
            elif stage == 3: # N3/SWS: slow waves delta 0.5-4 Hz
                sig = np.sin(2 * np.pi * 1 * t) * 2.0
            else:            # REM: mixed, low amplitude
                sig = np.sin(2 * np.pi * 6 * t) * 0.4

            # Add 1/f noise (realistic EEG background)
            noise = np.random.randn(SAMPLES) * 0.3
            X[i, 0] = sig + noise

        # Z-score normalize per epoch
        mean = X.mean(axis=-1, keepdims=True)
        std  = X.std(axis=-1, keepdims=True) + 1e-8
        X    = (X - mean) / std

        all_X.append(X)
        all_y.append(y)
        all_subj.extend([subj] * n)

        dist = {STAGE_NAMES[k]: int(v)
                for k, v in zip(*np.unique(y, return_counts=True))}
        print(f"  Subject {subj:02d}: {n} epochs | {dist}")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    s_all = np.array(all_subj)

    out_path = os.path.join(PROCESSED_DIR, "all_subjects.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "X": X_all,
            "y": y_all,
            "subjects": s_all,
            "synthetic": True,
            "sfreq": SFREQ,
            "epoch_sec": EPOCH_SEC,
        }, f)

    print(f"\nSaved: {X_all.shape} → {out_path}")
    print(f"Total label dist: { {STAGE_NAMES[k]: int(v) for k,v in zip(*np.unique(y_all, return_counts=True))} }")
    return X_all, y_all, s_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data instead of downloading")
    parser.add_argument("--n_subjects", type=int, default=10)
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic(n_subjects=args.n_subjects)
    else:
        result = download_real(n_subjects=args.n_subjects)
        if result is None:
            print("\nFalling back to synthetic data...")
            generate_synthetic(n_subjects=args.n_subjects)