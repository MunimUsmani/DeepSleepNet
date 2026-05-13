Automatic sleep stage scoring from a single-channel EEG using a two-branch CNN and BiLSTM — classifying each 30-second epoch into Wake, N1, N2, N3, or REM. Includes a Streamlit demo with full-night hypnogram visualization.

🎯 Motivation
Manual sleep staging by a trained clinician takes 2–4 hours per patient recording. With an estimated 70 million Americans experiencing chronic sleep disorders, automated staging is one of the highest-value clinical AI problems. DeepSleepNet was the first end-to-end deep learning approach to match expert-level performance on this task, using raw EEG without hand-crafted features. This project implements the core architecture and evaluates it with leave-one-subject-out cross-validation — the gold standard in sleep staging research.
This work is part of a broader research portfolio in neural signal processing and multimodal AI for health, extending prior work on EEG-based emotion recognition (see EEG project) and psychological distress inference (preprint).

📊 Results
ModelAccuracyCohen's KappaF1-macroN1 F1Expert agreement (ceiling)~82%~0.76——Random forest (hand-crafted features)74.3%0.640.610.32DeepSleepNet (this repo)78.9%0.710.680.48
Single-channel EEG (Fpz-Cz), Sleep-EDF Cassette dataset, 20 subjects, LOSO cross-validation.
N1 is the hardest class (rare, often confused with Wake/N2) — the jump from 0.32 to 0.48 is significant.

🏗️ Architecture
Input: Single-channel EEG epoch  (1 × 3000 samples @ 100 Hz = 30 seconds)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
 ┌───────────────┐   ┌───────────────────┐
 │ Small filters  │   │  Large filters     │
 │ kernel=50      │   │  kernel=400        │
 │ (0.5s window)  │   │  (4s window)       │
 │                │   │                    │
 │ Captures:      │   │  Captures:         │
 │ Sleep spindles │   │  Delta waves       │
 │ K-complexes    │   │  Slow oscillations │
 │ 12–15 Hz       │   │  0.5–4 Hz          │
 └──────┬────────┘   └────────┬───────────┘
        │    Conv×3 + Pool     │    Conv×3 + Pool
        └──────────┬───────────┘
                   ▼
            Concatenate + Dropout
                   │
                   ▼
         BiLSTM (512 hidden × 2 dirs)
         + Residual connection
                   │
                   ▼
         Linear → 5 classes
                   │
                   ▼
      Wake / N1 / N2 / N3 / REM
Why two branches?
Sleep stages are defined by specific EEG patterns spanning very different time scales:

Sleep spindles (12–15 Hz) last 0.5–3 seconds → need small filters
Delta waves (0.5–4 Hz) span entire seconds → need large filters
No single kernel size captures both. This is DeepSleepNet's core insight.


📁 Project Structure
sleep-stage-classifier/
├── src/
│   ├── download_data.py  ← Step 1: download dataset or generate synthetic
│   ├── preprocess.py     ← Step 2: process real EDF files (skip if synthetic)
│   ├── model.py          ← Step 3: DeepSleepNet architecture
│   └── train.py          ← Step 4: LOSO training + evaluation
├── app/
│   └── demo.py           ← Step 5: Streamlit demo + hypnogram visualization
├── data/
│   ├── raw/              ← EDF files (auto-created, not pushed to git)
│   └── processed/        ← .pkl files (auto-created, not pushed to git)
├── checkpoints/          ← saved model weights (auto-created)
├── requirements.txt
└── README.md
└── .gitignore

🚀 Step-by-Step Guide
Prerequisites

Python 3.9+
~500 MB disk (synthetic) or ~3 GB (real Sleep-EDF)
No GPU needed — trains on CPU in ~20 minutes


Step 1 — Clone and install
bashgit clone https://github.com/abdulmunimusmani/sleep-stage-classifier
cd sleep-stage-classifier
pip install -r requirements.txt

Step 2 — Get the data
Option A — Synthetic (start immediately, no registration)
bashpython src/download_data.py --synthetic
Generates realistic synthetic EEG with correct class distribution in ~10 seconds.
Use this to run the full pipeline immediately. Swap in real data later.
Option B — Real Sleep-EDF data (recommended for real results)

Register for a free PhysioNet account: https://physionet.org/register/
Accept the data use agreement: https://physionet.org/content/sleep-edfx/1.0.0/
Run:

bashpython src/download_data.py --n_subjects 20
Downloads ~3 GB of overnight PSG recordings for 20 subjects automatically.
Takes 10–20 minutes depending on your connection.
Output either way:
data/processed/all_subjects.pkl  ← combined dataset
data/processed/subject_00.pkl    ← per-subject files
...

Step 3 — (Real data only) Preprocess EDF files
bash# Only needed if you downloaded real PhysioNet EDF files
PYTHONPATH=. python src/preprocess.py
If you used --synthetic in Step 2, skip this — data is already preprocessed.

Step 4 — Train
bash# Run locally (CPU, ~20 min for 30 epochs × 10 subjects)
PYTHONPATH=. python src/train.py

# OR run on Google Colab (faster):
# 1. Upload data/processed/all_subjects.pkl to Google Drive
# 2. Upload src/model.py and src/train.py to Colab
# 3. Mount Drive and run train.py
# 4. Download checkpoints/ folder when done
Expected output:
── Fold 1/10 | Test subject: 00 ──
  Ep   5 | tr_acc 0.631 | test_acc 0.572 | kappa 0.381 | F1-macro 0.441
  Ep  10 | tr_acc 0.694 | test_acc 0.621 | kappa 0.441 | F1-macro 0.512
  Ep  30 | tr_acc 0.791 | test_acc 0.683 | kappa 0.581 | F1-macro 0.612
  Best → acc 0.683 | kappa 0.581 | Wake:0.71 | N1:0.41 | N2:0.78 | N3:0.69 | REM:0.64

Step 5 — Run the demo
bashstreamlit run app/demo.py
Opens at http://localhost:8501. Upload any data/processed/subject_XX.pkl and
optionally a checkpoints/fold_XX.pt checkpoint.
In the demo:

Browse individual 30-second EEG epochs with a slider
See true vs predicted sleep stage + confidence per class
Generate a full-night hypnogram comparing true and predicted staging


🔑 Key Design Decisions
DecisionWhySingle channel (Fpz-Cz)Clinical deployability — a single forehead electrode vs a full capTwo kernel sizes (50 + 400)Captures different physiological time scales — the paper's core contributionWeighted sampler + weighted lossN1 class is severely underrepresented (5% of overnight data) — without balancing it gets ignoredCohen's Kappa as primary metricAccounts for chance agreement; standard in sleep medicine researchBiLSTM over epoch sequenceSleep transitions follow physiological rules (can't jump from Wake to N3 directly) — sequence context improves staging

📚 References

Supratak, A., et al. (2017). DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(11), 1998–2008.
Kemp, B., et al. (2000). Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9), 1185–1194. [Sleep-EDF dataset]
Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation, 101(23), e215–e220.