# DeepSleepNet Sleep Stage Classifier

Automatic sleep stage scoring from **single-channel EEG** using a DeepSleepNet-style two-branch CNN architecture. The system classifies each 30-second EEG epoch into one of five sleep stages:

**Wake, N1, N2, N3, REM**

The project includes real Sleep-EDF preprocessing, leave-one-subject-out evaluation, trained checkpoints, and an interactive Streamlit research demo with clinical visualizations such as hypnogram comparison, confusion matrix, transition-error analysis, learned filter visualization, and per-class F1 benchmarking.

---

## Preview

### 1. Streamlit Demo

<img width="959" height="436" alt="image" src="https://github.com/user-attachments/assets/8c05c97c-4323-4c7e-a71b-3b12bbca6951" />
<img width="959" height="410" alt="image" src="https://github.com/user-attachments/assets/eb97a9cf-21f8-4fe0-b41f-c50434a2c302" />

### 2. True vs Predicted Hypnogram

<img width="785" height="435" alt="image" src="https://github.com/user-attachments/assets/2597ffe1-d78b-4dcd-851d-8c56700ed0c5" />
<img width="755" height="350" alt="image" src="https://github.com/user-attachments/assets/8289c5c9-8011-487d-96b9-b6148072927e" />

### 3. Confusion Matrix

<img width="781" height="438" alt="image" src="https://github.com/user-attachments/assets/47fc0f0f-141d-4c7e-8687-d372da0ec07d" />
<img width="748" height="243" alt="image" src="https://github.com/user-attachments/assets/5cd688ab-2b61-4ef3-8039-e7a1a3fa9fa6" />


### 4. Transition Error Analysis

<img width="741" height="1168" alt="image" src="https://github.com/user-attachments/assets/332372b7-1644-484c-8054-869fb3abd2d1" />
<img width="733" height="367" alt="image" src="https://github.com/user-attachments/assets/e41e8563-7d83-4d21-bf4f-b2fef4768130" />

### 5. Learned CNN Filter Visualization

<img width="1460" height="467" alt="image" src="https://github.com/user-attachments/assets/013aaa6c-8ad2-4438-b24a-6ad954dc5bc9" />
<img width="769" height="407" alt="image" src="https://github.com/user-attachments/assets/5ee3e53c-2483-4517-a07e-e440069f6c10" />


### 6. F1 Benchmark

<img width="1460" height="572" alt="image" src="https://github.com/user-attachments/assets/7051aa8e-03ec-4147-9e32-4a2eb176be24" />

<img width="794" height="392" alt="image" src="https://github.com/user-attachments/assets/bebec3b4-29f6-40c0-aa8e-2deb6734bae2" />

---

## Motivation

Manual sleep staging is time-consuming and requires trained clinical experts to score overnight recordings epoch by epoch. Automated sleep staging is therefore a high-value clinical AI problem because it can reduce review time, support sleep laboratories, and make large-scale sleep analysis more practical.

This project implements and extends a DeepSleepNet-style approach for automatic sleep staging from raw EEG. Beyond classification accuracy, the demo emphasizes interpretable research visuals that help explain what the model learned and where it fails.

The focus is not only to build a classifier, but to reproduce the kind of clinical analysis expected in sleep-stage scoring research:

- Full-night hypnogram comparison
- Per-class F1 analysis
- Confusion matrix
- Transition-rule error analysis
- Learned CNN filter visualization

---

## Dataset

This project uses the **Sleep-EDF Expanded Cassette** dataset from PhysioNet.

The processed dataset used in this project contains:

| Item | Value |
|---|---:|
| Dataset | Sleep-EDF Expanded Cassette |
| Signal | Single-channel EEG |
| Sampling rate | 100 Hz |
| Epoch length | 30 seconds |
| Samples per epoch | 3000 |
| Number of subjects used | 30 |
| Total epochs | 3,784 |
| Classes | Wake, N1, N2, N3, REM |

The processed training file is:

```text
data/processed/all_subjects.pkl
```

The raw EDF files are not required to run the Streamlit demo after preprocessing is complete.

---

## Final Results

The model was evaluated using **leave-one-subject-out cross-validation** over 30 subjects.

| Metric | Result |
|---|---:|
| Accuracy | 0.674 ± 0.106 |
| Cohen's Kappa | 0.568 ± 0.120 |
| F1-macro | 0.625 ± 0.103 |

### Per-Class F1

| Stage | F1 Score |
|---|---:|
| Wake | 0.648 |
| N1 | 0.529 |
| N2 | 0.633 |
| N3 | 0.815 |
| REM | 0.502 |

### Interpretation

N3 achieves the highest F1 score because deep sleep is dominated by high-amplitude delta activity, which is easier for the model to identify. N1 and REM are more difficult because they are transitional or mixed-frequency stages and are commonly confused with neighboring stages such as Wake and N2.

---

## Key Research Visuals

### 1. Confusion Matrix

**Recommended README location:** after the results table.

The confusion matrix shows which sleep stages the model confuses most often.

Use it to explain:

- N1 is frequently confused with Wake or N2.
- REM can be confused with Wake/N1 due to mixed-frequency EEG.
- N3 is usually easier because delta waves are distinctive.

Recommended image:

```markdown
![Confusion Matrix](assets/confusion-matrix.png)
```

---

### 2. Transition Error Analysis

**Recommended README location:** after the confusion matrix.

Instead of only showing class-level mistakes, transition analysis asks:

> Given the previous true sleep stage, what transition did the model fail on?

This is clinically useful because many sleep-stage errors happen at boundaries such as Wake → N1, N1 → N2, and REM → Wake.

Recommended image:

```markdown
![Transition Error Analysis](assets/transition-errors.png)
```

---

### 3. Learned Filter Visualization

**Recommended README location:** after the architecture section.

The two CNN branches are intended to learn EEG patterns at different time scales:

| Branch | Expected pattern |
|---|---|
| Small temporal filters | Sleep spindles around 12–15 Hz |
| Large temporal filters | Delta activity around 0.5–4 Hz |

This visualization helps show that the network learned meaningful EEG-related filters rather than arbitrary patterns.

Recommended image:

```markdown
![Learned Filters](assets/learned-filters.png)
```

---

### 4. Full-Night Hypnogram Comparison

**Recommended README location:** after the Streamlit demo section.

A hypnogram is the sleep-stage timeline that a sleep lab would inspect. Showing the true and predicted hypnogram side by side makes the model output clinically understandable.

Recommended image:

```markdown
![Hypnogram Comparison](assets/hypnogram-comparison.png)
```

---

### 5. F1 Benchmark

**Recommended README location:** after final results or in a benchmarking section.

The F1 benchmark helps compare your model against reference values from the original paper or another baseline.

Recommended image:

```markdown
![F1 Benchmark](assets/f1-benchmark.png)
```

---

## Architecture

The model follows the core idea of DeepSleepNet: use two convolutional branches to learn features at different temporal resolutions.

```text
Input EEG epoch
Shape: 1 × 3000
30 seconds @ 100 Hz
        │
        ├───────────────────────────┐
        │                           │
        ▼                           ▼
 Small-filter CNN branch      Large-filter CNN branch
 Captures short patterns      Captures slow patterns
 such as spindles             such as delta waves
        │                           │
        └──────────────┬────────────┘
                       ▼
              Feature concatenation
                       │
                       ▼
                  Classifier
                       │
                       ▼
        Wake / N1 / N2 / N3 / REM
```

### Why Two CNN Branches?

Sleep stages are defined by EEG patterns occurring at different time scales:

- **Sleep spindles** occur around 12–15 Hz and are important for N2.
- **Delta waves** occur around 0.5–4 Hz and are important for N3.
- REM and Wake contain mixed-frequency, low-amplitude EEG patterns.

A single kernel size may not capture all of these patterns effectively, so the two-branch design helps the model learn both short-duration and long-duration EEG features.

---

## Project Structure

```text
DeepSleepNet/
├── app/
│   └── demo.py
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── train_resume.py
├── data/
│   ├── raw/
│   │   └── sleep-cassette EDF files
│   └── processed/
│       └── all_subjects.pkl
├── checkpoints/
│   ├── fold_00.pt
│   ├── fold_01.pt
│   ├── ...
│   └── results_summary_resume.pkl
├── assets/
│   ├── streamlit-demo.png
│   ├── hypnogram-comparison.png
│   ├── confusion-matrix.png
│   ├── transition-errors.png
│   ├── learned-filters.png
│   └── f1-benchmark.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/DeepSleepNet.git
cd DeepSleepNet
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\activate
```

```bash
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install streamlit mne
```

---

## Running the Streamlit Demo

After downloading or generating the required files, run:

```bash
python -m streamlit run app/demo.py
```

The app should open at:

```text
http://localhost:8501
```

The demo can load:

```text
data/processed/all_subjects.pkl
checkpoints/fold_XX.pt
checkpoints/results_summary_resume.pkl
```

### Demo Features

The Streamlit app includes:

- Epoch-level EEG explorer
- True vs predicted sleep stage
- Prediction confidence bars
- Full-night hypnogram comparison
- Confusion matrix
- Transition-rule error analysis
- Learned CNN filter visualization
- Per-class F1 benchmark
- Final LOSO summary metrics

---

## Data Preparation

### Option 1: Use Existing Processed Data

If you already have:

```text
data/processed/all_subjects.pkl
```

you can skip preprocessing and directly run training or the demo.

### Option 2: Download Sleep-EDF with wget

In Google Colab or Linux/macOS:

```bash
wget -r -N -c -np -A "*.edf" -P data/raw https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
```

### Option 3: Preprocess EDF Files

After downloading EDF files:

```bash
PYTHONPATH=. python src/preprocess.py
```

Expected output:

```text
Saved combined: (3784, 1, 3000) → data/processed/all_subjects.pkl
```

---

## Training

### Train from Scratch

```bash
PYTHONPATH=. python src/train.py
```

### Resume Training

If Colab disconnects or training stops midway, use the resume script:

```bash
PYTHONPATH=. python src/train_resume.py
```

The resume script skips folds that already have checkpoint files and continues training missing folds.

Checkpoints are saved in:

```text
checkpoints/
```

---

## Checkpoints

This project uses leave-one-subject-out training, so each checkpoint corresponds to one held-out test subject/fold.

Example:

```text
checkpoints/fold_00.pt
checkpoints/fold_01.pt
checkpoints/fold_03.pt
...
checkpoints/fold_42.pt
```

There is not one single final checkpoint. Instead, each fold checkpoint is a model trained with one subject held out.

The final cross-validation summary is stored in:

```text
checkpoints/results_summary_resume.pkl
```

---

## Important Notes

### Why Some Fold Numbers Are Missing

The original downloaded EDF files may include PSG files without matching hypnogram labels. Those subjects are skipped during preprocessing. Because of this, checkpoint numbers may not be continuous.

For example, this is normal:

```text
fold_00.pt
fold_01.pt
fold_03.pt
fold_06.pt
...
```

### Why N1 Is Difficult

N1 is a transitional sleep stage and is often underrepresented. It is visually subtle and commonly confused with Wake and N2. This makes N1 one of the hardest classes in sleep staging.

### Why LOSO Evaluation Matters

Leave-one-subject-out cross-validation tests whether the model generalizes to unseen subjects. This is more realistic than randomly splitting epochs from the same subject into train and test sets.

---

## Recommended README Image Layout

Create an `assets/` folder:

```text
assets/
├── streamlit-demo.png
├── hypnogram-comparison.png
├── confusion-matrix.png
├── transition-errors.png
├── learned-filters.png
└── f1-benchmark.png
```

Recommended order in README:

| Section | Image |
|---|---|
| Preview | `streamlit-demo.png` |
| Final Results | `confusion-matrix.png` |
| Final Results | `f1-benchmark.png` |
| Clinical Analysis | `transition-errors.png` |
| Architecture | `learned-filters.png` |
| Streamlit Demo | `hypnogram-comparison.png` |

---

## Suggested Presentation Talking Points

- This project implements automatic sleep staging from raw single-channel EEG.
- The model was trained and evaluated on real Sleep-EDF Cassette data.
- LOSO cross-validation was used to test subject-level generalization.
- N3 is the easiest stage because delta activity is visually distinctive.
- N1 and REM are harder because they are transitional or mixed-frequency stages.
- The confusion matrix and transition analysis provide clinical insight into model errors.
- The filter visualization supports the idea that the CNN learned EEG-relevant frequency patterns.
- The hypnogram comparison shows the model output in the same format used by sleep labs.

---

## Results Summary for Report

```text
Dataset: Sleep-EDF Expanded Cassette
Input: Single-channel EEG
Epoch length: 30 seconds
Sampling rate: 100 Hz
Subjects: 30
Total epochs: 3,784
Evaluation: Leave-one-subject-out cross-validation

Accuracy: 0.674 ± 0.106
Cohen's Kappa: 0.568 ± 0.120
F1-macro: 0.625 ± 0.103

Per-class F1:
Wake: 0.648
N1:   0.529
N2:   0.633
N3:   0.815
REM:  0.502
```

---

## References
1. Supratak, A., Dong, H., Wu, C., & Guo, Y. (2017). **DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG.** IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(11), 1998–2008.

2. Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A. C., & Oberye, J. J. L. (2000). **Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.** IEEE Transactions on Biomedical Engineering, 47(9), 1185–1194.

3. Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). **PhysioBank, PhysioToolkit, and PhysioNet.** Circulation, 101(23), e215–e220.
