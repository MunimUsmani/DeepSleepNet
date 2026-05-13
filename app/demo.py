import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.model import DeepSleepNet


# ─────────────────────────────────────────────────────────────
# App config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepSleepNet Sleep Stage Classifier",
    page_icon="😴",
    layout="wide",
)

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

STAGE_COLORS = {
    "Wake": "#E74C3C",
    "N1": "#F39C12",
    "N2": "#3498DB",
    "N3": "#2ECC71",
    "REM": "#9B59B6",
}

STAGE_DESC = {
    "Wake": "Awake — high-frequency, low-amplitude EEG",
    "N1": "Light sleep — transitional sleep, theta activity",
    "N2": "Established sleep — sleep spindles and K-complexes",
    "N3": "Deep sleep — high-amplitude delta waves",
    "REM": "Dreaming sleep — mixed frequency EEG",
}

HYPNOGRAM_ORDER = {
    "Wake": 0,
    "REM": 1,
    "N1": 2,
    "N2": 3,
    "N3": 4,
}

INV_HYPNOGRAM_ORDER = {v: k for k, v in HYPNOGRAM_ORDER.items()}

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_subjects.pkl"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "checkpoints"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_from_path(checkpoint_path: str | None):
    model = DeepSleepNet(n_classes=5)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)

    model.eval()
    return model


def load_model_from_upload(uploaded_model):
    model = DeepSleepNet(n_classes=5)

    if uploaded_model is not None:
        state = torch.load(uploaded_model, map_location="cpu")
        model.load_state_dict(state)

    model.eval()
    return model


@st.cache_data
def load_pickle_from_path(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_pickle_from_upload(uploaded_file):
    return pickle.load(uploaded_file)


def normalize_data(data):
    """
    Supports:
    1. all_subjects.pkl:
       {
          "X": (N, 1, 3000),
          "y": (N,),
          "subjects": (N,),
          ...
       }

    2. single subject pkl:
       {
          "X": (N, 1, 3000),
          "y": (N,),
          "subject": ...
       }
    """
    X = data["X"]
    y = data["y"]

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.int64)

    if "subjects" in data:
        subjects = np.asarray(data["subjects"])
    else:
        subject_value = data.get("subject", 0)
        subjects = np.full(len(y), subject_value)

    return X, y, subjects, data


def predict_all(model, X, batch_size=256):
    preds = []
    probs_all = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).float()
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            preds.append(probs.argmax(axis=1))

    return np.concatenate(preds), np.concatenate(probs_all)


def get_stage_labels(values):
    return [STAGE_NAMES[int(v)] for v in values]


def plot_confusion_matrix(y_true, y_pred, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))

    if normalize:
        cm_display = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        title = "Normalized confusion matrix"
        fmt = ".2f"
    else:
        cm_display = cm
        title = "Confusion matrix"
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm_display)

    ax.set_title(title)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_yticklabels(STAGE_NAMES)

    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                format(cm_display[i, j], fmt),
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def compute_transition_error_table(y_true, y_pred):
    """
    Builds a table of errors based on previous true stage -> current true stage,
    and what the model predicted for the current stage.

    Example row:
    Wake → N1 predicted as Wake

    This highlights clinically meaningful mistakes like:
    N1 confused as Wake
    N1 confused as N2
    REM confused as Wake/N1
    """
    rows = []

    for i in range(1, len(y_true)):
        prev_true = int(y_true[i - 1])
        curr_true = int(y_true[i])
        curr_pred = int(y_pred[i])

        if curr_true != curr_pred:
            rows.append({
                "Previous true stage": STAGE_NAMES[prev_true],
                "Current true stage": STAGE_NAMES[curr_true],
                "Predicted stage": STAGE_NAMES[curr_pred],
                "Transition": f"{STAGE_NAMES[prev_true]} → {STAGE_NAMES[curr_true]}",
                "Mistake": f"{STAGE_NAMES[prev_true]} → {STAGE_NAMES[curr_true]} predicted as {STAGE_NAMES[curr_pred]}",
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    counts = (
        df.groupby(["Transition", "Predicted stage"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    total_by_transition = counts.groupby("Transition")["Count"].transform("sum")
    counts["Percent within transition errors"] = counts["Count"] / total_by_transition * 100

    return counts


def plot_transition_heatmap(transition_error_df):
    if transition_error_df.empty:
        return None

    pivot = transition_error_df.pivot_table(
        index="Transition",
        columns="Predicted stage",
        values="Count",
        fill_value=0,
        aggfunc="sum",
    )

    # Keep columns in sleep-stage order
    for stage in STAGE_NAMES:
        if stage not in pivot.columns:
            pivot[stage] = 0
    pivot = pivot[STAGE_NAMES]

    fig_height = max(4, 0.35 * len(pivot))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    im = ax.imshow(pivot.values)

    ax.set_title("Transition-rule error analysis")
    ax.set_xlabel("Predicted current stage")
    ax.set_ylabel("True transition")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = int(pivot.values[i, j])
            if value > 0:
                ax.text(j, i, str(value), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_hypnogram_comparison(y_true, y_pred):
    true_hyp = [HYPNOGRAM_ORDER[STAGE_NAMES[int(s)]] for s in y_true]
    pred_hyp = [HYPNOGRAM_ORDER[STAGE_NAMES[int(s)]] for s in y_pred]

    epochs = np.arange(len(y_true))
    hours = epochs * 30 / 3600

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.step(hours, true_hyp, where="post", label="True hypnogram", linewidth=1.5)
    ax.step(hours, pred_hyp, where="post", label="Predicted hypnogram", linewidth=1.2, alpha=0.8)

    ax.set_title("Full-night hypnogram comparison")
    ax.set_xlabel("Time since recording start (hours)")
    ax.set_ylabel("Sleep stage")
    ax.set_yticks(list(INV_HYPNOGRAM_ORDER.keys()))
    ax.set_yticklabels([INV_HYPNOGRAM_ORDER[i] for i in INV_HYPNOGRAM_ORDER])
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_epoch_signal(epoch):
    t = np.linspace(0, 30, epoch.shape[-1])
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, epoch[0])
    ax.set_title("30-second EEG epoch")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def find_conv1d_layers(model):
    convs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            convs.append((name, module))
    return convs


def plot_filter_visualization(model, fs=100):
    """
    Visualizes learned Conv1D filters.

    We try to separate the smallest-kernel and largest-kernel Conv1D layers.
    This is robust enough for most DeepSleepNet-style implementations:
    - small temporal filters should capture short patterns like spindles
    - large temporal filters should capture slower rhythm patterns
    """
    convs = find_conv1d_layers(model)

    if len(convs) == 0:
        return None, "No Conv1d layers found in the loaded model."

    conv_info = []
    for name, conv in convs:
        k = conv.weight.detach().cpu().numpy().shape[-1]
        conv_info.append((name, conv, k))

    conv_info_sorted = sorted(conv_info, key=lambda x: x[2])

    small_name, small_conv, small_k = conv_info_sorted[0]
    large_name, large_conv, large_k = conv_info_sorted[-1]

    selected = [
        ("Small-filter branch", small_name, small_conv, small_k),
        ("Large-filter branch", large_name, large_conv, large_k),
    ]

    fig_time, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, (title, name, conv, k) in zip(axes, selected):
        weights = conv.weight.detach().cpu().numpy()

        # Shape usually: (out_channels, in_channels, kernel_size)
        filters = weights[:, 0, :]
        n_plot = min(8, filters.shape[0])

        for i in range(n_plot):
            f = filters[i]
            f = (f - f.mean()) / (f.std() + 1e-8)
            ax.plot(f, alpha=0.8)

        ax.set_title(f"{title}\n{name}, kernel={k}")
        ax.set_xlabel("Filter sample")
        ax.set_ylabel("Normalized weight")
        ax.grid(True, alpha=0.3)

    fig_time.suptitle("Learned CNN filters in time domain", y=1.02)
    fig_time.tight_layout()

    fig_freq, ax = plt.subplots(figsize=(10, 4))

    for title, name, conv, k in selected:
        weights = conv.weight.detach().cpu().numpy()
        filters = weights[:, 0, :]

        # Average magnitude response across filters
        fft_mag = np.abs(np.fft.rfft(filters, axis=1))
        freqs = np.fft.rfftfreq(filters.shape[1], d=1 / fs)
        mean_response = fft_mag.mean(axis=0)

        if mean_response.max() > 0:
            mean_response = mean_response / mean_response.max()

        ax.plot(freqs, mean_response, label=f"{title} ({name}, k={k})")

    ax.axvspan(0.5, 4, alpha=0.15, label="Delta band 0.5–4 Hz")
    ax.axvspan(12, 15, alpha=0.15, label="Spindle band 12–15 Hz")

    ax.set_xlim(0, 30)
    ax.set_title("Average frequency response of learned CNN filters")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig_freq.tight_layout()

    note = (
        f"Detected Conv1D layers: {len(convs)}. "
        f"Smallest kernel: {small_name} k={small_k}. "
        f"Largest kernel: {large_name} k={large_k}."
    )

    return (fig_time, fig_freq), note


def plot_f1_benchmark(y_true, y_pred, paper_f1_values=None):
    your_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(5)),
        zero_division=0,
    )

    if paper_f1_values is None:
        paper_f1_values = [np.nan] * 5

    x = np.arange(len(STAGE_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, your_f1, width, label="Your model")

    if not all(np.isnan(paper_f1_values)):
        ax.bar(x + width / 2, paper_f1_values, width, label="Paper/reference")

    ax.set_title("Per-class F1 comparison")
    ax.set_xlabel("Sleep stage")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    df = pd.DataFrame({
        "Stage": STAGE_NAMES,
        "Your F1": your_f1,
        "Paper/reference F1": paper_f1_values,
    })

    return fig, df


def display_probability_bars(probs):
    for i, name in enumerate(STAGE_NAMES):
        color = STAGE_COLORS[name]
        pct = float(probs[i]) * 100
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;margin:6px 0;">
                <div style="
                    background:{color};
                    width:{max(pct, 1):.1f}%;
                    height:22px;
                    border-radius:5px;">
                </div>
                <span style="margin-left:10px;"><b>{name}</b> {pct:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

st.title("😴 DeepSleepNet Sleep Stage Classifier")
st.caption(
    "Research demo for single-channel EEG sleep staging with DeepSleepNet-style CNN + sequence modeling."
)
st.divider()

with st.sidebar:
    st.header("Load data")

    use_local_data = st.checkbox(
        "Use local data/processed/all_subjects.pkl",
        value=DEFAULT_DATA_PATH.exists(),
    )

    uploaded_data = None
    if not use_local_data:
        uploaded_data = st.file_uploader("Subject/all_subjects .pkl file", type=["pkl"])

    st.divider()

    st.header("Load checkpoint")

    checkpoint_files = []
    if DEFAULT_CKPT_DIR.exists():
        checkpoint_files = sorted(DEFAULT_CKPT_DIR.glob("fold_*.pt"))

    use_local_checkpoint = st.checkbox(
        "Use local checkpoint",
        value=len(checkpoint_files) > 0,
    )

    selected_checkpoint_path = None
    uploaded_model = None

    if use_local_checkpoint and checkpoint_files:
        checkpoint_names = [p.name for p in checkpoint_files]
        selected_name = st.selectbox(
            "Choose checkpoint",
            checkpoint_names,
            index=0,
        )
        selected_checkpoint_path = str(DEFAULT_CKPT_DIR / selected_name)
    else:
        uploaded_model = st.file_uploader("Model checkpoint .pt", type=["pt"])

    st.divider()

    st.subheader("Sleep stages")
    for name in STAGE_NAMES:
        color = STAGE_COLORS[name]
        desc = STAGE_DESC[name]
        st.markdown(
            f"<span style='color:{color}'>■</span> **{name}** — {desc}",
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Reference F1 values")
    st.caption("Optional: enter paper/reference F1 values for visual comparison.")

    use_reference_f1 = st.checkbox("Show paper/reference bars", value=False)

    reference_f1 = []
    if use_reference_f1:
        for stage in STAGE_NAMES:
            value = st.number_input(
                f"{stage} reference F1",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
            )
            reference_f1.append(value)
    else:
        reference_f1 = None


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────

if use_local_checkpoint and selected_checkpoint_path:
    model = load_model_from_path(selected_checkpoint_path)
    st.sidebar.success(f"Loaded {Path(selected_checkpoint_path).name}")
elif uploaded_model is not None:
    model = load_model_from_upload(uploaded_model)
    st.sidebar.success("Uploaded checkpoint loaded")
else:
    model = DeepSleepNet(n_classes=5)
    model.eval()
    st.sidebar.warning("No checkpoint loaded — random weights")


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────

data = None

if use_local_data:
    if DEFAULT_DATA_PATH.exists():
        data = load_pickle_from_path(str(DEFAULT_DATA_PATH))
        st.sidebar.success("Loaded local all_subjects.pkl")
    else:
        st.sidebar.error("Local all_subjects.pkl not found")
else:
    if uploaded_data is not None:
        data = load_pickle_from_upload(uploaded_data)
        st.sidebar.success("Uploaded data loaded")


# ─────────────────────────────────────────────────────────────
# No data fallback
# ─────────────────────────────────────────────────────────────

if data is None:
    st.info("Load `data/processed/all_subjects.pkl` or upload a processed `.pkl` file to start.")

    st.subheader("Project pipeline")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### 1. EDF EEG")
        st.caption("Sleep-EDF PSG signal files.")

    with c2:
        st.markdown("### 2. Preprocessing")
        st.caption("30-second EEG epochs, stage labels.")

    with c3:
        st.markdown("### 3. DeepSleepNet")
        st.caption("CNN feature extraction + classifier.")

    with c4:
        st.markdown("### 4. Clinical visuals")
        st.caption("Hypnogram, confusion matrix, transitions, filters.")

    st.stop()


# ─────────────────────────────────────────────────────────────
# Prepare data
# ─────────────────────────────────────────────────────────────

X_all, y_all, subjects_all, raw_data = normalize_data(data)

is_synthetic = raw_data.get("synthetic", False)
sfreq = raw_data.get("sfreq", 100)
epoch_sec = raw_data.get("epoch_sec", 30)

if is_synthetic:
    st.warning("Synthetic data loaded — use real Sleep-EDF data for meaningful results.")

unique_subjects = np.unique(subjects_all)

st.success(
    f"Loaded dataset: {X_all.shape[0]} epochs | "
    f"{len(unique_subjects)} subjects | "
    f"{X_all.shape[-1]} samples/epoch | "
    f"sfreq={sfreq} Hz"
)

subject_options = ["All subjects"] + [str(s) for s in unique_subjects]

selected_subject = st.selectbox(
    "Select subject/night for visualization",
    subject_options,
    index=0,
)

if selected_subject == "All subjects":
    X = X_all
    y = y_all
    selected_subject_label = "All subjects"
else:
    selected_subject_value = type(unique_subjects[0])(selected_subject)
    mask = subjects_all == selected_subject_value
    X = X_all[mask]
    y = y_all[mask]
    selected_subject_label = f"Subject {selected_subject}"

if len(X) == 0:
    st.error("No epochs found for selected subject.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────────────────────

with st.spinner("Running model predictions..."):
    preds, probs_all = predict_all(model, X)

acc = accuracy_score(y, preds)
kappa = cohen_kappa_score(y, preds)
f1_macro = f1_score(
    y,
    preds,
    average="macro",
    labels=list(range(5)),
    zero_division=0,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected data", selected_subject_label)
m2.metric("Accuracy", f"{acc * 100:.1f}%")
m3.metric("Cohen Kappa", f"{kappa:.3f}")
m4.metric("F1-macro", f"{f1_macro:.3f}")


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────

tab_epoch, tab_hypno, tab_conf, tab_trans, tab_filters, tab_bench = st.tabs(
    [
        "Epoch explorer",
        "Hypnogram comparison",
        "Confusion matrix",
        "Transition rules",
        "Learned filters",
        "F1 benchmark",
    ]
)


# ─────────────────────────────────────────────────────────────
# Tab 1: Epoch explorer
# ─────────────────────────────────────────────────────────────

with tab_epoch:
    st.subheader("Epoch-level prediction explorer")

    idx = st.slider("Epoch", 0, len(y) - 1, 0)

    epoch = X[idx]
    true_l = int(y[idx])
    pred_l = int(preds[idx])
    probs = probs_all[idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("True stage", STAGE_NAMES[true_l])
    c2.metric("Predicted stage", STAGE_NAMES[pred_l])
    c3.metric("Correct?", "✅" if true_l == pred_l else "❌")

    display_probability_bars(probs)

    st.pyplot(plot_epoch_signal(epoch), use_container_width=True)

    st.caption(
        f"Epoch {idx}: true={STAGE_NAMES[true_l]}, predicted={STAGE_NAMES[pred_l]}. "
        f"Each epoch is {epoch_sec} seconds long."
    )


# ─────────────────────────────────────────────────────────────
# Tab 2: Hypnogram comparison
# ─────────────────────────────────────────────────────────────

with tab_hypno:
    st.subheader("True vs predicted full-night hypnogram")

    st.pyplot(plot_hypnogram_comparison(y, preds), use_container_width=True)

    st.markdown(
        """
        **Why this matters:**  
        A hypnogram is the clinical output sleep labs care about.  
        It shows the sleep architecture over time and makes systematic model errors visible,
        for example repeated confusion during N1 transition periods.
        """
    )

    hyp_df = pd.DataFrame({
        "Epoch": np.arange(len(y)),
        "True": get_stage_labels(y),
        "Predicted": get_stage_labels(preds),
        "Correct": y == preds,
    })

    with st.expander("View epoch-by-epoch hypnogram table"):
        st.dataframe(hyp_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 3: Confusion matrix
# ─────────────────────────────────────────────────────────────

with tab_conf:
    st.subheader("Confusion matrix")

    c1, c2 = st.columns([2, 1])

    with c1:
        normalize_cm = st.checkbox("Normalize by true class", value=True)
        st.pyplot(
            plot_confusion_matrix(y, preds, normalize=normalize_cm),
            use_container_width=True,
        )

    with c2:
        st.markdown("### Clinical interpretation")
        st.markdown(
            """
            Common sleep-stage confusions are expected:

            - **N1 ↔ Wake**: N1 is transitional and visually subtle.
            - **N1 ↔ N2**: N1 can look similar before clear spindles/K-complexes.
            - **REM ↔ Wake/N1**: EEG may be mixed-frequency and low-amplitude.
            - **N3 is often easier** because delta waves are more distinctive.
            """
        )

    report = classification_report(
        y,
        preds,
        labels=list(range(5)),
        target_names=STAGE_NAMES,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).T

    with st.expander("Classification report"):
        st.dataframe(report_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 4: Transition-rule analysis
# ─────────────────────────────────────────────────────────────

with tab_trans:
    st.subheader("Transition-rule error analysis")

    transition_errors = compute_transition_error_table(y, preds)

    if transition_errors.empty:
        st.success("No transition-related errors found for this selection.")
    else:
        fig = plot_transition_heatmap(transition_errors)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

        st.markdown(
            """
            **What this shows:**  
            Instead of only asking “which class was wrong?”, this asks:

            > Given the previous true sleep stage, which stage transition did the model fail on?

            This is more clinically useful because sleep staging errors often happen near
            stage boundaries, especially around **Wake → N1**, **N1 → N2**, and **REM/Wake-like** transitions.
            """
        )

        st.markdown("### Most common transition mistakes")
        st.dataframe(
            transition_errors.head(20),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────
# Tab 5: Learned filter visualization
# ─────────────────────────────────────────────────────────────

with tab_filters:
    st.subheader("Learned CNN filter visualization")

    result, note = plot_filter_visualization(model, fs=sfreq)

    st.info(note)

    if result is None:
        st.warning("Could not visualize filters for this model.")
    else:
        fig_time, fig_freq = result

        st.pyplot(fig_time, use_container_width=True)
        st.pyplot(fig_freq, use_container_width=True)

        st.markdown(
            """
            **How to explain this in your presentation:**

            DeepSleepNet uses different temporal filter sizes to capture different EEG patterns.

            - Shorter filters can capture short events such as **sleep spindles** around 12–15 Hz.
            - Longer filters can capture slower oscillations such as **delta activity** around 0.5–4 Hz.
            - If the learned frequency response has energy near these bands, it supports the claim
              that the network learned meaningful EEG features, not just arbitrary patterns.
            """
        )


# ─────────────────────────────────────────────────────────────
# Tab 6: F1 benchmark
# ─────────────────────────────────────────────────────────────

with tab_bench:
    st.subheader("Per-class F1 benchmark")

    fig, f1_df = plot_f1_benchmark(
        y,
        preds,
        paper_f1_values=reference_f1,
    )

    st.pyplot(fig, use_container_width=True)
    st.dataframe(f1_df, use_container_width=True)

    st.markdown(
        """
        **How to use this section:**

        - Your model bars are computed from the currently selected subject or all subjects.
        - Enable **paper/reference bars** in the sidebar and enter the values from the research paper/table.
        - This creates a direct reproduction-style comparison for your report.
        """
    )

    st.markdown("### Current summary")
    st.write(
        {
            "accuracy": float(acc),
            "cohen_kappa": float(kappa),
            "f1_macro": float(f1_macro),
            "epochs": int(len(y)),
            "selection": selected_subject_label,
        }
    )