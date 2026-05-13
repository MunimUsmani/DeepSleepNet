"""
Step 4: Dataset + Training Loop
- Handles heavy class imbalance (N2 dominates overnight recordings)
- Leave-One-Subject-Out cross-validation
- Weighted loss + weighted sampler for balanced training
- Reports per-class F1 (standard sleep staging metric)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
)
from src.model import DeepSleepNet

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


# ── Dataset ─────────────────────────────────────────────────────────────────

class SleepDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, 1, 3000)  y: (N,)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Oversample minority classes (N1 is very rare)."""
    classes, counts = np.unique(y, return_counts=True)
    class_weights   = 1.0 / counts
    sample_weights  = class_weights[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(y),
        replacement=True,
    )


# ── Training utilities ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        preds   = model(X_batch).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc    = accuracy_score(all_labels, all_preds)
    kappa  = cohen_kappa_score(all_labels, all_preds)
    f1_per = f1_score(all_labels, all_preds, average=None,
                      labels=list(range(5)), zero_division=0)
    f1_mac = f1_score(all_labels, all_preds, average="macro",
                      labels=list(range(5)), zero_division=0)
    cm     = confusion_matrix(all_labels, all_preds, labels=list(range(5)))

    return acc, kappa, f1_per, f1_mac, cm


# ── LOSO Training ────────────────────────────────────────────────────────────

def train_loso(
    data_path:  str   = "data/processed/all_subjects.pkl",
    n_epochs:   int   = 30,
    batch_size: int   = 128,
    lr:         float = 1e-3,
    ckpt_dir:   str   = "checkpoints",
):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    X_all   = data["X"]
    y_all   = data["y"]
    s_all   = data["subjects"]
    is_syn  = data.get("synthetic", False)

    if is_syn:
        print("NOTE: Training on synthetic data — for pipeline testing only")

    unique_subjects = np.unique(s_all)
    print(f"Dataset: {X_all.shape} | {len(unique_subjects)} subjects | 5 classes")
    print(f"Class dist: { {STAGE_NAMES[k]: int(v) for k,v in zip(*np.unique(y_all, return_counts=True))} }\n")

    # Class weights for loss (inverse frequency)
    _, counts     = np.unique(y_all, return_counts=True)
    class_weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)

    fold_results = []

    for fold, test_subj in enumerate(unique_subjects):
        print(f"── Fold {fold+1}/{len(unique_subjects)} | Test subject: {test_subj:02d} ──")

        test_mask  = s_all == test_subj
        train_mask = ~test_mask
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]

        sampler  = make_weighted_sampler(y_tr)
        train_dl = DataLoader(SleepDataset(X_tr, y_tr),
                              batch_size=batch_size, sampler=sampler)
        test_dl  = DataLoader(SleepDataset(X_te, y_te),
                              batch_size=batch_size, shuffle=False)

        model     = DeepSleepNet(n_classes=5).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_kappa = -1.0
        ckpt_path  = os.path.join(ckpt_dir, f"fold_{test_subj:02d}.pt")

        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_dl, optimizer, criterion, device)
            acc, kappa, f1_per, f1_mac, _ = evaluate(model, test_dl, device)
            scheduler.step()

            if kappa > best_kappa:
                best_kappa = kappa
                torch.save(model.state_dict(), ckpt_path)

            if epoch % 5 == 0 or epoch == n_epochs:
                print(f"  Ep {epoch:3d} | tr_acc {tr_acc:.3f} | "
                      f"test_acc {acc:.3f} | kappa {kappa:.3f} | "
                      f"F1-macro {f1_mac:.3f}")

        # Final eval with best checkpoint
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        acc, kappa, f1_per, f1_mac, cm = evaluate(model, test_dl, device)
        fold_results.append({
            "subject": test_subj,
            "acc": acc, "kappa": kappa,
            "f1_per": f1_per, "f1_mac": f1_mac, "cm": cm,
        })
        f1_str = " | ".join(f"{STAGE_NAMES[i]}:{f1_per[i]:.2f}"
                            for i in range(5))
        print(f"  Best → acc {acc:.3f} | kappa {kappa:.3f} | {f1_str}\n")

    # Summary
    accs    = [r["acc"]    for r in fold_results]
    kappas  = [r["kappa"]  for r in fold_results]
    f1_macs = [r["f1_mac"] for r in fold_results]

    print("=" * 60)
    print("LOSO Results — DeepSleepNet")
    print(f"  Accuracy  : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Kappa     : {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
    print(f"  F1-macro  : {np.mean(f1_macs):.3f} ± {np.std(f1_macs):.3f}")
    print("\n  Per-class F1 (mean across subjects):")
    mean_f1 = np.mean([r["f1_per"] for r in fold_results], axis=0)
    for i, name in enumerate(STAGE_NAMES):
        print(f"    {name:5s}: {mean_f1[i]:.3f}")
    print("=" * 60)

    summary_path = os.path.join(ckpt_dir, "results_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(fold_results, f)

    return fold_results


if __name__ == "__main__":
    train_loso(
        data_path  = "data/processed/all_subjects.pkl",
        n_epochs   = 30,
        batch_size = 128,
        lr         = 1e-3,
    )