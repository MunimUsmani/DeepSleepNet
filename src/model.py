"""
Step 3: DeepSleepNet Architecture
Implements the representation learning part of DeepSleepNet from:
Supratak et al., "DeepSleepNet: A Model for Automatic Sleep Stage Scoring
Based on Raw Single-Channel EEG", IEEE TNSRE 2017.
DOI: 10.1109/TNSRE.2017.2721116

Key idea: two parallel CNN branches capture DIFFERENT temporal scales
  - Small filters (L branch): captures sleep spindles, K-complexes (high freq)
  - Large filters (S branch): captures slow-wave delta patterns (low freq)
  → Concatenated → BiLSTM for sequence context → classifier

Input: single-channel EEG epoch (1 × 3000 samples @ 100 Hz = 30 seconds)
Output: 5-class sleep stage (W, N1, N2, N3, REM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallFilterBranch(nn.Module):
    """
    Small-filter CNN branch (L branch in paper).
    Kernel size = Fs/2 = 50 (0.5-second window at 100 Hz).
    Captures high-frequency features: sleep spindles (12-15 Hz), K-complexes.
    """

    def __init__(self):
        super().__init__()
        # Stage 1
        self.conv1  = nn.Conv1d(1,  64, kernel_size=50, stride=6, padding=24, bias=False)
        self.bn1    = nn.BatchNorm1d(64)
        self.pool1  = nn.MaxPool1d(kernel_size=8, stride=8)
        self.drop1  = nn.Dropout(0.5)

        # Stage 2: deeper feature extraction
        self.conv2a = nn.Conv1d(64, 128, kernel_size=8, padding=4, bias=False)
        self.bn2a   = nn.BatchNorm1d(128)
        self.conv2b = nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False)
        self.bn2b   = nn.BatchNorm1d(128)
        self.conv2c = nn.Conv1d(128, 128, kernel_size=8, padding=3, bias=False)
        self.bn2c   = nn.BatchNorm1d(128)
        self.pool2  = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop2  = nn.Dropout(0.5)

    def forward(self, x):
        # x: (B, 1, 3000)
        x = self.drop1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = F.elu(self.bn2a(self.conv2a(x)))
        x = F.elu(self.bn2b(self.conv2b(x)))
        x = F.elu(self.bn2c(self.conv2c(x)))
        x = self.drop2(self.pool2(x))
        return x   # (B, 128, T_small)


class LargeFilterBranch(nn.Module):
    """
    Large-filter CNN branch (S branch in paper).
    Kernel size = 4×Fs = 400 (4-second window at 100 Hz).
    Captures low-frequency features: delta waves (0.5-4 Hz), slow oscillations.
    """

    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv1d(1,  64, kernel_size=400, stride=50, padding=200, bias=False)
        self.bn1    = nn.BatchNorm1d(64)
        self.pool1  = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1  = nn.Dropout(0.5)

        self.conv2a = nn.Conv1d(64, 128, kernel_size=6, padding=3, bias=False)
        self.bn2a   = nn.BatchNorm1d(128)
        self.conv2b = nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False)
        self.bn2b   = nn.BatchNorm1d(128)
        self.conv2c = nn.Conv1d(128, 128, kernel_size=6, padding=2, bias=False)
        self.bn2c   = nn.BatchNorm1d(128)
        self.pool2  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2  = nn.Dropout(0.5)

    def forward(self, x):
        x = self.drop1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = F.elu(self.bn2a(self.conv2a(x)))
        x = F.elu(self.bn2b(self.conv2b(x)))
        x = F.elu(self.bn2c(self.conv2c(x)))
        x = self.drop2(self.pool2(x))
        return x   # (B, 128, T_large)


class DeepSleepNet(nn.Module):
    """
    DeepSleepNet: Two-branch CNN + BiLSTM for sleep stage scoring.

    Architecture:
      Input EEG epoch
        → [SmallFilterBranch] → flatten
        → [LargeFilterBranch] → flatten
        → Concatenate → Dropout
        → BiLSTM (sequence context across epochs)
        → Residual connection
        → Classifier
    """

    def __init__(self, n_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        self.small_branch = SmallFilterBranch()
        self.large_branch = LargeFilterBranch()

        # Compute combined feature size with dummy forward pass
        self._feat_size = self._get_feature_size()

        self.drop_merged = nn.Dropout(dropout)

        # BiLSTM: processes a sequence of epochs (captures transition context)
        # e.g., N2→N3→N3→REM transitions are physiologically constrained
        self.lstm = nn.LSTM(
            input_size=self._feat_size,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Residual projection (feat_size → 1024 to match BiLSTM output)
        self.residual_proj = nn.Linear(self._feat_size, 1024)

        self.drop_lstm = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, n_classes)

    def _get_feature_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 3000)
            s = self.small_branch(dummy).flatten(1)
            l = self.large_branch(dummy).flatten(1)
            return s.shape[1] + l.shape[1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract merged CNN features from a single epoch.
        x: (B, 1, 3000)
        returns: (B, feat_size)
        """
        s = self.small_branch(x).flatten(1)
        l = self.large_branch(x).flatten(1)
        merged = torch.cat([s, l], dim=1)
        return self.drop_merged(merged)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Epoch-by-epoch classification (no sequence context).
        x: (B, 1, 3000)
        returns logits: (B, n_classes)
        """
        feat = self.forward_features(x)
        # Project residual
        res = self.residual_proj(feat)
        # Single-step LSTM (sequence_len=1)
        lstm_out, _ = self.lstm(feat.unsqueeze(1))
        out = lstm_out.squeeze(1) + res
        return self.classifier(self.drop_lstm(out))

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Sequence-aware classification (better accuracy).
        x_seq: (B, seq_len, 1, 3000) — batch of epoch sequences
        returns logits: (B, seq_len, n_classes)
        """
        B, T, C, S = x_seq.shape
        # Extract features for all epochs in parallel
        x_flat = x_seq.view(B * T, C, S)
        feats   = self.forward_features(x_flat)
        feats   = feats.view(B, T, -1)

        # BiLSTM over sequence
        lstm_out, _ = self.lstm(feats)   # (B, T, 1024)

        # Residual
        res = self.residual_proj(feats)   # (B, T, 1024)
        out = lstm_out + res              # (B, T, 1024)

        return self.classifier(self.drop_lstm(out))   # (B, T, n_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DeepSleepNet(n_classes=5)
    dummy = torch.zeros(8, 1, 3000)

    print("=" * 50)
    print("DeepSleepNet")
    print(f"  Feature size : {model._feat_size:,}")
    print(f"  Parameters   : {model.count_parameters():,}")
    print(f"  Input shape  : {tuple(dummy.shape)}")
    print(f"  Output shape : {tuple(model(dummy).shape)}")

    # Test sequence mode
    seq = torch.zeros(4, 10, 1, 3000)   # batch=4, seq_len=10
    seq_out = model.forward_sequence(seq)
    print(f"  Sequence out : {tuple(seq_out.shape)}")
    print("=" * 50)
    print("Model OK")