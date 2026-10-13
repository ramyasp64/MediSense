"""
MediSense 1D-CNN Autoencoder — operates on raw signal windows.

Why CNN instead of feature vectors:
    The 17-dim feature vector discards temporal structure (peak morphology,
    waveform shape, inter-beat variability) that distinguishes normal from
    abnormal PPG. A 1D-CNN preserves this structure at every layer.

Architecture (input: batch × 1 × 1250 samples):

    Encoder:
        Conv1d(1→32,  k=7, s=2) → 625
        Conv1d(32→64, k=5, s=2) → 313
        Conv1d(64→128,k=5, s=2) → 157
        Conv1d(128→64,k=3, s=2) → 79
        Flatten → Linear(5056→128) = latent

    Decoder (mirror):
        Linear(128→5056) → reshape(64, 79)
        ConvTranspose1d(64→128, k=3, s=2)         → 157
        ConvTranspose1d(128→64, k=5, s=2)         → 313
        ConvTranspose1d(64→32,  k=5, s=2)         → 625
        ConvTranspose1d(32→1,   k=7, s=2, op=1)   → 1250

Anomaly score: mean squared reconstruction error per window.
Normal signals reconstruct with low error.
Abnormal signals (unseen patterns) reconstruct poorly — high error.

Transfer learning:
    Pre-train on NASA Bearing vibration data (same structure: periodic signal,
    normal→degraded→failure). Fine-tune on medical PPG with encoder frozen.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


FIXED_LEN = 1250   # all windows resampled to this length before model


@dataclass
class CNNConfig:
    input_len: int = FIXED_LEN
    latent_dim: int = 256
    dropout: float = 0.15


class CNNEncoder(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Conv1d(32, 64,  kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        # 64 channels × 79 time steps = 5056
        self.flat_dim = 64 * 79
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 512), nn.GELU(),
            nn.Linear(512, cfg.latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 1250)
        h = self.conv(x)            # (B, 64, 79)
        h = h.flatten(1)            # (B, 5056)
        return self.fc(h)           # (B, latent_dim)


class CNNDecoder(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.flat_dim = 64 * 79
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim, 512), nn.GELU(),
            nn.Linear(512, self.flat_dim)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.ConvTranspose1d(64, 32,  kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(cfg.dropout),
            # output_padding=1 to recover exact 1250 from 625
            nn.ConvTranspose1d(32, 1,   kernel_size=7, stride=2, padding=3,
                               output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)              # (B, 5056)
        h = h.view(-1, 64, 79)     # (B, 64, 79)
        return self.deconv(h)       # (B, 1, 1250)


class MediSenseCNN(nn.Module):
    """
    1D-CNN autoencoder for raw PPG/sensor anomaly detection.
    Input/output shape: (batch, 1, FIXED_LEN)
    """

    def __init__(self, cfg: CNNConfig = None):
        super().__init__()
        self.cfg = cfg or CNNConfig()
        self.encoder = CNNEncoder(self.cfg)
        self.decoder = CNNDecoder(self.cfg)

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (reconstruction, latent_code)."""
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent vectors, shape (N, latent_dim). Always eval mode."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _, z = self.forward(x)
        if was_training:
            self.train()
        return z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error, shape (N,). Always eval mode."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            scores = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))
        if was_training:
            self.train()
        return scores

    def fit_mahalanobis(self, normal_x: torch.Tensor) -> None:
        """
        Fit Mahalanobis distance scorer on normal training windows.
        Stores mean and precision matrix of latent vectors.
        Call after fine-tuning, before evaluation.
        """
        import numpy as np
        from sklearn.covariance import EmpiricalCovariance
        z = self.encode(normal_x).cpu().numpy()  # (N, D)
        self._maha_mean = z.mean(axis=0)         # (D,)
        cov = EmpiricalCovariance().fit(z)
        self._maha_precision = cov.precision_     # (D, D)

    def mahalanobis_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mahalanobis distance from normal cluster in latent space.
        Higher = more anomalous. Shape (N,).
        Requires fit_mahalanobis() to have been called first.
        """
        import numpy as np
        z = self.encode(x).cpu().numpy()  # (N, D)
        diff = z - self._maha_mean        # (N, D)
        # dist_i = sqrt(diff_i @ P @ diff_i^T)
        scores = np.sqrt(np.einsum("nd,dd,nd->n", diff, self._maha_precision, diff))
        return torch.tensor(scores, dtype=torch.float32)

    def freeze_encoder_layers(self, n_conv_blocks: int = 2) -> None:
        """Freeze first n conv blocks of encoder for fine-tuning."""
        # Each block: Conv1d + BatchNorm1d + activation + optional Dropout = 3-4 modules
        frozen = 0
        for module in self.encoder.conv:
            if isinstance(module, nn.Conv1d):
                if frozen < n_conv_blocks:
                    for p in module.parameters():
                        p.requires_grad = False
                frozen += 1

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def save(self, path: Union[str, Path]) -> None:
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MediSenseCNN":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["cfg"])
        model.load_state_dict(ckpt["state_dict"])
        return model
