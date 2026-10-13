"""
MediSense Autoencoder — the core model for anomaly detection.

Architecture: Symmetric encoder-decoder.
    Input (17 features) → Encoder → Latent (8-dim) → Decoder → Reconstruction

Anomaly detection: reconstruction error.
    Normal signals reconstruct well (low error).
    Anomalies reconstruct poorly (high error) — the model has never seen them.

This is the same architecture used for BOTH:
    - Industrial pre-training (NASA bearing, CMAPSS, SKAB)
    - Medical fine-tuning (BIDMC, CapnoBase, MIMIC)

Transfer learning: pre-trained encoder weights are loaded, early layers frozen,
and the decoder is retrained on medical data. The latent space generalises across
domains because both industrial and medical sensors share the same fundamental
anomaly structure: normal → degraded → failure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class AutoencoderConfig:
    input_dim: int = 17           # feature vector size (from signal pipeline)
    hidden_dims: tuple = (64, 32, 16)
    latent_dim: int = 8
    dropout: float = 0.2
    activation: str = "relu"      # relu | leaky_relu | gelu


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.1), "gelu": nn.GELU()}[name]


class Encoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        act = _activation(cfg.activation)
        dims = [cfg.input_dim] + list(cfg.hidden_dims) + [cfg.latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(_activation(cfg.activation))
            if cfg.dropout > 0 and i < len(dims) - 2:
                layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        dims = [cfg.latent_dim] + list(reversed(cfg.hidden_dims)) + [cfg.input_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(_activation(cfg.activation))
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MediSenseAutoencoder(nn.Module):
    """
    Full autoencoder. Input → latent → reconstruction.
    Anomaly score = mean squared reconstruction error per sample.
    """

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_code)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error, shape (N,). Always runs in eval mode."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            scores = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)
        if was_training:
            self.train()
        return scores

    def freeze_encoder_layers(self, n_layers: int) -> None:
        """
        Freeze the first n_layers Linear layers of the encoder.
        Called before medical fine-tuning to preserve general sensor features.
        """
        linear_count = 0
        for module in self.encoder.net:
            if isinstance(module, nn.Linear):
                if linear_count < n_layers:
                    for param in module.parameters():
                        param.requires_grad = False
                linear_count += 1

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path: Union[str, Path]) -> None:
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MediSenseAutoencoder":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(checkpoint["cfg"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
