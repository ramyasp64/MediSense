"""
Training engine for MediSense autoencoder.

Handles both phases:
    Phase 1 — Industrial pre-training (train on normal industrial windows only)
    Phase 2 — Medical fine-tuning   (freeze early encoder, train on normal medical windows)

Autoencoders for anomaly detection are trained on NORMAL data only.
At inference, anomalies are identified by high reconstruction error.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from .autoencoder import MediSenseAutoencoder, AutoencoderConfig


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    freeze_encoder_layers: int = 2    # for fine-tuning phase
    device: str = "auto"              # auto | cpu | cuda | mps
    save_dir: str = "experiments/models"
    run_name: str = "run"


@dataclass
class TrainingLog:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    training_time_sec: float = 0.0
    phase: str = ""

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def _get_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _make_loader(
    features: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    t = torch.tensor(features, dtype=torch.float32)
    return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)


class MediSenseTrainer:
    def __init__(
        self,
        model: MediSenseAutoencoder,
        cfg: TrainingConfig,
    ):
        self.model = model
        self.cfg = cfg
        self.device = _get_device(cfg.device)
        self.model.to(self.device)
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    def pretrain(
        self,
        train_normal: np.ndarray,
        val_normal: np.ndarray,
    ) -> TrainingLog:
        """
        Phase 1: Pre-train on industrial normal data.
        All encoder/decoder parameters are trainable.
        """
        print(f"\n{'='*60}")
        print("PHASE 1: Industrial Pre-training")
        print(f"  Train: {len(train_normal)} normal windows")
        print(f"  Val:   {len(val_normal)} normal windows")
        print(f"  Device: {self.device}")
        print(f"{'='*60}")

        self.model.unfreeze_all()
        return self._train_loop(
            train_normal, val_normal, phase="pretrain"
        )

    def finetune(
        self,
        train_normal: np.ndarray,
        val_normal: np.ndarray,
        pretrained_path: Optional[Union[str, Path]] = None,
    ) -> TrainingLog:
        """
        Phase 2: Fine-tune on medical normal data.
        Loads pre-trained weights (if path given), freezes early encoder layers,
        then trains the rest on medical data.
        """
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])
            print(f"\nLoaded pre-trained weights from {pretrained_path}")

        self.model.freeze_encoder_layers(self.cfg.freeze_encoder_layers)

        n_frozen = sum(
            1 for p in self.model.parameters() if not p.requires_grad
        )
        n_total = sum(1 for p in self.model.parameters())
        print(f"\n{'='*60}")
        print("PHASE 2: Medical Fine-tuning")
        print(f"  Train: {len(train_normal)} normal windows")
        print(f"  Val:   {len(val_normal)} normal windows")
        print(f"  Frozen params: {n_frozen}/{n_total}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}")

        return self._train_loop(
            train_normal, val_normal, phase="finetune"
        )

    def train_baseline(
        self,
        train_normal: np.ndarray,
        val_normal: np.ndarray,
    ) -> TrainingLog:
        """
        Baseline: train from scratch on medical data only (no industrial pre-training).
        Used for comparison — the transfer learning experiment competes against this.
        """
        print(f"\n{'='*60}")
        print("BASELINE: Medical-only training (no transfer learning)")
        print(f"  Train: {len(train_normal)} normal windows")
        print(f"  Val:   {len(val_normal)} normal windows")
        print(f"{'='*60}")

        self.model.unfreeze_all()
        # Re-initialise weights — baseline must start fresh
        self.model.apply(self._reset_weights)
        return self._train_loop(
            train_normal, val_normal, phase="baseline"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        train_normal: np.ndarray,
        val_normal: np.ndarray,
        phase: str,
    ) -> TrainingLog:
        epochs = (
            self.cfg.epochs
            if phase in ("pretrain", "baseline")
            else max(self.cfg.epochs // 2, 30)
        )

        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.learning_rate,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        train_loader = _make_loader(train_normal, self.cfg.batch_size, shuffle=True)
        val_loader = _make_loader(val_normal, self.cfg.batch_size, shuffle=False)

        log = TrainingLog(phase=phase)
        best_state = None
        patience_counter = 0
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                x_hat, _ = self.model(batch)
                loss = criterion(x_hat, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(train_normal)

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    x_hat, _ = self.model(batch)
                    val_loss += criterion(x_hat, batch).item() * len(batch)
            val_loss /= len(val_normal)

            scheduler.step()
            log.train_losses.append(train_loss)
            log.val_losses.append(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:4d}/{epochs} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
                )

            # Early stopping & checkpoint
            if val_loss < log.best_val_loss:
                log.best_val_loss = val_loss
                log.best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if best_state:
            self.model.load_state_dict(best_state)

        log.training_time_sec = time.time() - t0
        save_path = Path(self.cfg.save_dir) / f"{self.cfg.run_name}_{phase}.pt"
        self.model.save(save_path)
        log.save(Path(self.cfg.save_dir) / f"{self.cfg.run_name}_{phase}_log.json")

        print(
            f"\n  Best val_loss={log.best_val_loss:.6f} at epoch {log.best_epoch} "
            f"({log.training_time_sec:.1f}s). Saved → {save_path}"
        )
        return log

    @staticmethod
    def _reset_weights(m: nn.Module) -> None:
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
