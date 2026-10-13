"""
Shared base classes and types for all MediSense dataset loaders.
Every loader returns WindowedDataset — the model only ever sees this contract.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np


class DataDomain(Enum):
    MEDICAL = "medical"
    INDUSTRIAL = "industrial"


class AnomalyLabel(Enum):
    NORMAL = 0
    SENSOR_MALFUNCTION = 1   # Level 1: equipment failure
    PATIENT_ANOMALY = 2      # Level 2: abnormal vital signs


@dataclass
class VitalSigns:
    """Ground-truth vital signs from reference medical equipment."""
    heart_rate_bpm: Optional[float] = None        # beats per minute
    respiratory_rate_bpm: Optional[float] = None  # breaths per minute
    spo2_pct: Optional[float] = None              # blood oxygen %
    temperature_c: Optional[float] = None         # body temperature


@dataclass
class Window:
    """A single analysis window — the atomic unit fed into the model."""
    signal: np.ndarray          # shape: (n_samples,) or (n_channels, n_samples)
    fs: float                   # sampling rate in Hz
    domain: DataDomain
    label: AnomalyLabel
    ground_truth: Optional[VitalSigns] = None
    patient_id: Optional[str] = None
    window_idx: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class WindowedDataset:
    """Full collection of windows from one dataset, ready for training."""
    windows: list[Window]
    domain: DataDomain
    dataset_name: str
    n_normal: int = 0
    n_anomaly: int = 0

    def __len__(self):
        return len(self.windows)

    def signals(self) -> np.ndarray:
        """Stack all window signals into shape (N, n_samples)."""
        return np.stack([w.signal for w in self.windows])

    def labels(self) -> np.ndarray:
        """Integer labels array of shape (N,)."""
        return np.array([w.label.value for w in self.windows])

    def binary_labels(self) -> np.ndarray:
        """0=normal, 1=any anomaly — for binary anomaly detection."""
        return (self.labels() > 0).astype(int)


def sliding_windows(
    signal: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """Slice a 1D signal into overlapping fixed-length windows."""
    window_samples = int(window_sec * fs)
    step = int(window_samples * (1 - overlap))
    windows = []
    start = 0
    while start + window_samples <= len(signal):
        windows.append(signal[start : start + window_samples])
        start += step
    return windows
