"""
CapnoBase IEEE TBME Respiratory Rate Benchmark loader.

Dataset: https://borealisdata.ca/dataverse/capnobase
42 patients during anesthesia, 8-minute recordings at 300 Hz.
Format: MATLAB v7.3 HDF5 (.mat files).

Signals used:
    signal/pleth/y          — PPG (photoplethysmogram) — main input signal
    param/samplingrate/pleth — sampling rate (300 Hz)

Ground truth (directly stored, not computed):
    reference/hr/pleth/y    — HR in BPM at each PPG beat (from expert peak labels)
    reference/rr/co2/y      — RR in breaths/min at each breath (from capnography)

These are the gold-standard reference values used in the benchmark paper:
    Charlton et al. (2018) "Breathing Rate Estimation from the Electrocardiogram
    and Photoplethysmogram: A Review", IEEE Reviews in Biomedical Engineering.
"""

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from .base import (
    AnomalyLabel,
    DataDomain,
    VitalSigns,
    Window,
    WindowedDataset,
    sliding_windows,
)

# Clinical alert thresholds
HR_LOW,  HR_HIGH = 40,  140
RR_LOW,  RR_HIGH = 6,   35


def _classify_vitals(hr: Optional[float], rr: Optional[float]) -> AnomalyLabel:
    if hr is not None and not (HR_LOW <= hr <= HR_HIGH):
        return AnomalyLabel.PATIENT_ANOMALY
    if rr is not None and not (RR_LOW <= rr <= RR_HIGH):
        return AnomalyLabel.PATIENT_ANOMALY
    return AnomalyLabel.NORMAL


def _load_mat(mat_path: Path) -> Optional[tuple]:
    """
    Load one CapnoBase .mat (HDF5/v7.3) file.
    Returns (ppg, fs, median_hr_bpm, median_rr_bpm) or None.
    """
    try:
        with h5py.File(str(mat_path), "r") as f:
            fs   = float(f["param/samplingrate/pleth"][0, 0])
            ppg  = np.array(f["signal/pleth/y"]).flatten().astype(np.float32)

            # Reference HR: per-beat BPM values → take median for the recording
            hr_vals = np.array(f["reference/hr/pleth/y"]).flatten()
            hr_vals = hr_vals[(hr_vals > 20) & (hr_vals < 300)]  # sanity filter
            hr = float(np.median(hr_vals)) if len(hr_vals) > 0 else None

            # Reference RR: per-breath breaths/min values → take median
            rr_vals = np.array(f["reference/rr/co2/y"]).flatten()
            rr_vals = rr_vals[(rr_vals > 2) & (rr_vals < 60)]   # sanity filter
            rr = float(np.median(rr_vals)) if len(rr_vals) > 0 else None

        return ppg, fs, hr, rr

    except Exception:
        return None


def load_capnobase(
    data_dir: Union[str, Path],
    window_sec: float = 10.0,
    overlap: float = 0.5,
    max_records: Optional[int] = None,
) -> WindowedDataset:
    """
    Load all CapnoBase .mat records and return a WindowedDataset.

    Args:
        data_dir: Path to data/raw/capnobase (contains *_8min.mat files).
        window_sec: Window length in seconds.
        overlap: Fractional overlap between consecutive windows.
        max_records: Cap number of records for fast dev runs.

    Returns:
        WindowedDataset with domain=MEDICAL.
    """
    data_dir  = Path(data_dir)
    mat_files = sorted(data_dir.glob("*.mat"))

    if max_records:
        mat_files = mat_files[:max_records]

    windows:  list[Window] = []
    n_normal = n_anomaly = n_skipped = 0

    for mat_path in mat_files:
        result = _load_mat(mat_path)
        if result is None:
            n_skipped += 1
            continue

        ppg, fs, hr, rr = result
        vitals = VitalSigns(heart_rate_bpm=hr, respiratory_rate_bpm=rr)
        label  = _classify_vitals(hr, rr)

        for idx, win_signal in enumerate(
            sliding_windows(ppg, fs, window_sec, overlap)
        ):
            windows.append(
                Window(
                    signal=win_signal,
                    fs=fs,
                    domain=DataDomain.MEDICAL,
                    label=label,
                    ground_truth=vitals,
                    patient_id=mat_path.stem,
                    window_idx=idx,
                )
            )
            if label == AnomalyLabel.NORMAL:
                n_normal += 1
            else:
                n_anomaly += 1

    print(
        f"[CapnoBase] Loaded {len(windows)} windows from "
        f"{len(mat_files) - n_skipped}/{len(mat_files)} records "
        f"({n_normal} normal, {n_anomaly} anomaly)"
    )
    return WindowedDataset(
        windows=windows,
        domain=DataDomain.MEDICAL,
        dataset_name="CapnoBase",
        n_normal=n_normal,
        n_anomaly=n_anomaly,
    )
