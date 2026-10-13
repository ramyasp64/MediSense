"""
MIMIC-III Waveform Database loader.

Dataset: https://physionet.org/content/mimic3wdb/1.0/
Requires PhysioNet credentialing. Apply at: physionet.org/settings/credentialing

Contains ICU waveforms (PPG, ECG, arterial BP) and vital signs numerics.
Ground truth: bedside monitor measurements (HR, SpO2, RR, ABP).

This loader is ready and waiting — plug in your data_dir once access is granted.
Download a subset first to validate:
    wfdb.dl_database('mimic3wdb-matched', 'data/raw/mimic', records=['p00/p000020/3141595'])

Full download is large (many TB). Use mimic3wdb-matched (matched waveforms subset).
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import wfdb

from .base import (
    AnomalyLabel,
    DataDomain,
    VitalSigns,
    Window,
    WindowedDataset,
    sliding_windows,
)

# Clinical alert thresholds (consistent with BIDMC loader)
HR_LOW, HR_HIGH = 40, 140
RR_LOW, RR_HIGH = 6, 35
SPO2_LOW = 90  # % — below 90% = hypoxemia alert


def _classify_vitals(hr, rr, spo2) -> AnomalyLabel:
    if hr is not None and not (HR_LOW <= hr <= HR_HIGH):
        return AnomalyLabel.PATIENT_ANOMALY
    if rr is not None and not (RR_LOW <= rr <= RR_HIGH):
        return AnomalyLabel.PATIENT_ANOMALY
    if spo2 is not None and spo2 < SPO2_LOW:
        return AnomalyLabel.PATIENT_ANOMALY
    return AnomalyLabel.NORMAL


def load_mimic(
    data_dir: Union[str, Path],
    window_sec: float = 10.0,
    overlap: float = 0.5,
    max_records: Optional[int] = None,
    signal_channel: str = "PLETH",  # PPG channel name in MIMIC
) -> WindowedDataset:
    """
    Load MIMIC-III waveform records.

    Args:
        data_dir: Root of downloaded mimic3wdb-matched directory.
        window_sec: Window size in seconds.
        overlap: Overlap between windows.
        max_records: Limit records for dev/test runs.
        signal_channel: Waveform channel to extract ('PLETH' for PPG).

    Returns:
        WindowedDataset with domain=MEDICAL.
    """
    data_dir = Path(data_dir)
    windows: list[Window] = []
    n_normal = n_anomaly = 0

    # MIMIC record list is in RECORDS file
    records_file = data_dir / "RECORDS"
    if not records_file.exists():
        print(
            "[MIMIC] RECORDS file not found. "
            "Run: python scripts/download_data.py --mimic once access is granted."
        )
        return WindowedDataset(
            windows=[],
            domain=DataDomain.MEDICAL,
            dataset_name="MIMIC-III",
        )

    record_paths = records_file.read_text().strip().splitlines()
    if max_records:
        record_paths = record_paths[:max_records]

    for rel_path in record_paths:
        full_path = str(data_dir / rel_path)
        try:
            record = wfdb.rdrecord(full_path)
        except Exception:
            continue

        if record.sig_name is None:
            continue

        sig_names_upper = [s.upper() for s in record.sig_name]
        if signal_channel not in sig_names_upper:
            continue

        ch_idx = sig_names_upper.index(signal_channel)
        ppg = record.p_signal[:, ch_idx].astype(np.float32)
        fs = record.fs

        # Try to load numerics record for HR/RR/SpO2
        hr = rr = spo2 = None
        try:
            num = wfdb.rdrecord(full_path + "n")
            num_names = [s.upper() for s in num.sig_name]
            def _median_numeric(name):
                if name in num_names:
                    col = num.p_signal[:, num_names.index(name)]
                    col = col[~np.isnan(col)]
                    return float(np.median(col)) if len(col) > 0 else None
                return None
            hr = _median_numeric("HR")
            rr = _median_numeric("RESP")
            spo2 = _median_numeric("SpO2")
        except Exception:
            pass

        vitals = VitalSigns(heart_rate_bpm=hr, respiratory_rate_bpm=rr, spo2_pct=spo2)
        label = _classify_vitals(hr, rr, spo2)

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
                    patient_id=rel_path,
                    window_idx=idx,
                )
            )
            if label == AnomalyLabel.NORMAL:
                n_normal += 1
            else:
                n_anomaly += 1

    print(
        f"[MIMIC-III] Loaded {len(windows)} windows from {len(record_paths)} records "
        f"({n_normal} normal, {n_anomaly} anomaly)"
    )
    return WindowedDataset(
        windows=windows,
        domain=DataDomain.MEDICAL,
        dataset_name="MIMIC-III",
        n_normal=n_normal,
        n_anomaly=n_anomaly,
    )
