"""
PhysioNet BIDMC PPG and Respiration Dataset loader.

Dataset: https://physionet.org/content/bidmc/1.0.0/
53 ICU patients, 8-minute segments sampled at 125 Hz.

Channels in each record:
    0: RESP   — impedance pneumography (reference for RR)
    1: PLETH  — photoplethysmogram / PPG (our main signal for HR)
    2: V      — ECG lead V
    3: AVR    — ECG lead AVR
    4: II     — ECG lead II (reference for HR via R-peak detection)

Ground truth:
    HR  — derived from ECG (lead II) R-peak intervals  [Charlton et al. 2018]
    RR  — derived from RESP impedance pneumography peaks

This matches the benchmark protocol in the BIDMC dataset paper.

Download:
    python3 -c "import wfdb; wfdb.dl_database('bidmc', 'data/raw/bidmc')"
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

from .base import (
    AnomalyLabel,
    DataDomain,
    VitalSigns,
    Window,
    WindowedDataset,
    sliding_windows,
)

# Clinical alert thresholds (WHO / standard ICU practice)
HR_LOW,  HR_HIGH  = 40, 140   # BPM
RR_LOW,  RR_HIGH  = 6,  35    # breaths/min

FS = 125.0   # BIDMC sampling rate (Hz)

# Channel indices
CH_RESP  = 0
CH_PLETH = 1
CH_ECG   = 4   # Lead II — best for R-peak detection


def _bandpass(sig: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def _compute_hr_from_ecg(ecg: np.ndarray, fs: float) -> Optional[float]:
    """
    Estimate heart rate from ECG lead II via R-peak detection.
    Returns median HR in BPM, or None if fewer than 2 peaks found.
    """
    # Bandpass 0.5–40 Hz to isolate QRS complex
    filtered = _bandpass(ecg, fs, 0.5, 40.0)
    # Peaks: min distance 0.3s (200 BPM max), prominence ≥ 0.3× range
    sig_range = filtered.max() - filtered.min() + 1e-10
    peaks, _ = find_peaks(
        filtered,
        distance=int(0.3 * fs),
        prominence=0.3 * sig_range,
    )
    if len(peaks) < 2:
        return None
    rr_intervals_sec = np.diff(peaks) / fs
    # Filter physiologically impossible intervals (< 0.3s or > 3s)
    rr_intervals_sec = rr_intervals_sec[
        (rr_intervals_sec >= 0.3) & (rr_intervals_sec <= 3.0)
    ]
    if len(rr_intervals_sec) == 0:
        return None
    return float(60.0 / np.median(rr_intervals_sec))


def _compute_rr_from_resp(resp: np.ndarray, fs: float) -> Optional[float]:
    """
    Estimate respiratory rate from impedance pneumography signal.
    Returns median RR in breaths/min, or None if fewer than 2 peaks found.
    """
    # Bandpass 0.1–0.5 Hz to isolate breathing cycles
    filtered = _bandpass(resp, fs, 0.1, 0.5)
    sig_range = filtered.max() - filtered.min() + 1e-10
    peaks, _ = find_peaks(
        filtered,
        distance=int(1.5 * fs),   # min 1.5s between breaths (40 breaths/min max)
        prominence=0.2 * sig_range,
    )
    if len(peaks) < 2:
        return None
    breath_intervals_sec = np.diff(peaks) / fs
    # Filter physiologically impossible intervals
    breath_intervals_sec = breath_intervals_sec[
        (breath_intervals_sec >= 1.5) & (breath_intervals_sec <= 20.0)
    ]
    if len(breath_intervals_sec) == 0:
        return None
    return float(60.0 / np.median(breath_intervals_sec))


def _classify_vitals(hr: Optional[float], rr: Optional[float]) -> AnomalyLabel:
    if hr is not None and not (HR_LOW <= hr <= HR_HIGH):
        return AnomalyLabel.PATIENT_ANOMALY
    if rr is not None and not (RR_LOW <= rr <= RR_HIGH):
        return AnomalyLabel.PATIENT_ANOMALY
    return AnomalyLabel.NORMAL


def load_bidmc(
    data_dir: Union[str, Path],
    window_sec: float = 10.0,
    overlap: float = 0.5,
    max_records: Optional[int] = None,
) -> WindowedDataset:
    """
    Load all BIDMC records and return a WindowedDataset.

    Ground truth HR is computed from ECG lead II R-peak intervals.
    Ground truth RR is computed from RESP impedance pneumography peaks.
    Both match the Charlton et al. 2018 BIDMC benchmark protocol.

    Args:
        data_dir: Path to data/raw/bidmc (contains bidmc01.dat, bidmc01.hea, etc.)
        window_sec: Window length in seconds (default 10s).
        overlap: Fractional overlap between consecutive windows (default 0.5).
        max_records: Cap number of records (useful for fast dev runs).

    Returns:
        WindowedDataset with domain=MEDICAL.
    """
    data_dir = Path(data_dir)
    # Find all records — exclude any with 'n' suffix (numerics, not present here)
    record_ids = sorted(
        p.stem for p in data_dir.glob("bidmc*.hea")
        if not p.stem.endswith("n")
    )

    if max_records:
        record_ids = record_ids[:max_records]

    windows: list[Window] = []
    n_normal = n_anomaly = 0
    n_skipped = 0

    for rec_id in record_ids:
        try:
            record = wfdb.rdrecord(str(data_dir / rec_id))
        except Exception as e:
            n_skipped += 1
            continue

        if record.p_signal is None or record.p_signal.shape[1] < 5:
            n_skipped += 1
            continue

        ppg  = record.p_signal[:, CH_PLETH].astype(np.float32)
        ecg  = record.p_signal[:, CH_ECG].astype(np.float32)
        resp = record.p_signal[:, CH_RESP].astype(np.float32)

        # Ground truth from reference signals
        hr = _compute_hr_from_ecg(ecg, FS)
        rr = _compute_rr_from_resp(resp, FS)
        vitals = VitalSigns(heart_rate_bpm=hr, respiratory_rate_bpm=rr)
        label  = _classify_vitals(hr, rr)

        for idx, win_signal in enumerate(
            sliding_windows(ppg, FS, window_sec, overlap)
        ):
            windows.append(
                Window(
                    signal=win_signal,
                    fs=FS,
                    domain=DataDomain.MEDICAL,
                    label=label,
                    ground_truth=vitals,
                    patient_id=rec_id,
                    window_idx=idx,
                )
            )
            if label == AnomalyLabel.NORMAL:
                n_normal += 1
            else:
                n_anomaly += 1

    print(
        f"[BIDMC] Loaded {len(windows)} windows from "
        f"{len(record_ids) - n_skipped}/{len(record_ids)} records "
        f"({n_normal} normal, {n_anomaly} anomaly)"
    )
    return WindowedDataset(
        windows=windows,
        domain=DataDomain.MEDICAL,
        dataset_name="BIDMC",
        n_normal=n_normal,
        n_anomaly=n_anomaly,
    )
