"""
NASA Industrial Sensor Dataset loaders.

Three datasets used for industrial pre-training:

1. NASA Bearing (IMS / FEMTO-ST)
   Source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
   Vibration sensors on bearings run to failure. Normal → degradation → failure.
   Files: 1st_test/, 2nd_test/, 3rd_test/ (space-separated CSVs, no headers)

2. NASA CMAPSS Turbofan Engine Degradation
   Source: same repository
   21 sensor readings per engine, run-to-failure cycles.
   Files: train_FD001.txt, train_FD002.txt, test_FD001.txt, RUL_FD001.txt

3. SKAB (Skoltech Anomaly Benchmark)
   Source: https://github.com/waico/SKAB
   Pump/valve sensor time-series with precise anomaly timestamps.
   Files: valve1/*.csv, valve2/*.csv, other/*.csv
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import (
    AnomalyLabel,
    DataDomain,
    Window,
    WindowedDataset,
    sliding_windows,
)

# NASA Bearing sampling rate (IMS dataset)
BEARING_FS = 20480  # Hz
# CMAPSS has no physical fs — treat each row as one timestep
CMAPSS_FS = 1.0


# ---------------------------------------------------------------------------
# NASA Bearing
# ---------------------------------------------------------------------------

def load_nasa_bearing(
    data_dir: Union[str, Path],
    window_sec: float = 0.05,   # 1024 samples at 20480 Hz ≈ 50ms
    overlap: float = 0.5,
    failure_fraction: float = 0.15,  # last 15% of run = failure/anomaly
) -> WindowedDataset:
    """
    Load NASA Bearing IMS dataset.

    Directory layout expected:
        data/raw/nasa_bearing/
            1st_test/  (2003-10-22_12_06_24)
            2nd_test/  (2004-02-12_10_32_39)
            3rd_test/  (2004-04-05_08_29_20)

    Each file is a space-separated CSV with 4 columns (one per bearing channel).
    Files are named by timestamp and sorted chronologically.

    Args:
        data_dir: Path to data/raw/nasa_bearing
        window_sec: Window size in seconds.
        overlap: Window overlap fraction.
        failure_fraction: Fraction of the END of each run labelled as anomaly.
    """
    data_dir = Path(data_dir)
    windows: list[Window] = []
    n_normal = n_anomaly = 0

    for test_dir in sorted(data_dir.glob("*test*")):
        # 3rd_test stores files inside a nested 'txt/' subdirectory
        candidate = test_dir / "txt"
        search_dir = candidate if candidate.is_dir() else test_dir
        files = sorted(f for f in search_dir.glob("*") if f.is_file())
        if not files:
            continue

        n_files = len(files)
        failure_start_idx = int(n_files * (1 - failure_fraction))

        for file_idx, fpath in enumerate(files):
            try:
                df = pd.read_csv(fpath, sep=r"\s+", header=None)
            except Exception:
                continue

            label = (
                AnomalyLabel.SENSOR_MALFUNCTION
                if file_idx >= failure_start_idx
                else AnomalyLabel.NORMAL
            )

            # Use channel 0 (bearing 1) — representative single channel
            signal = df.iloc[:, 0].values.astype(np.float32)

            for idx, win in enumerate(
                sliding_windows(signal, BEARING_FS, window_sec, overlap)
            ):
                windows.append(
                    Window(
                        signal=win,
                        fs=BEARING_FS,
                        domain=DataDomain.INDUSTRIAL,
                        label=label,
                        patient_id=f"{test_dir.name}_{fpath.name}",
                        window_idx=idx,
                        metadata={"source": "nasa_bearing", "test": test_dir.name},
                    )
                )
                if label == AnomalyLabel.NORMAL:
                    n_normal += 1
                else:
                    n_anomaly += 1

    print(
        f"[NASA Bearing] Loaded {len(windows)} windows "
        f"({n_normal} normal, {n_anomaly} anomaly)"
    )
    return WindowedDataset(
        windows=windows,
        domain=DataDomain.INDUSTRIAL,
        dataset_name="NASA_Bearing",
        n_normal=n_normal,
        n_anomaly=n_anomaly,
    )


# ---------------------------------------------------------------------------
# NASA CMAPSS Turbofan
# ---------------------------------------------------------------------------

CMAPSS_COLS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance in FD001 — drop them
CMAPSS_DROP_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}

# Remaining useful life threshold: ≤30 cycles = anomaly (imminent failure)
RUL_ANOMALY_THRESHOLD = 30


def load_nasa_cmapss(
    data_dir: Union[str, Path],
    subsets: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004"),
    window_cycles: int = 30,
    overlap: float = 0.5,
) -> WindowedDataset:
    """
    Load NASA CMAPSS turbofan engine degradation dataset.

    Files expected: data/raw/nasa_cmapss/train_FD001.txt, RUL_FD001.txt, etc.

    Args:
        data_dir: Path to data/raw/nasa_cmapss
        subsets: Which FD subsets to include.
        window_cycles: Number of engine cycles per window.
        overlap: Window overlap fraction.
    """
    data_dir = Path(data_dir)
    windows: list[Window] = []
    n_normal = n_anomaly = 0

    sensor_cols = [c for c in CMAPSS_COLS if c.startswith("s") and c not in CMAPSS_DROP_SENSORS]

    for subset in subsets:
        train_file = data_dir / f"train_{subset}.txt"
        rul_file = data_dir / f"RUL_{subset}.txt"

        if not train_file.exists():
            continue

        df = pd.read_csv(
            train_file, sep=r"\s+", header=None, names=CMAPSS_COLS
        )

        # Compute RUL per cycle per engine
        max_cycles = df.groupby("unit")["cycle"].max()
        df["rul"] = df.apply(
            lambda r: max_cycles[r["unit"]] - r["cycle"], axis=1
        )

        for unit_id, group in df.groupby("unit"):
            sensor_data = group[sensor_cols].values.astype(np.float32)
            rul_values = group["rul"].values

            step = max(1, int(window_cycles * (1 - overlap)))
            for start in range(0, len(sensor_data) - window_cycles + 1, step):
                window_data = sensor_data[start : start + window_cycles]
                window_rul = rul_values[start + window_cycles - 1]

                label = (
                    AnomalyLabel.SENSOR_MALFUNCTION
                    if window_rul <= RUL_ANOMALY_THRESHOLD
                    else AnomalyLabel.NORMAL
                )

                # Flatten multi-sensor window → 1D feature vector
                signal = window_data.flatten()

                windows.append(
                    Window(
                        signal=signal,
                        fs=CMAPSS_FS,
                        domain=DataDomain.INDUSTRIAL,
                        label=label,
                        patient_id=f"{subset}_unit{unit_id}",
                        metadata={
                            "source": "nasa_cmapss",
                            "subset": subset,
                            "rul": float(window_rul),
                        },
                    )
                )
                if label == AnomalyLabel.NORMAL:
                    n_normal += 1
                else:
                    n_anomaly += 1

    print(
        f"[NASA CMAPSS] Loaded {len(windows)} windows "
        f"({n_normal} normal, {n_anomaly} anomaly)"
    )
    return WindowedDataset(
        windows=windows,
        domain=DataDomain.INDUSTRIAL,
        dataset_name="NASA_CMAPSS",
        n_normal=n_normal,
        n_anomaly=n_anomaly,
    )


# ---------------------------------------------------------------------------
# SKAB (Skoltech Anomaly Benchmark)
# ---------------------------------------------------------------------------

def load_skab(
    data_dir: Union[str, Path],
    window_sec: float = 10.0,
    overlap: float = 0.5,
    fs: float = 1.0,  # 1 sample/second
) -> WindowedDataset:
    """
    Load SKAB pump/valve sensor dataset.

    Expected layout: data/raw/skab/valve1/*.csv, valve2/*.csv, other/*.csv
    Each CSV has columns: datetime, sensor readings..., anomaly (0/1), changepoint (0/1)

    Args:
        data_dir: Path to data/raw/skab
        window_sec: Window size in seconds.
        overlap: Window overlap fraction.
        fs: Sampling frequency (1 Hz for SKAB).
    """
    data_dir = Path(data_dir)
    windows: list[Window] = []
    n_normal = n_anomaly = 0

    csv_files = list(data_dir.rglob("*.csv"))
    # Exclude the all-normal allgood.csv if present
    csv_files = [f for f in csv_files if "allgood" not in f.name.lower()]

    for csv_path in sorted(csv_files):
        try:
            df = pd.read_csv(csv_path, sep=";", parse_dates=["datetime"], index_col="datetime")
        except Exception:
            try:
                df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            except Exception:
                continue

        if "anomaly" not in df.columns:
            continue

        anomaly_col = df["anomaly"].values
        sensor_cols = [c for c in df.columns if c not in ("anomaly", "changepoint")]
        if not sensor_cols:
            continue

        # Use first sensor channel for windowing (consistent with bearing approach)
        signal = df[sensor_cols[0]].values.astype(np.float32)
        anomaly_flags = anomaly_col.astype(int)

        window_samples = int(window_sec * fs)
        step = max(1, int(window_samples * (1 - overlap)))

        for start in range(0, len(signal) - window_samples + 1, step):
            win_signal = signal[start : start + window_samples]
            win_anomaly = anomaly_flags[start : start + window_samples]

            # If >10% of the window is anomalous, label the whole window anomalous
            label = (
                AnomalyLabel.SENSOR_MALFUNCTION
                if win_anomaly.mean() > 0.1
                else AnomalyLabel.NORMAL
            )

            windows.append(
                Window(
                    signal=win_signal,
                    fs=fs,
                    domain=DataDomain.INDUSTRIAL,
                    label=label,
                    patient_id=csv_path.stem,
                    metadata={"source": "skab", "file": str(csv_path.relative_to(data_dir))},
                )
            )
            if label == AnomalyLabel.NORMAL:
                n_normal += 1
            else:
                n_anomaly += 1

    print(
        f"[SKAB] Loaded {len(windows)} windows from {len(csv_files)} files "
        f"({n_normal} normal, {n_anomaly} anomaly)"
    )
    return WindowedDataset(
        windows=windows,
        domain=DataDomain.INDUSTRIAL,
        dataset_name="SKAB",
        n_normal=n_normal,
        n_anomaly=n_anomaly,
    )
