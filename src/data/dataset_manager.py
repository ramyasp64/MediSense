"""
DatasetManager — unified interface to load, merge, and split all datasets.

Usage:
    from src.data.dataset_manager import DatasetManager

    dm = DatasetManager("experiments/configs/base_config.yaml")
    industrial = dm.load_industrial()   # pre-training data
    medical    = dm.load_medical()      # fine-tuning / evaluation data
    splits     = dm.get_splits(medical) # train / val / test with patient-level split
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import yaml

from .base import DataDomain, Window, WindowedDataset
from .bidmc_loader import load_bidmc
from .capnobase_loader import load_capnobase
from .mimic_loader import load_mimic
from .nasa_loader import load_nasa_bearing, load_nasa_cmapss, load_skab


class DatasetManager:
    def __init__(self, config_path: Union[str, Path]):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.raw = Path(self.cfg["paths"]["raw_data"])
        self.seed = self.cfg.get("seed", 42)
        self.window_sec = self.cfg["signal"]["window_size_sec"]
        self.overlap = self.cfg["signal"]["overlap"]

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_industrial(self, max_records: Optional[int] = None) -> WindowedDataset:
        """Load all three industrial datasets merged into one WindowedDataset."""
        datasets = []

        bearing_dir = self.raw / "nasa_bearing"
        if bearing_dir.exists() and any(bearing_dir.iterdir()):
            datasets.append(
                load_nasa_bearing(bearing_dir, window_sec=0.05, overlap=self.overlap)
            )

        cmapss_dir = self.raw / "nasa_cmapss"
        if cmapss_dir.exists() and any(cmapss_dir.iterdir()):
            datasets.append(load_nasa_cmapss(cmapss_dir, window_cycles=30))

        skab_dir = self.raw / "skab"
        if skab_dir.exists() and any(skab_dir.iterdir()):
            datasets.append(
                load_skab(skab_dir, window_sec=self.window_sec, overlap=self.overlap)
            )

        return self._merge(datasets, "Industrial_Combined", DataDomain.INDUSTRIAL)

    def load_medical(
        self,
        include_mimic: bool = True,
        max_records: Optional[int] = None,
    ) -> WindowedDataset:
        """Load BIDMC + CapnoBase (+ MIMIC if available) merged."""
        datasets = []

        bidmc_dir = self.raw / "bidmc"
        if bidmc_dir.exists() and any(bidmc_dir.iterdir()):
            datasets.append(
                load_bidmc(bidmc_dir, self.window_sec, self.overlap, max_records)
            )

        capno_dir = self.raw / "capnobase"
        if capno_dir.exists() and any(capno_dir.iterdir()):
            datasets.append(
                load_capnobase(capno_dir, self.window_sec, self.overlap, max_records)
            )

        mimic_dir = self.raw / "mimic"
        if include_mimic and mimic_dir.exists() and any(mimic_dir.iterdir()):
            datasets.append(
                load_mimic(mimic_dir, self.window_sec, self.overlap, max_records)
            )

        if not datasets:
            raise FileNotFoundError(
                "No medical dataset directories found with data. "
                "Run: python scripts/download_data.py"
            )

        return self._merge(datasets, "Medical_Combined", DataDomain.MEDICAL)

    def get_splits(
        self, dataset: WindowedDataset
    ) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
        """
        Patient-level train/val/test split (60/20/20).
        Splitting by patient_id prevents data leakage across splits.
        """
        split_cfg = self.cfg["splits"]
        train_frac = split_cfg["train"]
        val_frac = split_cfg["val"]

        # Group windows by patient
        patient_ids = list({w.patient_id for w in dataset.windows if w.patient_id})
        rng = np.random.default_rng(self.seed)
        rng.shuffle(patient_ids)

        n = len(patient_ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_ids = set(patient_ids[:n_train])
        val_ids = set(patient_ids[n_train : n_train + n_val])
        test_ids = set(patient_ids[n_train + n_val :])

        def _subset(ids: set, name: str) -> WindowedDataset:
            wins = [w for w in dataset.windows if w.patient_id in ids]
            n_norm = sum(1 for w in wins if w.label.value == 0)
            return WindowedDataset(
                windows=wins,
                domain=dataset.domain,
                dataset_name=f"{dataset.dataset_name}_{name}",
                n_normal=n_norm,
                n_anomaly=len(wins) - n_norm,
            )

        train = _subset(train_ids, "train")
        val = _subset(val_ids, "val")
        test = _subset(test_ids, "test")

        print(
            f"[Split] Train: {len(train)} windows ({len(train_ids)} patients) | "
            f"Val: {len(val)} windows | Test: {len(test)} windows"
        )
        print("  *** Test set locked — do not use until final evaluation ***")
        return train, val, test

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge(
        self, datasets: list[WindowedDataset], name: str, domain: DataDomain
    ) -> WindowedDataset:
        all_windows: list[Window] = []
        n_normal = n_anomaly = 0
        for ds in datasets:
            all_windows.extend(ds.windows)
            n_normal += ds.n_normal
            n_anomaly += ds.n_anomaly
        merged = WindowedDataset(
            windows=all_windows,
            domain=domain,
            dataset_name=name,
            n_normal=n_normal,
            n_anomaly=n_anomaly,
        )
        print(
            f"[Merged: {name}] {len(merged)} total windows "
            f"({n_normal} normal, {n_anomaly} anomaly)"
        )
        return merged
