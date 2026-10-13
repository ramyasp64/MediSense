"""
Master experiment runner — runs the full MediSense thesis experiment.

Experiment sequence:
    1. Load industrial datasets → pre-train autoencoder
    2. Load medical datasets → fine-tune autoencoder (transfer learning)
    3. Train baseline model from scratch on medical data only
    4. Evaluate both on locked test set
    5. Run McNemar's test for statistical significance
    6. Run data efficiency experiment across training fractions
    7. Save all results as JSON for thesis tables/figures

Usage:
    python scripts/run_experiment.py --config experiments/configs/base_config.yaml
    python scripts/run_experiment.py --config experiments/configs/base_config.yaml --quick
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_manager import DatasetManager
from src.models.autoencoder import MediSenseAutoencoder, AutoencoderConfig
from src.models.trainer import MediSenseTrainer, TrainingConfig
from src.signal.pipeline import PipelineConfig, process_dataset
from src.evaluation.metrics import (
    calibrate_threshold,
    evaluate,
    mcnemars_test,
    data_efficiency_experiment,
    save_results,
)


def build_pipeline_cfg(cfg: dict) -> PipelineConfig:
    s = cfg["signal"]
    return PipelineConfig(
        fs=s["fs_medical"],
        band_low_hz=s["bandpass"]["hr_low"],
        band_high_hz=s["bandpass"]["hr_high"],
        kalman_process_noise=s["kalman"]["process_noise"],
        kalman_measurement_noise=s["kalman"]["measurement_noise"],
    )


def build_model(cfg: dict) -> MediSenseAutoencoder:
    m = cfg["model"]
    ac = AutoencoderConfig(
        input_dim=m["input_dim"],
        hidden_dims=tuple(m["hidden_dims"]),
        latent_dim=m["latent_dim"],
        dropout=m["dropout"],
        activation=m["activation"],
    )
    return MediSenseAutoencoder(ac)


def run(config_path: str, quick: bool = False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["seed"])

    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg["paths"]["models"] if "models" in cfg["paths"] else "experiments/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    max_records = 3 if quick else None
    pipe_cfg = build_pipeline_cfg(cfg)
    dm = DatasetManager(config_path)

    # ------------------------------------------------------------------
    # Step 1: Load & process industrial data
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Loading industrial datasets for pre-training")
    print("="*60)
    industrial = dm.load_industrial(max_records=max_records)
    # Pre-training: use only normal windows
    ind_normal = [w.signal for w in industrial.windows if w.label.value == 0]
    if not ind_normal:
        print("WARNING: No industrial data found. Run: python scripts/download_data.py --nasa")
        return

    # 80/20 split for industrial (no patient-level concern here)
    rng = np.random.default_rng(cfg["seed"])
    ind_normal_arr = list(ind_normal)
    rng.shuffle(ind_normal_arr)
    split = int(0.8 * len(ind_normal_arr))
    ind_train_raw, ind_val_raw = ind_normal_arr[:split], ind_normal_arr[split:]

    # Industrial feature extraction — apply_filters=False for short/multi-sensor windows.
    # The filters are PPG-specific; industrial signals use statistical/frequency features only.
    ind_pipe = PipelineConfig(fs=20480.0, band_low_hz=100.0, band_high_hz=9000.0)
    ind_train_feats, (ind_mean, ind_std) = process_dataset(
        ind_train_raw, ind_pipe, normalize=True, apply_filters=False
    )
    ind_val_feats, _ = process_dataset(
        ind_val_raw, ind_pipe, normalize=True,
        fit_stats=(ind_mean, ind_std), apply_filters=False
    )

    # ------------------------------------------------------------------
    # Step 2: Load & process medical data
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Loading medical datasets")
    print("="*60)
    medical = dm.load_medical(max_records=max_records)
    train_set, val_set, test_set = dm.get_splits(medical)

    def featurise(windows, fit_stats=None):
        raw = [w.signal for w in windows]
        labels = np.array([w.label.value > 0 for w in windows], dtype=int)
        feats, stats = process_dataset(
            raw, pipe_cfg, normalize=True, fit_stats=fit_stats
        )
        return feats, labels, stats

    # Training: fit stats on normal training windows only (no leakage)
    train_normal_windows = [w for w in train_set.windows if w.label.value == 0]
    train_feats, train_labels, (med_mean, med_std) = featurise(train_normal_windows)
    val_feats, val_labels, _ = featurise(val_set.windows, (med_mean, med_std))
    test_feats, test_labels, _ = featurise(test_set.windows, (med_mean, med_std))

    val_normal_feats = val_feats[val_labels == 0]

    t_cfg = cfg["training"]
    train_cfg = TrainingConfig(
        epochs=10 if quick else t_cfg["epochs_pretrain"],
        batch_size=t_cfg["batch_size"],
        learning_rate=t_cfg["learning_rate"],
        early_stopping_patience=3 if quick else t_cfg["early_stopping_patience"],
        freeze_encoder_layers=t_cfg["freeze_layers"],
        save_dir=str(models_dir),
        run_name="medisense",
    )

    # ------------------------------------------------------------------
    # Step 3: Industrial pre-training
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Industrial pre-training")
    print("="*60)
    transfer_model = build_model(cfg)
    trainer = MediSenseTrainer(transfer_model, train_cfg)
    pretrain_log = trainer.pretrain(ind_train_feats, ind_val_feats)

    # ------------------------------------------------------------------
    # Step 4: Medical fine-tuning (transfer learning)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: Medical fine-tuning (transfer learning)")
    print("="*60)
    train_cfg.epochs = 5 if quick else t_cfg["epochs_finetune"]
    finetune_log = trainer.finetune(
        train_feats,
        val_normal_feats,
        pretrained_path=models_dir / "medisense_pretrain.pt",
    )

    # ------------------------------------------------------------------
    # Step 5: Baseline (medical only, no pre-training)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 5: Baseline — medical-only training (no transfer learning)")
    print("="*60)
    baseline_model = build_model(cfg)
    baseline_trainer = MediSenseTrainer(baseline_model, train_cfg)
    baseline_log = baseline_trainer.train_baseline(train_feats, val_normal_feats)

    # ------------------------------------------------------------------
    # Step 6: Evaluate on LOCKED test set
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 6: Final evaluation on LOCKED test set")
    print("="*60)
    det = cfg["detection"]

    transfer_threshold = calibrate_threshold(
        transfer_model, val_normal_feats,
        percentile=det["patient_anomaly"]["threshold_percentile"]
    )
    baseline_threshold = calibrate_threshold(
        baseline_model, val_normal_feats,
        percentile=det["patient_anomaly"]["threshold_percentile"]
    )

    transfer_result = evaluate(
        transfer_model, test_feats, test_labels, transfer_threshold,
        run_name="Transfer Learning (industrial pre-train + medical fine-tune)"
    )
    baseline_result = evaluate(
        baseline_model, test_feats, test_labels, baseline_threshold,
        run_name="Baseline (medical-only training)"
    )

    # ------------------------------------------------------------------
    # Step 7: Statistical significance
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 7: McNemar's test — statistical significance")
    print("="*60)
    import torch
    t_tensor = torch.tensor(test_feats)
    transfer_model.eval()
    baseline_model.eval()
    with torch.no_grad():
        t_preds = (transfer_model.anomaly_score(t_tensor).numpy() >= transfer_threshold).astype(int)
        b_preds = (baseline_model.anomaly_score(t_tensor).numpy() >= baseline_threshold).astype(int)

    mcnemar = mcnemars_test(test_labels, t_preds, b_preds)

    # ------------------------------------------------------------------
    # Step 8: Data efficiency experiment
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 8: Data efficiency experiment")
    print("="*60)
    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    transfer_by_fraction = {}
    baseline_by_fraction = {}

    for frac in fractions:
        n = max(1, int(len(train_feats) * frac))
        subset = train_feats[:n]

        # Transfer at this fraction
        m_t = build_model(cfg)
        tr_t = MediSenseTrainer(m_t, TrainingConfig(
            epochs=5 if quick else 30,
            early_stopping_patience=3,
            save_dir=str(models_dir),
            run_name=f"efficiency_transfer_{frac}",
        ))
        tr_t.finetune(subset, val_normal_feats,
                      pretrained_path=models_dir / "medisense_pretrain.pt")
        thr_t = calibrate_threshold(m_t, val_normal_feats,
                                    det["patient_anomaly"]["threshold_percentile"])
        transfer_by_fraction[frac] = evaluate(
            m_t, test_feats, test_labels, thr_t, run_name=f"Transfer frac={frac}"
        )

        # Baseline at this fraction
        m_b = build_model(cfg)
        tr_b = MediSenseTrainer(m_b, TrainingConfig(
            epochs=5 if quick else 30,
            early_stopping_patience=3,
            save_dir=str(models_dir),
            run_name=f"efficiency_baseline_{frac}",
        ))
        tr_b.train_baseline(subset, val_normal_feats)
        thr_b = calibrate_threshold(m_b, val_normal_feats,
                                    det["patient_anomaly"]["threshold_percentile"])
        baseline_by_fraction[frac] = evaluate(
            m_b, test_feats, test_labels, thr_b, run_name=f"Baseline frac={frac}"
        )

    efficiency = data_efficiency_experiment(
        transfer_by_fraction,
        baseline_by_fraction,
        target_accuracy=cfg["targets"]["patient_anomaly_accuracy"],
    )

    # ------------------------------------------------------------------
    # Save all results
    # ------------------------------------------------------------------
    all_results = {
        "transfer_learning": {
            "accuracy": transfer_result.accuracy,
            "sensitivity": transfer_result.sensitivity,
            "specificity": transfer_result.specificity,
            "auroc": transfer_result.auroc,
            "false_negatives": transfer_result.n_false_negatives,
            "inference_latency_ms": transfer_result.inference_latency_ms,
        },
        "baseline": {
            "accuracy": baseline_result.accuracy,
            "sensitivity": baseline_result.sensitivity,
            "auroc": baseline_result.auroc,
        },
        "improvement_pp": round(
            (transfer_result.accuracy - baseline_result.accuracy) * 100, 2
        ),
        "mcnemars_test": mcnemar,
        "data_efficiency": efficiency,
        "thesis_targets_met": {
            "patient_anomaly_accuracy_90pct": transfer_result.accuracy >= 0.90,
            "improvement_5pp": (transfer_result.accuracy - baseline_result.accuracy) >= 0.05,
            "data_efficiency_50pct": (
                efficiency["data_efficiency_gain"] is not None
                and efficiency["data_efficiency_gain"] >= 50
            ),
            "latency_under_100ms": transfer_result.inference_latency_ms < 100,
            "mcnemars_significant": mcnemar["significant"],
        },
    }

    save_results(all_results, results_dir / "experiment_results.json")
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    for target, met in all_results["thesis_targets_met"].items():
        status = "✓ MET" if met else "✗ NOT MET"
        print(f"  {status}: {target.replace('_', ' ')}")
    print(f"\nFull results: {results_dir / 'experiment_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/base_config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 records, 10 epochs — for testing the pipeline")
    args = parser.parse_args()
    run(args.config, quick=args.quick)
