"""
MediSense Full Experiment — run this directly:
    python3 scripts/run_full_experiment.py

Runs all 9 steps and prints results to terminal.
Takes ~5-10 minutes on CPU.
"""

import sys, warnings, json, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import torch
from pathlib import Path

from src.data.bidmc_loader import load_bidmc
from src.data.capnobase_loader import load_capnobase
from src.data.nasa_loader import load_nasa_bearing, load_nasa_cmapss, load_skab
from src.data.base import AnomalyLabel
from src.signal.pipeline import PipelineConfig, process_dataset
from src.models.autoencoder import MediSenseAutoencoder, AutoencoderConfig
from src.models.trainer import MediSenseTrainer, TrainingConfig
from src.evaluation.metrics import (
    calibrate_threshold, evaluate, mcnemars_test,
    data_efficiency_experiment, save_results
)
from src.alerts.clinical_alerts import generate_alert

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
Path("experiments/models").mkdir(parents=True, exist_ok=True)
Path("experiments/results").mkdir(parents=True, exist_ok=True)

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Loading all datasets")
print("="*60)

bidmc   = load_bidmc('data/raw/bidmc')
capno   = load_capnobase('data/raw/capnobase')
bearing = load_nasa_bearing('data/raw/nasa_bearing')
cmapss  = load_nasa_cmapss('data/raw/nasa_cmapss')
skab    = load_skab('data/raw/skab', window_sec=30.0)

# ── STEP 2: SIGNAL PROCESSING ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Signal processing -> 17-dim feature extraction")
print("="*60)

med_cfg = PipelineConfig(fs=125.0, band_low_hz=0.5, band_high_hz=4.0)
ind_cfg = PipelineConfig(fs=20480.0, band_low_hz=100.0, band_high_hz=9000.0)

# Patient-level 60/20/20 split (prevents leakage)
all_medical = bidmc.windows + capno.windows
rng = np.random.default_rng(SEED)
patient_ids = list({w.patient_id for w in all_medical})
rng.shuffle(patient_ids)
n = len(patient_ids)
train_ids = set(patient_ids[:int(n*0.6)])
val_ids   = set(patient_ids[int(n*0.6):int(n*0.8)])
test_ids  = set(patient_ids[int(n*0.8):])

train_wins = [w for w in all_medical if w.patient_id in train_ids]
val_wins   = [w for w in all_medical if w.patient_id in val_ids]
test_wins  = [w for w in all_medical if w.patient_id in test_ids]

print(f"  Patient split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test patients")
print(f"  Window split:  {len(train_wins)} / {len(val_wins)} / {len(test_wins)}")
print("  *** Test set LOCKED — not used until Step 6 ***")

train_normal_wins = [w for w in train_wins if w.label == AnomalyLabel.NORMAL]
train_feats, (med_mean, med_std) = process_dataset(
    [w.signal for w in train_normal_wins], med_cfg, normalize=True)
val_feats, _ = process_dataset(
    [w.signal for w in val_wins], med_cfg, normalize=True, fit_stats=(med_mean, med_std))
test_feats, _ = process_dataset(
    [w.signal for w in test_wins], med_cfg, normalize=True, fit_stats=(med_mean, med_std))

val_labels  = np.array([w.label != AnomalyLabel.NORMAL for w in val_wins],  dtype=int)
test_labels = np.array([w.label != AnomalyLabel.NORMAL for w in test_wins], dtype=int)
val_normal_feats = val_feats[val_labels == 0]

print(f"  Features: {train_feats.shape[1]}-dim per window")
print(f"  Train normal windows: {len(train_feats)}")
print(f"  Test anomaly rate: {test_labels.mean()*100:.1f}%")

ind_normal = [w.signal for w in bearing.windows + cmapss.windows + skab.windows
              if w.label == AnomalyLabel.NORMAL]
rng.shuffle(ind_normal)
# Cap at 15,000 — statistically sufficient for pre-training; avoids excessive CPU time
IND_CAP = 15000
ind_normal = ind_normal[:IND_CAP]
split = int(0.8 * len(ind_normal))
ind_train_feats, (ind_mean, ind_std) = process_dataset(
    ind_normal[:split], ind_cfg, normalize=True, apply_filters=False)
ind_val_feats, _ = process_dataset(
    ind_normal[split:], ind_cfg, normalize=True,
    fit_stats=(ind_mean, ind_std), apply_filters=False)
print(f"  Industrial pre-train windows: {len(ind_train_feats):,} (capped at {IND_CAP} for efficiency)")

# ── STEP 3: INDUSTRIAL PRE-TRAINING ───────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Industrial pre-training — learning sensor anomaly structure")
print("="*60)

model_cfg = AutoencoderConfig(input_dim=17, hidden_dims=(64, 32, 16), latent_dim=8)

transfer_model = MediSenseAutoencoder(model_cfg)
trainer = MediSenseTrainer(transfer_model, TrainingConfig(
    epochs=50, batch_size=256, learning_rate=1e-3,
    early_stopping_patience=8, save_dir='experiments/models', run_name='medisense'))
trainer.pretrain(ind_train_feats, ind_val_feats)

# ── STEP 4: MEDICAL FINE-TUNING ───────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Medical fine-tuning (transfer learning)")
print("="*60)

trainer.finetune(train_feats, val_normal_feats,
                 pretrained_path='experiments/models/medisense_pretrain.pt')

# ── STEP 5: BASELINE ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Baseline — train from scratch on medical data only")
print("="*60)

baseline_model = MediSenseAutoencoder(model_cfg)
MediSenseTrainer(baseline_model, TrainingConfig(
    epochs=50, batch_size=256, learning_rate=1e-3,
    early_stopping_patience=8, save_dir='experiments/models', run_name='baseline'
)).train_baseline(train_feats, val_normal_feats)

# ── STEP 6: EVALUATION ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Final evaluation on LOCKED test set")
print("="*60)

t_thr = calibrate_threshold(transfer_model, val_normal_feats, percentile=90.0)
b_thr = calibrate_threshold(baseline_model,  val_normal_feats, percentile=90.0)

t_result = evaluate(transfer_model, test_feats, test_labels, t_thr,
                    run_name="Transfer Learning")
b_result = evaluate(baseline_model,  test_feats, test_labels, b_thr,
                    run_name="Baseline (medical only)")

# ── STEP 7: McNEMAR'S TEST ────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: McNemar's test")
print("="*60)

model_device = next(transfer_model.parameters()).device
t_tensor = torch.tensor(test_feats).to(model_device)
with torch.no_grad():
    t_preds = (transfer_model.anomaly_score(t_tensor).cpu().numpy() >= t_thr).astype(int)
    b_preds = (baseline_model.anomaly_score(t_tensor).cpu().numpy() >= b_thr).astype(int)
mcnemar = mcnemars_test(test_labels, t_preds, b_preds)

# ── STEP 8: DATA EFFICIENCY ───────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8: Data efficiency experiment")
print("="*60)

fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
t_by_frac, b_by_frac = {}, {}
for frac in fractions:
    n_sub = max(10, int(len(train_feats) * frac))
    subset = train_feats[:n_sub]
    print(f"\n  --- Fraction {frac:.0%} ({n_sub} windows) ---")

    m_t = MediSenseAutoencoder(model_cfg)
    MediSenseTrainer(m_t, TrainingConfig(
        epochs=30, batch_size=128, early_stopping_patience=5,
        save_dir='experiments/models', run_name=f'eff_t_{frac}'
    )).finetune(subset, val_normal_feats,
                pretrained_path='experiments/models/medisense_pretrain.pt')
    thr_t = calibrate_threshold(m_t, val_normal_feats, 90.0)
    t_by_frac[frac] = evaluate(m_t, test_feats, test_labels, thr_t,
                               run_name=f"Transfer {frac:.0%}")

    m_b = MediSenseAutoencoder(model_cfg)
    MediSenseTrainer(m_b, TrainingConfig(
        epochs=30, batch_size=128, early_stopping_patience=5,
        save_dir='experiments/models', run_name=f'eff_b_{frac}'
    )).train_baseline(subset, val_normal_feats)
    thr_b_frac = calibrate_threshold(m_b, val_normal_feats, 90.0)
    b_by_frac[frac] = evaluate(m_b, test_feats, test_labels, thr_b_frac,
                               run_name=f"Baseline {frac:.0%}")

efficiency = data_efficiency_experiment(t_by_frac, b_by_frac, target_accuracy=0.90)

# ── STEP 9: CLINICAL ALERT DEMO ───────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 9: Clinical alert generation on real patient windows")
print("="*60)

model_device = next(transfer_model.parameters()).device
with torch.no_grad():
    all_scores = transfer_model.anomaly_score(
        torch.tensor(test_feats).to(model_device)).cpu().numpy()

print()
shown = 0
for i, w in enumerate(test_wins):
    if shown >= 6:
        break
    alert = generate_alert(
        sensor_anomaly_score=all_scores[i],
        patient_anomaly_score=all_scores[i],
        sensor_threshold=t_thr * 1.5,
        patient_threshold=t_thr,
        vitals=w.ground_truth,
    )
    gt = w.ground_truth
    hr_s = f"{gt.heart_rate_bpm:.0f} BPM" if gt and gt.heart_rate_bpm else "N/A"
    rr_s = f"{gt.respiratory_rate_bpm:.0f} br/min" if gt and gt.respiratory_rate_bpm else "N/A"
    print(f"  [{w.patient_id}] HR={hr_s} RR={rr_s} | Ground truth: {w.label.name}")
    print(f"  Anomaly score: {all_scores[i]:.4f} (threshold: {t_thr:.4f})")
    print(alert.display())
    print()
    shown += 1

# ── SAVE ──────────────────────────────────────────────────────────────────────
results = {
    "transfer_learning": {
        "accuracy":             round(t_result.accuracy, 4),
        "sensitivity":          round(t_result.sensitivity, 4),
        "specificity":          round(t_result.specificity, 4),
        "f1_score":             round(t_result.f1_score, 4),
        "auroc":                round(t_result.auroc, 4),
        "false_negatives":      t_result.n_false_negatives,
        "false_positives":      t_result.n_false_positives,
        "inference_latency_ms": round(t_result.inference_latency_ms, 3),
        "classification_report":t_result.report,
    },
    "baseline": {
        "accuracy":    round(b_result.accuracy, 4),
        "sensitivity": round(b_result.sensitivity, 4),
        "specificity": round(b_result.specificity, 4),
        "auroc":       round(b_result.auroc, 4),
    },
    "improvement_pp":  round((t_result.accuracy - b_result.accuracy) * 100, 2),
    "mcnemars_test":   mcnemar,
    "data_efficiency": efficiency,
    "thesis_targets_met": {
        "patient_anomaly_accuracy_90pct": t_result.accuracy >= 0.90,
        "improvement_5pp":               (t_result.accuracy - b_result.accuracy) >= 0.05,
        "data_efficiency_50pct":         (efficiency.get("data_efficiency_gain") or 0) >= 50,
        "latency_under_100ms":           t_result.inference_latency_ms < 100,
        "mcnemars_significant":          mcnemar["significant"],
    },
}

with open("experiments/results/experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
print(f"\n  Transfer accuracy  : {t_result.accuracy:.4f}")
print(f"  Baseline accuracy  : {b_result.accuracy:.4f}")
print(f"  Improvement        : +{(t_result.accuracy-b_result.accuracy)*100:.2f} pp")
print(f"  AUROC (transfer)   : {t_result.auroc:.4f}")
print(f"  Latency            : {t_result.inference_latency_ms:.2f} ms/window")
print(f"  McNemar p          : {mcnemar['p_value']:.4f}")
print(f"  Data efficiency    : {efficiency.get('data_efficiency_gain')}% less data needed")
print()
print("  THESIS TARGETS:")
for k, v in results["thesis_targets_met"].items():
    status = "MET    " if v else "NOT MET"
    print(f"    [{status}] {k.replace('_',' ')}")
print(f"\n  Full results -> experiments/results/experiment_results.json")
