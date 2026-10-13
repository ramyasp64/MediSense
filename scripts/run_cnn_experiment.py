"""
MediSense CNN Experiment — full thesis pipeline with 1D-CNN autoencoder.

Run:
    python3 scripts/run_cnn_experiment.py

What changed from v1:
    - Raw PPG windows (1250 samples) replace 17-dim feature vectors
    - 1D-CNN autoencoder preserves temporal/waveform structure
    - Only NASA Bearing used for pre-training (genuine time-series signal)
    - All windows resampled to 1250 samples before model
"""

import sys, warnings, json, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import resample
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

from src.data.bidmc_loader import load_bidmc
from src.data.capnobase_loader import load_capnobase
from src.data.nasa_loader import load_nasa_bearing
from src.data.base import AnomalyLabel
from src.models.cnn_autoencoder import MediSenseCNN, CNNConfig, FIXED_LEN
from src.alerts.clinical_alerts import generate_alert
from src.evaluation.metrics import mcnemars_test, data_efficiency_experiment, save_results, EvaluationResult

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
Path("experiments/models").mkdir(parents=True, exist_ok=True)
Path("experiments/results").mkdir(parents=True, exist_ok=True)

# ── HELPERS ──────────────────────────────────────────────────────────────────

from scipy.signal import butter, filtfilt

def bandpass(sig: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    nyq = fs / 2.0
    lo_n, hi_n = lo / nyq, hi / nyq
    if lo_n <= 0 or hi_n >= 1 or lo_n >= hi_n or len(sig) < 30:
        return sig
    b, a = butter(3, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, sig).astype(np.float32)

def normalise(sig: np.ndarray) -> np.ndarray:
    """Min-max scale to [-1, 1] — better for CNN than z-score."""
    lo, hi = sig.min(), sig.max()
    rng = hi - lo
    if rng < 1e-8:
        return np.zeros_like(sig, dtype=np.float32)
    return (2.0 * (sig - lo) / rng - 1.0).astype(np.float32)

def to_fixed(sig: np.ndarray, fs: float = 125.0) -> np.ndarray:
    """Bandpass → resample to FIXED_LEN → min-max normalise."""
    sig = bandpass(sig.astype(np.float32), fs, lo=0.5, hi=min(4.0, fs/2 - 0.1))
    if len(sig) != FIXED_LEN:
        sig = resample(sig, FIXED_LEN)
    return normalise(sig)

def windows_to_tensor(windows) -> torch.Tensor:
    """List[Window] → (N, 1, FIXED_LEN) float32 tensor."""
    arr = np.stack([to_fixed(w.signal, fs=float(w.fs)) for w in windows])
    return torch.tensor(arr[:, None, :], dtype=torch.float32)

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

def weighted_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Peak-weighted MSE: errors near signal peaks penalised 3× more.
    Forces the model to learn waveform morphology, not just amplitude.
    """
    # Identify peaks: values in top 20% of signal range = peak region
    with torch.no_grad():
        sig_min = x.min(dim=2, keepdim=True).values
        sig_max = x.max(dim=2, keepdim=True).values
        thresh  = sig_min + 0.8 * (sig_max - sig_min)
        weights = torch.where(x >= thresh,
                              torch.full_like(x, 3.0),
                              torch.ones_like(x))
    return (weights * (x_hat - x) ** 2).mean()

def train_loop(model, train_t, val_t, epochs, batch_size, lr, patience, tag,
               warmup_epochs=10):
    """Training loop with LR warmup + cosine decay + peak-weighted loss."""
    device = next(model.parameters()).device
    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()),
               lr=lr, weight_decay=1e-5)

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_dl = DataLoader(TensorDataset(train_t), batch_size=batch_size,
                          shuffle=True, drop_last=True)
    val_dl   = DataLoader(TensorDataset(val_t),   batch_size=batch_size, shuffle=False)

    best_loss, best_state, wait = float('inf'), None, 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for (b,) in train_dl:
            b = b.to(device)
            opt.zero_grad()
            loss = weighted_mse(model(b)[0], b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tl += loss.item() * len(b)
        tl /= len(train_t)

        model.eval()
        with torch.no_grad():
            vl = sum(weighted_mse(model(b.to(device))[0], b.to(device)).item() * len(b)
                     for (b,) in val_dl) / len(val_t)
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{tag}] Epoch {epoch:3d}/{epochs} | train={tl:.4f} val={vl:.4f} lr={sched.get_last_lr()[0]:.5f}")
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{tag}] Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"  [{tag}] Best val={best_loss:.4f}  ({time.time()-t0:.1f}s)")
    return best_loss

def contrastive_finetune(model, train_normal_t, train_anomaly_t, val_normal_t,
                         epochs=80, batch_size=64, lr=5e-5, patience=15,
                         margin=10.0, tag="CONTRAST", warmup_epochs=5):
    """
    Fine-tune with contrastive loss:
      - Normal windows: reconstruction MSE (learn to reconstruct normal signals)
      - Both splits:    centre loss pulls normal latents together,
                        hinge loss pushes anomaly latents away from normal centre
    This shapes the latent space so Mahalanobis distance becomes discriminative.
    """
    device = next(model.parameters()).device
    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    norm_dl = DataLoader(TensorDataset(train_normal_t),  batch_size=batch_size, shuffle=True,  drop_last=True)
    anom_dl = DataLoader(TensorDataset(train_anomaly_t), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl  = DataLoader(TensorDataset(val_normal_t),    batch_size=batch_size, shuffle=False)

    best_loss, best_state, wait = float('inf'), None, 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        anom_iter = iter(anom_dl)
        for (x_n,) in norm_dl:
            x_n = x_n.to(device)
            try:
                (x_a,) = next(anom_iter)
            except StopIteration:
                anom_iter = iter(anom_dl)
                (x_a,) = next(anom_iter)
            x_a = x_a.to(device)

            opt.zero_grad()

            # 1. Reconstruction loss on normal windows
            x_hat, z_n = model(x_n)
            recon_loss = weighted_mse(x_hat, x_n)

            # 2. Centre loss: pull normal latents toward their mean
            mu_n = z_n.mean(dim=0, keepdim=True)               # (1, D)
            centre_loss = ((z_n - mu_n) ** 2).mean()

            # 3. Hinge loss: push anomaly latents away from normal centre
            _, z_a = model(x_a)
            dist_a = ((z_a - mu_n.detach()) ** 2).sum(dim=1).sqrt()   # (B,)
            hinge_loss = torch.clamp(margin - dist_a, min=0.0).mean()

            loss = recon_loss + 0.1 * centre_loss + 0.5 * hinge_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tl += loss.item() * len(x_n)
        tl /= len(train_normal_t)

        model.eval()
        with torch.no_grad():
            vl = sum(weighted_mse(model(b.to(device))[0], b.to(device)).item() * len(b)
                     for (b,) in val_dl) / len(val_normal_t)
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{tag}] Epoch {epoch:3d}/{epochs} | train={tl:.4f} val={vl:.4f} lr={sched.get_last_lr()[0]:.5f}")
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{tag}] Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"  [{tag}] Best val={best_loss:.4f}  ({time.time()-t0:.1f}s)")
    return best_loss

def calibrate(model, val_normal_t, percentile=90.0):
    device = next(model.parameters()).device
    with torch.no_grad():
        scores = model.anomaly_score(val_normal_t.to(device)).cpu().numpy()
    return float(np.percentile(scores, percentile))

def calibrate_maha(model, train_normal_t, val_t, val_labels):
    """Fit Mahalanobis scorer on train normals, pick threshold maximising F1 on val."""
    device = next(model.parameters()).device
    print("  [Mahalanobis] Fitting normal cluster in latent space...")
    model.fit_mahalanobis(train_normal_t.to(device))
    val_scores = model.mahalanobis_score(val_t.to(device)).numpy()
    # Use the 85th percentile of ALL val scores as threshold.
    # This targets ~15% flag rate — appropriate for a 5-10% anomaly prevalence.
    best_thr = float(np.percentile(val_scores, 85))
    preds = (val_scores >= best_thr).astype(int)
    tp = int(((preds == 1) & (val_labels == 1)).sum())
    fp = int(((preds == 1) & (val_labels == 0)).sum())
    fn = int(((preds == 0) & (val_labels == 1)).sum())
    rec = tp / (tp + fn + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    print(f"  [Mahalanobis] Val sensitivity={rec:.3f} F1={f1:.4f} at p85 threshold={best_thr:.4f}")
    return best_thr

def evaluate_model_maha(model, X_t, y_true, threshold, name=""):
    """Evaluate using Mahalanobis distance score (requires fit_mahalanobis called first)."""
    device = next(model.parameters()).device
    t0 = time.perf_counter()
    scores = model.mahalanobis_score(X_t.to(device)).numpy()
    latency_ms = (time.perf_counter() - t0) * 1000 / len(y_true)

    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    acc  = accuracy_score(y_true, preds)
    sens = tp / (tp + fn + 1e-10)
    spec = tn / (tn + fp + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    f1   = 2*prec*sens / (prec + sens + 1e-10)
    try:    auroc = roc_auc_score(y_true, scores)
    except: auroc = 0.5

    print(f"\n{'='*55}")
    if name: print(f"  Results: {name}")
    print(f"  Accuracy    : {acc:.4f}  (target ≥0.90)")
    print(f"  Sensitivity : {sens:.4f}  (catch emergencies)")
    print(f"  Specificity : {spec:.4f}  (avoid alert fatigue)")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  AUROC       : {auroc:.4f}")
    print(f"  False Neg   : {fn}  (missed emergencies)")
    print(f"  False Pos   : {fp}  (false alarms)")
    print(f"  Latency     : {latency_ms:.3f} ms/window  (target <100ms)")
    print(f"{'='*55}")

    return EvaluationResult(
        accuracy=float(acc), sensitivity=float(sens), specificity=float(spec),
        precision=float(prec), f1_score=float(f1), auroc=float(auroc), auprc=0.0,
        n_false_negatives=int(fn), n_false_positives=int(fp),
        n_true_positives=int(tp), n_true_negatives=int(tn),
        threshold=threshold, inference_latency_ms=latency_ms,
        report=classification_report(y_true, preds, target_names=["Normal","Anomaly"])
    ), scores

def evaluate_model(model, X_t, y_true, threshold, name=""):
    device = next(model.parameters()).device
    t0 = time.perf_counter()
    with torch.no_grad():
        scores = model.anomaly_score(X_t.to(device)).cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000 / len(y_true)

    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    acc  = accuracy_score(y_true, preds)
    sens = tp / (tp + fn + 1e-10)
    spec = tn / (tn + fp + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    f1   = 2*prec*sens / (prec + sens + 1e-10)
    try:    auroc = roc_auc_score(y_true, scores)
    except: auroc = 0.5

    print(f"\n{'='*55}")
    if name: print(f"  Results: {name}")
    print(f"  Accuracy    : {acc:.4f}  (target ≥0.90)")
    print(f"  Sensitivity : {sens:.4f}  (catch emergencies)")
    print(f"  Specificity : {spec:.4f}  (avoid alert fatigue)")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  AUROC       : {auroc:.4f}")
    print(f"  False Neg   : {fn}  (missed emergencies)")
    print(f"  False Pos   : {fp}  (false alarms)")
    print(f"  Latency     : {latency_ms:.3f} ms/window  (target <100ms)")
    print(f"{'='*55}")

    return EvaluationResult(
        accuracy=float(acc), sensitivity=float(sens), specificity=float(spec),
        precision=float(prec), f1_score=float(f1), auroc=float(auroc), auprc=0.0,
        n_false_negatives=int(fn), n_false_positives=int(fp),
        n_true_positives=int(tp), n_true_negatives=int(tn),
        threshold=threshold, inference_latency_ms=latency_ms,
        report=classification_report(y_true, preds, target_names=["Normal","Anomaly"])
    ), scores

# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Loading datasets")
print("="*60)

bidmc    = load_bidmc('data/raw/bidmc')
capno    = load_capnobase('data/raw/capnobase')
bearing  = load_nasa_bearing('data/raw/nasa_bearing')

# ── STEP 2: SPLIT ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Patient-level 60/20/20 split")
print("="*60)

all_medical = bidmc.windows + capno.windows
rng = np.random.default_rng(SEED)
pids = sorted({w.patient_id for w in all_medical})  # sort for determinism
rng.shuffle(pids)
n = len(pids)
train_ids = set(pids[:int(n*0.60)])
val_ids   = set(pids[int(n*0.60):int(n*0.80)])
test_ids  = set(pids[int(n*0.80):])

train_wins = [w for w in all_medical if w.patient_id in train_ids]
val_wins   = [w for w in all_medical if w.patient_id in val_ids]
test_wins  = [w for w in all_medical if w.patient_id in test_ids]

print(f"  Patients — Train:{len(train_ids)}  Val:{len(val_ids)}  Test:{len(test_ids)}")
print(f"  Windows  — Train:{len(train_wins)} Val:{len(val_wins)} Test:{len(test_wins)}")
print("  *** Test set LOCKED until Step 6 ***")

# ── STEP 3: CONVERT TO TENSORS ───────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Resample all windows → 1250 samples, z-score normalise")
print("="*60)

train_normal_wins = [w for w in train_wins if w.label == AnomalyLabel.NORMAL]
train_anomaly_wins = [w for w in train_wins if w.label != AnomalyLabel.NORMAL]
val_normal_wins   = [w for w in val_wins   if w.label == AnomalyLabel.NORMAL]

train_t        = windows_to_tensor(train_normal_wins)    # normal only (for autoencoder)
train_anomaly_t = windows_to_tensor(train_anomaly_wins)  # anomalies (for contrastive)
val_normal_t   = windows_to_tensor(val_normal_wins)
val_t          = windows_to_tensor(val_wins)
test_t         = windows_to_tensor(test_wins)

val_labels  = np.array([w.label != AnomalyLabel.NORMAL for w in val_wins],  dtype=int)
test_labels = np.array([w.label != AnomalyLabel.NORMAL for w in test_wins], dtype=int)

print(f"  Train normal:   {train_t.shape}")
print(f"  Train anomaly:  {train_anomaly_t.shape}  ← used for contrastive loss")
print(f"  Val:            {val_t.shape}  ({val_labels.mean()*100:.1f}% anomaly)")
print(f"  Test:           {test_t.shape} ({test_labels.mean()*100:.1f}% anomaly)")

# Industrial: NASA Bearing — genuine periodic vibration time-series
# Cap at 10,000 normal windows (more than enough for pre-training)
ind_normal_wins = [w for w in bearing.windows if w.label == AnomalyLabel.NORMAL]
rng.shuffle(ind_normal_wins)
ind_normal_wins = ind_normal_wins[:10000]
split = int(0.8 * len(ind_normal_wins))
ind_train_t = windows_to_tensor(ind_normal_wins[:split])
ind_val_t   = windows_to_tensor(ind_normal_wins[split:])
print(f"  Industrial (NASA Bearing): {ind_train_t.shape[0]} train / {ind_val_t.shape[0]} val")

device = get_device()
print(f"\n  Device: {device}")

# ── STEP 4: INDUSTRIAL PRE-TRAINING ───────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Industrial pre-training on NASA Bearing vibration data")
print("="*60)

cfg = CNNConfig(latent_dim=256, dropout=0.15)
transfer_model = MediSenseCNN(cfg).to(device)

_pretrain_path = Path("experiments/models/cnn_pretrain.pt")
if _pretrain_path.exists():
    print("  [SKIP] cnn_pretrain.pt already exists — loading saved weights.")
    transfer_model.load_state_dict(torch.load(_pretrain_path, map_location=device, weights_only=False)["state_dict"])
else:
    train_loop(transfer_model,
               ind_train_t, ind_val_t,
               epochs=120, batch_size=128, lr=3e-4, patience=15,
               tag="PRETRAIN", warmup_epochs=10)
    transfer_model.save("experiments/models/cnn_pretrain.pt")

# ── STEP 5: MEDICAL FINE-TUNING ───────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Medical fine-tuning (transfer learning)")
print("="*60)

# Contrastive fine-tuning — shapes latent space for Mahalanobis discrimination
# Always retrain (contrastive loss requires anomaly labels, old checkpoint didn't use them)
transfer_model.freeze_encoder_layers(n_conv_blocks=2)
n_frozen = sum(1 for p in transfer_model.parameters() if not p.requires_grad)
n_total  = sum(1 for p in transfer_model.parameters())
print(f"  Frozen encoder params: {n_frozen}/{n_total}")
print(f"  Training normal: {len(train_normal_wins)} | anomaly: {len(train_anomaly_wins)} windows")
contrastive_finetune(transfer_model,
                     train_t, train_anomaly_t, val_normal_t,
                     epochs=80, batch_size=64, lr=5e-5, patience=15,
                     margin=10.0, tag="CONTRAST-T", warmup_epochs=5)
transfer_model.unfreeze_all()
transfer_model.save("experiments/models/cnn_finetune.pt")

# ── STEP 6: BASELINE ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Baseline — train from scratch on medical data only (contrastive)")
print("="*60)

baseline_model = MediSenseCNN(cfg).to(device)
# Train autoencoder first, then contrastive fine-tune
train_loop(baseline_model, train_t, val_normal_t,
           epochs=60, batch_size=64, lr=3e-4, patience=15,
           tag="BASE-PRETRAIN", warmup_epochs=10)
contrastive_finetune(baseline_model,
                     train_t, train_anomaly_t, val_normal_t,
                     epochs=60, batch_size=64, lr=5e-5, patience=15,
                     margin=10.0, tag="CONTRAST-B", warmup_epochs=5)
baseline_model.save("experiments/models/cnn_baseline.pt")

# ── STEP 7: FINAL EVALUATION ON LOCKED TEST SET ───────────────────────────────
print("\n" + "="*60)
print("STEP 7: Final evaluation — LOCKED test set (first use)")
print("="*60)

# Use Mahalanobis distance in latent space — far superior to raw reconstruction error
t_thr = calibrate_maha(transfer_model, train_t, val_t, val_labels)
b_thr = calibrate_maha(baseline_model,  train_t, val_t, val_labels)

t_result, t_scores = evaluate_model_maha(
    transfer_model, test_t, test_labels, t_thr,
    name="Transfer Learning (NASA Bearing → Medical PPG)")
b_result, b_scores = evaluate_model_maha(
    baseline_model, test_t, test_labels, b_thr,
    name="Baseline (medical-only training)")

# ── STEP 8: McNEMAR'S TEST ────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8: McNemar's test — statistical significance")
print("="*60)

t_preds = (t_scores >= t_thr).astype(int)
b_preds = (b_scores >= b_thr).astype(int)
mcnemar = mcnemars_test(test_labels, t_preds, b_preds)

# ── STEP 9: DATA EFFICIENCY ───────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 9: Data efficiency — accuracy vs. training fraction")
print("="*60)

fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
t_by_frac, b_by_frac = {}, {}

for frac in fractions:
    n_sub = max(16, int(len(train_t) * frac))
    subset_t = train_t[:n_sub]
    print(f"\n  Fraction {frac:.0%} ({n_sub} windows)")

    # Transfer
    m_t = MediSenseCNN(cfg).to(device)
    m_t.load_state_dict(
        torch.load("experiments/models/cnn_pretrain.pt",
                   weights_only=False)["state_dict"])
    m_t.freeze_encoder_layers(2)
    train_loop(m_t, subset_t, val_normal_t,
               epochs=80, batch_size=64, lr=1e-4, patience=15,
               tag=f"T-{frac:.0%}", warmup_epochs=5)
    m_t.unfreeze_all()
    thr_t = calibrate_maha(m_t, subset_t, val_t, val_labels)
    t_by_frac[frac], _ = evaluate_model_maha(m_t, test_t, test_labels, thr_t,
                                             name=f"Transfer {frac:.0%}")

    # Baseline
    m_b = MediSenseCNN(cfg).to(device)
    train_loop(m_b, subset_t, val_normal_t,
               epochs=80, batch_size=64, lr=3e-4, patience=15,
               tag=f"B-{frac:.0%}", warmup_epochs=10)
    thr_b = calibrate_maha(m_b, subset_t, val_t, val_labels)
    b_by_frac[frac], _ = evaluate_model_maha(m_b, test_t, test_labels, thr_b,
                                             name=f"Baseline {frac:.0%}")

efficiency = data_efficiency_experiment(t_by_frac, b_by_frac, target_accuracy=0.90)

# ── STEP 10: CLINICAL ALERT DEMO ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 10: Clinical alert generation on real patient windows")
print("="*60)

device_t = next(transfer_model.parameters()).device
all_scores = transfer_model.mahalanobis_score(test_t.to(device_t)).numpy()

print()
shown = 0
for i, w in enumerate(test_wins):
    if shown >= 6: break
    alert = generate_alert(
        sensor_anomaly_score=all_scores[i],
        patient_anomaly_score=all_scores[i],
        sensor_threshold=t_thr * 1.5,
        patient_threshold=t_thr,
        vitals=w.ground_truth,
    )
    gt = w.ground_truth
    hr_s = f"{gt.heart_rate_bpm:.0f} BPM"   if gt and gt.heart_rate_bpm   else "N/A"
    rr_s = f"{gt.respiratory_rate_bpm:.0f}/min" if gt and gt.respiratory_rate_bpm else "N/A"
    print(f"  [{w.patient_id}] HR={hr_s} RR={rr_s} | Truth:{w.label.name} | Score:{all_scores[i]:.4f}")
    print(f"  {alert.display()}\n")
    shown += 1

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
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
        "classification_report": t_result.report,
    },
    "baseline": {
        "accuracy":    round(b_result.accuracy, 4),
        "sensitivity": round(b_result.sensitivity, 4),
        "specificity": round(b_result.specificity, 4),
        "f1_score":    round(b_result.f1_score, 4),
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

with open("experiments/results/cnn_experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("EXPERIMENT COMPLETE — CNN RESULTS")
print("="*60)
print(f"\n  Transfer accuracy  : {t_result.accuracy:.4f}")
print(f"  Baseline accuracy  : {b_result.accuracy:.4f}")
print(f"  Improvement        : +{(t_result.accuracy - b_result.accuracy)*100:.2f} pp")
print(f"  Transfer AUROC     : {t_result.auroc:.4f}")
print(f"  Sensitivity        : {t_result.sensitivity:.4f}")
print(f"  Latency            : {t_result.inference_latency_ms:.3f} ms/window")
print(f"  McNemar p          : {mcnemar['p_value']:.4f}")
eff_gain = efficiency.get('data_efficiency_gain')
print(f"  Data efficiency    : {eff_gain}% less data needed")
print()
print("  THESIS TARGETS:")
for k, v in results["thesis_targets_met"].items():
    status = "MET    " if v else "NOT MET"
    print(f"    [{status}] {k.replace('_',' ')}")
print(f"\n  Results → experiments/results/cnn_experiment_results.json")
