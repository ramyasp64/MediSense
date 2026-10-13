"""
Self-Supervised Baseline — MediSense
=====================================
Prof. von Gladiß baseline #2: Self-supervised pretraining on the medical
data itself (no external source, no labels).

Method: Masked Time-Series Autoencoder (MTSA)
  - Randomly mask 40% of each window with zeros
  - Train the CNN autoencoder to reconstruct the original (unmasked) signal
  - This is self-supervised: no anomaly labels used during pretraining
  - Fine-tune with contrastive loss (same as bearing experiment)
  - Score with Mahalanobis distance in latent space

If this AUROC < bearing AUROC → bearing pretraining is better than
  the best you can do within the medical domain alone (without labels).
If this AUROC ≈ bearing AUROC → external industrial data adds little
  beyond what self-supervision on medical data provides.

Run:
  /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3 scripts/run_selfsupervised_baseline.py
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from scipy.signal import resample as sp_resample

device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
print(f"\n  Device: {device}")

from src.models.cnn_autoencoder import MediSenseCNN, CNNConfig
from src.data.bidmc_loader import load_bidmc
from src.data.capnobase_loader import load_capnobase
from src.data.base import AnomalyLabel

FIXED_LEN    = 1250
SSL_PRETRAIN = os.path.join(ROOT, "experiments", "models", "cnn_ssl_pretrain.pt")
SSL_FINETUNE = os.path.join(ROOT, "experiments", "models", "cnn_ssl_finetune.pt")
RESULTS_PATH = os.path.join(ROOT, "experiments", "results", "ssl_baseline_results.json")
os.makedirs(os.path.join(ROOT, "experiments", "models"),  exist_ok=True)
os.makedirs(os.path.join(ROOT, "experiments", "results"), exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_fixed(sig, fs=125.0):
    sig = sig.astype(np.float32)
    if len(sig) != FIXED_LEN:
        sig = sp_resample(sig, FIXED_LEN)
    std = sig.std() + 1e-9
    return (sig - sig.mean()) / std

def windows_to_tensor(wins):
    arr = np.stack([_to_fixed(w.signal, float(w.fs)) for w in wins])
    return torch.tensor(arr[:, None, :], dtype=torch.float32)

def apply_mask(x, mask_ratio=0.40):
    """Zero-out mask_ratio fraction of timesteps (per sample, random positions)."""
    B, C, L = x.shape
    x_masked = x.clone()
    n_mask = int(L * mask_ratio)
    for i in range(B):
        idx = torch.randperm(L)[:n_mask]
        x_masked[i, :, idx] = 0.0
    return x_masked, idx   # idx is just the last sample's, used for reporting


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MEDICAL DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_medical():
    BIDMC_DIR = os.path.join(ROOT, "data", "raw", "bidmc")
    CAPNO_DIR = os.path.join(ROOT, "data", "raw", "capnobase")
    bidmc = load_bidmc(BIDMC_DIR)
    capno = load_capnobase(CAPNO_DIR)
    all_wins = bidmc.windows + capno.windows

    pids = sorted({w.patient_id for w in all_wins})
    rng  = np.random.default_rng(42)
    rng.shuffle(pids)
    n    = len(pids)
    tr_ids = set(pids[:int(n*0.6)])
    va_ids = set(pids[int(n*0.6):int(n*0.8)])
    te_ids = set(pids[int(n*0.8):])

    def split(ids):
        n_ = [w for w in all_wins if w.patient_id in ids
              and w.label == AnomalyLabel.NORMAL]
        a_ = [w for w in all_wins if w.patient_id in ids
              and w.label != AnomalyLabel.NORMAL]
        return n_, a_

    tr_n, tr_a = split(tr_ids)
    va_n, va_a = split(va_ids)
    te_n, te_a = split(te_ids)
    all_a = tr_a + va_a + te_a   # pool all anomalies for evaluation

    print(f"  Medical: train_normal={len(tr_n)}  train_anomaly={len(tr_a)}"
          f"  val_normal={len(va_n)}  test_normal={len(te_n)}"
          f"  total_anomaly={len(all_a)}")
    return tr_n, tr_a, va_n, te_n, all_a


# ══════════════════════════════════════════════════════════════════════════════
# MASKED AUTOENCODER PRETRAINING (self-supervised)
# ══════════════════════════════════════════════════════════════════════════════

def ssl_pretrain(model, all_normal_wins, epochs=60, batch_size=64,
                 lr=1e-3, patience=10, mask_ratio=0.40):
    """
    Masked reconstruction: train the autoencoder to reconstruct the full
    signal from a partially-masked version. No labels used.
    """
    # Use ALL normal windows (train + val) for SSL pretraining — no labels needed
    data_t = windows_to_tensor(all_normal_wins)
    N = data_t.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, wait, best_state = float("inf"), 0, None
    val_size = max(100, N // 10)

    print(f"  SSL pretraining on {N} normal windows (mask ratio={mask_ratio:.0%})")

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N)
        train_idx = idx[val_size:]
        val_idx   = idx[:val_size]
        ep_loss = 0.0

        for start in range(0, len(train_idx), batch_size):
            b_idx = train_idx[start:start+batch_size]
            x_orig   = data_t[b_idx].to(device)           # (B,1,L) original
            x_masked, _ = apply_mask(x_orig, mask_ratio)  # masked input

            x_hat, _ = model(x_masked)
            # Reconstruct ORIGINAL from MASKED — this is the SSL pretext task
            loss = ((x_hat - x_orig) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(b_idx)
        ep_loss /= len(train_idx)

        # Validation: measure reconstruction of masked → original
        model.eval()
        with torch.no_grad():
            x_v = data_t[val_idx].to(device)
            x_v_masked, _ = apply_mask(x_v, mask_ratio)
            xv_hat, _ = model(x_v_masked)
            val_loss = ((xv_hat - x_v) ** 2).mean().item()

        if ep % 10 == 0:
            print(f"  [SSL] Epoch {ep:3d}/{epochs}  train={ep_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            wait       = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"  [SSL] Early stop at epoch {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)


# ══════════════════════════════════════════════════════════════════════════════
# CONTRASTIVE FINE-TUNING (identical to bearing experiment)
# ══════════════════════════════════════════════════════════════════════════════

def contrastive_finetune(model, train_normal_t, train_anomaly_t, val_normal_t,
                         epochs=80, batch_size=64, lr=5e-5,
                         patience=15, margin=10.0, tag="CONTRAST"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, wait, best_state = float("inf"), 0, None
    Nn, Na = train_normal_t.shape[0], train_anomaly_t.shape[0]

    model.eval()
    with torch.no_grad():
        _, z = model(train_normal_t[:256].to(device))
        mu_n = z.mean(0)

    for ep in range(1, epochs + 1):
        model.train()
        idx_n = torch.randperm(Nn)
        ep_loss, nb = 0.0, 0

        for start in range(0, Nn, batch_size):
            x_n = train_normal_t[idx_n[start:start+batch_size]].to(device)
            x_a = train_anomaly_t[torch.randint(0, Na, (len(x_n),))].to(device)
            x_hat, z_n = model(x_n)
            _,     z_a = model(x_a)
            recon  = ((x_hat - x_n)**2).mean()
            centre = ((z_n - mu_n)**2).mean()
            hinge  = torch.clamp(margin - torch.norm(z_a - mu_n, dim=1), min=0).mean()
            loss   = recon + 0.1*centre + 0.5*hinge
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            xv, _ = model(val_normal_t.to(device))
            val_loss = ((xv - val_normal_t.to(device))**2).mean().item()

        if ep % 20 == 0:
            print(f"  [{tag}] Epoch {ep:3d}/{epochs}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{tag}] Early stop epoch {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, test_normal_wins, all_anomaly_wins, tag):
    from sklearn.metrics import roc_auc_score, accuracy_score

    tn_t = windows_to_tensor(test_normal_wins)
    ta_t = windows_to_tensor(all_anomaly_wins)

    model.eval()
    model.fit_mahalanobis(tn_t.to(device))

    all_t  = torch.cat([tn_t, ta_t])
    scores = model.mahalanobis_score(all_t.to(device)).cpu().numpy()
    labels = np.array([0]*len(test_normal_wins) + [1]*len(all_anomaly_wins))

    auroc = roc_auc_score(labels, scores)
    thr   = float(np.percentile(
        model.mahalanobis_score(tn_t.to(device)).cpu().numpy(), 85))
    preds = (scores > thr).astype(int)
    acc   = accuracy_score(labels, preds)
    sens  = (preds[labels==1] == 1).mean() if (labels==1).any() else 0.0

    print(f"\n  [{tag}] AUROC={auroc:.4f}  Accuracy={acc:.4f}  Sensitivity={sens:.4f}")
    return {"auroc": float(auroc), "accuracy": float(acc), "sensitivity": float(sens)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Self-Supervised Baseline (Masked Autoencoder) — MediSense")
    print("  Pretraining on medical data only, no labels, no external source")
    print("="*60)

    cfg = CNNConfig()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  [Step 1] Loading medical data…")
    tr_n, tr_a, va_n, te_n, all_a = load_medical()

    tr_t  = windows_to_tensor(tr_n)
    tr_at = windows_to_tensor(tr_a)
    va_t  = windows_to_tensor(va_n)

    # ── SSL pretraining ───────────────────────────────────────────────────────
    if os.path.exists(SSL_PRETRAIN):
        print("\n  [SKIP] SSL pretrain checkpoint found, loading…")
        ssl_model = MediSenseCNN(cfg).to(device)
        ckpt = torch.load(SSL_PRETRAIN, map_location=device, weights_only=False)
        ssl_model.load_state_dict(ckpt["state_dict"])
    else:
        print("\n  [Step 2] Masked autoencoder pretraining on medical data…")
        ssl_model = MediSenseCNN(cfg).to(device)
        all_normal = tr_n + va_n   # all normal windows, no labels needed
        ssl_pretrain(ssl_model, all_normal, epochs=60, batch_size=64,
                     lr=1e-3, patience=10, mask_ratio=0.40)
        torch.save({"state_dict": ssl_model.state_dict(), "cfg": cfg}, SSL_PRETRAIN)
        print(f"  Saved → {SSL_PRETRAIN}")

    # ── Contrastive fine-tuning ───────────────────────────────────────────────
    print("\n  [Step 3] Contrastive fine-tuning SSL model…")
    contrastive_finetune(ssl_model, tr_t, tr_at, va_t,
                         epochs=80, batch_size=64, lr=5e-5,
                         patience=15, margin=10.0, tag="SSL-FT")
    torch.save({"state_dict": ssl_model.state_dict(), "cfg": cfg}, SSL_FINETUNE)
    print(f"  Saved → {SSL_FINETUNE}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n  [Step 4] Evaluating SSL model…")
    ssl_results = evaluate(ssl_model, te_n, all_a, "SSL-MEDICAL")

    # ── Load previous results for comparison ──────────────────────────────────
    prev_cnn   = os.path.join(ROOT, "experiments", "results", "cnn_experiment_results.json")
    prev_stock = os.path.join(ROOT, "experiments", "results", "stock_baseline_results.json")
    bearing_auroc = baseline_auroc = stock_auroc = None
    if os.path.exists(prev_cnn):
        with open(prev_cnn) as f: d = json.load(f)
        bearing_auroc = d["transfer_learning"]["auroc"]
        baseline_auroc = d["baseline"]["auroc"]
    if os.path.exists(prev_stock):
        with open(prev_stock) as f: d = json.load(f)
        stock_auroc = d["stock_pretrain"]["auroc"]

    # ── Save ──────────────────────────────────────────────────────────────────
    comparison = {
        "ssl_medical": ssl_results,
        "stock_pretrain_auroc": stock_auroc,
        "bearing_pretrain_auroc": bearing_auroc,
        "baseline_auroc": baseline_auroc,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    # ── Print comparison ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FULL COMPARISON TABLE (so far)")
    print("="*60)
    if baseline_auroc: print(f"  Random init baseline          : AUROC {baseline_auroc:.4f}")
    if stock_auroc:    print(f"  Stock prices pretrain         : AUROC {stock_auroc:.4f}")
    print(f"  Self-supervised (medical only) : AUROC {ssl_results['auroc']:.4f}  ← NEW")
    if bearing_auroc:  print(f"  NASA bearing pretrain         : AUROC {bearing_auroc:.4f}")
    print("  Time-series foundation model  : run scripts/run_foundation_baseline.py")
    print("="*60)
    print(f"\n  Results saved → {RESULTS_PATH}\n")
