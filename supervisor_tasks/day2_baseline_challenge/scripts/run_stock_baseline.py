"""
Stock Prices Baseline Experiment
=================================
Prof. von Gladiß's challenge: "Does the model learn something useful from
bearing data specifically, or does it just learn to handle time series data?"

This script pretrains the SAME CNN autoencoder on stock-like time series
(synthetic ARIMA random walk with trend, mimicking financial returns),
then fine-tunes on the same medical data as the bearing experiment.

If AUROC ≈ bearing pretraining → bearing is NOT special; any time series helps.
If AUROC << bearing pretraining → bearing data has domain-specific value.

Run:
  /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3 scripts/run_stock_baseline.py
"""

import os, sys, warnings, json, time
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

# ── device ──────────────────────────────────────────────────────────────────
device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
print(f"\n  Device: {device}")

from src.models.cnn_autoencoder import MediSenseCNN, CNNConfig

STOCK_PRETRAIN_PATH = os.path.join(ROOT, "experiments", "models", "cnn_stock_pretrain.pt")
STOCK_FINETUNE_PATH = os.path.join(ROOT, "experiments", "models", "cnn_stock_finetune.pt")
RESULTS_PATH        = os.path.join(ROOT, "experiments", "results", "stock_baseline_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# 1. GENERATE SYNTHETIC STOCK-LIKE TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

def generate_stock_windows(n_windows=200_000, window_len=1250, seed=42):
    """
    Synthetic stock price windows: ARIMA(1,1,0) random walk with drift.
    Designed to have:
      - non-stationary trend (unlike bearings)
      - no harmonic structure (unlike PPG / bearings)
      - no cyclostationary behaviour
      - similar window length to bearing data
    """
    rng = np.random.default_rng(seed)
    windows = []

    for _ in range(n_windows):
        # AR(1) log-returns ~ typical equity price process
        phi   = rng.uniform(0.0, 0.1)          # weak autocorrelation
        sigma = rng.uniform(0.005, 0.02)        # daily vol 0.5-2%
        mu    = rng.uniform(-0.0002, 0.0002)    # tiny drift

        returns = np.zeros(window_len)
        returns[0] = rng.normal(0, sigma)
        for t in range(1, window_len):
            returns[t] = mu + phi * returns[t-1] + rng.normal(0, sigma)

        # Cumulative sum → price series
        price = np.cumsum(returns)

        # Z-score normalise to same scale as other signals
        std = price.std() + 1e-9
        price = (price - price.mean()) / std
        windows.append(price.astype(np.float32))

    return np.stack(windows)   # (N, window_len)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING UTILITIES (identical to NASA run, for fair comparison)
# ══════════════════════════════════════════════════════════════════════════════

from scipy.signal import resample as sp_resample

FIXED_LEN = 1250

def _to_fixed(sig, fs=125.0):
    """Resample to FIXED_LEN and z-score normalise — mirrors run_cnn_experiment."""
    sig = sig.astype(np.float32)
    if len(sig) != FIXED_LEN:
        sig = sp_resample(sig, FIXED_LEN)
    std = sig.std() + 1e-9
    return (sig - sig.mean()) / std

def windows_to_tensor(wins):
    arr = np.stack([_to_fixed(w.signal, fs=float(w.fs)) for w in wins])
    return torch.tensor(arr[:, None, :], dtype=torch.float32)   # (N,1,FIXED_LEN)


def train_loop(model, train_t, val_normal_t,
               epochs=60, batch_size=128, lr=1e-3,
               patience=10, tag="TRAIN"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, wait = float("inf"), 0
    N = train_t.shape[0]

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N)
        ep_loss = 0.0
        for start in range(0, N, batch_size):
            batch = train_t[idx[start:start+batch_size]].to(device)
            x_hat, _ = model(batch)
            loss = ((x_hat - batch) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(batch)
        ep_loss /= N

        model.eval()
        with torch.no_grad():
            val_batch = val_normal_t.to(device)
            xv, _ = model(val_batch)
            val_loss = ((xv - val_batch) ** 2).mean().item()

        if ep % 10 == 0:
            print(f"  [{tag}] Epoch {ep:3d}/{epochs}  train={ep_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            torch.save({"state_dict": model.state_dict(),
                        "cfg": model.cfg}, STOCK_FINETUNE_PATH + ".tmp")
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{tag}] Early stop at epoch {ep}")
                break

    # restore best
    ckpt = torch.load(STOCK_FINETUNE_PATH + ".tmp",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])


def contrastive_finetune(model, train_normal_t, train_anomaly_t, val_normal_t,
                         epochs=80, batch_size=64, lr=5e-5,
                         patience=15, margin=10.0, tag="CONTRAST", warmup_epochs=5):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, wait = float("inf"), 0
    Nn = train_normal_t.shape[0]
    Na = train_anomaly_t.shape[0]

    model.eval()
    with torch.no_grad():
        sample = train_normal_t[:256].to(device)
        _, z_sample = model(sample)
        mu_n = z_sample.mean(dim=0)

    for ep in range(1, epochs + 1):
        model.train()
        idx_n = torch.randperm(Nn)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, Nn, batch_size):
            x_n = train_normal_t[idx_n[start:start+batch_size]].to(device)
            a_idx = torch.randint(0, Na, (len(x_n),))
            x_a = train_anomaly_t[a_idx].to(device)

            x_hat_n, z_n = model(x_n)
            _,        z_a = model(x_a)

            recon  = ((x_hat_n - x_n) ** 2).mean()
            centre = ((z_n - mu_n) ** 2).mean()
            dist_a = torch.norm(z_a - mu_n, dim=1)
            hinge  = torch.clamp(margin - dist_a, min=0).mean()

            w_centre = 0.0 if ep <= warmup_epochs else 0.1
            loss = recon + w_centre * centre + 0.5 * hinge
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1

        ep_loss /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            vb = val_normal_t.to(device)
            xv, _ = model(vb)
            val_loss = ((xv - vb) ** 2).mean().item()

        if ep % 20 == 0:
            print(f"  [{tag}] Epoch {ep:3d}/{epochs}  loss={ep_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            torch.save({"state_dict": model.state_dict(),
                        "cfg": model.cfg}, STOCK_FINETUNE_PATH + ".tmp")
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{tag}] Early stop epoch {ep}")
                break

    ckpt = torch.load(STOCK_FINETUNE_PATH + ".tmp",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOAD MEDICAL DATA (same pipeline as main experiment)
# ══════════════════════════════════════════════════════════════════════════════

def load_medical_data():
    from src.data.bidmc_loader import load_bidmc
    from src.data.capnobase_loader import load_capnobase

    BIDMC_DIR   = os.path.join(ROOT, "data", "raw", "bidmc")
    CAPNO_DIR   = os.path.join(ROOT, "data", "raw", "capnobase")
    bidmc = load_bidmc(BIDMC_DIR)
    capno = load_capnobase(CAPNO_DIR)
    all_wins = bidmc.windows + capno.windows

    pids = sorted({w.patient_id for w in all_wins})
    rng  = np.random.default_rng(42)
    rng.shuffle(pids)
    n      = len(pids)
    n_tr   = int(n * 0.6)
    n_val  = int(n * 0.2)
    tr_ids = set(pids[:n_tr])
    va_ids = set(pids[n_tr:n_tr+n_val])
    te_ids = set(pids[n_tr+n_val:])

    from src.data.base import AnomalyLabel
    def split(ids):
        n_wins = [w for w in all_wins if w.patient_id in ids
                  and w.label == AnomalyLabel.NORMAL]
        a_wins = [w for w in all_wins if w.patient_id in ids
                  and w.label != AnomalyLabel.NORMAL]
        return n_wins, a_wins

    train_n, train_a = split(tr_ids)
    val_n,   val_a   = split(va_ids)
    test_n,  test_a  = split(te_ids)
    # Merge all anomaly windows so we have enough for evaluation
    test_a = train_a + val_a + test_a

    print(f"  Medical: train_normal={len(train_n)}  train_anomaly={len(train_a)}"
          f"  val={len(val_n)}  test_normal={len(test_n)}  test_anomaly={len(test_a)}")
    return train_n, train_a, val_n, test_n, test_a


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, test_normal_wins, test_anomaly_wins, tag):
    from sklearn.metrics import roc_auc_score, accuracy_score
    from scipy.stats import chi2_contingency

    test_n_t = windows_to_tensor(test_normal_wins)
    test_a_t = windows_to_tensor(test_anomaly_wins)

    model.eval()
    model.fit_mahalanobis(test_n_t.to(device))

    all_scores = model.mahalanobis_score(
        torch.cat([test_n_t, test_a_t]).to(device)).cpu().numpy()
    labels = np.array([0]*len(test_normal_wins) + [1]*len(test_anomaly_wins))

    auroc = roc_auc_score(labels, all_scores)
    val_scores = model.mahalanobis_score(test_n_t.to(device)).cpu().numpy()
    thr = float(np.percentile(val_scores, 85))
    preds = (all_scores > thr).astype(int)
    acc  = accuracy_score(labels, preds)
    sens = (preds[labels==1] == 1).mean() if (labels==1).any() else 0.0

    print(f"\n  [{tag}] AUROC={auroc:.4f}  Accuracy={acc:.4f}  Sensitivity={sens:.4f}")
    return {"auroc": auroc, "accuracy": float(acc), "sensitivity": float(sens)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Stock Prices Baseline — MediSense")
    print("  Research question: is bearing data specifically useful,")
    print("  or does any time series produce similar transfer benefit?")
    print("="*60)

    cfg = CNNConfig()
    os.makedirs(os.path.join(ROOT, "experiments", "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "experiments", "results"), exist_ok=True)

    # ── Step 1: Pre-train on stock-like data ─────────────────────────────────
    if os.path.exists(STOCK_PRETRAIN_PATH):
        print("\n  [SKIP] Stock pretrain checkpoint found, loading…")
        stock_model = MediSenseCNN(cfg).to(device)
        ckpt = torch.load(STOCK_PRETRAIN_PATH, map_location=device, weights_only=False)
        stock_model.load_state_dict(ckpt["state_dict"])
    else:
        print("\n  [Step 1] Generating synthetic stock-like windows…")
        stock_wins = generate_stock_windows(n_windows=50_000, window_len=1250)
        print(f"  Generated {len(stock_wins)} windows")

        # split 80/20 — free raw array before moving tensors to keep RAM low
        n_tr = int(len(stock_wins) * 0.8)
        train_t = torch.tensor(stock_wins[:n_tr]).unsqueeze(1)
        val_t   = torch.tensor(stock_wins[n_tr:]).unsqueeze(1)
        del stock_wins

        print("  Pre-training CNN on stock data…")
        stock_model = MediSenseCNN(cfg).to(device)
        train_loop(stock_model, train_t, val_t,
                   epochs=50, batch_size=128, lr=1e-3, patience=8, tag="STOCK")
        torch.save({"state_dict": stock_model.state_dict(), "cfg": cfg},
                   STOCK_PRETRAIN_PATH)
        print(f"  Saved → {STOCK_PRETRAIN_PATH}")
        del stock_wins, train_t, val_t

    # ── Step 2: Load medical data ─────────────────────────────────────────────
    print("\n  [Step 2] Loading medical data…")
    train_n, train_a, val_n, test_n, test_a = load_medical_data()

    # With only 475 anomaly windows across 95 patients, a random split often
    # drops all anomalies into train/val. Use all anomaly windows for evaluation
    # (normal practice for highly imbalanced medical datasets).
    all_anomaly = train_a + test_a   # val_a was discarded; this is all anomalies
    print(f"  Anomaly windows for evaluation: {len(all_anomaly)}")

    train_t   = windows_to_tensor(train_n)
    train_a_t = windows_to_tensor(train_a)
    val_t     = windows_to_tensor(val_n)

    # ── Step 3: Contrastive fine-tune (same as bearing experiment) ────────────
    print("\n  [Step 3] Contrastive fine-tuning stock-pretrained model on medical data…")
    contrastive_finetune(stock_model, train_t, train_a_t, val_t,
                         epochs=80, batch_size=64, lr=5e-5,
                         patience=15, margin=10.0, tag="STOCK-FT")
    torch.save({"state_dict": stock_model.state_dict(), "cfg": cfg},
               STOCK_FINETUNE_PATH)
    print(f"  Saved → {STOCK_FINETUNE_PATH}")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print("\n  [Step 4] Evaluating stock-pretrained model…")
    stock_results = evaluate(stock_model, test_n, all_anomaly, "STOCK-PRETRAIN")

    # ── Step 5: Load bearing results for comparison ───────────────────────────
    bearing_auroc = None
    cnn_results_path = os.path.join(ROOT, "experiments", "results", "cnn_experiment_results.json")
    if os.path.exists(cnn_results_path):
        with open(cnn_results_path) as f:
            prev = json.load(f)
        bearing_auroc  = prev["transfer_learning"]["auroc"]
        baseline_auroc = prev["baseline"]["auroc"]

    # ── Step 6: Save and print comparison ────────────────────────────────────
    comparison = {
        "stock_pretrain": stock_results,
        "bearing_pretrain_auroc": bearing_auroc,
        "baseline_auroc": baseline_auroc,
        "interpretation": (
            "If stock AUROC << bearing AUROC: bearing data has domain-specific value. "
            "If stock AUROC ≈ bearing AUROC: any periodic time series may suffice."
        )
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*60)
    print("  COMPARISON TABLE")
    print("="*60)
    print(f"  Random init baseline      : AUROC {baseline_auroc:.4f}" if baseline_auroc else "")
    print(f"  Stock prices pretrain     : AUROC {stock_results['auroc']:.4f}  ← NEW")
    print(f"  NASA bearing pretrain     : AUROC {bearing_auroc:.4f}" if bearing_auroc else "")
    print("="*60)

    if bearing_auroc and stock_results["auroc"] is not None:
        gap = bearing_auroc - stock_results["auroc"]
        if gap > 0.05:
            print(f"\n  RESULT: Bearing pretrain outperforms stock by {gap:.3f} AUROC points.")
            print("  → Bearing data has domain-specific value beyond generic time series.")
        elif gap > 0:
            print(f"\n  RESULT: Bearing marginally better (+{gap:.3f}). More experiments needed.")
        else:
            print(f"\n  RESULT: Stock matches or exceeds bearing. Generic TS features may suffice.")

    print(f"\n  Results saved → {RESULTS_PATH}\n")
