"""
Time-Series Foundation Model Baseline — MediSense
===================================================
Prof. von Gladiß baseline #3: Chronos (Amazon), a foundation model
pretrained on millions of diverse time series from every domain.

What Chronos is:
  A transformer model trained on weather, energy, finance, traffic,
  sensor, and medical time series — the "GPT of time series." It learns
  universal patterns of normal behaviour and generates probabilistic
  forecasts. We use it for zero-shot anomaly detection: a window that
  Chronos cannot predict well (wide prediction intervals) is likely
  anomalous.

Anomaly scoring method:
  For each 1250-sample window, use the first 60% (750 samples) as
  context. Ask Chronos to forecast the remaining 40% (500 samples).
  Anomaly score = mean width of the 10th–90th percentile prediction
  interval. Wider interval = Chronos is more uncertain = more anomalous.

Install (run once):
  pip install chronos-forecasting

Run:
  /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3 scripts/run_foundation_baseline.py
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.signal import resample as sp_resample

RESULTS_PATH = os.path.join(ROOT, "experiments", "results", "foundation_baseline_results.json")
os.makedirs(os.path.join(ROOT, "experiments", "results"), exist_ok=True)

FIXED_LEN = 1250

# ── check Chronos is installed ───────────────────────────────────────────────
try:
    import torch
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
    print("\n  Chronos found ✓")
except ImportError:
    CHRONOS_AVAILABLE = False
    print("\n  ✗ Chronos not installed.")
    print("  Install it with:")
    print("    pip install chronos-forecasting")
    print("  Then run this script again.\n")
    sys.exit(1)

from src.data.bidmc_loader import load_bidmc
from src.data.capnobase_loader import load_capnobase
from src.data.base import AnomalyLabel
from sklearn.metrics import roc_auc_score, accuracy_score


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_fixed(sig, fs=125.0):
    sig = sig.astype(np.float32)
    if len(sig) != FIXED_LEN:
        sig = sp_resample(sig, FIXED_LEN)
    std = sig.std() + 1e-9
    return (sig - sig.mean()) / std


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
    te_ids = set(pids[int(n*0.8):])

    te_n = [w for w in all_wins if w.patient_id in te_ids
            and w.label == AnomalyLabel.NORMAL]
    all_a = [w for w in all_wins if w.label != AnomalyLabel.NORMAL]

    print(f"  Evaluation: {len(te_n)} normal (test patients) + {len(all_a)} anomaly (all)")
    return te_n, all_a


# ══════════════════════════════════════════════════════════════════════════════
# CHRONOS ANOMALY SCORING
# ══════════════════════════════════════════════════════════════════════════════

def chronos_anomaly_score(pipe, windows, batch_size=8,
                          label="scoring"):
    """
    Score each window using Chronos prediction interval width.
    Wider interval = model more uncertain = higher anomaly score.
    Uses context_len=128, prediction_length=32 for fast CPU inference.
    """
    forecast_len = 32
    context_len  = 128   # short context so transformer is fast on CPU
    scores = []

    total = len(windows)
    for start in range(0, total, batch_size):
        batch = windows[start:start+batch_size]
        # Prepare context tensors
        contexts = [
            torch.tensor(_to_fixed(w.signal, float(w.fs))[:context_len],
                         dtype=torch.float32)
            for w in batch
        ]

        # Chronos zero-shot forecast
        with torch.no_grad():
            forecast = pipe.predict(
                context=contexts,
                prediction_length=forecast_len,
                num_samples=10,
            )
        # forecast shape: (batch, num_samples, forecast_len)
        # Anomaly score = mean of (90th pct - 10th pct) = interval width
        lo  = np.percentile(forecast.numpy(), 10, axis=1)   # (batch, L)
        hi  = np.percentile(forecast.numpy(), 90, axis=1)   # (batch, L)
        width = (hi - lo).mean(axis=1)                       # (batch,)
        scores.extend(width.tolist())

        if (start // batch_size) % 5 == 0:
            pct = min(start + batch_size, total) / total * 100
            print(f"  [{label}] {pct:.0f}% ({min(start+batch_size, total)}/{total})")

    return np.array(scores, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Foundation Model Baseline (Chronos) — MediSense")
    print("  Zero-shot anomaly detection — no fine-tuning, no labels")
    print("="*60)

    # ── Load Chronos (smallest model, fits in 4 GB RAM) ───────────────────────
    print("\n  [Step 1] Loading Chronos-T5-Small…")
    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",    # use CPU to avoid MPS memory issues
        torch_dtype=torch.float32,
    )
    print("  Chronos loaded ✓")

    # ── Load medical data ─────────────────────────────────────────────────────
    print("\n  [Step 2] Loading medical data…")
    te_n, all_a = load_medical()

    # ── Score ─────────────────────────────────────────────────────────────────
    print("\n  [Step 3] Scoring normal windows…")
    norm_scores = chronos_anomaly_score(pipe, te_n, label="normal")

    print("\n  [Step 4] Scoring anomaly windows…")
    anom_scores = chronos_anomaly_score(pipe, all_a, label="anomaly")

    # ── AUROC ─────────────────────────────────────────────────────────────────
    all_scores = np.concatenate([norm_scores, anom_scores])
    labels     = np.array([0]*len(te_n) + [1]*len(all_a))
    auroc      = roc_auc_score(labels, all_scores)
    thr        = float(np.percentile(norm_scores, 85))
    preds      = (all_scores > thr).astype(int)
    acc        = accuracy_score(labels, preds)
    sens       = (preds[labels==1] == 1).mean() if (labels==1).any() else 0.0

    print(f"\n  [CHRONOS] AUROC={auroc:.4f}  Accuracy={acc:.4f}  Sensitivity={sens:.4f}")

    # ── Load previous results ─────────────────────────────────────────────────
    bearing_auroc = stock_auroc = ssl_auroc = baseline_auroc = None
    for path, key in [
        (os.path.join(ROOT, "experiments", "results", "cnn_experiment_results.json"),
         [("transfer_learning", "auroc"), ("baseline", "auroc")]),
        (os.path.join(ROOT, "experiments", "results", "stock_baseline_results.json"),
         [("stock_pretrain", "auroc")]),
        (os.path.join(ROOT, "experiments", "results", "ssl_baseline_results.json"),
         [("ssl_medical", "auroc")]),
    ]:
        if os.path.exists(path):
            with open(path) as f: d = json.load(f)
            for keys in key:
                if isinstance(keys, tuple) and len(keys) == 2:
                    k1, k2 = keys
                    if k1 in d:
                        if k1 == "transfer_learning": bearing_auroc = d[k1][k2]
                        elif k1 == "baseline": baseline_auroc = d[k1][k2]
                        elif k1 == "stock_pretrain": stock_auroc = d[k1][k2]
                        elif k1 == "ssl_medical": ssl_auroc = d[k1][k2]

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "chronos_zero_shot": {
            "auroc": float(auroc),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "model": "amazon/chronos-t5-small",
            "method": "prediction interval width (zero-shot, no fine-tuning)"
        }
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ── Full comparison table ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  COMPLETE COMPARISON TABLE")
    print("="*60)
    if baseline_auroc: print(f"  Random init baseline           : AUROC {baseline_auroc:.4f}")
    if stock_auroc:    print(f"  Stock prices pretrain          : AUROC {stock_auroc:.4f}")
    if ssl_auroc:      print(f"  Self-supervised (medical only) : AUROC {ssl_auroc:.4f}")
    print(f"  Foundation model (Chronos)     : AUROC {auroc:.4f}  ← NEW")
    if bearing_auroc:  print(f"  NASA bearing pretrain          : AUROC {bearing_auroc:.4f}")
    print("="*60)
    print(f"\n  Results saved → {RESULTS_PATH}\n")
