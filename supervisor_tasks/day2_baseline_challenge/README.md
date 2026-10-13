# Day 2: Baseline Challenge - Proving Bearing Pretraining Has Domain-Specific Value

**Date:** October 2026
**Assigned by:** Dr. Anselm von Gladiß, University of Koblenz
**Status:** Completed. Consolidated email sent with all results.

---

## Background: Why MediSense matters

MediSense is low-cost vital signs monitoring for the 3 billion people in rural India,
Sub-Saharan Africa, and Southeast Asia who lack basic healthcare diagnostics.

The child with pneumonia whose respiratory rate goes unmeasured. The elderly patient with
sepsis whose heart rate is not monitored. Every person who dies from a treatable condition
simply because vital signs are not being tracked. This thesis establishes the technical
foundation proving that medical-grade accuracy is achievable at one-tenth the cost through
intelligent software. MediSense is both a technical innovation and a social entrepreneurship
initiative targeting 480 lives saved annually by 2031.

The model is fine-tuned once on the shared BIDMC and CapnoBase dataset, then deployed to any
rural clinic without modification. No per-clinic data collection is needed. Human physiology
is universal: normal and abnormal PPG thresholds do not vary by geography. Every MediSense
deployment at every rural community hospital uses the same fine-tuned model.

---

## The Challenge

Dr. von Gladiß requested comparison of bearing pretraining against three baselines:
(1) stock prices as an irrelevant domain, (2) self-supervised learning on the medical data
itself, and (3) a time-series foundation model.

---

## Experimental Design

Four conditions were run against the same evaluation protocol:
- Same medical dataset (BIDMC and CapnoBase, 95 patients, 5.3% anomaly prevalence)
- Same model architecture (1D-CNN Autoencoder, 1.45M parameters)
- Same fine-tuning procedure (contrastive: MSE + centre loss + hinge loss)
- Same evaluation metric (AUROC, correct for class-imbalanced anomaly detection)
- Same threshold (p85: 85th percentile of validation normal scores)
- Same patient split (60/20/20, deterministic seed 42)

### Condition 1: Random Initialisation (existing MediSense baseline)

No pretraining. CNN trained from scratch on medical data only.

**Result: AUROC 0.464** (near random, confirms task is hard without pretraining)

### Condition 2: Stock Prices Pretraining

**Script:** `scripts/run_stock_baseline.py`

Generated 50,000 synthetic windows from an ARIMA random walk with AR(1) log-returns. Pre-trained
the CNN autoencoder on stock windows, then fine-tuned on medical data using the same contrastive
procedure.

Stock prices are not cyclostationary (random walk, no periodic statistical regularity), have no
amplitude-modulated anomaly structure, and are spectrally flat rather than narrow-band harmonic.

**Result: AUROC 0.616.** Better than random, but significantly below bearing (0.746).

### Condition 3: Self-Supervised Learning on Medical Data

**Script:** `scripts/run_selfsupervised_baseline.py`

Masked Time-Series Autoencoder (MTSA): mask 40% of timesteps at random, train the CNN to
reconstruct the original signal from the masked version. No labels required. Pre-trained on
all 6,840 normal windows from train and validation splits. Then fine-tuned with the same
contrastive procedure.

Key distinction from bearing pretraining: SSL requires 6,840 normal windows from the medical
dataset at the pretraining stage. Bearing pretraining requires zero medical data for pretraining.
It uses publicly available NASA industrial data, so the full pipeline can be built without
needing access to medical institutional data at the pretraining stage.

**Result: AUROC 0.714.** Strong, but below bearing (0.746).

### Condition 4: Foundation Model (Chronos, Zero-Shot)

**Script:** `scripts/run_foundation_baseline.py`

Amazon Chronos-T5-Small (Ansari et al., 2024): a transformer pre-trained on millions of diverse
time series spanning weather, energy, finance, traffic, sensor, and medical records. Used
zero-shot: no fine-tuning, no labels. Anomaly score equals mean prediction interval width.

**Result: AUROC 0.517.** Barely above random. Stock prices (0.616) and SSL (0.714) both surpass
it. The strongest available generic time-series prior fails on PPG anomaly detection.

### Condition 5: NASA Bearing Pretraining (MediSense)

**Script:** `scripts/run_cnn_experiment.py` (main repository)

Pre-trained on real NASA FEMTO bearing vibration data. Fine-tuned with contrastive procedure.

**Result: AUROC 0.746.** Best across all five conditions.

> Note: This is not a final result. Optimisation toward clinical feasibility continues.
> The goal of MediSense is to reach clinically meaningful sensitivity for low-cost deployment
> in clinics serving underserved populations. The 0.746 figure reflects the current state.
> Work continues until that target is met.

---

## Results Summary

| Condition | Pretraining Domain | AUROC | Delta vs Random |
|---|---|---|---|
| Random initialisation | None | 0.464 | baseline |
| Chronos (zero-shot) | Universal (millions of series) | 0.517 | +5.3 pp |
| Stock prices | Financial (ARIMA random walk) | 0.616 | +15.2 pp |
| Self-supervised (medical) | Medical PPG (no labels) | 0.714 | +25.0 pp |
| NASA bearing (MediSense) | Industrial vibration | 0.746 | +28.2 pp |

Statistical significance: McNemar's test, p < 0.0001 (bearing vs random baseline).

---

## Decomposing the Bearing Advantage

**Component 1: Generic time-series structure (+15.2 pp)**
Any pre-trained representation beats random. Stock prices confirm this.

**Component 2: Cyclostationary signal structure (+25.0 pp)**
Self-supervised medical pretraining outperforms stock prices because PPG is cyclostationary
(statistical properties repeat at the heart rate frequency). A model trained on the medical
signal captures this structure even without labels.

**Component 3: Bearing-specific degradation physics (+28.2 pp, or +3.2 pp over SSL)**
Bearing data specifically adds the final increment. Bearing vibration and PPG share:
- Cyclostationary structure (periodic statistics)
- Amplitude-modulated anomaly signatures (bearing faults produce growing impulse amplitude; cardiac anomalies produce amplitude-modulated waveform deviations)
- Narrow-band harmonic structure (dominant fundamental with harmonic decay)

These three properties are shared between the two domains and absent in stock prices and
generic foundation model training corpora.

---

## Why Chronos Failed

A transformer trained on millions of diverse time series performs worse than stock prices
(0.616) and far worse than medical SSL (0.714) or bearing (0.746). The PPG anomaly detection
task requires cyclostationary, amplitude-modulated structural priors. Broad domain exposure
does not provide them.

---

## The SSL Pretraining Distinction

The SSL baseline required 6,840 normal windows from the shared medical dataset for its
pretraining phase. Bearing pretraining uses NASA industrial data: zero medical data required
for pretraining. Since MediSense deploys the same fine-tuned model to every clinic (human
physiology is universal), the bearing approach keeps the entire pretraining pipeline
independent of medical institutional data availability.

---

## Files in This Folder

```
day2_baseline_challenge/
+-- README.md
+-- scripts/
|   +-- run_stock_baseline.py
|   +-- run_selfsupervised_baseline.py
|   +-- run_foundation_baseline.py
+-- results/
    +-- cnn_experiment_results.json
    +-- stock_baseline_results.json
    +-- ssl_baseline_results.json
    +-- foundation_baseline_results.json
```

---

## Email sent to Dr. von Gladiß

A single consolidated email was sent after all four experiments completed, containing the full
comparison table, the three-component decomposition, the theoretical argument, and next steps.
