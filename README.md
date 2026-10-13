# MediSense: Vital Signs Anomaly Detection for Low-Resource Healthcare Settings

**Status:** Active Research · Universität Koblenz, Germany · 2025–2026  
**Domain:** Healthcare AI · Transfer Learning · Medical Signal Processing  
**Author:** Ramya Subramanian Porselva Bharathi

---

## Research Question

Millions of patients in low-resource clinical settings (community health posts, rural hospitals, field clinics) go unmonitored because accurate vital signs monitoring requires either expensive hardware or large quantities of labelled medical training data.

**MediSense challenges both constraints.**

The core insight: the anomaly structure that appears in factory sensor degradation (*normal → degraded → failure*) is structurally identical to patient vital sign deterioration. A model trained to detect bearing failures learns representations that generalise to cardiac and respiratory emergencies, with remarkably little medical fine-tuning data.

This research bridges **industrial quality control** and **healthcare AI** to serve populations who need it most. Through rigorous technical validation and a clear pathway to deployment, MediSense demonstrates that affordable, accurate vital signs monitoring for low-resource settings is achievable.

---

## System Architecture

```
INDUSTRIAL PRE-TRAINING                    MEDICAL FINE-TUNING
─────────────────────                      ─────────────────────
NASA Bearing Vibration  ─┐                 BIDMC Waveform PPG ─┐
CMAPSS Turbofan Data    ─┼─► Autoencoder   CapnoBase PPG      ─┼─► Fine-tuned Model
SKAB Hydraulic Pump     ─┘   (Encoder      MIMIC-III Waveform ─┘   (Encoder frozen,
                              frozen after                         decoder retrained)
                              pre-training)
                                    │
                                    ▼
                         ┌─────────────────────────────────┐
                         │   DUAL DETECTION PIPELINE        │
                         │                                  │
                         │  Level 1: Sensor Malfunction     │
                         │  Level 2: Patient Anomaly        │
                         │                                  │
                         │  ↓                               │
                         │  CLINICAL ALERT ENGINE           │
                         │  GREEN  — Normal                 │
                         │  YELLOW — Equipment issue        │
                         │  RED    — Urgent assessment      │
                         └─────────────────────────────────┘
```

---

## Model Variants

### Variant A: Feature-Based Autoencoder

Works on **17 hand-engineered features** extracted from a signal window: statistical, frequency-domain, and peak-based characteristics.

```
Input (17 features)
    → Linear(17→64) → BN → ReLU → Dropout
    → Linear(64→32) → BN → ReLU → Dropout
    → Linear(32→16) → BN → ReLU
    → Latent (8-dim)
    → Linear(8→16) → BN → ReLU
    → Linear(16→32) → BN → ReLU
    → Linear(32→64) → BN → ReLU
    → Linear(64→17)
    → Reconstruction

Anomaly Score = MSE(input, reconstruction)
```

**17 features:**
- `mean, std, skewness, kurtosis, min, max` (statistical profile)
- `dominant_freq_hz, dominant_freq_power, spectral_entropy, spectral_centroid, band_power_ratio` (frequency content)
- `n_peaks, mean_peak_interval, std_peak_interval, mean_peak_amplitude, peak_amplitude_cv, signal_quality_index` (rhythm and regularity)

### Variant B: 1D-CNN Autoencoder

Works on **raw signal windows** (1250 samples), preserving full temporal morphology.

```
Input (1 × 1250)
    → Conv1d(1→32,  k=7, s=2) → BN → GELU → 625
    → Conv1d(32→64, k=5, s=2) → BN → GELU → 313
    → Conv1d(64→128,k=5, s=2) → BN → GELU → 157
    → Conv1d(128→64,k=3, s=2) → BN → GELU → 79
    → Flatten → Linear(5056→256) = Latent
    ─────────────── (symmetric decoder) ───────────────
    → ConvTranspose1d ×4 → Output (1 × 1250)

Anomaly Score = MSE(input, reconstruction) per window
Optional: Mahalanobis distance in latent space
```

---

## Signal Processing Pipeline

```python
Stage 1: Bandpass Filter          # Butterworth, zero-phase filtfilt
         HR band:  0.5 – 4.0 Hz  # isolates cardiac frequencies
         RR band:  0.1 – 0.5 Hz  # isolates respiratory frequencies

Stage 2: Baseline Wander Removal  # High-pass 0.5 Hz
         Eliminates slow drift caused by patient movement

Stage 3: Kalman Smoother          # Exponential moving average (steady-state gain)
         Reduces HF noise while preserving peak morphology
         ~100× faster than iterative Kalman

Stage 4: Feature Extraction       # 17-dimensional vector
         Statistical (6) + Frequency (5) + Temporal/Peak (6)

Stage 5: Z-score Normalisation    # Fit on training set only (no data leakage)
```

---

## Dual-Level Clinical Alert System

Designed for healthcare workers with limited medical training.

| Level | Trigger | Alert | Action |
|-------|---------|-------|--------|
| **Level 1 (Equipment)** | Sensor anomaly score >= threshold | YELLOW: "Equipment Problem Detected" | Check sensor placement, reattach, wait 30s |
| **Level 2 (Patient)** | Patient anomaly score >= threshold | RED: Specific vital sign finding | Assess patient immediately |
| **Level 0 (Normal)** | Both scores below threshold | GREEN: "Vital Signs Normal" | Continue monitoring, recheck in 15 min |

**Patient alert specificity.** The system distinguishes:
- Tachycardia (HR >100 BPM) vs. Bradycardia (HR <60 BPM)
- Tachypnea (RR >20/min) vs. Bradypnea (RR <12/min)
- Hypoxemia (SpO2 <95%) with critical threshold at <90%
- Fever (≥38°C) vs. Hypothermia (<36°C)
- Multi-parameter deterioration (all findings reported together)

All thresholds follow WHO and standard clinical practice guidelines.

---

## Datasets

| Dataset | Domain | Source | Use |
|---------|--------|--------|-----|
| BIDMC Waveform | Medical PPG + vital signs | physionet.org | Fine-tuning and evaluation |
| CapnoBase | Medical PPG + capnography | capnobase.org | Fine-tuning and evaluation |
| MIMIC-III Waveform | ICU waveforms (PPG, ECG) | physionet.org | Fine-tuning and evaluation |
| NASA Bearing | Industrial vibration (4 bearings × 3 runs) | NASA IMS Repository | Industrial pre-training |
| CMAPSS | Turbofan engine run-to-failure | NASA | Industrial pre-training |
| SKAB | Hydraulic pump anomalies | skoltech.ru | Industrial pre-training |

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Patient anomaly detection accuracy | ≥ 90% | Clinical standard for screening tools |
| Sensor malfunction detection accuracy | ≥ 95% | Prevent false reassurance from faulty equipment |
| Transfer learning improvement over baseline | ≥ 5 pp | Prove domain transfer adds measurable value |
| Data efficiency gain | ≥ 50% reduction in labelled data | Core claim: less medical data needed |
| Inference latency | < 100 ms/window | Real-time monitoring requirement |
| Statistical significance | McNemar's p < 0.05 | Confirm improvement is not random |

---

## Repository Structure

```
MediSense/
├── src/
│   ├── data/
│   │   ├── base.py                  # VitalSigns dataclass, shared types
│   │   ├── dataset_manager.py       # Unified dataset loading interface
│   │   ├── bidmc_loader.py          # BIDMC Waveform loader
│   │   ├── capnobase_loader.py      # CapnoBase loader
│   │   ├── mimic_loader.py          # MIMIC-III loader
│   │   └── nasa_loader.py           # NASA Bearing + CMAPSS loader
│   ├── signal/
│   │   └── pipeline.py              # Full signal processing pipeline (17 features)
│   ├── models/
│   │   ├── autoencoder.py           # Feature-based autoencoder (Variant A)
│   │   ├── cnn_autoencoder.py       # 1D-CNN autoencoder (Variant B)
│   │   └── trainer.py               # Training loop with transfer learning support
│   ├── evaluation/
│   │   └── metrics.py               # All metrics + McNemar's + data efficiency
│   └── alerts/
│       └── clinical_alerts.py       # Dual-level alert engine
├── scripts/
│   ├── download_data.py             # Automated dataset downloads
│   ├── run_experiment.py            # Feature-based model experiment
│   ├── run_cnn_experiment.py        # CNN model experiment
│   └── run_full_experiment.py       # End-to-end: pre-train → fine-tune → evaluate
├── data/
│   └── raw/                         # Downloaded datasets (not tracked in git)
├── Ramya_SubramanianPorselvaBharathi_MediSenseThesisProposal.pdf
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/ramyasp64/MediSense.git
cd MediSense

pip install torch torchvision numpy scipy scikit-learn matplotlib

python scripts/download_data.py

python scripts/run_full_experiment.py
```

---

## Key Design Decisions

**Why transfer learning from industrial data?**  
Industrial run-to-failure datasets are large, publicly available, and fully labelled. Medical waveform data, by contrast, is scarce, expensive to annotate, and privacy-constrained. Both domains share the same fundamental signal structure: a normal regime, a degradation phase, and a failure event. The encoder learns domain-agnostic anomaly representations.

**Why an autoencoder rather than a classifier?**  
Autoencoders learn the structure of normal signals. Anomalies are detected by their poor reconstruction, not by a labelled anomaly class. This means the model can detect novel clinical deterioration patterns it has never seen during training. Labelled anomaly examples are not required.

**Why dual-level detection?**  
Sensor malfunctions mimic patient deterioration in raw signal features. Without Level 1 separation, a disconnected probe triggers the same alert as a genuine cardiac event. Dual-level detection eliminates this false-alarm pathway while maintaining sensitivity to real emergencies.

---

## Author

**Ramya Subramanian Porselva Bharathi**  
M.S. Web and Data Science · Universität Koblenz, Germany  
ramyasp64@gmail.com · [LinkedIn](https://www.linkedin.com/in/ramya_sp)
