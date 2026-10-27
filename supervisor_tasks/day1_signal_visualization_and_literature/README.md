# Day 1: Signal Visualization and Literature Gap Analysis

**Date:** September 2026
**Assigned by:** Thesis Supervisor, University of Koblenz
**Status:** Completed and sent via email

---

## What was asked

My supervisor assigned two tasks to establish the research foundation:

**Task 1: Signal Visualization.** Generate figures from real data showing that NASA bearing
vibration signals and medical PPG biosignals share structural similarity, justifying the
transfer learning hypothesis.

**Task 2: Literature Gap Analysis.** Survey the existing literature and map where MediSense
sits: which methods exist, what gaps remain, and how this work is novel.

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
rural clinic. No per-clinic data collection is needed. Human physiology is universal: normal
and abnormal PPG thresholds do not vary by geography.

---

## Task 1: Signal Visualization

### What was generated

Six figures from real data (no synthetic images, no internet downloads):

| Figure | File | What it shows |
|---|---|---|
| Fig 1 | `figures/fig1_signal_morphology.png` | Side-by-side waveforms: NASA bearing vibration (normal vs fault) and PPG (normal vs anomaly). Both show periodic structure with anomaly as amplitude deviation. |
| Fig 2 | `figures/fig2_frequency_spectrum.png` | FFT frequency spectra of both signals. Both have a dominant fundamental frequency with harmonic decay, a spectrally narrow-band periodic source. |
| Fig 3 | `figures/fig3_anomaly_patterns.png` | Anomaly patterns overlaid on normal waveforms for both domains, showing the structural similarity of fault signatures. |
| Fig 4 | `figures/fig4_results_dashboard.png` | MediSense model results dashboard: AUROC, sensitivity, specificity, inference latency. |
| Fig 5 | `figures/fig5_spectrogram.png` | Time-frequency spectrograms (STFT) of both signals. Both show stable harmonic bands interrupted at anomaly events. |
| Fig 6 | `figures/fig6_novelty_map.png` | Research novelty map: existing methods (SimCLR, TNC, TS-TCC, CLOCS) vs MediSense positioning. |

### How figures were generated

Script: `generate_visualizations.py`

- NASA bearing data loaded from `data/raw/nasa_bearing/` (FEMTO bearing dataset, IMS dataset)
- PPG data loaded from `data/raw/bidmc/` and `data/raw/capnobase/` using project data loaders
- Matplotlib with MediSense brand palette: teal `#2B7E7E` / orange `#C9613A`

### Key finding from visualization

Both bearing vibration and PPG signals are cyclostationary: their statistical properties
(mean, variance) repeat at the fundamental frequency. This is the core structural property
that justifies transfer learning between the two domains.

Stock prices and generic financial data are not cyclostationary. They follow a random walk
with no periodic statistical regularity. This distinction was later confirmed empirically
in Day 2.

---

## Task 2: Literature Gap Analysis

### Research landscape

| Method | Domain | Approach | Limitation |
|---|---|---|---|
| SimCLR (Chen et al., 2020) | Images | Contrastive self-supervised | Not designed for time series |
| TNC (Tonekaboni et al., 2021) | Medical time series | Temporal neighbour contrast | No cross-domain pretraining |
| TS-TCC (Eldele et al., 2021) | General time series | Temporal and contextual contrast | Needs labelled target data |
| CLOCS (Kiyasseh et al., 2021) | ECG | Patient-level contrastive | ECG-specific, not PPG |
| TimesNet (Wu et al., 2023) | General time series | 2D temporal variation modelling | Supervised, not anomaly detection |
| Chronos (Ansari et al., 2024) | Universal time series | Foundation model (T5 transformer) | Zero-shot prediction, not reconstruction-based anomaly |

### Identified gaps

**Gap 1: No method transfers from industrial to medical biosignals.** All existing cross-domain
work stays within one modality (e.g., ECG to ECG, or bearing to bearing).

**Gap 2: No method targets low-cost clinical deployment for underserved populations.** Existing
approaches assume cloud compute or large labelled datasets. MediSense targets deployment in
clinics serving patients without access to conventional diagnostic equipment, across rural
India, Sub-Saharan Africa, and Southeast Asia.

**Gap 3: Foundation models fail on PPG anomaly detection.** Broad domain coverage does not
substitute for cyclostationary structural priors (confirmed in Day 2 experiments).

**Gap 4: No quantified data efficiency benefit.** Papers report absolute performance but not
how much labelled data they require relative to a baseline.

### MediSense novelty claim

MediSense is the first work to:
- Pre-train on publicly available industrial bearing vibration data (NASA FEMTO)
- Fine-tune via contrastive learning on medical PPG biosignals
- Target anomaly detection for low-cost deployment in clinics serving underserved populations
- Demonstrate statistical significance of improvement (McNemar's test, p < 0.0001)

---

## Email sent to supervisor

A single consolidated email was sent with all six figures attached and a written summary of
the literature gap analysis, establishing the structural similarity argument and the novelty
positioning before the Day 2 challenge was assigned.
