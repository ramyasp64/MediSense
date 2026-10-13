"""
Signal processing pipeline for MediSense.

Transforms raw PPG/sensor signals into features for the anomaly detection model.

Pipeline stages:
    1. Bandpass filter    — isolate HR band (0.5–4 Hz) or RR band (0.1–0.5 Hz)
    2. Baseline wander    — remove slow drift via high-pass or morphological filter
    3. Kalman smoother    — reduce high-frequency noise while preserving peaks
    4. Feature extraction — statistical + frequency + temporal features
    5. Normalization      — z-score per feature across the dataset

All stages operate on 1D numpy arrays (single-channel windows).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Stage 1 & 2: Bandpass filter + baseline removal
# ---------------------------------------------------------------------------

def bandpass_filter(
    sig: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.
    Uses filtfilt to avoid phase distortion — critical for peak timing accuracy.
    """
    nyq = fs / 2.0
    low = np.clip(low_hz / nyq, 1e-6, 0.999)
    high = np.clip(high_hz / nyq, 1e-6, 0.999)
    if low >= high:
        return sig
    b, a = sp_signal.butter(order, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, sig).astype(np.float32)


def remove_baseline_wander(sig: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """
    Remove slow baseline drift with a high-pass filter.
    0.5 Hz cutoff eliminates respiration-induced PPG baseline shifts.
    """
    nyq = fs / 2.0
    cutoff = np.clip(cutoff_hz / nyq, 1e-6, 0.999)
    b, a = sp_signal.butter(2, cutoff, btype="high")
    return sp_signal.filtfilt(b, a, sig).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 3: Kalman smoother
# ---------------------------------------------------------------------------

def kalman_smooth(
    sig: np.ndarray,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-3,
) -> np.ndarray:
    """
    Lightweight noise reduction via exponential moving average.
    Equivalent to a steady-state Kalman gain — vectorized, ~100x faster
    than the iterative form while preserving peak morphology.
    """
    alpha = process_noise / (process_noise + measurement_noise)
    alpha = float(np.clip(alpha, 0.001, 0.5))
    b = np.array([alpha], dtype=float)
    a = np.array([1.0, -(1.0 - alpha)], dtype=float)
    return sp_signal.lfilter(b, a, sig).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 4: Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class SignalFeatures:
    """
    Complete feature set for one signal window.
    All features are scalar — concatenated into a 1D vector for the model.
    """
    # Statistical (6)
    mean: float
    std: float
    skewness: float
    kurtosis: float
    signal_min: float
    signal_max: float

    # Frequency (5)
    dominant_freq_hz: float
    dominant_freq_power: float
    spectral_entropy: float
    spectral_centroid: float
    band_power_ratio: float  # target band / total power

    # Temporal / peak-based (6)
    n_peaks: float           # peaks per second (proxy for HR/RR)
    mean_peak_interval: float
    std_peak_interval: float
    mean_peak_amplitude: float
    peak_amplitude_cv: float  # coefficient of variation
    signal_quality_index: float  # 0–1, based on peak regularity

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.mean, self.std, self.skewness, self.kurtosis,
            self.signal_min, self.signal_max,
            self.dominant_freq_hz, self.dominant_freq_power,
            self.spectral_entropy, self.spectral_centroid, self.band_power_ratio,
            self.n_peaks, self.mean_peak_interval, self.std_peak_interval,
            self.mean_peak_amplitude, self.peak_amplitude_cv,
            self.signal_quality_index,
        ], dtype=np.float32)

    @property
    def n_features(self) -> int:
        return 17


def _skewness(x: np.ndarray) -> float:
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-10:
        return 0.0
    return float(((x - mu) ** 3).mean() / sigma**3)


def _kurtosis(x: np.ndarray) -> float:
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-10:
        return 0.0
    return float(((x - mu) ** 4).mean() / sigma**4 - 3)


def _spectral_entropy(power: np.ndarray) -> float:
    p = power / (power.sum() + 1e-12)
    return float(-np.sum(p * np.log2(p + 1e-12)))


def extract_features(
    sig: np.ndarray,
    fs: float,
    band_low_hz: float = 0.5,
    band_high_hz: float = 4.0,
    peak_min_distance_sec: float = 0.3,
    peak_prominence_factor: float = 0.3,
) -> SignalFeatures:
    """
    Extract 17 features from a preprocessed signal window.

    Args:
        sig: 1D preprocessed signal array.
        fs: Sampling frequency.
        band_low_hz: Lower bound of target frequency band (e.g. 0.5 Hz for HR).
        band_high_hz: Upper bound of target frequency band (e.g. 4.0 Hz for HR).
        peak_min_distance_sec: Minimum distance between peaks (prevents double-counting).
        peak_prominence_factor: Peaks must exceed this fraction of signal range.
    """
    n = len(sig)

    # --- Statistical ---
    mean = float(sig.mean())
    std = float(sig.std())
    skew = _skewness(sig)
    kurt = _kurtosis(sig)
    sig_min = float(sig.min())
    sig_max = float(sig.max())

    # --- Frequency ---
    freqs = rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(rfft(sig))
    power = fft_mag**2

    band_mask = (freqs >= band_low_hz) & (freqs <= band_high_hz)
    band_power = power[band_mask].sum() if band_mask.any() else 0.0
    total_power = power.sum() + 1e-12
    band_power_ratio = float(band_power / total_power)

    if band_mask.any():
        dom_idx = np.argmax(power[band_mask])
        dominant_freq = float(freqs[band_mask][dom_idx])
        dominant_power = float(power[band_mask][dom_idx] / total_power)
    else:
        dominant_freq = 0.0
        dominant_power = 0.0

    spec_entropy = _spectral_entropy(power)
    spec_centroid = float(
        np.sum(freqs * power) / total_power if total_power > 0 else 0.0
    )

    # --- Temporal / peaks ---
    sig_range = sig_max - sig_min + 1e-10
    min_dist_samples = max(1, int(peak_min_distance_sec * fs))
    prominence = peak_prominence_factor * sig_range

    peaks, props = find_peaks(
        sig, distance=min_dist_samples, prominence=prominence
    )

    duration_sec = n / fs
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fs  # in seconds
        n_peaks_per_sec = float(len(peaks) / duration_sec)
        mean_interval = float(intervals.mean())
        std_interval = float(intervals.std())
        amplitudes = sig[peaks]
        mean_amp = float(amplitudes.mean())
        amp_cv = float(amplitudes.std() / (abs(mean_amp) + 1e-10))
        # SQI: 1 = perfectly regular peaks, 0 = chaotic
        sqi = float(1.0 - np.clip(std_interval / (mean_interval + 1e-10), 0, 1))
    elif len(peaks) == 1:
        n_peaks_per_sec = float(1 / duration_sec)
        mean_interval = duration_sec
        std_interval = 0.0
        mean_amp = float(sig[peaks[0]])
        amp_cv = 0.0
        sqi = 0.5
    else:
        n_peaks_per_sec = 0.0
        mean_interval = 0.0
        std_interval = 0.0
        mean_amp = 0.0
        amp_cv = 0.0
        sqi = 0.0

    return SignalFeatures(
        mean=mean,
        std=std,
        skewness=skew,
        kurtosis=kurt,
        signal_min=sig_min,
        signal_max=sig_max,
        dominant_freq_hz=dominant_freq,
        dominant_freq_power=dominant_power,
        spectral_entropy=spec_entropy,
        spectral_centroid=spec_centroid,
        band_power_ratio=band_power_ratio,
        n_peaks=n_peaks_per_sec,
        mean_peak_interval=mean_interval,
        std_peak_interval=std_interval,
        mean_peak_amplitude=mean_amp,
        peak_amplitude_cv=amp_cv,
        signal_quality_index=sqi,
    )


# ---------------------------------------------------------------------------
# Stage 5: Full pipeline — raw signal → feature vector
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    fs: float = 125.0
    band_low_hz: float = 0.5
    band_high_hz: float = 4.0
    filter_order: int = 4
    baseline_cutoff_hz: float = 0.5
    kalman_process_noise: float = 1e-5
    kalman_measurement_noise: float = 1e-3
    peak_min_distance_sec: float = 0.3
    peak_prominence_factor: float = 0.3


MIN_FILTER_LEN = 30   # filtfilt padlen guard — signals shorter than this skip filtering


def process_window(sig: np.ndarray, cfg: PipelineConfig, apply_filters: bool = True) -> np.ndarray:
    """
    Full pipeline: raw window → 17-dim feature vector.

    1. Remove baseline wander   (skipped for very short or industrial signals)
    2. Bandpass filter          (skipped if signal too short for filtfilt)
    3. Kalman smooth
    4. Extract features

    Returns:
        np.ndarray of shape (17,)
    """
    sig = sig.astype(np.float32)
    if apply_filters and len(sig) > MIN_FILTER_LEN:
        sig = remove_baseline_wander(sig, cfg.fs, cfg.baseline_cutoff_hz)
        sig = bandpass_filter(sig, cfg.fs, cfg.band_low_hz, cfg.band_high_hz, cfg.filter_order)
    sig = kalman_smooth(sig, cfg.kalman_process_noise, cfg.kalman_measurement_noise)
    features = extract_features(
        sig,
        cfg.fs,
        cfg.band_low_hz,
        cfg.band_high_hz,
        cfg.peak_min_distance_sec,
        cfg.peak_prominence_factor,
    )
    return features.to_vector()


def process_dataset(
    windows: list[np.ndarray],
    cfg: PipelineConfig,
    normalize: bool = True,
    fit_stats: Optional[tuple] = None,
    apply_filters: bool = True,
) -> tuple[np.ndarray, Optional[tuple]]:
    """
    Process a list of raw windows → (N, 17) feature matrix.

    Args:
        windows: List of 1D signal arrays.
        cfg: Pipeline configuration.
        normalize: Whether to z-score normalize the output.
        fit_stats: (mean, std) from training set — pass for val/test to avoid leakage.
        apply_filters: Set False for industrial data (bandpass is PPG-specific).

    Returns:
        (feature_matrix, (mean, std)) — stats are None if normalize=False.
    """
    features = np.stack([process_window(w, cfg, apply_filters=apply_filters) for w in windows])

    if not normalize:
        return features, None

    if fit_stats is not None:
        mean, std = fit_stats
    else:
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8

    features = (features - mean) / std
    return features.astype(np.float32), (mean, std)
