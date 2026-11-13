"""Signal-analysis tools for Particula Primordial Kick Laboratory.

All reported values are computed from the selected WAV or rendered buffer.
The low-frequency ridge is an autocorrelation-based estimate, not a guaranteed
fundamental-frequency tracker for every transient or noisy signal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import ShortTimeFFT, correlate, windows

EPS = 1e-12


def _as_mono_float(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio)
    if data.ndim == 2:
        data = np.mean(data.astype(np.float64), axis=1)
    else:
        data = data.astype(np.float64).reshape(-1)
    if data.size == 0:
        raise ValueError("Audio buffer is empty.")
    original_dtype = np.asarray(audio).dtype
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        if np.issubdtype(original_dtype, np.unsignedinteger):
            midpoint = (float(info.max) + 1.0) / 2.0
            data = (data - midpoint) / midpoint
        else:
            scale = float(max(abs(info.min), abs(info.max)))
            data = data / scale
    if not np.all(np.isfinite(data)):
        raise ValueError("Audio contains NaN or infinite values.")
    return data


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), EPS)))


def _rms_envelope(audio: np.ndarray, sample_rate: int, frame_ms: float = 10.0, hop_ms: float = 5.0):
    frame = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if len(audio) < frame:
        padded = np.pad(audio, (0, frame - len(audio)))
        frames = padded[None, :]
    else:
        starts = np.arange(0, len(audio) - frame + 1, hop)
        frames = np.stack([audio[i : i + frame] for i in starts])
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    times = (np.arange(len(rms)) * hop + frame / 2) / sample_rate
    return times, rms


def _decay_times(times: np.ndarray, envelope: np.ndarray) -> dict[str, float | None]:
    if envelope.size == 0 or float(np.max(envelope)) <= EPS:
        return {"t_minus_6_db_ms": None, "t_minus_20_db_ms": None, "t_minus_40_db_ms": None}
    peak_index = int(np.argmax(envelope))
    peak = float(envelope[peak_index])
    result: dict[str, float | None] = {}
    for threshold_db in (6, 20, 40):
        threshold = peak * (10.0 ** (-threshold_db / 20.0))
        below = np.flatnonzero(envelope[peak_index:] <= threshold)
        key = f"t_minus_{threshold_db}_db_ms"
        result[key] = (float((times[peak_index + below[0]] - times[peak_index]) * 1000.0) if below.size else None)
    return result


def _low_frequency_ridge(audio: np.ndarray, sample_rate: int, min_hz: float = 25.0, max_hz: float = 250.0):
    """Estimate a periodic low-frequency component using short-frame autocorrelation."""
    frame = min(4096, len(audio))
    if frame < 512:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    hop = max(128, int(sample_rate * 0.01))
    min_lag = max(2, int(sample_rate / max_hz))
    max_lag = min(frame - 2, int(sample_rate / min_hz))
    if max_lag <= min_lag:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    window = np.hanning(frame)
    estimates, confidence, centers = [], [], []
    starts = list(range(0, max(1, len(audio) - frame + 1), hop))
    if not starts:
        starts = [0]
    for start in starts:
        segment = audio[start : start + frame]
        if len(segment) < frame:
            segment = np.pad(segment, (0, frame - len(segment)))
        segment = (segment - np.mean(segment)) * window
        energy = float(np.sqrt(np.mean(segment * segment)))
        centers.append((start + frame / 2) / sample_rate)
        if energy < 1e-5:
            estimates.append(np.nan)
            confidence.append(0.0)
            continue
        corr = correlate(segment, segment, mode="full", method="fft")[frame - 1 :]
        corr /= max(float(corr[0]), EPS)
        search = corr[min_lag : max_lag + 1]
        lag_index = int(np.argmax(search)) + min_lag
        # Parabolic interpolation gives a sub-bin lag estimate.
        if 1 <= lag_index < len(corr) - 1:
            left, mid, right = corr[lag_index - 1 : lag_index + 2]
            denom = left - 2.0 * mid + right
            offset = 0.5 * (left - right) / denom if abs(denom) > EPS else 0.0
        else:
            offset = 0.0
        lag = max(1.0, lag_index + float(np.clip(offset, -0.5, 0.5)))
        estimates.append(float(sample_rate / lag))
        confidence.append(float(corr[lag_index]))
    return np.asarray(centers), np.asarray(estimates), np.asarray(confidence)


def analyze_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Compute reproducible time, level and spectral descriptors for one signal."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    data = _as_mono_float(audio)
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data * data)))
    crest = _db(peak / max(rms, EPS))

    # Whole-signal magnitude spectrum (windowed to reduce leakage).
    windowed = data * np.hanning(len(data))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
    spectral_power = spectrum * spectrum
    mag_sum = float(np.sum(spectrum))
    centroid = float(np.sum(frequencies * spectrum) / mag_sum) if mag_sum > EPS else 0.0
    dominant_mask = (frequencies >= 20.0) & (frequencies <= min(12_000.0, sample_rate / 2))
    dominant_frequency = float(frequencies[dominant_mask][np.argmax(spectrum[dominant_mask])]) if np.any(dominant_mask) else 0.0

    bands = [("20_60_hz", 20, 60), ("60_120_hz", 60, 120), ("120_250_hz", 120, 250),
             ("250_1000_hz", 250, 1000), ("1k_5k_hz", 1000, 5000), ("5k_12k_hz", 5000, 12000)]
    total_band_power = float(np.sum(spectral_power[(frequencies >= 20) & (frequencies <= 12000)]))
    band_energy_pct: dict[str, float] = {}
    for name, low, high in bands:
        mask = (frequencies >= low) & (frequencies < min(high, sample_rate / 2))
        power = float(np.sum(spectral_power[mask])) if np.any(mask) else 0.0
        band_energy_pct[name] = 100.0 * power / max(total_band_power, EPS)

    env_t, env = _rms_envelope(data, sample_rate)
    decay = _decay_times(env_t, env)
    ridge_t, ridge_hz, ridge_conf = _low_frequency_ridge(data, sample_rate)

    return {
        "duration_s": len(data) / sample_rate,
        "sample_rate_hz": int(sample_rate),
        "samples": int(len(data)),
        "peak_dbfs": _db(peak),
        "rms_dbfs": _db(rms),
        "crest_db": crest,
        "spectral_centroid_hz": centroid,
        "dominant_frequency_hz": dominant_frequency,
        "band_energy_pct": band_energy_pct,
        "decay": decay,
        "waveform": {"time_ms": np.arange(len(data)) / sample_rate * 1000.0, "amplitude": data},
        "spectrum": {"frequency_hz": frequencies, "magnitude_db": 20.0 * np.log10(np.maximum(spectrum / max(float(np.max(spectrum)), EPS), 1e-8))},
        "envelope": {"time_ms": env_t * 1000.0, "rms": env, "rms_db": 20.0 * np.log10(np.maximum(env / max(float(np.max(env)), EPS), 1e-8))},
        "ridge": {"time_s": ridge_t, "frequency_hz": ridge_hz, "confidence": ridge_conf},
    }


def spectrogram_data(audio: np.ndarray, sample_rate: int, max_frequency_hz: float = 8000.0):
    """Return a log-relative STFT magnitude image for plotting."""
    data = _as_mono_float(audio)
    nperseg = min(4096, max(256, len(data)))
    noverlap = min(nperseg - 1, int(nperseg * 0.875))
    hop = max(1, nperseg - noverlap)
    transform = ShortTimeFFT(
        windows.hann(nperseg, sym=False), hop, fs=sample_rate,
        fft_mode="onesided", mfft=nperseg, scale_to="magnitude",
    )
    z = transform.stft(data)
    frequencies = transform.f
    times = transform.t(len(data))
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(z) / max(float(np.max(np.abs(z))), EPS), 1e-6))
    mask = (frequencies >= 20.0) & (frequencies <= min(max_frequency_hz, sample_rate / 2))
    return frequencies[mask], times, magnitude_db[mask]
