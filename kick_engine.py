"""Procedural kick synthesizer for Particula Primordial.

The engine is deterministic: identical parameters and seed produce identical audio.
It generates mono floating-point audio at 48 kHz by default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KickParams:
    sample_rate: int = 48_000
    duration: float = 0.72
    start_freq: float = 148.0
    end_freq: float = 48.0
    pitch_decay_ms: float = 46.0
    pitch_curve: float = 1.0
    body: float = 0.88
    sub: float = 0.22
    harmonics: float = 0.16
    harmonic_tilt: float = 1.4
    attack_ms: float = 2.0
    decay_ms: float = 480.0
    transient: float = 0.12
    transient_tone: float = 2_200.0
    transient_decay_ms: float = 8.0
    saturation: float = 0.12
    output_db: float = -1.0

    def validated(self) -> "KickParams":
        """Return a safely bounded copy of these parameters."""
        return replace(
            self,
            sample_rate=int(np.clip(self.sample_rate, 22_050, 96_000)),
            duration=float(np.clip(self.duration, 0.12, 2.5)),
            start_freq=float(np.clip(self.start_freq, 45.0, 400.0)),
            end_freq=float(np.clip(self.end_freq, 25.0, 120.0)),
            pitch_decay_ms=float(np.clip(self.pitch_decay_ms, 5.0, 250.0)),
            pitch_curve=float(np.clip(self.pitch_curve, 0.45, 2.5)),
            body=float(np.clip(self.body, 0.0, 1.0)),
            sub=float(np.clip(self.sub, 0.0, 1.0)),
            harmonics=float(np.clip(self.harmonics, 0.0, 1.0)),
            harmonic_tilt=float(np.clip(self.harmonic_tilt, 0.5, 3.0)),
            attack_ms=float(np.clip(self.attack_ms, 0.1, 25.0)),
            decay_ms=float(np.clip(self.decay_ms, 80.0, 2_000.0)),
            transient=float(np.clip(self.transient, 0.0, 1.0)),
            transient_tone=float(np.clip(self.transient_tone, 500.0, 8_000.0)),
            transient_decay_ms=float(np.clip(self.transient_decay_ms, 1.0, 50.0)),
            saturation=float(np.clip(self.saturation, 0.0, 1.0)),
            output_db=float(np.clip(self.output_db, -12.0, 0.0)),
        )


def _one_pole_lowpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Small dependency-free one-pole low-pass filter."""
    cutoff = float(np.clip(cutoff_hz, 20.0, sample_rate * 0.45))
    alpha = np.exp(-2.0 * np.pi * cutoff / sample_rate)
    out = np.empty_like(signal, dtype=np.float64)
    previous = 0.0
    for index, value in enumerate(signal):
        previous = (1.0 - alpha) * float(value) + alpha * previous
        out[index] = previous
    return out


def _phase_from_frequency(frequency: np.ndarray, sample_rate: int) -> np.ndarray:
    """Integrate instantaneous frequency into continuous oscillator phase."""
    increments = (frequency[:-1] + frequency[1:]) * (0.5 / sample_rate)
    cycles = np.concatenate(([0.0], np.cumsum(increments)))
    return 2.0 * np.pi * cycles


def generate_kick(params: KickParams, seed: int = 0) -> np.ndarray:
    """Synthesize a single normalized mono kick as float32 in [-1, 1]."""
    p = params.validated()
    sr = p.sample_rate
    count = max(2, int(round(p.duration * sr)))
    time = np.arange(count, dtype=np.float64) / sr
    rng = np.random.default_rng(seed)

    # Exponential pitch fall: fast initial descent, then a stable low-frequency body.
    tau = p.pitch_decay_ms / 1_000.0
    pitch_progress = np.exp(-np.power(time / max(tau, 1e-5), p.pitch_curve))
    instantaneous_frequency = p.end_freq + (p.start_freq - p.end_freq) * pitch_progress
    phase = _phase_from_frequency(instantaneous_frequency, sr)

    # Fundamental body and restrained upper partials preserve a rounded low-end.
    body_wave = np.sin(phase)
    harmonic_wave = np.zeros(count, dtype=np.float64)
    partial_levels = (0.34, 0.20, 0.12, 0.075, 0.045, 0.025)
    for partial_number, base_level in enumerate(partial_levels, start=2):
        level = base_level / (partial_number ** (p.harmonic_tilt - 1.0))
        harmonic_wave += level * np.sin(phase * partial_number)
    harmonic_wave *= p.harmonics

    # A quiet, slightly lower sub oscillator adds chest weight without a hard click.
    sub_frequency = max(27.0, p.end_freq * 0.78)
    sub_wave = np.sin(2.0 * np.pi * sub_frequency * time)

    attack_seconds = max(p.attack_ms / 1_000.0, 1.0 / sr)
    attack_envelope = 1.0 - np.exp(-time / attack_seconds)
    decay_seconds = max(p.decay_ms / 1_000.0, 0.02)
    amplitude_envelope = attack_envelope * np.exp(-time / decay_seconds)

    # Transient: a small tonal tick plus a short, band-limited noise impulse.
    transient_env = np.exp(-time / max(p.transient_decay_ms / 1_000.0, 1.0 / sr))
    transient_env *= np.exp(-time / 0.035)
    click_tone = np.sin(2.0 * np.pi * p.transient_tone * time)
    noise = rng.standard_normal(count)
    noise_hp = noise - _one_pole_lowpass(noise, 650.0, sr)
    noise_bp = _one_pole_lowpass(noise_hp, 7_000.0, sr)
    noise_bp /= max(float(np.max(np.abs(noise_bp))), 1e-9)
    click = (0.62 * click_tone + 0.38 * noise_bp) * transient_env

    signal = amplitude_envelope * (
        p.body * body_wave
        + p.sub * sub_wave
        + harmonic_wave
    )
    signal += p.transient * click

    # Soft saturation adds density; no hard clipping stage is used.
    drive = 1.0 + p.saturation * 8.0
    signal = np.tanh(signal * drive) / np.tanh(drive)

    peak = float(np.max(np.abs(signal)))
    if peak > 1e-12:
        target_peak = 10.0 ** (p.output_db / 20.0)
        signal *= target_peak / peak

    return np.asarray(np.clip(signal, -1.0, 1.0), dtype=np.float32)


def params_to_dict(params: KickParams) -> dict[str, Any]:
    return asdict(params.validated())


def params_from_dict(values: dict[str, Any]) -> KickParams:
    allowed = set(KickParams.__dataclass_fields__.keys())
    clean = {key: value for key, value in values.items() if key in allowed}
    return KickParams(**clean).validated()


def make_variation(base: KickParams, amount: float, rng: np.random.Generator) -> KickParams:
    """Create a musically bounded variation around a base kick."""
    p = base.validated()
    amount = float(np.clip(amount, 0.0, 1.0))

    def vary(value: float, spread: float, low: float, high: float) -> float:
        return float(np.clip(value * (1.0 + rng.uniform(-spread, spread) * amount), low, high))

    return replace(
        p,
        start_freq=vary(p.start_freq, 0.18, 55.0, 360.0),
        end_freq=vary(p.end_freq, 0.12, 28.0, 100.0),
        pitch_decay_ms=vary(p.pitch_decay_ms, 0.35, 8.0, 220.0),
        pitch_curve=vary(p.pitch_curve, 0.25, 0.5, 2.3),
        body=vary(p.body, 0.16, 0.35, 1.0),
        sub=vary(p.sub, 0.40, 0.0, 0.65),
        harmonics=vary(p.harmonics, 0.55, 0.0, 0.55),
        decay_ms=vary(p.decay_ms, 0.22, 140.0, 1_200.0),
        transient=vary(p.transient, 0.50, 0.0, 0.45),
        saturation=vary(p.saturation, 0.50, 0.0, 0.55),
    ).validated()


def analyze_kick(audio: np.ndarray, sample_rate: int = 48_000) -> dict[str, float]:
    """Return useful engineering measurements for a generated kick."""
    data = np.asarray(audio, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return {"peak_dbfs": -120.0, "rms_dbfs": -120.0, "crest_db": 0.0, "centroid_hz": 0.0}
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data))))
    spectrum = np.abs(np.fft.rfft(data * np.hanning(data.size)))
    frequencies = np.fft.rfftfreq(data.size, 1.0 / sample_rate)
    magnitude_sum = float(np.sum(spectrum))
    centroid = float(np.sum(frequencies * spectrum) / magnitude_sum) if magnitude_sum > 0 else 0.0
    db = lambda value: 20.0 * np.log10(max(value, 1e-9))
    return {
        "peak_dbfs": db(peak),
        "rms_dbfs": db(rms),
        "crest_db": db(peak / max(rms, 1e-9)),
        "centroid_hz": centroid,
    }
