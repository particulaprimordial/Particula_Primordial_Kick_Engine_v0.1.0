"""Rebuild CSV/JSON data and static paper figures from bundled factory WAVs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kick_analysis import analyze_audio, spectrogram_data  # noqa: E402

FACTORY = ROOT / "examples" / "factory-kicks"
DATA = ROOT / "research" / "data"
FIGURES = ROOT / "research" / "figures"
DATA.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#fbfaf7",
    "axes.facecolor": "#fbfaf7",
    "axes.edgecolor": "#b9b6ad",
    "axes.labelcolor": "#33342f",
    "xtick.color": "#55564f",
    "ytick.color": "#55564f",
    "text.color": "#262722",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titleweight": "semibold",
})


def main() -> None:
    rows = []
    loaded = []
    for path in sorted(FACTORY.glob("*.wav")):
        sample_rate, audio = wavfile.read(path)
        metrics = analyze_audio(audio, sample_rate)
        band = metrics["band_energy_pct"]
        decay = metrics["decay"]
        row = {
            "sample": path.stem.replace("PP_Factory_", "").replace("_", " "),
            "file": path.name,
            "sample_rate_hz": sample_rate,
            "duration_ms": round(metrics["duration_s"] * 1000, 3),
            "peak_dbfs": round(metrics["peak_dbfs"], 4),
            "rms_dbfs": round(metrics["rms_dbfs"], 4),
            "crest_factor_db": round(metrics["crest_db"], 4),
            "spectral_centroid_hz": round(metrics["spectral_centroid_hz"], 3),
            "dominant_frequency_hz": round(metrics["dominant_frequency_hz"], 3),
            "energy_20_60_hz_pct": round(band["20_60_hz"], 4),
            "energy_60_120_hz_pct": round(band["60_120_hz"], 4),
            "energy_120_250_hz_pct": round(band["120_250_hz"], 4),
            "energy_250_1000_hz_pct": round(band["250_1000_hz"], 4),
            "energy_1k_5k_hz_pct": round(band["1k_5k_hz"], 4),
            "energy_5k_12k_hz_pct": round(band["5k_12k_hz"], 4),
            "time_to_minus_6db_ms": round(decay["t_minus_6_db_ms"], 3) if decay["t_minus_6_db_ms"] is not None else None,
            "time_to_minus_20db_ms": round(decay["t_minus_20_db_ms"], 3) if decay["t_minus_20_db_ms"] is not None else None,
            "time_to_minus_40db_ms": round(decay["t_minus_40_db_ms"], 3) if decay["t_minus_40_db_ms"] is not None else None,
        }
        rows.append(row)
        loaded.append((path, sample_rate, audio, metrics))

    if not rows:
        raise SystemExit(f"No factory WAV files found in {FACTORY}")

    csv_path = DATA / "factory_measurements.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (DATA / "factory_measurements.json").write_text(json.dumps({
        "project": "Particula Primordial Kick Engine",
        "dataset": "Factory kick WAVs bundled with version 0.2.0",
        "method": "Computed from WAV audio using kick_analysis.py",
        "n_samples": len(rows),
        "measurements": rows,
        "interpretation_note": "Descriptive digital-signal measures only; no subjective listening scores or quality rankings are included.",
    }, indent=2), encoding="utf-8")

    labels = [r["sample"] for r in rows]
    centroids = [r["spectral_centroid_hz"] for r in rows]
    decay20 = [r["time_to_minus_20db_ms"] if r["time_to_minus_20db_ms"] is not None else np.nan for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax.scatter(centroids, decay20, s=90, color="#a78a4b", edgecolor="#292a25", linewidth=.8, zorder=3)
    for x, y, label in zip(centroids, decay20, labels):
        ax.annotate(label, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Spectral centroid (Hz)")
    ax.set_ylabel("Time to -20 dB (ms)")
    ax.set_title("Factory specimens: spectral centroid vs. envelope decay")
    ax.grid(True, alpha=.22)
    fig.savefig(FIGURES / "factory_measurement_map.png", dpi=180)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 7.0), constrained_layout=True)
    x = np.arange(len(rows))
    low_20_60 = [r["energy_20_60_hz_pct"] for r in rows]
    ax1.bar(x, low_20_60, width=.56, color="#526052")
    ax1.set_xticks(x, labels, rotation=10, ha="right")
    ax1.set_ylabel("Share (%)")
    ax1.set_title("20–60 Hz band — dominant low-frequency energy")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=.2)

    bands = [
        ("60–120 Hz", "energy_60_120_hz_pct", "#c3a66a"),
        ("120–250 Hz", "energy_120_250_hz_pct", "#83927e"),
        ("250–1,000 Hz", "energy_250_1000_hz_pct", "#777c8a"),
    ]
    width = .23
    for idx, (label, key, color) in enumerate(bands):
        values = [r[key] for r in rows]
        ax2.bar(x + (idx - 1) * width, values, width=width, color=color, label=label)
    ax2.set_xticks(x, labels, rotation=10, ha="right")
    ax2.set_ylabel("Share (%)")
    ax2.set_title("Upper low-frequency and low-mid bands — zoomed scale")
    ax2.legend(frameon=False, ncol=3)
    ax2.grid(axis="y", alpha=.2)
    fig.suptitle("Factory WAV spectral-energy distribution", fontweight="semibold")
    fig.savefig(FIGURES / "factory_band_energy.png", dpi=180)
    plt.close(fig)

    # The first included sample is used as a reproducible visual example.
    path, sr, audio, metrics = loaded[0]
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if np.issubdtype(np.asarray(audio).dtype, np.integer):
        info = np.iinfo(np.asarray(audio).dtype)
        mono = mono / float(max(abs(info.min), abs(info.max)))
    time_ms = np.arange(len(mono)) / sr * 1000
    stride = max(1, len(mono) // 6000)
    freqs = metrics["spectrum"]["frequency_hz"]
    spec_db = metrics["spectrum"]["magnitude_db"]
    mask = (freqs >= 20) & (freqs <= min(12000, sr / 2))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.2), constrained_layout=True)
    ax1.plot(time_ms[::stride], mono[::stride], color="#a78a4b", linewidth=.7)
    ax1.set_xlim(0, time_ms[-1])
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Time-domain waveform — {path.stem}")
    ax1.grid(True, alpha=.2)
    ax2.plot(freqs[mask], spec_db[mask], color="#526052", linewidth=1.0)
    ax2.set_xscale("log")
    ax2.set_xlim(20, min(12000, sr / 2))
    ax2.set_ylim(-90, 3)
    ax2.set_xlabel("Frequency (Hz, log scale)")
    ax2.set_ylabel("Relative magnitude (dB)")
    ax2.set_title("Whole-file windowed magnitude spectrum")
    ax2.grid(True, which="both", alpha=.18)
    fig.savefig(FIGURES / "factory_reference_signal.png", dpi=180)
    plt.close(fig)

    sf, stime, sdb = spectrogram_data(mono, sr)
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(stime * 1000, sf, sdb, shading="auto", cmap="cividis", vmin=-80, vmax=0)
    ax.set_yscale("log")
    ax.set_ylim(20, min(8000, sr / 2))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz, log scale)")
    ax.set_title("STFT spectrogram — first factory specimen")
    fig.colorbar(mesh, ax=ax, label="Relative magnitude (dB)")
    fig.savefig(FIGURES / "factory_reference_spectrogram.png", dpi=180)
    plt.close(fig)

    print(f"Generated {len(rows)} rows -> {csv_path}")
    print(f"Generated paper figures -> {FIGURES}")


if __name__ == "__main__":
    main()
