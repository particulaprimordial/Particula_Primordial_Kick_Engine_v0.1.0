"""Regenerate factory WAV specimens and their parameter/measurement manifest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kick_analysis import analyze_audio  # noqa: E402
from kick_engine import generate_kick, params_to_dict  # noqa: E402
from presets import PRESETS  # noqa: E402

OUT = ROOT / "examples" / "factory-kicks"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sample_records = []
    for index, (preset_name, params) in enumerate(PRESETS.items(), start=1):
        audio = generate_kick(params, seed=1716 + index)
        file_name = f"PP_Factory_{index:02d}_{preset_name.split(' · ', 1)[1].replace(' ', '_')}.wav"
        wavfile.write(OUT / file_name, params.sample_rate, np.asarray(audio, dtype=np.float32))
        measured = analyze_audio(audio, params.sample_rate)
        sample_records.append({
            "preset": preset_name,
            "file": file_name,
            "parameters": params_to_dict(params),
            "analysis": {
                "peak_dbfs": measured["peak_dbfs"],
                "rms_dbfs": measured["rms_dbfs"],
                "crest_db": measured["crest_db"],
                "spectral_centroid_hz": measured["spectral_centroid_hz"],
                "band_energy_pct": measured["band_energy_pct"],
                "decay": measured["decay"],
            },
        })
    manifest = {
        "project": "Particula Primordial Kick Engine",
        "version": "0.2.0",
        "sample_rate": 48000,
        "format": "32-bit float mono WAV",
        "generation_seed_rule": "seed = 1716 + 1-based preset index",
        "samples": sample_records,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Regenerated {len(sample_records)} factory WAVs in {OUT}")


if __name__ == "__main__":
    main()
