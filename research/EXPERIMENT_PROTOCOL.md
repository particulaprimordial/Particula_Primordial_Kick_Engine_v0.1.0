# Experimental Protocol — Kick Engine

## Purpose

Build a repeatable evidence trail for sound-design decisions. This is an engineering and listening protocol, not a clinical or psychoacoustic study.

## Experiment A — One-parameter sensitivity

**Question:** What changes in the signal when one synthesis parameter changes and all others remain fixed?

1. Choose one reference preset and save its JSON parameter file.
2. Select one variable: start frequency, end frequency, pitch-decay time, body, sub, harmonics, transient, amplitude decay or saturation.
3. Render at least three levels: low, reference and high. Keep all other settings unchanged.
4. Use a fixed random seed for renders containing the noise transient.
5. Export WAV and analysis JSON for every render.
6. Compare waveform, spectrum, spectrogram, RMS envelope, low-frequency ridge and band-energy shares.
7. Listen at matched playback level, first in solo and then in the same kick-bass loop.
8. Record objective measurements and subjective judgments in separate columns.

## Experiment B — Kick/bass coexistence

1. Choose one bassline and keep its MIDI/audio, level, processing and arrangement fixed.
2. Compare candidate kicks with the same pattern and the same playback chain.
3. Record peak and RMS values for the kick and combined loop; do not treat these as loudness measurements.
4. Inspect spectral overlap and listen for masking, uncontrolled low-frequency buildup and loss of kick definition.
5. Prefer the sample that serves the groove, not the sample with the largest waveform.

## Experiment C — Variation-family audit

1. Choose a reference kick and variation depth.
2. Record the random seed and number of variants.
3. Generate a family and export the ZIP manifest.
4. Sort candidates by objective measurements only to organize auditioning.
5. Select finalists by listening in the actual musical context.
6. Preserve the original seed and manifest so a selected candidate can be reproduced.

## Controls and reporting

- Report sample rate, bit depth/format, duration, parameter values and random seed.
- Match playback level before subjective A/B comparisons.
- Use the same monitoring chain and musical loop within a comparison.
- Do not infer perceived warmth, physical impact or danceability from spectral values alone.
- Do not call sample peak or RMS `LUFS` or claim ITU compliance.
- Include null/failed estimates rather than inventing a value when a measurement cannot be computed.

## Suggested listening-rating fields

Use a 1–7 scale and define anchors before collecting ratings: perceived low-end weight, attack definition, smoothness, translation on small speakers, fit with bassline, groove fit, and overall preference. These ratings are subjective observations, not objective signal properties.
