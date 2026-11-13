"""Factory presets for the Particula Primordial Kick Engine."""
from kick_engine import KickParams

PRESETS: dict[str, KickParams] = {
    "01 · Rounded Chest": KickParams(
        duration=1.40, start_freq=142, end_freq=47, pitch_decay_ms=48,
        body=0.94, sub=0.28, harmonics=0.10, harmonic_tilt=1.7,
        attack_ms=2.5, decay_ms=560, transient=0.07,
        transient_tone=1_650, transient_decay_ms=9, saturation=0.10,
    ),
    "02 · Velvet Pressure": KickParams(
        duration=1.55, start_freq=126, end_freq=43, pitch_decay_ms=58,
        body=0.96, sub=0.34, harmonics=0.08, harmonic_tilt=1.9,
        attack_ms=3.5, decay_ms=640, transient=0.045,
        transient_tone=1_250, transient_decay_ms=11, saturation=0.08,
    ),
    "03 · Controlled Punch": KickParams(
        duration=1.05, start_freq=176, end_freq=52, pitch_decay_ms=31,
        body=0.84, sub=0.18, harmonics=0.22, harmonic_tilt=1.2,
        attack_ms=1.2, decay_ms=390, transient=0.20,
        transient_tone=2_800, transient_decay_ms=5, saturation=0.17,
    ),
    "04 · Organic Pulse": KickParams(
        duration=1.75, start_freq=118, end_freq=49, pitch_decay_ms=72,
        pitch_curve=0.86, body=0.91, sub=0.20, harmonics=0.17,
        harmonic_tilt=1.5, attack_ms=4.0, decay_ms=720,
        transient=0.09, transient_tone=1_850, transient_decay_ms=13,
        saturation=0.06,
    ),
    "05 · Deep Ritual": KickParams(
        duration=2.00, start_freq=154, end_freq=39, pitch_decay_ms=52,
        body=0.89, sub=0.42, harmonics=0.07, harmonic_tilt=2.0,
        attack_ms=3.0, decay_ms=820, transient=0.05,
        transient_tone=1_400, transient_decay_ms=10, saturation=0.11,
    ),
}
