import numpy as np

from kick_engine import KickParams, analyze_kick, generate_kick, make_variation


def test_generation_is_deterministic_and_finite():
    params = KickParams(duration=0.25)
    first = generate_kick(params, seed=44)
    second = generate_kick(params, seed=44)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert np.max(np.abs(first)) <= 1.0


def test_generation_has_expected_length():
    params = KickParams(duration=0.5, sample_rate=48_000)
    audio = generate_kick(params)
    assert len(audio) == 24_000


def test_variation_is_bounded_and_analyzable():
    rng = np.random.default_rng(10)
    varied = make_variation(KickParams(), 0.5, rng)
    audio = generate_kick(varied, seed=3)
    stats = analyze_kick(audio, varied.sample_rate)
    assert 28 <= varied.end_freq <= 100
    assert -120 <= stats["peak_dbfs"] <= 0
    assert stats["centroid_hz"] > 0


def test_analysis_module_reports_real_signal_features():
    from kick_analysis import analyze_audio

    sample_rate = 48_000
    time = np.arange(int(sample_rate * 0.8)) / sample_rate
    audio = (0.7 * np.sin(2 * np.pi * 50 * time) * np.exp(-time / 0.35)).astype(np.float32)
    measured = analyze_audio(audio, sample_rate)

    assert measured["sample_rate_hz"] == sample_rate
    assert measured["duration_s"] == 0.8
    assert abs(measured["dominant_frequency_hz"] - 50) <= 1.0
    assert measured["band_energy_pct"]["20_60_hz"] > 90
    assert measured["decay"]["t_minus_6_db_ms"] is not None
    ridge = measured["ridge"]["frequency_hz"]
    valid = ridge[np.isfinite(ridge)]
    assert valid.size > 0
    assert abs(float(np.median(valid)) - 50) < 3


def test_analysis_accepts_stereo_integer_pcm():
    from kick_analysis import analyze_audio

    sample_rate = 44_100
    time = np.arange(int(sample_rate * 0.25)) / sample_rate
    mono = (np.sin(2 * np.pi * 60 * time) * 20_000).astype(np.int16)
    stereo = np.column_stack([mono, mono])
    measured = analyze_audio(stereo, sample_rate)
    assert measured["samples"] == len(mono)
    assert measured["peak_dbfs"] <= 0
    assert measured["duration_s"] == 0.25


def test_analysis_centers_unsigned_eight_bit_pcm():
    from kick_analysis import analyze_audio

    sample_rate = 8_000
    time = np.arange(sample_rate // 4) / sample_rate
    # 8-bit PCM WAV is unsigned and centered around 128.
    pcm = np.clip(128 + 90 * np.sin(2 * np.pi * 100 * time), 0, 255).astype(np.uint8)
    measured = analyze_audio(pcm, sample_rate)
    assert measured["peak_dbfs"] < 0
    assert measured["rms_dbfs"] < 0
