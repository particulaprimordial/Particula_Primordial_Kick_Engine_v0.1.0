from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile

from kick_analysis import analyze_audio, spectrogram_data
from kick_engine import KickParams, generate_kick, make_variation, params_to_dict
from presets import PRESETS

ROOT = Path(__file__).parent
FACTORY_DIR = ROOT / "examples" / "factory-kicks"

st.set_page_config(
    page_title="Particula Primordial · Kick Engine",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#e9e8e3; --muted:#92928d; --gold:#d4b36a; --line:rgba(233,232,227,.12); }
    .stApp { background: radial-gradient(ellipse at 50% -15%, #262722 0%, #171816 54%, #111210 100%); color:var(--ink); }
    html, body, [class*="css"] { font-family:'Manrope', sans-serif; }
    .block-container { max-width:1400px; padding-top:2.1rem; padding-bottom:3.5rem; }
    h1,h2,h3 { color:var(--ink)!important; letter-spacing:-.045em; }
    h1 { font-size:2.3rem!important; font-weight:800!important; }
    h2 { font-size:1.25rem!important; font-weight:700!important; }
    p, label, .stMarkdown { color:var(--ink); }
    [data-testid="stCaptionContainer"], .muted { color:var(--muted)!important; }
    .eyebrow { font:500 10px 'DM Mono',monospace; color:var(--gold); letter-spacing:.22em; text-transform:uppercase; }
    .hero { border-bottom:1px solid var(--line); padding:0 0 1.3rem 0; margin-bottom:1.6rem; }
    .hero-sub { color:#aaa9a1; font-size:.91rem; margin-top:-.5rem; }
    .panel { background:rgba(255,255,255,.025); border:1px solid var(--line); border-radius:15px; padding:1.05rem 1.15rem; }
    .metric-label { font:10px 'DM Mono',monospace; color:var(--muted); letter-spacing:.12em; text-transform:uppercase; }
    .metric-value { font:500 1.22rem 'DM Mono',monospace; color:var(--ink); margin-top:.25rem; }
    .section-tag { font:500 10px 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; color:var(--gold); margin: .3rem 0 .8rem; }
    div[data-testid="stTabs"] button { color:#a8a8a1; font-size:.82rem; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--gold); }
    .stButton button, .stDownloadButton button { border-radius:9px; min-height:2.6rem; font-weight:700; letter-spacing:.04em; }
    .stButton button[kind="primary"] { background:var(--gold); color:#171816; border:1px solid var(--gold); }
    .stDownloadButton button { background:rgba(212,179,106,.10); border:1px solid rgba(212,179,106,.42); color:var(--ink); }
    div[data-baseweb="select"] > div, .stNumberInput input { background:#20211e; border-color:var(--line); }
    .stSlider [data-testid="stTickBar"] { display:none; }
    [data-testid="stAudio"] { width:100%; }
    hr { border-color:var(--line); }
    footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    for key, value in params_to_dict(KickParams()).items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("preset_choice", "01 · Rounded Chest")
    st.session_state.setdefault("audio", None)
    st.session_state.setdefault("audio_params", None)
    st.session_state.setdefault("batch_zip", None)
    st.session_state.setdefault("batch_count", 0)


def apply_selected_preset() -> None:
    preset = PRESETS[st.session_state.preset_choice]
    for key, value in params_to_dict(preset).items():
        st.session_state[key] = value
    st.session_state.audio = None
    st.session_state.audio_params = None


def current_params() -> KickParams:
    values = {key: st.session_state[key] for key in KickParams.__dataclass_fields__}
    return KickParams(**values).validated()


def wav_bytes(audio: np.ndarray, sample_rate: int = 48_000) -> bytes:
    stream = io.BytesIO()
    wavfile.write(stream, sample_rate, np.asarray(audio, dtype=np.float32))
    return stream.getvalue()


def plot_layout(fig: go.Figure, height: int = 250, x_title: str = "", y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=32),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.015)",
        font=dict(color="#a6a69e", size=10), showlegend=False,
        xaxis=dict(title=x_title, gridcolor="rgba(255,255,255,.07)", zeroline=False),
        yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.2)"),
    )
    return fig


def plot_waveform(metrics: dict) -> go.Figure:
    x = metrics["waveform"]["time_ms"]
    y = metrics["waveform"]["amplitude"]
    stride = max(1, int(np.ceil(len(y) / 5000)))
    fig = go.Figure(go.Scatter(x=x[::stride], y=y[::stride], mode="lines", line=dict(color="#d4b36a", width=1.0)))
    fig.update_yaxes(range=[-1.05, 1.05])
    return plot_layout(fig, 235, "TIME · ms", "AMPLITUDE")


def plot_spectrum(metrics: dict) -> go.Figure:
    f = metrics["spectrum"]["frequency_hz"]
    db = metrics["spectrum"]["magnitude_db"]
    mask = (f >= 20) & (f <= 12_000)
    fig = go.Figure(go.Scatter(x=f[mask], y=db[mask], mode="lines", line=dict(color="#9cae9b", width=1.15), fill="tozeroy", fillcolor="rgba(156,174,155,.07)"))
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[-90, 3])
    return plot_layout(fig, 235, "FREQUENCY · Hz · LOG SCALE", "RELATIVE LEVEL · dB")


def plot_spectrogram(audio: np.ndarray, sample_rate: int) -> go.Figure:
    f, t, db = spectrogram_data(audio, sample_rate)
    fig = go.Figure(go.Heatmap(
        x=t * 1000.0, y=f, z=db, zmin=-80, zmax=0,
        colorscale=[[0, "#161815"], [0.45, "#526052"], [0.78, "#a58a50"], [1, "#f0dfaa"]],
        colorbar=dict(title="dB", thickness=9, len=.78),
    ))
    fig.update_layout(height=330, margin=dict(l=8, r=8, t=8, b=32), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.015)", font=dict(color="#a6a69e", size=10))
    fig.update_xaxes(title="TIME · ms", gridcolor="rgba(255,255,255,.07)")
    fig.update_yaxes(title="FREQUENCY · Hz", type="log", gridcolor="rgba(255,255,255,.07)")
    return fig


def plot_envelope(metrics: dict) -> go.Figure:
    env = metrics["envelope"]
    fig = go.Figure(go.Scatter(x=env["time_ms"], y=env["rms_db"], mode="lines", line=dict(color="#d4b36a", width=1.5), fill="tozeroy", fillcolor="rgba(212,179,106,.08)"))
    fig.update_yaxes(range=[-80, 2])
    return plot_layout(fig, 220, "TIME · ms", "RMS ENVELOPE · dB RELATIVE")


def plot_pitch_ridge(metrics: dict) -> go.Figure:
    ridge = metrics["ridge"]
    t, f, c = ridge["time_s"], ridge["frequency_hz"], ridge["confidence"]
    valid = np.isfinite(f) & (c >= 0.10)
    fig = go.Figure(go.Scatter(x=t[valid] * 1000.0, y=f[valid], mode="lines+markers", marker=dict(size=4), line=dict(color="#9cae9b", width=1.5), name="Estimated ridge"))
    fig.update_yaxes(range=[20, 260])
    return plot_layout(fig, 220, "TIME · ms", "ESTIMATED LOW-FREQUENCY RIDGE · Hz")


@st.cache_data(show_spinner=False)
def factory_dataset() -> list[dict]:
    rows = []
    for path in sorted(FACTORY_DIR.glob("*.wav")):
        sr, audio = wavfile.read(path)
        m = analyze_audio(audio, sr)
        decay = m["decay"]
        rows.append({
            "sample": path.stem.replace("PP_Factory_", "").replace("_", " "),
            "duration_ms": round(m["duration_s"] * 1000, 1),
            "peak_dBFS": round(m["peak_dbfs"], 2),
            "RMS_dBFS": round(m["rms_dbfs"], 2),
            "crest_dB": round(m["crest_db"], 2),
            "centroid_Hz": round(m["spectral_centroid_hz"], 1),
            "20–120_Hz_%": round(m["band_energy_pct"]["20_60_hz"] + m["band_energy_pct"]["60_120_hz"], 1),
            "120–1000_Hz_%": round(m["band_energy_pct"]["120_250_hz"] + m["band_energy_pct"]["250_1000_hz"], 1),
            "t_-20dB_ms": round(decay["t_minus_20_db_ms"], 1) if decay["t_minus_20_db_ms"] is not None else None,
            "file": path.name,
            "sample_rate": sr,
        })
    return rows


def json_safe_metrics(metrics: dict) -> dict:
    return {
        key: value for key, value in metrics.items()
        if key not in {"waveform", "spectrum", "envelope", "ridge"}
    }


init_state()

st.markdown(
    '<div class="hero"><div class="eyebrow">PARTICULA PRIMORDIAL / AUDIO SYSTEMS 001</div>'
    '<h1>Kick Engine <span style="color:#d4b36a">/ 01</span></h1>'
    '<div class="hero-sub">Procedural low-frequency instrument · synthesize, measure, document and develop original kick drums</div></div>',
    unsafe_allow_html=True,
)

(tab_synth, tab_lab, tab_variations, tab_research, tab_presets) = st.tabs([
    "01  SYNTHESIZER", "02  KICK LABORATORY", "03  VARIATION LAB", "04  RESEARCH NOTEBOOK", "05  PRESETS & METHOD"
])

with tab_synth:
    top_left, top_right = st.columns([1.1, 1.0], gap="large")
    with top_left:
        st.markdown('<div class="section-tag">Starting point</div>', unsafe_allow_html=True)
        st.selectbox("FACTORY PRESET", list(PRESETS.keys()), key="preset_choice", on_change=apply_selected_preset)
        st.caption("Choose a starting character, then shape the sound with the controls below.")
    with top_right:
        st.markdown('<div class="section-tag">Output format</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><span class="metric-label">RENDER FORMAT</span><div class="metric-value">48 kHz · 32-bit float · Mono WAV</div><span class="muted">Designed for clean import into Ableton Live.</span></div>', unsafe_allow_html=True)

    st.divider()
    a, b, c, d = st.columns(4, gap="large")
    with a:
        st.markdown('<div class="section-tag">01 / Pitch contour</div>', unsafe_allow_html=True)
        st.slider("Start frequency · Hz", 55.0, 360.0, key="start_freq", step=1.0)
        st.slider("End frequency · Hz", 28.0, 100.0, key="end_freq", step=1.0)
        st.slider("Pitch decay · ms", 8.0, 220.0, key="pitch_decay_ms", step=1.0)
        st.slider("Pitch curve", 0.5, 2.3, key="pitch_curve", step=0.05)
    with b:
        st.markdown('<div class="section-tag">02 / Low-end body</div>', unsafe_allow_html=True)
        st.slider("Body", 0.0, 1.0, key="body", step=0.01)
        st.slider("Sub weight", 0.0, 1.0, key="sub", step=0.01)
        st.slider("Harmonic content", 0.0, 0.7, key="harmonics", step=0.01)
        st.slider("Harmonic tilt", 0.5, 3.0, key="harmonic_tilt", step=0.05)
    with c:
        st.markdown('<div class="section-tag">03 / Impact</div>', unsafe_allow_html=True)
        st.slider("Transient amount", 0.0, 0.6, key="transient", step=0.01)
        st.slider("Transient tone · Hz", 500.0, 8_000.0, key="transient_tone", step=50.0)
        st.slider("Transient decay · ms", 1.0, 50.0, key="transient_decay_ms", step=1.0)
        st.slider("Attack · ms", 0.1, 25.0, key="attack_ms", step=0.1)
    with d:
        st.markdown('<div class="section-tag">04 / Envelope & tone</div>', unsafe_allow_html=True)
        st.slider("Duration · seconds", 0.12, 2.0, key="duration", step=0.01)
        st.slider("Decay · ms", 100.0, 1_500.0, key="decay_ms", step=10.0)
        st.slider("Saturation", 0.0, 0.7, key="saturation", step=0.01)
        st.slider("Output ceiling · dBFS", -12.0, 0.0, key="output_db", step=0.5)

    render_col, note_col = st.columns([1, 2.3], gap="large", vertical_alignment="center")
    with render_col:
        if st.button("GENERATE KICK", type="primary", use_container_width=True):
            p = current_params()
            st.session_state.audio = generate_kick(p, seed=17)
            st.session_state.audio_params = params_to_dict(p)
    with note_col:
        st.caption("Deterministic rendering: identical parameters and seed reproduce the same audio. Use Kick Laboratory to inspect measured results.")

    if st.session_state.audio is not None:
        audio = st.session_state.audio
        p = KickParams(**st.session_state.audio_params)
        st.audio(wav_bytes(audio, p.sample_rate), format="audio/wav")
        metrics = analyze_audio(audio, p.sample_rate)
        m1, m2, m3, m4 = st.columns(4)
        for col, label, value in [
            (m1, "PEAK", f"{metrics['peak_dbfs']:.1f} dBFS"),
            (m2, "RMS", f"{metrics['rms_dbfs']:.1f} dBFS"),
            (m3, "CREST FACTOR", f"{metrics['crest_db']:.1f} dB"),
            (m4, "SPECTRAL CENTROID", f"{metrics['spectral_centroid_hz']:.0f} Hz"),
        ]:
            with col:
                st.markdown(f'<div class="panel"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        filename = f"PP_Kick_{int(p.start_freq)}-{int(p.end_freq)}Hz.wav"
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("DOWNLOAD WAV · 32-BIT FLOAT", wav_bytes(audio, p.sample_rate), file_name=filename, mime="audio/wav", use_container_width=True)
        with dl2:
            st.download_button("DOWNLOAD PRESET · JSON", json.dumps(params_to_dict(p), indent=2), file_name=filename.replace(".wav", ".json"), mime="application/json", use_container_width=True)
    else:
        st.markdown('<div class="panel"><div class="metric-label">READY TO SYNTHESIZE</div><div class="metric-value">Shape the pitch, weight and transient. Then generate your first kick.</div><p class="muted">Start with the low-frequency body. Add only enough transient and harmonics to translate the kick in a mix.</p></div>', unsafe_allow_html=True)

with tab_lab:
    st.markdown("## Kick Laboratory")
    st.caption("Measurement and visualization module. Select a rendered kick, a factory WAV or your own WAV. Every graph below is computed from the selected audio file.")
    source_options = ["Factory sample", "Upload WAV"]
    if st.session_state.audio is not None:
        source_options.insert(0, "Last rendered kick")
    source = st.radio("ANALYSIS SOURCE", source_options, horizontal=True)
    selected_audio = None
    selected_sr = 48_000
    selected_name = ""
    raw_playback = None

    if source == "Last rendered kick":
        selected_audio = st.session_state.audio
        selected_sr = int(st.session_state.audio_params["sample_rate"])
        selected_name = "Current synthesizer render"
        raw_playback = wav_bytes(selected_audio, selected_sr)
    elif source == "Factory sample":
        factory_files = sorted(FACTORY_DIR.glob("*.wav"))
        choice = st.selectbox("FACTORY WAV", factory_files, format_func=lambda x: x.stem.replace("PP_Factory_", "").replace("_", " "))
        selected_sr, selected_audio = wavfile.read(choice)
        selected_name = choice.name
        raw_playback = choice.read_bytes()
    else:
        uploaded = st.file_uploader("Upload a WAV file", type=["wav"], help="Mono or stereo WAV. Integer PCM and floating-point WAV are supported.")
        if uploaded is not None:
            try:
                selected_sr, selected_audio = wavfile.read(io.BytesIO(uploaded.getvalue()))
                selected_name = uploaded.name
                raw_playback = uploaded.getvalue()
            except Exception as exc:
                st.error(f"Could not read this WAV file: {exc}")

    if selected_audio is not None:
        try:
            measured = analyze_audio(selected_audio, int(selected_sr))
            st.markdown(f'<div class="panel"><span class="metric-label">CURRENT SPECIMEN</span><div class="metric-value">{selected_name}</div><span class="muted">{measured["sample_rate_hz"]:,} Hz · {measured["duration_s"] * 1000:.1f} ms · {measured["samples"]:,} samples</span></div>', unsafe_allow_html=True)
            st.audio(raw_playback, format="audio/wav")
            mcols = st.columns(5)
            metric_items = [
                ("PEAK", f"{measured['peak_dbfs']:.2f} dBFS"),
                ("RMS", f"{measured['rms_dbfs']:.2f} dBFS"),
                ("CREST", f"{measured['crest_db']:.2f} dB"),
                ("CENTROID", f"{measured['spectral_centroid_hz']:.1f} Hz"),
                ("DOMINANT BIN", f"{measured['dominant_frequency_hz']:.1f} Hz"),
            ]
            for col, (label, value) in zip(mcols, metric_items):
                with col:
                    st.markdown(f'<div class="panel"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-tag">01 / Time domain</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_waveform(measured), use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div class="section-tag">02 / Time-frequency domain · STFT</div>', unsafe_allow_html=True)
            st.caption("The spectrogram shows how spectral energy changes over time. Its time/frequency resolution depends on the analysis window.")
            st.plotly_chart(plot_spectrogram(selected_audio, int(selected_sr)), use_container_width=True, config={"displayModeBar": False})
            pcol, ecol = st.columns(2, gap="large")
            with pcol:
                st.markdown('<div class="section-tag">03 / Spectrum</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_spectrum(measured), use_container_width=True, config={"displayModeBar": False})
            with ecol:
                st.markdown('<div class="section-tag">04 / Amplitude decay</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_envelope(measured), use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div class="section-tag">05 / Low-frequency pitch estimate</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_pitch_ridge(measured), use_container_width=True, config={"displayModeBar": False})
            st.caption("The low-frequency ridge is estimated by frame-based autocorrelation. Treat it as an analysis aid, not an exact pitch meter: transient noise, strong harmonics, short duration and changing pitch can create ambiguous estimates.")

            st.markdown("### Frequency-band energy distribution")
            band_names = ["20–60 Hz", "60–120 Hz", "120–250 Hz", "250–1k Hz", "1k–5k Hz", "5k–12k Hz"]
            band_keys = ["20_60_hz", "60_120_hz", "120_250_hz", "250_1000_hz", "1k_5k_hz", "5k_12k_hz"]
            band_vals = [measured["band_energy_pct"][key] for key in band_keys]
            band_fig = go.Figure(go.Bar(x=band_names, y=band_vals, marker=dict(color="#9cae9b")))
            band_fig.update_layout(height=220, margin=dict(l=8, r=8, t=8, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.015)", font=dict(color="#a6a69e", size=10), yaxis_title="SHARE OF 20 Hz–12 kHz SPECTRAL POWER · %")
            st.plotly_chart(band_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("### Decay measurements")
            decay = measured["decay"]
            decay_rows = [{"measurement": "Time to -6 dB", "value_ms": decay["t_minus_6_db_ms"]},
                          {"measurement": "Time to -20 dB", "value_ms": decay["t_minus_20_db_ms"]},
                          {"measurement": "Time to -40 dB", "value_ms": decay["t_minus_40_db_ms"]}]
            st.dataframe(decay_rows, hide_index=True, use_container_width=True)
            report = json_safe_metrics(measured)
            report["source"] = selected_name
            report["analysis_note"] = "Relative spectral measurements; autocorrelation low-frequency ridge is an estimate, not a guaranteed F0 track."
            st.download_button("DOWNLOAD MEASUREMENT REPORT · JSON", json.dumps(report, indent=2), file_name=Path(selected_name).stem + "_analysis.json", mime="application/json", use_container_width=True)
            band_csv = "band,energy_share_percent\n" + "\n".join(f"{name},{value:.6f}" for name, value in zip(band_names, band_vals))
            st.download_button("DOWNLOAD FREQUENCY BANDS · CSV", band_csv, file_name=Path(selected_name).stem + "_bands.csv", mime="text/csv", use_container_width=True)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    st.divider()
    st.markdown("### Factory sample comparison")
    st.caption("This table is measured directly from the five WAV files shipped with the project. It is not a subjective quality ranking.")
    factory_rows = factory_dataset()
    st.dataframe(factory_rows, hide_index=True, use_container_width=True)
    if factory_rows:
        scatter = go.Figure(go.Scatter(
            x=[r["centroid_Hz"] for r in factory_rows], y=[r["t_-20dB_ms"] for r in factory_rows],
            mode="markers+text", text=[r["sample"] for r in factory_rows], textposition="top center",
            marker=dict(size=11, color="#d4b36a", line=dict(width=1, color="#e9e8e3")),
        ))
        scatter.update_layout(height=300, margin=dict(l=8, r=8, t=24, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.015)", font=dict(color="#a6a69e", size=10), xaxis_title="SPECTRAL CENTROID · Hz", yaxis_title="TIME TO -20 dB · ms")
        st.plotly_chart(scatter, use_container_width=True, config={"displayModeBar": False})
    csv_text = "\n".join([",".join(factory_rows[0].keys())] + [",".join(str(row[k]) for k in factory_rows[0].keys()) for row in factory_rows]) if factory_rows else ""
    st.download_button("DOWNLOAD FACTORY DATASET · CSV", csv_text, file_name="PP_Factory_Kick_Measurements.csv", mime="text/csv", use_container_width=True)

with tab_variations:
    st.markdown("## Variation Lab")
    st.caption("Generate a family of related kicks around the current synthesizer settings. Each file is rendered independently and accompanied by exact parameters and measured features.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        batch_size = st.slider("Number of variations", 4, 32, 12, 1)
        variation_amount = st.slider("Variation depth", 0.05, 1.0, 0.35, 0.05)
        seed_value = st.number_input("Random seed", min_value=0, max_value=999_999, value=2026, step=1)
    with right:
        st.markdown('<div class="panel"><div class="metric-label">WHAT CHANGES</div><div class="metric-value">Pitch · decay · body · sub · harmonics · transient · saturation</div><p class="muted">Bounded perturbations explore a parameter neighborhood around the selected kick, rather than making unrelated sounds.</p></div>', unsafe_allow_html=True)
    if st.button("GENERATE VARIATION FAMILY", type="primary", use_container_width=True):
        base = current_params()
        rng = np.random.default_rng(int(seed_value))
        memory = io.BytesIO()
        manifest = []
        with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index in range(int(batch_size)):
                varied = make_variation(base, float(variation_amount), rng)
                audio = generate_kick(varied, seed=int(seed_value) + index + 1)
                name = f"PP_Kick_Var_{index + 1:02d}.wav"
                archive.writestr(name, wav_bytes(audio, varied.sample_rate))
                manifest.append({"file": name, "parameters": params_to_dict(varied), "analysis": analyze_audio(audio, varied.sample_rate) | {"waveform": None, "spectrum": None, "envelope": None, "ridge": None}})
            archive.writestr("manifest.json", json.dumps({"project": "Particula Primordial Kick Engine", "created_utc": datetime.now(timezone.utc).isoformat(), "seed": int(seed_value), "variations": manifest}, indent=2))
        st.session_state.batch_zip = memory.getvalue()
        st.session_state.batch_count = int(batch_size)
    if st.session_state.batch_zip:
        st.success(f"Created {st.session_state.batch_count} WAV variations with parameter and measurement manifest.")
        st.download_button("DOWNLOAD SOUND-DESIGN PACK · ZIP", st.session_state.batch_zip, file_name="Particula_Primordial_Kick_Variations.zip", mime="application/zip", use_container_width=True)

with tab_research:
    st.markdown("## Research Notebook")
    st.caption("A technical and educational layer for documenting how the instrument works, what is measured, and what can legitimately be concluded.")
    st.markdown('<div class="panel"><div class="metric-label">PROJECT THESIS</div><div class="metric-value">A controllable kick is both a musical object and a measurable signal.</div><p class="muted">The synthesizer creates the sound. The laboratory describes its measurable behavior. The research notebook records the method, evidence, interpretation and limitations.</p></div>', unsafe_allow_html=True)
    paper_tabs = st.tabs(["ABSTRACT", "INTRODUCTION", "METHODS", "RESULTS & DISCUSSION", "LIMITATIONS", "LEARNING PATH", "REFERENCES"])
    with paper_tabs[0]:
        st.markdown("""
### Abstract

**Purpose.** This project develops a reproducible, parameterized software instrument for designing low-frequency kick drums and a companion analysis environment for inspecting their signal properties.

**Method.** A kick is synthesized from a time-varying sinusoidal oscillator, a sub component, weighted harmonic partials, a short transient, an amplitude envelope and a soft nonlinear saturation stage. The resulting waveform is measured using level statistics, a windowed Fourier spectrum, short-time Fourier analysis, RMS-envelope decay measurements and an autocorrelation-based low-frequency pitch estimate.

**Current status.** Version 0.2 establishes the instrument, five factory specimens, a browser-based laboratory and a reproducible workflow for exporting audio and measurements. The factory comparison provides descriptive signal measurements; it does not establish perceptual superiority or listener preference.

**Contribution.** The project connects procedural sound synthesis, visual signal inspection and a documented learning/research workflow for iterative kick design.

**Keywords:** procedural audio, kick drum synthesis, digital signal processing, spectral analysis, envelope, reproducibility, sound design.
""")
    with paper_tabs[1]:
        st.markdown("""
### 1. Introduction

A kick drum must perform several roles at once: establish rhythmic weight, occupy a low-frequency region, remain perceptible on different playback systems, and coexist with the bassline. These are related but distinct requirements. A waveform that looks large is not necessarily perceived as powerful; a high peak does not guarantee sustained low-frequency energy; and a strong sub component can consume headroom while translating poorly on small speakers.

The project therefore treats kick design as a controlled signal-generation problem with a musical evaluation stage. Parameters are exposed as interpretable controls rather than hidden behind a sample selector.

**Research question.** How can a compact procedural model expose useful controls over the pitch contour, spectral structure, transient and decay of a club-oriented kick while preserving reproducibility and making its output measurable?

**Subquestions.** (1) How do pitch-contour parameters affect the low-frequency trajectory? (2) How do body, sub and harmonic controls affect spectral-energy distribution? (3) How do attack and decay parameters affect the measured amplitude envelope? (4) Which descriptors help distinguish generated specimens without claiming to predict subjective quality?
""")
    with paper_tabs[2]:
        st.markdown(r"""
### 2. Methods

#### 2.1 Synthesis model

The principal oscillator has a time-varying instantaneous frequency:

\[f(t)=f_{end}+(f_{start}-f_{end})e^{-(t/\tau)^p}\]

where \(f_{start}\) is the initial frequency, \(f_{end}\) is the final frequency, \(\tau\) controls pitch-transition time, and \(p\) controls the curve shape. The oscillator phase is obtained by integrating instantaneous frequency:

\[\phi(t)=2\pi\int_0^t f(u)\,du\]

The body is a sinusoid at this phase. Upper partials are generated at integer multiples of phase and weighted by a harmonic-tilt parameter. A separate low sine component contributes sub energy. A short tonal/noise transient adds attack definition. The combined signal is multiplied by an amplitude envelope and passed through a tanh-based saturation function.

#### 2.2 Analysis model

- **Peak level:** maximum absolute sample amplitude, expressed in dBFS.
- **RMS level:** square-root mean-square amplitude, expressed in dBFS.
- **Crest factor:** peak-to-RMS ratio, in dB.
- **Spectral centroid:** magnitude-weighted mean frequency; a descriptor of spectral balance, not a perceived-brightness verdict.
- **Band-energy share:** proportion of FFT power within predefined frequency bands (20–60, 60–120, 120–250, 250–1,000, 1,000–5,000 and 5,000–12,000 Hz), relative to the measured 20 Hz–12 kHz range.
- **Spectrogram:** short-time Fourier magnitude displayed over time and frequency. Window length governs the time/frequency resolution trade-off.
- **Amplitude decay:** time from the envelope peak until the smoothed RMS envelope crosses -6, -20 and -40 dB relative to that peak.
- **Low-frequency ridge:** frame-based autocorrelation estimate constrained to 25–250 Hz. It is an estimate, not a guaranteed fundamental-frequency tracker.

#### 2.3 Reproducibility

Each rendered sample is tied to a parameter dictionary and random seed. The exported sound-design pack contains WAV files, the seed, parameter values and measured descriptors. Factory measurements are recalculated from the WAV files at application runtime.
""")
    with paper_tabs[3]:
        st.markdown("""
### 3. Results and Discussion

The five factory WAVs are the initial dataset. The Kick Laboratory calculates their measurements directly from the audio files and provides a CSV export. The comparison scatter plot places spectral centroid against time to -20 dB; it is a descriptive map of the current sample set, not a quality score.

**Interpretation rule:** report what the measurement supports. For example, a longer time-to--20 dB indicates a longer measured envelope tail under this analysis method. It does not, by itself, prove that the kick sounds warmer, larger or more danceable. Those judgments require controlled listening and playback-context evaluation.

**Next experiment:** create parameter-controlled pairs in which only one factor changes (for example, transient level or pitch-decay time), match playback level, compare in the same loop and record listener judgments separately from signal measurements.
""")
        figure_left, figure_right = st.columns(2, gap="large")
        with figure_left:
            st.image(str(ROOT / "research" / "figures" / "factory_measurement_map.png"), caption="Figure 1 · Spectral centroid versus measured envelope decay", width=620)
        with figure_right:
            st.image(str(ROOT / "research" / "figures" / "factory_band_energy.png"), caption="Figure 2 · Measured spectral-energy distribution", width=620)
        st.image(str(ROOT / "research" / "figures" / "factory_reference_spectrogram.png"), caption="Figure 3 · STFT spectrogram of the Rounded Chest factory specimen", width=900)
        rows = factory_dataset()
        st.markdown("#### Initial factory dataset · measured from included WAV files")
        st.caption("Across these five samples: the 20–60 Hz band contains roughly 98.7–99.7% of measured 20 Hz–12 kHz spectral power; time to the -20 dB RMS-envelope threshold ranges from roughly 960 to 1,650 ms. This describes the prototype samples, not a universal target.")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        if rows:
            st.download_button("DOWNLOAD INITIAL DATASET · CSV", "\n".join([",".join(rows[0].keys())] + [",".join(str(row[k]) for k in rows[0].keys()) for row in rows]), file_name="PP_Initial_Factory_Dataset.csv", mime="text/csv")
    with paper_tabs[4]:
        st.markdown("""
### 4. Limitations

1. **The synthesis model is intentionally compact.** It does not yet model a physical drum, membrane modes, room coupling or complex analog circuitry.
2. **Spectral centroid and band energy are signal descriptors.** They do not directly predict perceived warmth, punch, chest impact or danceability.
3. **The autocorrelation pitch trace can be ambiguous.** Strong transients, noisy attacks, short frames, strong harmonics and rapidly changing pitch can confuse the estimate.
4. **RMS threshold decay is operationally defined.** The result depends on frame length, hop size and threshold convention; it is not a standardized acoustic decay time such as reverberation time.
5. **The initial dataset is small and synthetic.** Five factory samples are useful for software verification and design comparison, not general scientific conclusions.
6. **Playback is not perceptually calibrated.** Listener tests, level matching, headphones/speaker documentation and kick-bass mix tests remain future work.

Do not describe the current peak or RMS statistics as ITU programme loudness. This application does not implement the full ITU-R BS.1770 algorithm.
""")
    with paper_tabs[5]:
        st.markdown("""
### Learning path

1. **Waveform and sampling:** amplitude, sample rate, duration, quantization and headroom.
2. **Oscillators and phase:** frequency as phase change; why pitch sweeps need phase integration.
3. **Envelopes:** attack, decay, sustain and release; amplitude contour versus pitch contour.
4. **Harmonics:** integer partials, spectral balance and why the fundamental is not the whole sound.
5. **Fourier analysis:** FFT, windowing, leakage, frequency resolution and logarithmic frequency axes.
6. **Time-frequency analysis:** STFT/spectrogram and the resolution trade-off.
7. **Measurement versus perception:** descriptors can compare signals, but listening in context determines musical usefulness.

**Practical exercise:** render one kick, export its JSON parameters, change only pitch-decay time, render again and compare the pitch ridge and spectrogram. Repeat with only the transient amount, then compare the waveform and high-frequency energy share.
""")
    with paper_tabs[6]:
        st.markdown("""
### References and further reading

1. **Julius O. Smith III — Spectral Audio Signal Processing.** [Read online](https://www.dsprelated.com/freebooks/sasp/). Fourier analysis, spectra, STFT and time-frequency displays.
2. **Udo Zölzer (ed.) — DAFX: Digital Audio Effects, 2nd edition.** [Wiley / DOI](https://doi.org/10.1002/9781119991298). Digital audio processing, nonlinear processing and spectral effects.
3. **SciPy — ShortTimeFFT documentation.** [Technical reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html). The STFT implementation used by the Kick Laboratory.
4. **SciPy — Signal processing tutorial.** [STFT explanation](https://docs.scipy.org/doc/scipy/tutorial/signal.html#short-time-fourier-transform). Windowed, overlapping Fourier analysis.
5. **ITU-R BS.1770-5.** [Official recommendation](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en). Programme loudness and true-peak metering. The current application does not implement this standard.

The complete bibliography and reading order are also available in `research/REFERENCES.md`.
""")

with tab_presets:
    st.markdown("## Factory characters")
    st.caption("These are starting points, not finished rules. Adjust them to fit the kick-bass relationship in your track.")
    descriptions = {
        "01 · Rounded Chest": "Soft leading edge, stable low body, subtle upper harmonics. The reference starting point for a rounded club kick.",
        "02 · Velvet Pressure": "Longer and smoother, with restrained transient energy and a deeper sustained body.",
        "03 · Controlled Punch": "Shorter envelope, faster pitch movement and a more defined attack without an industrial edge.",
        "04 · Organic Pulse": "Slower pitch gesture and longer decay for a less mechanical, breathing quality.",
        "05 · Deep Ritual": "Low-tuned body, extended weight and minimal transient. Check the kick against your bassline for headroom.",
    }
    for name, description in descriptions.items():
        with st.expander(name, expanded=(name == "01 · Rounded Chest")):
            st.write(description)
            st.json(params_to_dict(PRESETS[name]))
    st.divider()
    st.markdown("### Sound-design method")
    st.markdown("""
1. **Tune the body first.** Set the ending frequency and listen against the bassline.
2. **Shape the pitch fall.** Start frequency and pitch-decay time control the initial impact.
3. **Add weight before attack.** Increase body and sub carefully.
4. **Use harmonics sparingly.** Add enough upper partials for translation without making the kick clicky.
5. **Set the transient last.** Add only enough definition for the kick to remain audible in the loop.
6. **Export, measure, compare and curate.** Keep sounds that work in the complete groove, not only in solo.
""")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">PARTICULA PRIMORDIAL · COMPUTATIONAL SOUND DESIGN · BUILD 0.2.0</div>', unsafe_allow_html=True)
