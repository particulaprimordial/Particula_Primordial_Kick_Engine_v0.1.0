from __future__ import annotations

import html
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile

from kick_analysis import analyze_audio, spectrogram_data
from kick_engine import KickParams, generate_kick, make_variation, params_to_dict
from presets import PRESETS

ROOT = Path(__file__).parent
FACTORY_DIR = ROOT / "examples" / "factory-kicks"
FIG_DIR = ROOT / "research" / "figures"

# ───────────────────────────── design tokens ─────────────────────────────
PAPER, PANEL = "#f7f8fa", "#eceff3"
INK, MUTED, LINE, GRID = "#12161c", "#667080", "#d3d8e0", "#e6e9ee"
ACCENT, TEAL, AMBER, VIOLET, ROSE = "#2438ff", "#0b8a7f", "#b86e00", "#7a3ff2", "#c2255c"
SERIES = [ACCENT, INK, TEAL, AMBER, VIOLET, ROSE]
MONO = "IBM Plex Mono, ui-monospace, monospace"

NEW_API = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2]) >= (1, 50)
FULL = {"width": "stretch"} if NEW_API else {"use_container_width": True}

st.set_page_config(
    page_title="Kick Engine · Particula Primordial",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,600;1,6..72,300;1,6..72,400&display=swap');
:root { --ink:%INK%; --muted:%MUTED%; --line:%LINE%; --paper:%PAPER%; --panel:%PANEL%; --accent:%ACCENT%; }
html, body, .stApp, [class*="css"] { font-family:'Newsreader', Georgia, serif; color:var(--ink); }
.stApp { background:var(--paper); }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
.block-container { max-width:1360px; padding:2.2rem 2.6rem 4rem; }
h1, h2, h3, h4 { font-family:'Newsreader', Georgia, serif; font-weight:400; letter-spacing:-.02em; color:var(--ink); }
h2 { font-size:1.7rem; margin:.2rem 0 .2rem; } h3 { font-size:1.25rem; } h4 { font-size:1.05rem; font-weight:600; }
p, li { font-size:1.04rem; line-height:1.62; }
[data-testid="stCaptionContainer"], .muted { color:var(--muted); font-size:.9rem; }
[data-testid="stWidgetLabel"] p { font:400 .76rem 'IBM Plex Mono', monospace; color:var(--ink); }
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }
[data-testid="stSidebar"] h3 { font-size:1.05rem; margin-top:1.2rem; }
div[data-baseweb="tab-list"] { gap:1.7rem; border-bottom:1px solid var(--line); }
button[data-baseweb="tab"] { background:transparent; padding:.7rem 0; font:500 .82rem 'IBM Plex Mono', monospace; color:var(--muted); }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--ink); }
div[data-baseweb="tab-highlight"] { background:var(--accent); height:2px; }
div[data-baseweb="tab-border"] { background:var(--line); }
.stButton button, .stDownloadButton button { border-radius:2px; min-height:2.5rem; font:500 .8rem 'IBM Plex Mono', monospace; border:1px solid var(--ink); background:transparent; color:var(--ink); }
.stButton button:hover, .stDownloadButton button:hover { border-color:var(--accent); color:var(--accent); }
.stButton button[kind="primary"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.stButton button[kind="primary"]:hover { filter:brightness(1.1); color:#fff; }
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input { border-radius:2px; background:#fff; border-color:var(--line); font-family:'IBM Plex Mono', monospace; font-size:.82rem; }
.stSlider [data-testid="stTickBar"] { display:none; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:2px; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
hr { border-color:var(--line); }
.hero { display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; flex-wrap:wrap; border-bottom:1px solid var(--ink); padding-bottom:1.1rem; margin-bottom:1.4rem; }
.hero h1 { font-size:3.5rem; font-weight:300; line-height:1; margin:0; }
.hero h1 em { font-style:italic; color:var(--accent); font-weight:300; }
.hero p { color:var(--muted); margin:.55rem 0 0; max-width:48ch; }
.specs { font:400 .74rem 'IBM Plex Mono', monospace; color:var(--muted); text-align:right; line-height:1.9; }
.sec { font:500 .8rem 'IBM Plex Mono', monospace; border-top:1px solid var(--ink); padding-top:.55rem; margin:1.7rem 0 .8rem; }
.sec span { color:var(--muted); font-weight:400; margin-left:.8rem; }
.sheet { display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); border-top:1px solid var(--ink); margin:.5rem 0 1.2rem; }
.cell { padding:.6rem 1rem .55rem 0; border-bottom:1px solid var(--line); }
.cell .k { font:400 .72rem 'IBM Plex Mono', monospace; color:var(--muted); }
.cell .v { font:300 1.75rem 'Newsreader', serif; letter-spacing:-.02em; }
.cell .v small { font:400 .72rem 'IBM Plex Mono', monospace; color:var(--muted); margin-left:.3rem; letter-spacing:0; }
.specimen { font:300 1.5rem 'Newsreader', serif; margin:.4rem 0 0; }
.specimen small { font:400 .76rem 'IBM Plex Mono', monospace; color:var(--muted); margin-left:1rem; }
.figcap { font-size:.88rem; color:var(--muted); border-top:1px solid var(--line); padding-top:.4rem; margin:-.3rem 0 1.5rem; max-width:92ch; }
.figcap b { color:var(--ink); font-weight:600; }
.note { border-left:2px solid var(--accent); padding:.1rem 0 .1rem .9rem; color:var(--muted); margin:.6rem 0 1rem; max-width:80ch; }
.status { font:400 .74rem 'IBM Plex Mono', monospace; color:var(--muted); }
.status i { display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-right:.5rem; background:var(--accent); }
.status i.stale { background:#b86e00; } .status i.idle { background:var(--line); }
.foot { font:400 .72rem 'IBM Plex Mono', monospace; color:var(--muted); border-top:1px solid var(--line); margin-top:3rem; padding-top:.8rem; }
</style>
"""
for _k, _v in dict(INK=INK, MUTED=MUTED, LINE=LINE, PAPER=PAPER, PANEL=PANEL, ACCENT=ACCENT).items():
    CSS = CSS.replace(f"%{_k}%", _v)
st.markdown(CSS, unsafe_allow_html=True)

# ───────────────────────────── parameter model ─────────────────────────────
# key: (label, min, max, step, help)
SLIDERS: dict[str, tuple[str, float, float, float, str]] = {
    "start_freq": ("Start frequency (Hz)", 55.0, 360.0, 1.0, "Oscillator frequency at the first sample. Higher values give a sharper beater-like onset."),
    "end_freq": ("End frequency (Hz)", 28.0, 100.0, 1.0, "Resting pitch of the body. Tune this against the bassline."),
    "pitch_decay_ms": ("Pitch decay (ms)", 8.0, 220.0, 1.0, "Time scale τ of the pitch fall."),
    "pitch_curve": ("Pitch curve p", 0.5, 2.3, 0.05, "Curvature exponent of the fall: below 1 is front-loaded, above 1 is rounder."),
    "body": ("Body", 0.0, 1.0, 0.01, "Level of the main swept sine."),
    "sub": ("Sub weight", 0.0, 1.0, 0.01, "Level of the separate low sine component."),
    "harmonics": ("Harmonic content", 0.0, 0.7, 0.01, "Overall level of integer partials above the body."),
    "harmonic_tilt": ("Harmonic tilt", 0.5, 3.0, 0.05, "How quickly partial levels fall with partial number."),
    "transient": ("Transient amount", 0.0, 0.6, 0.01, "Level of the short tonal/noise onset."),
    "transient_tone": ("Transient tone (Hz)", 500.0, 8000.0, 50.0, "Centre of the tonal part of the transient."),
    "transient_decay_ms": ("Transient decay (ms)", 1.0, 50.0, 1.0, "How quickly the transient dies away."),
    "attack_ms": ("Attack (ms)", 0.1, 25.0, 0.1, "Amplitude rise time. Longer values round the leading edge."),
    "duration": ("Duration (s)", 0.12, 2.0, 0.01, "Length of the rendered file."),
    "decay_ms": ("Amplitude decay (ms)", 100.0, 1500.0, 10.0, "Time scale of the amplitude fall. Distinct from pitch decay."),
    "saturation": ("Saturation", 0.0, 0.7, 0.01, "Amount of tanh waveshaping."),
    "output_db": ("Output ceiling (dBFS)", -12.0, 0.0, 0.5, "Target sample peak of the render."),
}
GROUPS = [
    ("Pitch contour", "How the oscillator falls to its resting pitch.", ["start_freq", "end_freq", "pitch_decay_ms", "pitch_curve"]),
    ("Low-end body", "Weight and upper-partial balance.", ["body", "sub", "harmonics", "harmonic_tilt"]),
    ("Impact", "Onset definition.", ["transient", "transient_tone", "transient_decay_ms", "attack_ms"]),
    ("Envelope and tone", "Length, fall and saturation.", ["duration", "decay_ms", "saturation", "output_db"]),
]

BANDS = [("20–60 Hz", "20_60_hz"), ("60–120 Hz", "60_120_hz"), ("120–250 Hz", "120_250_hz"),
         ("250 Hz–1 kHz", "250_1000_hz"), ("1–5 kHz", "1k_5k_hz"), ("5–12 kHz", "5k_12k_hz")]

FEATURES = {
    "duration_ms": ("Duration", "ms"), "peak_dBFS": ("Peak", "dBFS"), "rms_dBFS": ("RMS", "dBFS"),
    "crest_dB": ("Crest factor", "dB"), "centroid_Hz": ("Spectral centroid", "Hz"), "dominant_Hz": ("Dominant bin", "Hz"),
    "pct_20_60": ("20–60 Hz share", "%"), "pct_60_120": ("60–120 Hz share", "%"), "pct_120_250": ("120–250 Hz share", "%"),
    "pct_250_1k": ("250 Hz–1 kHz share", "%"), "pct_1k_12k": ("1–12 kHz share", "%"),
    "t_minus_6dB_ms": ("Time to −6 dB", "ms"), "t_minus_20dB_ms": ("Time to −20 dB", "ms"), "t_minus_40dB_ms": ("Time to −40 dB", "ms"),
}
MATRIX_DEFAULT = ["centroid_Hz", "rms_dBFS", "crest_dB", "pct_20_60", "pct_60_120", "pct_120_250", "t_minus_6dB_ms", "t_minus_20dB_ms"]

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "kick_engine_figure"},
}

FIG_COUNTER = {"n": 0}


# ───────────────────────────── state & helpers ─────────────────────────────
def init_state() -> None:
    for key, value in params_to_dict(KickParams()).items():
        st.session_state.setdefault(key, float(value) if key in SLIDERS else value)
    defaults = {
        "preset_choice": "01 · Rounded Chest", "audio": None, "audio_params": None, "audio_seed": None,
        "slots": {}, "variations": [], "batch_zip": None, "render_seed": 17, "live": True, "rand_depth": 0.25, "rand_n": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clamp_into_state(values: dict) -> None:
    for key, value in values.items():
        if key in SLIDERS:
            _, lo, hi, _, _ = SLIDERS[key]
            st.session_state[key] = float(min(max(float(value), lo), hi))
        else:
            st.session_state[key] = value


def apply_preset() -> None:
    clamp_into_state(params_to_dict(PRESETS[st.session_state.preset_choice]))
    st.session_state.audio = None
    st.session_state.audio_params = None


def choose_preset(name: str) -> None:
    st.session_state.preset_choice = name
    apply_preset()


def randomize() -> None:
    st.session_state.rand_n += 1
    rng = np.random.default_rng(int(st.session_state.render_seed) + st.session_state.rand_n)
    clamp_into_state(params_to_dict(make_variation(current_params(), float(st.session_state.rand_depth), rng)))


def load_variation(name: str) -> None:
    for item in st.session_state.variations:
        if item["name"] == name:
            clamp_into_state(item["params"])


def current_params() -> KickParams:
    values = {key: st.session_state[key] for key in KickParams.__dataclass_fields__}
    return KickParams(**values).validated()


def render() -> None:
    p = current_params()
    st.session_state.audio = generate_kick(p, seed=int(st.session_state.render_seed))
    st.session_state.audio_params = params_to_dict(p)
    st.session_state.audio_seed = int(st.session_state.render_seed)


def to_float(audio) -> np.ndarray:
    a = np.asarray(audio)
    if a.dtype.kind == "i":
        a = a.astype(np.float64) / float(-np.iinfo(a.dtype).min)
    elif a.dtype.kind == "u":
        half = (np.iinfo(a.dtype).max + 1) / 2
        a = (a.astype(np.float64) - half) / half
    else:
        a = a.astype(np.float64)
    return a.mean(axis=1) if a.ndim > 1 else a


def wav_bytes(audio, sample_rate: int = 48_000, gain: float = 1.0) -> bytes:
    stream = io.BytesIO()
    wavfile.write(stream, int(sample_rate), (to_float(audio) * gain).astype(np.float32))
    return stream.getvalue()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(text).stem).strip("_") or "specimen"


def json_default(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


@st.cache_data(show_spinner=False, max_entries=64)
def measure(audio: np.ndarray, sr: int) -> dict:
    return analyze_audio(audio, sr)


@st.cache_data(show_spinner=False, max_entries=32)
def spectro(audio: np.ndarray, sr: int):
    return spectrogram_data(audio, sr)


def feature_row(name: str, m: dict) -> dict:
    band, decay = m["band_energy_pct"], m["decay"]
    return {
        "specimen": name,
        "duration_ms": m["duration_s"] * 1000.0, "peak_dBFS": m["peak_dbfs"], "rms_dBFS": m["rms_dbfs"],
        "crest_dB": m["crest_db"], "centroid_Hz": m["spectral_centroid_hz"], "dominant_Hz": m["dominant_frequency_hz"],
        "pct_20_60": band["20_60_hz"], "pct_60_120": band["60_120_hz"], "pct_120_250": band["120_250_hz"],
        "pct_250_1k": band["250_1000_hz"], "pct_1k_12k": band["1k_5k_hz"] + band["5k_12k_hz"],
        "t_minus_6dB_ms": decay["t_minus_6_db_ms"], "t_minus_20dB_ms": decay["t_minus_20_db_ms"],
        "t_minus_40dB_ms": decay["t_minus_40_db_ms"],
    }


def json_safe_metrics(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in {"waveform", "spectrum", "envelope", "ridge"}}


@st.cache_data(show_spinner=False)
def factory_library() -> dict:
    lib = {}
    for path in sorted(FACTORY_DIR.glob("*.wav")):
        sr, audio = wavfile.read(path)
        lib[path.stem.replace("PP_Factory_", "").replace("_", " ")] = (int(sr), audio)
    return lib


def build_library() -> dict:
    lib: dict = {}
    if st.session_state.audio is not None:
        lib["Current render"] = (int(st.session_state.audio_params["sample_rate"]), st.session_state.audio)
    for slot, (sr, audio, _) in st.session_state.slots.items():
        lib[f"Slot {slot}"] = (sr, audio)
    lib.update(factory_library())
    for item in st.session_state.variations:
        lib[item["name"]] = (item["sr"], item["audio"])
    for up in st.session_state.get("uploads") or []:
        try:
            sr, audio = wavfile.read(io.BytesIO(up.getvalue()))
            lib[up.name] = (int(sr), audio)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"{up.name}: could not read WAV ({exc})")
    return lib


def df_of(names: list[str], lib: dict) -> pd.DataFrame:
    return pd.DataFrame([feature_row(n, measure(lib[n][1], lib[n][0])) for n in names])


def column_config() -> dict:
    return {k: st.column_config.NumberColumn(f"{lbl} ({unit})", format="%.2f") for k, (lbl, unit) in FEATURES.items()}


def axis_label(key: str) -> str:
    lbl, unit = FEATURES[key]
    return f"{lbl} ({unit})"


# ───────────────────────────── page furniture ─────────────────────────────
def sec(title: str, hint: str = "") -> None:
    extra = f"<span>{html.escape(hint)}</span>" if hint else ""
    st.markdown(f'<div class="sec">{html.escape(title)}{extra}</div>', unsafe_allow_html=True)


def sheet(cells: list[tuple[str, str, str]]) -> None:
    body = "".join(f'<div class="cell"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}<small>{html.escape(u)}</small></div></div>' for k, v, u in cells)
    st.markdown(f'<div class="sheet">{body}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    FIG_COUNTER["n"] += 1
    st.markdown(f'<div class="figcap"><b>Figure {FIG_COUNTER["n"]}.</b> {text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def status(kind: str, text: str) -> None:
    st.markdown(f'<div class="status"><i class="{kind}"></i>{html.escape(text)}</div>', unsafe_allow_html=True)


# ───────────────────────────── plots ─────────────────────────────
def style(fig: go.Figure, height: int = 260, x: str = "", y: str = "", hover: str = "x unified") -> go.Figure:
    axis = dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE,
                title_font=dict(size=11), title_standoff=8)
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=14, b=4), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, size=11, color=MUTED), hovermode=hover, showlegend=False,
        hoverlabel=dict(font_family=MONO, font_size=11, bgcolor="#fff", bordercolor=LINE, font_color=INK),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=ACCENT),
    )
    fig.update_xaxes(title_text=x, showspikes=True, spikecolor=MUTED, spikethickness=1, spikedash="dot", spikemode="across", **axis)
    fig.update_yaxes(title_text=y, **axis)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key=key, **FULL)


def freq_axis(fig: go.Figure, log: bool, fmin: float, fmax: float, axis: str = "x") -> None:
    rng = [np.log10(fmin), np.log10(fmax)] if log else [0, fmax]
    getattr(fig, f"update_{axis}axes")(type="log" if log else "linear", range=rng)


def plot_pitch_model() -> go.Figure:
    s = st.session_state
    t = np.linspace(0, 0.35, 500)
    tau = max(float(s.pitch_decay_ms), 1e-3) / 1000.0
    f = s.end_freq + (s.start_freq - s.end_freq) * np.exp(-((t / tau) ** s.pitch_curve))
    fig = go.Figure(go.Scatter(x=t * 1000, y=f, mode="lines", line=dict(color=ACCENT, width=2), name="f(t)", hovertemplate="%{y:.1f} Hz"))
    fig.add_hline(y=s.end_freq, line=dict(color=MUTED, width=1, dash="dot"))
    return style(fig, 210, "Time (ms)", "Frequency (Hz)")


def plot_waveform(m: dict, win: tuple[float, float]) -> go.Figure:
    x, y = np.asarray(m["waveform"]["time_ms"]), np.asarray(m["waveform"]["amplitude"])
    keep = (x >= win[0]) & (x <= win[1])
    x, y = x[keep], y[keep]
    stride = max(1, int(np.ceil(len(y) / 6000)))
    fig = go.Figure(go.Scatter(x=x[::stride], y=y[::stride], mode="lines", line=dict(color=INK, width=1), hovertemplate="%{y:.3f}"))
    fig.update_yaxes(range=[-1.05, 1.05])
    return style(fig, 250, "Time (ms)", "Amplitude")


def plot_spectrum(m: dict, fmin: float, fmax: float, floor: float, log: bool) -> go.Figure:
    f, db = np.asarray(m["spectrum"]["frequency_hz"]), np.asarray(m["spectrum"]["magnitude_db"])
    keep = (f >= fmin) & (f <= fmax)
    fig = go.Figure(go.Scatter(x=f[keep], y=db[keep], mode="lines", line=dict(color=TEAL, width=1.4), hovertemplate="%{y:.1f} dB"))
    peak = m["dominant_frequency_hz"]
    fig.add_vline(x=peak, line=dict(color=ACCENT, width=1, dash="dot"), annotation_text=f"{peak:.1f} Hz", annotation_font=dict(size=10, color=ACCENT))
    freq_axis(fig, log, fmin, fmax)
    fig.update_yaxes(range=[floor, 3])
    return style(fig, 250, "Frequency (Hz)", "Relative level (dB)")


def plot_spectrogram(audio, sr: int, win, fmin, fmax, floor, log, cmap, ridge, ridge_conf) -> go.Figure:
    f, t, db = spectro(audio, sr)
    f, db = np.asarray(f), np.asarray(db)
    keep = f > 0
    fig = go.Figure(go.Heatmap(x=np.asarray(t) * 1000.0, y=f[keep], z=db[keep], zmin=floor, zmax=0, colorscale=cmap,
                               colorbar=dict(title=dict(text="dB", side="right"), thickness=8, len=0.8, outlinewidth=0),
                               hovertemplate="%{x:.0f} ms, %{y:.0f} Hz, %{z:.1f} dB<extra></extra>"))
    if ridge is not None:
        tt, ff, cc = (np.asarray(ridge[k]) for k in ("time_s", "frequency_hz", "confidence"))
        ok = np.isfinite(ff) & (cc >= ridge_conf)
        fig.add_trace(go.Scatter(x=tt[ok] * 1000, y=ff[ok], mode="lines", line=dict(color="#fff", width=1.6, dash="dot"), hoverinfo="skip"))
    freq_axis(fig, log, fmin, fmax, "y")
    fig.update_xaxes(range=list(win))
    return style(fig, 340, "Time (ms)", "Frequency (Hz)", hover="closest")


def plot_envelope(m: dict, win, floor: float) -> go.Figure:
    env = m["envelope"]
    x, y = np.asarray(env["time_ms"]), np.asarray(env["rms_db"])
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=ACCENT, width=1.8), hovertemplate="%{y:.1f} dB"))
    t_peak = float(x[int(np.nanargmax(y))])
    for label, thr, key in (("−6", -6, "t_minus_6_db_ms"), ("−20", -20, "t_minus_20_db_ms"), ("−40", -40, "t_minus_40_db_ms")):
        val = m["decay"][key]
        if val is not None:
            fig.add_trace(go.Scatter(x=[t_peak + val], y=[thr], mode="markers+text", text=[f"{label} dB · {val:.0f} ms"], textposition="top right",
                                     marker=dict(size=7, color=INK), textfont=dict(size=10, color=INK), hoverinfo="skip"))
    fig.update_xaxes(range=list(win))
    fig.update_yaxes(range=[max(floor, -90), 3])
    return style(fig, 250, "Time (ms)", "RMS envelope (dB re peak)")


def plot_ridge(m: dict, win, conf: float) -> go.Figure:
    r = m["ridge"]
    t, f, c = np.asarray(r["time_s"]) * 1000, np.asarray(r["frequency_hz"]), np.asarray(r["confidence"])
    ok = np.isfinite(f) & (c >= conf)
    fig = go.Figure(go.Scatter(x=t[ok], y=f[ok], mode="lines+markers", line=dict(color=LINE, width=1),
                               marker=dict(size=4 + 8 * c[ok], color=c[ok], colorscale=[[0, "#c9cfda"], [1, ACCENT]], cmin=0, cmax=1, showscale=True,
                                           colorbar=dict(title=dict(text="conf.", side="right"), thickness=8, len=0.8, outlinewidth=0)),
                               hovertemplate="%{y:.1f} Hz"))
    fig.update_xaxes(range=list(win))
    fig.update_yaxes(range=[20, 260])
    return style(fig, 250, "Time (ms)", "Estimated ridge (Hz)")


def plot_bands(series: dict[str, dict], log: bool) -> go.Figure:
    fig = go.Figure()
    for i, (name, m) in enumerate(series.items()):
        vals = [m["band_energy_pct"][k] for _, k in BANDS]
        fig.add_trace(go.Bar(name=name, x=[b for b, _ in BANDS], y=[max(v, 1e-4) if log else v for v in vals], customdata=vals,
                             marker=dict(color=SERIES[i % len(SERIES)]), hovertemplate="%{customdata:.4f} %<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group", showlegend=len(series) > 1)
    fig.update_yaxes(type="log" if log else "linear")
    return style(fig, 280, "", "Share of 20 Hz–12 kHz power (%)", hover="closest")


def explorer(df: pd.DataFrame, key: str, dx="centroid_Hz", dy="t_minus_20dB_ms") -> None:
    cols = [c for c in FEATURES if c in df.columns and df[c].notna().any()]
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X axis", cols, index=cols.index(dx) if dx in cols else 0, format_func=axis_label, key=f"{key}_x")
    y = c2.selectbox("Y axis", cols, index=cols.index(dy) if dy in cols else min(1, len(cols) - 1), format_func=axis_label, key=f"{key}_y")
    colour = c3.selectbox("Colour", ["none"] + cols, format_func=lambda c: "none" if c == "none" else axis_label(c), key=f"{key}_c")
    marker = dict(size=12, line=dict(width=1, color=INK))
    if colour == "none":
        marker["color"] = ACCENT
    else:
        marker.update(color=df[colour], colorscale="Viridis", showscale=True, colorbar=dict(title=dict(text=FEATURES[colour][1], side="right"), thickness=8, outlinewidth=0))
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="markers+text" if len(df) <= 12 else "markers", text=df["specimen"], textposition="top center",
                               textfont=dict(size=10, color=INK), marker=marker, customdata=df["specimen"],
                               hovertemplate="%{customdata}<br>" + FEATURES[x][0] + ": %{x:.2f}<br>" + FEATURES[y][0] + ": %{y:.2f}<extra></extra>"))
    show(style(fig, 360, axis_label(x), axis_label(y), hover="closest"), f"{key}_plot")


# ───────────────────────────── sidebar ─────────────────────────────
init_state()
with st.sidebar:
    st.markdown("### Render")
    st.toggle("Live render", key="live", help="Re-render the kick after every control change. Rendering is fast; switch off to render on demand.")
    st.number_input("Random seed", 0, 999_999, key="render_seed", step=1, help="Seeds the noise transient. Same parameters and seed reproduce the same audio.")
    st.markdown("### Display")
    cmap = st.selectbox("Spectrogram colour map", ["Cividis", "Viridis", "Magma", "Inferno", "Turbo", "Greys"], index=0)
    floor = float(st.slider("Dynamic range floor (dB)", -120, -40, -80, 5))
    fscale = st.radio("Frequency axis", ["Logarithmic", "Linear"], horizontal=True)
    fmax = float(st.select_slider("Maximum frequency (Hz)", [300, 500, 1000, 2000, 5000, 12000], value=2000))
    ridge_on = st.toggle("Overlay pitch ridge on spectrogram", value=True)
    ridge_conf = float(st.slider("Ridge confidence threshold", 0.05, 0.9, 0.10, 0.05, help="Autocorrelation frames below this confidence are hidden."))
    st.markdown("### Import")
    st.file_uploader("Add WAV files", type=["wav"], accept_multiple_files=True, key="uploads", help="Mono or stereo, integer PCM or float. Imported files appear in Measure and Compare.")
    st.caption("Peak and RMS are sample statistics, not LUFS or true-peak measurements.")
LOG = fscale == "Logarithmic"
FMIN = 20.0

# ───────────────────────────── header ─────────────────────────────
st.markdown(
    '<div class="hero"><div><h1>Kick Engine <em>/ 01</em></h1>'
    "<p>A procedural instrument and measurement laboratory for low-frequency drums: synthesize, measure, compare and document.</p></div>"
    '<div class="specs">Particula Primordial<br>Version 0.2.0<br>48 kHz, 32-bit float, mono</div></div>',
    unsafe_allow_html=True,
)

tab_synth, tab_lab, tab_cmp, tab_var, tab_note, tab_pre = st.tabs(["Synthesize", "Measure", "Compare", "Variations", "Notebook", "Presets"])

# ───────────────────────────── synthesize ─────────────────────────────
with tab_synth:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        sec("Starting point")
        st.selectbox("Factory preset", list(PRESETS.keys()), key="preset_choice", on_change=apply_preset)
        b1, b2 = st.columns(2)
        b1.button("Reset to preset", on_click=apply_preset, **FULL)
        b2.button("Randomize", on_click=randomize, help="Bounded random perturbation of every parameter around the current settings.", **FULL)
        st.slider("Randomize depth", 0.05, 1.0, step=0.05, key="rand_depth")
    with right:
        sec("Pitch contour", "model preview of Eq. 1, not measured audio")
        show(plot_pitch_model(), "pitch_model")

    for col, (title, blurb, keys) in zip(st.columns(4, gap="large"), GROUPS):
        with col:
            sec(title)
            st.caption(blurb)
            for k in keys:
                label, lo, hi, step, helptext = SLIDERS[k]
                st.slider(label, lo, hi, step=step, key=k, help=helptext)

    sec("Render")
    try:
        if st.session_state.live:
            render()
        else:
            st.button("Render kick", type="primary", on_click=render)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not render with these parameters: {exc}")

    if st.session_state.audio is None:
        status("idle", "No render yet. Press Render kick or switch on Live render.")
    else:
        audio, ap = st.session_state.audio, st.session_state.audio_params
        p = KickParams(**ap)
        try:
            stale = params_to_dict(current_params()) != ap or st.session_state.audio_seed != int(st.session_state.render_seed)
        except Exception:  # noqa: BLE001
            stale = True
        status("stale" if stale else "", "Controls changed since this render." if stale else "Render matches the current controls.")
        m = measure(audio, p.sample_rate)
        sheet([("Peak", f"{m['peak_dbfs']:.1f}", "dBFS"), ("RMS", f"{m['rms_dbfs']:.1f}", "dBFS"), ("Crest factor", f"{m['crest_db']:.1f}", "dB"),
               ("Spectral centroid", f"{m['spectral_centroid_hz']:.0f}", "Hz"), ("Dominant bin", f"{m['dominant_frequency_hz']:.1f}", "Hz"),
               ("Duration", f"{m['duration_s'] * 1000:.0f}", "ms")])
        st.audio(wav_bytes(audio, p.sample_rate), format="audio/wav")
        w1, w2 = st.columns(2, gap="large")
        with w1:
            show(plot_waveform(m, (0.0, m["duration_s"] * 1000)), "syn_wave")
        with w2:
            show(plot_spectrum(m, FMIN, fmax, floor, LOG), "syn_spec")
        name = f"PP_Kick_{int(p.start_freq)}-{int(p.end_freq)}Hz"
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Download WAV", wav_bytes(audio, p.sample_rate), file_name=name + ".wav", mime="audio/wav", **FULL)
        d2.download_button("Download preset JSON", json.dumps(params_to_dict(p), indent=2), file_name=name + ".json", mime="application/json", **FULL)
        if d3.button("Store as slot A", **FULL):
            st.session_state.slots["A"] = (p.sample_rate, audio.copy(), ap)
        if d4.button("Store as slot B", **FULL):
            st.session_state.slots["B"] = (p.sample_rate, audio.copy(), ap)
        if st.session_state.slots:
            st.caption("Stored slots: " + ", ".join(st.session_state.slots) + ". Compare them in the Compare tab.")

# ───────────────────────────── measure ─────────────────────────────
with tab_lab:
    FIG_COUNTER["n"] = 0
    lib = build_library()
    st.markdown("## Kick laboratory")
    st.caption("Every figure is computed from the selected audio file.")
    c1, c2 = st.columns([1.2, 1], gap="large")
    names = list(lib.keys())
    chosen = c1.selectbox("Specimen", names, key="lab_specimen")
    panels = c2.multiselect("Panels", ["Waveform", "Spectrogram", "Spectrum", "Envelope", "Pitch ridge", "Bands"],
                            default=["Waveform", "Spectrogram", "Spectrum", "Envelope", "Pitch ridge", "Bands"])
    sr_sel, audio_sel = lib[chosen]
    try:
        m = measure(audio_sel, int(sr_sel))
        dur = float(m["duration_s"] * 1000)
        o1, o2 = st.columns([2, 1], gap="large")
        win = o1.slider("Time window (ms)", 0.0, dur, (0.0, dur), step=max(dur / 400, 0.5), help="Zoom every time-based panel to the same interval, for example the first 80 ms to inspect the attack.")
        grid = o2.radio("Layout", ["Stacked", "Two columns"], horizontal=True)
        st.markdown(f'<div class="specimen">{html.escape(chosen)}<small>{m["sample_rate_hz"]:,} Hz</small><small>{m["samples"]:,} samples</small></div>', unsafe_allow_html=True)
        d = m["decay"]
        sheet([("Peak", f"{m['peak_dbfs']:.2f}", "dBFS"), ("RMS", f"{m['rms_dbfs']:.2f}", "dBFS"), ("Crest factor", f"{m['crest_db']:.2f}", "dB"),
               ("Spectral centroid", f"{m['spectral_centroid_hz']:.1f}", "Hz"), ("Dominant bin", f"{m['dominant_frequency_hz']:.1f}", "Hz"),
               ("Time to −20 dB", "n/a" if d["t_minus_20_db_ms"] is None else f"{d['t_minus_20_db_ms']:.0f}", "ms")])
        st.audio(wav_bytes(audio_sel, sr_sel), format="audio/wav")

        band_log = False
        if "Bands" in panels:
            band_log = st.toggle("Logarithmic band axis", value=True, help="The 20–60 Hz band dominates most kicks; a log axis makes the small bands legible.")
        ridge = m["ridge"] if ridge_on else None

        def p_wave():
            show(plot_waveform(m, win), "lab_wave")
            caption(f"Time-domain waveform of {html.escape(chosen)}.")

        def p_spectrogram():
            show(plot_spectrogram(audio_sel, int(sr_sel), win, FMIN, fmax, floor, LOG, cmap, ridge, ridge_conf), "lab_spectrogram")
            caption("STFT magnitude. Window length trades time resolution against frequency resolution." + (" Dotted line: autocorrelation ridge." if ridge_on else ""))

        def p_spectrum():
            show(plot_spectrum(m, FMIN, fmax, floor, LOG), "lab_spectrum")
            caption("Whole-file windowed magnitude spectrum relative to its own maximum. Dotted line marks the dominant bin.")

        def p_env():
            show(plot_envelope(m, win, floor), "lab_env")
            caption("Smoothed RMS envelope. Markers show the −6, −20 and −40 dB crossings measured from the envelope peak.")

        def p_ridge():
            show(plot_ridge(m, win, ridge_conf), "lab_ridge")
            caption("Frame-based autocorrelation estimate constrained to 25–250 Hz. It is an analysis aid, not an exact pitch meter.")

        def p_bands():
            show(plot_bands({chosen: m}, band_log), "lab_bands")
            caption("Share of spectral power in each band, relative to 20 Hz–12 kHz.")

        order = [("Waveform", p_wave), ("Spectrogram", p_spectrogram), ("Spectrum", p_spectrum), ("Envelope", p_env), ("Pitch ridge", p_ridge), ("Bands", p_bands)]
        active = [(n, fn) for n, fn in order if n in panels]
        if grid == "Two columns":
            for i in range(0, len(active), 2):
                cols = st.columns(2, gap="large")
                for col, (n, fn) in zip(cols, active[i:i + 2]):
                    with col:
                        sec(n)
                        fn()
        else:
            for n, fn in active:
                sec(n)
                fn()

        sec("Export")
        report = json_safe_metrics(m)
        report["source"] = chosen
        report["analysis_note"] = "Relative spectral measurements; autocorrelation ridge is an estimate, not a guaranteed F0 track."
        bands_df = pd.DataFrame({"band": [b for b, _ in BANDS], "energy_share_percent": [m["band_energy_pct"][k] for _, k in BANDS]})
        e1, e2, e3 = st.columns(3)
        e1.download_button("Measurement report (JSON)", json.dumps(report, indent=2, default=json_default), file_name=safe_name(chosen) + "_analysis.json", mime="application/json", **FULL)
        e2.download_button("Frequency bands (CSV)", bands_df.to_csv(index=False), file_name=safe_name(chosen) + "_bands.csv", mime="text/csv", **FULL)
        e3.download_button("Descriptors (CSV)", pd.DataFrame([feature_row(chosen, m)]).to_csv(index=False), file_name=safe_name(chosen) + "_descriptors.csv", mime="text/csv", **FULL)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analysis failed: {exc}")

# ───────────────────────────── compare ─────────────────────────────
with tab_cmp:
    FIG_COUNTER["n"] = 0
    lib = build_library()
    st.markdown("## Compare")
    st.caption("Overlay specimens, quantify differences against a reference and audition at matched level.")
    names = list(lib.keys())
    default_sel = [n for n in ["Slot A", "Slot B"] if n in lib] or [n for n in names if n[:2] in ("01", "03")][:2] or names[:2]
    sel = st.multiselect("Specimens", names, default=default_sel)
    if len(sel) < 1:
        note("Choose at least one specimen. Store slots A and B in the Synthesize tab, import WAV files in the sidebar, or pick factory samples.")
    else:
        ref = st.selectbox("Reference specimen", sel)
        data = {n: measure(lib[n][1], lib[n][0]) for n in sel}
        df = df_of(sel, lib)

        sec("Descriptor table")
        mode = st.radio("Show", ["Values", "Difference from reference"], horizontal=True, label_visibility="collapsed")
        view = df.copy()
        if mode != "Values":
            base = df[df["specimen"] == ref].iloc[0]
            for c in FEATURES:
                view[c] = df[c] - base[c]
        st.dataframe(view, hide_index=True, column_config=column_config(), **FULL)
        st.download_button("Download table (CSV)", df.to_csv(index=False), file_name="PP_comparison.csv", mime="text/csv")

        t_spec, t_env, t_band, t_exp, t_mat = st.tabs(["Spectra", "Envelopes", "Bands", "Explorer", "Matrix"])
        with t_spec:
            fig = go.Figure()
            for i, n in enumerate(sel):
                f_, db_ = np.asarray(data[n]["spectrum"]["frequency_hz"]), np.asarray(data[n]["spectrum"]["magnitude_db"])
                keep = (f_ >= FMIN) & (f_ <= fmax)
                fig.add_trace(go.Scatter(x=f_[keep], y=db_[keep], mode="lines", name=n, line=dict(color=SERIES[i % len(SERIES)], width=1.5), hovertemplate="%{y:.1f} dB"))
            fig.update_layout(showlegend=True)
            freq_axis(fig, LOG, FMIN, fmax)
            fig.update_yaxes(range=[floor, 3])
            show(style(fig, 340, "Frequency (Hz)", "Relative level (dB)"), "cmp_spec")
            caption("Spectra, each normalized to its own maximum. They show shape, not absolute level.")
        with t_env:
            fig = go.Figure()
            for i, n in enumerate(sel):
                e = data[n]["envelope"]
                fig.add_trace(go.Scatter(x=e["time_ms"], y=e["rms_db"], mode="lines", name=n, line=dict(color=SERIES[i % len(SERIES)], width=1.6), hovertemplate="%{y:.1f} dB"))
            fig.update_layout(showlegend=True)
            fig.update_yaxes(range=[max(floor, -90), 3])
            show(style(fig, 340, "Time (ms)", "RMS envelope (dB re peak)"), "cmp_env")
            caption("Smoothed RMS envelopes. Compare tail length before auditioning in a repeated kick and bass pattern.")
        with t_band:
            blog = st.toggle("Logarithmic axis", value=True, key="cmp_blog")
            show(plot_bands(data, blog), "cmp_bands")
            caption("Spectral power share per band for each specimen.")
        with t_exp:
            explorer(df, "cmp_exp")
            caption("Choose any two descriptors. These are signal measures, not quality scores.")
        with t_mat:
            feats = st.multiselect("Descriptors", list(FEATURES), default=[f for f in MATRIX_DEFAULT if df[f].notna().any()], format_func=axis_label)
            if feats and len(df) > 0:
                sub = df[feats].astype(float)
                span = (sub.max() - sub.min()).replace(0, np.nan)
                norm = ((sub - sub.min()) / span).fillna(0.5)
                fig = go.Figure(go.Heatmap(z=norm.values, x=[FEATURES[f][0] for f in feats], y=df["specimen"], text=sub.round(1).astype(str).values, texttemplate="%{text}",
                                           colorscale="Cividis", showscale=False, hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>", xgap=2, ygap=2))
                fig.update_yaxes(autorange="reversed")
                show(style(fig, 90 + 46 * len(df), hover="closest"), "cmp_matrix")
                caption("Descriptor matrix. Colour is min-max scaled within each column across the selected specimens; printed numbers are the real values.")

        sec("Audition", "level-matched playback")
        a1, a2, a3 = st.columns([1, 1, 1], gap="large")
        pick_a = a1.selectbox("First", sel, index=0, key="aud_a")
        pick_b = a2.selectbox("Second", sel, index=min(1, len(sel) - 1), key="aud_b")
        match = a3.toggle("Match RMS level", value=True, help="Attenuates the louder specimen to the RMS of the quieter one, so neither clips and level does not bias the comparison.")
        rms = {n: 10 ** (data[n]["rms_dbfs"] / 20.0) for n in (pick_a, pick_b)}
        target = min(rms.values())
        for col, n in zip(st.columns(2, gap="large"), (pick_a, pick_b)):
            with col:
                gain = target / rms[n] if match and rms[n] > 0 else 1.0
                st.markdown(f"**{html.escape(n)}**" + (f" · gain {20 * np.log10(gain):+.1f} dB" if match else ""))
                st.audio(wav_bytes(lib[n][1], lib[n][0], gain), format="audio/wav")
        st.caption("Listen first in solo, then in the same kick and bass loop. Spectral values do not predict warmth, chest impact or danceability.")

# ───────────────────────────── variations ─────────────────────────────
with tab_var:
    st.markdown("## Variation lab")
    st.caption("Generate a family of related kicks around the current settings. Audition, inspect and load any member back into the synthesizer.")
    vl, vr = st.columns([1, 1.2], gap="large")
    with vl:
        n_var = st.slider("Number of variations", 4, 32, 12, 1)
        depth = st.slider("Variation depth", 0.05, 1.0, 0.35, 0.05)
        vseed = st.number_input("Variation seed", 0, 999_999, 2026, 1)
        if st.button("Generate family", type="primary", **FULL):
            try:
                base, rng = current_params(), np.random.default_rng(int(vseed))
                items = []
                with st.spinner("Rendering and measuring"):
                    for i in range(int(n_var)):
                        v = make_variation(base, float(depth), rng)
                        audio = generate_kick(v, seed=int(vseed) + i + 1).astype(np.float32)
                        items.append({"name": f"Variation {i + 1:02d}", "sr": int(v.sample_rate), "audio": audio, "params": params_to_dict(v)})
                st.session_state.variations = items
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    manifest = []
                    for it in items:
                        zf.writestr(it["name"].replace(" ", "_") + ".wav", wav_bytes(it["audio"], it["sr"]))
                        manifest.append({"file": it["name"], "parameters": it["params"], "analysis": json_safe_metrics(measure(it["audio"], it["sr"]))})
                    zf.writestr("manifest.json", json.dumps({"project": "Particula Primordial Kick Engine", "created_utc": datetime.now(timezone.utc).isoformat(),
                                                             "seed": int(vseed), "depth": float(depth), "variations": manifest}, indent=2, default=json_default))
                st.session_state.batch_zip = mem.getvalue()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not generate variations: {exc}")
    with vr:
        note("Bounded perturbations explore the neighbourhood of the current kick rather than producing unrelated sounds. Sort by measurements to organize listening, then choose finalists by ear in the actual loop.")

    items = st.session_state.variations
    if items:
        FIG_COUNTER["n"] = 0
        vdf = pd.DataFrame([{**feature_row(it["name"], measure(it["audio"], it["sr"])), **{f"p_{k}": it["params"][k] for k in SLIDERS if k in it["params"]}} for it in items])
        sec("Family map")
        explorer(vdf, "var_exp")
        caption("Each point is one variation. Colour any descriptor to see where the family spreads.")

        sec("Parameters against outcomes")
        pcols = [f"p_{k}" for k in SLIDERS if f"p_{k}" in vdf and vdf[f"p_{k}"].nunique() > 1]
        fcols = [c for c in ("centroid_Hz", "crest_dB", "t_minus_20dB_ms", "rms_dBFS") if vdf[c].notna().all() and vdf[c].nunique() > 1]
        dims = [dict(label=SLIDERS[c[2:]][0].split(" (")[0], values=vdf[c]) for c in pcols] + [dict(label=FEATURES[c][0], values=vdf[c]) for c in fcols]
        if len(dims) >= 2:
            colour = vdf["centroid_Hz"] if "centroid_Hz" in fcols else vdf[pcols[0]]
            fig = go.Figure(go.Parcoords(line=dict(color=colour, colorscale="Viridis", showscale=False), dimensions=dims,
                                         labelfont=dict(family=MONO, size=10, color=INK), tickfont=dict(family=MONO, size=9, color=MUTED), rangefont=dict(family=MONO, size=9, color=MUTED)))
            fig.update_layout(height=340, margin=dict(l=40, r=40, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(family=MONO))
            show(fig, "var_par")
            caption("Parallel coordinates. Drag along any axis to select a range and see which parameters lead to which measured outcomes. Colour follows spectral centroid.")

        sec("Audition")
        pick = st.selectbox("Variation", [it["name"] for it in items])
        it = next(x for x in items if x["name"] == pick)
        st.audio(wav_bytes(it["audio"], it["sr"]), format="audio/wav")
        mv = measure(it["audio"], it["sr"])
        w1, w2 = st.columns(2, gap="large")
        with w1:
            show(plot_waveform(mv, (0.0, mv["duration_s"] * 1000)), "var_wave")
        with w2:
            show(plot_spectrum(mv, FMIN, fmax, floor, LOG), "var_spec")
        g1, g2, g3 = st.columns(3)
        g1.button("Load into synthesizer", on_click=load_variation, args=(pick,), **FULL)
        g2.download_button("Download this WAV", wav_bytes(it["audio"], it["sr"]), file_name=safe_name(pick) + ".wav", mime="audio/wav", **FULL)
        if st.session_state.batch_zip:
            g3.download_button("Download pack (ZIP)", st.session_state.batch_zip, file_name="Particula_Primordial_Kick_Variations.zip", mime="application/zip", **FULL)
        st.dataframe(vdf[["specimen"] + list(FEATURES)], hide_index=True, column_config=column_config(), **FULL)

# ───────────────────────────── notebook ─────────────────────────────
with tab_note:
    st.markdown("## Research notebook")
    st.caption("How the instrument works, what is measured, and what can legitimately be concluded.")
    note("A controllable kick is both a musical object and a measurable signal. The synthesizer creates the sound, the laboratory describes its measurable behaviour, and this notebook records method, evidence, interpretation and limitations.")
    nb = st.tabs(["Abstract", "Introduction", "Methods", "Results", "Limitations", "Learning path", "References"])
    with nb[0]:
        st.markdown("""
### Abstract

**Purpose.** This project develops a reproducible, parameterized software instrument for designing low-frequency kick drums and a companion analysis environment for inspecting their signal properties.

**Method.** A kick is synthesized from a time-varying sinusoidal oscillator, a sub component, weighted harmonic partials, a short transient, an amplitude envelope and a soft nonlinear saturation stage. The resulting waveform is measured using level statistics, a windowed Fourier spectrum, short-time Fourier analysis, RMS-envelope decay measurements and an autocorrelation-based low-frequency pitch estimate.

**Current status.** Version 0.2 establishes the instrument, five factory specimens, a browser-based laboratory and a reproducible workflow for exporting audio and measurements. The factory comparison provides descriptive signal measurements; it does not establish perceptual superiority or listener preference.

**Contribution.** The project connects procedural sound synthesis, visual signal inspection and a documented learning/research workflow for iterative kick design.

**Keywords:** procedural audio, kick drum synthesis, digital signal processing, spectral analysis, envelope, reproducibility, sound design.
""")
    with nb[1]:
        st.markdown("""
### 1. Introduction

A kick drum must perform several roles at once: establish rhythmic weight, occupy a low-frequency region, remain perceptible on different playback systems, and coexist with the bassline. These are related but distinct requirements. A waveform that looks large is not necessarily perceived as powerful; a high peak does not guarantee sustained low-frequency energy; and a strong sub component can consume headroom while translating poorly on small speakers.

The project therefore treats kick design as a controlled signal-generation problem with a musical evaluation stage. Parameters are exposed as interpretable controls rather than hidden behind a sample selector.

**Research question.** How can a compact procedural model expose useful controls over the pitch contour, spectral structure, transient and decay of a club-oriented kick while preserving reproducibility and making its output measurable?

**Subquestions.** (1) How do pitch-contour parameters affect the low-frequency trajectory? (2) How do body, sub and harmonic controls affect spectral-energy distribution? (3) How do attack and decay parameters affect the measured amplitude envelope? (4) Which descriptors help distinguish generated specimens without claiming to predict subjective quality?
""")
    with nb[2]:
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
    with nb[3]:
        st.markdown("""
### 3. Results and discussion

The five factory WAVs are the initial dataset. The laboratory calculates their measurements directly from the audio files. The comparison scatter plot places spectral centroid against time to -20 dB; it is a descriptive map of the current sample set, not a quality score.

**Interpretation rule:** report what the measurement supports. A longer time-to--20 dB indicates a longer measured envelope tail under this analysis method. It does not, by itself, prove that the kick sounds warmer, larger or more danceable. Those judgments require controlled listening and playback-context evaluation.

**Next experiment:** create parameter-controlled pairs in which only one factor changes (for example, transient level or pitch-decay time), match playback level, compare in the same loop and record listener judgments separately from signal measurements.
""")
        factory = factory_library()
        if factory:
            fdf = df_of(list(factory), factory)
            explorer(fdf, "nb_exp")
            caption("Factory specimens: any two measured descriptors, computed live from the bundled WAV files.")
            st.dataframe(fdf, hide_index=True, column_config=column_config(), **FULL)
            st.caption("Across these five samples the 20–60 Hz band holds roughly 98.7–99.7% of measured 20 Hz–12 kHz spectral power, and time to −20 dB ranges from roughly 960 to 1,650 ms. This describes the prototype samples, not a universal target.")
            st.download_button("Download initial dataset (CSV)", fdf.to_csv(index=False), file_name="PP_Initial_Factory_Dataset.csv", mime="text/csv")
        shots = [(FIG_DIR / "factory_measurement_map.png", "Static figure: spectral centroid versus measured envelope decay."),
                 (FIG_DIR / "factory_band_energy.png", "Static figure: measured spectral-energy distribution."),
                 (FIG_DIR / "factory_reference_spectrogram.png", "Static figure: STFT spectrogram of the Rounded Chest specimen.")]
        for path, cap in shots:
            if path.exists():
                st.image(str(path), caption=cap, **FULL)
    with nb[4]:
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
    with nb[5]:
        st.markdown("""
### Learning path

1. **Waveform and sampling:** amplitude, sample rate, duration, quantization and headroom.
2. **Oscillators and phase:** frequency as phase change; why pitch sweeps need phase integration.
3. **Envelopes:** attack, decay, sustain and release; amplitude contour versus pitch contour.
4. **Harmonics:** integer partials, spectral balance and why the fundamental is not the whole sound.
5. **Fourier analysis:** FFT, windowing, leakage, frequency resolution and logarithmic frequency axes.
6. **Time-frequency analysis:** STFT/spectrogram and the resolution trade-off.
7. **Measurement versus perception:** descriptors can compare signals, but listening in context determines musical usefulness.

**Practical exercise:** render one kick, export its JSON parameters, change only pitch-decay time, render again and compare the pitch ridge and spectrogram in the Compare tab. Repeat with only the transient amount, then compare the waveform and high-frequency energy share.
""")
    with nb[6]:
        st.markdown("""
### References and further reading

1. **Julius O. Smith III. Spectral Audio Signal Processing.** [Read online](https://www.dsprelated.com/freebooks/sasp/). Fourier analysis, spectra, STFT and time-frequency displays.
2. **Udo Zölzer (ed.). DAFX: Digital Audio Effects, 2nd edition.** [Wiley / DOI](https://doi.org/10.1002/9781119991298). Digital audio processing, nonlinear processing and spectral effects.
3. **SciPy. ShortTimeFFT documentation.** [Technical reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html). The STFT implementation used by the laboratory.
4. **SciPy. Signal processing tutorial.** [STFT explanation](https://docs.scipy.org/doc/scipy/tutorial/signal.html#short-time-fourier-transform). Windowed, overlapping Fourier analysis.
5. **ITU-R BS.1770-5.** [Official recommendation](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en). Programme loudness and true-peak metering. The current application does not implement this standard.

The complete bibliography and reading order are also available in `research/REFERENCES.md`.
""")

# ───────────────────────────── presets ─────────────────────────────
with tab_pre:
    st.markdown("## Factory characters")
    st.caption("Starting points, not finished rules. Adjust them to fit the kick and bass relationship in your track.")
    descriptions = {
        "01 · Rounded Chest": "Soft leading edge, stable low body, subtle upper harmonics. The reference starting point for a rounded club kick.",
        "02 · Velvet Pressure": "Longer and smoother, with restrained transient energy and a deeper sustained body.",
        "03 · Controlled Punch": "Shorter envelope, faster pitch movement and a more defined attack without an industrial edge.",
        "04 · Organic Pulse": "Slower pitch gesture and longer decay for a less mechanical, breathing quality.",
        "05 · Deep Ritual": "Low-tuned body, extended weight and minimal transient. Check the kick against your bassline for headroom.",
    }
    for name, text in descriptions.items():
        if name not in PRESETS:
            continue
        with st.expander(name, expanded=(name == "01 · Rounded Chest")):
            pc1, pc2 = st.columns([2, 1], gap="large")
            with pc1:
                st.write(text)
                st.button("Load into synthesizer", key=f"load_{name}", on_click=choose_preset, args=(name,))
            with pc2:
                pdict = params_to_dict(PRESETS[name])
                st.dataframe(pd.DataFrame({"parameter": list(pdict), "value": [str(v) for v in pdict.values()]}), hide_index=True, height=240, **FULL)
    sec("Sound-design method", "an ordered workflow")
    st.markdown("""
1. **Tune the body first.** Set the ending frequency and listen against the bassline.
2. **Shape the pitch fall.** Start frequency and pitch-decay time control the initial impact.
3. **Add weight before attack.** Increase body and sub carefully.
4. **Use harmonics sparingly.** Add enough upper partials for translation without making the kick clicky.
5. **Set the transient last.** Add only enough definition for the kick to remain audible in the loop.
6. **Export, measure, compare and curate.** Keep sounds that work in the complete groove, not only in solo.
""")

st.markdown('<div class="foot">Particula Primordial, computational sound design. Build 0.2.0.</div>', unsafe_allow_html=True)
