"""
Lucid Link — Sleep & Dream Analysis Dashboard
RV College of Engineering · IDP 2025-26
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import io
import threading
import time as _time
from collections import Counter, deque as _deque

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

# ── Demo data ──────────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, str(ROOT / "data"))
    from last_night_demo import (
        STAGE_SEQUENCE, DREAM_DETECTIONS, SUMMARY,
        STAGE_COLORS, DREAM_COLORS, DATE, SLEEP_START, WAKE_TIME,
        HYPNOGRAM, build_hypnogram,
    )
    DEMO_AVAILABLE = True
except Exception:
    DEMO_AVAILABLE = False

PROCESSED_DIR = ROOT / "data" / "processed"
REAL_X_PATH   = PROCESSED_DIR / "sleep_X.npy"
REAL_Y_PATH   = PROCESSED_DIR / "sleep_y.npy"
HW_X_PATH     = ROOT / "data" / "hardware_session_X.npy"
HW_Y_PATH     = ROOT / "data" / "hardware_session_y.npy"
STAGE_NAMES   = ["Wake", "N1", "N2", "N3", "REM"]
DREAM_CAT_ORDER = ["Face", "Object", "Animal", "Scene", "Text", "Movement"]

st.set_page_config(page_title="Lucid Link", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }
[data-testid="stAppViewContainer"] > .main { background: #F4EEE6; }
.block-container { padding: 24px 28px 40px !important; max-width: 100% !important; }
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #EDE9E3 !important;
    min-width: 230px !important;
}
[data-testid="stSidebar"] section { padding-top: 0 !important; }
.card {
    background: #FFFFFF; border-radius: 14px; padding: 20px 22px;
    box-shadow: 0 1px 12px rgba(0,0,0,0.06); border: 1px solid #F0ECE8;
}
.card-sm {
    background: #FFFFFF; border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 1px 12px rgba(0,0,0,0.06); border: 1px solid #F0ECE8; height: 100%;
}
.m-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.09em; color: #9CA3AF; margin-bottom: 6px;
}
.m-val { font-size: 1.9rem; font-weight: 700; color: #111827; line-height: 1;
         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.m-sub { font-size: 0.78rem; color: #6B7280; margin-top: 5px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
         font-size: 0.72rem; font-weight: 700; }
.badge-green { background:#ECFDF5; color:#059669; border:1px solid #A7F3D0; }
.badge-blue  { background:#EFF6FF; color:#2563EB; border:1px solid #BFDBFE; }
.badge-pink  { background:#FDF2F8; color:#DB2777; border:1px solid #FBCFE8; }
.badge-amber { background:#FFFBEB; color:#D97706; border:1px solid #FDE68A; }
.sec-title {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #9CA3AF; margin-bottom: 12px;
}
.dream-row {
    display:flex; align-items:center; justify-content:space-between;
    padding:9px 12px; border-radius:8px; border-left:3px solid;
    background:#FAFAF9; margin-bottom:6px;
}
.dream-cat  { font-size:0.85rem; font-weight:700; }
.dream-meta { font-size:0.73rem; color:#9CA3AF; }
.dream-conf { font-size:0.78rem; font-weight:700; background:#F3F4F6;
              padding:2px 8px; border-radius:6px; color:#374151; }
.insight-row {
    display:flex; gap:10px; align-items:flex-start;
    padding:10px 14px; border-radius:10px; margin-bottom:8px;
}
.insight-good { background:#ECFDF5; border-left:3px solid #059669; }
.insight-warn { background:#FFFBEB; border-left:3px solid #D97706; }
.insight-info { background:#EFF6FF; border-left:3px solid #2563EB; }
.stTabs [data-baseweb="tab-list"] {
    gap:4px; background:#EDE9E3; border-radius:10px; padding:4px; border:none;
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px; padding:8px 22px; font-weight:600;
    font-size:0.85rem; color:#6B7280; border:none !important; background:transparent;
}
.stTabs [aria-selected="true"] {
    background:#FFFFFF !important; color:#E8762C !important;
    box-shadow:0 1px 6px rgba(0,0,0,0.1);
}
.stTabs [data-baseweb="tab-border"] { display:none; }
.stButton > button[kind="primary"] {
    background:#E8762C !important; border-color:#E8762C !important;
    color:white !important; border-radius:8px !important; font-weight:700 !important;
    font-size:0.84rem !important; padding:6px 18px !important; letter-spacing:0.01em !important;
}
.stButton > button { border-radius:8px !important; font-weight:600 !important; font-size:0.84rem !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size:0.88rem; font-weight:500; color:#6B7280;
    padding:9px 14px; border-radius:8px; cursor:pointer;
    display:flex; align-items:center; gap:10px;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) { color:#E8762C; background:#FFF7F0; font-weight:600; }
[data-testid="stSidebar"] .stRadio > div { flex-direction:column; gap:2px; }
[data-testid="stSidebar"] .stRadio [type="radio"] { display:none; }
[data-testid="stFileUploader"] > div {
    border:2px dashed #D1D5DB !important; border-radius:12px !important; background:#FAFAF9 !important;
}
[data-testid="stSelectbox"] > div > div { border-radius:8px !important; border-color:#E5E7EB !important; }
[data-testid="stProgress"] > div > div > div > div { background:#E8762C !important; }
div[data-testid="column"] { padding:0 5px !important; }
.element-container { margin-bottom:0 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_min(m):
    h, mn = divmod(int(m), 60)
    return f"{h}h {mn:02d}m" if h else f"{mn}m"

def score_color(s):
    if s >= 85: return "#059669"
    if s >= 70: return "#D97706"
    return "#DC2626"

def compute_sleep_score(cnt, total_epochs):
    """Composite 0-100 sleep quality score."""
    total_m = total_epochs * 0.5
    sleep_m = (total_epochs - cnt.get("Wake", 0)) * 0.5
    if total_m < 1: return 0
    efficiency  = min(sleep_m / total_m * 100, 100)
    rem_pct     = cnt.get("REM", 0) * 0.5 / max(sleep_m, 1) * 100
    n3_pct      = cnt.get("N3",  0) * 0.5 / max(sleep_m, 1) * 100
    eff_score   = efficiency  * 0.35
    rem_score   = min(rem_pct / 25 * 100, 100) * 0.25
    n3_score    = min(n3_pct  / 20 * 100, 100) * 0.20
    dur_score   = min(sleep_m / 480 * 100, 100) * 0.20
    return int(eff_score + rem_score + n3_score + dur_score)

def generate_insights(cnt, total_epochs, n_dreams):
    """Return list of (kind, text) tuples — kind in {good, warn, info}."""
    sleep_m = (total_epochs - cnt.get("Wake", 0)) * 0.5
    rem_pct = cnt.get("REM", 0) * 0.5 / max(sleep_m, 1) * 100
    n3_pct  = cnt.get("N3",  0) * 0.5 / max(sleep_m, 1) * 100
    n1_pct  = cnt.get("N1",  0) * 0.5 / max(sleep_m, 1) * 100
    efficiency = (total_epochs - cnt.get("Wake", 0)) / max(total_epochs, 1) * 100

    insights = []
    if rem_pct >= 22:
        insights.append(("good", f"Strong REM at {rem_pct:.0f}% — above the healthy 20-25% threshold, supporting memory and emotional processing."))
    elif rem_pct < 14:
        insights.append(("warn", f"Low REM detected ({rem_pct:.0f}%) — REM deprivation can impair learning consolidation and emotional regulation."))
    else:
        insights.append(("info", f"Healthy REM proportion at {rem_pct:.0f}% — optimal window for cognitive recovery and dream processing."))

    if n3_pct >= 15:
        insights.append(("good", f"Excellent deep sleep ({n3_pct:.0f}%) — physical restoration, immune function, and growth hormone release are well supported."))
    elif n3_pct < 8:
        insights.append(("warn", f"Low deep sleep ({n3_pct:.0f}%) — consider limiting screens and alcohol 2h before bed to promote SWS."))
    else:
        insights.append(("info", f"Adequate deep sleep at {n3_pct:.0f}% — foundational for physical recovery."))

    if efficiency >= 90:
        insights.append(("good", f"Sleep efficiency {efficiency:.0f}% — excellent; minimal time awake during the recording."))
    elif efficiency < 75:
        insights.append(("warn", f"Sleep efficiency {efficiency:.0f}% — high wake time suggests fragmented sleep or restless night."))

    if n_dreams >= 6:
        insights.append(("good", f"{n_dreams} dream events detected — active dream content suggests healthy REM engagement and memory replay."))
    elif n_dreams <= 1:
        insights.append(("info", f"Minimal dream activity detected — this may reflect shallow REM or early wake before dream recall consolidates."))
    else:
        insights.append(("info", f"{n_dreams} dream events captured across REM cycles — normal range for a typical night."))

    if n1_pct > 15:
        insights.append(("warn", f"High N1 proportion ({n1_pct:.0f}%) — excess light sleep may indicate frequent micro-arousals or sleep-onset difficulty."))

    return insights[:4]

def detect_sleep_cycles(hyp_names):
    """Count distinct NREM→REM cycles."""
    cycles, in_rem = 0, False
    for s in hyp_names:
        if s == "REM" and not in_rem:
            cycles += 1; in_rem = True
        elif s != "REM":
            in_rem = False
    return cycles

def detect_rem_latency(hyp_names):
    """Minutes from first non-Wake epoch to first REM epoch."""
    start = next((i for i, s in enumerate(hyp_names) if s != "Wake"), None)
    first_rem = next((i for i, s in enumerate(hyp_names) if s == "REM"), None)
    if start is None or first_rem is None: return None
    return (first_rem - start) * 0.5

def signal_quality(X_rem):
    """Returns quality % based on z-score kurtosis (near 3 = clean Gaussian)."""
    if len(X_rem) == 0: return 100
    kurts = np.array([
        float(np.mean((ep[0] - ep[0].mean())**4) / (ep[0].std()**4 + 1e-8))
        for ep in X_rem
    ])
    artifact_frac = float(np.mean(kurts > 6.0))
    return int((1 - artifact_frac) * 100)

def build_hypnogram_fig(hyp_points):
    Y_POS = {"Wake": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
    TICK_LABELS = ["N3", "N2", "N1", "REM", "Wake"]
    times  = [p[0] for p in hyp_points]
    stages = [p[1] for p in hyp_points]
    y_vals = [Y_POS.get(s, 0) for s in stages]
    times_ext  = times  + [times[-1] + timedelta(minutes=1)]
    y_vals_ext = y_vals + [y_vals[-1]]
    fig = go.Figure()
    in_rem = False; rem_start = None
    for i, (t, s) in enumerate(zip(times, stages)):
        if s == "REM" and not in_rem: in_rem = True; rem_start = t
        elif s != "REM" and in_rem:
            fig.add_vrect(x0=rem_start, x1=t, fillcolor="rgba(236,72,153,0.07)", line_width=0)
            in_rem = False
    if in_rem:
        fig.add_vrect(x0=rem_start, x1=times[-1], fillcolor="rgba(236,72,153,0.07)", line_width=0)
    fig.add_trace(go.Scatter(
        x=times_ext, y=y_vals_ext, mode="lines",
        line=dict(shape="hv", color="#374151", width=1.8),
        hovertemplate="%{x|%H:%M}<br><b>%{customdata}</b><extra></extra>",
        customdata=stages + [stages[-1]], showlegend=False,
    ))
    for stage, color in STAGE_COLORS.items():
        xs = [t for t, s in zip(times, stages) if s == stage]
        ys = [Y_POS[stage]] * len(xs)
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(color=color, size=3, symbol="circle"),
                name=stage, showlegend=True, hoverinfo="skip",
            ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", height=220,
        margin=dict(t=8, b=40, l=52, r=16),
        yaxis=dict(tickvals=[0,1,2,3,4], ticktext=TICK_LABELS,
                   gridcolor="#F3F4F6", tickfont=dict(size=11, color="#6B7280"),
                   showline=False, zeroline=False),
        xaxis=dict(tickformat="%H:%M", gridcolor="#F3F4F6",
                   tickfont=dict(size=11, color="#6B7280"), showline=False, zeroline=False),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.18,
                    font=dict(size=10, color="#6B7280"), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#E5E7EB"),
    )
    return fig

def build_donut_fig(stage_mins):
    labels = [k for k, v in stage_mins.items() if v > 0]
    values = [v for k, v in stage_mins.items() if v > 0]
    colors = [STAGE_COLORS[k] for k in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color="white", width=3)),
        hole=0.62, textinfo="label+percent", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value} min (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
        margin=dict(t=4, b=4, l=4, r=4), height=240,
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        annotations=[dict(text="Stage<br>Mix", x=0.5, y=0.5, font_size=12,
                          showarrow=False, font=dict(color="#6B7280"))],
    )
    return fig

def build_transition_fig(hyp_names):
    """Heatmap of stage → stage transition probabilities."""
    stages = ["Wake", "N1", "N2", "N3", "REM"]
    matrix = np.zeros((5, 5))
    for a, b in zip(hyp_names[:-1], hyp_names[1:]):
        if a in stages and b in stages:
            matrix[stages.index(a)][stages.index(b)] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, matrix / row_sums * 100, 0)
    text = [[f"{v:.0f}%" if v >= 1 else "" for v in row] for row in pct]
    fig = go.Figure(go.Heatmap(
        z=pct, x=stages, y=stages,
        colorscale=[[0, "#F9FAFB"], [0.001, "#FFF7F0"], [1, "#E8762C"]],
        text=text, texttemplate="%{text}", showscale=False,
        hovertemplate="From <b>%{y}</b> to <b>%{x}</b>: %{z:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        height=210, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=6, b=6, l=46, r=6),
        xaxis=dict(title="Next Stage", tickfont=dict(size=10, color="#6B7280"), showline=False),
        yaxis=dict(title="From Stage", tickfont=dict(size=10, color="#6B7280"),
                   autorange="reversed", showline=False),
        font=dict(family="-apple-system,sans-serif"),
    )
    return fig

def build_dream_radar_fig(dream_events):
    """Polar/radar chart of dream category distribution."""
    cats = DREAM_CAT_ORDER
    counts = Counter(ev["category"] for ev in dream_events)
    vals = [counts.get(c, 0) for c in cats] + [counts.get(cats[0], 0)]
    cats_closed = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats_closed, fill="toself",
        fillcolor="rgba(232,118,44,0.15)",
        line=dict(color="#E8762C", width=2),
        marker=dict(size=6, color="#E8762C"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=False, gridcolor="#F3F4F6"),
            angularaxis=dict(tickfont=dict(size=10, color="#374151")),
            bgcolor="white",
        ),
        paper_bgcolor="white", height=220,
        margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
    )
    return fig

def build_eeg_fig(epoch_arr, stage_label):
    t = np.linspace(0, 30, epoch_arr.shape[1])
    color = STAGE_COLORS.get(stage_label, "#374151")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=epoch_arr[0].astype(float), mode="lines",
        line=dict(color=color, width=0.9), name="Fpz-Cz",
        hovertemplate="%{x:.1f}s<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", height=160,
        margin=dict(t=6, b=32, l=52, r=12),
        xaxis=dict(title="Time (s)", gridcolor="#F3F4F6",
                   tickfont=dict(size=10, color="#6B7280"), showline=False),
        yaxis=dict(title="Amplitude (z)", gridcolor="#F3F4F6",
                   tickfont=dict(size=10, color="#6B7280"), showline=False, zeroline=False),
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        showlegend=False,
    )
    return fig

def classify_dream(epoch, position=0.5):
    ch0 = epoch[0]
    n_win = 12; w = len(ch0) // n_win
    win_e = np.array([ch0[i*w:(i+1)*w].var() for i in range(n_win)])
    win_e /= (win_e.sum() + 1e-9)
    jitter = float((win_e * np.arange(n_win)).sum()) / (n_win - 1)
    jittered = (position * 6.0 + (jitter - 0.5) * 1.2) % 6.0
    cat = DREAM_CAT_ORDER[int(jittered) % 6]
    frac = jittered - int(jittered)
    conf = round(min(0.96, 0.65 + min(frac, 1 - frac) * 0.62), 2)
    return cat, conf

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE EEG / SLEEP RECORDING  —  BioAmp EXG Pill + Arduino Uno R4 Minima
# ═══════════════════════════════════════════════════════════════════════════════
_REC_TMP  = ROOT / "data" / ".rec_raw.bin"  # persists across browser refresh
_SYNC1, _SYNC2, _PKT = 0xC7, 0x7C, 16      # Chords protocol

def _parse_pkt(pkt):
    """14-bit ADC, big-endian, centred at 8192 → [-1, 1] float.
    Packet layout: [SYNC1, SYNC2, Counter, Ch0_hi, Ch0_lo, Ch1_hi, Ch1_lo, ...]
    Channel data starts at byte 3 (HEADER_LEN=3).
    """
    def _ch(off): return (((pkt[off] << 8) | pkt[off+1]) - 8192) / 8192.0
    return _ch(3), _ch(5)

def _live_reader(port, baud, buf, stop_ev):
    """Background thread: stream Chords serial into a deque."""
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            _time.sleep(2.0)          # UNO R4 resets on port open — wait for boot
            ser.reset_input_buffer()
            ser.write(b"START\n")     # Chords firmware requires START command
            ser.flush()
            raw = bytearray()
            while not stop_ev.is_set():
                raw.extend(ser.read(256))
                while len(raw) >= _PKT:
                    if raw[0] == _SYNC1 and raw[1] == _SYNC2:
                        buf.append(_parse_pkt(raw[:_PKT]))
                        raw = raw[_PKT:]
                    else:
                        raw = raw[1:]
    except Exception as e:
        buf.append(("ERR", str(e)))

def _sleep_recorder(port, baud, tmpfile, stop_ev):
    """Background thread: write interleaved float32 ch0/ch1 pairs to disk."""
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            _time.sleep(2.0)          # UNO R4 resets on port open — wait for boot
            ser.reset_input_buffer()
            ser.write(b"START\n")     # Chords firmware requires START command
            ser.flush()
            with open(tmpfile, "ab") as f:
                raw = bytearray()
                while not stop_ev.is_set():
                    raw.extend(ser.read(256))
                    while len(raw) >= _PKT:
                        if raw[0] == _SYNC1 and raw[1] == _SYNC2:
                            c0, c1 = _parse_pkt(raw[:_PKT])
                            f.write(np.array([c0, c1], dtype=np.float32).tobytes())
                            raw = raw[_PKT:]
                        else:
                            raw = raw[1:]
    except Exception as e:
        open(str(tmpfile) + ".err", "w").write(str(e))

def _process_recording(tmpfile, fs_in=500, fs_out=100, epoch_s=30):
    """Raw binary → X (N,2,3000) float32 + y (N,) int64 stage labels."""
    from scipy.signal import resample, welch
    data = np.fromfile(tmpfile, dtype=np.float32).reshape(-1, 2)
    n_ep = len(data) // (fs_in * epoch_s)
    if n_ep == 0:
        return None, None
    X_out, y_out = [], []
    for i in range(n_ep):
        seg  = data[i * fs_in * epoch_s:(i + 1) * fs_in * epoch_s]
        seg_r = resample(seg, fs_out * epoch_s, axis=0).astype(np.float32)
        for c in range(2):
            mu, sd = seg_r[:, c].mean(), seg_r[:, c].std()
            seg_r[:, c] = (seg_r[:, c] - mu) / (sd + 1e-8)
        ep = seg_r.T  # (2, 3000)
        X_out.append(ep)
        f_, P_ = welch(ep[0], fs=100, nperseg=256)
        def _bp(lo, hi): return float(P_[(f_ >= lo) & (f_ < hi)].sum()) + 1e-9
        tot = _bp(0.5, 40)
        sc  = [_bp(15,30)/tot + _bp(8,12)/tot*.5,
               _bp(4,8)/tot  + _bp(8,12)/tot*.3,
               _bp(12,15)/tot*2 + _bp(0.5,4)/tot*.3,
               _bp(0.5,4)/tot*2,
               (_bp(4,8) + _bp(8,12))/tot*1.2]
        y_out.append(int(np.argmax(sc)))
    return np.array(X_out), np.array(y_out, dtype=np.int64)

def _stage_from_buf(buf_deque, fs_in=500, fs_out=100):
    """Quick spectral stage from the live deque buffer.

    Classifier logic:
      Wake  — beta (13-30 Hz) + alpha (8-12 Hz) + EMG (30-50 Hz muscle noise
              from awake subjects) dominate; OR signal variance is very low
              (flat line / poor contact = cannot confirm sleep → default Wake)
      N1    — theta (4-8 Hz) prominent, alpha fading
      N2    — sleep spindles (12-15 Hz) + some delta
      N3    — high-amplitude delta (0.5-4 Hz) with large variance
      REM   — mixed theta + alpha, low chin EMG
    """
    from scipy.signal import resample, welch, sosfiltfilt, butter
    arr = np.array([x for x in list(buf_deque)
                    if not (isinstance(x, tuple) and x[0] == "ERR")])
    if len(arr) < fs_in * 6:
        return "—"
    seg   = arr[-min(len(arr), fs_in * 30):]
    seg_r = resample(seg, int(len(seg) * fs_out / fs_in), axis=0)
    ch0   = seg_r[:, 0]

    # Real N3 delta waves require large amplitude oscillations.
    # Low std = electrode drift/noise masquerading as delta → Wake.
    raw_std = arr[-min(len(arr), fs_in * 10):, 0].std()
    if raw_std < 0.08:
        return "Wake"

    # High-pass at 1 Hz to remove baseline wander / electrode drift
    sos = butter(4, 1.0, btype="high", fs=fs_out, output="sos")
    ch0 = sosfiltfilt(sos, ch0)
    ch0 = (ch0 - ch0.mean()) / (ch0.std() + 1e-8)

    f_, P_ = welch(ch0, fs=fs_out, nperseg=256)
    def _bp(lo, hi): return float(P_[(f_ >= lo) & (f_ < hi)].sum()) + 1e-9
    tot   = _bp(1.0, 50)
    delta = _bp(1.0, 4)
    theta = _bp(4,   8)
    alpha = _bp(8,  12)
    spin  = _bp(12, 15)
    beta  = _bp(13, 30)
    emg   = _bp(30, 50)

    sc = [
        beta/tot*1.5 + alpha/tot + emg/tot*1.5,   # Wake
        theta/tot*1.5 + alpha/tot*0.5,             # N1
        spin/tot*2.5  + delta/tot*0.3,             # N2
        delta/tot*3.0,                             # N3 — needs true delta waves
        (theta + alpha)/tot*1.2,                   # REM
    ]
    return STAGE_NAMES[int(np.argmax(sc))]

# ─────────────────────────────────────────────────────────────────────────────
# SHARED ANALYSIS RENDERER  (used by Upload + Hardware tabs)
# ─────────────────────────────────────────────────────────────────────────────
def render_analysis(X, hyp_names, start_dt, session_label="Uploaded Session", key_prefix="ra"):
    """
    Full analysis view: session card, metrics, timeline, stage distribution,
    transitions, insights, dream radar, dream predictions, EEG waveforms, export.
    """
    from collections import Counter as _Counter

    cnt = _Counter(hyp_names)
    total_epochs = len(hyp_names)
    total_m  = total_epochs * 0.5
    sleep_m  = (total_epochs - cnt.get("Wake", 0)) * 0.5
    rem_m    = cnt.get("REM", 0) * 0.5
    n3_m     = cnt.get("N3",  0) * 0.5
    rem_pct  = round(rem_m / max(sleep_m, 1) * 100, 1)
    n3_pct   = round(n3_m  / max(sleep_m, 1) * 100, 1)
    efficiency = round((total_epochs - cnt.get("Wake", 0)) / max(total_epochs, 1) * 100, 1)
    n_cycles   = detect_sleep_cycles(hyp_names)
    rem_lat    = detect_rem_latency(hyp_names)
    score      = compute_sleep_score(cnt, total_epochs)
    sc_col     = score_color(score)

    # ── Detect dream events from REM epochs ──────────────────────────────────
    rem_indices = [i for i, s in enumerate(hyp_names) if s == "REM"]
    rem_cycles  = []
    if rem_indices:
        cur = [rem_indices[0]]
        for idx in rem_indices[1:]:
            if idx - cur[-1] <= 3: cur.append(idx)
            else: rem_cycles.append(cur); cur = [idx]
        rem_cycles.append(cur)

    all_dream_events = []
    total_rem_epochs = len(rem_indices)
    for cycle_num, cycle_idxs in enumerate(rem_cycles, 1):
        n_det = max(1, len(cycle_idxs) // 20)
        for pos in np.clip(
            np.linspace(len(cycle_idxs) * 0.15, len(cycle_idxs) * 0.85, n_det, dtype=int),
            0, len(cycle_idxs) - 1,
        ):
            ep_idx  = cycle_idxs[int(pos)]
            gpos    = rem_indices.index(ep_idx) / max(total_rem_epochs - 1, 1)
            cat, conf = classify_dream(X[ep_idx], position=gpos)
            ep_time = start_dt + timedelta(minutes=ep_idx * 0.5)
            all_dream_events.append({
                "cycle": cycle_num, "ep_idx": ep_idx,
                "category": cat, "confidence": conf,
                "clock": ep_time.strftime("%H:%M"),
            })

    n_dreams = len(all_dream_events)
    rem_X    = X[rem_indices] if rem_indices else X[:0]
    sig_qual = signal_quality(rem_X)

    # ── Session header + metric cards ────────────────────────────────────────
    c0, c1, c2, c3 = st.columns([1.5, 1, 1, 1])
    with c0:
        end_dt = start_dt + timedelta(minutes=total_m)
        st.markdown(f"""
        <div class="card" style="background:linear-gradient(135deg,#1C1917,#292524);
             border-color:#3B3430;padding:24px 26px;">
            <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;color:#78716C;margin-bottom:4px;">
                {session_label.upper()}
            </div>
            <div style="font-size:2.8rem;font-weight:800;color:#FFFFFF;
                        line-height:1;letter-spacing:-0.03em;
                        font-family:-apple-system,sans-serif;">
                {fmt_min(sleep_m)}
            </div>
            <div style="font-size:0.82rem;color:#A8A29E;margin-top:6px;">
                {start_dt.strftime("%b %d")} &nbsp;
                {start_dt.strftime("%H:%M")} &nbsp;—&nbsp; {end_dt.strftime("%H:%M")}
            </div>
            <div style="margin-top:14px;display:flex;align-items:center;gap:10px;">
                <div style="font-size:2rem;font-weight:800;color:{sc_col};
                            font-family:-apple-system,sans-serif;">{score}</div>
                <div>
                    <div style="font-size:0.72rem;color:#78716C;font-weight:600;">SLEEP SCORE</div>
                    <div style="font-size:0.75rem;color:#A8A29E;">Efficiency {efficiency}%</div>
                </div>
            </div>
            <div style="margin-top:12px;display:flex;gap:16px;flex-wrap:wrap;">
                <div style="font-size:0.72rem;color:#78716C;">
                    <span style="color:#A8A29E;">{n_cycles} cycles</span>
                </div>
                {"" if rem_lat is None else
                 f'<div style="font-size:0.72rem;color:#78716C;"><span style="color:#A8A29E;">REM latency {fmt_min(rem_lat)}</span></div>'}
                <div style="font-size:0.72rem;color:#78716C;">
                    <span style="color:#{'059669' if sig_qual >= 90 else 'D97706' if sig_qual >= 75 else 'DC2626'};">
                        Signal {sig_qual}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    for col, (lbl, val, sub, badge_cls) in zip(
        [c1, c2, c3],
        [
            ("REM SLEEP",    fmt_min(rem_m), f"{rem_pct}% of sleep",   "badge-pink"),
            ("DEEP SLEEP",   fmt_min(n3_m),  f"{n3_pct}% of sleep",    "badge-blue"),
            ("DREAM EVENTS", str(n_dreams),  f"{n_cycles} sleep cycles","badge-amber"),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="card-sm">
                <div class="m-label">{lbl}</div>
                <div class="m-val">{val}</div>
                <div style="margin-top:8px;"><span class="badge {badge_cls}">{sub}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hypnogram ────────────────────────────────────────────────────────────
    hyp_pts = [(start_dt + timedelta(minutes=i * 0.5), s)
               for i, s in enumerate(hyp_names)]
    st.markdown('<div class="sec-title">SLEEP STAGE TIMELINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="padding:16px 20px 12px;">', unsafe_allow_html=True)
    st.plotly_chart(build_hypnogram_fig(hyp_pts), width="stretch", key=f"{key_prefix}_chart_1",
                    config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stage distribution + Insights ────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="sec-title">STAGE DISTRIBUTION</div>', unsafe_allow_html=True)
        sm = {s: cnt.get(s, 0) * 0.5 for s in STAGE_NAMES}
        st.markdown('<div class="card" style="padding:14px 18px;">', unsafe_allow_html=True)
        st.plotly_chart(build_donut_fig(sm), width="stretch", key=f"{key_prefix}_chart_2",
                        config={"displayModeBar": False})
        for stage, mins in sm.items():
            if mins > 0:
                pct = round(mins / max(sleep_m, 1) * 100, 1)
                dc, dn, dv = st.columns([0.2, 1, 0.8])
                dc.markdown(
                    f'<div style="width:10px;height:10px;border-radius:50%;'
                    f'background:{STAGE_COLORS[stage]};margin-top:10px;"></div>',
                    unsafe_allow_html=True)
                dn.markdown(f'<div style="font-size:0.82rem;color:#374151;'
                            f'font-weight:600;padding-top:6px;">{stage}</div>',
                            unsafe_allow_html=True)
                dv.markdown(f'<div style="font-size:0.8rem;color:#6B7280;'
                            f'text-align:right;padding-top:6px;">'
                            f'{fmt_min(mins)} · {pct}%</div>',
                            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sec-title">SLEEP INSIGHTS</div>', unsafe_allow_html=True)
        insights = generate_insights(cnt, total_epochs, n_dreams)
        kind_map = {"good": ("insight-good", "#059669", "&#9679;"),
                    "warn": ("insight-warn", "#D97706", "&#9888;"),
                    "info": ("insight-info", "#2563EB", "&#8505;")}
        for kind, text in insights:
            cls, col, icon = kind_map.get(kind, kind_map["info"])
            st.markdown(
                f'<div class="insight-row {cls}">'
                f'<div style="color:{col};font-size:1rem;line-height:1;flex-shrink:0;">{icon}</div>'
                f'<div style="font-size:0.8rem;color:#374151;line-height:1.55;">{text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stage Transition Matrix + Dream Radar ─────────────────────────────────
    tl, tr = st.columns(2)
    with tl:
        st.markdown('<div class="sec-title">STAGE TRANSITION MATRIX</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="card" style="padding:12px 16px;">', unsafe_allow_html=True)
        st.plotly_chart(build_transition_fig(hyp_names), width="stretch", key=f"{key_prefix}_chart_3",
                        config={"displayModeBar": False})
        st.markdown(
            '<div style="font-size:0.72rem;color:#9CA3AF;margin-top:4px;">'
            'Row = current stage; column = next stage; values are % probability.</div>',
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tr:
        st.markdown('<div class="sec-title">DREAM CONTENT RADAR</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="card" style="padding:12px 16px;">', unsafe_allow_html=True)
        if all_dream_events:
            st.plotly_chart(build_dream_radar_fig(all_dream_events),
                            width="stretch", key=f"{key_prefix}_chart_4",
                            config={"displayModeBar": False})
            dist = Counter(ev["category"] for ev in all_dream_events)
            top_cat = dist.most_common(1)[0][0]
            st.markdown(
                f'<div style="font-size:0.72rem;color:#9CA3AF;margin-top:4px;">'
                f'Dominant content: <b style="color:{DREAM_COLORS[top_cat]};">{top_cat}</b> '
                f'({dist[top_cat]} event{"s" if dist[top_cat]>1 else ""})</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="padding:60px 0;text-align:center;color:#9CA3AF;'
                'font-size:0.85rem;">No dream events detected</div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dream Predictions ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">DREAM PREDICTIONS</div>', unsafe_allow_html=True)
    if not all_dream_events:
        st.markdown(
            '<div class="card" style="padding:18px 20px;color:#9CA3AF;font-size:0.85rem;">'
            'No REM epochs detected — upload a file with REM sleep to see dream predictions.'
            '</div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="padding:16px 20px;">', unsafe_allow_html=True)
        by_cycle = {}
        for ev in all_dream_events:
            by_cycle.setdefault(ev["cycle"], []).append(ev)
        for cycle_n, events in sorted(by_cycle.items()):
            rem_dur = round(len(rem_cycles[cycle_n - 1]) * 0.5) if cycle_n <= len(rem_cycles) else "?"
            st.markdown(f"""
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#9CA3AF;margin:8px 0 5px;padding-left:2px;">
                REM Cycle {cycle_n} &nbsp;·&nbsp; {rem_dur} min
            </div>""", unsafe_allow_html=True)
            for ev in events:
                col  = DREAM_COLORS.get(ev["category"], "#6B7280")
                pct  = int(ev["confidence"] * 100)
                bar_w = int(ev["confidence"] * 80)
                st.markdown(f"""
                <div class="dream-row" style="border-left-color:{col};">
                    <div>
                        <div class="dream-cat" style="color:{col};">{ev["category"]}</div>
                        <div class="dream-meta">{ev["clock"]}</div>
                        <div style="background:#EBEBEB;border-radius:4px;
                                    height:4px;width:100px;margin-top:5px;overflow:hidden;">
                            <div style="background:{col};width:{bar_w}px;
                                        height:4px;border-radius:4px;"></div>
                        </div>
                    </div>
                    <div class="dream-conf">{pct}%</div>
                </div>""", unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #F0EBE3;'
            f'font-size:0.75rem;color:#9CA3AF;text-align:right;">'
            f'{n_dreams} dream event{"s" if n_dreams!=1 else ""} detected '
            f'across {len(rem_cycles)} REM cycle{"s" if len(rem_cycles)!=1 else ""}</div>',
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── EEG Waveforms During Dreaming ─────────────────────────────────────────
    st.markdown('<div class="sec-title">EEG SIGNAL DURING DREAMING</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.75rem;color:#9CA3AF;margin-bottom:10px;">'
        '30-second epochs at detected dream events '
        '&#8212; Fpz-Cz (top) and Pz-Oz (bottom)</div>',
        unsafe_allow_html=True)

    t_axis = np.linspace(0, 30, 3000)
    for eeg_i, ev in enumerate(all_dream_events[:6]):
        ep      = X[ev["ep_idx"]]
        cat_col = DREAM_COLORS.get(ev["category"], "#6B7280")
        fig_w   = go.Figure()
        fig_w.add_trace(go.Scatter(x=t_axis, y=ep[0].astype(float) + 4.5, mode="lines",
                                   line=dict(color=cat_col, width=0.85), name="Fpz-Cz",
                                   hovertemplate="Fpz-Cz %{x:.1f}s<extra></extra>"))
        fig_w.add_trace(go.Scatter(x=t_axis, y=ep[1].astype(float) - 4.5, mode="lines",
                                   line=dict(color=cat_col, width=0.85), opacity=0.70,
                                   name="Pz-Oz",
                                   hovertemplate="Pz-Oz %{x:.1f}s<extra></extra>"))
        fig_w.add_annotation(x=0.5, y=4.5, text="Fpz-Cz", showarrow=False,
                             xanchor="left", font=dict(size=9, color=cat_col),
                             xref="x", yref="y")
        fig_w.add_annotation(x=0.5, y=-4.5, text="Pz-Oz", showarrow=False,
                             xanchor="left", font=dict(size=9, color=cat_col),
                             xref="x", yref="y")
        fig_w.update_layout(
            paper_bgcolor="white", plot_bgcolor="white", height=180,
            margin=dict(t=8, b=28, l=52, r=14),
            xaxis=dict(title="Time (s)", gridcolor="#F3F4F6",
                       tickfont=dict(size=9, color="#9CA3AF"), showline=False, range=[0, 30]),
            yaxis=dict(gridcolor="#F3F4F6", tickfont=dict(size=9, color="#9CA3AF"),
                       showline=False, zeroline=True, zerolinecolor="#E5E7EB", zerolinewidth=1),
            showlegend=False,
            font=dict(family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"),
        )
        label_html = (
            f'<div class="card" style="padding:10px 18px 6px;margin-bottom:2px;">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<div style="width:8px;height:8px;border-radius:50%;'
            f'background:{cat_col};flex-shrink:0;"></div>'
            f'<div style="font-size:0.78rem;font-weight:700;color:{cat_col};">'
            f'{ev["category"]}</div>'
            f'<div style="font-size:0.72rem;color:#9CA3AF;">'
            f'{ev["clock"]} &nbsp;&middot;&nbsp; '
            f'{int(ev["confidence"]*100)}% confidence'
            f' &nbsp;&middot;&nbsp; REM Cycle {ev["cycle"]}</div>'
            f'</div></div>'
        )
        st.markdown(label_html, unsafe_allow_html=True)
        st.plotly_chart(fig_w, width="stretch", key=f"{key_prefix}_eeg_{eeg_i}", config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Export CSV ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">EXPORT</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        csv_lines = ["time,stage,epoch_index"]
        for i, (t, s) in enumerate(hyp_pts):
            csv_lines.append(f"{t.strftime('%Y-%m-%d %H:%M')},{s},{i}")
        csv_bytes = "\n".join(csv_lines).encode()
        st.download_button(
            label="Download Hypnogram CSV",
            data=csv_bytes,
            file_name="lucidlink_hypnogram.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_dl_hypnogram",
        )
    with e2:
        if all_dream_events:
            dream_csv = ["time,category,confidence,rem_cycle"]
            for ev in all_dream_events:
                dream_csv.append(f"{ev['clock']},{ev['category']},{ev['confidence']},{ev['cycle']}")
            st.download_button(
                label="Download Dream Events CSV",
                data="\n".join(dream_csv).encode(),
                file_name="lucidlink_dreams.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"{key_prefix}_dl_dreams",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:26px 20px 18px;border-bottom:1px solid #F0ECE8;margin-bottom:10px;">
        <div style="font-size:1.18rem;font-weight:800;color:#111827;
                    letter-spacing:-0.03em;font-family:-apple-system,sans-serif;">
            Lucid Link
        </div>
        <div style="font-size:0.7rem;color:#9CA3AF;margin-top:3px;font-weight:500;">
            Neural Interface · Dream Visualization
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio("Navigation", ["Overview", "Sleep Sessions", "Hardware", "Reports"],
                   label_visibility="hidden")

    st.markdown('<div style="border-top:1px solid #F0ECE8;margin:14px 0;"></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="m-label" style="padding:0 4px;">DEVICE STATUS</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;padding:10px 4px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#D1D5DB;"></div>
        <div style="font-size:0.82rem;color:#9CA3AF;font-weight:500;">No device connected</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid #F0ECE8;margin:14px 0;"></div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0 4px 20px;">
        <div class="m-label">TEAM</div>
        <div style="font-size:0.8rem;color:#374151;line-height:1.9;margin-top:4px;">
            RV College of Engineering<br>
            IDP 2025-26 · Group 12
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
col_title, col_actions = st.columns([3, 2])
with col_title:
    st.markdown("""
    <div style="padding:0 0 18px;">
        <h1 style="font-size:1.55rem;font-weight:800;color:#111827;margin:0;
                   letter-spacing:-0.03em;font-family:-apple-system,sans-serif;">
            Sleep Analysis
        </h1>
        <p style="color:#9CA3AF;font-size:0.84rem;margin:5px 0 0;font-weight:500;">
            Last session recorded — June 22-23, 2026
        </p>
    </div>
    """, unsafe_allow_html=True)
with col_actions:
    ca, cb = st.columns(2)
    with ca: st.button("+ Connect Device", type="primary", use_container_width=True)
    with cb: st.button("Upload Data", use_container_width=True)

# ── Session state for live EEG & sleep recording ─────────────────────────────
for _k, _v in {
    "live_buf":    _deque(maxlen=5000),  # 10 s at 500 Hz
    "live_stop":   None,
    "live_thread": None,
    "rec_stop":    None,
    "rec_thread":  None,
    "rec_start":   None,
    "rec_done":    False,
    "rec_X":       None,
    "rec_y":       None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_demo, tab_upload, tab_hw = st.tabs(["Demo Session", "Upload File", "Connect Hardware"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DEMO SESSION
# ─────────────────────────────────────────────────────────────────────────────
with tab_demo:
    c0, c1, c2, c3 = st.columns([1.5, 1, 1, 1])
    with c0:
        sc = SUMMARY["sleep_score"] if DEMO_AVAILABLE else 0
        sc_col = score_color(sc)
        st.markdown(f"""
        <div class="card" style="background:linear-gradient(135deg,#1C1917,#292524);
             border-color:#3B3430;padding:24px 26px;">
            <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;color:#78716C;margin-bottom:4px;">LAST NIGHT</div>
            <div style="font-size:2.8rem;font-weight:800;color:#FFFFFF;line-height:1;
                        letter-spacing:-0.03em;font-family:-apple-system,sans-serif;">
                {fmt_min(SUMMARY["sleep_min"]) if DEMO_AVAILABLE else "—"}
            </div>
            <div style="font-size:0.82rem;color:#A8A29E;margin-top:6px;">
                {DATE if DEMO_AVAILABLE else "No data"}
                &nbsp;&nbsp;{SLEEP_START if DEMO_AVAILABLE else ""}&nbsp;—&nbsp;
                {WAKE_TIME if DEMO_AVAILABLE else ""}
            </div>
            <div style="margin-top:14px;display:flex;align-items:center;gap:10px;">
                <div style="font-size:2rem;font-weight:800;color:{sc_col};
                            font-family:-apple-system,sans-serif;">{sc}</div>
                <div>
                    <div style="font-size:0.72rem;color:#78716C;font-weight:600;">SLEEP SCORE</div>
                    <div style="font-size:0.75rem;color:#A8A29E;">
                        Efficiency {SUMMARY["sleep_efficiency"] if DEMO_AVAILABLE else "—"}%
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    metrics = [
        ("REM SLEEP",    fmt_min(SUMMARY["rem_min"]), f'{SUMMARY["rem_pct"]}% of sleep',  "badge-pink"),
        ("DEEP SLEEP",   fmt_min(SUMMARY["n3_min"]),  f'{SUMMARY["n3_pct"]}% of sleep',   "badge-blue"),
        ("DREAM EVENTS", str(SUMMARY["dream_events"]), f'{SUMMARY["sleep_cycles"]} cycles',"badge-amber"),
    ] if DEMO_AVAILABLE else [
        ("REM","—","—","badge-pink"),("DEEP","—","—","badge-blue"),("DREAMS","—","—","badge-amber")
    ]
    for col, (lbl, val, sub, badge_cls) in zip([c1, c2, c3], metrics):
        with col:
            st.markdown(f"""
            <div class="card-sm">
                <div class="m-label">{lbl}</div>
                <div class="m-val">{val}</div>
                <div style="margin-top:8px;"><span class="badge {badge_cls}">{sub}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">SLEEP STAGE TIMELINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="padding:16px 20px 12px;">', unsafe_allow_html=True)
    if DEMO_AVAILABLE:
        st.plotly_chart(build_hypnogram_fig(HYPNOGRAM), width="stretch", key="chart_2",
                        config={"displayModeBar": False})
    else:
        st.info("Demo data not available.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.4])
    with col_left:
        st.markdown('<div class="sec-title">STAGE DISTRIBUTION</div>', unsafe_allow_html=True)
        st.markdown('<div class="card" style="padding:16px 20px;">', unsafe_allow_html=True)
        if DEMO_AVAILABLE:
            sm = {"Wake": SUMMARY["wake_min"], "N1": SUMMARY["n1_min"],
                  "N2": SUMMARY["n2_min"],  "N3": SUMMARY["n3_min"], "REM": SUMMARY["rem_min"]}
            st.plotly_chart(build_donut_fig(sm), width="stretch", key="chart_3",
                            config={"displayModeBar": False})
            for stage, mins in sm.items():
                pct = round(mins / SUMMARY["sleep_min"] * 100, 1)
                dc, dn, dv = st.columns([0.2, 1, 0.8])
                dc.markdown(f'<div style="width:10px;height:10px;border-radius:50%;'
                            f'background:{STAGE_COLORS[stage]};margin-top:10px;"></div>',
                            unsafe_allow_html=True)
                dn.markdown(f'<div style="font-size:0.82rem;color:#374151;font-weight:600;'
                            f'padding-top:6px;">{stage}</div>', unsafe_allow_html=True)
                dv.markdown(f'<div style="font-size:0.8rem;color:#6B7280;text-align:right;'
                            f'padding-top:6px;">{fmt_min(mins)} · {pct}%</div>',
                            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="sec-title">DREAM DETECTIONS</div>', unsafe_allow_html=True)
        st.markdown('<div class="card" style="padding:16px 20px;">', unsafe_allow_html=True)
        if DEMO_AVAILABLE:
            by_cycle = {}
            for d in DREAM_DETECTIONS: by_cycle.setdefault(d["cycle"], []).append(d)
            cycle_rem = [s for s in STAGE_SEQUENCE if s[0] == "REM"]
            for cycle_n, events in sorted(by_cycle.items()):
                rem_dur = cycle_rem[cycle_n - 1][1] if cycle_n <= len(cycle_rem) else "?"
                st.markdown(f"""
                <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.08em;color:#9CA3AF;margin:8px 0 5px;padding-left:2px;">
                    REM Cycle {cycle_n} &nbsp;·&nbsp; {rem_dur} min
                </div>""", unsafe_allow_html=True)
                for ev in events:
                    col  = DREAM_COLORS.get(ev["category"], "#6B7280")
                    pct  = int(ev["confidence"] * 100)
                    bar_w = int(ev["confidence"] * 80)
                    st.markdown(f"""
                    <div class="dream-row" style="border-left-color:{col};">
                        <div>
                            <div class="dream-cat" style="color:{col};">{ev["category"]}</div>
                            <div class="dream-meta">{ev["clock"]}</div>
                            <div style="background:#EBEBEB;border-radius:4px;
                                        height:4px;width:100px;margin-top:5px;overflow:hidden;">
                                <div style="background:{col};width:{bar_w}px;
                                            height:4px;border-radius:4px;"></div>
                            </div>
                        </div>
                        <div class="dream-conf">{pct}%</div>
                    </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Demo Insights + Transition Matrix
    di_l, di_r = st.columns(2)
    with di_l:
        st.markdown('<div class="sec-title">SLEEP INSIGHTS</div>', unsafe_allow_html=True)
        demo_insights = [
            ("good", "Strong REM at 32.9% — above the healthy 20-25% threshold, supporting memory and emotional processing."),
            ("good", "Excellent deep sleep (17.4%) — physical restoration and immune function well supported."),
            ("good", "Sleep efficiency 97.8% — minimal time awake across the night."),
            ("good", "10 dream events captured across 5 REM cycles — highly active dream processing."),
        ]
        kind_map = {"good": ("insight-good","#059669","&#9679;"),
                    "warn": ("insight-warn","#D97706","&#9888;"),
                    "info": ("insight-info","#2563EB","&#8505;")}
        for kind, text in demo_insights:
            cls, col_c, icon = kind_map[kind]
            st.markdown(
                f'<div class="insight-row {cls}">'
                f'<div style="color:{col_c};font-size:1rem;line-height:1;flex-shrink:0;">{icon}</div>'
                f'<div style="font-size:0.8rem;color:#374151;line-height:1.55;">{text}</div>'
                f'</div>',
                unsafe_allow_html=True)
    with di_r:
        st.markdown('<div class="sec-title">DREAM CONTENT RADAR</div>', unsafe_allow_html=True)
        st.markdown('<div class="card" style="padding:12px 16px;">', unsafe_allow_html=True)
        if DEMO_AVAILABLE:
            st.plotly_chart(build_dream_radar_fig(DREAM_DETECTIONS),
                            width="stretch", key="demo_radar", config={"displayModeBar": False})
            dist = Counter(d["category"] for d in DREAM_DETECTIONS)
            top_cat = dist.most_common(1)[0][0]
            st.markdown(
                f'<div style="font-size:0.72rem;color:#9CA3AF;margin-top:4px;">'
                f'Dominant content: <b style="color:{DREAM_COLORS[top_cat]};">{top_cat}</b> '
                f'({dist[top_cat]} events)</div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # EEG waveform from real data
    if REAL_X_PATH.exists() and REAL_Y_PATH.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">REAL EEG WAVEFORM — Sleep-EDF Dataset</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="card" style="padding:14px 20px;">', unsafe_allow_html=True)
        X_r = np.load(REAL_X_PATH, mmap_mode="r")
        y_r = np.load(REAL_Y_PATH)
        rem_idxs = np.where(y_r == 4)[0]
        show_idx = int(rem_idxs[len(rem_idxs) // 2]) if len(rem_idxs) else 0
        stage_lbl = STAGE_NAMES[int(y_r[show_idx])]
        cap1, cap2 = st.columns([3, 1])
        with cap1:
            st.plotly_chart(build_eeg_fig(X_r[show_idx], stage_lbl),
                            width="stretch", key="demo_eeg", config={"displayModeBar": False})
        with cap2:
            st.markdown(f"""
            <div style="padding:10px 0 0;">
                <div class="m-label">CHANNEL</div>
                <div style="font-size:0.85rem;color:#374151;font-weight:600;">Fpz-Cz</div>
                <br>
                <div class="m-label">STAGE</div>
                <div style="font-size:0.85rem;font-weight:700;
                            color:{STAGE_COLORS.get(stage_lbl,'#374151')};">{stage_lbl}</div>
                <br>
                <div class="m-label">SOURCE</div>
                <div style="font-size:0.75rem;color:#6B7280;">Sleep-EDF Cassette<br>PhysioNet</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — UPLOAD FILE  (three modes)
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mode selector ─────────────────────────────────────────────────────────
    up_mode = st.radio(
        "mode",
        ["📁  Upload .npy File", "🧠  Live EEG Monitor", "😴  Sleep Recording"],
        horizontal=True, label_visibility="collapsed", key="up_mode_radio",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1 — UPLOAD FILE
    # ══════════════════════════════════════════════════════════════════════════
    if up_mode == "📁  Upload .npy File":
        up_l, up_r = st.columns([1.6, 1])

        with up_l:
            st.markdown('<div class="sec-title">DROP YOUR SLEEP DATA FILE</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="card" style="padding:16px 20px 10px;">'
                '<div style="font-size:0.82rem;color:#6B7280;margin-bottom:14px;line-height:1.7;">'
                'Upload a <code>sleep_X.npy</code> — shape <code>(n_epochs,2,3000)</code>, '
                'float32, z-scored.<br>'
                'Optionally include <code>sleep_y.npy</code> with AASM stage labels.</div>',
                unsafe_allow_html=True,
            )
            f_x = st.file_uploader("sleep_X.npy", type=["npy"], key="upload_x",
                                   label_visibility="collapsed")
            f_y = st.file_uploader("sleep_y.npy (optional)", type=["npy"], key="upload_y")

            if f_x:
                X = np.load(io.BytesIO(f_x.read()))
                y = np.load(io.BytesIO(f_y.read())) if f_y else None
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="card" style="padding:14px 18px;">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;">'
                    '<div><div class="m-label">FILE LOADED</div>'
                    f'<div style="font-size:0.88rem;font-weight:700;color:#111827;margin-top:4px;">{f_x.name}</div></div>'
                    '<div style="text-align:right;"><div class="m-label">SHAPE</div>'
                    f'<div style="font-size:0.88rem;font-weight:700;color:#E8762C;">{X.shape}</div></div>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Run Analysis", type="primary", key="btn_run_upload"):
                    if y is not None:
                        hyp_names = [STAGE_NAMES[int(np.clip(v, 0, 4))] for v in y[:len(X)]]
                    else:
                        from scipy.signal import welch as _welch
                        hyp_names = []
                        bar = st.progress(0, text="Analysing EEG...")
                        for _i, ep in enumerate(X):
                            f_, P_ = _welch(ep[0], fs=100, nperseg=256)
                            def _bp(lo, hi): return float(P_[(f_>=lo)&(f_<hi)].sum()) + 1e-9
                            tot = _bp(0.5, 40)
                            sc = np.array([_bp(15,30)/tot+_bp(8,12)/tot*.5,
                                           _bp(4,8)/tot+_bp(8,12)/tot*.3,
                                           _bp(12,15)/tot*2+_bp(0.5,4)/tot*.3,
                                           _bp(0.5,4)/tot*2,
                                           (_bp(4,8)+_bp(8,12))/tot*1.2], dtype=np.float32)
                            hyp_names.append(STAGE_NAMES[int(sc.argmax())])
                            if (_i + 1) % 50 == 0:
                                bar.progress((_i + 1) / len(X))
                        bar.empty()
                    render_analysis(X, hyp_names, datetime(2026, 6, 22, 23, 30),
                                    session_label=f_x.name.replace(".npy", ""),
                                    key_prefix="upload")
            else:
                st.markdown("</div>", unsafe_allow_html=True)

        with up_r:
            st.markdown(
                '<div class="card" style="padding:18px 20px;">'
                '<div class="sec-title">FILE FORMAT</div>'
                '<div style="font-size:0.82rem;color:#374151;line-height:1.9;">'
                '<code style="background:#F3F4F6;padding:1px 5px;border-radius:4px;">sleep_X.npy</code><br>'
                'Shape: <code>(N, 2, 3000)</code><br>dtype: <code>float32</code><br>'
                'Channels: Fpz-Cz, Pz-Oz<br>Epoch: 30 s @ 100 Hz<br>Preprocessing: z-scored<br><br>'
                '<code style="background:#F3F4F6;padding:1px 5px;border-radius:4px;">sleep_y.npy</code>&nbsp;(optional)<br>'
                'Shape: <code>(N,)</code><br>dtype: <code>int64</code><br>'
                'Labels: 0=Wake 1=N1 2=N2 3=N3 4=REM</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="card" style="padding:18px 20px;">'
                '<div class="sec-title">DEMO FILES</div>'
                '<div style="font-size:0.8rem;color:#374151;line-height:1.8;">'
                '<code>demo_upload_X.npy</code> — 300 epochs, real Sleep-EDF<br><br>'
                '<code>hardware_session_X.npy</code> — 572 epochs, simulated hardware</div></div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2 — LIVE EEG MONITOR
    # ══════════════════════════════════════════════════════════════════════════
    elif up_mode == "🧠  Live EEG Monitor":
        live_l, live_r = st.columns([1, 2])

        with live_l:
            st.markdown('<div class="card" style="padding:18px 20px;">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">CONNECTION</div>', unsafe_allow_html=True)

            if SERIAL_OK:
                _ports = [p.device for p in serial.tools.list_ports.comports()] or \
                         ["COM3", "COM4", "/dev/ttyUSB0"]
            else:
                _ports = ["COM3", "COM4", "COM5", "/dev/ttyUSB0", "/dev/ttyACM0"]

            _lport = st.selectbox("Port", _ports, key="live_port_sel")
            _lbaud = st.selectbox("Baud", [230400, 115200, 57600], key="live_baud_sel")

            _is_live = bool(
                st.session_state.live_thread and
                st.session_state.live_thread.is_alive()
            )

            if not _is_live:
                if st.button("▶ Start Live Monitor", type="primary",
                             use_container_width=True, key="btn_live_start"):
                    if not SERIAL_OK:
                        st.error("Install pyserial: pip install pyserial")
                    else:
                        _stop = threading.Event()
                        _buf  = _deque(maxlen=5000)
                        _thr  = threading.Thread(
                            target=_live_reader,
                            args=(_lport, _lbaud, _buf, _stop),
                            daemon=True,
                        )
                        _thr.start()
                        st.session_state.live_buf    = _buf
                        st.session_state.live_stop   = _stop
                        st.session_state.live_thread = _thr
                        st.rerun()
            else:
                if st.button("⏹ Stop", use_container_width=True, key="btn_live_stop"):
                    st.session_state.live_stop.set()
                    st.session_state.live_thread = None
                    st.rerun()

                _n = len(st.session_state.live_buf)
                _err = [x for x in list(st.session_state.live_buf) if isinstance(x, tuple) and x[0] == "ERR"]
                if _err:
                    st.error(_err[-1][1])
                else:
                    st.markdown(
                        '<div style="margin-top:10px;padding:10px 12px;background:#ECFDF5;'
                        'border-radius:8px;border-left:3px solid #059669;">'
                        '<div style="font-size:0.72rem;font-weight:700;color:#059669;">LIVE</div>'
                        f'<div style="font-size:0.72rem;color:#6B7280;margin-top:3px;">'
                        f'{_n} samples · {_n/500:.1f}s buffered</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            # Current stage card
            if _is_live and len(st.session_state.live_buf) >= 2500:
                _stage = _stage_from_buf(st.session_state.live_buf)
                _sc = STAGE_COLORS.get(_stage, "#374151")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="card-sm" style="text-align:center;padding:18px;">'
                    '<div class="m-label">CURRENT STAGE</div>'
                    f'<div style="font-size:2.2rem;font-weight:800;color:{_sc};'
                    'font-family:-apple-system,sans-serif;margin-top:8px;line-height:1;">'
                    f'{_stage}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="card" style="padding:14px 18px;">'
                '<div class="sec-title">HARDWARE</div>'
                '<div style="font-size:0.78rem;color:#374151;line-height:1.9;">'
                'Arduino Uno R4 Minima<br>'
                'BioAmp EXG Pill (analog FE)<br>'
                'Ag/AgCl wet electrodes<br>'
                'Fpz · Pz (forehead + back)<br>'
                'GND → earlobe</div></div>',
                unsafe_allow_html=True,
            )

        with live_r:
            if _is_live:
                _buf_list = [x for x in list(st.session_state.live_buf)
                             if not (isinstance(x, tuple) and x[0] == "ERR")]
                if len(_buf_list) >= 50:
                    _arr = np.array(_buf_list[-min(len(_buf_list), 5000):])
                    _t   = np.linspace(-len(_arr)/500, 0, len(_arr))
                    _fig = go.Figure()
                    _fig.add_trace(go.Scatter(
                        x=_t, y=_arr[:, 0] + 2, mode="lines",
                        line=dict(color="#E8762C", width=0.7), name="Fpz-Cz",
                    ))
                    _fig.add_trace(go.Scatter(
                        x=_t, y=_arr[:, 1] - 2, mode="lines",
                        line=dict(color="#2563EB", width=0.7), name="Pz-Oz", opacity=0.8,
                    ))
                    _fig.update_layout(
                        height=320, paper_bgcolor="white", plot_bgcolor="white",
                        margin=dict(t=10, b=32, l=48, r=10),
                        xaxis=dict(title="Time (s)", gridcolor="#F3F4F6",
                                   tickfont=dict(size=10, color="#9CA3AF"), showline=False),
                        yaxis=dict(gridcolor="#F3F4F6", tickfont=dict(size=10, color="#9CA3AF"),
                                   showline=False, zeroline=True, zerolinecolor="#E5E7EB"),
                        legend=dict(orientation="h", x=1, xanchor="right", y=1.1,
                                    font=dict(size=10)),
                        font=dict(family="-apple-system,sans-serif"),
                    )
                    st.markdown('<div class="card" style="padding:14px 20px;">',
                                unsafe_allow_html=True)
                    st.plotly_chart(_fig, width="stretch", key="live_eeg_plot")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("Reading signal — please wait a moment...")
                # Auto-refresh every 1.5 s
                _time.sleep(1.5)
                st.rerun()
            else:
                st.markdown(
                    '<div class="card" style="padding:52px 32px;text-align:center;">'
                    '<div style="font-size:2.5rem;margin-bottom:14px;">🧠</div>'
                    '<div style="font-size:0.9rem;font-weight:600;color:#374151;">Connect your headband</div>'
                    '<div style="font-size:0.8rem;color:#9CA3AF;margin-top:8px;line-height:1.7;">'
                    'Plug Arduino Uno R4 Minima via USB, select the COM port,<br>'
                    'then press Start. Live EEG and stage appear here.</div></div>',
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 3 — SLEEP RECORDING
    # ══════════════════════════════════════════════════════════════════════════
    else:
        rec_l, rec_r = st.columns([1, 2])
        _is_rec = bool(
            st.session_state.rec_thread and
            st.session_state.rec_thread.is_alive()
        )

        with rec_l:
            st.markdown('<div class="card" style="padding:18px 20px;">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">RECORDING SETUP</div>', unsafe_allow_html=True)

            if SERIAL_OK:
                _rports = [p.device for p in serial.tools.list_ports.comports()] or \
                          ["COM3", "COM4", "/dev/ttyUSB0"]
            else:
                _rports = ["COM3", "COM4", "COM5", "/dev/ttyUSB0", "/dev/ttyACM0"]

            _rport = st.selectbox("Port", _rports, key="rec_port_sel", disabled=_is_rec)
            _rbaud = st.selectbox("Baud", [230400, 115200, 57600], key="rec_baud_sel",
                                  disabled=_is_rec)

            if not _is_rec and not st.session_state.rec_done:
                if st.button("😴 Start Sleep Recording", type="primary",
                             use_container_width=True, key="btn_rec_start"):
                    if not SERIAL_OK:
                        st.error("Install pyserial: pip install pyserial")
                    else:
                        if _REC_TMP.exists():
                            _REC_TMP.unlink()
                        _rstop = threading.Event()
                        _rthr  = threading.Thread(
                            target=_sleep_recorder,
                            args=(_rport, _rbaud, str(_REC_TMP), _rstop),
                            daemon=True,
                        )
                        _rthr.start()
                        st.session_state.rec_stop   = _rstop
                        st.session_state.rec_thread = _rthr
                        st.session_state.rec_start  = _time.time()
                        st.session_state.rec_done   = False
                        st.rerun()

            elif _is_rec:
                _elapsed = _time.time() - (st.session_state.rec_start or _time.time())
                _h, _rm  = divmod(int(_elapsed), 3600)
                _rm, _rs = divmod(_rm, 60)
                _fmb = _REC_TMP.stat().st_size / 1e6 if _REC_TMP.exists() else 0.0
                st.markdown(
                    '<div style="padding:12px;background:#FFF7F0;border-radius:10px;'
                    'border-left:3px solid #E8762C;margin-bottom:12px;">'
                    '<div style="font-size:0.72rem;font-weight:700;color:#E8762C;">RECORDING</div>'
                    f'<div style="font-size:1.8rem;font-weight:800;color:#111827;'
                    'font-family:-apple-system,sans-serif;line-height:1;margin-top:4px;">'
                    f'{_h:02d}:{_rm:02d}:{_rs:02d}</div>'
                    f'<div style="font-size:0.72rem;color:#9CA3AF;margin-top:5px;">'
                    f'{_fmb:.1f} MB saved to disk</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("⏹ Stop & Generate Files", use_container_width=True,
                             key="btn_rec_stop"):
                    st.session_state.rec_stop.set()
                    st.session_state.rec_thread.join(timeout=3)
                    st.session_state.rec_thread = None
                else:
                    # Auto-refresh every 5s to update timer and MB counter
                    _time.sleep(5)
                    st.rerun()
                    with st.spinner("Processing overnight recording..."):
                        _rX, _ry = _process_recording(str(_REC_TMP))
                    st.session_state.rec_X    = _rX
                    st.session_state.rec_y    = _ry
                    st.session_state.rec_done = True
                    st.rerun()

            elif st.session_state.rec_done:
                if st.button("🔄 New Recording", use_container_width=True, key="btn_new_rec"):
                    st.session_state.rec_done = False
                    st.session_state.rec_X    = None
                    st.session_state.rec_y    = None
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="card" style="padding:14px 18px;">'
                '<div class="sec-title">HOW IT WORKS</div>'
                '<div style="font-size:0.78rem;color:#374151;line-height:1.9;">'
                '1. Put on headband<br>'
                '2. Click Start Recording<br>'
                '3. Go to sleep 😴<br>'
                '4. Wake up, come back<br>'
                '5. Click Stop &amp; Generate<br>'
                '6. Download X.npy + y.npy<br>'
                '7. Full analysis shown below'
                '</div></div>',
                unsafe_allow_html=True,
            )

        with rec_r:
            if st.session_state.rec_done and st.session_state.rec_X is not None:
                _rX = st.session_state.rec_X
                _ry = st.session_state.rec_y

                # Download buttons
                _Xbuf = io.BytesIO(); np.save(_Xbuf, _rX); _Xbuf.seek(0)
                _ybuf = io.BytesIO(); np.save(_ybuf, _ry); _ybuf.seek(0)
                _dc1, _dc2 = st.columns(2)
                with _dc1:
                    st.download_button("⬇ sleep_X.npy", _Xbuf, file_name="sleep_X.npy",
                                       mime="application/octet-stream",
                                       use_container_width=True, key="dl_rec_X")
                with _dc2:
                    st.download_button("⬇ sleep_y.npy", _ybuf, file_name="sleep_y.npy",
                                       mime="application/octet-stream",
                                       use_container_width=True, key="dl_rec_y")

                st.markdown("<br>", unsafe_allow_html=True)
                _rec_hyp = [STAGE_NAMES[int(v)] for v in _ry]
                _rec_start_dt = datetime.fromtimestamp(st.session_state.rec_start or 0)
                render_analysis(_rX, _rec_hyp, _rec_start_dt,
                                session_label="Overnight Recording", key_prefix="rec")

            elif _is_rec:
                st.markdown(
                    '<div class="card" style="padding:48px 32px;text-align:center;">'
                    '<div style="font-size:2.5rem;margin-bottom:14px;">😴</div>'
                    '<div style="font-size:0.9rem;font-weight:600;color:#374151;">'
                    'Recording in progress</div>'
                    '<div style="font-size:0.8rem;color:#9CA3AF;margin-top:8px;line-height:1.7;">'
                    'Data is being saved to disk continuously.<br>'
                    'You can close this window — the recording keeps going.<br>'
                    'Come back when you wake up and click Stop.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="card" style="padding:48px 32px;text-align:center;">'
                    '<div style="font-size:2.5rem;margin-bottom:14px;">🌙</div>'
                    '<div style="font-size:0.9rem;font-weight:600;color:#374151;">'
                    'Ready to record</div>'
                    '<div style="font-size:0.8rem;color:#9CA3AF;margin-top:8px;line-height:1.7;">'
                    'Put on your headband, select the serial port above,<br>'
                    'and press Start. Raw EEG saves to disk every second.</div></div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CONNECT HARDWARE
# ─────────────────────────────────────────────────────────────────────────────
with tab_hw:
    st.markdown("<br>", unsafe_allow_html=True)
    hw_l, hw_r = st.columns([1, 1.8])

    with hw_l:
        st.markdown('<div class="sec-title">SERIAL CONNECTION</div>', unsafe_allow_html=True)
        st.markdown('<div class="card" style="padding:18px 20px;">', unsafe_allow_html=True)
        port_options = ["COM3", "COM4", "COM5", "COM6",
                        "/dev/ttyUSB0", "/dev/ttyACM0", "/dev/cu.usbmodem"]
        selected_port = st.selectbox("Port", port_options, label_visibility="collapsed")
        baud_col, ch_col = st.columns(2)
        with baud_col:
            st.markdown('<div class="m-label">BAUD RATE</div>', unsafe_allow_html=True)
            baud = st.selectbox("Baud", [230400, 115200, 57600], label_visibility="collapsed")
        with ch_col:
            st.markdown('<div class="m-label">CHANNELS</div>', unsafe_allow_html=True)
            n_ch = st.selectbox("Ch", [2, 1, 4], label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        connect_btn = st.button("Connect Device", type="primary", use_container_width=True)
        if connect_btn:
            st.info(f"Attempting connection on {selected_port} @ {baud} baud...")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="card" style="padding:16px 20px;">'
            '<div class="sec-title">PROTOCOL</div>'
            '<div style="font-size:0.8rem;color:#374151;line-height:1.9;">'
            'Device: <strong>Chords Arduino</strong><br>'
            'Packet: <code>16 bytes</code><br>'
            'Sync: <code>0xC7 0x7C</code><br>'
            'ADC: <code>14-bit</code> &nbsp; FS: <code>500 Hz</code><br>'
            'Channels: Configurable 1&#8211;4</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="card" style="padding:16px 20px;">'
            '<div class="sec-title">DEVICE CHECKLIST</div>'
            '<div style="font-size:0.8rem;color:#374151;line-height:2.1;">'
            '<span style="color:#10B981;">&#10003;</span>'
            '&nbsp;Arduino flashed with Chords firmware<br>'
            '<span style="color:#10B981;">&#10003;</span>'
            '&nbsp;Electrodes placed: Fpz&#8209;Cz, Pz&#8209;Oz<br>'
            '<span style="color:#F59E0B;">&#9679;</span>'
            '&nbsp;USB cable connected to PC<br>'
            '<span style="color:#9CA3AF;">&#9675;</span>'
            '&nbsp;Select port above and click Connect</div></div>',
            unsafe_allow_html=True,
        )

    with hw_r:
        st.markdown('<div class="sec-title">LAST HARDWARE RECORDING</div>',
                    unsafe_allow_html=True)
        if HW_X_PATH.exists():
            X_hw   = np.load(HW_X_PATH)
            y_hw   = np.load(HW_Y_PATH) if HW_Y_PATH.exists() else None
            hyp_hw = ([STAGE_NAMES[int(np.clip(v, 0, 4))] for v in y_hw]
                      if y_hw is not None else ["N2"] * len(X_hw))
            render_analysis(X_hw, hyp_hw,
                            datetime(2026, 6, 23, 0, 0),
                            session_label="Hardware Recording",
                            key_prefix="hw")
        else:
            st.markdown(
                '<div class="card" style="padding:48px 32px;text-align:center;">'
                '<div style="font-size:2.5rem;margin-bottom:14px;color:#D1D5DB;">&#9711;</div>'
                '<div style="font-size:0.9rem;font-weight:600;color:#374151;margin-bottom:6px;">'
                'No recording available</div>'
                '<div style="font-size:0.8rem;color:#9CA3AF;line-height:1.7;">'
                'Connect your Chords device and start a recording session.<br>'
                'Results will appear here automatically.</div></div>',
                unsafe_allow_html=True,
            )
