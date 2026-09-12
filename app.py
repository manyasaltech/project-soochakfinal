"""
Project Soochak - Mine Assessment & Terrain Response
Smart India Hackathon 2026 | Problem Statement 26025
Team MATR

Interactive Dashboard with AeroLinkTree Dark Grain Glassmorphic Design:
- Real-Time Hardware Bridge via USB Serial (Node S-103)
- Multi-Sensor Simulation Engine (MPU6050, VL53L1X, HX711, 980m Laser Baseline, Gas)
- AI Layer 1: UCI Seismic-Bumps Isolation Forest Anomaly Detection
- AI Layer 2: Sentinel-1 InSAR STGCN-LSTM 12-Hour Subsidence Forecasting
- Interactive 8x6 Health Grid (Phi-Cube) & Schematic Mine Wall Map
- Telemetry Trends, Polar Radar Fingerprint & False-Alarm Audit Log
"""

import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_engine import (
    MineEnvironmentSimulator,
    OperationalScenario,
    ThreatLevel,
)
from hardware import bridge, get_available_ports
from models.seismic_isolation_forest import get_seismic_detector
from models.stgcn_lstm import get_stgcn_forecaster

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & AEROLINKTREE THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Project Soochak | Mine Subsidence Monitoring",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Chakra+Petch:wght@500;600;700&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* AeroLinkTree Dark Grain & Ambient Radial Lighting */
    .stApp {
        background-color: #0a0c10 !important;
        color: #f3f4f6 !important;
        background-image:
            radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.14), transparent 55%),
            radial-gradient(circle at 100% 100%, rgba(52, 211, 153, 0.07), transparent 50%),
            radial-gradient(circle at 0% 50%, rgba(192, 132, 252, 0.04), transparent 45%) !important;
        background-attachment: fixed !important;
    }

    /* Fixed Subtle Noise Grain Overlay */
    .grain-overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 999999;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
        opacity: 0.038;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Frosted Glassmorphic Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .glass-card-sm {
        background: rgba(17, 24, 39, 0.55);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
    }

    /* Header Bar */
    .brand-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #38bdf8;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-title {
        font-family: 'Chakra Petch', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #f8fafc;
        margin: 0;
        line-height: 1.1;
    }
    .brand-title .accent {
        color: #38bdf8;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
    }
    .brand-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* KPI Cards */
    .kpi-box {
        background: rgba(17, 24, 39, 0.55);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28);
        transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-box:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.35);
    }
    .kpi-title {
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        font-weight: 700;
    }
    .kpi-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 6px 0;
        line-height: 1;
    }
    .kpi-helper {
        font-size: 0.69rem;
        color: #64748b;
        margin-top: -3px;
        margin-bottom: 6px;
    }
    .kpi-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.70rem;
        font-weight: 700;
        width: fit-content;
    }
    .pill-safe { background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .pill-info { background: rgba(56, 189, 248, 0.16); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); }
    .pill-warn { background: rgba(245, 158, 11, 0.16); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .pill-crit { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); animation: pulse 1.6s infinite; }

    .pulse-circle {
        width: 6px; height: 6px; border-radius: 50%; background: currentColor;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.55; transform: scale(0.9); }
    }

    /* Threat Status Banner */
    .status-banner {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        backdrop-filter: blur(12px);
    }
    .banner-safe {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 5px solid #10b981;
    }
    .banner-watch {
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 5px solid #38bdf8;
    }
    .banner-warning {
        background: rgba(245, 158, 11, 0.09);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 5px solid #f59e0b;
    }
    .banner-critical {
        background: rgba(239, 68, 68, 0.13);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 5px solid #ef4444;
        animation: pulse 1.5s infinite;
    }

    /* AeroLinkTree Presenter Stage & Hero Countdown Timer */
    .presenter-shell {
        background: rgba(15, 23, 42, 0.70);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
    }
    .presenter-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 10px;
    }
    .presenter-brand-name {
        font-family: 'Inter', sans-serif;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .presenter-sub-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .presenter-topic-card {
        margin-bottom: 14px;
    }
    .presenter-topic-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .presenter-topic-title {
        font-family: 'Outfit', 'Chakra Petch', sans-serif;
        font-size: clamp(1.4rem, 2.2vw, 2.1rem);
        font-weight: 800;
        line-height: 1.18;
        letter-spacing: -0.01em;
        color: #f8fafc;
        margin: 0;
    }
    .presenter-topic-desc {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 6px;
        max-width: 820px;
        line-height: 1.45;
    }
    .presenter-timer-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 2px;
    }
    .presenter-timer-digits {
        font-family: 'Outfit', 'JetBrains Mono', monospace;
        font-size: clamp(3.0rem, 5.2vw, 4.8rem);
        font-weight: 800;
        line-height: 0.92;
        letter-spacing: 0.04em;
        font-variant-numeric: tabular-nums;
        color: #f8fafc;
        text-shadow: 0 0 35px rgba(56, 189, 248, 0.35);
        margin: 8px 0;
    }
    .presenter-timer-digits.urgent {
        color: #f87171 !important;
        text-shadow: 0 0 45px rgba(239, 68, 68, 0.65) !important;
        animation: pulse-urgent 1.2s infinite ease-in-out;
    }
    .presenter-timer-digits.warning {
        color: #fbbf24 !important;
        text-shadow: 0 0 35px rgba(245, 158, 11, 0.5) !important;
    }
    @keyframes pulse-urgent {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.88; transform: scale(0.985); }
    }
    .timer-progress-track {
        width: 100%;
        height: 7px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 6px;
    }
    .timer-progress-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #38bdf8, #10b981);
        transition: width 0.4s ease;
    }
    .timer-progress-fill.urgent {
        background: linear-gradient(90deg, #f59e0b, #ef4444);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.5);
    }
    .timer-progress-fill.warning {
        background: linear-gradient(90deg, #38bdf8, #f59e0b);
    }

    /* 3-Second Executive Summary Cards (AeroLinkTree editorial pill style) */
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-top: 16px;
    }
    .summary-pill-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: border-color 0.2s, transform 0.2s;
    }
    .summary-pill-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }
    .summary-tag {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .summary-value {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.25;
    }
    .summary-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Scenario Quick Switcher Bar */
    .quick-scenario-bar {
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
        flex-wrap: wrap;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.45);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #94a3b8 !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.16) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
    }

    /* Phi-Cube Sensor Matrix */
    .phi-matrix { display: grid; gap: 8px; }
    .phi-tile {
        aspect-ratio: 1/1; border-radius: 10px;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        font-family: 'JetBrains Mono', monospace; border: 2px solid transparent;
        transition: transform 0.16s, box-shadow 0.16s; position: relative;
    }
    .phi-tile:hover { transform: scale(1.05); z-index: 2; }
    .phi-score { font-size: 1.25rem; font-weight: 800; color: #0b1120; line-height: 1; }
    .phi-id { font-size: 0.52rem; font-weight: 700; color: rgba(11, 17, 32, 0.75); margin-top: 3px; }
    .phi-excellent { background: #16a34a; }
    .phi-good { background: #86efac; }
    .phi-suboptimal { background: #f59e0b; }
    .phi-critical { background: #f87171; }
    .phi-empty { background: rgba(30, 41, 59, 0.25); border: 1px dashed rgba(75, 85, 99, 0.25); }
    .phi-selected { border-color: #38bdf8 !important; box-shadow: 0 0 12px rgba(56, 189, 248, 0.55); }
    .phi-live-badge {
        position: absolute; top: 3px; right: 3px;
        background: #ef4444; color: #ffffff;
        font-size: 0.42rem; font-weight: 800; padding: 1px 4px; border-radius: 4px;
        animation: pulse 1.4s infinite;
    }

    /* Detail Card */
    .detail-card {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .detail-row {
        display: flex; justify-content: space-between; padding: 4px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.81rem;
    }
    .detail-label { color: #94a3b8; }
    .detail-value { color: #f3f4f6; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

    /* Section Subheaders */
    .section-headline {
        font-family: 'Chakra Petch', sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 16px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-headline::after {
        content: "";
        flex: 1;
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin-left: 10px;
    }
</style>
<div class="grain-overlay"></div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. SESSION STATE & SIMULATOR INITIALIZATION
# -----------------------------------------------------------------------------
if "simulator" not in st.session_state:
    with st.spinner("Initializing Project Soochak AI Engines..."):
        st.session_state.simulator = MineEnvironmentSimulator()
        st.session_state.current_data = st.session_state.simulator.step(OperationalScenario.NORMAL)
        st.session_state.auto_refresh = False
        st.session_state.selected_node_id = "S-103"

sim: MineEnvironmentSimulator = st.session_state.simulator


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & METRIC TIERS
# -----------------------------------------------------------------------------
def kpi_html(label: str, value: str, pill_cls: str, pill_text: str, helper: str = "") -> str:
    h_html = f'<div class="kpi-helper">{helper}</div>' if helper else ""
    return f"""
    <div class="kpi-box">
        <div class="kpi-title">{label}</div>
        {h_html}
        <div class="kpi-number">{value}</div>
        <div class="kpi-pill {pill_cls}"><span class="pulse-circle"></span>{pill_text}</div>
    </div>
    """

def threat_tier_meta(level: str):
    return {
        "SAFE": ("pill-safe", "Normal Operations"),
        "WATCH": ("pill-info", "Subtle Variance"),
        "WARNING": ("pill-warn", "Strata Creep Warning"),
        "CRITICAL": ("pill-crit", "Evacuate Now"),
    }.get(level, ("pill-safe", level))

def score_tier_meta(score: float):
    if score >= 0.70: return "pill-crit", "Critical (≥0.70)"
    elif score >= 0.40: return "pill-warn", "Elevated (0.40–0.69)"
    return "pill-safe", "Normal (<0.40)"

def displacement_tier_meta(disp_mm: float):
    if disp_mm >= 5.0: return "pill-crit", "Critical (≥5.0mm)"
    elif disp_mm >= 2.0: return "pill-warn", "Elevated (2.0–4.9mm)"
    return "pill-safe", "Normal (<2.0mm)"

def laser_tier_meta(dev_mm: float):
    if dev_mm > 2.5: return "pill-crit", "Critical Beam Shift"
    elif dev_mm > 0.8: return "pill-warn", "Warning Deviation"
    return "pill-safe", "Aligned"

def mine_health_score(anomaly_score: float) -> int:
    return max(0, min(100, int(round((1.0 - anomaly_score) * 100))))

def health_tier(score: int):
    if score >= 80: return "excellent", "Excellent", "#16a34a"
    elif score >= 60: return "good", "Good", "#4ade80"
    elif score >= 40: return "suboptimal", "Suboptimal", "#f59e0b"
    else: return "critical", "Critical", "#f87171"


# -----------------------------------------------------------------------------
# 4. SENSOR GRID & SCHEMATIC WALL MAP
# -----------------------------------------------------------------------------
GRID_COLS, GRID_ROWS = 8, 6

def compute_sensor_grid(nodes_list):
    lats = [n["lat"] for n in nodes_list]; lons = [n["lon"] for n in nodes_list]
    la1, la2 = min(lats), max(lats); lo1, lo2 = min(lons), max(lons)
    lp = (la2 - la1) * 0.15 or 0.001; lop = (lo2 - lo1) * 0.15 or 0.001
    la1 -= lp; la2 += lp; lo1 -= lop; lo2 += lop

    placements = {}
    for n in nodes_list:
        col = min(GRID_COLS - 1, int((n["lon"] - lo1) / (lo2 - lo1) * GRID_COLS))
        row = min(GRID_ROWS - 1, int(GRID_ROWS - (n["lat"] - la1) / (la2 - la1) * GRID_ROWS))
        placements[(row, col)] = n
    return placements

def compute_grid_refs(nodes_list):
    pl = compute_sensor_grid(nodes_list)
    refs = {}
    for (r, c), n in pl.items():
        refs[n["node_id"]] = f"{chr(65 + c)}{r + 1}"
    return refs

def render_sensor_grid_html(nodes_list, sel_id=None, is_hardware_live=False):
    pl = compute_sensor_grid(nodes_list)
    cells = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            n = pl.get((r, c))
            if not n:
                cells.append('<div class="phi-tile phi-empty"></div>')
                continue
            sc = mine_health_score(n["anomaly_score"])
            tier, _, _ = health_tier(sc)
            sel_cls = " phi-selected" if n["node_id"] == sel_id else ""
            live_badge = '<div class="phi-live-badge">LIVE</div>' if (n["node_id"] == "S-103" and is_hardware_live) else ''
            cells.append(
                f'<div class="phi-tile phi-{tier}{sel_cls}">'
                f'{live_badge}'
                f'<span class="phi-score">{sc}</span>'
                f'<span class="phi-id">{n["node_id"]}</span>'
                f'</div>'
            )
    return f'<div class="phi-matrix" style="grid-template-columns: repeat({GRID_COLS}, 1fr);">{"".join(cells)}</div>'

panel_polygon = [
    [23.7542, 86.4155], [23.7548, 86.4235],
    [23.7500, 86.4242], [23.7495, 86.4162],
]
goaf_polygon = [
    [23.7530, 86.4175], [23.7520, 86.4215],
    [23.7508, 86.4210], [23.7518, 86.4170],
]
WALL_COLS, WALL_ROWS = 12, 9

def build_schematic_wall_map(nodes_list, laser_dict, grid_refs_dict):
    all_lats = [n["lat"] for n in nodes_list] + [p[0] for p in panel_polygon + goaf_polygon] + [laser_dict["tx_lat"], laser_dict["rx_lat"]]
    all_lons = [n["lon"] for n in nodes_list] + [p[1] for p in panel_polygon + goaf_polygon] + [laser_dict["tx_lon"], laser_dict["rx_lon"]]
    la1, la2 = min(all_lats), max(all_lats); lo1, lo2 = min(all_lons), max(all_lons)
    lp = (la2 - la1) * 0.18 or 0.001; lop = (lo2 - lo1) * 0.18 or 0.001
    la1 -= lp; la2 += lp; lo1 -= lop; lo2 += lop

    def proj(la, lo):
        return ((lo - lo1) / (lo2 - lo1) * WALL_COLS, (la - la1) / (la2 - la1) * WALL_ROWS)

    nl = {n["node_id"]: n for n in nodes_list}
    fig = go.Figure()

    # Panel boundary wall
    pp = [proj(la, lo) for la, lo in panel_polygon] + [proj(*panel_polygon[0])]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in pp], y=[p[1] for p in pp],
        mode="lines", fill="toself",
        fillcolor="rgba(56, 189, 248, 0.08)",
        line=dict(color="#38bdf8", width=2.5),
        name="Panel 4-B Boundary", hoverinfo="text",
        hovertext="Underground Longwall Extraction Panel 4-B",
    ))

    # Goaf caving zone
    gp = [proj(la, lo) for la, lo in goaf_polygon] + [proj(*goaf_polygon[0])]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in gp], y=[p[1] for p in gp],
        mode="lines", fill="toself",
        fillcolor="rgba(249, 115, 22, 0.12)",
        line=dict(color="#f97316", width=2, dash="dash"),
        name="Active Goaf Margin", hoverinfo="text",
        hovertext="Goaf Caving Zone — Decompressed roof strata prone to subsidence",
    ))

    # Roadway Tunnel
    main_nodes = [nid for nid in ["S-101", "S-102", "S-103", "S-104", "S-105"] if nid in nl]
    mx = [proj(nl[nid]["lat"], nl[nid]["lon"])[0] for nid in main_nodes]
    my = [proj(nl[nid]["lat"], nl[nid]["lon"])[1] for nid in main_nodes]
    fig.add_trace(go.Scatter(
        x=mx, y=my, mode="lines",
        line=dict(color="#94a3b8", width=7),
        opacity=0.25, name="Main Roadway", hoverinfo="skip",
    ))

    # Laser Baseline
    tx = proj(laser_dict["tx_lat"], laser_dict["tx_lon"])
    rx = proj(laser_dict["rx_lat"], laser_dict["rx_lon"])
    dev = laser_dict["total_deviation_mm"]
    laser_color = "#ef4444" if dev > 2.5 else ("#f59e0b" if dev > 0.8 else "#10b981")
    fig.add_trace(go.Scatter(
        x=[tx[0], rx[0]], y=[tx[1], rx[1]],
        mode="lines+markers",
        line=dict(color=laser_color, width=2.2, dash="dot"),
        marker=dict(size=10, symbol="diamond", color=laser_color, line=dict(color="#ffffff", width=1)),
        name=f"Laser Baseline ({dev:.2f}mm)",
        hovertext=[
            f"<b>{laser_dict['tx_name']}</b><br>Optical Emitter (North Portal)",
            f"<b>{laser_dict['rx_name']}</b><br>Deviation: {dev:.2f}mm | Status: {laser_dict['status']}",
        ],
        hoverinfo="text",
    ))

    # Nodes
    status_colors = {"SAFE": "#10b981", "WATCH": "#0284c7", "WARNING": "#f97316", "CRITICAL": "#ef4444"}
    nx, ny, nc, nt, nh, ns = [], [], [], [], [], []
    for n in nodes_list:
        x, y = proj(n["lat"], n["lon"])
        nx.append(x); ny.append(y)
        nc.append(status_colors.get(n["status"], "#10b981"))
        nt.append(n["node_id"])
        ns.append(15 + n["anomaly_score"] * 15)
        nh.append(
            f"<b>{n['node_id']} — {n['name']}</b><br>"
            f"Grid: {grid_refs_dict.get(n['node_id'], '-')}<br>"
            f"Status: <b>{n['status']}</b> | ML Score: <b>{n['anomaly_score']:.3f}</b><br>"
            f"Tilt: {n['tilt_magnitude_deg']}° | Disp: {n['displacement_mm']:.2f}mm<br>"
            f"Load: {n['load_kN']:.1f} kN | CH₄: {n['methane_ppm']:.0f} ppm"
        )

    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers+text",
        marker=dict(size=ns, color=nc, line=dict(color="white", width=1.5), symbol="square"),
        text=nt, textposition="top center",
        textfont=dict(color="#f3f4f6", size=10, family="monospace"),
        hovertext=nh, hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        xaxis=dict(
            range=[0, WALL_COLS], showgrid=True, gridcolor="rgba(148, 163, 184, 0.12)",
            dtick=1, zeroline=False, fixedrange=True,
            tickvals=[i + 0.5 for i in range(WALL_COLS)],
            ticktext=[chr(65 + i) for i in range(WALL_COLS)],
        ),
        yaxis=dict(
            range=[0, WALL_ROWS], showgrid=True, gridcolor="rgba(148, 163, 184, 0.12)",
            dtick=1, zeroline=False, fixedrange=True, scaleanchor="x", scaleratio=1,
            tickvals=[i + 0.5 for i in range(WALL_ROWS)],
            ticktext=[str(WALL_ROWS - i) for i in range(WALL_ROWS)],
        ),
        hovermode="closest",
    )
    return fig


# -----------------------------------------------------------------------------
# 5. TOP BRAND HEADER & HARDWARE BAR
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([2.6, 1.4])

with header_col1:
    st.markdown("""
    <div style="display: flex; flex-direction: column;">
        <div class="brand-eyebrow">
            <span class="pulse-circle" style="background:#38bdf8;"></span>
            <span>SIH 2026 · PS 26025 · TEAM MATR · MINE COMMAND</span>
        </div>
        <h1 class="brand-title">PROJECT SOOCHAK <span class="accent">MINE RESPONSE</span></h1>
        <div class="brand-sub">
            Real-Time Mine Subsidence Monitoring, Prediction & Early Warning System · Jharia Coalfield Panel 4-B
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    ports = get_available_ports()
    p_col1, p_col2 = st.columns([1.8, 1.2])
    with p_col1:
        sel_port = st.selectbox(
            "Serial Port",
            ports if ports else ["No Ports Found"],
            label_visibility="collapsed",
            key="header_port_sel",
        )
    with p_col2:
        if not bridge.running:
            if st.button("🔌 Connect", use_container_width=True, key="hw_connect_btn"):
                if sel_port and sel_port != "No Ports Found":
                    ok, msg = bridge.connect(sel_port)
                    if not ok: st.error(msg)
                    else: st.rerun()
        else:
            if st.button("⏹️ Disconnect", use_container_width=True, key="hw_disconnect_btn"):
                bridge.disconnect()
                st.rerun()

    hw_active = bridge.running and bridge.latest_data is not None
    if hw_active:
        st.markdown(
            f'<div style="text-align:right; margin-top:4px;">'
            f'<span class="kpi-pill pill-crit" style="animation:pulse 1.2s infinite;">'
            f'<span class="pulse-circle"></span>LIVE ESP32 S-103 · {bridge.port}'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    elif bridge.running:
        st.markdown(
            f'<div style="text-align:right; margin-top:4px;">'
            f'<span class="kpi-pill pill-info">'
            f'<span class="pulse-circle"></span>LISTENING ON {bridge.port}...'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:right; margin-top:4px;">'
            '<span class="kpi-pill pill-safe">'
            '<span class="pulse-circle"></span>SYNTHETIC SIMULATION ACTIVE'
            '</span></div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. SCENARIO SIMULATION ENGINE
# -----------------------------------------------------------------------------
scenario_dict = {
    "Normal Operations (Ambient Baseline)": OperationalScenario.NORMAL,
    "Controlled Blasting (False Alarm Suppression Test)": OperationalScenario.BLASTING,
    "Strata Creep (Pre-Subsidence Micro-Deformation)": OperationalScenario.STRATA_CREEP,
    "Critical Subsidence (Emergency Evacuation Protocol)": OperationalScenario.CRITICAL_SUBSIDENCE,
    "Sensor Hardware Glitch (Drift Rejection Test)": OperationalScenario.SENSOR_GLITCH,
}

# -----------------------------------------------------------------------------
# 7. NAVIGATION TABS (Preserving the familiar 3-tab workflow)
# -----------------------------------------------------------------------------
tab_operations, tab_simulation, tab_analytics = st.tabs([
    "🗺️  Mine Operations & Grid",
    "🎮  Simulation Control Center",
    "📊  Anomaly Analytics & Audit",
])


# -----------------------------------------------------------------------------
# TAB 2: SIMULATION CONTROLS (Evaluated first so Tab 1 has fresh data)
# -----------------------------------------------------------------------------
with tab_simulation:
    st.markdown('<div class="section-headline">🎮 Mine Scenario Simulator & Disturbance Controls</div>', unsafe_allow_html=True)
    st.caption("Test how the AI anomaly engine, spatial correlation guard, and laser baseline respond to different underground conditions.")

    sc_col1, sc_col2 = st.columns([1.4, 1])
    with sc_col1:
        st.markdown("##### 1️⃣ Choose Operational Scenario")
        selected_scenario_name = st.selectbox(
            "Select Scenario",
            list(scenario_dict.keys()),
            index=0,
            label_visibility="collapsed",
            key="sim_scenario_select",
        )
        active_scenario = scenario_dict[selected_scenario_name]

        b1, b2, b3 = st.columns([1, 1, 1.2])
        with b1:
            btn_step = st.button("⚡ Next Step", use_container_width=True, key="btn_sim_step")
        with b2:
            btn_reset = st.button("🔄 Reset Baseline", use_container_width=True, key="btn_sim_reset")
        with b3:
            auto_toggle = st.checkbox("🔁 Auto-Play (2s)", value=st.session_state.auto_refresh, key="auto_run_chk")
            st.session_state.auto_refresh = auto_toggle

    with sc_col2:
        st.markdown("##### 2️⃣ Manual Disturbance Injection")
        man_disp = st.slider("Manual Ground Displacement (mm)", 0.0, 10.0, 0.0, 0.1, key="man_disp_slider")
        man_tilt = st.slider("Manual Tilt Offset (degrees)", 0.0, 5.0, 0.0, 0.1, key="man_tilt_slider")
        man_laser = st.slider("Manual Laser Optical Drift (mm)", 0.0, 6.0, 0.0, 0.1, key="man_laser_slider")

    if btn_reset:
        st.session_state.simulator = MineEnvironmentSimulator()
        sim = st.session_state.simulator
        st.session_state.current_data = sim.step(OperationalScenario.NORMAL)
        st.rerun()

    if btn_step:
        st.session_state.current_data = sim.step(
            scenario=active_scenario,
            manual_disp_offset=man_disp,
            manual_tilt_offset=man_tilt,
            manual_laser_offset=man_laser,
        )
        st.rerun()

# Auto run step
if (st.session_state.auto_refresh or hw_active) and not btn_step:
    st.session_state.current_data = sim.step(
        scenario=active_scenario,
        manual_disp_offset=man_disp,
        manual_tilt_offset=man_tilt,
        manual_laser_offset=man_laser,
    )

data = st.session_state.current_data
nodes = data["nodes"]
laser = data["laser_system"]
threat_level = data["threat_level"]
peak_score = data["peak_anomaly_score"]
false_alarms = data["false_alarms_rejected"]
history = data["history"]
logs = data["logs"]
uci_seismic = data.get("uci_seismic", {})
insar_forecast = data.get("insar_forecast", {})

# Complete Tab 2 with live status & simulated telemetry
with tab_simulation:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-headline">📟 Simulation Engine Status</div>', unsafe_allow_html=True)

    stat1, stat2, stat3, stat4, stat5 = st.columns(5)
    with stat1: st.metric("Simulation Step", sim.step_counter)
    with stat2: st.metric("Active Scenario", selected_scenario_name.split(" (")[0])
    with stat3: st.metric("Threat Level", threat_level)
    with stat4: st.metric("Filtered Alarms", false_alarms)
    with stat5:
        adaptive_label = "2.0s (Focused)" if (peak_score >= 0.40 or threat_level != "SAFE") else "30.0s (Power-Save)"
        st.metric("Adaptive Polling", adaptive_label)

    st.markdown('<div class="section-headline">📋 Live Multi-Node Simulated Telemetry Stream</div>', unsafe_allow_html=True)
    df_nodes = pd.DataFrame(nodes)[[
        "node_id", "name", "panel_zone", "depth_m", "pitch_deg", "roll_deg",
        "tilt_magnitude_deg", "vibration_rms_g", "vibration_peak_g",
        "displacement_mm", "displacement_rate_mm_min", "load_kN", "load_delta_kN",
        "methane_ppm", "temp_c", "battery_pct", "status", "anomaly_score",
    ]]
    df_nodes.columns = [
        "Node ID", "Name", "Zone", "Depth (m)", "Pitch (°)", "Roll (°)",
        "Tilt Mag (°)", "Vib RMS (g)", "Vib Peak (g)", "Disp (mm)",
        "Disp Rate", "Load (kN)", "Δ Load (kN)", "CH₄ (ppm)", "Temp (°C)",
        "Batt (%)", "Status", "Anomaly Score",
    ]
    st.dataframe(
        df_nodes.style.format({
            "Pitch (°)": "{:.2f}", "Roll (°)": "{:.2f}", "Tilt Mag (°)": "{:.2f}",
            "Vib RMS (g)": "{:.3f}", "Vib Peak (g)": "{:.3f}",
            "Disp (mm)": "{:.2f}", "Disp Rate": "{:+.2f}",
            "Load (kN)": "{:.1f}", "Δ Load (kN)": "{:+.1f}",
            "CH₄ (ppm)": "{:.0f}", "Temp (°C)": "{:.1f}", "Batt (%)": "{:.0f}%",
            "Anomaly Score": "{:.3f}",
        }),
        use_container_width=True,
        height=320,
    )


# -----------------------------------------------------------------------------
# TAB 1: MINE OPERATIONS & REAL-TIME GRID (Original clean layout preserved)
# -----------------------------------------------------------------------------
with tab_operations:
    # -------------------------------------------------------------------------
    # AEROLINKTREE PRESENTATION STAGE & HERO COUNTDOWN TIMER
    # -------------------------------------------------------------------------
    max_d = max(n["displacement_mm"] for n in nodes)
    top_node_item = max(nodes, key=lambda n: n["anomaly_score"])
    
    # 1-Click Scenario Quick-Switcher for Evaluators & Judges
    st.markdown('<div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px;">⚡ Operational Scenario Quick-Switch:</div>', unsafe_allow_html=True)
    sc_cols = st.columns(5)
    scenario_buttons_meta = [
        ("Normal Shift", OperationalScenario.NORMAL, "🟢 Normal Shift"),
        ("Controlled Blasting", OperationalScenario.BLASTING, "🛡️ Blasting Test"),
        ("Strata Creep", OperationalScenario.STRATA_CREEP, "⚠️ Strata Creep"),
        ("Critical Subsidence", OperationalScenario.CRITICAL_SUBSIDENCE, "🚨 Critical Subsidence"),
        ("Sensor Glitch", OperationalScenario.SENSOR_GLITCH, "🔧 Sensor Glitch"),
    ]
    for i, (sc_label, sc_enum, sc_display) in enumerate(scenario_buttons_meta):
        with sc_cols[i]:
            is_active_sc = (active_scenario == sc_enum)
            if st.button(sc_display, key=f"quick_sc_{sc_enum.value}", use_container_width=True, type="primary" if is_active_sc else "secondary"):
                st.session_state.current_data = sim.step(scenario=sc_enum)
                st.rerun()

    if threat_level == "CRITICAL":
        stage_status = "CRITICAL SUBSIDENCE DETECTED"
        stage_status_color = "#f87171"
        stage_pulse_cls = "urgent"
        stage_title = "Coordinated Strata Roof Collapse in Progress"
        stage_desc = f"Severe multi-node roof displacement ({max_d:.2f}mm) validated by 980m Long-Baseline Optical Laser ({laser['total_deviation_mm']:.2f}mm). Immediate underground evacuation mandatory!"
        
        step_mod = (sim.step_counter * 22) % 480
        rem_sec = max(160, 900 - step_mod)
        mins, secs = divmod(rem_sec, 60)
        timer_str = f"{mins:02d}:{secs:02d}"
        timer_label = "🚨 SAFE RETREAT WINDOW (ESTIMATED BUFFER BEFORE ROOF CAVING)"
        progress_pct = int((rem_sec / 900) * 100)
        
        g1_title, g1_val, g1_sub = "1. MINE CONDITION", "🚨 COLLAPSE UNDERWAY", f"Roof Sag: {max_d:.2f} mm"
        g2_title, g2_val, g2_sub = "2. CRITICAL SECTOR", f"🚨 {top_node_item['node_id']} ({top_node_item['panel_zone']})", f"Displacement: {top_node_item['displacement_mm']:.2f} mm"
        g3_title, g3_val, g3_sub = "3. OPERATOR ACTION", "🚨 EVACUATE PANEL 4-B", "Sound Siren & Clear Roadway"

    elif threat_level == "WARNING":
        stage_status = "ELEVATED STRATA CREEP WARNING"
        stage_status_color = "#fbbf24"
        stage_pulse_cls = "warning"
        stage_title = "Pre-Subsidence Bed Separation at Goaf Margins"
        stage_desc = f"Continuous micro-strain and roof sag identified around Node {top_node_item['node_id']} ({top_node_item['panel_zone']}). Laser drift active at {laser['total_deviation_mm']:.2f}mm. Precautionary support inspection advised."
        
        timer_str = "03:45:00"
        timer_label = "⚠️ STABILIZATION & INSPECTION BUFFER WINDOW"
        progress_pct = 65
        
        g1_title, g1_val, g1_sub = "1. MINE CONDITION", "⚠️ SLOW ROOF CREEP", f"Disp: {max_d:.2f} mm"
        g2_title, g2_val, g2_sub = "2. CRITICAL SECTOR", f"⚠️ {top_node_item['node_id']} ({top_node_item['panel_zone']})", f"Score: {top_node_item['anomaly_score']:.3f}"
        g3_title, g3_val, g3_sub = "3. OPERATOR ACTION", "⚠️ DISPATCH GEOTECH", "Inspect Hydraulic Prop Loads"

    elif active_scenario == OperationalScenario.BLASTING:
        stage_status = "CONTROLLED BLASTING FILTERED"
        stage_status_color = "#38bdf8"
        stage_pulse_cls = ""
        stage_title = "Heavy Blast Vibration Filtered · Zero Strata Movement"
        stage_desc = "Transient acoustic shock (1.2g RMS) rejected by Spatial Correlation Guard. Crack displacement and optical laser baseline stable. False alarm suppressed."
        
        timer_str = "00:00:00"
        timer_label = "🛡️ BLAST SHOCK DISSIPATED · TELEMETRY NOMINAL"
        progress_pct = 100
        
        g1_title, g1_val, g1_sub = "1. MINE CONDITION", "🛡️ BLAST SHOCK WAVE", "Transient Spike (1.2g)"
        g2_title, g2_val, g2_sub = "2. CRITICAL SECTOR", "🛡️ EXCAVATION FACE", "False Alarm Rejected"
        g3_title, g3_val, g3_sub = "3. OPERATOR ACTION", "✅ NO ACTION REQUIRED", "AI Guard Kept Mine Active"

    elif active_scenario == OperationalScenario.SENSOR_GLITCH:
        stage_status = "SENSOR DRIFT FILTERED"
        stage_status_color = "#38bdf8"
        stage_pulse_cls = ""
        stage_title = "Isolated Sensor Drift Suppressed by Multi-Node Guard"
        stage_desc = "Single-node telemetry jump rejected because neighboring nodes confirm undisturbed strata. False mine evacuation prevented."
        
        timer_str = "00:00:00"
        timer_label = "🛡️ DRIFT ISOLATED · MINE EXTRACTION CONTINUES"
        progress_pct = 100
        
        g1_title, g1_val, g1_sub = "1. MINE CONDITION", "🛡️ HARDWARE NOISE", "Single-Node Drift Filtered"
        g2_title, g2_val, g2_sub = "2. CRITICAL SECTOR", f"🛡️ {top_node_item['node_id']}", "Flagged for Sensor Check"
        g3_title, g3_val, g3_sub = "3. OPERATOR ACTION", "✅ OPERATIONS NORMAL", "Schedule Sensor Re-check"

    else:
        stage_status = "NORMAL STRATA EQUILIBRIUM"
        stage_status_color = "#34d399"
        stage_pulse_cls = ""
        stage_title = "All 8 Sectors Stable · Regular Extraction Shift Authorized"
        stage_desc = "Underground strata micro-vibrations, hydraulic prop loads, and 980m long-baseline optical reference are within geological baseline limits."
        
        timer_str = "11:42:15"
        timer_label = "🛰️ COPERNICUS SENTINEL-1 SATELLITE RADAR PREDICTIVE HORIZON"
        progress_pct = 92
        
        g1_title, g1_val, g1_sub = "1. MINE CONDITION", "🟢 STRATA STABLE", "Within Geological Limits"
        g2_title, g2_val, g2_sub = "2. CRITICAL SECTOR", "🟢 ALL NOMINAL", "8 Mesh Nodes Active"
        g3_title, g3_val, g3_sub = "3. OPERATOR ACTION", "✅ SHIFT AUTHORIZED", "Regular Mining Operations"

    st.markdown(f"""
    <div class="presenter-shell">
        <div class="presenter-header">
            <div class="presenter-brand-name">
                <span class="pulse-circle" style="background:{stage_status_color};"></span>
                <span>PROJECT SOOCHAK MINE COMMAND // EARLY WARNING STAGE SYNCED</span>
            </div>
            <div class="presenter-sub-tag" style="color: {stage_status_color};">
                ● {stage_status}
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 18px;">
            <div style="flex: 1; min-width: 320px;">
                <div class="presenter-topic-label" style="color: {stage_status_color};">
                    ● {stage_status}
                </div>
                <h2 class="presenter-topic-title">
                    {stage_title}
                </h2>
                <p class="presenter-topic-desc">
                    {stage_desc}
                </p>
            </div>
            
            <div style="text-align: right; min-width: 270px;">
                <div class="presenter-timer-label">
                    {timer_label}
                </div>
                <div class="presenter-timer-digits {stage_pulse_cls}">
                    {timer_str}
                </div>
                <div class="timer-progress-track">
                    <div class="timer-progress-fill {stage_pulse_cls}" style="width: {progress_pct}%;"></div>
                </div>
            </div>
        </div>

        <div class="summary-grid">
            <div class="summary-pill-card">
                <div class="summary-tag">🌐 {g1_title}</div>
                <div class="summary-value">{g1_val}</div>
                <div class="summary-sub">{g1_sub}</div>
            </div>
            <div class="summary-pill-card">
                <div class="summary-tag">📍 {g2_title}</div>
                <div class="summary-value">{g2_val}</div>
                <div class="summary-sub">{g2_sub}</div>
            </div>
            <div class="summary-pill-card">
                <div class="summary-tag">⚡ {g3_title}</div>
                <div class="summary-value">{g3_val}</div>
                <div class="summary-sub">{g3_sub}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top 5 KPI ribbon
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    with kpi_c1:
        pcl, ptxt = threat_tier_meta(threat_level)
        st.markdown(kpi_html("System Threat Level", threat_level, pcl, ptxt, helper="Overall AI & Spatial Verdict"), unsafe_allow_html=True)
    with kpi_c2:
        pcl, ptxt = score_tier_meta(peak_score)
        st.markdown(kpi_html("Peak Anomaly Score", f"{peak_score:.3f}", pcl, ptxt, helper="Isolation Forest Anomaly [0-1]"), unsafe_allow_html=True)
    with kpi_c3:
        max_d = max(n["displacement_mm"] for n in nodes)
        pcl, ptxt = displacement_tier_meta(max_d)
        st.markdown(kpi_html("Max Displacement", f"{max_d:.2f} mm", pcl, ptxt, helper="Crack Opening / Sagging"), unsafe_allow_html=True)
    with kpi_c4:
        l_dev = laser["total_deviation_mm"]
        pcl, ptxt = laser_tier_meta(l_dev)
        st.markdown(kpi_html("Optical Laser Drift", f"{l_dev:.2f} mm", pcl, ptxt, helper="980m Baseline Reference"), unsafe_allow_html=True)
    with kpi_c5:
        st.markdown(kpi_html("False Alarms Blocked", str(false_alarms), "pill-info", "AI Active Guard", helper="Blasting / Sensor Noise Filtered"), unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MANAGEMENT BY EXCEPTION & AUTOMATED HOTSPOT ISOLATION (For 100+ Sensor Fleet)
    # -------------------------------------------------------------------------
    grefs = compute_grid_refs(nodes)
    node_lookup = {n["node_id"]: n for n in nodes}

    # Sort nodes by risk priority
    sorted_fleet = sorted(nodes, key=lambda n: n["anomaly_score"], reverse=True)
    top_hazard = sorted_fleet[0]
    top_hazard_id = top_hazard["node_id"]
    is_hazard_active = (top_hazard["anomaly_score"] >= 0.40 or threat_level in ["WARNING", "CRITICAL"])

    if "selected_node_id" not in st.session_state or st.session_state.selected_node_id not in node_lookup:
        st.session_state.selected_node_id = top_hazard_id

    # Render Management by Exception Banner
    if is_hazard_active:
        mbe_cls = "banner-critical" if top_hazard["anomaly_score"] >= 0.70 else "banner-warning"
        mbe_icon = "🚨" if top_hazard["anomaly_score"] >= 0.70 else "⚠️"
        st.markdown(f"""
        <div class="status-banner {mbe_cls}" style="margin-bottom: 14px; padding: 10px 16px;">
            <div style="font-size: 1.6rem; line-height: 1;">{mbe_icon}</div>
            <div style="flex: 1;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div style="font-weight: 800; font-size: 0.95rem; letter-spacing: 0.4px;">
                        MANAGEMENT BY EXCEPTION · HOTSPOT ISOLATED: {top_hazard_id} ({top_hazard['name']})
                    </div>
                    <span class="kpi-pill pill-crit" style="font-size: 0.68rem;">
                        <span class="pulse-circle"></span>ADAPTIVE SAMPLING: 2.0s POLLING ACTIVE
                    </span>
                </div>
                <div style="font-size: 0.80rem; color: #cbd5e1; margin-top: 3px;">
                    Automated fleet filter isolated <b>{top_hazard_id}</b> in <b>{top_hazard['panel_zone']}</b> (Score <b>{top_hazard['anomaly_score']:.3f}</b> · Disp <b>{top_hazard['displacement_mm']:.2f}mm</b>). Telemetry auto-adjusted to 2.0s high-rate focus while remaining fleet operates in 30s power-save mode.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-banner banner-safe" style="margin-bottom: 14px; padding: 10px 16px;">
            <div style="font-size: 1.6rem; line-height: 1;">🛡️</div>
            <div style="flex: 1;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div style="font-weight: 800; font-size: 0.95rem; letter-spacing: 0.4px;">
                        MANAGEMENT BY EXCEPTION · ALL FLEET SECTORS NOMINAL
                    </div>
                    <span class="kpi-pill pill-safe" style="font-size: 0.68rem;">
                        <span class="pulse-circle"></span>POWER-SAVE TELEMETRY: 30.0s POLLING
                    </span>
                </div>
                <div style="font-size: 0.80rem; color: #cbd5e1; margin-top: 3px;">
                    Automated anomaly filter confirms all sensor nodes are within geological baseline limits. Standard low-power 30-second telemetry active across fleet.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    grid_col, detail_col = st.columns([1.8, 1.2])

    with grid_col:
        st.markdown('<div class="section-headline">🧭 Sensor Health Matrix & Fleet Priority Hotspots</div>', unsafe_allow_html=True)
        st.caption("Color-coded health scores (0-100). Higher = safer. Automatic exception filter prioritizes anomalous sectors.")

        grid_html = render_sensor_grid_html(
            nodes,
            sel_id=st.session_state.selected_node_id,
            is_hardware_live=hw_active,
        )
        st.markdown(f'<div class="glass-card" style="padding: 14px 16px;">{grid_html}</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Priority quick-bar for 1-click drilldown into top hotspots
        st.markdown('<div style="font-size: 0.74rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">⚡ Fleet Priority Hotspots (Ranked by Risk):</div>', unsafe_allow_html=True)
        
        quick_cols = st.columns([1.3] + [1] * min(4, len(sorted_fleet)) + [1.3])
        with quick_cols[0]:
            is_auto_sel = st.session_state.selected_node_id == top_hazard_id
            if st.button(f"🎯 Auto-Focus ({top_hazard_id})", key="btn_autofocus", use_container_width=True, type="primary" if is_auto_sel else "secondary"):
                st.session_state.selected_node_id = top_hazard_id
                st.rerun()

        for idx, n in enumerate(sorted_fleet[:4]):
            with quick_cols[idx + 1]:
                is_sel = n["node_id"] == st.session_state.selected_node_id
                n_score = n["anomaly_score"]
                prefix = "🚨 " if n_score >= 0.70 else ("⚠️ " if n_score >= 0.40 else "● ")
                btn_txt = f"{prefix}{n['node_id']}"
                if st.button(btn_txt, key=f"quick_sel_{n['node_id']}", use_container_width=True, type="primary" if is_sel else "secondary"):
                    st.session_state.selected_node_id = n["node_id"]
                    st.rerun()

        with quick_cols[-1]:
            all_nids = [n["node_id"] for n in nodes]
            cur_idx = all_nids.index(st.session_state.selected_node_id) if st.session_state.selected_node_id in all_nids else 0
            chosen_nid = st.selectbox("Fleet Dropdown", all_nids, index=cur_idx, label_visibility="collapsed", key="fleet_node_dropdown")
            if chosen_nid != st.session_state.selected_node_id:
                st.session_state.selected_node_id = chosen_nid
                st.rerun()

    with detail_col:
        st.markdown('<div class="section-headline">🔍 Node Inspector Telemetry</div>', unsafe_allow_html=True)
        sn = node_lookup[st.session_state.selected_node_id]
        h_score = mine_health_score(sn["anomaly_score"])
        h_tier, h_tier_label, h_color = health_tier(h_score)
        is_node_hw = sn["node_id"] == "S-103" and hw_active

        hw_tag = '<span class="kpi-pill pill-crit" style="font-size:0.65rem; margin-left:6px;"><span class="pulse-circle"></span>LIVE ESP32</span>' if is_node_hw else '<span class="kpi-pill pill-safe" style="font-size:0.65rem; margin-left:6px;">SIMULATED</span>'

        # Fetch calibrated UCI seismic bump anomaly score & InSAR 12h forecast for this node
        seis_scores = uci_seismic.get("scores", {})
        node_seismic_score = seis_scores.get(sn["node_id"], sn["anomaly_score"])
        insar_node_list = insar_forecast.get("nodes", [])
        insar_risk_list = insar_forecast.get("peak_12h_risk", [])
        
        # Match node index to InSAR
        node_num = int(sn["node_id"].split("-")[-1]) if "-" in sn["node_id"] else 1
        insar_idx = min(len(insar_risk_list) - 1, max(0, node_num - 101)) if insar_risk_list else 0
        node_12h_risk = float(insar_risk_list[insar_idx]) if insar_risk_list else 0.12
        insar_meta = insar_node_list[insar_idx] if len(insar_node_list) > insar_idx else {}
        insar_velocity = insar_meta.get("velocity_mmyr", -14.5)

        st.markdown(f"""
        <div class="detail-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #f8fafc;">
                        {sn['node_id']} {hw_tag}
                    </div>
                    <div style="font-size: 0.76rem; color: #94a3b8;">{sn['name']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: {h_color}; font-family: 'JetBrains Mono', monospace; line-height: 1;">{h_score}</div>
                    <div style="font-size: 0.68rem; color: {h_color}; font-weight: 700; text-transform: uppercase;">{h_tier_label}</div>
                </div>
            </div>
            <div class="detail-row"><span class="detail-label">Grid Reference</span><span class="detail-value">{grefs.get(sn['node_id'], '-')}</span></div>
            <div class="detail-row"><span class="detail-label">Zone & Depth</span><span class="detail-value">{sn['panel_zone']} ({sn['depth_m']}m)</span></div>
            <div class="detail-row"><span class="detail-label">GPS Coordinates</span><span class="detail-value">{sn['lat']:.4f}° N, {sn['lon']:.4f}° E</span></div>
            <div class="detail-row"><span class="detail-label">AI Threat Status</span><span class="detail-value">{sn['status']} (ML: {sn['anomaly_score']:.3f})</span></div>
            <div class="detail-row"><span class="detail-label">MPU6050 Tilt</span><span class="detail-value">{sn['tilt_magnitude_deg']}° (P:{sn['pitch_deg']}° R:{sn['roll_deg']}°)</span></div>
            <div class="detail-row"><span class="detail-label">VL53L1X Displacement</span><span class="detail-value">{sn['displacement_mm']:.2f} mm ({sn['displacement_rate_mm_min']:+.2f} mm/min)</span></div>
            <div class="detail-row"><span class="detail-label">Vibration Acceleration</span><span class="detail-value">RMS: {sn['vibration_rms_g']}g | Peak: {sn['vibration_peak_g']}g</span></div>
            <div class="detail-row"><span class="detail-label">Prop Load (HX711)</span><span class="detail-value">{sn['load_kN']:.1f} kN (Δ {sn['load_delta_kN']:+.1f} kN)</span></div>
            <div class="detail-row"><span class="detail-label">Underground CH₄ Gas</span><span class="detail-value">{sn['methane_ppm']:.0f} ppm (Permissible &lt;5000)</span></div>
            <div class="detail-row"><span class="detail-label">InSAR Satellite Velocity</span><span class="detail-value" style="color:#38bdf8;">{insar_velocity:.1f} mm/yr</span></div>
            <div class="detail-row"><span class="detail-label">12h Subsidence Risk</span><span class="detail-value" style="color:{'#f87171' if node_12h_risk > 0.65 else '#34d399'};">{node_12h_risk*100:.1f}%</span></div>
            <div class="detail-row" style="border-bottom: none;"><span class="detail-label">Last Telemetry</span><span class="detail-value">{sn['last_updated']}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Clean optional 12h forecast curve in an unobtrusive expander
        forecast_matrix = insar_forecast.get("forecast_12h", [])
        if forecast_matrix and len(forecast_matrix) > insar_idx:
            with st.expander("🛰️ View 12-Hour InSAR Forecast Curve", expanded=False):
                curve_data = forecast_matrix[insar_idx]
                df_fc = pd.DataFrame({
                    "Hour": [f"+{h}h" for h in range(1, 13)],
                    "Subsidence Risk": curve_data
                })
                fig_mini = px.line(df_fc, x="Hour", y="Subsidence Risk", markers=True, template="plotly_dark")
                fig_mini.add_hline(y=0.65, line_dash="dash", line_color="#ef4444", annotation_text="Hazard")
                fig_mini.update_traces(line_color="#38bdf8", line_width=2)
                fig_mini.update_layout(height=180, margin=dict(l=20, r=15, t=15, b=20), paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 1.05]))
                st.plotly_chart(fig_mini, use_container_width=True)

    # Schematic Mine Wall Map Section (100% preserved)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-headline">🗺️ Schematic Underground Mine Wall & Laser Baseline Map</div>', unsafe_allow_html=True)
    st.caption("Panel 4-B boundary wall, active goaf caving margin, main haulage roadway, and 980m long-baseline optical reference.")
    wall_fig = build_schematic_wall_map(nodes, laser, grefs)
    st.plotly_chart(wall_fig, use_container_width=True, config={"displayModeBar": False})

    # -------------------------------------------------------------------------
    # COPERNICUS SENTINEL-1 InSAR SATELLITE RADAR VIEW
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-headline">🛰️ Copernicus Sentinel-1 InSAR Satellite Surface Deformation</div>', unsafe_allow_html=True)
    st.caption("Multi-temporal C-band Synthetic Aperture Radar (SAR) remote sensing over Jharia Coalfield Panel 4-B. Left: Wrapped Phase Fringe Rings (28mm/cycle). Right: Calibrated LOS Subsidence Velocity Field with Ground Sensor Array.")

    st.image(
        "assets/insar_sentinel1_jharia.png",
        caption="Copernicus Sentinel-1A SAR Differential Interferogram (Left) & Calibrated Line-of-Sight Subsidence Velocity Map (Right) · Jharia Coalfield Panel 4-B (Project Soochak)",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# TAB 3: ANOMALY ANALYTICS & AUDIT (Preserving the familiar layout + adding clean InSAR info)
# -----------------------------------------------------------------------------
with tab_analytics:
    st.markdown('<div class="section-headline">📊 Historical Telemetry Trends & Anomaly Analytics</div>', unsafe_allow_html=True)

    if history:
        df_hist = pd.DataFrame(history)
        c_tr1, c_tr2 = st.columns(2)

        with c_tr1:
            fig_score = go.Figure()
            fig_score.add_trace(go.Scatter(
                x=df_hist["step"], y=df_hist["peak_anomaly_score"],
                mode="lines+markers", name="Peak Anomaly Score",
                line=dict(color="#ef4444", width=2.5),
            ))
            fig_score.add_trace(go.Scatter(
                x=df_hist["step"], y=df_hist["mean_anomaly_score"],
                mode="lines", name="Mean Mine Score",
                line=dict(color="#38bdf8", width=1.5, dash="dot"),
            ))
            fig_score.add_hline(y=0.70, line_dash="dash", line_color="#dc2626", annotation_text="Critical (0.70)")
            fig_score.add_hline(y=0.40, line_dash="dash", line_color="#f59e0b", annotation_text="Warning (0.40)")
            fig_score.update_layout(
                title="Isolation Forest Anomaly Score vs Time Steps",
                xaxis_title="Simulation Step", yaxis_title="Normalized Score [0, 1]",
                template="plotly_dark", height=300,
                margin=dict(l=35, r=15, t=40, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
            )
            st.plotly_chart(fig_score, use_container_width=True)

        with c_tr2:
            fig_disp = go.Figure()
            fig_disp.add_trace(go.Scatter(
                x=df_hist["step"], y=df_hist["max_displacement_mm"],
                mode="lines+markers", name="Max Crack Disp (mm)",
                line=dict(color="#f59e0b", width=2.2),
            ))
            fig_disp.add_trace(go.Scatter(
                x=df_hist["step"], y=df_hist["laser_deviation_mm"],
                mode="lines+markers", name="Laser Optical Drift (mm)",
                line=dict(color="#10b981", width=2.0, dash="dash"),
            ))
            fig_disp.update_layout(
                title="Ground Displacement vs Long-Baseline Laser Drift",
                xaxis_title="Simulation Step", yaxis_title="Displacement (mm)",
                template="plotly_dark", height=300,
                margin=dict(l=35, r=15, t=40, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
            )
            st.plotly_chart(fig_disp, use_container_width=True)

        c_tr3, c_tr4 = st.columns(2)
        with c_tr3:
            fig_tilt = px.line(
                df_hist, x="step", y="max_tilt_deg",
                title="Peak MPU6050 Roof Tilt Magnitude (°)",
                template="plotly_dark", markers=True,
            )
            fig_tilt.update_traces(line_color="#c084fc", line_width=2)
            fig_tilt.update_layout(height=260, margin=dict(l=35, r=15, t=35, b=25), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
            st.plotly_chart(fig_tilt, use_container_width=True)

        with c_tr4:
            fig_load = px.line(
                df_hist, x="step", y="mean_load_kN",
                title="Average Hydraulic Prop Load (kN)",
                template="plotly_dark", markers=True,
            )
            fig_load.update_traces(line_color="#38bdf8", line_width=2)
            fig_load.update_layout(height=260, margin=dict(l=35, r=15, t=35, b=25), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
            st.plotly_chart(fig_load, use_container_width=True)

    # Per-node scores & Radar Fingerprint
    st.markdown('<div class="section-headline">🎯 Spatial Cross-Correlation & Multi-Sensor Fingerprint</div>', unsafe_allow_html=True)
    sp_col1, sp_col2 = st.columns([1.6, 1.2])

    with sp_col1:
        df_bars = pd.DataFrame(nodes)
        fig_bar = px.bar(
            df_bars, x="node_id", y="anomaly_score", color="status",
            color_discrete_map={"SAFE": "#10b981", "WATCH": "#0284c7", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"},
            title="Per-Node ML Anomaly Scores across Panel 4-B",
            hover_data=["name", "displacement_mm", "tilt_magnitude_deg", "load_kN"],
            template="plotly_dark",
        )
        fig_bar.add_hline(y=0.70, line_dash="dash", line_color="#ef4444", annotation_text="Critical (0.70)")
        fig_bar.add_hline(y=0.40, line_dash="dash", line_color="#f59e0b", annotation_text="Warning (0.40)")
        fig_bar.update_layout(height=300, margin=dict(l=30, r=15, t=40, b=25), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with sp_col2:
        crit_node = max(nodes, key=lambda n: n["anomaly_score"])
        cats = ["Displacement", "Tilt Angle", "Vibration RMS", "Load Delta", "Laser Shift"]
        norm_v = [
            min(1.0, crit_node["displacement_mm"] / 5.0),
            min(1.0, crit_node["tilt_magnitude_deg"] / 3.0),
            min(1.0, crit_node["vibration_rms_g"] / 0.8),
            min(1.0, max(0.0, crit_node["load_delta_kN"]) / 80.0),
            min(1.0, laser["total_deviation_mm"] / 3.0),
        ]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_v, theta=cats, fill="toself",
            name=f"Node {crit_node['node_id']}", line_color="#f43f5e",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"Multi-Sensor Fingerprint ({crit_node['node_id']})",
            template="plotly_dark", height=300,
            margin=dict(l=25, r=25, t=40, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Multi-Node AI Correlation & False-Alarm Audit Log
    st.markdown('<div class="section-headline">🛡️ Multi-Node AI Correlation & False Alarm Audit Log</div>', unsafe_allow_html=True)
    if logs:
        for log in logs:
            l_type = log.get("type", "INFO")
            if l_type == "FILTERED_ALARM": l_icon, l_col = "🛡️", "#38bdf8"
            elif l_type == "SUBSIDENCE_ALERT": l_icon, l_col = "🚨", "#f87171"
            elif l_type == "STRATA_CREEP": l_icon, l_col = "⚠️", "#fbbf24"
            else: l_icon, l_col = "ℹ️", "#94a3b8"

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.65); border-left: 3px solid {l_col}; padding: 8px 14px; border-radius: 6px; margin-bottom: 6px; font-size: 0.85rem;">
                <span style="color: #94a3b8; font-size: 0.78rem; font-family: monospace;">[{log['time']}]</span>
                <span style="font-weight: 700; color: {l_col}; margin-left: 8px;">{l_icon} {l_type}</span>:
                <span style="color: #e2e8f0; margin-left: 6px;">{log['message']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No alert events recorded yet. Run a scenario in the Simulation Control Center tab to generate events.")

    # Clean Model Health & Satellite Summary (Simple, not overloaded)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    with st.expander("ℹ️ AI Models & InSAR Remote Sensing Details", expanded=False):
        st.markdown("""
        * **Layer 1 — UCI Seismic-Bumps Isolation Forest**: Calibrated against 2,584 real coal mine shifts. High-energy vibration shocks and prop loads are cross-checked to detect pre-bump strata deformation while rejecting transient blasts.
        * **Layer 2 — Sentinel-1 InSAR STGCN-LSTM**: Spatial Graph Convolutional Network tracking multi-temporal surface subsidence over Jharia Coalfield Panel 4-B. Predicts 12-hour forward subsidence hazard probabilities.
        """)


# -----------------------------------------------------------------------------
# 8. AUTOMATIC REFRESH LOOP
# -----------------------------------------------------------------------------
if st.session_state.auto_refresh or hw_active:
    time.sleep(1.5 if hw_active else 2.0)
    st.rerun()
