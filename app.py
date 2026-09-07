"""
Project Soochak — Minimal Dashboard
"""

import time
import math

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

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Soochak",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: AeroLinkTree dark grain glassmorphism ───────────────────────────────
st.markdown("""
<div class="grain-overlay"></div>
<style>
    /* Hide sidebar toggle + hamburger */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="stAppViewContainer"] {
        background-color: #0a0c10;
        color: #f3f4f6;
        background-image:
            radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.12), transparent 55%),
            radial-gradient(circle at 100% 100%, rgba(52, 211, 153, 0.06), transparent 50%);
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .grain-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none; z-index: 999999;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
        opacity: 0.04;
    }
    /* Glass cards */
    .glass { background: rgba(17,24,39,0.55); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 18px; }
    .glass-sm { background: rgba(17,24,39,0.55); backdrop-filter: blur(14px); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 12px 14px; }
    /* KPI */
    .kpi { text-align: center; }
    .kpi .label { font-size: 0.72rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .kpi .val { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 800; margin: 4px 0; }
    .kpi .tag { font-size: 0.7rem; font-weight: 700; padding: 2px 10px; border-radius: 20px; display: inline-block; }
    .tag-safe { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
    .tag-warn { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
    .tag-crit { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .tag-info { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
    /* Grid tiles */
    .phi-grid { display: grid; gap: 8px; }
    .phi-tile { aspect-ratio: 1/1; border-radius: 10px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; border: 2px solid transparent; cursor: default; transition: transform 0.15s; }
    .phi-tile:hover { transform: scale(1.06); }
    .phi-score { font-size: 1.2rem; font-weight: 800; color: #0b1120; }
    .phi-id { font-size: 0.5rem; font-weight: 700; color: rgba(11,17,32,0.6); margin-top: 2px; }
    .phi-excellent { background: #16a34a; }
    .phi-good { background: #86efac; }
    .phi-suboptimal { background: #f59e0b; }
    .phi-critical { background: #f87171; }
    .phi-empty { background: rgba(30,41,59,0.3); border: 1px dashed rgba(75,85,99,0.2); }
    .phi-selected { border-color: #38bdf8; box-shadow: 0 0 0 3px rgba(56,189,248,0.3); }
    .phi-live { position: relative; }
    .phi-live::after { content: "LIVE"; position: absolute; top: 3px; right: 3px; font-size: 0.4rem; background: #ef4444; color: white; padding: 1px 4px; border-radius: 4px; font-weight: 800; animation: blink 1.5s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }
    /* Detail card */
    .detail-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.8rem; }
    .detail-row .dl { color: #9ca3af; }
    .detail-row .dv { color: #f3f4f6; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    /* Hardware badge */
    .hw-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .hw-on { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
    .hw-off { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid #f87171; }
    .pulse-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: blink 1.5s infinite; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────────────
def kpi(label, value, tag_text, tag_class):
    return f'<div class="glass-sm kpi"><div class="label">{label}</div><div class="val">{value}</div><div class="tag {tag_class}">{tag_text}</div></div>'

def threat_tag(level):
    m = {"SAFE": ("tag-safe", "Normal"), "WATCH": ("tag-info", "Watch"), "WARNING": ("tag-warn", "Warning"), "CRITICAL": ("tag-crit", "Evacuate")}
    return m.get(level, ("tag-safe", level))

def score_tag(s):
    if s >= 0.70: return "tag-crit", "Critical"
    if s >= 0.40: return "tag-warn", "Elevated"
    return "tag-safe", "Normal"

def disp_tag(d):
    if d >= 5.0: return "tag-crit", "Critical"
    if d >= 2.0: return "tag-warn", "Elevated"
    return "tag-safe", "Normal"

def mine_health_score(a): return max(0, min(100, int(round((1.0 - a) * 100))))

def health_tier(s):
    if s >= 80: return "excellent", "Excellent"
    if s >= 60: return "good", "Good"
    if s >= 40: return "suboptimal", "Suboptimal"
    return "critical", "Critical"


# ── Sensor Grid ──────────────────────────────────────────────────────────────
GRID_C, GRID_R = 8, 6

def compute_grid(nodes_list):
    lats = [n["lat"] for n in nodes_list]; lons = [n["lon"] for n in nodes_list]
    la1, la2 = min(lats), max(lats); lo1, lo2 = min(lons), max(lons)
    lp = (la2-la1)*0.15 or 0.001; lop = (lo2-lo1)*0.15 or 0.001
    la1-=lp; la2+=lp; lo1-=lop; lo2+=lop
    p = {}
    for n in nodes_list:
        c = min(GRID_C-1, int((n["lon"]-lo1)/(lo2-lo1)*GRID_C))
        r = min(GRID_R-1, int(GRID_R - (n["lat"]-la1)/(la2-la1)*GRID_R))
        p[(r,c)] = n
    return p

def grid_refs(nodes_list):
    refs = {}
    for (r,c), n in compute_grid(nodes_list).items():
        refs[n["node_id"]] = f"{chr(65+c)}{r+1}"
    return refs

def grid_html(nodes_list, sel_id=None):
    pl = compute_grid(nodes_list)
    cells = []
    for r in range(GRID_R):
        for c in range(GRID_C):
            n = pl.get((r,c))
            if not n:
                cells.append('<div class="phi-tile phi-empty"></div>')
                continue
            sc = mine_health_score(n["anomaly_score"])
            t, _ = health_tier(sc)
            sel = " phi-selected" if n["node_id"] == sel_id else ""
            live = " phi-live" if n["node_id"] == "S-103" and bridge.running else ""
            cells.append(f'<div class="phi-tile phi-{t}{sel}{live}"><span class="phi-score">{sc}</span><span class="phi-id">{n["node_id"]}</span></div>')
    return f'<div class="phi-grid" style="grid-template-columns:repeat({GRID_C},1fr);">{"".join(cells)}</div>'


# ── Wall Map ─────────────────────────────────────────────────────────────────
panel_poly = [[23.7542,86.4155],[23.7548,86.4235],[23.7500,86.4242],[23.7495,86.4162]]
goaf_poly = [[23.7530,86.4175],[23.7520,86.4215],[23.7508,86.4210],[23.7518,86.4170]]
WC, WR = 12, 9

def build_map(nodes_list, laser_dict, grefs):
    all_lats = [n["lat"] for n in nodes_list]+[p[0] for p in panel_poly+goaf_poly]+[laser_dict["tx_lat"],laser_dict["rx_lat"]]
    all_lons = [n["lon"] for n in nodes_list]+[p[1] for p in panel_poly+goaf_poly]+[laser_dict["tx_lon"],laser_dict["rx_lon"]]
    la1,la2,lo1,lo2 = min(all_lats),max(all_lats),min(all_lons),max(all_lons)
    lp=(la2-la1)*0.18 or 0.001; lop=(lo2-lo1)*0.18 or 0.001
    la1-=lp; la2+=lp; lo1-=lop; lo2+=lop
    def pr(la,lo): return ((lo-lo1)/(lo2-lo1)*WC, (la-la1)/(la2-la1)*WR)
    nl = {n["node_id"]:n for n in nodes_list}
    fig = go.Figure()
    # Panel
    pp = [pr(la,lo) for la,lo in panel_poly]+[pr(*panel_poly[0])]
    fig.add_trace(go.Scatter(x=[p[0] for p in pp],y=[p[1] for p in pp],mode="lines",fill="toself",fillcolor="rgba(59,130,246,0.07)",line=dict(color="#3b82f6",width=3),name="Panel 4-B"))
    # Goaf
    gp = [pr(la,lo) for la,lo in goaf_poly]+[pr(*goaf_poly[0])]
    fig.add_trace(go.Scatter(x=[p[0] for p in gp],y=[p[1] for p in gp],mode="lines",fill="toself",fillcolor="rgba(249,115,22,0.12)",line=dict(color="#f97316",width=2,dash="dash"),name="Goaf Zone"))
    # Roadway
    road = [nid for nid in ["S-101","S-102","S-103","S-104","S-105"] if nid in nl]
    rx = [pr(nl[n]["lat"],nl[n]["lon"])[0] for n in road]
    ry = [pr(nl[n]["lat"],nl[n]["lon"])[1] for n in road]
    fig.add_trace(go.Scatter(x=rx,y=ry,mode="lines",line=dict(color="#94a3b8",width=8),opacity=0.25,name="Main Gate",hoverinfo="skip"))
    # Laser
    tx=pr(laser_dict["tx_lat"],laser_dict["tx_lon"]); rxp=pr(laser_dict["rx_lat"],laser_dict["rx_lon"])
    dev=laser_dict["total_deviation_mm"]
    lc = "#ef4444" if dev>2.5 else ("#f59e0b" if dev>0.8 else "#10b981")
    fig.add_trace(go.Scatter(x=[tx[0],rxp[0]],y=[tx[1],rxp[1]],mode="lines+markers",line=dict(color=lc,width=2,dash="dot"),marker=dict(size=10,symbol="diamond",color=lc),name=f"Laser ({dev:.2f}mm)"))
    # Nodes
    sc = {"SAFE":"#10b981","WATCH":"#0284c7","WARNING":"#f97316","CRITICAL":"#ef4444"}
    nx,ny,nc,nt,nh,ns = [],[],[],[],[],[]
    for n in nodes_list:
        x,y = pr(n["lat"],n["lon"]); nx.append(x); ny.append(y)
        nc.append(sc.get(n["status"],"#10b981")); nt.append(n["node_id"])
        ns.append(14+n["anomaly_score"]*14)
        nh.append(f"<b>{n['node_id']}</b> ({n['status']})<br>Tilt:{n['tilt_magnitude_deg']}° Disp:{n['displacement_mm']:.2f}mm")
    fig.add_trace(go.Scatter(x=nx,y=ny,mode="markers+text",marker=dict(size=ns,color=nc,line=dict(color="white",width=1.5),symbol="square"),text=nt,textposition="top center",textfont=dict(color="#e5e7eb",size=10,family="monospace"),hovertext=nh,hoverinfo="text",showlegend=False))
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0f172a",height=420,margin=dict(l=10,r=10,t=10,b=10),legend=dict(orientation="h",y=1.02,font=dict(size=9)),
        xaxis=dict(range=[0,WC],showgrid=True,gridcolor="rgba(148,163,184,0.12)",dtick=1,zeroline=False,fixedrange=True),
        yaxis=dict(range=[0,WR],showgrid=True,gridcolor="rgba(148,163,184,0.12)",dtick=1,zeroline=False,scaleanchor="x",scaleratio=1,fixedrange=True),hovermode="closest")
    return fig


# ── Init ─────────────────────────────────────────────────────────────────────
if "simulator" not in st.session_state:
    with st.spinner("Initializing AI Engine..."):
        st.session_state.simulator = MineEnvironmentSimulator()
        st.session_state.current_data = st.session_state.simulator.step(OperationalScenario.NORMAL)
        st.session_state.auto_refresh = False

sim = st.session_state.simulator

# ── Header row ───────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#38bdf8;letter-spacing:2px;margin-bottom:0;">SIH 2026 · PS 26025 · TEAM MATR</p>', unsafe_allow_html=True)
    st.markdown('<h1 style="margin:0;font-size:2rem;">PROJECT SOOCHAK</h1>', unsafe_allow_html=True)
with h2:
    # Hardware connect inline
    ports = get_available_ports()
    pc1, pc2 = st.columns([2, 1])
    with pc1:
        sel_port = st.selectbox("Port", ports if ports else ["No ports"], label_visibility="collapsed", key="port_sel")
    with pc2:
        if not bridge.running:
            if st.button("Connect", use_container_width=True):
                if sel_port and sel_port != "No ports":
                    ok, msg = bridge.connect(sel_port)
                    if not ok: st.error(msg)
                    else: st.rerun()
        else:
            if st.button("Disconnect", use_container_width=True):
                bridge.disconnect()
                st.rerun()
    if bridge.running:
        st.markdown(f'<span class="hw-badge hw-on"><span class="pulse-dot"></span>LIVE · {bridge.port}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="hw-badge hw-off">OFFLINE</span>', unsafe_allow_html=True)

# ── Simulation Controls (compact row) ────────────────────────────────────────
scenarios = {
    "Normal": OperationalScenario.NORMAL,
    "Blasting (False Alarm)": OperationalScenario.BLASTING,
    "Strata Creep": OperationalScenario.STRATA_CREEP,
    "Critical Subsidence": OperationalScenario.CRITICAL_SUBSIDENCE,
    "Sensor Glitch": OperationalScenario.SENSOR_GLITCH,
}

with st.container():
    sc1, sc2, sc3, sc4, sc5 = st.columns([2, 1, 1, 1, 1.5])
    with sc1:
        scenario_label = st.selectbox("Scenario", list(scenarios.keys()), label_visibility="collapsed")
        current_scenario = scenarios[scenario_label]
    with sc2:
        step_clicked = st.button("⚡ Step", use_container_width=True)
    with sc3:
        reset_clicked = st.button("🔄 Reset", use_container_width=True)
    with sc4:
        auto_run = st.checkbox("Auto", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_run
    with sc5:
        manual_disp = st.slider("Inject Disp (mm)", 0.0, 10.0, 0.0, 0.5, label_visibility="collapsed")

manual_tilt = 0.0
manual_laser = 0.0

if reset_clicked:
    st.session_state.simulator = MineEnvironmentSimulator()
    sim = st.session_state.simulator
    st.session_state.current_data = sim.step(OperationalScenario.NORMAL)
    st.rerun()

if step_clicked:
    st.session_state.current_data = sim.step(scenario=current_scenario, manual_disp_offset=manual_disp, manual_tilt_offset=manual_tilt, manual_laser_offset=manual_laser)
    st.rerun()

# When hardware is connected OR auto is checked, always advance to pick up fresh data
hw_live = bridge.running and bridge.latest_data is not None
if (st.session_state.auto_refresh or hw_live) and not step_clicked:
    st.session_state.current_data = sim.step(scenario=current_scenario, manual_disp_offset=manual_disp, manual_tilt_offset=manual_tilt, manual_laser_offset=manual_laser)

data = st.session_state.current_data
nodes = data["nodes"]; laser = data["laser_system"]; threat_level = data["threat_level"]
peak_score = data["peak_anomaly_score"]; false_alarms = data["false_alarms_rejected"]
history = data["history"]; logs = data["logs"]

# ── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    tc, tl = threat_tag(threat_level)
    st.markdown(kpi("Threat Level", threat_level, tl, tc), unsafe_allow_html=True)
with k2:
    tc, tl = score_tag(peak_score)
    st.markdown(kpi("Peak ML Score", f"{peak_score:.3f}", tl, tc), unsafe_allow_html=True)
with k3:
    mx = max(n["displacement_mm"] for n in nodes)
    tc, tl = disp_tag(mx)
    st.markdown(kpi("Max Displacement", f"{mx:.2f} mm", tl, tc), unsafe_allow_html=True)
with k4:
    st.markdown(kpi("False Alarms Blocked", str(false_alarms), "AI Active", "tag-info"), unsafe_allow_html=True)

# ── Main Content: Grid + Detail | Map ────────────────────────────────────────
grefs = grid_refs(nodes)
node_lookup = {n["node_id"]: n for n in nodes}

if "selected_node_id" not in st.session_state or st.session_state.selected_node_id not in node_lookup:
    st.session_state.selected_node_id = max(nodes, key=lambda n: n["anomaly_score"])["node_id"]

left, right = st.columns([1.6, 1])

with left:
    # Grid
    st.markdown(grid_html(nodes, st.session_state.selected_node_id), unsafe_allow_html=True)
    # Node selector buttons
    cols = st.columns(len(nodes))
    for i, n in enumerate(nodes):
        with cols[i]:
            sel = n["node_id"] == st.session_state.selected_node_id
            if st.button(n["node_id"], key=f"s_{n['node_id']}", use_container_width=True, type="primary" if sel else "secondary"):
                st.session_state.selected_node_id = n["node_id"]
                st.rerun()

with right:
    sn = node_lookup[st.session_state.selected_node_id]
    sc = mine_health_score(sn["anomaly_score"])
    t, tl = health_tier(sc)
    tc = {"excellent":"#16a34a","good":"#4ade80","suboptimal":"#f59e0b","critical":"#f87171"}[t]
    is_live = sn["node_id"] == "S-103" and bridge.running

    live_badge = '<span style="background:#ef4444;color:white;padding:2px 8px;border-radius:10px;font-size:0.65rem;font-weight:800;margin-left:8px;animation:blink 1.5s infinite;">● LIVE ESP32</span>' if is_live else ''

    st.markdown(f"""
    <div class="glass">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
            <div><span style="font-size:1rem;font-weight:800;color:#e5e7eb;">{sn['node_id']}</span>{live_badge}<br><span style="font-size:0.7rem;color:#9ca3af;">{sn['name']}</span></div>
            <div style="text-align:right;"><span style="font-size:1.5rem;font-weight:800;color:{tc};font-family:JetBrains Mono,monospace;">{sc}</span><br><span style="font-size:0.6rem;color:{tc};font-weight:700;">{tl}</span></div>
        </div>
        <div class="detail-row"><span class="dl">Grid Ref</span><span class="dv">{grefs.get(sn['node_id'],'-')}</span></div>
        <div class="detail-row"><span class="dl">Status</span><span class="dv">{sn['status']}</span></div>
        <div class="detail-row"><span class="dl">Tilt</span><span class="dv">{sn['tilt_magnitude_deg']}°</span></div>
        <div class="detail-row"><span class="dl">Displacement</span><span class="dv">{sn['displacement_mm']:.2f} mm</span></div>
        <div class="detail-row"><span class="dl">Vibration</span><span class="dv">{sn['vibration_rms_g']} g</span></div>
        <div class="detail-row"><span class="dl">Load</span><span class="dv">{sn['load_kN']:.1f} kN</span></div>
        <div class="detail-row"><span class="dl">Temp</span><span class="dv">{sn['temp_c']:.1f}°C</span></div>
        <div class="detail-row"><span class="dl">Methane</span><span class="dv">{sn['methane_ppm']:.0f} ppm</span></div>
        <div class="detail-row" style="border-bottom:none;"><span class="dl">Battery</span><span class="dv">{sn['battery_pct']:.0f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── Wall Map ─────────────────────────────────────────────────────────────────
st.plotly_chart(build_map(nodes, laser, grefs), use_container_width=True, config={"displayModeBar": False})

# ── Analytics ────────────────────────────────────────────────────────────────
if history:
    df = pd.DataFrame(history)
    if not df.empty:
        a1, a2 = st.columns(2)
        with a1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["step"],y=df["peak_anomaly_score"],mode="lines+markers",name="Peak Score",line=dict(color="#ef4444",width=2)))
            fig.add_trace(go.Scatter(x=df["step"],y=df["mean_anomaly_score"],mode="lines",name="Mean",line=dict(color="#3b82f6",width=1.5,dash="dot")))
            fig.add_hline(y=0.70,line_dash="dash",line_color="#dc2626"); fig.add_hline(y=0.40,line_dash="dash",line_color="#d97706")
            fig.update_layout(title="Anomaly Score",template="plotly_dark",height=280,margin=dict(l=30,r=10,t=35,b=25),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with a2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["step"],y=df["max_displacement_mm"],mode="lines+markers",name="Max Displacement",line=dict(color="#f59e0b",width=2)))
            fig.add_trace(go.Scatter(x=df["step"],y=df["laser_deviation_mm"],mode="lines+markers",name="Laser Dev",line=dict(color="#10b981",width=2,dash="dash")))
            fig.update_layout(title="Displacement vs Laser",template="plotly_dark",height=280,margin=dict(l=30,r=10,t=35,b=25),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        a3, a4 = st.columns(2)
        with a3:
            dn = pd.DataFrame(nodes)
            fig = px.bar(dn,x="node_id",y="anomaly_score",color="status",color_discrete_map={"SAFE":"#10b981","WATCH":"#0284c7","WARNING":"#f59e0b","CRITICAL":"#ef4444"},template="plotly_dark")
            fig.add_hline(y=0.70,line_dash="dash",line_color="#ef4444"); fig.add_hline(y=0.40,line_dash="dash",line_color="#f59e0b")
            fig.update_layout(title="Per-Node Scores",height=280,margin=dict(l=30,r=10,t=35,b=25),paper_bgcolor="rgba(0,0,0,0)",showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with a4:
            cn = max(nodes,key=lambda n:n["anomaly_score"])
            cats = ["Displacement","Tilt","Vibration","Load Δ","Laser"]
            vals = [min(1,cn["displacement_mm"]/5),min(1,cn["tilt_magnitude_deg"]/3),min(1,cn["vibration_rms_g"]/0.8),min(1,max(0,cn["load_delta_kN"])/80),min(1,laser["total_deviation_mm"]/3)]
            fig = go.Figure(go.Scatterpolar(r=vals,theta=cats,fill='toself',line_color="#f43f5e"))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),title=f"Fingerprint ({cn['node_id']})",template="plotly_dark",height=280,margin=dict(l=30,r=30,t=35,b=25),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

# ── Auto refresh ─────────────────────────────────────────────────────────────
if st.session_state.auto_refresh or hw_live:
    time.sleep(1.5 if hw_live else 2.0)
    st.rerun()
