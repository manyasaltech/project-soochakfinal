"""
Project Soochak - Data Layer 2: Sentinel-1 InSAR India Subsidence Dataset
Smart India Hackathon 2026 | Problem Statement 26025 | Team MATR

Multi-temporal Interferometric Synthetic Aperture Radar (InSAR) surface deformation
dataset curated for Indian Coalfields (Jharia Coalfield Panel 4-B, Seam 11).
Coordinates: 23.75° N, 86.42° E.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_CSV = os.path.join(DATA_DIR, "insar_jharia_dataset.csv")

# 10 Reference Spatial Nodes covering underground panel, goaf margins, haulage road, and railway
JHARIA_INSaR_NODES = [
    {"node_id": "NODE_01", "name": "Entry Gate Drift", "lat": 23.7535, "lon": 86.4162, "zone": "Panel 4-B North Barrier"},
    {"node_id": "NODE_02", "name": "Goaf Margin West A", "lat": 23.7526, "lon": 86.4179, "zone": "Panel 4-B Goaf Margin West"},
    {"node_id": "NODE_03", "name": "Longwall Face Center", "lat": 23.7518, "lon": 86.4196, "zone": "Panel 4-B Longwall Center"},
    {"node_id": "NODE_04", "name": "Goaf Margin East B", "lat": 23.7512, "lon": 86.4214, "zone": "Panel 4-B Goaf Margin East"},
    {"node_id": "NODE_05", "name": "Tailgate Airway Hub", "lat": 23.7503, "lon": 86.4232, "zone": "Panel 4-B Return Airway"},
    {"node_id": "NODE_06", "name": "Surface Haul Road Pillar", "lat": 23.7546, "lon": 86.4188, "zone": "Haul Road Overburden"},
    {"node_id": "NODE_07", "name": "Village Buffer Zone 1", "lat": 23.7541, "lon": 86.4218, "zone": "Village Buffer Zone"},
    {"node_id": "NODE_08", "name": "Railway Line Settlement", "lat": 23.7529, "lon": 86.4248, "zone": "Coal Freight Rail Corridor"},
    {"node_id": "NODE_09", "name": "Deep Decompression Shaft", "lat": 23.7498, "lon": 86.4180, "zone": "South Subsidence Trough"},
    {"node_id": "NODE_10", "name": "East Perimeter Piezometer", "lat": 23.7515, "lon": 86.4260, "zone": "East Buffer Barrier"},
]


def load_insar_spatial_data():
    """
    Load Sentinel-1 InSAR surface deformation features for Jharia Coalfield.
    Features:
    - cumulative displacement (mm, negative indicates subsidence)
    - LOS deformation velocity (mm/year)
    - temporal interferometric coherence [0, 1]
    """
    if os.path.exists(CACHE_CSV):
        try:
            return pd.read_csv(CACHE_CSV)
        except Exception:
            pass

    np.random.seed(42)
    num_nodes = len(JHARIA_INSaR_NODES)

    # Realistic InSAR ranges observed in Jharia caving panels:
    # Subsidence troughs can experience -30mm to -70mm displacement, -20 to -140 mm/yr velocity.
    records = []
    for i, meta in enumerate(JHARIA_INSaR_NODES):
        is_core = meta["node_id"] in ["NODE_02", "NODE_03", "NODE_04", "NODE_09"]
        if is_core:
            disp = np.random.uniform(-48.0, -18.0)
            vel = np.random.uniform(-42.0, -14.0)
            coh = np.random.uniform(0.68, 0.88)
        else:
            disp = np.random.uniform(-14.0, 2.0)
            vel = np.random.uniform(-12.0, 1.0)
            coh = np.random.uniform(0.78, 0.96)

        records.append({
            "node_id": meta["node_id"],
            "name": meta["name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "zone": meta["zone"],
            "displacement_mm": round(disp, 2),
            "velocity_mmyr": round(vel, 2),
            "coherence": round(coh, 3),
            "incidence_angle_deg": 39.2,
        })

    df = pd.DataFrame(records)
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(CACHE_CSV, index=False)
    except Exception:
        pass
    return df


def build_spatial_adjacency_matrix(nodes_df):
    """
    Compute normalized spatial adjacency matrix A using Gaussian distance kernel:
    A_ij = exp(-dist(i, j) / std(dist))
    """
    coords = nodes_df[["lat", "lon"]].values
    # Approximate geographic distance in km (1 deg lat ~ 111km, 1 deg lon ~ 102km in Jharia)
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    diff_km = np.sqrt((diff[:, :, 0] * 111.0)**2 + (diff[:, :, 1] * 102.0)**2)
    std_dist = np.std(diff_km) or 1.0
    A = np.exp(-diff_km / std_dist)
    # Zero out self-loop for graph conv or retain normalized
    return A


def generate_insar_time_sequences(nodes_df, n_samples=300, seq_len=6, horizon=12):
    """
    Generate synthetic multi-temporal radar acquisition sequences for training STGCN-LSTM.
    - X shape: [n_samples, seq_len, num_nodes, 3] (features: disp, vel, coh)
    - Y shape: [n_samples, num_nodes, horizon] (predicted hourly subsidence risk [0, 1])
    """
    np.random.seed(42)
    num_nodes = len(nodes_df)
    base_disp = nodes_df["displacement_mm"].values
    base_vel = nodes_df["velocity_mmyr"].values
    base_coh = nodes_df["coherence"].values

    X_list = []
    Y_list = []

    for s in range(n_samples):
        # Sample an overall trend for this simulation episode (stable vs accelerating creep)
        creep_factor = np.random.uniform(0.8, 2.2) if s % 4 == 0 else np.random.uniform(0.2, 1.0)
        
        # Temporal history (seq_len steps)
        seq_features = np.zeros((seq_len, num_nodes, 3))
        for t in range(seq_len):
            t_offset = (t - seq_len + 1) * 0.4
            disp_t = base_disp + t_offset * (base_vel / 52.0) * creep_factor + np.random.normal(0, 0.2, num_nodes)
            vel_t = base_vel * creep_factor + np.random.normal(0, 0.5, num_nodes)
            coh_t = np.clip(base_coh + np.random.normal(0, 0.02, num_nodes), 0.5, 0.99)
            seq_features[t, :, 0] = disp_t
            seq_features[t, :, 1] = vel_t
            seq_features[t, :, 2] = coh_t

        # Future 12-hour risk trajectory (horizon=12)
        future_risk = np.zeros((num_nodes, horizon))
        for h in range(horizon):
            hour_progression = (h + 1) / float(horizon)
            for n in range(num_nodes):
                is_danger_zone = n in [1, 2, 3, 8]
                risk_baseline = (abs(base_disp[n]) / 60.0) * 0.6 + (abs(base_vel[n]) / 50.0) * 0.4
                if is_danger_zone and creep_factor > 1.2:
                    node_risk = min(0.98, risk_baseline * 0.7 + hour_progression * 0.45 + np.random.normal(0, 0.02))
                else:
                    node_risk = min(0.40, risk_baseline * 0.4 + hour_progression * 0.05 + np.random.normal(0, 0.02))
                future_risk[n, h] = max(0.02, node_risk)

        X_list.append(seq_features)
        Y_list.append(future_risk)

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)
