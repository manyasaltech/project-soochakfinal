# Project Soochak ⛏️

AI-Enabled Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System.

**Smart India Hackathon 2026 · Problem Statement 26025 · Team MATR**

## What It Does

A real-time mine subsidence monitoring dashboard that combines:
- **Live ESP32 hardware telemetry** via LoRa (SX1278 @ 433MHz)
- **Isolation Forest ML** anomaly detection across 8 sensor nodes
- **Interactive mine grid** with health scoring (0–100)
- **Wall map** showing panel boundaries, goaf zones, and laser baseline
- **Analytics** — anomaly trends, displacement charts, radar fingerprints

Node **S-103** receives live data from a physical ESP32 transmitter (MPU6050 + VL53L0X + HX711) over LoRa, parsed via USB serial on the receiver side.

## Hardware

| Sensor | Purpose |
|--------|---------|
| MPU6050 | 6-DoF IMU (tilt, vibration) |
| VL53L0X | Laser Time-of-Flight (crack displacement) |
| HX711 | Strain gauge load cell (strata load) |
| SX1278 | LoRa 433MHz radio (wireless link) |

**Transmitter**: ESP32 DevKit with all sensors → LoRa TX every 3s  
**Receiver**: ESP32-C3 Super Mini → USB Serial → Python dashboard

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, select your ESP32 receiver port in the top-right, and click **Connect**.

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard (UI, grid, map, charts) |
| `data_engine.py` | Simulation engine + Isolation Forest ML |
| `hardware.py` | Serial bridge to ESP32 receiver |
| `server.py` | Standalone HTML API server (alternative frontend) |
| `index.html` | Lightweight HTML dashboard |

## Screenshots

*Connect your ESP32 and open the dashboard to see live telemetry on S-103.*
