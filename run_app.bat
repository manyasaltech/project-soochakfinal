@echo off
title Project Soochak - Mine Subsidence Monitoring Dashboard
echo =====================================================================
echo  PROJECT SOOCHAK: MINE SUBSIDENCE MONITORING SYSTEM
echo  Smart India Hackathon 2026 - Problem Statement 26025
echo  Team MATR
echo =====================================================================
echo.
echo Starting Streamlit & Folium Anomaly Heatmap Application...
echo.
cd /d "%~dp0"
"%~dp0.venv\Scripts\streamlit.exe" run app.py
pause
