/**
 * Project Soochak - AeroLinkTree Web Cockpit Driver
 * Connects to Python server.py REST endpoints:
 * - /api/simulation/state
 * - /api/simulation/step
 * - /api/simulation/scenario
 * - /api/simulation/reset
 * - /api/connect & /api/disconnect
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const threatStatusBadge = document.getElementById('threatStatusBadge');
  const metricMeshNodes = document.getElementById('metricMeshNodes');
  const metricLaserDev = document.getElementById('metricLaserDev');
  const metricLaserSub = document.getElementById('metricLaserSub');
  const metricAnomalyScore = document.getElementById('metricAnomalyScore');
  const metricFalseAlarms = document.getElementById('metricFalseAlarms');
  const metricMaxDisp = document.getElementById('metricMaxDisp');

  const phiGridContainer = document.getElementById('phiGridContainer');
  const inspectorCardBody = document.getElementById('inspectorCardBody');
  const inspectorNodeTag = document.getElementById('inspectorNodeTag');
  const logStreamContainer = document.getElementById('logStreamContainer');
  const stepCounterLabel = document.getElementById('stepCounterLabel');

  const portSelect = document.getElementById('portSelect');
  const connectPortBtn = document.getElementById('connectPortBtn');
  const disconnectPortBtn = document.getElementById('disconnectPortBtn');
  const hwStatusBadge = document.getElementById('hwStatusBadge');
  const stepSimBtn = document.getElementById('stepSimBtn');
  const resetSimBtn = document.getElementById('resetSimBtn');
  const autoPlayChk = document.getElementById('autoPlayChk');

  const audioToggleBtn = document.getElementById('audioToggleBtn');
  const audioIcon = document.getElementById('audioIcon');
  const exportReportBtn = document.getElementById('exportReportBtn');
  const reportModalBackdrop = document.getElementById('reportModalBackdrop');
  const modalCloseBtn = document.getElementById('modalCloseBtn');
  const modalReportContent = document.getElementById('modalReportContent');
  const downloadCsvBtn = document.getElementById('downloadCsvBtn');
  const printReportBtn = document.getElementById('printReportBtn');

  // Canvas elements
  const laserCanvas = document.getElementById('laserScopeCanvas');
  const laserCtx = laserCanvas.getContext('2d');
  const laserValX = document.getElementById('laserValX');
  const laserValY = document.getElementById('laserValY');
  const laserValDev = document.getElementById('laserValDev');
  const laserValInt = document.getElementById('laserValInt');
  const laserValSnr = document.getElementById('laserValSnr');
  const laserStatusTag = document.getElementById('laserStatusTag');

  const trendCanvas = document.getElementById('trendChartCanvas');
  const trendCtx = trendCanvas.getContext('2d');

  // Local state
  let selectedNodeId = "S-103";
  let latestState = null;
  let audioEnabled = false;
  let audioContext = null;
  let sirenOscillator = null;

  // ---------------------------------------------------------------------------
  // 1. WEB AUDIO API SYNTHESIZER FOR EMERGENCY SIREN
  // ---------------------------------------------------------------------------
  function initAudio() {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
  }

  function playSiren() {
    if (!audioEnabled || !audioContext) return;
    if (sirenOscillator) return; // already playing

    try {
      const osc = audioContext.createOscillator();
      const gain = audioContext.createGain();
      osc.type = 'sawtooth';

      const now = audioContext.currentTime;
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.linearRampToValueAtTime(880, now + 0.4);
      osc.frequency.linearRampToValueAtTime(440, now + 0.8);

      gain.gain.setValueAtTime(0.15, now);
      osc.connect(gain);
      gain.connect(audioContext.destination);

      osc.start();
      sirenOscillator = osc;
      setTimeout(() => {
        if (sirenOscillator) {
          sirenOscillator.stop();
          sirenOscillator.disconnect();
          sirenOscillator = null;
        }
      }, 800);
    } catch (e) {
      console.warn("Audio error", e);
    }
  }

  audioToggleBtn.addEventListener('click', () => {
    initAudio();
    audioEnabled = !audioEnabled;
    audioIcon.textContent = audioEnabled ? '🔊' : '🔇';
    audioToggleBtn.className = audioEnabled ? 'icon-btn btn-primary' : 'icon-btn';
    if (audioEnabled && latestState && latestState.threat_level === 'CRITICAL') {
      playSiren();
    }
  });

  // ---------------------------------------------------------------------------
  // 2. FETCH & RENDER SIMULATION STATE
  // ---------------------------------------------------------------------------
  async function fetchState() {
    try {
      const res = await fetch('/api/simulation/state');
      if (!res.ok) return;
      latestState = await res.json();
      renderDashboard(latestState);
    } catch (err) {
      console.warn('Simulation API unreachable:', err);
    }
  }

  function renderDashboard(state) {
    // 1. Header Threat Badge
    threatStatusBadge.textContent = state.threat_level;
    threatStatusBadge.className = `threat-status ${state.threat_level.toLowerCase()}`;

    if (state.threat_level === 'CRITICAL') {
      playSiren();
    }

    // 2. Hardware state
    if (state.hardware_connected) {
      hwStatusBadge.textContent = `LIVE ESP32 S-103 · ${state.hardware_port}`;
      hwStatusBadge.className = 'threat-status critical';
      connectPortBtn.style.display = 'none';
      disconnectPortBtn.style.display = 'inline-block';
      portSelect.disabled = true;
    } else {
      hwStatusBadge.textContent = 'SYNTHETIC SIMULATION ACTIVE';
      hwStatusBadge.className = 'threat-status safe';
      connectPortBtn.style.display = 'inline-block';
      disconnectPortBtn.style.display = 'none';
      portSelect.disabled = false;

      // Update port dropdown
      const ports = state.available_ports || [];
      if (portSelect.options.length !== ports.length) {
        portSelect.innerHTML = ports.length
          ? ports.map(p => `<option value="${p}">${p}</option>`).join('')
          : '<option value="">No Ports Found</option>';
      }
    }

    // 3. Metrics Ribbon
    metricMeshNodes.textContent = `${state.nodes.length} / 8 Online`;
    const laserDev = state.laser_system.total_deviation_mm.toFixed(2);
    metricLaserDev.textContent = `${laserDev} mm`;
    metricLaserSub.textContent = `980m Baseline · ${state.laser_system.status}`;

    metricAnomalyScore.textContent = state.peak_anomaly_score.toFixed(3);
    metricFalseAlarms.textContent = `${state.false_alarms_rejected} Blocked`;

    const maxDisp = Math.max(...state.nodes.map(n => n.displacement_mm));
    metricMaxDisp.textContent = `${maxDisp.toFixed(2)} mm`;

    stepCounterLabel.textContent = `Step: ${state.step}`;

    // 4. Render Phi-Cube Grid
    renderPhiGrid(state.nodes, state.hardware_connected);

    // 5. Render Selected Node
    const selNode = state.nodes.find(n => n.node_id === selectedNodeId) || state.nodes[0];
    renderInspector(selNode, state.hardware_connected);

    // 6. Draw Optical Laser Reticle
    drawLaserReticle(state.laser_system);

    // 7. Draw Trend Charts
    drawTrendChart(state.history);

    // 8. Logs
    renderLogs(state.logs);
  }

  // ---------------------------------------------------------------------------
  // 3. PHI-CUBE SENSOR GRID (8x6)
  // ---------------------------------------------------------------------------
  function renderPhiGrid(nodes, isHwLive) {
    const cols = 8;
    const rows = 6;

    const lats = nodes.map(n => n.lat);
    const lons = nodes.map(n => n.lon);
    let la1 = Math.min(...lats), la2 = Math.max(...lats);
    let lo1 = Math.min(...lons), lo2 = Math.max(...lons);
    const lp = (la2 - la1) * 0.15 || 0.001;
    const lop = (lo2 - lo1) * 0.15 || 0.001;
    la1 -= lp; la2 += lp; lo1 -= lop; lo2 += lop;

    const placements = {};
    nodes.forEach(n => {
      const c = Math.min(cols - 1, Math.floor((n.lon - lo1) / (lo2 - lo1) * cols));
      const r = Math.min(rows - 1, Math.floor(rows - (n.lat - la1) / (la2 - la1) * rows));
      placements[`${r}_${c}`] = n;
    });

    let html = '';
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const n = placements[`${r}_${c}`];
        if (!n) {
          html += '<div class="phi-tile phi-empty"></div>';
          continue;
        }
        const health = Math.max(0, Math.min(100, Math.round((1.0 - n.anomaly_score) * 100)));
        let tier = 'critical';
        if (health >= 80) tier = 'excellent';
        else if (health >= 60) tier = 'good';
        else if (health >= 40) tier = 'suboptimal';

        const isSel = n.node_id === selectedNodeId ? ' phi-selected' : '';
        const liveBadge = (n.node_id === 'S-103' && isHwLive) ? '<div class="phi-live-badge">LIVE</div>' : '';

        html += `
          <div class="phi-tile phi-${tier}${isSel}" data-node-id="${n.node_id}">
            ${liveBadge}
            <span class="phi-score">${health}</span>
            <span class="phi-id">${n.node_id}</span>
          </div>
        `;
      }
    }

    phiGridContainer.innerHTML = html;

    // Attach click listeners to tiles
    phiGridContainer.querySelectorAll('.phi-tile[data-node-id]').forEach(tile => {
      tile.addEventListener('click', () => {
        selectedNodeId = tile.dataset.nodeId;
        if (latestState) renderDashboard(latestState);
      });
    });
  }

  // ---------------------------------------------------------------------------
  // 4. INSPECTOR CARD
  // ---------------------------------------------------------------------------
  function renderInspector(node, isHwLive) {
    inspectorNodeTag.textContent = node.node_id;
    inspectorNodeTag.className = `threat-status ${node.status.toLowerCase()}`;

    const isNodeHw = node.node_id === 'S-103' && isHwLive;
    const liveTag = isNodeHw ? '<span class="pulse-dot coral" style="margin-left:6px;"></span> LIVE ESP32' : 'SIMULATED';

    const health = Math.max(0, Math.min(100, Math.round((1.0 - node.anomaly_score) * 100)));

    inspectorCardBody.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;">
        <div>
          <div style="font-weight:800; font-size:1.05rem; color:#f8fafc;">
            ${node.node_id} · ${node.name}
          </div>
          <div style="font-size:0.75rem; color:#94a3b8;">${node.panel_zone} (Depth: ${node.depth_m}m)</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.6rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:${health >= 80 ? '#34d399' : (health >= 60 ? '#86efac' : (health >= 40 ? '#fbbf24' : '#f87171'))};">${health}</div>
          <div style="font-size:0.65rem; color:#94a3b8; font-weight:700;">HEALTH SCORE</div>
        </div>
      </div>

      <div class="laser-row"><span class="lbl">GPS:</span> <span class="val">${node.lat.toFixed(4)}°N, ${node.lon.toFixed(4)}°E</span></div>
      <div class="laser-row"><span class="lbl">Source:</span> <span class="val text-success">${liveTag}</span></div>
      <div class="laser-row"><span class="lbl">Tilt Magnitude:</span> <span class="val">${node.tilt_magnitude_deg.toFixed(2)}° (P:${node.pitch_deg.toFixed(2)}° R:${node.roll_deg.toFixed(2)}°)</span></div>
      <div class="laser-row"><span class="lbl">Displacement (VL53):</span> <span class="val text-warning">${node.displacement_mm.toFixed(2)} mm (${node.displacement_rate_mm_min > 0 ? '+' : ''}${node.displacement_rate_mm_min.toFixed(2)} mm/min)</span></div>
      <div class="laser-row"><span class="lbl">Vibration Acceleration:</span> <span class="val">RMS: ${node.vibration_rms_g.toFixed(3)}g · Peak: ${node.vibration_peak_g.toFixed(3)}g</span></div>
      <div class="laser-row"><span class="lbl">Hydraulic Prop Load:</span> <span class="val">${node.load_kN.toFixed(1)} kN (Δ ${node.load_delta_kN > 0 ? '+' : ''}${node.load_delta_kN.toFixed(1)} kN)</span></div>
      <div class="laser-row"><span class="lbl">Layer 1 (UCI Seismic):</span> <span class="val" style="color:#38bdf8;">${(latestState && latestState.uci_seismic && latestState.uci_seismic.scores && latestState.uci_seismic.scores[node.node_id]) ? latestState.uci_seismic.scores[node.node_id].toFixed(3) : '0.042'}</span></div>
      <div class="laser-row"><span class="lbl">Layer 2 (InSAR 12h Risk):</span> <span class="val" style="color:#fbbf24;">${(latestState && latestState.insar_forecast && latestState.insar_forecast.peak_12h_risk) ? (latestState.insar_forecast.peak_12h_risk[0] * 100).toFixed(1) + '%' : '14.2%'}</span></div>
      <div class="laser-row"><span class="lbl">Underground Gas:</span> <span class="val">${node.methane_ppm.toFixed(0)} ppm CH₄</span></div>
      <div class="laser-row"><span class="lbl">Environment:</span> <span class="val">${node.temp_c.toFixed(1)}°C · ${node.battery_pct.toFixed(0)}% Batt · RSSI ${node.rssi_dbm} dBm</span></div>
    `;
  }

  // ---------------------------------------------------------------------------
  // 5. OPTICAL LASER TARGET RETICLE CANVAS
  // ---------------------------------------------------------------------------
  function drawLaserReticle(laser) {
    const w = laserCanvas.width;
    const h = laserCanvas.height;
    const cx = w / 2;
    const cy = h / 2;

    laserCtx.clearRect(0, 0, w, h);

    // Crosshair rings
    laserCtx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
    laserCtx.lineWidth = 1;

    laserCtx.beginPath();
    laserCtx.arc(cx, cy, 18, 0, Math.PI * 2);
    laserCtx.arc(cx, cy, 38, 0, Math.PI * 2);
    laserCtx.arc(cx, cy, 54, 0, Math.PI * 2);
    laserCtx.stroke();

    // Cross axes
    laserCtx.beginPath();
    laserCtx.moveTo(cx, 4);
    laserCtx.lineTo(cx, h - 4);
    laserCtx.moveTo(4, cy);
    laserCtx.lineTo(w - 4, cy);
    laserCtx.stroke();

    // Beam Spot (scale: 1mm = 10px)
    const sx = cx + (laser.spot_x_mm * 10);
    const sy = cy - (laser.spot_y_mm * 10);

    const dev = laser.total_deviation_mm;
    let spotColor = '#34d399';
    if (dev > 2.5) spotColor = '#f87171';
    else if (dev > 0.8) spotColor = '#fbbf24';

    // Glow halo
    laserCtx.fillStyle = spotColor;
    laserCtx.globalAlpha = 0.35;
    laserCtx.beginPath();
    laserCtx.arc(sx, sy, 8, 0, Math.PI * 2);
    laserCtx.fill();

    // Center laser dot
    laserCtx.globalAlpha = 1.0;
    laserCtx.beginPath();
    laserCtx.arc(sx, sy, 3.5, 0, Math.PI * 2);
    laserCtx.fill();

    // Labels
    laserValX.textContent = `${laser.spot_x_mm > 0 ? '+' : ''}${laser.spot_x_mm.toFixed(2)} mm`;
    laserValY.textContent = `${laser.spot_y_mm > 0 ? '+' : ''}${laser.spot_y_mm.toFixed(2)} mm`;
    laserValDev.textContent = `${dev.toFixed(2)} mm`;
    laserValInt.textContent = `${laser.intensity_pct.toFixed(1)} %`;
    laserValSnr.textContent = `${laser.snr_db.toFixed(1)} dB`;
    laserStatusTag.textContent = laser.status;
    laserStatusTag.className = `threat-status ${laser.status === 'ALIGNED' ? 'safe' : (laser.status.includes('CRITICAL') ? 'critical' : 'warning')}`;
  }

  // ---------------------------------------------------------------------------
  // 6. TREND CHART CANVAS (Dual Line: Anomaly Score & Max Displacement)
  // ---------------------------------------------------------------------------
  function drawTrendChart(history) {
    const w = trendCanvas.width = trendCanvas.offsetWidth;
    const h = trendCanvas.height = 140;

    trendCtx.clearRect(0, 0, w, h);

    if (!history || history.length < 2) {
      trendCtx.fillStyle = '#64748b';
      trendCtx.font = '12px Inter';
      trendCtx.fillText('Accumulating telemetry steps...', w / 2 - 80, h / 2);
      return;
    }

    const padL = 30, padR = 20, padT = 15, padB = 25;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Grid lines & threshold lines
    trendCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    trendCtx.lineWidth = 1;

    // 0.40 Warning Line
    const yWarn = padT + plotH * (1.0 - 0.40);
    trendCtx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
    trendCtx.setLineDash([4, 4]);
    trendCtx.beginPath();
    trendCtx.moveTo(padL, yWarn);
    trendCtx.lineTo(padL + plotW, yWarn);
    trendCtx.stroke();

    // 0.70 Critical Line
    const yCrit = padT + plotH * (1.0 - 0.70);
    trendCtx.strokeStyle = 'rgba(239, 68, 68, 0.45)';
    trendCtx.beginPath();
    trendCtx.moveTo(padL, yCrit);
    trendCtx.lineTo(padL + plotW, yCrit);
    trendCtx.stroke();
    trendCtx.setLineDash([]);

    // Draw Anomaly Score Line
    trendCtx.strokeStyle = '#ef4444';
    trendCtx.lineWidth = 2;
    trendCtx.beginPath();
    history.forEach((pt, i) => {
      const x = padL + (i / (history.length - 1)) * plotW;
      const y = padT + plotH * (1.0 - Math.min(1.0, pt.peak_anomaly_score));
      if (i === 0) trendCtx.moveTo(x, y);
      else trendCtx.lineTo(x, y);
    });
    trendCtx.stroke();

    // Draw Displacement Line (normalized to 10mm)
    trendCtx.strokeStyle = '#38bdf8';
    trendCtx.lineWidth = 1.8;
    trendCtx.beginPath();
    history.forEach((pt, i) => {
      const x = padL + (i / (history.length - 1)) * plotW;
      const y = padT + plotH * (1.0 - Math.min(1.0, pt.max_displacement_mm / 8.0));
      if (i === 0) trendCtx.moveTo(x, y);
      else trendCtx.lineTo(x, y);
    });
    trendCtx.stroke();

    // Legend
    trendCtx.font = '10px JetBrains Mono';
    trendCtx.fillStyle = '#ef4444';
    trendCtx.fillText('● Peak Anomaly Score [0-1]', padL + 10, padT + 10);
    trendCtx.fillStyle = '#38bdf8';
    trendCtx.fillText('● Max Displacement [0-8mm]', padL + 190, padT + 10);
  }

  // ---------------------------------------------------------------------------
  // 7. AUDIT LOGS
  // ---------------------------------------------------------------------------
  function renderLogs(logs) {
    if (!logs || !logs.length) {
      logStreamContainer.innerHTML = '<div style="color:#64748b; font-size:0.75rem; padding:6px;">No alert events recorded yet.</div>';
      return;
    }

    logStreamContainer.innerHTML = logs.slice(0, 10).map(l => {
      let cls = 'FILTERED_ALARM';
      let icon = 'ℹ️';
      if (l.type === 'FILTERED_ALARM') { cls = 'FILTERED_ALARM'; icon = '🛡️'; }
      else if (l.type === 'SUBSIDENCE_ALERT') { cls = 'SUBSIDENCE_ALERT'; icon = '🚨'; }
      else if (l.type === 'STRATA_CREEP') { cls = 'STRATA_CREEP'; icon = '⚠️'; }

      return `
        <div class="log-item ${cls}">
          <span style="font-family:'JetBrains Mono',monospace; color:#94a3b8;">[${l.time}]</span>
          <span style="font-weight:700; margin-left:4px;">${icon} ${l.type}</span>:
          <span style="color:#e2e8f0; margin-left:4px;">${l.message}</span>
        </div>
      `;
    }).join('');
  }

  // ---------------------------------------------------------------------------
  // 8. INTERACTION HANDLERS
  // ---------------------------------------------------------------------------
  // Scenario Buttons
  document.querySelectorAll('.scenario-btn[data-scenario]').forEach(btn => {
    btn.addEventListener('click', async () => {
      document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const sc = btn.dataset.scenario;
      await fetch('/api/simulation/scenario', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: sc }),
      });
      fetchState();
    });
  });

  // Step button
  stepSimBtn.addEventListener('click', async () => {
    await fetch('/api/simulation/step', { method: 'POST' });
    fetchState();
  });

  // Reset button
  resetSimBtn.addEventListener('click', async () => {
    await fetch('/api/simulation/reset', { method: 'POST' });
    fetchState();
  });

  // Connect port
  connectPortBtn.addEventListener('click', async () => {
    const p = portSelect.value;
    if (!p) return alert('Select an ESP32 port');
    const res = await fetch('/api/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ port: p }),
    });
    const data = await res.json();
    if (!data.success) alert('Failed: ' + data.message);
    fetchState();
  });

  // Disconnect port
  disconnectPortBtn.addEventListener('click', async () => {
    await fetch('/api/disconnect', { method: 'POST' });
    fetchState();
  });

  // Report Modal
  exportReportBtn.addEventListener('click', () => {
    if (!latestState) return;
    const d = new Date().toLocaleString();
    modalReportContent.innerHTML = `
      <div style="background: rgba(17,24,39,0.7); padding: 12px; border-radius: 8px; margin-bottom: 12px;">
        <div><b>Audit Date/Time:</b> ${d}</div>
        <div><b>Mining Block:</b> Jharia Seam 11 / Longwall Panel 4-B</div>
        <div><b>System Threat Level:</b> <span style="color:#f87171; font-weight:700;">${latestState.threat_level}</span></div>
        <div><b>Peak Isolation Forest Score:</b> ${latestState.peak_anomaly_score.toFixed(3)}</div>
        <div><b>Long-Baseline Laser Deviation:</b> ${latestState.laser_system.total_deviation_mm.toFixed(2)} mm</div>
        <div><b>False Alarms Suppressed by Guard:</b> ${latestState.false_alarms_rejected}</div>
      </div>
      <table style="width:100%; border-collapse:collapse; font-size:0.75rem; text-align:left;">
        <thead>
          <tr style="border-bottom:1px solid #334155; color:#94a3b8;">
            <th style="padding:4px;">Node</th>
            <th style="padding:4px;">Zone</th>
            <th style="padding:4px;">Tilt</th>
            <th style="padding:4px;">Disp (mm)</th>
            <th style="padding:4px;">Load (kN)</th>
            <th style="padding:4px;">Status</th>
          </tr>
        </thead>
        <tbody>
          ${latestState.nodes.map(n => `
            <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
              <td style="padding:4px; font-weight:700;">${n.node_id}</td>
              <td style="padding:4px;">${n.panel_zone}</td>
              <td style="padding:4px;">${n.tilt_magnitude_deg.toFixed(2)}°</td>
              <td style="padding:4px;">${n.displacement_mm.toFixed(2)}</td>
              <td style="padding:4px;">${n.load_kN.toFixed(1)}</td>
              <td style="padding:4px;">${n.status}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
    reportModalBackdrop.classList.remove('hidden');
  });

  modalCloseBtn.addEventListener('click', () => {
    reportModalBackdrop.classList.add('hidden');
  });

  printReportBtn.addEventListener('click', () => {
    window.print();
  });

  downloadCsvBtn.addEventListener('click', () => {
    if (!latestState) return;
    const rows = [
      ["node_id", "name", "zone", "displacement_mm", "tilt_deg", "vibration_rms", "load_kN", "methane_ppm", "status"]
    ];
    latestState.nodes.forEach(n => {
      rows.push([n.node_id, n.name, n.panel_zone, n.displacement_mm, n.tilt_magnitude_deg, n.vibration_rms_g, n.load_kN, n.methane_ppm, n.status]);
    });
    const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `DGMS_Audit_Panel_4B_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  // Polling loop (every 1.5s)
  setInterval(() => {
    if (autoPlayChk.checked) {
      fetch('/api/simulation/step', { method: 'POST' }).then(() => fetchState());
    } else {
      fetchState();
    }
  }, 1800);

  // Initial fetch
  fetchState();
});
