document.addEventListener('DOMContentLoaded', () => {
  const portSelect = document.getElementById('port-select');
  const connectBtn = document.getElementById('connect-btn');
  const disconnectBtn = document.getElementById('disconnect-btn');
  const statusBadge = document.getElementById('conn-status');

  // DOM Elements for telemetry
  const accelEl = document.getElementById('val-accel');
  const gyroEl = document.getElementById('val-gyro');
  const tempEl = document.getElementById('val-temp');
  const distEl = document.getElementById('val-dist');
  const loadEl = document.getElementById('val-load');
  const rssiEl = document.getElementById('val-rssi');

  const mainScoreEl = document.getElementById('main-score');

  async function fetchTelemetry() {
    try {
      const res = await fetch('/api/telemetry');
      if (res.ok) {
        const data = await res.json();
        
        // Update Ports list if disconnected
        if (!data.connected && portSelect.options.length !== data.ports.length) {
          portSelect.innerHTML = data.ports.map(p => `<option value="${p}">${p}</option>`).join('');
        }

        // Update UI state
        if (data.connected) {
          statusBadge.textContent = `CONNECTED TO ${data.port}`;
          statusBadge.className = 'status-indicator status-connected';
          connectBtn.style.display = 'none';
          disconnectBtn.style.display = 'inline-block';
          portSelect.disabled = true;
        } else {
          statusBadge.textContent = 'DISCONNECTED';
          statusBadge.className = 'status-indicator status-disconnected';
          connectBtn.style.display = 'inline-block';
          disconnectBtn.style.display = 'none';
          portSelect.disabled = false;
        }

        // Update Telemetry
        const hw = data.hardware;
        if (hw) {
          accelEl.textContent = `X:${hw.accelX.toFixed(2)} Y:${hw.accelY.toFixed(2)} Z:${hw.accelZ.toFixed(2)}`;
          gyroEl.textContent = `X:${hw.gyroX.toFixed(2)} Y:${hw.gyroY.toFixed(2)} Z:${hw.gyroZ.toFixed(2)}`;
          tempEl.textContent = `${hw.tempC.toFixed(1)}°C`;
          distEl.textContent = `${hw.distanceMM} mm`;
          loadEl.textContent = `${hw.loadWeight.toFixed(1)} g`;
          rssiEl.textContent = `${hw.rssi} dBm`;

          // Fake a mine score based on displacement for demo
          const score = Math.max(0, 100 - (hw.distanceMM / 10));
          mainScoreEl.textContent = score.toFixed(0);
          if (score < 50) {
            mainScoreEl.className = 'card-value value-red';
          } else {
            mainScoreEl.className = 'card-value value-green';
          }
        }
      }
    } catch (e) {
      console.error("API error", e);
    }
  }

  // Connect Handler
  connectBtn.addEventListener('click', async () => {
    const port = portSelect.value;
    if (!port) return alert("Select a port");
    const res = await fetch('/api/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ port })
    });
    const result = await res.json();
    if (!result.success) alert("Failed to connect: " + result.message);
  });

  // Disconnect Handler
  disconnectBtn.addEventListener('click', async () => {
    await fetch('/api/disconnect', { method: 'POST' });
  });

  // Poll every 500ms
  setInterval(fetchTelemetry, 500);
});
