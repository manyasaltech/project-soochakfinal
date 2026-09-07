import serial
import serial.tools.list_ports
import threading
import time
import re

def get_available_ports():
    return [port.device for port in serial.tools.list_ports.comports()]

class HardwareBridge:
    def __init__(self):
        self.port = None
        self.baudrate = 115200
        self.serial_conn = None
        self.thread = None
        self.running = False
        
        self.latest_data = None
    
    def connect(self, port):
        if self.running:
            self.disconnect()
        self.port = port
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            return True, "Connected successfully"
        except Exception as e:
            return False, str(e)
        
    def disconnect(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            
    def _read_loop(self):
        while self.running:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("[RX]"):
                        # [RX] accelX,accelY,accelZ,gyroX,gyroY,gyroZ,tempC,distanceMM,loadWeight | RSSI: -65 dBm | SNR: 9.5 dB
                        parts = line.split("|")
                        if len(parts) >= 1:
                            csv_part = parts[0].replace("[RX]", "").strip()
                            vals = csv_part.split(",")
                            if len(vals) == 9:
                                data = {
                                    "accelX": float(vals[0]),
                                    "accelY": float(vals[1]),
                                    "accelZ": float(vals[2]),
                                    "gyroX": float(vals[3]),
                                    "gyroY": float(vals[4]),
                                    "gyroZ": float(vals[5]),
                                    "tempC": float(vals[6]),
                                    "distanceMM": float(vals[7]),
                                    "loadWeight": float(vals[8]),
                                    "rssi": -65
                                }
                                # parse RSSI
                                if len(parts) >= 2 and "RSSI:" in parts[1]:
                                    rssi_match = re.search(r"RSSI:\s*(-?\d+)", parts[1])
                                    if rssi_match:
                                        data["rssi"] = int(rssi_match.group(1))
                                self.latest_data = data
            except Exception as e:
                time.sleep(0.1)

# Global instance
bridge = HardwareBridge()
