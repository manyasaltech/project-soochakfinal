import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading
import sys
from hardware import bridge, get_available_ports

class APIHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/telemetry':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            data = bridge.latest_data if bridge.latest_data else {
                "accelX": 0, "accelY": 0, "accelZ": 0,
                "gyroX": 0, "gyroY": 0, "gyroZ": 0,
                "tempC": 25.0, "distanceMM": 450, "loadWeight": 0.0,
                "rssi": -100
            }
            # We add fake connection for testing if no hardware is connected, 
            # but since user wants real ESP data, we expose the bridge state.
            response = {
                "hardware": data,
                "ports": get_available_ports(),
                "connected": bridge.running,
                "port": bridge.port
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/connect':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                req = json.loads(post_data.decode('utf-8'))
                port = req.get('port')
                if port:
                    success, msg = bridge.connect(port)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": success, "message": msg}).encode('utf-8'))
                else:
                    self.send_response(400)
                    self.end_headers()
            except:
                self.send_response(400)
                self.end_headers()
        elif self.path == '/api/disconnect':
            bridge.disconnect()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, APIHandler)
    print(f'Starting Project Soochak HTML API Server on http://localhost:{port}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        bridge.disconnect()
        sys.exit(0)

if __name__ == '__main__':
    run()
