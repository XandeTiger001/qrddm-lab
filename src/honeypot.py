from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
import os

class HoneypotHandler(BaseHTTPRequestHandler):
    log_file = 'data/honeypot_logs.json'
    
    def log_attack(self, method, path, body, headers):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'method': method,
            'path': path,
            'body': body,
            'headers': dict(headers),
            'source_ip': self.client_address[0],
            'user_agent': headers.get('User-Agent', 'Unknown')
        }
        
        logs = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                try:
                    logs = json.load(f)
                except:
                    logs = []
        
        logs.append(log_entry)
        
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"🍯 Honeypot captured: {method} {path} from {self.client_address[0]}")
    
    def handle_request(self, method):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else ''
        
        self.log_attack(method, self.path, body, self.headers)
        
        # Fake responses to keep attackers engaged
        fake_responses = {
            '/admin': {'status': 'Admin panel', 'version': '1.0'},
            '/login': {'message': 'Login successful', 'token': 'fake_token_12345'},
            '/api/users': {'users': [{'id': 1, 'name': 'admin'}]},
            '/config': {'database': 'localhost', 'debug': True}
        }
        
        response = fake_responses.get(self.path, {'message': 'Resource found'})
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def do_GET(self):
        self.handle_request('GET')
    
    def do_POST(self):
        self.handle_request('POST')
    
    def log_message(self, format, *args):
        pass

def run_honeypot(port=9000):
    server = HTTPServer(('localhost', port), HoneypotHandler)
    print(f"🍯 Honeypot running on http://localhost:{port}")
    print("Endpoints: /admin, /login, /api/users, /config\n")
    server.serve_forever()

if __name__ == '__main__':
    run_honeypot()
