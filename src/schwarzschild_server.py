from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambdas'))
from schwarzschild_detector import lambda_handler as schwarzschild_handler
from redirector import lambda_handler as redirector_handler

class SchwarzschildHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else ''
            
            event = {
                'body': body,
                'headers': dict(self.headers),
                'requestContext': {
                    'requestRate': 10,
                    'identity': {'sourceIp': self.client_address[0]}
                }
            }
            
            result = schwarzschild_handler(event, None)
            response_body = json.loads(result['body'])
            
            # Log metric Schwarzschild
            if response_body.get('schwarzschild'):
                s = response_body['schwarzschild']
                print(f"📊 Φ={s['phi']:.3f} | M={s['M']:.3f} | r={s['r']:.3f} | C={s['curvature']:.3f}")
            
            if response_body['threat_detected']:
                action = response_body.get('action', 'REDIRECT')
                print(f"🚨 {action}: {response_body['classification']['level']}")
                
                if action in ['REDIRECT_HONEYPOT', 'BLOCK_IMMEDIATE']:
                    redirect_event = {'body': result['body'], 'requestContext': event['requestContext']}
                    result = redirector_handler(redirect_event, None)
            
            self.send_response(result['statusCode'])
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(result['body'].encode())
        except Exception as e:
            print(f"Error: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

def run_server(port=8000):
    server = HTTPServer(('localhost', port), SchwarzschildHandler)
    print(f"🌌 Schwarzschild Defense System running on http://localhost:{port}")
    print("Using curvature metrics for threat detecion\n")
    print("Φ(r) = 1 - 2DM(r)/(c²r)")
    print("Classification: Φ ≥ 0.5 (SAFE) | 0.2 < Φ < 0.5 (MONITOR) | Φ ≤ 0.2 (CRITICAL)\n")
    server.serve_forever()

if __name__ == '__main__':
    run_server()
