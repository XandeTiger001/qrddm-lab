from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambdas'))
from threat_detector import lambda_handler as detector_handler
from redirector import lambda_handler as redirector_handler

class EventHorizonHandler(BaseHTTPRequestHandler):
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
            
            result = detector_handler(event, None)
            response_body = json.loads(result['body'])
            
            if response_body['threat_detected']:
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
    server = HTTPServer(('localhost', port), EventHorizonHandler)
    print(f"🛡️  Cyber Event Horizon running on http://localhost:{port}")
    print("Send POST requests to test threat detection\n")
    server.serve_forever()

if __name__ == '__main__':
    run_server()
