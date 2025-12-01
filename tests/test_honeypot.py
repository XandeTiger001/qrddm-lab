import requests
import time

HONEYPOT_URL = 'http://localhost:9000'

attacks = [
    ('GET', '/admin', None),
    ('POST', '/login', {'username': 'admin', 'password': "' OR '1'='1"}),
    ('GET', '/api/users', None),
    ('POST', '/config', {'cmd': 'cat /etc/passwd'}),
    ('GET', '/../../../etc/passwd', None),
]

def test_honeypot():
    print("🧪 Testing Honeypot Endpoints\n")
    
    for method, path, payload in attacks:
        print(f"Attacking: {method} {path}")
        try:
            if method == 'GET':
                response = requests.get(f"{HONEYPOT_URL}{path}", timeout=5)
            else:
                response = requests.post(f"{HONEYPOT_URL}{path}", json=payload, timeout=5)
            
            print(f"  Response: {response.status_code}")
            print(f"  Data: {response.json()}\n")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error: {e}\n")

if __name__ == '__main__':
    test_honeypot()
