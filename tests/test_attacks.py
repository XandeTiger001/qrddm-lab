import requests
import json

BASE_URL = 'http://localhost:8000'

test_cases = [
    {
        'name': 'Normal Request',
        'payload': {'user': 'alice', 'action': 'login'}
    },
    {
        'name': 'SQL Injection',
        'payload': {'query': "SELECT * FROM users WHERE id=1 OR 1=1; DROP TABLE users;"}
    },
    {
        'name': 'XSS Attack',
        'payload': {'comment': "<script>alert('XSS')</script>"}
    },
    {
        'name': 'Path Traversal',
        'payload': {'file': '../../../etc/passwd'}
    },
    {
        'name': 'Large Payload',
        'payload': {'data': 'A' * 50000}
    }
]

def run_tests():
    print("🧪 Testing Cyber Event Horizon\n")
    
    for test in test_cases:
        print(f"Testing: {test['name']}")
        try:
            response = requests.post(BASE_URL, json=test['payload'], timeout=5)
            result = response.json()
            
            status = "🚨 BLOCKED" if result.get('threat_detected') else "✅ ALLOWED"
            score = result.get('threat_score', 0)
            
            print(f"  {status} | Threat Score: {score}")
            print(f"  Action: {result.get('action', 'N/A')}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")

if __name__ == '__main__':
    run_tests()
