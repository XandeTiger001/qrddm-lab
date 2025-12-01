import requests
import json

BASE_URL = 'http://localhost:8000'

test_cases = [
    {
        'name': 'Tráfego Normal',
        'payload': {'user': 'alice', 'action': 'login'},
        'expected': 'SAFE'
    },
    {
        'name': 'SQL Injection Moderado',
        'payload': {'query': "SELECT * FROM users WHERE id=1 OR 1=1"},
        'expected': 'MONITOR/CRITICAL'
    },
    {
        'name': 'XSS Attack',
        'payload': {'comment': "<script>alert('XSS')</script>"},
        'expected': 'MONITOR/CRITICAL'
    },
    {
        'name': 'Path Traversal',
        'payload': {'file': '../../../etc/passwd'},
        'expected': 'MONITOR/CRITICAL'
    },
    {
        'name': 'DDoS Simulado',
        'payload': {'data': 'A' * 50000},
        'expected': 'CRITICAL'
    }
]

def run_tests():
    print("🌌 Testando Sistema de Defesa Schwarzschild\n")
    print("=" * 80)
    
    for test in test_cases:
        print(f"\n📡 {test['name']}")
        try:
            response = requests.post(BASE_URL, json=test['payload'], timeout=5)
            result = response.json()
            
            # Métricas Schwarzschild
            if 'schwarzschild' in result:
                s = result['schwarzschild']
                print(f"   Φ(r) = {s['phi']:.4f}")
                print(f"   M(r) = {s['M']:.4f}")
                print(f"   r = {s['r']:.4f}")
                print(f"   r_s = {s['r_schwarzschild']:.4f}")
                print(f"   Curvatura = {s['curvature']:.4f}")
            
            # Classificação
            if 'classification' in result:
                c = result['classification']
                level = c['level']
                inside = c.get('inside_horizon', False)
                
                status = "🚨 CRÍTICO" if level == "CRITICAL" else "⚠️  MONITOR" if level == "MONITOR" else "✅ SEGURO"
                print(f"   {status} | Nível: {level}")
                
                if inside:
                    print(f"   ⚠️  DENTRO DO HORIZONTE DE EVENTOS!")
                
                print(f"   Ação: {result.get('action', 'N/A')}")
                print(f"   Esperado: {test['expected']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    run_tests()
