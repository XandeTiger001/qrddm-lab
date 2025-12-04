import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_mass import EnhancedSchwarzschildDefense, DefenseMode

def test_mass_components():
    print("🧪 Testing M Components(r)\n")
    
    defense = EnhancedSchwarzschildDefense(mode=DefenseMode.BALANCED)
    
    tests = [
        {
            'name': 'Simple Attack',
            'data': {
                'type': 'xss',
                'threat_score': 0.5,
                'target_layer': 3,
                'num_techniques': 1,
                'mutations': 0,
                'adversarial_noise': 0,
                'distance_to_core': 5.0,
                'affected_modules': 1,
                'total_modules': 10,
                'source_ip': '1.1.1.1'
            },
            'expected_M': 'Low'
        },
        {
            'name': 'Complex Near Attack',
            'data': {
                'type': 'adversarial_ml',
                'threat_score': 0.9,
                'target_layer': 0,
                'num_techniques': 5,
                'mutations': 10,
                'adversarial_noise': 8,
                'distance_to_core': 0.3,
                'affected_modules': 7,
                'total_modules': 10,
                'source_ip': '2.2.2.2'
            },
            'expected_M': 'alto'
        }
    ]
    
    for test in tests:
        result = defense.analyze_attack(test['data'])
        
        print(f"📡 {test['name']}")
        print(f"   M(r) = {result['M_total']:.4f} (esperado: {test['expected_M']})")
        print(f"   Φ(r) = {result['phi']:.4f}")
        print(f"   Classification: {result['classification']}")
        print(f"   Energia de Defesa: {result['field_warping']}\n")
        
        if test['expected_M'] == 'alto':
            assert result['M_total'] > 2.0, "M should be High"
        else:
            assert result['M_total'] < 2.0, "M should be Low"

def test_mode_adaptation():
    print("\n🧪 Testing mode adaptation\n")
    
    attack = {
        'type': 'sql_injection',
        'threat_score': 0.7,
        'target_layer': 2,
        'num_techniques': 2,
        'mutations': 1,
        'adversarial_noise': 0,
        'distance_to_core': 2.0,
        'affected_modules': 2,
        'total_modules': 10,
        'source_ip': '3.3.3.3'
    }
    
    modes = [DefenseMode.ECONOMY, DefenseMode.BALANCED, DefenseMode.PARANOID]
    
    for mode in modes:
        defense = EnhancedSchwarzschildDefense(mode=mode)
        result = defense.analyze_attack(attack)
        
        print(f"Modo {mode.value.upper()}:")
        print(f"   M(r) = {result['M_total']:.4f}")
        print(f"   Classification: {result['classification']}\n")

def test_frequency_effect():
    print("\n🧪 Testing frequency effect\n")
    
    defense = EnhancedSchwarzschildDefense()
    
    attack = {
        'type': 'ddos',
        'threat_score': 0.8,
        'target_layer': 1,
        'num_techniques': 1,
        'mutations': 0,
        'adversarial_noise': 0,
        'distance_to_core': 1.0,
        'affected_modules': 5,
        'total_modules': 10,
        'source_ip': '4.4.4.4'
    }
    
    print("Repeated attacks from the same IP address:")
    for i in range(5):
        result = defense.analyze_attack(attack)
        F = result['mass_components']['F_frequency']
        M = result['M_total']
        print(f"   Attack {i+1}: F={F:.4f}, M(r)={M:.4f}")

if __name__ == '__main__':
    test_mass_components()
    test_mode_adaptation()
    test_frequency_effect()
    print("\n✅ All tests passed!")
