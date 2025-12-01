#!/usr/bin/env python3
"""
TriSense Logic Engine Demonstration
Real-world cybersecurity threat assessment using ternary logic
"""

from trisense_engine import TriSenseEngine, enhance_ml_prediction

def demo_trisense_states():
    """Demonstrate all three TriSense states with realistic examples"""
    engine = TriSenseEngine()
    
    test_cases = [
        # Evidence (+1) cases - confirmed threats
        {
            'name': 'SQL Injection Attack',
            'payload': "admin' OR 1=1--",
            'rate': 5,
            'size': 15,
            'expected_state': 1
        },
        {
            'name': 'XSS Attack',
            'payload': '<script>alert("xss")</script>',
            'rate': 3,
            'size': 28,
            'expected_state': 1
        },
        {
            'name': 'Brute Force Pattern',
            'payload': 'login',
            'rate': 25,
            'size': 5,
            'expected_state': 1
        },
        
        # Shadow (-1) cases - suspicious but uncertain
        {
            'name': 'Suspicious Characters',
            'payload': "user'; --comment",
            'rate': 8,
            'size': 15,
            'expected_state': -1
        },
        {
            'name': 'Elevated Request Rate',
            'payload': 'normal request',
            'rate': 18,
            'size': 14,
            'expected_state': -1
        },
        {
            'name': 'Large Payload',
            'payload': 'A' * 150,
            'rate': 2,
            'size': 150,
            'expected_state': -1
        },
        
        # Normality (0) cases - benign traffic
        {
            'name': 'Normal Login',
            'payload': 'username=john&password=secret',
            'rate': 2,
            'size': 30,
            'expected_state': 0
        },
        {
            'name': 'API Request',
            'payload': '{"user_id": 123, "action": "get_profile"}',
            'rate': 5,
            'size': 42,
            'expected_state': 0
        }
    ]
    
    print("🔍 TriSense Logic Engine Demonstration")
    print("=" * 60)
    
    for case in test_cases:
        state, details = engine.synthesize_assessment(
            case['payload'], case['rate'], case['size']
        )
        
        state_symbols = {-1: "🌫️ ", 0: "✅", 1: "🚨"}
        
        print(f"\n{state_symbols[state]} {case['name']}")
        print(f"   Payload: {case['payload'][:50]}{'...' if len(case['payload']) > 50 else ''}")
        print(f"   Rate: {case['rate']}/min, Size: {case['size']} bytes")
        print(f"   TriSense State: {state} ({details['state_meaning']})")
        print(f"   Confidence: {details['confidence']:.2f}")
        print(f"   Recommendation: {details['recommendation']}")
        print(f"   Component Analysis: {details['assessments']}")

def demo_ml_fusion():
    """Demonstrate ML + TriSense fusion for enhanced threat detection"""
    print("\n\n🤖 ML + TriSense Fusion Demonstration")
    print("=" * 60)
    
    fusion_cases = [
        {
            'name': 'High ML + Evidence TriSense',
            'payload': "' UNION SELECT password FROM users--",
            'rate': 10,
            'size': 38,
            'ml_score': 0.85
        },
        {
            'name': 'Low ML + Shadow TriSense',
            'payload': 'user"; --',
            'rate': 12,
            'size': 8,
            'ml_score': 0.35
        },
        {
            'name': 'Medium ML + Normal TriSense',
            'payload': 'legitimate user query',
            'rate': 3,
            'size': 20,
            'ml_score': 0.45
        }
    ]
    
    for case in fusion_cases:
        result = enhance_ml_prediction(
            case['payload'], case['rate'], case['size'], case['ml_score']
        )
        
        threat_icon = "🚨" if result['threat_detected'] else "✅"
        
        print(f"\n{threat_icon} {case['name']}")
        print(f"   Original ML Score: {case['ml_score']:.3f}")
        print(f"   TriSense State: {result['trisense_state']}")
        print(f"   Final Threat Score: {result['threat_score']:.3f}")
        print(f"   Threat Detected: {result['threat_detected']}")
        print(f"   Fusion Logic: {result['fusion_logic']}")

def demo_state_transitions():
    """Show how TriSense states transition based on evidence accumulation"""
    print("\n\n🔄 TriSense State Transition Examples")
    print("=" * 60)
    
    engine = TriSenseEngine()
    
    # Progressive attack simulation
    payloads = [
        ("Initial probe", "admin"),
        ("Add suspicious char", "admin'"),
        ("SQL injection attempt", "admin' OR 1=1"),
        ("Full attack", "admin' OR 1=1; DROP TABLE users--")
    ]
    
    for stage, payload in payloads:
        state, details = engine.synthesize_assessment(payload, 5, len(payload))
        state_names = {-1: "Shadow", 0: "Normal", 1: "Evidence"}
        
        print(f"\n{stage}: '{payload}'")
        print(f"   State: {state} ({state_names[state]})")
        print(f"   Confidence: {details['confidence']:.2f}")

if __name__ == "__main__":
    demo_trisense_states()
    demo_ml_fusion()
    demo_state_transitions()
    
    print("\n\n💡 TriSense Logic Engine Summary")
    print("=" * 60)
    print("🌫️  Shadow (-1): Incomplete data, suspicious patterns, requires investigation")
    print("✅ Normal (0):  Baseline behavior, no threat indicators")
    print("🚨 Evidence (+1): Confirmed threats, high confidence, immediate action")
    print("\nInspired by quantum superposition - threats exist in uncertain states")
    print("until sufficient evidence collapses them into definitive classifications.")