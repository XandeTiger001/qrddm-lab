#!/usr/bin/env python3
"""
TriSense Logic Engine Demonstration

Shows how ternary logic enhances cybersecurity threat detection
beyond traditional binary classification.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trisense_engine import TriSenseEngine, enhance_ml_prediction
import json

def demo_trisense():
    print("🔮 TriSense Logic Engine - Quantum-Inspired Cybersecurity")
    print("=" * 60)
    print("States: -1=Shadow | 0=Normality | +1=Evidence")
    print("=" * 60)
    
    engine = TriSenseEngine()
    
    # Test cases from your dataset
    test_cases = [
        {
            "name": "Normal Login",
            "payload": '{"user":"alice","action":"login"}',
            "rate": 5,
            "size": 32,
            "ml_score": 0.1
        },
        {
            "name": "Suspicious Query",
            "payload": '{"query":"SELECT * FROM users"}',
            "rate": 8,
            "size": 25,
            "ml_score": 0.4
        },
        {
            "name": "SQL Injection",
            "payload": '{"query":"SELECT * FROM users WHERE id=1 OR 1=1"}',
            "rate": 15,
            "size": 45,
            "ml_score": 0.8
        },
        {
            "name": "XSS Attack",
            "payload": '{"comment":"<script>alert(\'XSS\')</script>"}',
            "rate": 12,
            "size": 38,
            "ml_score": 0.7
        },
        {
            "name": "Path Traversal",
            "payload": '{"file":"../../../etc/passwd"}',
            "rate": 18,
            "size": 28,
            "ml_score": 0.6
        },
        {
            "name": "Padding Attack",
            "payload": '{"data":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}',
            "rate": 25,
            "size": 52,
            "ml_score": 0.9
        },
        {
            "name": "Ambiguous Case",
            "payload": '{"search":"admin\' --"}',
            "rate": 7,
            "size": 20,
            "ml_score": 0.5
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Payload: {case['payload'][:50]}{'...' if len(case['payload']) > 50 else ''}")
        
        # Get TriSense assessment
        state, details = engine.synthesize_assessment(
            case['payload'], case['rate'], case['size']
        )
        
        # Get enhanced ML prediction
        enhanced = enhance_ml_prediction(
            case['payload'], case['rate'], case['size'], case['ml_score']
        )
        
        # Display results
        state_symbols = {-1: "🌫️ ", 0: "✅", 1: "🚨"}
        print(f"   TriSense State: {state_symbols[state]} {state} - {details['state_meaning']}")
        print(f"   Confidence: {details['confidence']:.2f}")
        print(f"   ML Score: {case['ml_score']:.2f} → Enhanced: {enhanced['threat_score']:.2f}")
        print(f"   Action: {details['recommendation']}")
        
        # Show individual assessments
        assessments = details['assessments']
        print(f"   Analysis: Entropy={assessments['entropy']}, Injection={assessments['injection']}, "
              f"Velocity={assessments['velocity']}, Traversal={assessments['traversal']}")

def demo_state_transitions():
    print("\n\n🔄 TriSense State Transition Rules")
    print("=" * 40)
    print("Evidence ≥ 2 → +1 (Evidence)")
    print("Evidence = 1, Shadow = 0 → +1 (Evidence)")
    print("Shadow ≥ 2 → -1 (Shadow)")
    print("Evidence = 1, Shadow ≥ 1 → -1 (Shadow)")
    print("Otherwise → 0 (Normality)")
    
    print("\n🎯 Real-world Applications:")
    print("• Adaptive honeypot routing")
    print("• Dynamic rate limiting")
    print("• Threat intelligence scoring")
    print("• Incident response prioritization")

def demo_quantum_inspiration():
    print("\n\n⚛️  Quantum Logic Inspiration")
    print("=" * 35)
    print("Traditional Binary: Threat | No Threat")
    print("TriSense Ternary: Evidence | Shadow | Normality")
    print()
    print("Like quantum superposition, threats exist in")
    print("uncertain states until evidence collapses them")
    print("into definitive classifications.")
    print()
    print("Shadow state captures the 'maybe' that binary")
    print("logic cannot express - crucial for cybersecurity")
    print("where false positives and negatives have real costs.")

if __name__ == "__main__":
    demo_trisense()
    demo_state_transitions()
    demo_quantum_inspiration()