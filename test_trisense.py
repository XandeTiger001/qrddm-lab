#!/usr/bin/env python3
"""Quick test to verify TriSense implementation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.trisense_engine import TriSenseEngine

def quick_test():
    engine = TriSenseEngine()
    
    # Test all three states
    tests = [
        ("SQL injection", "admin' OR 1=1--", 1),
        ("Suspicious chars", "user';", -1), 
        ("Normal request", "username=john", 0)
    ]
    
    print("🧪 Quick TriSense Test")
    print("-" * 30)
    
    for name, payload, expected in tests:
        state, _ = engine.synthesize_assessment(payload, 5, len(payload))
        status = "✅" if state == expected else "❌"
        print(f"{status} {name}: {state} (expected {expected})")
    
    print("\n✨ TriSense Logic Engine is ready!")

if __name__ == "__main__":
    quick_test()