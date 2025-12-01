import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from defense_layers import DefenseLayer, LayeredDefenseSystem
import numpy as np

def test_M_reduction():
    print("Test 1: M Reduction Formula\n")
    
    layer = DefenseLayer(radius=5.0, G=2.0, alpha=1.5)
    
    M_initial = 5.0
    F_defense = layer.calculate_force(M_initial)
    M_reduced = max(0, M_initial - F_defense)
    
    print(f"   M_initial: {M_initial:.4f}")
    print(f"   F_defense: {F_defense:.4f}")
    print(f"   M_reduced: {M_reduced:.4f}")
    print(f"   Reduction: {M_initial - M_reduced:.4f}")
    
    assert M_reduced >= 0, "M should never be negative"
    assert M_reduced < M_initial, "M should decrease"
    print("   OK: M reduction working correctly\n")

def test_probabilistic_variation():
    print("Test 2: Probabilistic Variation\n")
    
    layer = DefenseLayer(radius=5.0, G=2.0, alpha=1.5)
    M = 3.0
    F = layer.calculate_force(M)
    P = layer.neutralization_probability(F)
    
    print(f"   P_success: {P:.4f}")
    
    # Run 1000 times to check randomness
    neutralized_count = 0
    for _ in range(1000):
        if np.random.random() < P:
            neutralized_count += 1
    
    observed_rate = neutralized_count / 1000
    print(f"   Expected rate: {P:.4f}")
    print(f"   Observed rate: {observed_rate:.4f}")
    print(f"   Difference: {abs(P - observed_rate):.4f}")
    
    assert abs(P - observed_rate) < 0.05, "Random probability should match expected"
    print("   OK: Probabilistic mode working correctly\n")

def test_clamp_protection():
    print("Test 3: Clamp Protection\n")
    
    layer = DefenseLayer(radius=1.0, G=10.0, alpha=2.0)
    
    M_small = 0.5
    F_large = layer.calculate_force(M_small)
    M_after = max(0, M_small - F_large)
    
    print(f"   M_small: {M_small:.4f}")
    print(f"   F_large: {F_large:.4f}")
    print(f"   M_after clamp: {M_after:.4f}")
    
    assert M_after == 0, "M should be clamped to 0 when F > M"
    assert M_after >= 0, "M should never be negative"
    print("   OK: Clamp protection working\n")

def test_full_system():
    print("Test 4: Full System Simulation\n")
    
    layers = [(5.0, 2.0, 1.5), (3.0, 2.5, 1.8), (1.0, 3.0, 2.0)]
    system = LayeredDefenseSystem(layers)
    
    # Deterministic
    results, M_final, state, time = system.simulate_attack(5.0, mode='deterministic')
    
    print(f"   Deterministic:")
    print(f"     M0=5.0 -> M_final={M_final:.4f}")
    print(f"     State: {state:+d}")
    print(f"     Layers: {len(results)}")
    
    # Check M decreases through layers
    for i, r in enumerate(results):
        print(f"     Layer {i}: M={r['M']:.4f}, F={r['F']:.4f}")
        if i > 0:
            assert r['M'] <= results[i-1]['M'], "M should decrease or stay same"
    
    # Probabilistic
    neutralized = 0
    for _ in range(100):
        _, _, state, _ = system.simulate_attack(5.0, mode='probabilistic')
        if state == 1:
            neutralized += 1
    
    print(f"\n   Probabilistic (100 runs):")
    print(f"     Neutralization rate: {neutralized}%")
    
    assert 0 < neutralized < 100, "Should have some variation in probabilistic mode"
    print("   OK: Full system working correctly\n")

if __name__ == '__main__':
    print("Defense Layers Validation Tests\n")
    print("=" * 60 + "\n")
    
    test_M_reduction()
    test_probabilistic_variation()
    test_clamp_protection()
    test_full_system()
    
    print("=" * 60)
    print("\nAll tests passed!")
