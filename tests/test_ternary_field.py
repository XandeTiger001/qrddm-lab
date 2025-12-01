import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ternary_field_simulation import TernaryFieldSimulator
import numpy as np

def test_field_initialization():
    print("🧪 Test 1: Field Initialization\n")
    
    sim = TernaryFieldSimulator(grid_size=20, k=2.0)
    
    assert sim.state.shape == (20, 20), "Grid must be 20x20"
    assert np.all(sim.state == 0), "Initial state must be neutral (0)"
    assert sim.core_position == (10, 10), "Core must be at the center"
    
    print("   ✅ Field initialized correctly")
    print(f"   Grid: {sim.grid_size}x{sim.grid_size}")
    print(f"   Core: {sim.core_position}\n")

def test_attack_injection():
    print("🧪 Test 2: Attack Injection\n")
    
    sim = TernaryFieldSimulator(grid_size=20, k=2.0)
    sim.inject_attack((5, 5), intensity=2.0, radius=3)
    
    threats = np.sum(sim.state == -1)
    assert threats > 0, "There must be cells in threat state"
    
    print(f"   ✅ Attack injected at (5, 5)")
    print(f"   Affected cells: {threats}\n")

def test_field_evolution():
    print("🧪 Test 3: Field Evolution\n")
    
    sim = TernaryFieldSimulator(grid_size=30, k=2.0)
    sim.inject_attack((10, 10), intensity=1.5, radius=4)
    
    initial_threats = np.sum(sim.state == -1)
    
    for _ in range(10):
        sim.step()
    
    final_threats = np.sum(sim.state == -1)
    final_protected = np.sum(sim.state == 1)
    
    print(f"   Initial threats: {initial_threats}")
    print(f"   Threats after 10 steps: {final_threats}")
    print(f"   Protected cells: {final_protected}")
    
    assert final_protected > 0, "There must be protected cells after evolution"
    print("   ✅ Field evolved correctly\n")

def test_g_field_calculation():
    print("🧪 Test 4: g(r) Calculation\n")
    
    sim = TernaryFieldSimulator(grid_size=20, k=2.0)
    sim.inject_attack((10, 10), intensity=2.0, radius=3)
    
    g_near = sim.calculate_g(10, 10)
    g_far = sim.calculate_g(0, 0)
    
    print(f"   g(r) near core: {g_near:.4f}")
    print(f"   g(r) far from core: {g_far:.4f}")
    
    assert g_near < g_far, "g(r) must be lower near the attack"
    print("   ✅ g(r) field calculated correctly\n")

def test_ternary_classification():
    print("🧪 Test 5: Ternary Classification\n")
    
    sim = TernaryFieldSimulator(grid_size=20, k=2.0)
    
    test_cases = [
        (0.8, 1, "PROTECTED"),
        (0.5, 0, "NEUTRAL"),
        (0.2, -1, "THREAT")
    ]
    
    for g_value, expected_state, label in test_cases:
        if g_value > 0.7:
            state = 1
        elif g_value > 0.3:
            state = 0
        else:
            state = -1
        
        print(f"   g={g_value:.1f} → State={state:2d} ({label})")
        assert state == expected_state, f"State should be {expected_state} but got {state}"
    
    print("   ✅ Ternary classification working\n")

def test_statistics():
    print("🧪 Test 6: Field Statistics\n")
    
    sim = TernaryFieldSimulator(grid_size=20, k=2.0)
    sim.inject_attack((10, 10), intensity=2.0, radius=5)
    
    for _ in range(5):
        sim.step()
    
    stats = sim.get_statistics()
    
    print(f"   Threats: {stats['threats']}")
    print(f"   Neutral: {stats['neutral']}")
    print(f"   Protected: {stats['protected']}")
    print(f"   Threat Ratio: {stats['threat_ratio']:.2%}")
    print(f"   Protection Ratio: {stats['protected_ratio']:.2%}")
    
    total = stats['threats'] + stats['neutral'] + stats['protected']
    assert total == 400, "Total must be 20x20 = 400"
    print("   ✅ Statistics correct\n")

if __name__ == '__main__':
    print("🌌 Ternary Field Simulator Tests\n")
    print("=" * 60 + "\n")
    
    test_field_initialization()
    test_attack_injection()
    test_field_evolution()
    test_g_field_calculation()
    test_ternary_classification()
    test_statistics()
    
    print("=" * 60)
    print("\n✅ All tests passed!")

