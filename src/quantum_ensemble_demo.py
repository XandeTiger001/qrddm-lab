#!/usr/bin/env python3
"""
Quantum Attack States → AI + Physics Ensemble Demo
Generates difficult quantum attack states and feeds them into the ensemble
"""

from quantum_attack_generator import QuantumAttackGenerator, QuantumToClassicalConverter
from tier1_classic_defense import Tier1ClassicDefense, generate_synthetic_attacks
import numpy as np

def demo_quantum_ensemble_pipeline():
    """Demo: Quantum states → AI + Physics ensemble"""
    print("QUANTUM ATTACK STATES → AI + PHYSICS ENSEMBLE")
    print("=" * 60)
    
    # Initialize components
    quantum_gen = QuantumAttackGenerator()
    converter = QuantumToClassicalConverter()
    defense = Tier1ClassicDefense()
    
    # Train defense system
    training_data = generate_synthetic_attacks(50)
    defense.train_system(training_data)
    
    print("\n1. GENERATING DIFFICULT QUANTUM ATTACK STATES")
    print("-" * 50)
    
    # Generate quantum attack states
    quantum_states = quantum_gen.generate_difficult_attack_states(5)
    
    for i, state in enumerate(quantum_states):
        print(f"Quantum State {i+1}: Entropy={state.entropy:.3f}")
    
    print("\n2. FEEDING INTO AI + PHYSICS ENSEMBLE")
    print("-" * 50)
    
    results = []
    for i, quantum_state in enumerate(quantum_states):
        # Convert quantum state to classical attack data
        attack_data = converter.convert_quantum_to_attack_data(quantum_state)
        
        # Feed into AI + Physics ensemble
        result = defense.analyze_attack(attack_data)
        
        print(f"\nAttack {i+1} ({attack_data['type']}):")
        print(f"  Quantum Entropy: {quantum_state.entropy:.3f}")
        print(f"  AI Threat Prob: {result['threat_probability']:.3f}")
        print(f"  Physics Mass: {result['physics']['M']:.3f}")
        print(f"  Enhanced Threat: {result['enhanced_threat']:.3f}")
        print(f"  Decision: {result['decision']}")
        
        results.append({
            'quantum_entropy': quantum_state.entropy,
            'ai_threat': result['threat_probability'],
            'physics_mass': result['physics']['M'],
            'enhanced_threat': result['enhanced_threat'],
            'decision': result['decision']
        })
    
    print("\n3. ENSEMBLE ANALYSIS")
    print("-" * 50)
    
    # Analyze ensemble performance
    avg_entropy = np.mean([r['quantum_entropy'] for r in results])
    avg_ai_threat = np.mean([r['ai_threat'] for r in results])
    avg_physics_mass = np.mean([r['physics_mass'] for r in results])
    avg_enhanced = np.mean([r['enhanced_threat'] for r in results])
    
    blocked = sum(1 for r in results if r['decision'] == 'BLOCK')
    
    print(f"Average Quantum Entropy: {avg_entropy:.3f}")
    print(f"Average AI Threat: {avg_ai_threat:.3f}")
    print(f"Average Physics Mass: {avg_physics_mass:.3f}")
    print(f"Average Enhanced Threat: {avg_enhanced:.3f}")
    print(f"Blocked Attacks: {blocked}/{len(results)} ({blocked/len(results):.1%})")
    
    print(f"\nQuantum-enhanced ensemble successfully processed {len(results)} difficult attack states")

if __name__ == '__main__':
    demo_quantum_ensemble_pipeline()