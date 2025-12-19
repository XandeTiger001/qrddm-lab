#!/usr/bin/env python3
"""
Quantum Mechanics Enhanced Cyber Defense Demo
Demonstrates extreme scenario simulation, perturbation generation, 
state space analysis, and adversarial stress testing
"""

from tier1_classic_defense import Tier1ClassicDefense, generate_synthetic_attacks
from quantum_defense import QuantumEnhancedDefense, QuantumThreatSimulator, QuantumPerturbationGenerator
import numpy as np
import time

def demo_extreme_scenarios():
    """Quantum simulator of extreme scenarios"""
    print("QUANTUM EXTREME SCENARIO SIMULATION")
    print("=" * 60)
    
    simulator = QuantumThreatSimulator()
    
    # Create extreme multi-vector attack superposition
    extreme_weights = {
        'sql_inj': 0.4,
        'xss': 0.3,
        'ddos': 0.5,
        'malware': 0.6,
        'apt': 0.8,
        'zero_day': 0.9,
        'quantum_hack': 0.7
    }
    
    quantum_state = simulator.create_superposition_attack(extreme_weights)
    print(f"Extreme Attack Superposition Created")
    print(f"   States: {len(quantum_state.basis_labels)}")
    print(f"   Dominant amplitudes: {np.abs(quantum_state.amplitudes)[:3]}")
    
    # Simulate measurement collapse
    for i in range(5):
        measured_attack = quantum_state.measure()
        print(f"   Measurement {i+1}: {measured_attack}")
    
    # Create entangled attack scenario
    state1 = simulator.create_superposition_attack({'ddos': 0.8, 'malware': 0.6})
    state2 = simulator.create_superposition_attack({'apt': 0.9, 'zero_day': 0.7})
    entangled = simulator.quantum_entangle_attacks(state1, state2)
    
    print(f"\nEntangled Attack Scenario")
    print(f"   Entangled states: {len(entangled.basis_labels)}")
    print(f"   Sample entanglement: {entangled.basis_labels[0]}")

def demo_perturbation_generation():
    """Quantum generator of perturbations"""
    print("\nQUANTUM PERTURBATION GENERATION")
    print("=" * 60)
    
    generator = QuantumPerturbationGenerator()
    
    # Base attack to mutate
    base_attack = {
        'type': 'sql_injection',
        'payload': "' OR 1=1--",
        'request_size': 1024,
        'frequency': 10,
        'target_layer': 3
    }
    
    print(f"Base Attack: {base_attack['type']}")
    print(f"   Payload: {base_attack['payload']}")
    
    # Generate quantum variants
    variants = generator.generate_attack_perturbations(base_attack, 5)
    
    print(f"\nQuantum Variants Generated:")
    for i, variant in enumerate(variants):
        print(f"   Variant {i+1}:")
        print(f"     Payload: {variant['payload']}")
        print(f"     Size: {variant['request_size']:.1f}")
        if variant.get('quantum_tunnel'):
            print(f"     Quantum tunnel: {variant['bypass_probability']:.2f}")
    
    # Generate coherent attack wave
    wave = generator.create_coherent_attack_wave(attack_frequency=5.0, duration=2.0)
    print(f"\nCoherent Attack Wave")
    print(f"   Wave length: {len(wave)} samples")
    print(f"   Peak amplitude: {np.max(np.abs(wave)):.3f}")
    print(f"   Phase coherence: {np.std(np.angle(wave)):.3f}")

def demo_state_space_analysis():
    """Quantum analyzer of state spaces"""
    print("\nQUANTUM STATE SPACE ANALYSIS")
    print("=" * 60)
    
    defense = Tier1ClassicDefense()
    quantum_defense = defense.enable_quantum_mode()
    
    # Analyze different attack types in quantum state space
    test_attacks = [
        {
            'type': 'normal_traffic',
            'threat_score': 0.1,
            'frequency': 0.2,
            'complexity': 0.1
        },
        {
            'type': 'sql_injection',
            'threat_score': 0.7,
            'frequency': 0.4,
            'complexity': 0.6
        },
        {
            'type': 'advanced_persistent_threat',
            'threat_score': 0.9,
            'frequency': 0.8,
            'complexity': 0.95
        }
    ]
    
    print("Quantum State Analysis Results:")
    for attack in test_attacks:
        analysis = quantum_defense.state_analyzer.analyze_attack_state_space(attack)
        
        print(f"\n{attack['type']}:")
        print(f"   Quantum Entropy: {analysis['quantum_entropy']:.3f}")
        print(f"   State Purity: {analysis['state_purity']:.3f}")
        print(f"   Coherence: {analysis['coherence_measure']:.3f}")
        print(f"   Entanglement: {analysis['entanglement_degree']:.3f}")
        print(f"   Fidelity to Normal: {analysis['quantum_fidelity']:.3f}")
        print(f"   Phase Variance: {analysis['phase_variance']:.3f}")

def demo_adversarial_stress_test():
    """Quantum adversarial stress test"""
    print("\nQUANTUM ADVERSARIAL STRESS TESTING")
    print("=" * 60)
    
    # Initialize defense system
    defense = Tier1ClassicDefense()
    
    # Train with some data
    training_data = generate_synthetic_attacks(50)
    defense.train_system(training_data)
    
    # Enable quantum enhancement
    quantum_defense = defense.enable_quantum_mode()
    
    # Run quantum stress tests
    print("Running Quantum Stress Tests...")
    stress_results = quantum_defense.adversarial_tester.quantum_stress_test(defense, 10)
    
    # Analyze results
    total_tests = len(stress_results)
    successful_defenses = sum(1 for r in stress_results if r.get('defense_success', False))
    quantum_bypasses = sum(1 for r in stress_results if r.get('quantum_bypass_attempted', False))
    
    print(f"\nStress Test Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful Defenses: {successful_defenses}")
    print(f"   Defense Success Rate: {successful_defenses/total_tests:.1%}")
    print(f"   Quantum Bypass Attempts: {quantum_bypasses}")
    
    # Show detailed results for failed defenses
    failed_tests = [r for r in stress_results if not r.get('defense_success', False)]
    if failed_tests:
        print(f"\nFailed Defense Analysis:")
        for i, test in enumerate(failed_tests[:3]):
            if 'quantum_properties' in test:
                qp = test['quantum_properties']
                print(f"   Test {test['test_id']}:")
                print(f"     Superposition: {qp.get('superposition_factor', 0):.3f}")
                print(f"     Entanglement: {qp.get('entanglement_strength', 0):.3f}")
                print(f"     Decoherence: {qp.get('decoherence_time', 0):.3f}")

def demo_quantum_threat_simulation():
    """Comprehensive quantum threat simulation"""
    print("\nQUANTUM THREAT SIMULATION")
    print("=" * 60)
    
    # Initialize full system
    defense = Tier1ClassicDefense()
    training_data = generate_synthetic_attacks(75)
    defense.train_system(training_data)
    
    quantum_defense = defense.enable_quantum_mode()
    
    # Simulate quantum-enhanced attack
    quantum_attack = {
        'type': 'quantum_multi_vector',
        'payload': 'quantum_superposed_payload',
        'source_ip': '203.0.113.666',
        'request_size': 2048,
        'response_time': 0.001,  # Quantum speed
        'frequency': 100,
        'target_layer': 7,
        'num_techniques': 8,
        'threat_score': 0.95,
        'complexity': 0.99,
        'quantum_tunnel': True,
        'bypass_probability': 0.4
    }
    
    print("Analyzing Quantum Threat...")
    result = quantum_defense.analyze_with_quantum_enhancement(quantum_attack)
    
    print(f"\nAnalysis Results:")
    print(f"   Base Threat Probability: {result['threat_probability']:.3f}")
    print(f"   Quantum Enhanced Threat: {result['quantum_enhanced_threat']:.3f}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Decision: {result['decision']}")
    print(f"   Quantum State Measured: {result['quantum_state_measured']}")
    
    qa = result['quantum_analysis']
    print(f"\nQuantum Properties:")
    print(f"   Entropy: {qa['quantum_entropy']:.3f}")
    print(f"   Purity: {qa['state_purity']:.3f}")
    print(f"   Coherence: {qa['coherence_measure']:.3f}")
    print(f"   Fidelity: {qa['quantum_fidelity']:.3f}")
    
    if result.get('quantum_bypass_detected'):
        print(f"   Quantum tunneling detected!")

def main():
    """Run complete quantum mechanics demo"""
    print("QUANTUM MECHANICS ENHANCED CYBER DEFENSE")
    print("Extreme Scenarios • Perturbations • State Analysis • Adversarial Testing")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run all quantum demos
    demo_extreme_scenarios()
    demo_perturbation_generation()
    demo_state_space_analysis()
    demo_adversarial_stress_test()
    demo_quantum_threat_simulation()
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Quantum Defense Demo Complete")
    print(f"Total execution time: {elapsed:.2f} seconds")
    print(f"Quantum mechanics successfully integrated into cyber defense")
    print(f"Ready for extreme scenario simulation and adversarial testing")

if __name__ == '__main__':
    main()