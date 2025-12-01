import numpy as np
from defense_layers import DefenseLayer, LayeredDefenseSystem
from quantum_defense import QuantumDefenseModule

class QuantumLayeredDefense(LayeredDefenseSystem):
    """Hybrid AI + Quantum defense system"""
    
    def __init__(self, layers_config):
        super().__init__(layers_config)
        self.quantum = QuantumDefenseModule()
    
    def simulate_attack_quantum(self, M0, mode='quantum'):
        """Simulate attack with quantum-enhanced defense"""
        M = M0
        results = []
        total_processing_time = 0
        
        # Quantum mode selection
        q_mode, _ = self.quantum.quantum_threat_mode()
        
        for i, layer in enumerate(self.layers):
            # Quantum alpha replaces fixed alpha
            layer.alpha = self.quantum.generate_quantum_alpha()
            
            F_defense = layer.calculate_force(M)
            P_success = layer.neutralization_probability(F_defense)
            phi = layer.calculate_phi(M)
            dt_prime = layer.calculate_time_dilation(M)
            
            total_processing_time += dt_prime
            
            # Quantum entropy for randomness
            q_entropy = self.quantum.generate_quantum_entropy()
            G_boost = 1.0 + (dt_prime - 1.0) * 0.1 * (1 + q_entropy)
            F_defense_boosted = F_defense * G_boost
            
            if mode == 'quantum':
                # Use quantum randomness instead of classical
                if q_entropy < P_success:
                    results.append({
                        'layer': i, 'M': M, 'F': F_defense_boosted, 'P': P_success,
                        'phi': phi, 'dt_prime': dt_prime, 'alpha': layer.alpha,
                        'q_entropy': q_entropy, 'q_mode': q_mode, 'neutralized': True
                    })
                    return results, 0.0, 1, total_processing_time, q_mode
            
            M_new = max(0, M - F_defense_boosted)
            results.append({
                'layer': i, 'M': M, 'F': F_defense_boosted, 'P': P_success,
                'phi': phi, 'dt_prime': dt_prime, 'alpha': layer.alpha,
                'q_entropy': q_entropy, 'q_mode': q_mode, 'neutralized': False
            })
            M = M_new
            
            if M <= 0.01:
                return results, M, 1, total_processing_time, q_mode
        
        if M < 0.3:
            state = 1
        elif M < 0.7:
            state = 0
        else:
            state = -1
        
        return results, M, state, total_processing_time, q_mode

def demo_quantum_hybrid():
    print("Quantum-Enhanced Layered Defense\n")
    print("=" * 70)
    
    layers = [
        (10.0, 2.0, 1.5),
        (7.5, 2.5, 1.8),
        (5.0, 3.0, 2.0),
        (2.5, 3.5, 2.2),
        (1.0, 4.0, 2.5)
    ]
    
    system = QuantumLayeredDefense(layers)
    
    attacks = [('Weak', 1.0), ('Medium', 3.0), ('Strong', 5.0), ('Critical', 10.0)]
    
    print("\nQuantum Mode (100 runs per attack):\n")
    
    for name, M0 in attacks:
        neutralized = 0
        final_Ms = []
        modes_used = {'PARANOID': 0, 'BALANCED': 0, 'ADAPTIVE': 0, 'ECONOMY': 0}
        
        for _ in range(100):
            results, M_final, state, time, q_mode = system.simulate_attack_quantum(M0, mode='quantum')
            if state == 1:
                neutralized += 1
            final_Ms.append(M_final)
            modes_used[q_mode] += 1
        
        print(f"{name} Attack (M0={M0:.1f}):")
        print(f"  Neutralization: {neutralized}%")
        print(f"  Avg final M: {np.mean(final_Ms):.4f}")
        print(f"  Quantum modes: {modes_used}")
        print()
    
    print("=" * 70)
    print("\nHybrid AI + Quantum Defense Active")

if __name__ == '__main__':
    demo_quantum_hybrid()
