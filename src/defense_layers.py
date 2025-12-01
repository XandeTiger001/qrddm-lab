import numpy as np

class DefenseLayer:
    def __init__(self, radius, G, alpha, c=1.0):
        self.radius = radius
        self.G = G
        self.alpha = alpha
        self.c = c
    
    def calculate_force(self, M):
        """F_defense(r) = G*M / r^2"""
        return (self.G * M) / (self.radius ** 2)
    
    def calculate_phi(self, M):
        """Φ(r) = 1 - 2GM/(c²r)"""
        phi = 1 - (2 * self.G * M) / (self.c**2 * self.radius)
        return max(phi, 0.01)
    
    def calculate_time_dilation(self, M, dt=1.0):
        """Δt' = Δt / sqrt(Φ(r)) - Time dilation near horizon"""
        phi = self.calculate_phi(M)
        dt_prime = dt / np.sqrt(phi)
        return dt_prime
    
    def neutralization_probability(self, F_defense):
        """P_success = 1 - e^(-alpha * F_defense)"""
        return 1 - np.exp(-self.alpha * F_defense)

class LayeredDefenseSystem:
    def __init__(self, layers_config):
        self.layers = [DefenseLayer(r, G, alpha) for r, G, alpha in layers_config]
    
    def simulate_attack(self, M0, mode='deterministic'):
        """Simulate attack through layers with time dilation"""
        M = M0
        results = []
        total_processing_time = 0
        
        for i, layer in enumerate(self.layers):
            F_defense = layer.calculate_force(M)
            P_success = layer.neutralization_probability(F_defense)
            phi = layer.calculate_phi(M)
            dt_prime = layer.calculate_time_dilation(M)
            
            total_processing_time += dt_prime
            
            # AI uses extra time to boost defense
            G_boost = 1.0 + (dt_prime - 1.0) * 0.1
            F_defense_boosted = F_defense * G_boost
            
            if mode == 'probabilistic':
                if np.random.random() < P_success:
                    results.append({
                        'layer': i, 'M': M, 'F': F_defense_boosted, 'P': P_success,
                        'phi': phi, 'dt_prime': dt_prime, 'neutralized': True
                    })
                    return results, 0.0, 1, total_processing_time
            
            M_new = max(0, M - F_defense_boosted)
            results.append({
                'layer': i, 'M': M, 'F': F_defense_boosted, 'P': P_success,
                'phi': phi, 'dt_prime': dt_prime, 'neutralized': False
            })
            M = M_new
            
            if M <= 0.01:
                return results, M, 1, total_processing_time
        
        if M < 0.3:
            state = 1
        elif M < 0.7:
            state = 0
        else:
            state = -1
        
        return results, M, state, total_processing_time

def demo_layered_defense():
    print("🛡️  Layered Defense System with Quantum Probability\n")
    print("=" * 70)
    
    # Configure 5 concentric layers: (radius, G, alpha)
    layers = [
        (10.0, 2.0, 1.5),  # Outer layer
        (7.5, 2.5, 1.8),
        (5.0, 3.0, 2.0),   # Middle
        (2.5, 3.5, 2.2),
        (1.0, 4.0, 2.5)    # Core layer
    ]
    
    system = LayeredDefenseSystem(layers)
    
    # Test different attack intensities
    attacks = [
        ('Weak', 1.0),
        ('Medium', 3.0),
        ('Strong', 5.0),
        ('Critical', 10.0)
    ]
    
    print("\n📊 Deterministic Mode with Time Dilation:\n")
    for name, M0 in attacks:
        results, M_final, state, total_time = system.simulate_attack(M0, mode='deterministic')
        
        print(f"{name} Attack (M0={M0:.1f}):")
        print(f"  Final M: {M_final:.4f}")
        print(f"  State: {state:+d} ({'Protected' if state==1 else 'Attention' if state==0 else 'Compromised'})")
        print(f"  Processing time: {total_time:.2f}x (time dilation)")
        print(f"  Layers traversed: {len(results)}\n")
    
    print("\n🎲 Probabilistic Mode (100 runs):\n")
    for name, M0 in attacks:
        neutralized = 0
        final_Ms = []
        processing_times = []
        
        for _ in range(100):
            results, M_final, state, total_time = system.simulate_attack(M0, mode='probabilistic')
            if state == 1:
                neutralized += 1
            final_Ms.append(M_final)
            processing_times.append(total_time)
        
        print(f"{name} Attack (M0={M0:.1f}):")
        print(f"  Neutralization rate: {neutralized}%")
        print(f"  Avg final M: {np.mean(final_Ms):.4f}")
        print(f"  Avg processing time: {np.mean(processing_times):.2f}x\n")
    
    print("=" * 70)

if __name__ == '__main__':
    demo_layered_defense()
