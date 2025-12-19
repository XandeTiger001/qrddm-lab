import numpy as np
from tier1_classic_defense import Tier1ClassicDefense, generate_synthetic_attacks

class QuantumHardAttackGenerator:
    def __init__(self):
        self.vectors = ['sql', 'xss', 'ddos', 'overflow', 'traversal', 'malware']
    
    def superposition_attack(self, weights=None):
        """Create superposition of attack vectors"""
        n = len(self.vectors)
        if weights is None:
            amplitudes = np.ones(n) / np.sqrt(n)  # Equal superposition
        else:
            amplitudes = np.array([weights.get(v, 0) for v in self.vectors])
            amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        # Add random phases
        phases = np.random.uniform(0, 2*np.pi, n)
        state = amplitudes * np.exp(1j * phases)
        
        return self._to_attack_data(state)
    
    def max_entropy_attack(self):
        """Generate maximum entropy attack state"""
        n = len(self.vectors)
        amplitudes = np.ones(n) / np.sqrt(n)
        phases = np.random.uniform(0, 2*np.pi, n)
        state = amplitudes * np.exp(1j * phases)
        entropy = np.log2(n)  # Maximum entropy
        
        attack = self._to_attack_data(state)
        attack['quantum_entropy'] = entropy
        return attack
    
    def adversarial_noise_attack(self, base_state, noise=0.3):
        """Add adversarial noise to attack state"""
        noise_real = np.random.normal(0, noise, len(base_state))
        noise_imag = np.random.normal(0, noise, len(base_state))
        noisy_state = base_state + noise_real + 1j * noise_imag
        noisy_state = noisy_state / np.linalg.norm(noisy_state)
        
        return self._to_attack_data(noisy_state)
    
    def _to_attack_data(self, quantum_state):
        """Convert quantum state to attack data"""
        probs = np.abs(quantum_state) ** 2
        dominant = np.argmax(probs)
        attack_type = self.vectors[dominant]
        
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        
        payloads = {
            'sql': "' OR 1=1--", 'xss': "<script>alert(1)</script>",
            'ddos': "flood", 'overflow': "A"*1000, 'traversal': "../etc/passwd",
            'malware': "malicious_code"
        }
        
        return {
            'type': attack_type,
            'payload': payloads[attack_type],
            'source_ip': '203.0.113.42',
            'request_size': int(1024 * (1 + entropy)),
            'frequency': entropy * 10,
            'target_layer': 7,
            'num_techniques': int(entropy * 2) + 1,
            'quantum_entropy': entropy,
            'label': 1
        }

def demo():
    print("QUANTUM HARD ATTACKS -> AI + PHYSICS ENSEMBLE")
    print("=" * 50)
    
    # Setup
    gen = QuantumHardAttackGenerator()
    defense = Tier1ClassicDefense()
    defense.train_system(generate_synthetic_attacks(30))
    
    # Generate hard quantum attacks
    attacks = [
        gen.superposition_attack(),
        gen.max_entropy_attack(),
        gen.adversarial_noise_attack(np.ones(6) / np.sqrt(6))
    ]
    
    print("\nQuantum Attack States:")
    for i, attack in enumerate(attacks):
        result = defense.analyze_attack(attack)
        print(f"{i+1}. {attack['type']} | Entropy: {attack['quantum_entropy']:.2f} | "
              f"AI: {result['threat_probability']:.2f} | Physics: {result['physics']['M']:.2f} | "
              f"Decision: {result['decision']}")

if __name__ == '__main__':
    demo()