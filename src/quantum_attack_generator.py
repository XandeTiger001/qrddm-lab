import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass
from typing import List, Dict
import random

@dataclass
class QuantumAttackState:
    amplitudes: np.ndarray
    attack_vectors: List[str]
    entropy: float
    
    def measure(self) -> str:
        probabilities = np.abs(self.amplitudes) ** 2
        return np.random.choice(self.attack_vectors, p=probabilities)

class QuantumAttackGenerator:
    def __init__(self):
        self.attack_vectors = ['sql_injection', 'xss', 'ddos', 'buffer_overflow', 'path_traversal', 'malware', 'apt', 'zero_day']
        self.n_states = len(self.attack_vectors)
    
    def create_superposition_attack(self, weights: Dict[str, float] = None) -> QuantumAttackState:
        """Create superposition of multiple attack vectors"""
        if weights is None:
            # Equal superposition
            amplitudes = np.ones(self.n_states, dtype=complex) / np.sqrt(self.n_states)
        else:
            amplitudes = np.zeros(self.n_states, dtype=complex)
            for i, vector in enumerate(self.attack_vectors):
                weight = weights.get(vector, 0.0)
                phase = np.random.uniform(0, 2*np.pi)
                amplitudes[i] = weight * np.exp(1j * phase)
            
            # Normalize
            norm = np.linalg.norm(amplitudes)
            if norm > 0:
                amplitudes = amplitudes / norm
        
        entropy = self._calculate_entropy(amplitudes)
        return QuantumAttackState(amplitudes, self.attack_vectors, entropy)
    
    def create_maximum_entropy_attack(self) -> QuantumAttackState:
        """Generate attack state with maximum entropy (most unpredictable)"""
        # Maximum entropy = uniform distribution
        amplitudes = np.ones(self.n_states, dtype=complex) / np.sqrt(self.n_states)
        
        # Add random phases for maximum uncertainty
        phases = np.random.uniform(0, 2*np.pi, self.n_states)
        amplitudes = amplitudes * np.exp(1j * phases)
        
        entropy = self._calculate_entropy(amplitudes)
        return QuantumAttackState(amplitudes, self.attack_vectors, entropy)
    
    def add_adversarial_noise(self, state: QuantumAttackState, noise_level: float = 0.2) -> QuantumAttackState:
        """Add adversarial quantum noise to attack state"""
        # Generate adversarial noise
        noise = np.random.normal(0, noise_level, self.n_states) + 1j * np.random.normal(0, noise_level, self.n_states)
        
        # Apply noise to amplitudes
        noisy_amplitudes = state.amplitudes + noise
        
        # Renormalize
        norm = np.linalg.norm(noisy_amplitudes)
        if norm > 0:
            noisy_amplitudes = noisy_amplitudes / norm
        
        entropy = self._calculate_entropy(noisy_amplitudes)
        return QuantumAttackState(noisy_amplitudes, state.attack_vectors, entropy)
    
    def generate_difficult_attack_states(self, num_states: int = 5) -> List[QuantumAttackState]:
        """Generate multiple difficult quantum attack states"""
        states = []
        
        # 1. Maximum entropy state
        states.append(self.create_maximum_entropy_attack())
        
        # 2. Weighted superposition states
        for _ in range(num_states - 1):
            weights = {vector: np.random.uniform(0.1, 1.0) for vector in self.attack_vectors}
            base_state = self.create_superposition_attack(weights)
            # Add adversarial noise
            noisy_state = self.add_adversarial_noise(base_state, noise_level=0.3)
            states.append(noisy_state)
        
        return states
    
    def _calculate_entropy(self, amplitudes: np.ndarray) -> float:
        """Calculate von Neumann entropy"""
        probabilities = np.abs(amplitudes) ** 2
        probabilities = probabilities[probabilities > 1e-12]
        return -np.sum(probabilities * np.log2(probabilities + 1e-12))

class QuantumToClassicalConverter:
    def __init__(self):
        self.attack_templates = {
            'sql_injection': {'payload': "' OR 1=1--", 'request_size': 1024, 'frequency': 5},
            'xss': {'payload': "<script>alert('xss')</script>", 'request_size': 512, 'frequency': 3},
            'ddos': {'payload': "flood_request", 'request_size': 2048, 'frequency': 100},
            'buffer_overflow': {'payload': "A" * 1000, 'request_size': 4096, 'frequency': 2},
            'path_traversal': {'payload': "../../../etc/passwd", 'request_size': 256, 'frequency': 4},
            'malware': {'payload': "malicious_binary", 'request_size': 8192, 'frequency': 1},
            'apt': {'payload': "advanced_persistent", 'request_size': 1536, 'frequency': 0.5},
            'zero_day': {'payload': "unknown_exploit", 'request_size': 3072, 'frequency': 0.1}
        }
    
    def convert_quantum_to_attack_data(self, quantum_state: QuantumAttackState) -> Dict:
        """Convert quantum state to classical attack data"""
        # Measure the quantum state
        measured_attack = quantum_state.measure()
        template = self.attack_templates[measured_attack]
        
        # Create attack data with quantum-influenced parameters
        attack_data = {
            'type': measured_attack,
            'payload': template['payload'],
            'source_ip': f"203.0.113.{random.randint(1, 255)}",
            'request_size': template['request_size'] * (1 + quantum_state.entropy * 0.5),
            'response_time': 0.1 / (1 + quantum_state.entropy),
            'frequency': template['frequency'] * (1 + quantum_state.entropy),
            'target_layer': random.randint(3, 7),
            'num_techniques': int(quantum_state.entropy * 3) + 1,
            'quantum_entropy': quantum_state.entropy,
            'quantum_superposition': True,
            'label': 1  # Malicious
        }
        
        return attack_data

def demo_quantum_attack_generation():
    """Demo quantum attack state generation"""
    print("QUANTUM ATTACK STATE GENERATION")
    print("=" * 50)
    
    generator = QuantumAttackGenerator()
    converter = QuantumToClassicalConverter()
    
    # Generate difficult quantum attack states
    quantum_states = generator.generate_difficult_attack_states(3)
    
    print(f"Generated {len(quantum_states)} quantum attack states:")
    
    for i, state in enumerate(quantum_states):
        print(f"\nQuantum State {i+1}:")
        print(f"  Entropy: {state.entropy:.3f}")
        print(f"  Dominant vectors: {[v for j, v in enumerate(state.attack_vectors) if np.abs(state.amplitudes[j]) > 0.3]}")
        
        # Convert to classical attack
        attack_data = converter.convert_quantum_to_attack_data(state)
        print(f"  Measured attack: {attack_data['type']}")
        print(f"  Enhanced size: {attack_data['request_size']:.1f}")
        print(f"  Enhanced frequency: {attack_data['frequency']:.2f}")

if __name__ == '__main__':
    demo_quantum_attack_generation()