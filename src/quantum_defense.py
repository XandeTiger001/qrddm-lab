import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random

@dataclass
class QuantumState:
    amplitudes: np.ndarray
    basis_labels: List[str]
    
    def normalize(self):
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
    
    def measure(self) -> str:
        probabilities = np.abs(self.amplitudes) ** 2
        return np.random.choice(self.basis_labels, p=probabilities)

class QuantumThreatSimulator:
    def __init__(self):
        self.attack_basis = ['normal', 'sql_inj', 'xss', 'ddos', 'malware', 'apt', 'zero_day', 'quantum_hack']
        self.n_qubits = 3  # 8 states
        
    def create_superposition_attack(self, threat_weights: Dict[str, float]) -> QuantumState:
        """Create quantum superposition of multiple attack types"""
        amplitudes = np.zeros(len(self.attack_basis), dtype=complex)
        
        for i, attack_type in enumerate(self.attack_basis):
            weight = threat_weights.get(attack_type, 0.0)
            phase = np.random.uniform(0, 2*np.pi)
            amplitudes[i] = weight * np.exp(1j * phase)
        
        state = QuantumState(amplitudes, self.attack_basis)
        state.normalize()
        return state
    
    def quantum_entangle_attacks(self, attack1: QuantumState, attack2: QuantumState) -> QuantumState:
        """Create entangled attack scenarios"""
        # Tensor product for entanglement
        entangled_amplitudes = np.kron(attack1.amplitudes, attack2.amplitudes)
        entangled_labels = [f"{a1}x{a2}" for a1 in attack1.basis_labels for a2 in attack2.basis_labels]
        
        return QuantumState(entangled_amplitudes, entangled_labels)
    
    def apply_uncertainty_principle(self, state: QuantumState, measurement_precision: float = 0.8) -> QuantumState:
        """Simulate quantum uncertainty in attack detection"""
        # Add quantum noise based on uncertainty principle
        noise_amplitude = np.sqrt(1 - measurement_precision**2)
        noise = np.random.normal(0, noise_amplitude, len(state.amplitudes)) + \
                1j * np.random.normal(0, noise_amplitude, len(state.amplitudes))
        
        noisy_amplitudes = state.amplitudes + noise * 0.1
        noisy_state = QuantumState(noisy_amplitudes, state.basis_labels)
        noisy_state.normalize()
        return noisy_state

class QuantumPerturbationGenerator:
    def __init__(self):
        self.pauli_matrices = {
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            'I': np.array([[1, 0], [0, 1]], dtype=complex)
        }
    
    def generate_attack_perturbations(self, base_attack: Dict, num_variants: int = 5) -> List[Dict]:
        """Generate quantum-inspired attack variants"""
        variants = []
        
        for i in range(num_variants):
            # Apply random Pauli rotations to attack parameters
            variant = base_attack.copy()
            
            # Quantum bit-flip on payload
            if 'payload' in variant and random.random() < 0.3:
                payload = list(variant['payload'])
                flip_pos = random.randint(0, len(payload)-1)
                payload[flip_pos] = chr(ord(payload[flip_pos]) ^ 1)  # Quantum flip
                variant['payload'] = ''.join(payload)
            
            # Phase rotation on numerical parameters
            for key in ['request_size', 'frequency', 'target_layer']:
                if key in variant:
                    phase = np.random.uniform(0, 2*np.pi)
                    rotation_factor = 1 + 0.1 * np.cos(phase)
                    variant[key] = variant[key] * rotation_factor
            
            # Quantum tunneling effect - bypass normal constraints
            if random.random() < 0.2:
                variant['quantum_tunnel'] = True
                variant['bypass_probability'] = np.random.uniform(0.1, 0.4)
            
            variants.append(variant)
        
        return variants
    
    def create_coherent_attack_wave(self, attack_frequency: float, duration: float) -> np.ndarray:
        """Generate coherent quantum attack wave"""
        t = np.linspace(0, duration, int(duration * 100))
        # Quantum coherent state with Poissonian statistics
        alpha = np.sqrt(attack_frequency)  # Coherent state parameter
        wave = alpha * np.exp(1j * attack_frequency * t) * np.exp(-0.5 * np.abs(alpha)**2)
        return wave

class QuantumStateAnalyzer:
    def __init__(self):
        self.hilbert_dim = 8  # 2^3 for 3-qubit system
    
    def analyze_attack_state_space(self, attack_data: Dict) -> Dict:
        """Analyze attack in quantum state space"""
        # Map attack to quantum state
        state_vector = self._encode_attack_to_state(attack_data)
        
        # Calculate quantum properties
        entropy = self._von_neumann_entropy(state_vector)
        purity = self._calculate_purity(state_vector)
        coherence = self._calculate_coherence(state_vector)
        
        # Quantum phase analysis
        phases = np.angle(state_vector)
        phase_variance = np.var(phases)
        
        return {
            'quantum_entropy': entropy,
            'state_purity': purity,
            'coherence_measure': coherence,
            'phase_variance': phase_variance,
            'entanglement_degree': self._measure_entanglement(state_vector),
            'quantum_fidelity': self._calculate_fidelity_to_normal(state_vector)
        }
    
    def _encode_attack_to_state(self, attack_data: Dict) -> np.ndarray:
        """Encode attack parameters into quantum state"""
        # Simple encoding: map attack features to quantum amplitudes
        features = [
            attack_data.get('threat_score', 0.5),
            attack_data.get('frequency', 0.3),
            attack_data.get('complexity', 0.4)
        ]
        
        # Create quantum state from features
        angles = [f * np.pi for f in features]
        state = np.zeros(self.hilbert_dim, dtype=complex)
        
        for i in range(self.hilbert_dim):
            amplitude = 1.0
            for j, angle in enumerate(angles):
                if (i >> j) & 1:
                    amplitude *= np.sin(angle/2)
                else:
                    amplitude *= np.cos(angle/2)
            state[i] = amplitude
        
        return state / np.linalg.norm(state)
    
    def _von_neumann_entropy(self, state: np.ndarray) -> float:
        """Calculate quantum entropy"""
        rho = np.outer(state, np.conj(state))  # Density matrix
        eigenvals = np.real(np.linalg.eigvals(rho))
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        return -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
    
    def _calculate_purity(self, state: np.ndarray) -> float:
        """Calculate state purity"""
        rho = np.outer(state, np.conj(state))
        return np.real(np.trace(rho @ rho))
    
    def _calculate_coherence(self, state: np.ndarray) -> float:
        """Calculate quantum coherence"""
        rho = np.outer(state, np.conj(state))
        # l1-norm coherence
        coherence = np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))
        return coherence
    
    def _measure_entanglement(self, state: np.ndarray) -> float:
        """Measure entanglement (simplified)"""
        # Reshape for bipartite system
        state_matrix = state.reshape(2, 4)
        # Calculate Schmidt decomposition approximation
        U, s, Vh = np.linalg.svd(state_matrix)
        # Entanglement entropy
        s_normalized = s / np.linalg.norm(s)
        s_filtered = s_normalized[s_normalized > 1e-12]
        return -np.sum(s_filtered**2 * np.log2(s_filtered**2 + 1e-12))
    
    def _calculate_fidelity_to_normal(self, state: np.ndarray) -> float:
        """Calculate fidelity to normal traffic state"""
        normal_state = np.zeros(len(state), dtype=complex)
        normal_state[0] = 1.0  # |000⟩ state represents normal traffic
        return np.abs(np.vdot(normal_state, state))**2

class QuantumAdversarialTester:
    def __init__(self):
        self.quantum_gates = ['H', 'X', 'Y', 'Z', 'CNOT', 'T']
    
    def quantum_stress_test(self, defense_system, num_tests: int = 10) -> List[Dict]:
        """Perform quantum-enhanced adversarial testing"""
        results = []
        
        for i in range(num_tests):
            # Generate quantum adversarial attack
            quantum_attack = self._generate_quantum_adversarial_attack()
            
            # Test defense system
            try:
                defense_result = defense_system.analyze_attack(quantum_attack)
                success = defense_result.get('decision', 'ALLOW') != 'ALLOW'
                
                results.append({
                    'test_id': i,
                    'attack_type': 'quantum_adversarial',
                    'quantum_properties': quantum_attack.get('quantum_props', {}),
                    'defense_success': success,
                    'detection_confidence': defense_result.get('confidence', 0.0),
                    'quantum_bypass_attempted': quantum_attack.get('quantum_tunnel', False)
                })
            except Exception as e:
                results.append({
                    'test_id': i,
                    'error': str(e),
                    'defense_success': False
                })
        
        return results
    
    def _generate_quantum_adversarial_attack(self) -> Dict:
        """Generate quantum-enhanced adversarial attack"""
        base_attack = {
            'type': 'quantum_adversarial',
            'payload': self._quantum_scrambled_payload(),
            'source_ip': f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            'request_size': random.randint(500, 2000),
            'frequency': random.uniform(0.1, 0.8),
            'target_layer': random.randint(0, 3),
            'threat_score': random.uniform(0.3, 0.9)
        }
        
        # Add quantum properties
        base_attack['quantum_props'] = {
            'superposition_factor': random.uniform(0.2, 0.8),
            'entanglement_strength': random.uniform(0.1, 0.6),
            'decoherence_time': random.uniform(0.01, 0.1)
        }
        
        # Quantum tunneling attempt
        if random.random() < 0.3:
            base_attack['quantum_tunnel'] = True
            base_attack['bypass_probability'] = random.uniform(0.1, 0.4)
        
        return base_attack
    
    def _quantum_scrambled_payload(self) -> str:
        """Generate quantum-scrambled attack payload"""
        base_payloads = [
            "' OR 1=1--",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "UNION SELECT * FROM users"
        ]
        
        payload = random.choice(base_payloads)
        
        # Apply quantum scrambling
        scrambled = []
        for char in payload:
            # Quantum bit operations
            if random.random() < 0.1:  # Quantum noise
                scrambled.append(chr((ord(char) + random.randint(1, 3)) % 128))
            else:
                scrambled.append(char)
        
        return ''.join(scrambled)

class QuantumEnhancedDefense:
    def __init__(self, base_defense):
        self.base_defense = base_defense
        self.quantum_simulator = QuantumThreatSimulator()
        self.perturbation_gen = QuantumPerturbationGenerator()
        self.state_analyzer = QuantumStateAnalyzer()
        self.adversarial_tester = QuantumAdversarialTester()
    
    def analyze_with_quantum_enhancement(self, attack_data: Dict) -> Dict:
        """Enhanced analysis with quantum mechanics"""
        # Base analysis
        base_result = self.base_defense.analyze_attack(attack_data)
        
        # Quantum state analysis
        quantum_analysis = self.state_analyzer.analyze_attack_state_space(attack_data)
        
        # Quantum threat assessment
        threat_weights = {
            attack_data.get('type', 'normal'): base_result.get('threat_probability', 0.5)
        }
        quantum_state = self.quantum_simulator.create_superposition_attack(threat_weights)
        
        # Apply quantum uncertainty
        uncertain_state = self.quantum_simulator.apply_uncertainty_principle(quantum_state)
        
        # Enhanced decision with quantum factors
        quantum_factor = 1.0 - quantum_analysis['quantum_fidelity']
        enhanced_threat = base_result['enhanced_threat'] * (1 + quantum_factor * 0.3)
        
        # Quantum tunneling check
        if attack_data.get('quantum_tunnel', False):
            tunnel_prob = attack_data.get('bypass_probability', 0.0)
            if random.random() < tunnel_prob:
                enhanced_threat *= 0.7  # Partial bypass
        
        return {
            **base_result,
            'quantum_analysis': quantum_analysis,
            'quantum_enhanced_threat': enhanced_threat,
            'quantum_state_measured': uncertain_state.measure(),
            'quantum_bypass_detected': attack_data.get('quantum_tunnel', False)
        }

def demo_quantum_defense():
    print("⚛️  Quantum-Enhanced Cyber Defense")
    print("=" * 50)
    
    # Mock base defense for demo
    class MockDefense:
        def analyze_attack(self, attack_data):
            return {
                'threat_probability': random.uniform(0.3, 0.8),
                'enhanced_threat': random.uniform(0.4, 0.9),
                'decision': random.choice(['ALLOW', 'MONITOR', 'BLOCK']),
                'confidence': random.uniform(0.6, 0.95)
            }
    
    base_defense = MockDefense()
    quantum_defense = QuantumEnhancedDefense(base_defense)
    
    # Test scenarios
    test_attacks = [
        {
            'type': 'sql_injection',
            'payload': "' OR 1=1--",
            'threat_score': 0.7,
            'frequency': 0.3,
            'complexity': 0.6
        },
        {
            'type': 'quantum_adversarial',
            'payload': "quantum_scrambled_attack",
            'threat_score': 0.85,
            'frequency': 0.4,
            'complexity': 0.9,
            'quantum_tunnel': True,
            'bypass_probability': 0.3
        }
    ]
    
    for i, attack in enumerate(test_attacks):
        print(f"\n🔬 Test {i+1}: {attack['type']}")
        result = quantum_defense.analyze_with_quantum_enhancement(attack)
        
        qa = result['quantum_analysis']
        print(f"   Quantum Entropy: {qa['quantum_entropy']:.3f}")
        print(f"   State Purity: {qa['state_purity']:.3f}")
        print(f"   Coherence: {qa['coherence_measure']:.3f}")
        print(f"   Fidelity to Normal: {qa['quantum_fidelity']:.3f}")
        print(f"   Enhanced Threat: {result['quantum_enhanced_threat']:.3f}")
        print(f"   Quantum State: {result['quantum_state_measured']}")
        
        if result.get('quantum_bypass_detected'):
            print("   ⚠️  Quantum tunneling attempt detected!")
    
    print(f"\n{'='*50}")
    print("🧪 Running Quantum Stress Tests...")
    
    stress_results = quantum_defense.adversarial_tester.quantum_stress_test(base_defense, 5)
    success_rate = sum(1 for r in stress_results if r.get('defense_success', False)) / len(stress_results)
    print(f"   Defense Success Rate: {success_rate:.1%}")
    print(f"   Quantum Bypasses Attempted: {sum(1 for r in stress_results if 'quantum_bypass_attempted' in r)}")

if __name__ == '__main__':
    demo_quantum_defense()