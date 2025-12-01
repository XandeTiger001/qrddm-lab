import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random

@dataclass
class QuantumReading:
    """Local quantum measurement from a cluster"""
    layer: int
    n_qubits: int
    superposition: float
    noise: float
    entanglement: float
    interference_pattern: str
    ternary_vote: int

@dataclass
class ClusterDecision:
    """Combined decision from all quantum clusters"""
    attack_id: int
    readings: List[QuantumReading]
    combined_superposition: float
    combined_noise: float
    combined_entanglement: float
    final_ternary_state: int
    confidence: float
    countermeasure: str

class QuantumCluster:
    """Quantum cluster for local defense measurements"""
    
    def __init__(self, layer: int, n_qubits: int, cluster_type: str):
        self.layer = layer
        self.n_qubits = n_qubits
        self.cluster_type = cluster_type
        self.state_vector = self._initialize_state()
        
    def _initialize_state(self) -> np.ndarray:
        """Initialize quantum state in superposition"""
        n_states = 2 ** self.n_qubits
        return np.ones(n_states) / np.sqrt(n_states)
    
    def measure_superposition(self, attack_signal: np.ndarray) -> float:
        """Measure quantum superposition level"""
        # Apply attack signal to state
        perturbed_state = self.state_vector + 0.1 * attack_signal[:len(self.state_vector)]
        perturbed_state /= np.linalg.norm(perturbed_state)
        
        # Measure superposition (entropy of state)
        probs = np.abs(perturbed_state) ** 2
        superposition = -np.sum(probs * np.log2(probs + 1e-10))
        return superposition / self.n_qubits  # Normalize
    
    def measure_noise(self, attack_signal: np.ndarray) -> float:
        """Detect unexpected noise (subtle attacks)"""
        # Noise = deviation from expected uniform distribution
        expected = np.ones(len(self.state_vector)) / len(self.state_vector)
        perturbed = np.abs(self.state_vector + 0.1 * attack_signal[:len(self.state_vector)]) ** 2
        perturbed /= np.sum(perturbed)
        
        noise = np.sum((perturbed - expected) ** 2)
        return np.clip(noise * 10, 0, 1)
    
    def measure_entanglement(self, attack_signal: np.ndarray) -> float:
        """Measure entanglement (correlations for coordinated attacks)"""
        # Simplified entanglement measure using correlation
        if self.n_qubits < 2:
            return 0.0
        
        # Split state into bipartite system
        mid = len(self.state_vector) // 2
        part_a = self.state_vector[:mid]
        part_b = self.state_vector[mid:]
        
        # Add attack perturbation
        part_a = part_a + 0.05 * attack_signal[:len(part_a)]
        part_b = part_b + 0.05 * attack_signal[len(part_a):len(part_a)+len(part_b)]
        
        # Correlation as entanglement measure
        correlation = np.abs(np.corrcoef(np.abs(part_a), np.abs(part_b))[0, 1])
        return correlation
    
    def detect_interference(self, attack_signal: np.ndarray) -> str:
        """Detect interference patterns"""
        # Measure phase interference
        phases = np.angle(self.state_vector + 0.1 * attack_signal[:len(self.state_vector)] * 1j)
        
        # Classify interference pattern
        phase_variance = np.var(phases)
        if phase_variance > 2.0:
            return "DESTRUCTIVE"
        elif phase_variance > 1.0:
            return "MIXED"
        else:
            return "CONSTRUCTIVE"
    
    def local_ternary_vote(self, superposition: float, noise: float, 
                          entanglement: float) -> int:
        """Local ternary decision based on measurements"""
        threat_score = noise * 0.5 + (1 - superposition) * 0.3 + entanglement * 0.2
        
        if threat_score > 0.7:
            return 1  # Evidence
        elif threat_score > 0.3:
            return -1  # Shadow
        else:
            return 0  # Normal

class Layer1Cluster(QuantumCluster):
    """3-qubit cluster: Basic interference detection"""
    
    def __init__(self):
        super().__init__(layer=1, n_qubits=3, cluster_type="Interference Detector")
    
    def analyze(self, attack_signal: np.ndarray) -> QuantumReading:
        """Detect subtle attacks via interference"""
        superposition = self.measure_superposition(attack_signal)
        noise = self.measure_noise(attack_signal)
        entanglement = self.measure_entanglement(attack_signal)
        interference = self.detect_interference(attack_signal)
        vote = self.local_ternary_vote(superposition, noise, entanglement)
        
        print(f"   🔬 Layer 1 (3-qubit): Interference Detection")
        print(f"      Superposition: {superposition:.3f}")
        print(f"      Noise: {noise:.3f} {'⚠️' if noise > 0.5 else ''}")
        print(f"      Interference: {interference}")
        print(f"      Vote: {['Normal', 'Shadow', 'Evidence'][vote]}")
        
        return QuantumReading(
            layer=1,
            n_qubits=3,
            superposition=superposition,
            noise=noise,
            entanglement=entanglement,
            interference_pattern=interference,
            ternary_vote=vote
        )

class Layer2Cluster(QuantumCluster):
    """4-qubit cluster: Correlation detection"""
    
    def __init__(self):
        super().__init__(layer=2, n_qubits=4, cluster_type="Correlation Detector")
    
    def analyze(self, attack_signal: np.ndarray) -> QuantumReading:
        """Detect coordinated attacks via correlations"""
        superposition = self.measure_superposition(attack_signal)
        noise = self.measure_noise(attack_signal)
        entanglement = self.measure_entanglement(attack_signal)
        interference = self.detect_interference(attack_signal)
        vote = self.local_ternary_vote(superposition, noise, entanglement)
        
        print(f"   🔬 Layer 2 (4-qubit): Correlation Detection")
        print(f"      Superposition: {superposition:.3f}")
        print(f"      Entanglement: {entanglement:.3f} {'⚠️' if entanglement > 0.6 else ''}")
        print(f"      Coordinated Attack: {'YES' if entanglement > 0.6 else 'NO'}")
        print(f"      Vote: {['Normal', 'Shadow', 'Evidence'][vote]}")
        
        return QuantumReading(
            layer=2,
            n_qubits=4,
            superposition=superposition,
            noise=noise,
            entanglement=entanglement,
            interference_pattern=interference,
            ternary_vote=vote
        )

class Layer3Cluster(QuantumCluster):
    """5-qubit cluster: QAOA/VQE countermeasure selection"""
    
    def __init__(self):
        super().__init__(layer=3, n_qubits=5, cluster_type="QAOA/VQE Optimizer")
    
    def run_qaoa(self, threat_level: float) -> str:
        """Simplified QAOA for countermeasure optimization"""
        # QAOA finds optimal countermeasure configuration
        # Cost function: minimize threat while minimizing resource usage
        
        countermeasures = [
            "RATE_LIMIT",
            "IP_BLOCK",
            "HONEYPOT_REDIRECT",
            "DEEP_INSPECTION",
            "QUARANTINE"
        ]
        
        # Simulate QAOA optimization (simplified)
        costs = []
        for cm in countermeasures:
            # Cost = threat_level * effectiveness - resource_cost
            effectiveness = random.uniform(0.6, 0.95)
            resource_cost = random.uniform(0.1, 0.4)
            cost = -(threat_level * effectiveness - resource_cost)
            costs.append(cost)
        
        optimal_idx = np.argmin(costs)
        return countermeasures[optimal_idx]
    
    def run_vqe(self, readings: List[QuantumReading]) -> float:
        """Simplified VQE for threat energy estimation"""
        # VQE estimates ground state energy (threat level)
        avg_noise = np.mean([r.noise for r in readings])
        avg_entanglement = np.mean([r.entanglement for r in readings])
        
        # Hamiltonian expectation value
        energy = avg_noise * 0.6 + avg_entanglement * 0.4
        return energy
    
    def analyze(self, attack_signal: np.ndarray, 
                prev_readings: List[QuantumReading]) -> QuantumReading:
        """Run QAOA/VQE for countermeasure selection"""
        superposition = self.measure_superposition(attack_signal)
        noise = self.measure_noise(attack_signal)
        entanglement = self.measure_entanglement(attack_signal)
        interference = self.detect_interference(attack_signal)
        
        # Run VQE to estimate threat energy
        threat_energy = self.run_vqe(prev_readings + [QuantumReading(
            3, 5, superposition, noise, entanglement, interference, 0
        )])
        
        # Run QAOA to select countermeasure
        countermeasure = self.run_qaoa(threat_energy)
        
        vote = self.local_ternary_vote(superposition, noise, entanglement)
        
        print(f"   🔬 Layer 3 (5-qubit): QAOA/VQE Optimizer")
        print(f"      Threat Energy (VQE): {threat_energy:.3f}")
        print(f"      Optimal Countermeasure (QAOA): {countermeasure}")
        print(f"      Vote: {['Normal', 'Shadow', 'Evidence'][vote]}")
        
        reading = QuantumReading(
            layer=3,
            n_qubits=5,
            superposition=superposition,
            noise=noise,
            entanglement=entanglement,
            interference_pattern=interference,
            ternary_vote=vote
        )
        reading.countermeasure = countermeasure
        
        return reading

class QuantumClusterDefenseAI:
    """AI that combines quantum cluster readings for ternary decisions"""
    
    def __init__(self):
        self.layer1 = Layer1Cluster()
        self.layer2 = Layer2Cluster()
        self.layer3 = Layer3Cluster()
        self.decisions = []
        self.attack_counter = 0
    
    def generate_attack_signal(self, attack_type: str, intensity: float) -> np.ndarray:
        """Generate attack signal for quantum measurement"""
        n_samples = 32
        
        if attack_type == "SUBTLE":
            # Low amplitude, high frequency noise
            signal = np.random.normal(0, 0.1 * intensity, n_samples)
        elif attack_type == "COORDINATED":
            # Correlated patterns
            base = np.sin(np.linspace(0, 4*np.pi, n_samples))
            signal = base * intensity + np.random.normal(0, 0.05, n_samples)
        elif attack_type == "MASSIVE":
            # High amplitude across spectrum
            signal = np.random.uniform(-intensity, intensity, n_samples)
        else:
            signal = np.random.normal(0, 0.5 * intensity, n_samples)
        
        return signal
    
    def combine_readings(self, readings: List[QuantumReading]) -> Tuple[int, float, str]:
        """AI combines cluster readings for final ternary decision"""
        # Weighted voting (Layer 3 has highest weight)
        weights = [0.25, 0.35, 0.40]
        votes = [r.ternary_vote for r in readings]
        
        # Weighted average
        weighted_vote = sum(w * v for w, v in zip(weights, votes))
        
        # Final ternary classification
        if weighted_vote > 0.5:
            final_state = 1  # Evidence
        elif weighted_vote < -0.3:
            final_state = 0  # Normal
        else:
            final_state = -1  # Shadow
        
        # Combined measurements
        combined_superposition = np.mean([r.superposition for r in readings])
        combined_noise = np.mean([r.noise for r in readings])
        combined_entanglement = np.mean([r.entanglement for r in readings])
        
        # Confidence based on agreement
        vote_variance = np.var(votes)
        confidence = 1.0 / (1.0 + vote_variance)
        
        # Get countermeasure from Layer 3
        countermeasure = readings[2].countermeasure if hasattr(readings[2], 'countermeasure') else "MONITOR"
        
        return final_state, confidence, countermeasure
    
    def process_attack(self, attack_type: str, intensity: float) -> ClusterDecision:
        """Process attack through quantum cluster defense"""
        attack_id = self.attack_counter
        self.attack_counter += 1
        
        print(f"\n{'='*70}")
        print(f"🎯 ATTACK #{attack_id:03d}: {attack_type} (Intensity: {intensity:.2f})")
        print(f"{'='*70}")
        
        # Generate attack signal
        attack_signal = self.generate_attack_signal(attack_type, intensity)
        
        # Layer 1: Interference detection
        reading1 = self.layer1.analyze(attack_signal)
        
        # Layer 2: Correlation detection
        reading2 = self.layer2.analyze(attack_signal)
        
        # Layer 3: QAOA/VQE optimization
        reading3 = self.layer3.analyze(attack_signal, [reading1, reading2])
        
        readings = [reading1, reading2, reading3]
        
        # AI combines readings
        print(f"\n   🤖 AI COMBINING CLUSTER READINGS...")
        final_state, confidence, countermeasure = self.combine_readings(readings)
        
        decision = ClusterDecision(
            attack_id=attack_id,
            readings=readings,
            combined_superposition=np.mean([r.superposition for r in readings]),
            combined_noise=np.mean([r.noise for r in readings]),
            combined_entanglement=np.mean([r.entanglement for r in readings]),
            final_ternary_state=final_state,
            confidence=confidence,
            countermeasure=countermeasure
        )
        
        self.decisions.append(decision)
        
        state_names = {-1: "🌫️  SHADOW", 0: "✅ NORMAL", 1: "🚨 EVIDENCE"}
        print(f"\n   {'='*60}")
        print(f"   FINAL DECISION: {state_names[final_state]}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Countermeasure: {countermeasure}")
        print(f"   {'='*60}")
        
        return decision
    
    def visualize_cluster_analysis(self):
        """Visualize quantum cluster measurements"""
        if not self.decisions:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        attack_ids = [d.attack_id for d in self.decisions]
        
        # 1. Measurements by layer
        for layer in [1, 2, 3]:
            noise_vals = [d.readings[layer-1].noise for d in self.decisions]
            axes[0, 0].plot(attack_ids, noise_vals, 'o-', label=f'Layer {layer} ({[3,4,5][layer-1]}-qubit)', linewidth=2)
        
        axes[0, 0].set_xlabel('Attack ID', fontsize=11)
        axes[0, 0].set_ylabel('Noise Level', fontsize=11)
        axes[0, 0].set_title('Quantum Noise Detection by Layer', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ternary decisions
        states = [d.final_ternary_state for d in self.decisions]
        colors = ['green' if s == 0 else 'orange' if s == -1 else 'red' for s in states]
        axes[0, 1].scatter(attack_ids, states, c=colors, s=200, edgecolors='black', linewidth=2)
        axes[0, 1].set_xlabel('Attack ID', fontsize=11)
        axes[0, 1].set_ylabel('Ternary State', fontsize=11)
        axes[0, 1].set_yticks([-1, 0, 1])
        axes[0, 1].set_yticklabels(['Shadow', 'Normal', 'Evidence'])
        axes[0, 1].set_title('AI Ternary Decisions', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Entanglement (coordinated attacks)
        entanglement_vals = [d.combined_entanglement for d in self.decisions]
        axes[1, 0].bar(attack_ids, entanglement_vals, color='cyan', edgecolor='black', alpha=0.7)
        axes[1, 0].axhline(y=0.6, color='red', linestyle='--', label='Coordination threshold')
        axes[1, 0].set_xlabel('Attack ID', fontsize=11)
        axes[1, 0].set_ylabel('Entanglement', fontsize=11)
        axes[1, 0].set_title('Coordinated Attack Detection', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Countermeasures
        countermeasures = [d.countermeasure for d in self.decisions]
        cm_counts = {}
        for cm in countermeasures:
            cm_counts[cm] = cm_counts.get(cm, 0) + 1
        
        cms = list(cm_counts.keys())
        counts = list(cm_counts.values())
        axes[1, 1].barh(cms, counts, color='purple', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Frequency', fontsize=11)
        axes[1, 1].set_title('QAOA Countermeasure Selection', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('quantum_cluster_defense.png', dpi=150, bbox_inches='tight')
        plt.show()

def demo():
    """Demonstrate quantum cluster defense"""
    print("🌌 QUANTUM CLUSTER DEFENSE SYSTEM")
    print("="*70)
    print("Each layer has its own quantum cluster:")
    print("  Layer 1: 3-qubit → Interference detection (subtle attacks)")
    print("  Layer 2: 4-qubit → Correlation detection (coordinated attacks)")
    print("  Layer 3: 5-qubit → QAOA/VQE (countermeasure optimization)")
    print("="*70)
    
    ai = QuantumClusterDefenseAI()
    
    # Test scenarios
    scenarios = [
        ("SUBTLE", 0.4, "Low-level probe"),
        ("COORDINATED", 0.7, "Multi-vector attack"),
        ("MASSIVE", 0.9, "DDoS flood"),
        ("SUBTLE", 0.3, "Reconnaissance"),
        ("COORDINATED", 0.8, "APT campaign"),
        ("NORMAL", 0.1, "Legitimate traffic"),
    ]
    
    for attack_type, intensity, desc in scenarios:
        print(f"\n📡 Scenario: {desc}")
        ai.process_attack(attack_type, intensity)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 DEFENSE SUMMARY")
    print(f"{'='*70}")
    
    evidence = sum(1 for d in ai.decisions if d.final_ternary_state == 1)
    shadow = sum(1 for d in ai.decisions if d.final_ternary_state == -1)
    normal = sum(1 for d in ai.decisions if d.final_ternary_state == 0)
    
    print(f"🚨 Evidence (Confirmed Threats): {evidence}")
    print(f"🌫️  Shadow (Suspicious): {shadow}")
    print(f"✅ Normal (Benign): {normal}")
    
    avg_confidence = np.mean([d.confidence for d in ai.decisions])
    print(f"\n🎯 Average Confidence: {avg_confidence:.3f}")
    
    print("\n🎨 Generating visualization...")
    ai.visualize_cluster_analysis()
    
    return ai

if __name__ == '__main__':
    ai = demo()
