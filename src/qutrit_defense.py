import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import time

@dataclass
class QutritState:
    """Qutrit quantum state (3-level system)"""
    amplitudes: np.ndarray  # [a0, a1, a2] for |0⟩, |1⟩, |2⟩
    ternary_classical: int  # -1, 0, +1
    
@dataclass
class DefenseResult:
    """Defense processing result"""
    input_state: int
    qutrit_amplitudes: np.ndarray
    entropy: float
    interference: float
    entanglement: float
    output_state: int
    processing_time: float
    confidence: float

class QutritGates:
    """Quantum gates for qutrits"""
    
    @staticmethod
    def hadamard_qutrit() -> np.ndarray:
        """Hadamard-like gate for qutrits (Fourier transform)"""
        omega = np.exp(2j * np.pi / 3)
        H3 = np.array([
            [1, 1, 1],
            [1, omega, omega**2],
            [1, omega**2, omega**4]
        ]) / np.sqrt(3)
        return H3
    
    @staticmethod
    def phase_gate(theta: float) -> np.ndarray:
        """Phase rotation for qutrit"""
        return np.diag([1, np.exp(1j * theta), np.exp(2j * theta)])
    
    @staticmethod
    def ternary_rotation(alpha: float, beta: float) -> np.ndarray:
        """Ternary-specific rotation"""
        return np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha) * np.cos(beta), -np.sin(beta)],
            [0, np.sin(beta), np.cos(beta)]
        ])

class QutritDefense:
    """Qutrit-based quantum defense"""
    
    def __init__(self):
        self.gates = QutritGates()
        self.results = []
    
    def classical_to_qutrit(self, ternary_state: int) -> QutritState:
        """Convert classical ternary state to qutrit"""
        # Map: -1 → |0⟩, 0 → |1⟩, +1 → |2⟩
        state_map = {-1: 0, 0: 1, 1: 2}
        basis_idx = state_map[ternary_state]
        
        amplitudes = np.zeros(3, dtype=complex)
        amplitudes[basis_idx] = 1.0
        
        return QutritState(amplitudes, ternary_state)
    
    def apply_superposition(self, qutrit: QutritState) -> np.ndarray:
        """Apply Hadamard-like gate for ternary superposition"""
        H3 = self.gates.hadamard_qutrit()
        return H3 @ qutrit.amplitudes
    
    def measure_entropy(self, amplitudes: np.ndarray) -> float:
        """Measure quantum entropy"""
        probs = np.abs(amplitudes) ** 2
        probs = probs[probs > 1e-10]
        return -np.sum(probs * np.log2(probs))
    
    def measure_interference(self, amplitudes: np.ndarray) -> float:
        """Measure interference pattern"""
        phases = np.angle(amplitudes)
        phase_diff = np.abs(phases[1] - phases[0]) + np.abs(phases[2] - phases[1])
        return phase_diff / (2 * np.pi)
    
    def measure_entanglement(self, amp1: np.ndarray, amp2: np.ndarray) -> float:
        """Measure entanglement between two qutrits"""
        # Simplified: correlation between probability distributions
        prob1 = np.abs(amp1) ** 2
        prob2 = np.abs(amp2) ** 2
        correlation = np.abs(np.sum(prob1 * prob2))
        return 1 - correlation  # Higher = more entangled
    
    def qutrit_to_ternary(self, amplitudes: np.ndarray) -> Tuple[int, float]:
        """Map qutrit state back to ternary"""
        # Extract real parts and find dominant amplitude
        real_parts = np.real(amplitudes)
        probs = np.abs(amplitudes) ** 2
        dominant_idx = np.argmax(probs)
        dominant_real = real_parts[dominant_idx]
        confidence = probs[dominant_idx]
        
        # Map based on dominant amplitude
        if dominant_real < -0.3:
            return -1, confidence  # Shadow
        elif dominant_real > 0.3:
            return 1, confidence   # Evidence
        else:
            return 0, confidence   # Normal
    
    def process(self, ternary_state: int, attack_strength: float = 0.5) -> DefenseResult:
        """Full qutrit processing pipeline"""
        start_time = time.time()
        
        # 1. Classical ternary → qutrit
        qutrit = self.classical_to_qutrit(ternary_state)
        
        # 2. Apply superposition
        superposed = self.apply_superposition(qutrit)
        
        # 3. Apply attack-dependent phase rotation
        phase_gate = self.gates.phase_gate(attack_strength * np.pi)
        processed = phase_gate @ superposed
        
        # 4. Apply ternary rotation
        rotation = self.gates.ternary_rotation(attack_strength * 0.5, attack_strength * 0.3)
        processed = rotation @ processed
        
        # 5. Measure quantum properties
        entropy = self.measure_entropy(processed)
        interference = self.measure_interference(processed)
        
        # Create second qutrit for entanglement measurement
        qutrit2 = self.classical_to_qutrit(0)
        superposed2 = self.apply_superposition(qutrit2)
        entanglement = self.measure_entanglement(processed, superposed2)
        
        # 6. Qutrit → ternary
        output_state, confidence = self.qutrit_to_ternary(processed)
        
        processing_time = time.time() - start_time
        
        result = DefenseResult(
            input_state=ternary_state,
            qutrit_amplitudes=processed,
            entropy=entropy,
            interference=interference,
            entanglement=entanglement,
            output_state=output_state,
            processing_time=processing_time,
            confidence=confidence
        )
        
        self.results.append(result)
        return result

class QubitDefense:
    """Qubit-based quantum defense (for comparison)"""
    
    def __init__(self):
        self.results = []
    
    def classical_to_qubit(self, ternary_state: int) -> np.ndarray:
        """Convert ternary to qubit (lossy)"""
        # Map: -1,0 → |0⟩, +1 → |1⟩
        if ternary_state <= 0:
            return np.array([1.0, 0.0], dtype=complex)
        else:
            return np.array([0.0, 1.0], dtype=complex)
    
    def hadamard_qubit(self) -> np.ndarray:
        """Standard Hadamard gate"""
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    def process(self, ternary_state: int, attack_strength: float = 0.5) -> DefenseResult:
        """Qubit processing pipeline"""
        start_time = time.time()
        
        # Convert to qubit (loses ternary information)
        qubit = self.classical_to_qubit(ternary_state)
        
        # Apply Hadamard
        H = self.hadamard_qubit()
        superposed = H @ qubit
        
        # Phase rotation
        phase = np.array([[1, 0], [0, np.exp(1j * attack_strength * np.pi)]])
        processed = phase @ superposed
        
        # Measure properties (limited to 2 dimensions)
        probs = np.abs(processed) ** 2
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        interference = np.abs(np.angle(processed[1]) - np.angle(processed[0])) / np.pi
        entanglement = 0.5  # Simplified
        
        # Map back to ternary (lossy)
        if probs[0] > 0.7:
            output_state = -1
        elif probs[1] > 0.7:
            output_state = 1
        else:
            output_state = 0
        
        confidence = max(probs)
        processing_time = time.time() - start_time
        
        # Pad to 3 dimensions for comparison
        padded_amps = np.zeros(3, dtype=complex)
        padded_amps[:2] = processed
        
        result = DefenseResult(
            input_state=ternary_state,
            qutrit_amplitudes=padded_amps,
            entropy=entropy,
            interference=interference,
            entanglement=entanglement,
            output_state=output_state,
            processing_time=processing_time,
            confidence=confidence
        )
        
        self.results.append(result)
        return result

def compare_defenses():
    """Compare qutrit vs qubit defense effectiveness"""
    print("🌌 QUTRIT vs QUBIT DEFENSE COMPARISON")
    print("="*70)
    
    qutrit_defense = QutritDefense()
    qubit_defense = QubitDefense()
    
    # Test scenarios
    scenarios = [
        (-1, 0.3, "Shadow threat (low)"),
        (-1, 0.7, "Shadow threat (high)"),
        (0, 0.5, "Normal traffic"),
        (1, 0.4, "Evidence (medium)"),
        (1, 0.9, "Evidence (critical)"),
    ]
    
    print("\n📊 PROCESSING ATTACKS...\n")
    
    for ternary_in, attack_str, desc in scenarios:
        print(f"{'='*70}")
        print(f"🎯 Scenario: {desc}")
        print(f"   Input: {['Shadow', 'Normal', 'Evidence'][ternary_in]} ({ternary_in})")
        print(f"   Attack Strength: {attack_str:.2f}")
        
        # Process with qutrit
        print(f"\n   🔷 QUTRIT DEFENSE:")
        qutrit_result = qutrit_defense.process(ternary_in, attack_str)
        print(f"      Amplitudes: {qutrit_result.qutrit_amplitudes}")
        print(f"      Entropy: {qutrit_result.entropy:.4f}")
        print(f"      Interference: {qutrit_result.interference:.4f}")
        print(f"      Entanglement: {qutrit_result.entanglement:.4f}")
        print(f"      Output: {['Shadow', 'Normal', 'Evidence'][qutrit_result.output_state]} ({qutrit_result.output_state})")
        print(f"      Confidence: {qutrit_result.confidence:.4f}")
        
        # Process with qubit
        print(f"\n   🔶 QUBIT DEFENSE:")
        qubit_result = qubit_defense.process(ternary_in, attack_str)
        print(f"      Amplitudes: {qubit_result.qutrit_amplitudes[:2]}")
        print(f"      Entropy: {qubit_result.entropy:.4f}")
        print(f"      Interference: {qubit_result.interference:.4f}")
        print(f"      Output: {['Shadow', 'Normal', 'Evidence'][qubit_result.output_state]} ({qubit_result.output_state})")
        print(f"      Confidence: {qubit_result.confidence:.4f}")
        print()
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"📊 EFFECTIVENESS ANALYSIS")
    print(f"{'='*70}")
    
    # Accuracy (correct ternary preservation)
    qutrit_correct = sum(1 for r in qutrit_defense.results if r.input_state == r.output_state)
    qubit_correct = sum(1 for r in qubit_defense.results if r.input_state == r.output_state)
    
    print(f"\n🎯 State Preservation:")
    print(f"   Qutrit: {qutrit_correct}/{len(scenarios)} ({qutrit_correct/len(scenarios)*100:.1f}%)")
    print(f"   Qubit:  {qubit_correct}/{len(scenarios)} ({qubit_correct/len(scenarios)*100:.1f}%)")
    
    # Average entropy (information capacity)
    qutrit_entropy = np.mean([r.entropy for r in qutrit_defense.results])
    qubit_entropy = np.mean([r.entropy for r in qubit_defense.results])
    
    print(f"\n📈 Information Capacity (Entropy):")
    print(f"   Qutrit: {qutrit_entropy:.4f}")
    print(f"   Qubit:  {qubit_entropy:.4f}")
    print(f"   Advantage: {(qutrit_entropy/qubit_entropy - 1)*100:.1f}% more information")
    
    # Average confidence
    qutrit_conf = np.mean([r.confidence for r in qutrit_defense.results])
    qubit_conf = np.mean([r.confidence for r in qubit_defense.results])
    
    print(f"\n🎯 Decision Confidence:")
    print(f"   Qutrit: {qutrit_conf:.4f}")
    print(f"   Qubit:  {qubit_conf:.4f}")
    
    # Processing time
    qutrit_time = np.mean([r.processing_time for r in qutrit_defense.results]) * 1000
    qubit_time = np.mean([r.processing_time for r in qubit_defense.results]) * 1000
    
    print(f"\n⏱️  Processing Speed:")
    print(f"   Qutrit: {qutrit_time:.3f}ms")
    print(f"   Qubit:  {qubit_time:.3f}ms")
    
    # Verdict
    print(f"\n{'='*70}")
    print(f"🏆 VERDICT")
    print(f"{'='*70}")
    
    qutrit_score = (qutrit_correct/len(scenarios)) * 0.4 + qutrit_entropy * 0.3 + qutrit_conf * 0.3
    qubit_score = (qubit_correct/len(scenarios)) * 0.4 + qubit_entropy * 0.3 + qubit_conf * 0.3
    
    print(f"\nOverall Effectiveness Score:")
    print(f"   🔷 Qutrit: {qutrit_score:.4f}")
    print(f"   🔶 Qubit:  {qubit_score:.4f}")
    
    if qutrit_score > qubit_score:
        advantage = (qutrit_score / qubit_score - 1) * 100
        print(f"\n✅ QUTRIT DEFENSE IS MORE EFFECTIVE")
        print(f"   {advantage:.1f}% better than qubit-based defense")
        print(f"\n   Reasons:")
        print(f"   • Native ternary state representation")
        print(f"   • Higher information capacity (log₂3 vs log₂2)")
        print(f"   • No lossy conversion from ternary")
        print(f"   • Richer quantum interference patterns")
    else:
        print(f"\n⚠️  Results inconclusive")
    
    return qutrit_defense, qubit_defense

def visualize_comparison(qutrit_defense: QutritDefense, qubit_defense: QubitDefense):
    """Visualize qutrit vs qubit comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = range(len(qutrit_defense.results))
    
    # 1. State preservation
    qutrit_states = [r.output_state for r in qutrit_defense.results]
    qubit_states = [r.output_state for r in qubit_defense.results]
    input_states = [r.input_state for r in qutrit_defense.results]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    axes[0, 0].bar(x - width, input_states, width, label='Input', color='gray', alpha=0.7)
    axes[0, 0].bar(x, qutrit_states, width, label='Qutrit', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width, qubit_states, width, label='Qubit', color='orange', alpha=0.7)
    axes[0, 0].set_xlabel('Scenario', fontsize=11)
    axes[0, 0].set_ylabel('Ternary State', fontsize=11)
    axes[0, 0].set_title('State Preservation', fontsize=12, fontweight='bold')
    axes[0, 0].set_yticks([-1, 0, 1])
    axes[0, 0].set_yticklabels(['Shadow', 'Normal', 'Evidence'])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Entropy comparison
    qutrit_entropy = [r.entropy for r in qutrit_defense.results]
    qubit_entropy = [r.entropy for r in qubit_defense.results]
    
    axes[0, 1].plot(scenarios, qutrit_entropy, 'o-', label='Qutrit', color='blue', linewidth=2, markersize=8)
    axes[0, 1].plot(scenarios, qubit_entropy, 's-', label='Qubit', color='orange', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Scenario', fontsize=11)
    axes[0, 1].set_ylabel('Quantum Entropy', fontsize=11)
    axes[0, 1].set_title('Information Capacity', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confidence comparison
    qutrit_conf = [r.confidence for r in qutrit_defense.results]
    qubit_conf = [r.confidence for r in qubit_defense.results]
    
    axes[1, 0].bar(x - width/2, qutrit_conf, width, label='Qutrit', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, qubit_conf, width, label='Qubit', color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Scenario', fontsize=11)
    axes[1, 0].set_ylabel('Confidence', fontsize=11)
    axes[1, 0].set_title('Decision Confidence', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Overall metrics
    metrics = ['Accuracy', 'Entropy', 'Confidence']
    qutrit_metrics = [
        sum(1 for r in qutrit_defense.results if r.input_state == r.output_state) / len(qutrit_defense.results),
        np.mean(qutrit_entropy) / np.log2(3),  # Normalize
        np.mean(qutrit_conf)
    ]
    qubit_metrics = [
        sum(1 for r in qubit_defense.results if r.input_state == r.output_state) / len(qubit_defense.results),
        np.mean(qubit_entropy) / np.log2(2),  # Normalize
        np.mean(qubit_conf)
    ]
    
    x_metrics = np.arange(len(metrics))
    axes[1, 1].bar(x_metrics - width/2, qutrit_metrics, width, label='Qutrit', color='blue', alpha=0.7)
    axes[1, 1].bar(x_metrics + width/2, qubit_metrics, width, label='Qubit', color='orange', alpha=0.7)
    axes[1, 1].set_xticks(x_metrics)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylabel('Normalized Score', fontsize=11)
    axes[1, 1].set_title('Overall Effectiveness', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('qutrit_vs_qubit_defense.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    qutrit_def, qubit_def = compare_defenses()
    print("\n🎨 Generating visualization...")
    visualize_comparison(qutrit_def, qubit_def)
