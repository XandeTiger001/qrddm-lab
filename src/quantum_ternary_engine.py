import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import json

@dataclass
class QuantumAttack:
    """Attack with quantum signature"""
    id: int
    position: Tuple[int, int]
    M0: float
    qsig: str
    qfluc: float
    layer: int
    phi: float
    dt_prime: float
    G_adapted: float
    alpha_adapted: float
    M_final: float
    P_success: float
    ternary_state: int

class QuantumTernaryEngine:
    def __init__(self, n_layers=3, n_qubits=8):
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.attacks = []
        self.attack_counter = 0
        
    def hadamard_measure(self, seed: int) -> str:
        """Generate quantum signature via Hadamard-like measurement"""
        np.random.seed(seed)
        return ''.join(str(np.random.randint(0, 2)) for _ in range(self.n_qubits))
    
    def quantum_fluctuation(self, qsig: str) -> float:
        """Calculate quantum fluctuation (average of 1s)"""
        return sum(int(b) for b in qsig) / len(qsig)
    
    def calculate_phi(self, M0: float, layer: int) -> float:
        """Gravitational potential: φ = -GM/r"""
        r = layer + 1
        return -2.0 * M0 / r
    
    def time_dilation(self, phi: float) -> float:
        """Time dilation: dt' = dt * sqrt(1 + 2φ/c²), c=1"""
        return np.sqrt(max(0.01, 1 + 2 * phi))
    
    def adapt_parameters(self, G_base: float, alpha_base: float, qfluc: float) -> Tuple[float, float]:
        """Modulate G and alpha by quantum fluctuation"""
        G_adapted = G_base * (1 + 0.5 * qfluc)
        alpha_adapted = alpha_base * (1 - 0.3 * qfluc)
        return G_adapted, alpha_adapted
    
    def cnot_entangle(self, control_state: str, target_state: str) -> Tuple[str, str]:
        """CNOT gate: flip target if control is 1"""
        result_target = ''.join(
            str(int(t) ^ int(c)) if c == '1' else t
            for c, t in zip(control_state, target_state)
        )
        return control_state, result_target
    
    def pauli_z_rotate(self, state: str) -> str:
        """Pauli-Z: rotate phase (flip sign of |1⟩ states)"""
        # In computational basis, Z doesn't change bitstring but adds phase
        # We simulate by inverting the bitstring to show field rotation
        return ''.join('1' if b == '0' else '0' for b in state)
    
    def layer_propagation(self, M0: float, qfluc: float, layer: int) -> float:
        """Propagate mass through layers with quantum modulation"""
        G_base, alpha_base = 1.0, 0.8
        G, alpha = self.adapt_parameters(G_base, alpha_base, qfluc)
        
        phi = self.calculate_phi(M0, layer)
        dt_prime = self.time_dilation(phi)
        
        # M evolves: M_i+1 = M_i * exp(-alpha * dt')
        decay = np.exp(-alpha * dt_prime)
        return M0 * decay
    
    def calculate_success_probability(self, M_final: float, qfluc: float) -> float:
        """P_success based on final mass and quantum uncertainty"""
        base_prob = 1 / (1 + np.exp(-5 * (M_final - 0.5)))
        quantum_boost = 0.2 * qfluc
        return min(0.99, base_prob + quantum_boost)
    
    def classify_ternary(self, M_final: float, P_success: float) -> int:
        """Ternary classification: -1 (shadow), 0 (normal), +1 (evidence)"""
        if P_success > 0.7 and M_final > 0.6:
            return 1  # Evidence
        elif P_success > 0.3 or M_final > 0.3:
            return -1  # Shadow
        else:
            return 0  # Normal
    
    def process_attack(self, position: Tuple[int, int], M0: float) -> QuantumAttack:
        """Full quantum-enhanced attack processing"""
        attack_id = self.attack_counter
        self.attack_counter += 1
        
        # Generate quantum signature
        qsig = self.hadamard_measure(seed=attack_id * 42 + int(M0 * 1000))
        qfluc = self.quantum_fluctuation(qsig)
        
        # Process through layers
        M_current = M0
        logs = []
        current_qsig = qsig
        
        for layer in range(self.n_layers):
            phi = self.calculate_phi(M_current, layer)
            dt_prime = self.time_dilation(phi)
            G_adapted, alpha_adapted = self.adapt_parameters(1.0, 0.8, qfluc)
            
            M_next = self.layer_propagation(M_current, qfluc, layer)
            
            # CNOT: Entangle adjacent layers
            if layer < self.n_layers - 1:
                next_qsig = self.hadamard_measure(seed=attack_id * 100 + layer)
                current_qsig, entangled_next = self.cnot_entangle(current_qsig, next_qsig)
                qfluc_entangled = self.quantum_fluctuation(entangled_next)
            else:
                qfluc_entangled = qfluc
            
            # Pauli-Z: Rotate field if attack passes threshold
            attack_passes = M_next > 0.5
            if attack_passes:
                current_qsig = self.pauli_z_rotate(current_qsig)
                qfluc = self.quantum_fluctuation(current_qsig)
            
            logs.append({
                'layer': layer,
                'M': M_current,
                'phi': phi,
                'dt_prime': dt_prime,
                'G': G_adapted,
                'alpha': alpha_adapted,
                'qsig': current_qsig,
                'cnot_applied': layer < self.n_layers - 1,
                'pauli_z_applied': attack_passes
            })
            
            M_current = M_next
        
        M_final = M_current
        P_success = self.calculate_success_probability(M_final, qfluc)
        ternary_state = self.classify_ternary(M_final, P_success)
        
        attack = QuantumAttack(
            id=attack_id,
            position=position,
            M0=M0,
            qsig=logs[-1]['qsig'],  # Final quantum state after gates
            qfluc=qfluc,
            layer=self.n_layers - 1,
            phi=logs[-1]['phi'],
            dt_prime=logs[-1]['dt_prime'],
            G_adapted=logs[-1]['G'],
            alpha_adapted=logs[-1]['alpha'],
            M_final=M_final,
            P_success=P_success,
            ternary_state=ternary_state
        )
        
        # Store layer logs for detailed analysis
        attack.layer_logs = logs
        self.attacks.append(attack)
        return attack
    
    def generate_log(self, attack: QuantumAttack) -> str:
        """Generate formatted log entry"""
        state_symbols = {-1: "🌫️ SHADOW", 0: "✅ NORMAL", 1: "🚨 EVIDENCE"}
        
        log = f"""
╔══════════════════════════════════════════════════════════════╗
║ QUANTUM ATTACK ANALYSIS #{attack.id:03d}                           
╚══════════════════════════════════════════════════════════════╝
📍 Position: {attack.position}
🔢 M₀: {attack.M0:.4f} → M_final: {attack.M_final:.4f}

🌀 QUANTUM SIGNATURE (Final)
   qsig:  {attack.qsig}
   qfluc: {attack.qfluc:.4f} (quantum fluctuation)

⚡ FIELD PARAMETERS (Layer {attack.layer})
   φ:      {attack.phi:.4f} (gravitational potential)
   dt':    {attack.dt_prime:.4f} (time dilation)
   G:      {attack.G_adapted:.4f} (adapted coupling)
   α:      {attack.alpha_adapted:.4f} (adapted decay)

🔗 QUANTUM GATES APPLIED"""
        
        if hasattr(attack, 'layer_logs'):
            for layer_log in attack.layer_logs:
                cnot_icon = "🔗" if layer_log.get('cnot_applied') else "  "
                z_icon = "🔄" if layer_log.get('pauli_z_applied') else "  "
                log += f"""
   Layer {layer_log['layer']}: {cnot_icon} CNOT  {z_icon} Pauli-Z  |  qsig: {layer_log['qsig'][:8]}"""
        
        log += f"""

🎯 THREAT ASSESSMENT
   P_success: {attack.P_success:.4f}
   State:     {state_symbols[attack.ternary_state]}
"""
        return log
    
    def plot_results(self):
        """Plot M0 vs M_final colored by ternary classification"""
        if not self.attacks:
            print("No attacks to plot")
            return
        
        M0_vals = [a.M0 for a in self.attacks]
        M_final_vals = [a.M_final for a in self.attacks]
        states = [a.ternary_state for a in self.attacks]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot colored by ternary state
        colors = {-1: 'orange', 0: 'green', 1: 'red'}
        labels = {-1: 'Shadow', 0: 'Normal', 1: 'Evidence'}
        
        for state in [-1, 0, 1]:
            mask = [s == state for s in states]
            M0_filtered = [m for m, include in zip(M0_vals, mask) if include]
            M_final_filtered = [m for m, include in zip(M_final_vals, mask) if include]
            ax1.scatter(M0_filtered, M_final_filtered, c=colors[state], 
                       label=labels[state], s=100, alpha=0.7, edgecolors='black')
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='M₀=M_final')
        ax1.set_xlabel('M₀ (Initial Mass)', fontsize=12)
        ax1.set_ylabel('M_final (Final Mass)', fontsize=12)
        ax1.set_title('Quantum Attack Evolution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Quantum fluctuation vs P_success
        qfluc_vals = [a.qfluc for a in self.attacks]
        P_vals = [a.P_success for a in self.attacks]
        
        for state in [-1, 0, 1]:
            mask = [s == state for s in states]
            qf = [q for q, include in zip(qfluc_vals, mask) if include]
            pv = [p for p, include in zip(P_vals, mask) if include]
            ax2.scatter(qf, pv, c=colors[state], label=labels[state], 
                       s=100, alpha=0.7, edgecolors='black')
        
        ax2.set_xlabel('Quantum Fluctuation', fontsize=12)
        ax2.set_ylabel('P_success', fontsize=12)
        ax2.set_title('Quantum Uncertainty vs Success Probability', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quantum_ternary_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

def demo():
    """Demonstration of quantum ternary engine"""
    print("🌌 QUANTUM TERNARY ENGINE - Cybersecurity Analysis")
    print("=" * 70)
    
    engine = QuantumTernaryEngine(n_layers=3, n_qubits=8)
    
    # Simulate various attack scenarios
    scenarios = [
        ((10, 10), 0.9, "High intensity SQL injection"),
        ((20, 15), 0.7, "Medium XSS attack"),
        ((30, 25), 0.4, "Low suspicious probe"),
        ((15, 30), 0.85, "Brute force attempt"),
        ((40, 40), 0.3, "Normal traffic"),
        ((25, 35), 0.6, "Path traversal"),
        ((5, 45), 0.95, "Critical exploit"),
        ((35, 10), 0.2, "Benign request"),
    ]
    
    for pos, M0, desc in scenarios:
        attack = engine.process_attack(pos, M0)
        print(engine.generate_log(attack))
        print(f"📝 Scenario: {desc}\n")
    
    # Statistics
    print("\n" + "=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70)
    
    evidence = sum(1 for a in engine.attacks if a.ternary_state == 1)
    shadow = sum(1 for a in engine.attacks if a.ternary_state == -1)
    normal = sum(1 for a in engine.attacks if a.ternary_state == 0)
    
    print(f"🚨 Evidence (Confirmed Threats): {evidence}")
    print(f"🌫️  Shadow (Suspicious):          {shadow}")
    print(f"✅ Normal (Benign):              {normal}")
    print(f"\nTotal Attacks Analyzed: {len(engine.attacks)}")
    
    avg_qfluc = np.mean([a.qfluc for a in engine.attacks])
    avg_decay = np.mean([a.M0 - a.M_final for a in engine.attacks])
    
    print(f"\n⚛️  Average Quantum Fluctuation: {avg_qfluc:.4f}")
    print(f"📉 Average Mass Decay: {avg_decay:.4f}")
    
    # Generate plot
    print("\n🎨 Generating visualization...")
    engine.plot_results()
    
    return engine

if __name__ == '__main__':
    engine = demo()
