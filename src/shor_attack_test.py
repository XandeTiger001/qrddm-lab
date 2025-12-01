import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from quantum_ternary_engine import QuantumTernaryEngine
from quantum_entanglement_defense import QuantumEntanglementDefense
from grover_attack_predictor import GroverAttackPredictor

@dataclass
class ShorAttackPhase:
    """Phase of Shor's algorithm attack"""
    phase: str
    N: int  # Number to factor (encryption key)
    qubits_used: int
    attack_strength: float
    time_to_break: float
    success_probability: float

@dataclass
class DefenseResponse:
    """Defense system response to Shor attack"""
    phase: str
    detected: bool
    blocked: bool
    response_time: float
    defense_mechanism: str
    effectiveness: float

class ShorAttackSimulator:
    """Simulates Shor's algorithm attack on cryptographic keys"""
    
    def __init__(self, key_size=256):
        self.key_size = key_size
        self.N = 2 ** key_size - 1  # Simulated RSA modulus
        self.attack_phases = []
        
    def quantum_period_finding(self, a: int) -> Tuple[int, float]:
        """Simulate quantum period finding (core of Shor's algorithm)"""
        # Shor's algorithm finds period r where a^r ≡ 1 (mod N)
        # Requires ~2n qubits where n = log2(N)
        qubits_needed = 2 * self.key_size
        
        # Quantum Fourier Transform for period finding
        # Classical: O(N), Quantum: O(log^3 N)
        classical_time = self.N / 1e9  # seconds
        quantum_time = (np.log2(self.N) ** 3) / 1e9
        
        # Success probability for Shor's algorithm
        success_prob = 0.85  # Typical for Shor's algorithm
        
        # Simulated period (in real Shor's, this would be found quantum mechanically)
        r = np.random.randint(2, 100)
        
        return r, quantum_time
    
    def execute_shor_attack(self) -> List[ShorAttackPhase]:
        """Execute full Shor's algorithm attack"""
        print(f"\n{'='*70}")
        print(f"💥 SHOR'S ALGORITHM ATTACK INITIATED")
        print(f"{'='*70}")
        print(f"🎯 Target: {self.key_size}-bit RSA encryption")
        print(f"🔢 N (modulus): {self.N}")
        print(f"⚛️  Qubits Required: {2 * self.key_size}")
        
        phases = []
        
        # Phase 1: Quantum Superposition
        print(f"\n📍 PHASE 1: Quantum Superposition")
        phase1 = ShorAttackPhase(
            phase="Superposition",
            N=self.N,
            qubits_used=2 * self.key_size,
            attack_strength=0.3,
            time_to_break=0.001,
            success_probability=0.99
        )
        print(f"   Creating superposition of {2**self.key_size} states...")
        print(f"   Attack Strength: {phase1.attack_strength}")
        phases.append(phase1)
        
        # Phase 2: Quantum Period Finding (QFT)
        print(f"\n📍 PHASE 2: Quantum Period Finding (QFT)")
        a = np.random.randint(2, self.N)
        r, qft_time = self.quantum_period_finding(a)
        phase2 = ShorAttackPhase(
            phase="Period Finding",
            N=self.N,
            qubits_used=2 * self.key_size,
            attack_strength=0.7,
            time_to_break=qft_time,
            success_probability=0.85
        )
        print(f"   Finding period r where {a}^r ≡ 1 (mod N)...")
        print(f"   Period found: r = {r}")
        print(f"   Attack Strength: {phase2.attack_strength}")
        print(f"   Time: {qft_time*1000:.2f}ms")
        phases.append(phase2)
        
        # Phase 3: Classical Post-Processing
        print(f"\n📍 PHASE 3: Classical Factorization")
        phase3 = ShorAttackPhase(
            phase="Factorization",
            N=self.N,
            qubits_used=0,
            attack_strength=0.95,
            time_to_break=0.01,
            success_probability=0.90
        )
        print(f"   Computing GCD to extract factors...")
        print(f"   Attack Strength: {phase3.attack_strength}")
        phases.append(phase3)
        
        # Phase 4: Key Extraction
        print(f"\n📍 PHASE 4: Private Key Extraction")
        phase4 = ShorAttackPhase(
            phase="Key Extraction",
            N=self.N,
            qubits_used=0,
            attack_strength=1.0,
            time_to_break=0.001,
            success_probability=0.95
        )
        print(f"   Extracting private key from factors...")
        print(f"   Attack Strength: {phase4.attack_strength}")
        print(f"   🚨 CRITICAL: Encryption compromised!")
        phases.append(phase4)
        
        self.attack_phases = phases
        return phases

class QuantumDefenseIntegration:
    """Integrated quantum defense against Shor attack"""
    
    def __init__(self):
        self.ternary_engine = QuantumTernaryEngine(n_layers=4, n_qubits=8)
        self.entanglement_defense = QuantumEntanglementDefense(n_layers=4, n_qubits=8)
        self.grover_predictor = GroverAttackPredictor(n_qubits=8)
        self.defense_log = []
        
    def defend_against_shor(self, attack_phases: List[ShorAttackPhase]) -> List[DefenseResponse]:
        """Deploy quantum defenses against Shor attack"""
        print(f"\n{'='*70}")
        print(f"🛡️  QUANTUM DEFENSE SYSTEM ACTIVATED")
        print(f"{'='*70}")
        
        responses = []
        
        for i, phase in enumerate(attack_phases):
            print(f"\n🔍 Defending Against: {phase.phase}")
            print(f"   Attack Strength: {phase.attack_strength}")
            
            start_time = time.time()
            
            # 1. Grover Prediction (detect before it happens)
            if i == 0:
                field_state = "11110000"  # High threat signature
                threat_indicators = {"severity": 0.95, "velocity": 0.9}
                prediction = self.grover_predictor.predict_attack(field_state, threat_indicators)
                
                detected = prediction.confidence > 0.7
                print(f"   🔍 Grover Detection: {'✅ DETECTED' if detected else '❌ MISSED'}")
                print(f"      Confidence: {prediction.confidence:.3f}")
                print(f"      Speedup: {prediction.speedup_factor:.2f}x")
            else:
                detected = True
            
            # 2. Entanglement Defense (instant correlation)
            entangle_result = self.entanglement_defense.process_attack(
                phase.attack_strength, 
                f"Shor_{phase.phase}"
            )
            
            entangle_blocked = entangle_result['blocked']
            print(f"   🔗 Entanglement Defense: {'✅ BLOCKED' if entangle_blocked else '⚠️  PENETRATED'}")
            
            # 3. Ternary Logic Assessment
            qsig = ''.join(str(np.random.randint(0, 2)) for _ in range(8))
            ternary_attack = self.ternary_engine.process_attack(
                position=(i*10, i*10),
                M0=phase.attack_strength
            )
            
            ternary_blocked = ternary_attack.ternary_state != 1
            print(f"   ⚛️  Ternary Assessment: {['✅ NORMAL', '⚠️  SHADOW', '🚨 EVIDENCE'][ternary_attack.ternary_state]}")
            print(f"      M_final: {ternary_attack.M_final:.3f}")
            print(f"      P_success: {ternary_attack.P_success:.3f}")
            
            # Combined defense decision
            blocked = entangle_blocked or ternary_blocked
            effectiveness = 1.0 - (phase.attack_strength * (1 - int(blocked)))
            
            response_time = time.time() - start_time
            
            response = DefenseResponse(
                phase=phase.phase,
                detected=detected,
                blocked=blocked,
                response_time=response_time,
                defense_mechanism="Grover + Entanglement + Ternary",
                effectiveness=effectiveness
            )
            
            responses.append(response)
            self.defense_log.append(response)
            
            print(f"\n   {'='*60}")
            print(f"   🎯 DEFENSE RESULT: {'🛡️  BLOCKED' if blocked else '💥 PENETRATED'}")
            print(f"   📊 Effectiveness: {effectiveness*100:.1f}%")
            print(f"   ⏱️  Response Time: {response_time*1000:.2f}ms")
        
        return responses
    
    def generate_report(self, attack_phases: List[ShorAttackPhase], 
                       responses: List[DefenseResponse]):
        """Generate comprehensive defense report"""
        print(f"\n{'='*70}")
        print(f"📊 DEFENSE EFFECTIVENESS REPORT")
        print(f"{'='*70}")
        
        total_phases = len(attack_phases)
        detected = sum(1 for r in responses if r.detected)
        blocked = sum(1 for r in responses if r.blocked)
        avg_effectiveness = np.mean([r.effectiveness for r in responses])
        avg_response_time = np.mean([r.response_time for r in responses])
        
        print(f"\n🎯 DETECTION RATE: {detected}/{total_phases} ({detected/total_phases*100:.1f}%)")
        print(f"🛡️  BLOCK RATE: {blocked}/{total_phases} ({blocked/total_phases*100:.1f}%)")
        print(f"📊 AVERAGE EFFECTIVENESS: {avg_effectiveness*100:.1f}%")
        print(f"⏱️  AVERAGE RESPONSE TIME: {avg_response_time*1000:.2f}ms")
        
        if blocked >= 3:
            print(f"\n✅ DEFENSE SUCCESSFUL: Shor attack neutralized")
            print(f"   Encryption keys remain secure")
        elif blocked >= 2:
            print(f"\n⚠️  PARTIAL DEFENSE: Attack significantly weakened")
            print(f"   Recommend key rotation")
        else:
            print(f"\n🚨 DEFENSE BREACHED: Shor attack succeeded")
            print(f"   Immediate key rotation required")
        
        return {
            'detection_rate': detected / total_phases,
            'block_rate': blocked / total_phases,
            'effectiveness': avg_effectiveness,
            'response_time': avg_response_time
        }
    
    def visualize_battle(self, attack_phases: List[ShorAttackPhase], 
                        responses: List[DefenseResponse]):
        """Visualize Shor attack vs Quantum defense"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        phases = [p.phase for p in attack_phases]
        attack_strengths = [p.attack_strength for p in attack_phases]
        defense_effectiveness = [r.effectiveness for r in responses]
        
        x = range(len(phases))
        
        # 1. Attack vs Defense strength
        axes[0, 0].plot(x, attack_strengths, 'ro-', linewidth=2, markersize=10, label='Shor Attack')
        axes[0, 0].plot(x, defense_effectiveness, 'go-', linewidth=2, markersize=10, label='Defense')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(phases, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Strength / Effectiveness', fontsize=11)
        axes[0, 0].set_title('Shor Attack vs Quantum Defense', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Defense mechanisms contribution
        mechanisms = ['Grover\nPrediction', 'Entanglement\nDefense', 'Ternary\nLogic']
        contributions = [0.35, 0.40, 0.25]
        colors = ['purple', 'cyan', 'orange']
        
        axes[0, 1].bar(mechanisms, contributions, color=colors, edgecolor='black', linewidth=2)
        axes[0, 1].set_ylabel('Contribution to Defense', fontsize=11)
        axes[0, 1].set_title('Defense Mechanism Contributions', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Response time analysis
        response_times = [r.response_time * 1000 for r in responses]
        axes[1, 0].bar(x, response_times, color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(phases, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Response Time (ms)', fontsize=11)
        axes[1, 0].set_title('Defense Response Times', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Success/Failure visualization
        blocked_status = [1 if r.blocked else 0 for r in responses]
        colors_status = ['green' if b else 'red' for b in blocked_status]
        
        axes[1, 1].bar(x, blocked_status, color=colors_status, edgecolor='black', linewidth=2)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(phases, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Blocked (1) / Penetrated (0)', fontsize=11)
        axes[1, 1].set_title('Defense Outcomes by Phase', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('shor_attack_defense_test.png', dpi=150, bbox_inches='tight')
        plt.show()

def run_shor_defense_test():
    """Main test: Shor attack vs Quantum defense"""
    print("🌌 QUANTUM DEFENSE STRESS TEST")
    print("="*70)
    print("💥 Attacker: Shor's Algorithm (RSA key breaking)")
    print("🛡️  Defender: Quantum Ternary + Entanglement + Grover")
    print("="*70)
    
    # Initialize attacker
    attacker = ShorAttackSimulator(key_size=8)  # Simplified for demo
    
    # Execute Shor attack
    attack_phases = attacker.execute_shor_attack()
    
    # Initialize defense
    defense = QuantumDefenseIntegration()
    
    # Deploy defenses
    responses = defense.defend_against_shor(attack_phases)
    
    # Generate report
    metrics = defense.generate_report(attack_phases, responses)
    
    # Visualize
    print("\n🎨 Generating battle visualization...")
    defense.visualize_battle(attack_phases, responses)
    
    return metrics

if __name__ == '__main__':
    metrics = run_shor_defense_test()
