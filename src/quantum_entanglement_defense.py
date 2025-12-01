import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random

@dataclass
class EntangledLayer:
    """Defense layer with quantum entanglement"""
    id: int
    state: str  # quantum bitstring
    M_value: float
    response_time: float
    entangled_with: int
    is_primary: bool  # which layer responds first (hidden from attacker)

@dataclass
class EntanglementEvent:
    """Record of entanglement correlation"""
    attack_id: int
    layer_a: int
    layer_b: int
    state_a_before: str
    state_b_before: str
    state_a_after: str
    state_b_after: str
    correlation: float
    primary_responder: int
    response_delta_t: float

class QuantumEntanglementDefense:
    def __init__(self, n_layers=4, n_qubits=8):
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.layers = []
        self.entanglement_pairs = []
        self.events = []
        self.attack_counter = 0
        
        self._initialize_entangled_layers()
    
    def _initialize_entangled_layers(self):
        """Create entangled layer pairs"""
        for i in range(self.n_layers):
            state = ''.join(str(random.randint(0, 1)) for _ in range(self.n_qubits))
            layer = EntangledLayer(
                id=i,
                state=state,
                M_value=0.0,
                response_time=0.0,
                entangled_with=-1,
                is_primary=False
            )
            self.layers.append(layer)
        
        # Create entanglement pairs (0-1, 2-3, etc.)
        for i in range(0, self.n_layers - 1, 2):
            self.layers[i].entangled_with = i + 1
            self.layers[i + 1].entangled_with = i
            # Randomly assign primary responder (hidden from attacker)
            primary = random.choice([i, i + 1])
            self.layers[primary].is_primary = True
            self.entanglement_pairs.append((i, i + 1))
    
    def measure_correlation(self, state_a: str, state_b: str) -> float:
        """Calculate quantum correlation between entangled states"""
        matches = sum(a == b for a, b in zip(state_a, state_b))
        return matches / len(state_a)
    
    def apply_entanglement(self, layer_a: EntangledLayer, layer_b: EntangledLayer, 
                          attack_strength: float) -> Tuple[str, str, int]:
        """Apply entanglement: changing one instantly affects the other"""
        # Store original states
        state_a_orig = layer_a.state
        state_b_orig = layer_b.state
        
        # Determine which layer responds first (hidden from attacker)
        if layer_a.is_primary:
            primary_id = layer_a.id
            # Primary layer responds to attack
            new_state_a = self._respond_to_attack(state_a_orig, attack_strength)
            # Entangled layer INSTANTLY correlates (no time delay)
            new_state_b = self._entangled_response(new_state_a, state_b_orig)
        else:
            primary_id = layer_b.id
            # Primary layer responds to attack
            new_state_b = self._respond_to_attack(state_b_orig, attack_strength)
            # Entangled layer INSTANTLY correlates
            new_state_a = self._entangled_response(new_state_b, state_a_orig)
        
        return new_state_a, new_state_b, primary_id
    
    def _respond_to_attack(self, state: str, attack_strength: float) -> str:
        """Layer responds to attack by flipping qubits"""
        n_flips = int(attack_strength * len(state))
        state_list = list(state)
        flip_positions = random.sample(range(len(state)), n_flips)
        for pos in flip_positions:
            state_list[pos] = '1' if state_list[pos] == '0' else '0'
        return ''.join(state_list)
    
    def _entangled_response(self, primary_state: str, secondary_state: str) -> str:
        """Entangled layer responds instantly based on primary layer change"""
        # CNOT-like operation: secondary correlates with primary
        new_state = ''.join(
            str(int(s) ^ int(p)) if random.random() > 0.3 else s
            for p, s in zip(primary_state, secondary_state)
        )
        return new_state
    
    def process_attack(self, M0: float, attack_vector: str = "penetration") -> Dict:
        """Process attack through entangled defense layers"""
        attack_id = self.attack_counter
        self.attack_counter += 1
        
        print(f"\n{'='*70}")
        print(f"🎯 ATTACK #{attack_id:03d} | M₀={M0:.3f} | Vector: {attack_vector}")
        print(f"{'='*70}")
        
        M_current = M0
        attack_strength = M0
        
        for pair_idx, (i, j) in enumerate(self.entanglement_pairs):
            layer_a = self.layers[i]
            layer_b = self.layers[j]
            
            print(f"\n🔗 ENTANGLED PAIR: Layer {i} ⟷ Layer {j}")
            print(f"   Before: L{i}={layer_a.state[:8]}... | L{j}={layer_b.state[:8]}...")
            
            # Store states before
            state_a_before = layer_a.state
            state_b_before = layer_b.state
            
            # Apply entanglement (instant correlation)
            new_state_a, new_state_b, primary = self.apply_entanglement(
                layer_a, layer_b, attack_strength
            )
            
            # Update layers
            layer_a.state = new_state_a
            layer_b.state = new_state_b
            layer_a.M_value = M_current
            layer_b.M_value = M_current
            
            # Response time (primary responds instantly, secondary has 0 delay due to entanglement)
            response_time_primary = random.uniform(0.1, 0.5)
            response_time_secondary = 0.0  # INSTANT due to entanglement
            
            layer_a.response_time = response_time_primary if primary == i else response_time_secondary
            layer_b.response_time = response_time_primary if primary == j else response_time_secondary
            
            # Calculate correlation
            correlation = self.measure_correlation(new_state_a, new_state_b)
            
            print(f"   After:  L{i}={new_state_a[:8]}... | L{j}={new_state_b[:8]}...")
            print(f"   ⚡ Primary Responder: Layer {primary} (HIDDEN from attacker)")
            print(f"   📊 Correlation: {correlation:.3f}")
            print(f"   ⏱️  Response Times: L{i}={layer_a.response_time:.3f}s | L{j}={layer_b.response_time:.3f}s")
            
            # Record event
            event = EntanglementEvent(
                attack_id=attack_id,
                layer_a=i,
                layer_b=j,
                state_a_before=state_a_before,
                state_b_before=state_b_before,
                state_a_after=new_state_a,
                state_b_after=new_state_b,
                correlation=correlation,
                primary_responder=primary,
                response_delta_t=abs(layer_a.response_time - layer_b.response_time)
            )
            self.events.append(event)
            
            # Attack decays through entangled layers
            decay_factor = 0.7 * (1 - correlation * 0.3)
            M_current *= decay_factor
            attack_strength = M_current
            
            print(f"   🛡️  Defense Effectiveness: {(1-decay_factor)*100:.1f}%")
            print(f"   📉 M: {M0:.3f} → {M_current:.3f}")
        
        # Final assessment
        blocked = M_current < 0.3
        
        print(f"\n{'='*70}")
        print(f"🎯 FINAL ASSESSMENT")
        print(f"{'='*70}")
        print(f"   M_final: {M_current:.4f}")
        print(f"   Status: {'🛡️  BLOCKED' if blocked else '⚠️  PENETRATED'}")
        print(f"   Attacker Knowledge: ❓ UNCERTAIN (cannot determine response order)")
        
        return {
            'attack_id': attack_id,
            'M0': M0,
            'M_final': M_current,
            'blocked': blocked,
            'events': self.events[-len(self.entanglement_pairs):]
        }
    
    def visualize_entanglement(self):
        """Visualize entanglement correlations and response patterns"""
        if not self.events:
            print("No events to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Correlation over attacks
        correlations = [e.correlation for e in self.events]
        attack_ids = [e.attack_id for e in self.events]
        
        axes[0, 0].plot(attack_ids, correlations, 'o-', color='purple', linewidth=2, markersize=8)
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random correlation')
        axes[0, 0].set_xlabel('Attack ID', fontsize=11)
        axes[0, 0].set_ylabel('Quantum Correlation', fontsize=11)
        axes[0, 0].set_title('Entanglement Correlation Strength', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Primary responder distribution (hidden from attacker)
        primary_counts = {}
        for e in self.events:
            primary_counts[e.primary_responder] = primary_counts.get(e.primary_responder, 0) + 1
        
        layers = list(primary_counts.keys())
        counts = list(primary_counts.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
        
        axes[0, 1].bar(layers, counts, color=colors, edgecolor='black', linewidth=1.5)
        axes[0, 1].set_xlabel('Layer ID', fontsize=11)
        axes[0, 1].set_ylabel('Times as Primary Responder', fontsize=11)
        axes[0, 1].set_title('Primary Responder Distribution (Hidden)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Response time delta (shows instant entanglement)
        deltas = [e.response_delta_t for e in self.events]
        axes[1, 0].hist(deltas, bins=20, color='cyan', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='Classical limit')
        axes[1, 0].set_xlabel('Response Time Δt (seconds)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Entanglement Speed (Instant Correlation)', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Entanglement network
        ax = axes[1, 1]
        ax.set_xlim(-1, self.n_layers)
        ax.set_ylim(-1, 2)
        ax.axis('off')
        ax.set_title('Entangled Defense Network', fontsize=12, fontweight='bold')
        
        # Draw layers
        for i in range(self.n_layers):
            x = i
            y = 1
            circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, f'L{i}', ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Mark primary responders
            if self.layers[i].is_primary:
                star = plt.Circle((x, y + 0.5), 0.1, color='gold', ec='black')
                ax.add_patch(star)
        
        # Draw entanglement links
        for i, j in self.entanglement_pairs:
            ax.plot([i, j], [1, 1], 'r-', linewidth=3, alpha=0.6)
            ax.text((i+j)/2, 1.3, '⟷', ha='center', fontsize=16, color='red')
        
        plt.tight_layout()
        plt.savefig('quantum_entanglement_defense.png', dpi=150, bbox_inches='tight')
        plt.show()

def demo():
    """Demonstrate quantum entanglement defense"""
    print("🌌 QUANTUM ENTANGLEMENT DEFENSE SYSTEM")
    print("="*70)
    print("⚛️  Two layers entangled → Moving one INSTANTLY changes the other")
    print("❓ Attacker cannot determine which layer responds first")
    print("="*70)
    
    defense = QuantumEntanglementDefense(n_layers=4, n_qubits=8)
    
    print(f"\n🔗 Initialized {len(defense.entanglement_pairs)} entangled pairs:")
    for i, (a, b) in enumerate(defense.entanglement_pairs):
        primary = a if defense.layers[a].is_primary else b
        print(f"   Pair {i}: Layer {a} ⟷ Layer {b} (Primary: {primary} - HIDDEN)")
    
    # Simulate various attacks
    attacks = [
        (0.85, "SQL Injection"),
        (0.65, "XSS Attack"),
        (0.95, "Zero-day Exploit"),
        (0.45, "Brute Force"),
        (0.75, "Path Traversal")
    ]
    
    results = []
    for M0, attack_type in attacks:
        result = defense.process_attack(M0, attack_type)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 DEFENSE SUMMARY")
    print(f"{'='*70}")
    
    blocked = sum(1 for r in results if r['blocked'])
    print(f"🛡️  Attacks Blocked: {blocked}/{len(results)}")
    print(f"⚠️  Attacks Penetrated: {len(results) - blocked}/{len(results)}")
    
    avg_correlation = np.mean([e.correlation for e in defense.events])
    print(f"⚛️  Average Entanglement Correlation: {avg_correlation:.3f}")
    
    avg_response_delta = np.mean([e.response_delta_t for e in defense.events])
    print(f"⚡ Average Response Time Δt: {avg_response_delta:.4f}s (near-instant)")
    
    print("\n🎨 Generating visualization...")
    defense.visualize_entanglement()
    
    return defense

if __name__ == '__main__':
    defense = demo()
