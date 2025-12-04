import numpy as np
from enum import Enum

class DefenseMode(Enum):
    PARANOID = "paranoid"
    BALANCED = "balanced"
    ECONOMY = "economy"
    STUDY = "study"

class AdaptiveMassCalculator:
    def __init__(self, mode=DefenseMode.BALANCED):
        self.mode = mode
        self.coefficients = self._initialize_coefficients(mode)
    
    def _initialize_coefficients(self, mode):
        """Coefficients α, β, γ, δ, ε based on mode"""
        presets = {
            DefenseMode.PARANOID: {'alpha': 1.5, 'beta': 1.0, 'gamma': 0.8, 'delta': 1.5, 'epsilon': 0.7},
            DefenseMode.BALANCED: {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'epsilon': 1.0},
            DefenseMode.ECONOMY: {'alpha': 0.8, 'beta': 0.7, 'gamma': 0.5, 'delta': 0.8, 'epsilon': 0.5},
            DefenseMode.STUDY: {'alpha': 0.7, 'beta': 1.5, 'gamma': 0.6, 'delta': 0.7, 'epsilon': 0.8}
        }
        return presets[mode]
    
    def calculate_severity(self, attack_type, threat_score):
        """S = Attack severity"""
        severity_weights = {
            'sql_injection': 0.9,
            'xss': 0.7,
            'path_traversal': 0.8,
            'ddos': 0.95,
            'buffer_overflow': 0.85,
            'adversarial_ml': 0.9,
            'normal': 0.1
        }
        base_severity = severity_weights.get(attack_type, 0.5)
        return base_severity * threat_score
    
    def calculate_frequency(self, event_count, time_window=60):
        """F = Frequency of similar attacks"""
        return np.log1p(event_count / time_window)
    
    def calculate_velocity(self, current_pattern, previous_pattern):
        """V = Rate of attack change"""
        if previous_pattern is None:
            return 0.0
        diff = abs(current_pattern - previous_pattern)
        return np.tanh(diff)
    
    def calculate_distance(self, target_layer, core_layer=0):
        """D = Proximity to critical component (inverted)"""
        distance = abs(target_layer - core_layer)
        return 1.0 / (1.0 + distance)
    
    def calculate_complexity(self, num_techniques, mutations, adversarial_noise):
        """C = Attack complexity"""
        return np.log1p(num_techniques + mutations + adversarial_noise)
    
    def calculate_M(self, S, F, V, D, C):
        """M(r) = α·S + β·F + γ·V + δ·D + ε·C"""
        alpha = self.coefficients['alpha']
        beta = self.coefficients['beta']
        gamma = self.coefficients['gamma']
        delta = self.coefficients['delta']
        epsilon = self.coefficients['epsilon']
        
        M = alpha * S + beta * F + gamma * V + delta * D + epsilon * C
        return np.clip(M, 0.01, 100.0)
    
    def calculate_angular_spread(self, affected_modules, total_modules):
        """dΩ² = Angular spread (how many modules are affected)"""
        if total_modules == 0:
            return 0.0
        return (affected_modules / total_modules) * np.pi
    
    def adapt_coefficients(self, feedback):
        """AI adjusts coefficients based on feedback"""
        if feedback['false_positives'] > 0.3:
            self.coefficients['alpha'] *= 0.9
            self.coefficients['delta'] *= 0.9
        elif feedback['false_negatives'] > 0.3:
            self.coefficients['alpha'] *= 1.1
            self.coefficients['delta'] *= 1.1
        
        if feedback['study_mode']:
            self.coefficients['beta'] *= 1.2
        
        # Limit coefficients
        for key in self.coefficients:
            self.coefficients[key] = np.clip(self.coefficients[key], 0.1, 2.0)
    
def set_mode(self, mode):
    """Change defense mode"""
    self.mode = mode
    self.coefficients = self._initialize_coefficients(mode)


class EnhancedSchwarzschildDefense:
    def __init__(self, G=1.5, c=1.0, mode=DefenseMode.BALANCED):
        self.G = G
        self.c = c
        self.mass_calculator = AdaptiveMassCalculator(mode)
        self.attack_history = {}
    
    def analyze_attack(self, attack_data):
        """Full analysis with adaptive M(r)"""
        attack_type = attack_data.get('type', 'normal')
        threat_score = attack_data.get('threat_score', 0.5)
        source_ip = attack_data.get('source_ip', 'unknown')
        
        # Calculate M(r) components
        S = self.mass_calculator.calculate_severity(attack_type, threat_score)
        
        event_count = self.attack_history.get(source_ip, [])
        F = self.mass_calculator.calculate_frequency(len(event_count))
        
        previous = event_count[-1] if event_count else None
        V = self.mass_calculator.calculate_velocity(threat_score, previous)
        
        target_layer = attack_data.get('target_layer', 3)
        D = self.mass_calculator.calculate_distance(target_layer)
        
        C = self.mass_calculator.calculate_complexity(
            attack_data.get('num_techniques', 1),
            attack_data.get('mutations', 0),
            attack_data.get('adversarial_noise', 0)
        )
        
        # Calculate M(r)
        M = self.mass_calculator.calculate_M(S, F, V, D, C)
        
        # Distance to core
        r = attack_data.get('distance_to_core', 1.0)
        r = np.clip(r, 0.1, 100.0)
        
        # Schwarzschild metric
        phi = self._calculate_phi(r, M)
        r_s = self._schwarzschild_radius(M)
        curvature = self._curvature(r, M)
        
        # Angular spread
        omega = self.mass_calculator.calculate_angular_spread(
            attack_data.get('affected_modules', 1),
            attack_data.get('total_modules', 10)
        )
        
        # Update attack history
        if source_ip not in self.attack_history:
            self.attack_history[source_ip] = []
        self.attack_history[source_ip].append(threat_score)
        
        return {
            'mass_components': {
                'S_severity': round(float(S), 4),
                'F_frequency': round(float(F), 4),
                'V_velocity': round(float(V), 4),
                'D_distance': round(float(D), 4),
                'C_complexity': round(float(C), 4)
            },
            'coefficients': self.mass_calculator.coefficients,
            'M_total': round(float(M), 4),
            'r': round(float(r), 4),
            'phi': round(float(phi), 4),
            'r_schwarzschild': round(float(r_s), 4),
            'curvature': round(float(curvature), 4),
            'omega_spread': round(float(omega), 4),
            'classification': self._classify(phi),
            'field_warping': self._calculate_warping(M, r)
        }
    
    def _calculate_phi(self, r, M):
        denominator = max(self.c**2 * r, 1e-6)
        phi = 1 - (2 * self.G * M) / denominator
        return np.clip(phi, -10, 10)
    
    def _schwarzschild_radius(self, M):
        return (2 * self.G * M) / (self.c**2)
    
    def _curvature(self, r, M):
        denominator = max(r**3, 1e-6)
        return np.clip((2 * self.G * M) / denominator, 0, 1e6)
    
    def _classify(self, phi):
        if phi >= 0.5:
            return 'SAFE'
        elif phi > 0.2:
            return 'MONITOR'
        else:
            return 'CRITICAL'
    
    def _calculate_warping(self, M, r):
        """Energy required to defend (Cuvarture of digital space)"""
        warping = (M / r**2) * 100
        return round(float(np.clip(warping, 0, 1000)), 2)

def demo_adaptive_mass():
    defense = EnhancedSchwarzschildDefense(G=1.5, c=1.0, mode=DefenseMode.BALANCED)
    
    scenarios = [
        {
            'name': 'Simple SQL Injection',
            'type': 'sql_injection',
            'threat_score': 0.7,
            'target_layer': 2,
            'num_techniques': 1,
            'mutations': 0,
            'adversarial_noise': 0,
            'distance_to_core': 3.0,
            'affected_modules': 1,
            'total_modules': 10,
            'source_ip': '192.168.1.100'
        },
        {
            'name': 'Repeated Massive DDoS',
            'type': 'ddos',
            'threat_score': 0.95,
            'target_layer': 1,
            'num_techniques': 2,
            'mutations': 5,
            'adversarial_noise': 3,
            'distance_to_core': 0.5,
            'affected_modules': 8,
            'total_modules': 10,
            'source_ip': '10.0.0.50'
        },
        {
            'name': 'Adversarial ML Attack',
            'type': 'adversarial_ml',
            'threat_score': 0.85,
            'target_layer': 0,
            'num_techniques': 4,
            'mutations': 10,
            'adversarial_noise': 8,
            'distance_to_core': 0.2,
            'affected_modules': 3,
            'total_modules': 10,
            'source_ip': '172.16.0.20'
        }
    ]
    
def demo_adaptive_mass():
    print("🌌 Adaptive M(r) Defense System\n")
    print("M(r) = α·S + β·F + γ·V + δ·D + ε·C\n")
    print("=" * 80)
    
    for scenario in scenarios:
        result = defense.analyze_attack(scenario)
        
        print(f"\n📡 {scenario['name']}")
        print(f"\n   M(r) Components:")
        mc = result['mass_components']
        print(f"   • S (Severity): {mc['S_severity']:.4f}")
        print(f"   • F (Frequency): {mc['F_frequency']:.4f}")
        print(f"   • V (Velocity): {mc['V_velocity']:.4f}")
        print(f"   • D (Distance): {mc['D_distance']:.4f}")
        print(f"   • C (Complexity): {mc['C_complexity']:.4f}")
        
        print(f"\n   Coefficients (α,β,γ,δ,ε):")
        coef = result['coefficients']
        print(f"   {coef['alpha']:.2f}, {coef['beta']:.2f}, {coef['gamma']:.2f}, "
              f"{coef['delta']:.2f}, {coef['epsilon']:.2f}")
        
        print(f"\n   Schwarzschild Metric:")
        print(f"   • M(r) = {result['M_total']:.4f} (attack mass)")
        print(f"   • r = {result['r']:.4f} (distance to core)")
        print(f"   • Φ(r) = {result['phi']:.4f}")
        print(f"   • r_s = {result['r_schwarzschild']:.4f} (horizon)")
        print(f"   • Curvature = {result['curvature']:.4f}")
        print(f"   • Ω = {result['omega_spread']:.4f} rad (spread)")
        
        print(f"\n   🎯 Classification: {result['classification']}")
        print(f"   ⚡ Defense Energy: {result['field_warping']} units")
        
        if result['r'] <= result['r_schwarzschild']:
            print(f"   ⚠️  INSIDE THE EVENT HORIZON!")
    
    print("\n" + "=" * 80)
    print("\n🧠 Switching to PARANOID mode...")
    defense.mass_calculator.set_mode(DefenseMode.PARANOID)
    
    result = defense.analyze_attack(scenarios[0])
    print(f"\nSQL Injection in PARANOID mode:")
    print(f"   M(r) = {result['M_total']:.4f} (previously: ~0.7)")
    print(f"   Classification: {result['classification']}")


if __name__ == '__main__':
    demo_adaptive_mass()
