import numpy as np
import datetime
from typing import Dict, List, Tuple
from enum import Enum

class DefenseMode(Enum):
    BALANCED = "balanced"
    PARANOID = "paranoid"
    PERMISSIVE = "permissive"

class AdaptiveMassCalculator:
    def __init__(self, mode=DefenseMode.BALANCED):
        self.mode = mode
        self.coefficients = self._initialize_coefficients(mode)
    
    def _initialize_coefficients(self, mode):
        if mode == DefenseMode.PARANOID:
            return {'alpha': 1.5, 'beta': 1.2, 'gamma': 1.3, 'delta': 0.8, 'epsilon': 1.4}
        elif mode == DefenseMode.PERMISSIVE:
            return {'alpha': 0.7, 'beta': 0.8, 'gamma': 0.6, 'delta': 1.2, 'epsilon': 0.5}
        else:
            return {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'epsilon': 1.0}
    
    def calculate_severity(self, attack_type, threat_score):
        severity_map = {
            'sql_injection': 0.6, 'ddos': 0.9, 'adversarial_ml': 0.8,
            'ransomware': 0.95, 'apt': 0.85, 'normal': 0.1
        }
        base_severity = severity_map.get(attack_type, 0.5)
        return base_severity * threat_score
    
    def calculate_frequency(self, event_count):
        return min(event_count / 10.0, 1.0)
    
    def calculate_velocity(self, current_threat, previous_threat):
        if previous_threat is None:
            return 0.0
        return abs(current_threat - previous_threat)
    
    def calculate_distance(self, target_layer):
        return max(0.1, 1.0 / (target_layer + 1))
    
    def calculate_complexity(self, num_techniques, mutations, adversarial_noise):
        return (num_techniques * 0.2 + mutations * 0.05 + adversarial_noise * 0.1)
    
    def calculate_M(self, S, F, V, D, C):
        alpha = self.coefficients['alpha']
        beta = self.coefficients['beta']
        gamma = self.coefficients['gamma']
        delta = self.coefficients['delta']
        epsilon = self.coefficients['epsilon']
        
        M = alpha * S + beta * F + gamma * V + delta * D + epsilon * C
        return np.clip(M, 0.01, 100.0)
    
    def calculate_angular_spread(self, affected_modules, total_modules):
        if total_modules == 0:
            return 0.0
        return (affected_modules / total_modules) * np.pi
    
    def set_mode(self, mode):
        self.mode = mode
        self.coefficients = self._initialize_coefficients(mode)

class SimpleAIDefense:
    """Lightweight AI defense without TensorFlow dependencies"""
    
    def __init__(self):
        self.attack_patterns = {
            'sql_injection': {'confidence': 0.85, 'anomaly_score': 0.7},
            'ddos': {'confidence': 0.92, 'anomaly_score': 0.95},
            'adversarial_ml': {'confidence': 0.78, 'anomaly_score': 0.85}
        }
    
    def analyze_traffic(self, attack_data):
        attack_type = attack_data.get('type', 'normal')
        pattern = self.attack_patterns.get(attack_type, {'confidence': 0.5, 'anomaly_score': 0.3})
        
        # Simple heuristic-based analysis
        threat_score = attack_data.get('threat_score', 0.5)
        confidence = pattern['confidence'] * threat_score
        anomaly_score = pattern['anomaly_score'] * threat_score
        
        is_anomaly = anomaly_score > 0.6
        
        # Generate simple insight
        insights = {
            'sql_injection': 'Database injection pattern detected with query manipulation signatures',
            'ddos': 'Distributed denial of service attack overwhelming system resources',
            'adversarial_ml': 'Machine learning model poisoning attempt with evasion techniques'
        }
        
        ai_insight = insights.get(attack_type, f'{attack_type} pattern identified')
        
        # Assess threat level
        combined_score = confidence * anomaly_score
        if combined_score > 1.5:
            threat_level = "CRITICAL"
        elif combined_score > 0.8:
            threat_level = "HIGH"
        elif combined_score > 0.4:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        return {
            'ai_classification': attack_type.upper(),
            'confidence': round(confidence, 4),
            'is_anomaly': is_anomaly,
            'anomaly_score': round(anomaly_score, 4),
            'ai_insight': ai_insight,
            'threat_assessment': threat_level
        }

class EnhancedSchwarzschildDefense:
    def __init__(self, G=1.5, c=1.0, mode=DefenseMode.BALANCED):
        self.G = G
        self.c = c
        self.mass_calculator = AdaptiveMassCalculator(mode)
        self.attack_history = {}
    
    def analyze_attack(self, attack_data):
        attack_type = attack_data.get('type', 'normal')
        threat_score = attack_data.get('threat_score', 0.5)
        source_ip = attack_data.get('source_ip', 'unknown')
        
        # Calculate mass components
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
        
        # Calculate Schwarzschild metrics
        M = self.mass_calculator.calculate_M(S, F, V, D, C)
        r = attack_data.get('distance_to_core', 1.0)
        r = np.clip(r, 0.1, 100.0)
        
        phi = self._calculate_phi(r, M)
        r_s = self._schwarzschild_radius(M)
        curvature = self._curvature(r, M)
        
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
        warping = (M / r**2) * 100
        return round(float(np.clip(warping, 0, 1000)), 2)

def demo_adaptive_mass():
    print("*** Adaptive M(r) Defense System ***\n")
    print("M(r) = alpha*S + beta*F + gamma*V + delta*D + epsilon*C\n")
    print("=" * 80)
    
    defense = EnhancedSchwarzschildDefense(G=1.5, c=1.0, mode=DefenseMode.BALANCED)
    ai_defense = SimpleAIDefense()
    
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
            'name': 'Massive DDoS Attack',
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
    
    for scenario in scenarios:
        # Enhanced analysis with AI
        ai_result = ai_defense.analyze_traffic(scenario)
        result = defense.analyze_attack(scenario)
        
        print(f"\n[SCAN] {scenario['name']}")
        print(f"   [AI] Classification: {ai_result['ai_classification']} ({ai_result['confidence']:.1%})")
        print(f"   [ALERT] Anomaly: {'YES' if ai_result['is_anomaly'] else 'NO'} (score: {ai_result['anomaly_score']:.2f})")
        print(f"\n   M(r) Components:")
        mc = result['mass_components']
        print(f"   • S (Severity): {mc['S_severity']:.4f}")
        print(f"   • F (Frequency): {mc['F_frequency']:.4f}")
        print(f"   • V (Velocity): {mc['V_velocity']:.4f}")
        print(f"   • D (Distance): {mc['D_distance']:.4f}")
        print(f"   • C (Complexity): {mc['C_complexity']:.4f}")
        
        print(f"\n   Coefficients (alpha,beta,gamma,delta,epsilon):")
        coef = result['coefficients']
        print(f"   {coef['alpha']:.2f}, {coef['beta']:.2f}, {coef['gamma']:.2f}, "
              f"{coef['delta']:.2f}, {coef['epsilon']:.2f}")
        
        print(f"\n   Schwarzschild Metric:")
        print(f"   • M(r) = {result['M_total']:.4f} (attack mass)")
        print(f"   • r = {result['r']:.4f} (distance to core)")
        print(f"   • Phi(r) = {result['phi']:.4f}")
        print(f"   • r_s = {result['r_schwarzschild']:.4f} (horizon)")
        print(f"   • Curvature = {result['curvature']:.4f}")
        print(f"   • Omega = {result['omega_spread']:.4f} rad (spread)")
        
        print(f"\n   [RESULT] Schwarzschild Classification: {result['classification']}")
        print(f"   [THREAT] AI Threat Level: {ai_result['threat_assessment']}")
        print(f"   [ENERGY] Defense Energy: {result['field_warping']} units")
        print(f"   [INSIGHT] {ai_result['ai_insight']}")
        
        if result['r'] <= result['r_schwarzschild']:
            print(f"   [WARNING] INSIDE THE EVENT HORIZON!")
    
    print("\n" + "=" * 80)
    print("\n[MODE] Switching to PARANOID mode...")
    defense.mass_calculator.set_mode(DefenseMode.PARANOID)
    
    result = defense.analyze_attack(scenarios[0])
    print(f"\nSQL Injection in PARANOID mode:")
    print(f"   M(r) = {result['M_total']:.4f} (previously: ~0.7)")
    print(f"   Classification: {result['classification']}")
    
    print("\n[DEMO] Testing different attack scenarios...")
    
    # Test edge cases
    edge_cases = [
        {
            'name': 'Zero-Day Exploit',
            'type': 'zero_day',
            'threat_score': 0.99,
            'target_layer': 0,
            'num_techniques': 8,
            'mutations': 15,
            'adversarial_noise': 12,
            'distance_to_core': 0.1,
            'affected_modules': 9,
            'total_modules': 10,
            'source_ip': '203.0.113.1'
        }
    ]
    
    for case in edge_cases:
        ai_result = ai_defense.analyze_traffic(case)
        result = defense.analyze_attack(case)
        
        print(f"\n[EDGE] {case['name']}")
        print(f"   M(r) = {result['M_total']:.4f}")
        print(f"   Classification: {result['classification']}")
        print(f"   Threat Level: {ai_result['threat_assessment']}")
        
        if result['classification'] == 'CRITICAL':
            print(f"   [CRITICAL] Immediate response required!")

if __name__ == '__main__':
    demo_adaptive_mass()