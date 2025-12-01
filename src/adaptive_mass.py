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
        """Coeficientes α, β, γ, δ, ε baseados no modo"""
        presets = {
            DefenseMode.PARANOID: {'alpha': 1.5, 'beta': 1.0, 'gamma': 0.8, 'delta': 1.5, 'epsilon': 0.7},
            DefenseMode.BALANCED: {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'epsilon': 1.0},
            DefenseMode.ECONOMY: {'alpha': 0.8, 'beta': 0.7, 'gamma': 0.5, 'delta': 0.8, 'epsilon': 0.5},
            DefenseMode.STUDY: {'alpha': 0.7, 'beta': 1.5, 'gamma': 0.6, 'delta': 0.7, 'epsilon': 0.8}
        }
        return presets[mode]
    
    def calculate_severity(self, attack_type, threat_score):
        """S = Severidade do ataque"""
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
        """F = Frequência de ataques similares"""
        return np.log1p(event_count / time_window)
    
    def calculate_velocity(self, current_pattern, previous_pattern):
        """V = Velocidade de mudança do ataque"""
        if previous_pattern is None:
            return 0.0
        diff = abs(current_pattern - previous_pattern)
        return np.tanh(diff)
    
    def calculate_distance(self, target_layer, core_layer=0):
        """D = Proximidade ao componente crítico (invertido)"""
        distance = abs(target_layer - core_layer)
        return 1.0 / (1.0 + distance)
    
    def calculate_complexity(self, num_techniques, mutations, adversarial_noise):
        """C = Complexidade do ataque"""
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
        """dΩ² = Dispersão angular (quantos módulos afetados)"""
        if total_modules == 0:
            return 0.0
        return (affected_modules / total_modules) * np.pi
    
    def adapt_coefficients(self, feedback):
        """IA ajusta coeficientes baseado em feedback"""
        if feedback['false_positives'] > 0.3:
            self.coefficients['alpha'] *= 0.9
            self.coefficients['delta'] *= 0.9
        elif feedback['false_negatives'] > 0.3:
            self.coefficients['alpha'] *= 1.1
            self.coefficients['delta'] *= 1.1
        
        if feedback['study_mode']:
            self.coefficients['beta'] *= 1.2
        
        # Limita coeficientes
        for key in self.coefficients:
            self.coefficients[key] = np.clip(self.coefficients[key], 0.1, 2.0)
    
    def set_mode(self, mode):
        """Muda modo de defesa"""
        self.mode = mode
        self.coefficients = self._initialize_coefficients(mode)

class EnhancedSchwarzschildDefense:
    def __init__(self, G=1.5, c=1.0, mode=DefenseMode.BALANCED):
        self.G = G
        self.c = c
        self.mass_calculator = AdaptiveMassCalculator(mode)
        self.attack_history = {}
    
    def analyze_attack(self, attack_data):
        """Análise completa com M(r) adaptativo"""
        attack_type = attack_data.get('type', 'normal')
        threat_score = attack_data.get('threat_score', 0.5)
        source_ip = attack_data.get('source_ip', 'unknown')
        
        # Calcula componentes de M(r)
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
        
        # Calcula M(r)
        M = self.mass_calculator.calculate_M(S, F, V, D, C)
        
        # Distância ao núcleo
        r = attack_data.get('distance_to_core', 1.0)
        r = np.clip(r, 0.1, 100.0)
        
        # Métrica de Schwarzschild
        phi = self._calculate_phi(r, M)
        r_s = self._schwarzschild_radius(M)
        curvature = self._curvature(r, M)
        
        # Dispersão angular
        omega = self.mass_calculator.calculate_angular_spread(
            attack_data.get('affected_modules', 1),
            attack_data.get('total_modules', 10)
        )
        
        # Atualiza histórico
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
        """Energia necessária para defender (curvatura do espaço digital)"""
        warping = (M / r**2) * 100
        return round(float(np.clip(warping, 0, 1000)), 2)

def demo_adaptive_mass():
    defense = EnhancedSchwarzschildDefense(G=1.5, c=1.0, mode=DefenseMode.BALANCED)
    
    scenarios = [
        {
            'name': 'SQL Injection Simples',
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
            'name': 'DDoS Massivo Repetido',
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
    
    print("🌌 Sistema de Defesa com M(r) Adaptativo\n")
    print("M(r) = α·S + β·F + γ·V + δ·D + ε·C\n")
    print("=" * 80)
    
    for scenario in scenarios:
        result = defense.analyze_attack(scenario)
        
        print(f"\n📡 {scenario['name']}")
        print(f"\n   Componentes de M(r):")
        mc = result['mass_components']
        print(f"   • S (Severidade): {mc['S_severity']:.4f}")
        print(f"   • F (Frequência): {mc['F_frequency']:.4f}")
        print(f"   • V (Velocidade): {mc['V_velocity']:.4f}")
        print(f"   • D (Distância): {mc['D_distance']:.4f}")
        print(f"   • C (Complexidade): {mc['C_complexity']:.4f}")
        
        print(f"\n   Coeficientes (α,β,γ,δ,ε):")
        coef = result['coefficients']
        print(f"   {coef['alpha']:.2f}, {coef['beta']:.2f}, {coef['gamma']:.2f}, {coef['delta']:.2f}, {coef['epsilon']:.2f}")
        
        print(f"\n   Métrica de Schwarzschild:")
        print(f"   • M(r) = {result['M_total']:.4f} (massa do ataque)")
        print(f"   • r = {result['r']:.4f} (distância ao núcleo)")
        print(f"   • Φ(r) = {result['phi']:.4f}")
        print(f"   • r_s = {result['r_schwarzschild']:.4f} (horizonte)")
        print(f"   • Curvatura = {result['curvature']:.4f}")
        print(f"   • Ω = {result['omega_spread']:.4f} rad (dispersão)")
        
        print(f"\n   🎯 Classificação: {result['classification']}")
        print(f"   ⚡ Energia de Defesa: {result['field_warping']} unidades")
        
        if result['r'] <= result['r_schwarzschild']:
            print(f"   ⚠️  DENTRO DO HORIZONTE DE EVENTOS!")
    
    print("\n" + "=" * 80)
    print("\n🧠 Adaptando para modo PARANOID...")
    defense.mass_calculator.set_mode(DefenseMode.PARANOID)
    
    result = defense.analyze_attack(scenarios[0])
    print(f"\nSQL Injection em modo PARANOID:")
    print(f"   M(r) = {result['M_total']:.4f} (antes: ~0.7)")
    print(f"   Classificação: {result['classification']}")

if __name__ == '__main__':
    demo_adaptive_mass()
