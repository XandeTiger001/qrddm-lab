import numpy as np
import datetime
from typing import Dict, List, Tuple
from enum import Enum
from ai_integration import IntegratedAIDefense

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

class GravitationalLensing:
    def __init__(self, lens_strength=1.2):
        self.lens_strength = lens_strength
        self.photon_trajectories = []
    
    def detect_visual_distortions(self, original_data: np.ndarray, reconstructed_data: np.ndarray) -> Dict[str, float]:
        distortion_map = np.abs(original_data - reconstructed_data)
        deflection_angle = np.mean(distortion_map) * self.lens_strength
        magnification = 1.0 / (1.0 - deflection_angle) if deflection_angle < 0.99 else 10.0
        
        return {
            'deflection_angle': deflection_angle,
            'magnification': magnification,
            'distortion_strength': np.max(distortion_map),
            'lensing_detected': deflection_angle > 0.1
        }
    
    def reconstruct_photon_trajectories(self, distorted_data: np.ndarray, steps=5) -> List[np.ndarray]:
        trajectories = []
        
        for step in range(steps):
            curvature_factor = 1.0 - (step * 0.15)
            corrected_trajectory = distorted_data * curvature_factor + np.random.normal(0, 0.05, distorted_data.shape)
            lens_correction = self.lens_strength * np.sin(step * np.pi / steps)
            final_trajectory = corrected_trajectory + lens_correction * 0.1
            
            trajectories.append(final_trajectory)
            self.photon_trajectories.append({
                'step': step,
                'curvature_factor': curvature_factor,
                'trajectory': final_trajectory.tolist()
            })
        
        return trajectories

class ComputationalEventHorizon:
    def __init__(self, information_limit=0.95, uncertainty_threshold=0.8):
        self.information_limit = information_limit
        self.uncertainty_threshold = uncertainty_threshold
        self.recovery_attempts = []
    
    def calculate_information_boundary(self, reconstruction_confidence: float, 
                                    data_completeness: float, 
                                    noise_level: float) -> Dict[str, float]:
        information_density = reconstruction_confidence * data_completeness * (1.0 - noise_level)
        horizon_distance = 1.0 - (information_density / self.information_limit)
        beyond_horizon = horizon_distance <= 0.0
        leakage_rate = np.exp(-information_density * 2.0) if not beyond_horizon else 1.0
        
        return {
            'information_density': information_density,
            'horizon_distance': max(0.0, horizon_distance),
            'beyond_horizon': beyond_horizon,
            'recovery_reliability': 1.0 - leakage_rate,
            'hallucination_risk': leakage_rate
        }
    
    def enforce_recovery_limits(self, recovery_data: np.ndarray, 
                              confidence_scores: np.ndarray) -> Tuple[np.ndarray, bool]:
        reliable_mask = confidence_scores > self.uncertainty_threshold
        filtered_recovery = recovery_data.copy()
        filtered_recovery[~reliable_mask] = 0.0
        recovery_blocked = np.sum(~reliable_mask) > len(reliable_mask) * 0.5
        
        self.recovery_attempts.append({
            'timestamp': datetime.datetime.now(),
            'reliable_points': np.sum(reliable_mask),
            'total_points': len(reliable_mask),
            'recovery_blocked': recovery_blocked
        })
        
        return filtered_recovery, recovery_blocked

class BlackHoleEntropy:
    def __init__(self):
        self.entropy_history = []
    
    def calculate_scene_entropy(self, reconstructed_data: np.ndarray) -> float:
        normalized_data = (reconstructed_data - np.min(reconstructed_data)) / (np.max(reconstructed_data) - np.min(reconstructed_data) + 1e-8)
        hist, _ = np.histogram(normalized_data, bins=10, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        return entropy
    
    def adjust_confidence_by_entropy(self, base_confidence: float, entropy: float) -> float:
        normalized_entropy = min(entropy / 3.3, 1.0)
        entropy_penalty = normalized_entropy * 0.4
        adjusted_confidence = base_confidence * (1.0 - entropy_penalty)
        
        self.entropy_history.append({
            'entropy': entropy,
            'base_confidence': base_confidence,
            'adjusted_confidence': adjusted_confidence,
            'penalty': entropy_penalty
        })
        
        return max(0.1, adjusted_confidence)

class InformationDecay:
    def __init__(self, decay_rate=0.3):
        self.decay_rate = decay_rate
        self.decay_timeline = []
    
    def simulate_horizon_crossing(self, data: np.ndarray, steps=5) -> List[Dict]:
        decay_stages = []
        current_data = data.copy()
        
        for step in range(steps):
            scrambling_factor = (step + 1) / steps
            tidal_stretch = np.random.normal(1.0, scrambling_factor * 0.2, data.shape)
            current_data *= tidal_stretch
            
            decay_mask = np.random.random(data.shape) < (scrambling_factor * self.decay_rate)
            current_data[decay_mask] = np.random.normal(0, 0.1, np.sum(decay_mask))
            
            hawking_noise = np.random.normal(0, scrambling_factor * 0.05, data.shape)
            current_data += hawking_noise
            
            stage = {
                'step': step,
                'scrambling_factor': scrambling_factor,
                'data_integrity': 1.0 - scrambling_factor,
                'decayed_data': current_data.copy(),
                'recoverable': scrambling_factor < 0.7
            }
            
            decay_stages.append(stage)
            self.decay_timeline.append(stage)
        
        return decay_stages

class AdaptiveCurvatureTensor:
    def __init__(self, tensor_size=8):
        self.tensor_size = tensor_size
        self.curvature_tensor = np.eye(tensor_size)
        self.perception_history = []
    
    def calculate_riemann_tensor(self, information_density: np.ndarray) -> np.ndarray:
        inverse_density = 1.0 / (information_density + 0.1)
        curvature_strength = np.mean(inverse_density)
        
        tensor = np.eye(self.tensor_size)
        for i in range(self.tensor_size):
            for j in range(self.tensor_size):
                if i != j:
                    tensor[i, j] = curvature_strength * 0.1 * np.sin(i + j)
        
        return tensor
    
    def warp_ai_perception(self, input_data: np.ndarray, curvature_tensor: np.ndarray) -> np.ndarray:
        if len(input_data) != self.tensor_size:
            if len(input_data) < self.tensor_size:
                padded_data = np.zeros(self.tensor_size)
                padded_data[:len(input_data)] = input_data
                input_data = padded_data
            else:
                input_data = input_data[:self.tensor_size]
        
        warped_perception = curvature_tensor @ input_data
        metric_correction = np.sqrt(np.abs(np.diag(curvature_tensor)))
        warped_perception *= metric_correction
        
        self.perception_history.append({
            'original': input_data.tolist(),
            'warped': warped_perception.tolist(),
            'curvature_strength': np.trace(curvature_tensor)
        })
        
        return warped_perception
    
    def adapt_to_information_landscape(self, data_regions: List[np.ndarray]) -> np.ndarray:
        densities = []
        for region in data_regions:
            density = np.var(region) + 0.1
            densities.append(density)
        
        density_array = np.array(densities)
        if len(density_array) < self.tensor_size:
            avg_density = np.mean(density_array)
            density_array = np.pad(density_array, (0, self.tensor_size - len(density_array)), constant_values=avg_density)
        
        self.curvature_tensor = self.calculate_riemann_tensor(density_array[:self.tensor_size])
        return self.curvature_tensor

class EnhancedSchwarzschildDefense:
    def __init__(self, G=1.5, c=1.0, mode=DefenseMode.BALANCED):
        self.G = G
        self.c = c
        self.mass_calculator = AdaptiveMassCalculator(mode)
        self.attack_history = {}
        self.gravitational_lens = GravitationalLensing()
        self.event_horizon = ComputationalEventHorizon()
        self.black_hole_entropy = BlackHoleEntropy()
        self.information_decay = InformationDecay()
        self.curvature_tensor = AdaptiveCurvatureTensor()
    
    def analyze_attack(self, attack_data):
        attack_type = attack_data.get('type', 'normal')
        threat_score = attack_data.get('threat_score', 0.5)
        source_ip = attack_data.get('source_ip', 'unknown')
        
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
        
        if source_ip not in self.attack_history:
            self.attack_history[source_ip] = []
        self.attack_history[source_ip].append(threat_score)
        
        lensing_analysis = {}
        horizon_analysis = {}
        
        if 'original_data' in attack_data and 'reconstructed_data' in attack_data:
            original = np.array(attack_data['original_data'])
            reconstructed = np.array(attack_data['reconstructed_data'])
            
            lensing_analysis = self.gravitational_lens.detect_visual_distortions(original, reconstructed)
            trajectories = self.gravitational_lens.reconstruct_photon_trajectories(reconstructed)
            lensing_analysis['trajectory_count'] = len(trajectories)
            
            confidence = attack_data.get('reconstruction_confidence', 0.7)
            completeness = attack_data.get('data_completeness', 0.8)
            noise = attack_data.get('noise_level', 0.2)
            
            horizon_analysis = self.event_horizon.calculate_information_boundary(confidence, completeness, noise)
            
            scene_entropy = self.black_hole_entropy.calculate_scene_entropy(reconstructed)
            adjusted_confidence = self.black_hole_entropy.adjust_confidence_by_entropy(confidence, scene_entropy)
            
            decay_stages = self.information_decay.simulate_horizon_crossing(reconstructed)
            
            data_regions = [original[:4], reconstructed[:4]]
            curvature_tensor = self.curvature_tensor.adapt_to_information_landscape(data_regions)
            warped_perception = self.curvature_tensor.warp_ai_perception(reconstructed, curvature_tensor)
            
            horizon_analysis.update({
                'scene_entropy': scene_entropy,
                'entropy_adjusted_confidence': adjusted_confidence,
                'decay_stages': len(decay_stages),
                'final_data_integrity': decay_stages[-1]['data_integrity'] if decay_stages else 1.0,
                'curvature_strength': np.trace(curvature_tensor),
                'perception_warped': True
            })
        
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
            'field_warping': self._calculate_warping(M, r),
            'gravitational_lensing': lensing_analysis,
            'event_horizon': horizon_analysis
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
    ai_defense = IntegratedAIDefense()
    
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
            'source_ip': '172.16.0.20',
            'original_data': [1.0, 0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6],
            'reconstructed_data': [0.9, 0.7, 0.5, 0.8, 0.6, 0.4, 0.7, 0.5],
            'reconstruction_confidence': 0.6,
            'data_completeness': 0.7,
            'noise_level': 0.4
        }
    ]
    
    for scenario in scenarios:
        ai_result = ai_defense.analyze_traffic(scenario)
        result = defense.analyze_attack(scenario)
        
        print(f"\n[SCAN] {scenario['name']}")
        print(f"   [AI] Classification: {ai_result['ai_classification']} ({ai_result['confidence']:.1%})")
        print(f"   [ALERT] Anomaly: {'YES' if ai_result['is_anomaly'] else 'NO'} (score: {ai_result['anomaly_score']:.2f})")
        
        if 'event_horizon' in result and result['event_horizon']:
            horizon = result['event_horizon']
            if horizon.get('scene_entropy'):
                print(f"   [ENTROPY] Scene Entropy: {horizon['scene_entropy']:.3f}, Adjusted Confidence: {horizon.get('entropy_adjusted_confidence', 0):.3f}")
            if horizon.get('decay_stages'):
                print(f"   [DECAY] Information Decay: {horizon['decay_stages']} stages, Integrity: {horizon.get('final_data_integrity', 0):.1%}")
            if horizon.get('curvature_strength'):
                print(f"   [TENSOR] Curvature Tensor: Strength {horizon['curvature_strength']:.3f}, Perception Warped")
    
    print("\n[TEST] Testing Black Hole Entropy...")
    entropy_calc = BlackHoleEntropy()
    test_data = np.array([0.9, 0.7, 0.5, 0.8, 0.6, 0.4, 0.7, 0.5])
    entropy = entropy_calc.calculate_scene_entropy(test_data)
    adjusted_conf = entropy_calc.adjust_confidence_by_entropy(0.8, entropy)
    
    print(f"   Scene Entropy: {entropy:.3f} bits")
    print(f"   Confidence: 0.800 → {adjusted_conf:.3f} (entropy penalty)")

if __name__ == '__main__':
    demo_adaptive_mass()