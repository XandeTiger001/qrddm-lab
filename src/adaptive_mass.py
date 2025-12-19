import numpy as np
from enum import Enum
from collections import defaultdict, deque
import time
from typing import Dict, Optional, Tuple

class DefenseMode(Enum):
    PARANOID = "paranoid"
    BALANCED = "balanced"
    ECONOMY = "economy"
    STUDY = "study"

class AdaptiveMassCalculator:
    def __init__(self, mode=DefenseMode.BALANCED, temporal_alpha=0.7, history_window=300):
        self.mode = mode
        self.coefficients = self._initialize_coefficients(mode)
        self.feature_bounds = {
            'S_max': 1.0, 'F_max': 5.0, 'V_max': 1.0, 
            'D_max': 1.0, 'C_max': 8.0
        }
        self.coeff_bounds = {'min': 0.1, 'max': 2.0}
        
        # Temporal smoothing
        self.temporal_alpha = temporal_alpha  # EMA coefficient
        self.M_previous = 0.0
        
        # Historical data for F and V calculation
        self.history_window = history_window  # seconds
        self.event_history = defaultdict(deque)  # {source_ip: [(timestamp, pattern), ...]}
        self.pattern_history = deque(maxlen=100)  # Global pattern history
        
        self.feedback_history = []
    
    def _initialize_coefficients(self, mode):
        presets = {
            DefenseMode.PARANOID: {'alpha': 1.2, 'beta': 0.8, 'gamma': 0.6, 'delta': 1.4, 'epsilon': 0.5},
            DefenseMode.BALANCED: {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.8, 'delta': 1.0, 'epsilon': 0.7},
            DefenseMode.ECONOMY: {'alpha': 0.8, 'beta': 0.6, 'gamma': 0.4, 'delta': 0.7, 'epsilon': 0.3},
            DefenseMode.STUDY: {'alpha': 0.6, 'beta': 1.3, 'gamma': 0.5, 'delta': 0.6, 'epsilon': 0.4}
        }
        return presets[mode]
    
    def _normalize_feature(self, value, feature_name):
        """Normalize and clip features to [0,1]"""
        max_val = self.feature_bounds.get(f'{feature_name}_max', 1.0)
        return np.clip(value / max_val, 0.0, 1.0)
    
    def _clean_old_events(self, current_time):
        """Remove events older than history_window"""
        cutoff_time = current_time - self.history_window
        
        for source_ip in list(self.event_history.keys()):
            events = self.event_history[source_ip]
            while events and events[0][0] < cutoff_time:
                events.popleft()
            if not events:
                del self.event_history[source_ip]
    
    def calculate_severity(self, attack_type, threat_score):
        """S: Normalized severity [0,1]"""
        severity_weights = {
            'sql_injection': 0.9, 'xss': 0.7, 'path_traversal': 0.8,
            'ddos': 0.95, 'buffer_overflow': 0.85, 'adversarial_ml': 0.9, 
            'normal': 0.1
        }
        base_severity = severity_weights.get(attack_type, 0.5)
        S = base_severity * np.clip(threat_score, 0.0, 1.0)
        return self._normalize_feature(S, 'S')
    
    def calculate_frequency(self, source_ip, attack_type, current_time=None):
        """F: Frequency based on similar events in time window"""
        if current_time is None:
            current_time = time.time()
        
        self._clean_old_events(current_time)
        
        # Count similar events from same source
        events = self.event_history.get(source_ip, deque())
        similar_count = sum(1 for _, pattern in events if pattern.get('type') == attack_type)
        
        # Add current event to history
        current_pattern = {'type': attack_type, 'timestamp': current_time}
        self.event_history[source_ip].append((current_time, current_pattern))
        
        # Calculate frequency (events per minute)
        F = similar_count / (self.history_window / 60.0)
        return self._normalize_feature(F, 'F')
    
    def calculate_velocity(self, current_threat_score, source_ip, current_time=None):
        """V: Velocity based on threat score changes over time"""
        if current_time is None:
            current_time = time.time()
        
        events = self.event_history.get(source_ip, deque())
        
        if len(events) < 2:
            # No previous data - return low velocity
            V = 0.1
        else:
            # Get most recent previous event
            prev_time, prev_pattern = events[-2] if len(events) >= 2 else (current_time - 1, {'threat_score': 0})
            prev_threat_score = prev_pattern.get('threat_score', 0)
            
            # Calculate velocity as rate of change
            dt = max(current_time - prev_time, 0.1)  # Avoid division by zero
            dv = abs(current_threat_score - prev_threat_score)
            V = dv / dt
        
        # Add current pattern to global history
        self.pattern_history.append({
            'timestamp': current_time,
            'threat_score': current_threat_score,
            'source_ip': source_ip
        })
        
        return self._normalize_feature(V, 'V')
    
    def calculate_distance(self, target_layer, core_layer=0):
        """D: Normalized distance (inverted) [0,1]"""
        distance = abs(target_layer - core_layer)
        D = 1.0 / (1.0 + distance)
        return self._normalize_feature(D, 'D')
    
    def calculate_complexity(self, num_techniques, mutations, adversarial_noise):
        """C: Normalized complexity with entropy cap [0,1]"""
        C = np.log1p(num_techniques + mutations + adversarial_noise)
        C = np.clip(C, 0.0, self.feature_bounds['C_max'])
        return self._normalize_feature(C, 'C')
    
    def calculate_M_with_temporal_smoothing(self, S, F, V, D, C, has_missing_features=False):
        """M(r) with temporal smoothing: M_t = alpha*M_now + (1-alpha)*M_prev"""
        # Calculate current M
        alpha = np.clip(self.coefficients['alpha'], self.coeff_bounds['min'], self.coeff_bounds['max'])
        beta = np.clip(self.coefficients['beta'], self.coeff_bounds['min'], self.coeff_bounds['max'])
        gamma = np.clip(self.coefficients['gamma'], self.coeff_bounds['min'], self.coeff_bounds['max'])
        delta = np.clip(self.coefficients['delta'], self.coeff_bounds['min'], self.coeff_bounds['max'])
        epsilon = np.clip(self.coefficients['epsilon'], self.coeff_bounds['min'], self.coeff_bounds['max'])
        
        M_current = alpha * S + beta * F + gamma * V + delta * D + epsilon * C
        
        # Apply degradation if missing critical features
        if has_missing_features:
            M_current *= 0.7  # Degrade decision confidence
        
        # Temporal smoothing (exponential moving average)
        M_smoothed = self.temporal_alpha * M_current + (1 - self.temporal_alpha) * self.M_previous
        
        # Update previous value for next iteration
        self.M_previous = M_smoothed
        
        return np.clip(M_smoothed, 0.01, 10.0)
    
    def analyze_attack_with_history(self, attack_data):
        """Complete analysis with historical context"""
        current_time = time.time()
        source_ip = attack_data.get('source_ip', 'unknown')
        attack_type = attack_data.get('type', 'unknown')
        threat_score = attack_data.get('threat_score', 0.5)
        
        # Calculate features with historical context
        S = self.calculate_severity(attack_type, threat_score)
        
        # F and V with proper historical data
        F = self.calculate_frequency(source_ip, attack_type, current_time)
        V = self.calculate_velocity(threat_score, source_ip, current_time)
        
        D = self.calculate_distance(
            attack_data.get('target_layer', 3),
            attack_data.get('distance_to_core', 0.5)
        )
        
        C = self.calculate_complexity(
            attack_data.get('num_techniques', 1),
            attack_data.get('mutations', 0),
            attack_data.get('adversarial_noise', 0)
        )
        
        # Check for missing features
        has_missing_features = (
            source_ip == 'unknown' or 
            len(self.event_history.get(source_ip, [])) < 2
        )
        
        # Calculate M with temporal smoothing
        M_total = self.calculate_M_with_temporal_smoothing(S, F, V, D, C, has_missing_features)
        
        return {
            'M_total': M_total,
            'mass_components': {
                'S_severity': S, 'F_frequency': F, 'V_velocity': V,
                'D_distance': D, 'C_complexity': C
            },
            'temporal_smoothed': True,
            'has_missing_features': has_missing_features,
            'event_count': len(self.event_history.get(source_ip, [])),
            'M_previous': self.M_previous
        }
    
    def set_mode(self, mode):
        """Change defense mode"""
        self.mode = mode
        self.coefficients = self._initialize_coefficients(mode)
    
    def get_statistics(self):
        """Get system statistics"""
        total_sources = len(self.event_history)
        total_events = sum(len(events) for events in self.event_history.values())
        
        return {
            'mode': self.mode.value,
            'temporal_alpha': self.temporal_alpha,
            'tracked_sources': total_sources,
            'total_events': total_events,
            'M_previous': self.M_previous,
            'coefficients': self.coefficients.copy()
        }

def demo_adaptive_mass():
    """Demo with temporal smoothing and proper F/V calculation"""
    print("=== Cyber Event Horizon: Adaptive Mass with Temporal Smoothing ===\n")
    
    calculator = AdaptiveMassCalculator(mode=DefenseMode.BALANCED, temporal_alpha=0.7)
    
    # Simulate attack sequence from same source
    attack_sequence = [
        {
            'name': 'Initial Probe',
            'type': 'sql_injection',
            'threat_score': 0.3,
            'source_ip': '192.168.1.100',
            'target_layer': 2,
            'num_techniques': 1,
            'mutations': 0,
            'adversarial_noise': 0
        },
        {
            'name': 'Escalated Attack',
            'type': 'sql_injection', 
            'threat_score': 0.7,
            'source_ip': '192.168.1.100',
            'target_layer': 1,
            'num_techniques': 3,
            'mutations': 2,
            'adversarial_noise': 1
        },
        {
            'name': 'Advanced Persistent Threat',
            'type': 'sql_injection',
            'threat_score': 0.95,
            'source_ip': '192.168.1.100', 
            'target_layer': 0,
            'num_techniques': 5,
            'mutations': 8,
            'adversarial_noise': 3
        }
    ]
    
    print("Analyzing attack sequence with temporal smoothing...\n")
    
    for i, attack in enumerate(attack_sequence):
        # Add small delay to simulate time progression
        time.sleep(0.1)
        
        result = calculator.analyze_attack_with_history(attack)
        
        print(f"Attack {i+1}: {attack['name']}")
        print(f"  Threat Score: {attack['threat_score']:.2f}")
        print(f"  M(r) Components:")
        mc = result['mass_components']
        print(f"    S={mc['S_severity']:.3f}, F={mc['F_frequency']:.3f}, V={mc['V_velocity']:.3f}")
        print(f"    D={mc['D_distance']:.3f}, C={mc['C_complexity']:.3f}")
        print(f"  M_total (smoothed): {result['M_total']:.4f}")
        print(f"  M_previous: {result['M_previous']:.4f}")
        print(f"  Event count for source: {result['event_count']}")
        print(f"  Missing features: {result['has_missing_features']}")
        print()
    
    # Show statistics
    stats = calculator.get_statistics()
    print("System Statistics:")
    print(f"  Mode: {stats['mode']}")
    print(f"  Temporal Alpha: {stats['temporal_alpha']}")
    print(f"  Tracked Sources: {stats['tracked_sources']}")
    print(f"  Total Events: {stats['total_events']}")

if __name__ == '__main__':
    demo_adaptive_mass()