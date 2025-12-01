import numpy as np
import json
from datetime import datetime
from enum import Enum

class ThreatLevel(Enum):
    SAFE = -1
    MONITOR = 0
    CRITICAL = 1

class SchwarzschildDefense:
    def __init__(self, D=1.0, c=1.0, T_shadow=0.5, T_crit=0.2):
        self.D = D
        self.c = c
        self.T_shadow = T_shadow
        self.T_crit = T_crit
        self.M_min = 0.01
        self.M_max = 100.0
        self.r_min = 0.1
        self.r_max = 100.0
        self.event_history = []
        self.max_history = 100
    
    def calculate_M(self, threat_score, payload_size, request_rate):
        size_factor = np.log1p(payload_size / 1000)
        rate_factor = np.log1p(request_rate / 10)
        M = threat_score * (1 + size_factor + rate_factor)
        return np.clip(M, self.M_min, self.M_max)
    
    def calculate_r(self, time_to_impact, network_hops=1):
        r = time_to_impact * network_hops
        return np.clip(r, self.r_min, self.r_max)
    
    def phi(self, r, M):
        denominator = max(self.c**2 * r, 1e-6)
        phi_value = 1 - (2 * self.D * M) / denominator
        return np.clip(phi_value, -10, 10)
    
    def schwarzschild_radius(self, M):
        """Event Horizon Radius"""
        return (2 * self.D * M) / (self.c**2)
    
    def curvature(self, r, M):
        denominator = max(r**3, 1e-6)
        C = (2 * self.D * M) / denominator
        return np.clip(C, 0, 1e6)
    
    def classify_threat(self, phi_value):
        """Ternary classification based 12 Φ(r)"""
        if phi_value >= self.T_shadow:
            return ThreatLevel.SAFE
        elif phi_value > self.T_crit:
            return ThreatLevel.MONITOR
        else:
            return ThreatLevel.CRITICAL
    
    def analyze_event(self, threat_score, payload_size, request_rate, 
                      time_to_impact=1.0, network_hops=1):
        """Complete analysis using Schwarzschild Metric"""
        M = self.calculate_M(threat_score, payload_size, request_rate)
        r = self.calculate_r(time_to_impact, network_hops)
        
        phi_value = self.phi(r, M)
        r_s = self.schwarzschild_radius(M)
        C = self.curvature(r, M)
        
        threat_level = self.classify_threat(phi_value)
        
        inside_horizon = r <= r_s
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'M': round(M, 4),
                'r': round(r, 4),
                'phi': round(phi_value, 4),
                'r_schwarzschild': round(r_s, 4),
                'curvature': round(C, 4)
            },
            'classification': {
                'level': threat_level.name,
                'value': threat_level.value,
                'inside_horizon': inside_horizon
            },
            'action': self._get_action(threat_level, inside_horizon)
        }
    
    def _get_action(self, level, inside_horizon):
        """Determines action based on the threat level"""
        if inside_horizon:
            return 'BLOCK_IMMEDIATE'
        elif level == ThreatLevel.CRITICAL:
            return 'REDIRECT_HONEYPOT'
        elif level == ThreatLevel.MONITOR:
            return 'LOG_AND_MONITOR'
        else:
            return 'ALLOW'
    
    def update_baseline(self, phi_value):
        self.event_history.append(phi_value)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
    
    def get_adaptive_thresholds(self):
        if len(self.event_history) < 10:
            return self.T_shadow, self.T_crit
        baseline = np.median(self.event_history)
        std = np.std(self.event_history)
        T_shadow_adaptive = max(0.3, min(0.7, baseline + 0.5 * std))
        T_crit_adaptive = max(0.1, min(0.4, baseline - 0.5 * std))
        return T_shadow_adaptive, T_crit_adaptive
    
    def adapt_parameters(self, feedback_score):
        if feedback_score < 0.3:
            self.T_crit = max(0.05, self.T_crit * 0.95)
            self.T_shadow = max(0.2, self.T_shadow * 0.97)
        elif feedback_score > 0.8:
            self.T_crit = min(0.5, self.T_crit * 1.05)
            self.T_shadow = min(0.8, self.T_shadow * 1.03)

def simulate_attacks():
    """Simulation of attacks with Schwarzschild analysis"""
    defense = SchwarzschildDefense(D=1.5, c=1.0, T_shadow=0.5, T_crit=0.2)
    
    scenarios = [
        {'name': 'Normal Attack', 'threat': 0.3, 'size': 100, 'rate': 5, 'time': 2.0},
        {'name': 'SQL Injection', 'threat': 0.8, 'size': 500, 'rate': 15, 'time': 1.0},
        {'name': 'DDoS Massive', 'threat': 0.9, 'size': 10000, 'rate': 100, 'time': 0.5},
        {'name': 'Attack Near', 'threat': 0.7, 'size': 200, 'rate': 20, 'time': 0.2},
        {'name': 'Legitimate Traffic', 'threat': 0.1, 'size': 50, 'rate': 2, 'time': 5.0}
    ]
    
    print("🌌 Defense Simulation Schwarzschild\n")
    print("=" * 80)
    
    for scenario in scenarios:
        result = defense.analyze_event(
            scenario['threat'], 
            scenario['size'], 
            scenario['rate'],
            scenario['time']
        )
        
        print(f"\n📡 {scenario['name']}")
        print(f"   M(r) = {result['metrics']['M']:.4f} | r = {result['metrics']['r']:.4f}")
        print(f"   Φ(r) = {result['metrics']['phi']:.4f} | r_s = {result['metrics']['r_schwarzschild']:.4f}")
        print(f"   Curvature = {result['metrics']['curvature']:.4f}")
        print(f"   🎯 Classification: {result['classification']['level']}")
        print(f"   🛡️  Action: {result['action']}")
        
        if result['classification']['inside_horizon']:
            print(f"   ⚠️  INSIDE THE EVENT HORIZON!")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    simulate_attacks()
