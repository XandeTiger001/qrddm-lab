from integrated_defense import IntegratedDefenseSystem
from adaptive_mass import AdaptiveMassCalculator, DefenseMode
from observability import ProductionObservability
from defense_layers import LayeredDefenseSystem
import numpy as np
import time
from datetime import datetime

class ProductionDefenseSystem(IntegratedDefenseSystem):
    """Production-ready defense system with full observability"""
    
    def __init__(self, config=None, observability_window=5):
        super().__init__(config)
        self.observability = ProductionObservability(observability_window)
        self.mass_calculator = AdaptiveMassCalculator(mode=DefenseMode.BALANCED)
        
    def process_request_with_observability(self, source_ip, request_data=None, ground_truth=None):
        """Enhanced request processing with full observability"""
        start_time = time.time()
        
        # Step 1: Calculate adaptive mass and physics parameters
        mass_result = self.mass_calculator.analyze_attack_with_history(request_data or {})
        
        # Step 2: Calculate Schwarzschild parameters
        M_total = mass_result['M_total']
        phi, r_s = self._calculate_schwarzschild_params(M_total)
        
        # Step 3: Process through defense system
        response = super().process_request(source_ip, request_data)
        
        # Step 4: Calculate processing time and confidence
        processing_time = time.time() - start_time
        confidence = self._calculate_confidence(mass_result, response)
        
        # Step 5: Enhanced response with physics data
        enhanced_response = {
            **response,
            'physics': {
                'M_total': M_total,
                'phi': phi,
                'r_s': r_s,
                'mass_components': mass_result['mass_components']
            },
            'confidence': confidence,
            'latency': processing_time
        }
        
        # Step 6: Log everything for observability
        self.observability.log_defense_decision(
            source_ip=source_ip,
            features=mass_result['mass_components'],
            physics_params=enhanced_response['physics'],
            decision_data=enhanced_response,
            ground_truth=ground_truth
        )
        
        return enhanced_response
    
    def _calculate_schwarzschild_params(self, M_total):
        """Calculate Schwarzschild radius and metric"""
        G = 1.0  # Normalized gravitational constant
        c = 1.0  # Speed of light
        
        # Schwarzschild radius: r_s = 2GM/c²
        r_s = 2 * G * M_total / (c ** 2)
        
        # Assume we're at distance r = r_s + 1 (just outside event horizon)
        r = r_s + 1.0
        
        # Schwarzschild metric component: Φ(r) = 1 - r_s/r
        phi = max(0.01, 1 - r_s / r) if r > 0 else 0.01
        
        return phi, r_s
    
    def _calculate_confidence(self, mass_result, response):
        """Calculate model confidence based on various factors"""
        base_confidence = 0.5
        
        # Higher confidence for clear decisions
        M_total = mass_result['M_total']
        if M_total > 3.0:  # High threat mass
            base_confidence = 0.9
        elif M_total > 1.5:  # Medium threat mass
            base_confidence = 0.7
        elif M_total < 0.5:  # Low threat mass
            base_confidence = 0.8
        
        # Reduce confidence for missing features
        if mass_result.get('has_missing_features', False):
            base_confidence *= 0.8
        
        # Adjust based on action consistency
        if response['status'] == 'BLOCKED' and M_total > 2.0:
            base_confidence *= 1.1
        elif response['status'] == 'ALLOWED' and M_total < 1.0:
            base_confidence *= 1.1
        else:
            base_confidence *= 0.9
        
        return min(1.0, base_confidence)
    
    def export_metrics(self):
        """Export current metrics to monitoring systems"""
        return self.observability.export_metrics()
    
    def get_system_health(self):
        """Get comprehensive system health status"""
        base_status = super().get_system_status()
        metrics_result = self.export_metrics()
        
        health_status = {
            **base_status,
            'observability': {
                'metrics_available': metrics_result is not None,
                'active_alerts': len(metrics_result['alerts']) if metrics_result else 0,
                'last_export': datetime.utcnow().isoformat()
            },
            'mass_calculator': self.mass_calculator.get_statistics()
        }
        
        return health_status

def demo_production_system():
    """Demonstrate production system with observability"""
    print("Production Defense System with Observability")
    print("=" * 50)
    
    # Initialize production system
    prod_system = ProductionDefenseSystem(observability_window=1)
    
    # Simulate realistic traffic with ground truth
    traffic_scenarios = [
        # Benign traffic
        {'source_ip': '10.0.1.10', 'type': 'benign', 'threat_score': 0.1, 'ground_truth': 0},
        {'source_ip': '10.0.1.11', 'type': 'benign', 'threat_score': 0.15, 'ground_truth': 0},
        
        # Escalating attack
        {'source_ip': '203.0.113.50', 'type': 'sql_injection', 'threat_score': 0.6, 'ground_truth': 1},
        {'source_ip': '203.0.113.50', 'type': 'sql_injection', 'threat_score': 0.8, 'ground_truth': 1},
        {'source_ip': '203.0.113.50', 'type': 'sql_injection', 'threat_score': 0.95, 'ground_truth': 1},
        
        # DDoS attack
        {'source_ip': '185.220.101.20', 'type': 'ddos', 'threat_score': 0.9, 'ground_truth': 1},
        {'source_ip': '185.220.101.21', 'type': 'ddos', 'threat_score': 0.85, 'ground_truth': 1},
        
        # False positive scenario
        {'source_ip': '192.168.1.100', 'type': 'benign', 'threat_score': 0.4, 'ground_truth': 0}
    ]
    
    print("Processing traffic with full observability...\n")
    
    for i, scenario in enumerate(traffic_scenarios):
        ground_truth = scenario.pop('ground_truth')
        
        # Add realistic attack parameters
        scenario.update({
            'target_layer': np.random.randint(1, 4),
            'num_techniques': np.random.randint(1, 3),
            'mutations': np.random.randint(0, 2),
            'adversarial_noise': np.random.randint(0, 1)
        })
        
        response = prod_system.process_request_with_observability(
            scenario['source_ip'], 
            scenario, 
            ground_truth
        )
        
        print(f"Request {i+1}: {scenario['type']} from {scenario['source_ip']}")
        print(f"  Status: {response['status']}")
        print(f"  Confidence: {response['confidence']:.3f}")
        print(f"  M(r): {response['physics']['M_total']:.3f}")
        print(f"  Phi(r): {response['physics']['phi']:.3f}")
        print(f"  r_s: {response['physics']['r_s']:.3f}")
        print(f"  Latency: {response['latency']*1000:.2f}ms")
        print()
        
        time.sleep(0.1)  # Simulate realistic timing
    
    # Export metrics and show system health
    print("Exporting metrics and checking system health...\n")
    
    metrics_result = prod_system.export_metrics()
    if metrics_result:
        metrics = metrics_result['metrics']
        print("PERFORMANCE METRICS:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.3f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.3f}")
        print(f"  Average Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  Average Confidence: {metrics['avg_confidence']:.3f}")
        
        if metrics_result['alerts']:
            print(f"\nACTIVE ALERTS:")
            for alert in metrics_result['alerts']:
                print(f"  {alert['severity']}: {alert['message']}")
        else:
            print(f"\nNo active alerts")
    
    # System health check
    health = prod_system.get_system_health()
    print(f"\nSYSTEM HEALTH:")
    print(f"  Active blocks: {health['active_blocks']}")
    print(f"  Honeypot sessions: {health['honeypot_sessions']}")
    print(f"  Tracked sources: {health['mass_calculator']['tracked_sources']}")
    print(f"  Total events: {health['mass_calculator']['total_events']}")
    print(f"  Active alerts: {health['observability']['active_alerts']}")

if __name__ == '__main__':
    demo_production_system()