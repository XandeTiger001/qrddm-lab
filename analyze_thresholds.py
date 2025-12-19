#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset_generator import SyntheticDatasetGenerator
from integrated_defense import IntegratedDefenseSystem

def analyze_detection_thresholds():
    """Analyze why attacks aren't being detected"""
    print("Analyzing Detection Thresholds")
    print("=" * 40)
    
    generator = SyntheticDatasetGenerator()
    defense_system = IntegratedDefenseSystem()
    
    # Test different attack types
    attack_types = ['sql_injection', 'xss', 'ddos', 'adversarial']
    
    for attack_type in attack_types:
        print(f"\n{attack_type.upper()} Analysis:")
        
        for intensity in ['low', 'medium', 'high', 'critical']:
            sample = generator.generate_traffic_sample(attack_type, intensity)
            
            # Process through defense system
            response = defense_system.process_request(sample['source_ip'], sample)
            
            print(f"  {intensity:8s}: threat={sample['threat_score']:.3f} -> {response['status']:10s}")
            
            # Show action limiter state
            action_response = defense_system.action_limiter.evaluate_request(sample['source_ip'], sample)
            print(f"            action_level={action_response['action_level']:8s}, count={action_response['request_count']}")
    
    # Test rapid requests from same IP
    print(f"\nRAPID REQUESTS Analysis:")
    attacker_ip = "192.168.1.200"
    
    for i in range(15):
        sample = generator.generate_traffic_sample('sql_injection', 'high')
        sample['source_ip'] = attacker_ip
        
        response = defense_system.process_request(sample['source_ip'], sample)
        
        if i % 3 == 0 or response['status'] != 'ALLOWED':
            print(f"  Request {i+1:2d}: {response['status']:10s} - {response['message']}")

if __name__ == '__main__':
    analyze_detection_thresholds()