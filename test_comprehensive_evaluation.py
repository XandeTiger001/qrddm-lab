#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset_generator import SyntheticDatasetGenerator
from evaluation_metrics import DefenseEvaluator, BoundaryTestSuite, print_evaluation_results
from integrated_defense import IntegratedDefenseSystem
from adaptive_mass import AdaptiveMassCalculator, DefenseMode
import pandas as pd
import numpy as np
import time

class ComprehensiveTestSuite:
    def __init__(self):
        self.generator = SyntheticDatasetGenerator()
        self.defense_system = IntegratedDefenseSystem()
        self.evaluator = DefenseEvaluator()
        
    def test_standard_dataset(self, samples_per_type=200):
        """Test on standard synthetic dataset"""
        print("Generating and testing standard dataset...")
        
        dataset = self.generator.generate_dataset(samples_per_type=samples_per_type)
        
        for _, sample in dataset.iterrows():
            detection_start = time.time()
            
            # Convert sample to format expected by defense system
            request_data = {
                'source_ip': sample['source_ip'],
                'type': sample['type'],
                'threat_score': sample['threat_score'],
                'target_layer': sample.get('target_layer', 3),
                'num_techniques': sample.get('num_techniques', 1),
                'mutations': sample.get('mutations', 0),
                'adversarial_noise': sample.get('adversarial_noise', 0)
            }
            
            response = self.defense_system.process_request(
                sample['source_ip'], 
                request_data
            )
            
            detection_time = time.time() - detection_start
            
            # Determine if attack was detected
            detected = response['status'] in ['BLOCKED', 'REDIRECT', 'THROTTLED']
            
            self.evaluator.add_prediction(
                y_true=sample['label'],
                y_pred=1 if detected else 0,
                threat_score=sample['threat_score'],
                detection_time=detection_time,
                attack_type=sample['type']
            )
        
        return len(dataset)
    
    def test_adaptive_mass_integration(self, num_samples=100):
        """Test integration with adaptive mass calculator"""
        print("Testing adaptive mass integration...")
        
        mass_calculator = AdaptiveMassCalculator(mode=DefenseMode.PARANOID)
        
        # Generate attack sequence
        for i in range(num_samples):
            attack_type = np.random.choice(['sql_injection', 'xss', 'ddos', 'adversarial', 'benign'])
            intensity = np.random.choice(['low', 'medium', 'high'])
            
            sample = self.generator.generate_traffic_sample(attack_type, intensity)
            
            # Calculate adaptive mass
            mass_result = mass_calculator.analyze_attack_with_history(sample)
            
            # Use mass to enhance threat score
            enhanced_threat = min(1.0, sample['threat_score'] * (1 + mass_result['M_total'] / 10))
            sample['threat_score'] = enhanced_threat
            
            detection_start = time.time()
            response = self.defense_system.process_request(sample['source_ip'], sample)
            detection_time = time.time() - detection_start
            
            detected = response['status'] in ['BLOCKED', 'REDIRECT', 'THROTTLED']
            
            self.evaluator.add_prediction(
                y_true=sample['label'],
                y_pred=1 if detected else 0,
                threat_score=enhanced_threat,
                detection_time=detection_time,
                attack_type=f"{attack_type}_adaptive"
            )
    
    def run_full_evaluation(self):
        """Run complete evaluation suite"""
        print("Cyber Event Horizon - Comprehensive Evaluation Suite")
        print("=" * 60)
        
        # Reset evaluator
        self.evaluator.reset()
        
        # Test 1: Standard dataset
        standard_samples = self.test_standard_dataset(samples_per_type=150)
        print(f"Tested {standard_samples} standard samples")
        
        # Test 2: Adaptive mass integration
        self.test_adaptive_mass_integration(num_samples=100)
        print("Tested 100 adaptive mass samples")
        
        # Test 3: Boundary cases
        boundary_suite = BoundaryTestSuite(self.defense_system)
        boundary_results = boundary_suite.run_comprehensive_test()
        
        # Combine results
        overall_metrics = self.evaluator.calculate_metrics()
        per_attack_metrics = self.evaluator.calculate_per_attack_metrics()
        
        # Add boundary test results
        boundary_overall = boundary_results['overall_metrics']
        boundary_per_attack = boundary_results['per_attack_metrics']
        
        return {
            'standard_metrics': {
                'overall_metrics': overall_metrics,
                'per_attack_metrics': per_attack_metrics
            },
            'boundary_metrics': {
                'overall_metrics': boundary_overall,
                'per_attack_metrics': boundary_per_attack
            },
            'combined_sample_count': overall_metrics['total_samples'] + boundary_overall['total_samples']
        }

def main():
    """Run comprehensive evaluation"""
    test_suite = ComprehensiveTestSuite()
    
    print("Starting comprehensive evaluation...")
    start_time = time.time()
    
    results = test_suite.run_full_evaluation()
    
    total_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {total_time:.2f} seconds")
    print(f"Total samples processed: {results['combined_sample_count']}")
    
    # Print standard dataset results
    print("\nSTANDARD DATASET RESULTS:")
    print_evaluation_results(results['standard_metrics'])
    
    # Print boundary test results
    print("\nBOUNDARY TEST RESULTS:")
    print_evaluation_results(results['boundary_metrics'])
    
    # Summary comparison
    std_metrics = results['standard_metrics']['overall_metrics']
    boundary_metrics = results['boundary_metrics']['overall_metrics']
    
    print("\nPERFORMANCE COMPARISON:")
    print("=" * 40)
    print(f"{'Metric':<20} {'Standard':<10} {'Boundary':<10} {'Delta':<10}")
    print("-" * 50)
    
    metrics_to_compare = ['precision', 'recall', 'f1_score', 'false_positive_rate']
    for metric in metrics_to_compare:
        std_val = std_metrics[metric]
        boundary_val = boundary_metrics[metric]
        delta = boundary_val - std_val
        print(f"{metric:<20} {std_val:<10.3f} {boundary_val:<10.3f} {delta:<+10.3f}")
    
    # Detection latency comparison
    print(f"{'avg_detection_latency':<20} {std_metrics['avg_detection_latency']:<10.4f} {boundary_metrics['avg_detection_latency']:<10.4f} {boundary_metrics['avg_detection_latency'] - std_metrics['avg_detection_latency']:<+10.4f}")
    
    print("\nEVALUATION SUMMARY:")
    print("- Generated synthetic datasets with SQLi, XSS, DDoS, adversarial, and benign traffic")
    print("- Tested boundary cases: traffic spikes, burst patterns, evasive attacks")
    print("- Measured precision, recall, F1-score, false positive rate, detection latency")
    print("- Integrated with adaptive mass calculation for enhanced threat assessment")
    print("- Validated tiered defense system performance under various attack scenarios")

if __name__ == '__main__':
    main()