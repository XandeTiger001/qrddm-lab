import numpy as np
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from collections import defaultdict
# import matplotlib.pyplot as plt  # Optional for plotting

class DefenseEvaluator:
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
        self.detection_times = []
        self.threat_scores = []
        self.attack_types = []
        
    def add_prediction(self, y_true, y_pred, threat_score, detection_time, attack_type):
        """Add single prediction result"""
        self.ground_truth.append(y_true)
        self.predictions.append(y_pred)
        self.threat_scores.append(threat_score)
        self.detection_times.append(detection_time)
        self.attack_types.append(attack_type)
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        if not self.predictions:
            return {}
        
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        
        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Detection latency
        avg_detection_time = np.mean(self.detection_times)
        p95_detection_time = np.percentile(self.detection_times, 95)
        
        # AUC if we have threat scores
        auc = roc_auc_score(y_true, self.threat_scores) if len(set(y_true)) > 1 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'avg_detection_latency': avg_detection_time,
            'p95_detection_latency': p95_detection_time,
            'auc_score': auc,
            'total_samples': len(self.predictions)
        }
    
    def calculate_per_attack_metrics(self):
        """Calculate metrics per attack type"""
        attack_metrics = {}
        
        for attack_type in set(self.attack_types):
            indices = [i for i, at in enumerate(self.attack_types) if at == attack_type]
            
            if not indices:
                continue
                
            y_true_subset = [self.ground_truth[i] for i in indices]
            y_pred_subset = [self.predictions[i] for i in indices]
            detection_times_subset = [self.detection_times[i] for i in indices]
            
            if len(set(y_true_subset)) > 1:
                precision = precision_score(y_true_subset, y_pred_subset, zero_division=0)
                recall = recall_score(y_true_subset, y_pred_subset, zero_division=0)
                f1 = f1_score(y_true_subset, y_pred_subset, zero_division=0)
            else:
                precision = recall = f1 = 0
            
            attack_metrics[attack_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_detection_time': np.mean(detection_times_subset),
                'sample_count': len(indices)
            }
        
        return attack_metrics
    
    def reset(self):
        """Reset all stored results"""
        self.predictions.clear()
        self.ground_truth.clear()
        self.detection_times.clear()
        self.threat_scores.clear()
        self.attack_types.clear()

class BoundaryTestSuite:
    def __init__(self, defense_system):
        self.defense_system = defense_system
        self.evaluator = DefenseEvaluator()
    
    def test_traffic_spikes(self, spike_intensity=1000, duration_seconds=10):
        """Test system response to traffic spikes"""
        print(f"Testing traffic spike: {spike_intensity} requests in {duration_seconds}s")
        
        results = []
        start_time = time.time()
        
        for i in range(spike_intensity):
            request_time = time.time()
            
            # Simulate DDoS-like traffic
            sample = {
                'source_ip': f"192.168.1.{(i % 50) + 1}",
                'type': 'ddos',
                'threat_score': 0.9,
                'target_layer': 2,
                'num_techniques': 1,
                'mutations': 0,
                'adversarial_noise': 0
            }
            
            detection_start = time.time()
            response = self.defense_system.process_request(sample['source_ip'], sample)
            detection_time = time.time() - detection_start
            
            # Determine if attack was detected (blocked or redirected)
            detected = response['status'] in ['BLOCKED', 'REDIRECT', 'THROTTLED']
            
            self.evaluator.add_prediction(
                y_true=1,  # All are attacks
                y_pred=1 if detected else 0,
                threat_score=sample['threat_score'],
                detection_time=detection_time,
                attack_type='ddos_spike'
            )
            
            results.append({
                'request_id': i,
                'response_status': response['status'],
                'detection_time': detection_time,
                'detected': detected
            })
            
            # Small delay to simulate realistic timing
            if i % 100 == 0:
                time.sleep(0.01)
        
        total_time = time.time() - start_time
        print(f"Spike test completed in {total_time:.2f}s")
        
        return results
    
    def test_burst_patterns(self, num_bursts=5, burst_size=20, burst_interval=60):
        """Test response to burst attack patterns"""
        print(f"Testing burst patterns: {num_bursts} bursts of {burst_size} requests")
        
        results = []
        
        for burst_id in range(num_bursts):
            print(f"  Burst {burst_id + 1}/{num_bursts}")
            
            for i in range(burst_size):
                sample = {
                    'source_ip': f"10.0.0.{burst_id + 1}",
                    'type': 'sql_injection',
                    'threat_score': 0.7 + (i * 0.01),  # Gradually increasing threat
                    'target_layer': max(1, 3 - (i // 5)),
                    'num_techniques': min(5, 1 + (i // 3)),
                    'mutations': i // 2,
                    'adversarial_noise': 0
                }
                
                detection_start = time.time()
                response = self.defense_system.process_request(sample['source_ip'], sample)
                detection_time = time.time() - detection_start
                
                detected = response['status'] in ['BLOCKED', 'REDIRECT', 'THROTTLED']
                
                self.evaluator.add_prediction(
                    y_true=1,
                    y_pred=1 if detected else 0,
                    threat_score=sample['threat_score'],
                    detection_time=detection_time,
                    attack_type='sql_burst'
                )
                
                results.append({
                    'burst_id': burst_id,
                    'request_id': i,
                    'response_status': response['status'],
                    'detected': detected
                })
            
            # Wait between bursts
            if burst_id < num_bursts - 1:
                time.sleep(burst_interval / 100)  # Scaled down for testing
        
        return results
    
    def test_evasive_patterns(self, num_requests=50, evasion_techniques=None):
        """Test response to evasive/low-and-slow attacks"""
        print(f"Testing evasive patterns: {num_requests} subtle requests")
        
        if evasion_techniques is None:
            evasion_techniques = ['low_threat', 'distributed_sources', 'timing_variation']
        
        results = []
        
        for i in range(num_requests):
            # Vary evasion technique
            technique = evasion_techniques[i % len(evasion_techniques)]
            
            if technique == 'low_threat':
                threat_score = np.random.uniform(0.2, 0.4)  # Below typical thresholds
                source_ip = "172.16.1.100"
            elif technique == 'distributed_sources':
                threat_score = np.random.uniform(0.5, 0.7)
                source_ip = f"203.0.113.{np.random.randint(1, 255)}"
            else:  # timing_variation
                threat_score = np.random.uniform(0.4, 0.6)
                source_ip = "198.51.100.50"
                time.sleep(np.random.exponential(0.1))  # Random delays
            
            sample = {
                'source_ip': source_ip,
                'type': 'adversarial',
                'threat_score': threat_score,
                'target_layer': 3,
                'num_techniques': 2,
                'mutations': 3,
                'adversarial_noise': 2
            }
            
            detection_start = time.time()
            response = self.defense_system.process_request(sample['source_ip'], sample)
            detection_time = time.time() - detection_start
            
            detected = response['status'] in ['BLOCKED', 'REDIRECT', 'THROTTLED']
            
            self.evaluator.add_prediction(
                y_true=1,
                y_pred=1 if detected else 0,
                threat_score=sample['threat_score'],
                detection_time=detection_time,
                attack_type='evasive'
            )
            
            results.append({
                'request_id': i,
                'technique': technique,
                'threat_score': threat_score,
                'response_status': response['status'],
                'detected': detected
            })
        
        return results
    
    def run_comprehensive_test(self):
        """Run all boundary tests"""
        print("Running Comprehensive Boundary Test Suite")
        print("=" * 50)
        
        # Reset evaluator
        self.evaluator.reset()
        
        # Run tests
        spike_results = self.test_traffic_spikes(spike_intensity=200, duration_seconds=5)
        burst_results = self.test_burst_patterns(num_bursts=3, burst_size=15)
        evasive_results = self.test_evasive_patterns(num_requests=30)
        
        # Calculate metrics
        overall_metrics = self.evaluator.calculate_metrics()
        per_attack_metrics = self.evaluator.calculate_per_attack_metrics()
        
        return {
            'overall_metrics': overall_metrics,
            'per_attack_metrics': per_attack_metrics,
            'test_results': {
                'spike_results': spike_results,
                'burst_results': burst_results,
                'evasive_results': evasive_results
            }
        }

def print_evaluation_results(results):
    """Print formatted evaluation results"""
    print("\nEVALUATION RESULTS")
    print("=" * 40)
    
    overall = results['overall_metrics']
    print(f"Overall Performance:")
    print(f"  Precision: {overall['precision']:.3f}")
    print(f"  Recall: {overall['recall']:.3f}")
    print(f"  F1-Score: {overall['f1_score']:.3f}")
    print(f"  False Positive Rate: {overall['false_positive_rate']:.3f}")
    print(f"  AUC Score: {overall['auc_score']:.3f}")
    print(f"  Avg Detection Latency: {overall['avg_detection_latency']:.4f}s")
    print(f"  95th Percentile Latency: {overall['p95_detection_latency']:.4f}s")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {overall['true_positives']}")
    print(f"  False Positives: {overall['false_positives']}")
    print(f"  True Negatives: {overall['true_negatives']}")
    print(f"  False Negatives: {overall['false_negatives']}")
    
    print(f"\nPer-Attack Type Performance:")
    for attack_type, metrics in results['per_attack_metrics'].items():
        print(f"  {attack_type}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1-Score: {metrics['f1_score']:.3f}")
        print(f"    Avg Detection Time: {metrics['avg_detection_time']:.4f}s")
        print(f"    Sample Count: {metrics['sample_count']}")

if __name__ == '__main__':
    # This would be run with an actual defense system
    print("Evaluation metrics module loaded successfully")
    print("Use with: evaluator = DefenseEvaluator()")
    print("         test_suite = BoundaryTestSuite(defense_system)")