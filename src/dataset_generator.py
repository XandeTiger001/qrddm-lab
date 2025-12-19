import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from collections import defaultdict

class SyntheticDatasetGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.attack_patterns = {
            'sql_injection': {
                'base_threat': 0.8,
                'techniques': ['union', 'blind', 'time_based', 'error_based'],
                'payloads': ["' OR 1=1--", "'; DROP TABLE--", "UNION SELECT NULL--"]
            },
            'xss': {
                'base_threat': 0.7,
                'techniques': ['stored', 'reflected', 'dom_based'],
                'payloads': ["<script>alert('xss')</script>", "javascript:alert(1)", "onload=alert(1)"]
            },
            'ddos': {
                'base_threat': 0.9,
                'techniques': ['volumetric', 'protocol', 'application'],
                'patterns': ['burst', 'sustained', 'amplification']
            },
            'adversarial': {
                'base_threat': 0.85,
                'techniques': ['evasion', 'poisoning', 'model_inversion'],
                'noise_levels': [0.1, 0.3, 0.5, 0.8]
            },
            'benign': {
                'base_threat': 0.1,
                'patterns': ['normal_browsing', 'api_calls', 'file_access'],
                'user_agents': ['Chrome', 'Firefox', 'Safari', 'Edge']
            }
        }
    
    def generate_traffic_sample(self, attack_type, intensity='medium'):
        """Generate single traffic sample"""
        base_time = datetime.utcnow()
        
        if attack_type == 'benign':
            return self._generate_benign_traffic(base_time, intensity)
        else:
            return self._generate_attack_traffic(attack_type, base_time, intensity)
    
    def _generate_benign_traffic(self, timestamp, intensity):
        """Generate benign traffic patterns"""
        patterns = self.attack_patterns['benign']
        
        return {
            'timestamp': timestamp.isoformat(),
            'source_ip': self._generate_ip('benign'),
            'type': 'benign',
            'pattern': random.choice(patterns['patterns']),
            'user_agent': random.choice(patterns['user_agents']),
            'request_size': np.random.normal(1024, 256),
            'response_time': np.random.exponential(0.1),
            'threat_score': np.random.uniform(0.0, 0.2),
            'num_techniques': 0,
            'mutations': 0,
            'adversarial_noise': 0,
            'target_layer': random.randint(3, 5),
            'label': 0  # Benign
        }
    
    def _generate_attack_traffic(self, attack_type, timestamp, intensity):
        """Generate malicious traffic patterns"""
        patterns = self.attack_patterns[attack_type]
        
        intensity_multipliers = {'low': 0.5, 'medium': 1.0, 'high': 1.5, 'critical': 2.0}
        multiplier = intensity_multipliers.get(intensity, 1.0)
        
        base_threat = patterns['base_threat'] * multiplier
        threat_score = np.clip(np.random.normal(base_threat, 0.1), 0.0, 1.0)
        
        sample = {
            'timestamp': timestamp.isoformat(),
            'source_ip': self._generate_ip(attack_type),
            'type': attack_type,
            'threat_score': threat_score,
            'target_layer': self._get_target_layer(attack_type, intensity),
            'label': 1  # Malicious
        }
        
        if attack_type == 'sql_injection':
            sample.update({
                'technique': random.choice(patterns['techniques']),
                'payload': random.choice(patterns['payloads']),
                'num_techniques': random.randint(1, 4),
                'mutations': random.randint(0, 3),
                'adversarial_noise': 0
            })
        
        elif attack_type == 'xss':
            sample.update({
                'technique': random.choice(patterns['techniques']),
                'payload': random.choice(patterns['payloads']),
                'num_techniques': random.randint(1, 3),
                'mutations': random.randint(0, 2),
                'adversarial_noise': 0
            })
        
        elif attack_type == 'ddos':
            sample.update({
                'technique': random.choice(patterns['techniques']),
                'pattern': random.choice(patterns['patterns']),
                'request_rate': np.random.exponential(100) * multiplier,
                'num_techniques': random.randint(1, 2),
                'mutations': 0,
                'adversarial_noise': 0
            })
        
        elif attack_type == 'adversarial':
            sample.update({
                'technique': random.choice(patterns['techniques']),
                'noise_level': random.choice(patterns['noise_levels']),
                'num_techniques': random.randint(2, 5),
                'mutations': random.randint(3, 8),
                'adversarial_noise': random.randint(1, 5)
            })
        
        return sample
    
    def _generate_ip(self, attack_type):
        """Generate realistic IP addresses"""
        if attack_type == 'benign':
            # Corporate/home networks
            subnets = ['192.168.1', '10.0.0', '172.16.1']
        else:
            # Suspicious/foreign networks
            subnets = ['203.0.113', '198.51.100', '185.220.101']
        
        subnet = random.choice(subnets)
        host = random.randint(1, 254)
        return f"{subnet}.{host}"
    
    def _get_target_layer(self, attack_type, intensity):
        """Determine target layer based on attack sophistication"""
        layer_mapping = {
            'sql_injection': {'low': 4, 'medium': 3, 'high': 2, 'critical': 1},
            'xss': {'low': 4, 'medium': 3, 'high': 2, 'critical': 1},
            'ddos': {'low': 5, 'medium': 4, 'high': 3, 'critical': 2},
            'adversarial': {'low': 3, 'medium': 2, 'high': 1, 'critical': 0}
        }
        return layer_mapping.get(attack_type, {}).get(intensity, 3)
    
    def generate_dataset(self, samples_per_type=1000, include_boundary_cases=True):
        """Generate complete synthetic dataset"""
        dataset = []
        
        # Standard samples
        for attack_type in self.attack_patterns.keys():
            for intensity in ['low', 'medium', 'high']:
                count = samples_per_type if attack_type != 'benign' else samples_per_type * 2
                for _ in range(count):
                    sample = self.generate_traffic_sample(attack_type, intensity)
                    dataset.append(sample)
        
        # Boundary cases
        if include_boundary_cases:
            dataset.extend(self._generate_boundary_cases())
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        return pd.DataFrame(dataset)
    
    def _generate_boundary_cases(self):
        """Generate edge cases and boundary conditions"""
        boundary_cases = []
        
        # Traffic spikes
        spike_time = datetime.utcnow()
        for i in range(100):
            sample = self.generate_traffic_sample('ddos', 'critical')
            sample['timestamp'] = (spike_time + timedelta(seconds=i*0.1)).isoformat()
            sample['boundary_case'] = 'traffic_spike'
            boundary_cases.append(sample)
        
        # Burst patterns
        burst_time = datetime.utcnow() + timedelta(minutes=5)
        for burst in range(5):
            for i in range(20):
                sample = self.generate_traffic_sample('sql_injection', 'high')
                sample['timestamp'] = (burst_time + timedelta(seconds=burst*60 + i*0.5)).isoformat()
                sample['boundary_case'] = 'burst_pattern'
                boundary_cases.append(sample)
        
        # Evasive patterns (low and slow)
        evasive_time = datetime.utcnow() + timedelta(minutes=10)
        for i in range(50):
            sample = self.generate_traffic_sample('adversarial', 'low')
            sample['timestamp'] = (evasive_time + timedelta(minutes=i*2)).isoformat()
            sample['threat_score'] *= 0.3  # Make it subtle
            sample['boundary_case'] = 'evasive_pattern'
            boundary_cases.append(sample)
        
        return boundary_cases
    
    def generate_time_series_dataset(self, duration_hours=24, samples_per_hour=100):
        """Generate time-series dataset with realistic temporal patterns"""
        dataset = []
        start_time = datetime.utcnow()
        
        for hour in range(duration_hours):
            # Simulate daily patterns (more attacks at night)
            hour_of_day = hour % 24
            attack_probability = 0.3 + 0.4 * np.sin((hour_of_day - 6) * np.pi / 12)
            
            for sample_idx in range(samples_per_hour):
                timestamp = start_time + timedelta(hours=hour, minutes=sample_idx*0.6)
                
                if np.random.random() < attack_probability:
                    attack_type = np.random.choice(['sql_injection', 'xss', 'ddos', 'adversarial'], 
                                                 p=[0.3, 0.25, 0.25, 0.2])
                    intensity = np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.3, 0.2])
                else:
                    attack_type = 'benign'
                    intensity = 'medium'
                
                sample = self.generate_traffic_sample(attack_type, intensity)
                sample['timestamp'] = timestamp.isoformat()
                sample['hour_of_day'] = hour_of_day
                dataset.append(sample)
        
        return pd.DataFrame(dataset)

def save_datasets():
    """Generate and save datasets"""
    generator = SyntheticDatasetGenerator()
    
    # Standard dataset
    print("Generating standard dataset...")
    standard_df = generator.generate_dataset(samples_per_type=500)
    standard_df.to_csv('data/synthetic_dataset.csv', index=False)
    
    # Time series dataset
    print("Generating time series dataset...")
    timeseries_df = generator.generate_time_series_dataset(duration_hours=12, samples_per_hour=50)
    timeseries_df.to_csv('data/timeseries_dataset.csv', index=False)
    
    print(f"Standard dataset: {len(standard_df)} samples")
    print(f"Time series dataset: {len(timeseries_df)} samples")
    print("Attack type distribution:")
    print(standard_df['type'].value_counts())

if __name__ == '__main__':
    save_datasets()