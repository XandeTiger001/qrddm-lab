import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time
import random
from quantum_defense import QuantumEnhancedDefense

class FeatureExtractor:
    def __init__(self):
        self.attack_signatures = {
            'sql_injection': ['union', 'select', 'drop', 'insert', 'delete', '--', ';'],
            'xss': ['script', 'alert', 'onload', 'onerror', 'javascript:', '<', '>'],
            'ddos': ['flood', 'amplification', 'syn', 'udp', 'icmp'],
            'buffer_overflow': ['%n', 'AAAA', '\x90', 'shellcode', 'nop'],
            'path_traversal': ['../', '..\\', '%2e%2e', 'etc/passwd', 'boot.ini']
        }
    
    def extract_features(self, attack_data):
        """Extract 15 key features for classic ML"""
        features = []
        
        # Basic features (4)
        features.append(len(attack_data.get('payload', '')))
        features.append(sum(1 for c in attack_data.get('payload', '') if not c.isalnum()))
        features.append(attack_data.get('request_size', 1024))
        features.append(attack_data.get('response_time', 0.1))
        
        # Pattern matching features (5)
        payload = attack_data.get('payload', '').lower()
        for attack_type, signatures in self.attack_signatures.items():
            features.append(sum(1 for sig in signatures if sig in payload))
        
        # Behavioral features (6)
        features.append(attack_data.get('frequency', 1))
        features.append(self._get_ip_reputation(attack_data.get('source_ip', '')))
        features.append((time.time() % 86400) / 86400)  # Normalized hour
        features.append(attack_data.get('target_layer', 3))
        features.append(attack_data.get('num_techniques', 1))
        features.append(len(attack_data.get('payload', '')) / 100)  # Normalized payload length
        
        return np.array(features)
    
    def _get_ip_reputation(self, ip):
        """Simple IP reputation scoring"""
        if ip.startswith('192.168.') or ip.startswith('10.'):
            return 0.1  # Internal network
        elif ip.startswith('203.0.113.') or ip.startswith('198.51.100.'):
            return 0.9  # Known bad ranges
        else:
            return 0.5  # Unknown

class PhysicsEngine:
    def __init__(self, G=1.0, c=1.0):
        self.G = G  # Gravitational constant
        self.c = c  # Speed of light
    
    def calculate_schwarzschild_params(self, M):
        """Calculate Schwarzschild radius and metric"""
        r_s = 2 * self.G * M / (self.c ** 2)
        r = max(r_s + 0.1, 1.0)  # Observer distance
        phi = max(0.01, 1 - r_s / r)  # Schwarzschild metric
        return phi, r_s, r
    
    def calculate_threat_mass(self, features):
        """Convert features to threat mass M(r)"""
        # Normalize features first
        normalized_features = features / (np.max(features) + 1e-6)
        
        # Weighted combination of key features
        weights = np.array([0.1, 0.15, 0.05, 0.05, 0.2, 0.15, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.15, 0.2])
        M = np.dot(normalized_features, weights) * 5.0  # Scale to reasonable range
        return max(0.1, min(5.0, M))

class ClassicAIEnsemble:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        self.nn_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X, y):
        """Train both models"""
        X_scaled = self.scaler.fit_transform(X)
        self.rf_model.fit(X, y)
        self.nn_model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_proba(self, X):
        """Ensemble prediction with confidence"""
        if not self.is_trained:
            return np.array([[0.5, 0.5]] * len(X))
        
        X_scaled = self.scaler.transform(X)
        rf_proba = self.rf_model.predict_proba(X)
        nn_proba = self.nn_model.predict_proba(X_scaled)
        
        # Ensemble averaging
        ensemble_proba = (rf_proba + nn_proba) / 2
        return ensemble_proba

class DecisionPolicy:
    def __init__(self):
        self.thresholds = {
            'allow': 0.3,
            'monitor': 0.5,
            'throttle': 0.7,
            'block': 0.9
        }
    
    def make_decision(self, threat_prob, physics_params, confidence):
        """Make final decision based on ensemble + physics"""
        phi, r_s, r = physics_params
        
        # Physics-enhanced threat score
        physics_factor = (1 - phi) * 0.5  # Simplified physics enhancement
        enhanced_threat = threat_prob + physics_factor * threat_prob
        
        # Decision logic with confidence adjustment
        adjusted_threat = enhanced_threat * (0.5 + confidence * 0.5)
        
        if adjusted_threat >= self.thresholds['block']:
            return 'BLOCK', adjusted_threat
        elif adjusted_threat >= self.thresholds['throttle']:
            return 'THROTTLE', adjusted_threat
        elif adjusted_threat >= self.thresholds['monitor']:
            return 'MONITOR', adjusted_threat
        else:
            return 'ALLOW', adjusted_threat

class Tier1ClassicDefense:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.physics_engine = PhysicsEngine()
        self.ai_ensemble = ClassicAIEnsemble()
        self.decision_policy = DecisionPolicy()
        self.attack_history = []
    
    def train_system(self, training_data):
        """Train the classic AI models"""
        X = []
        y = []
        
        for attack in training_data:
            features = self.feature_extractor.extract_features(attack)
            X.append(features)
            y.append(attack['label'])
        
        X = np.array(X)
        y = np.array(y)
        
        self.ai_ensemble.train(X, y)
        print(f"Trained on {len(X)} samples")
    
    def analyze_attack(self, attack_data):
        """Complete Tier 1 analysis pipeline"""
        # Step 1: Feature extraction
        features = self.feature_extractor.extract_features(attack_data)
        
        # Step 2: Classic AI prediction
        proba = self.ai_ensemble.predict_proba([features])[0]
        threat_prob = proba[1]  # Probability of attack
        confidence = max(proba) - min(proba)  # Confidence measure
        
        # Step 3: Physics calculation
        M = self.physics_engine.calculate_threat_mass(features)
        phi, r_s, r = self.physics_engine.calculate_schwarzschild_params(M)
        
        # Step 4: Decision policy
        decision, enhanced_threat = self.decision_policy.make_decision(
            threat_prob, (phi, r_s, r), confidence
        )
        
        # Store for history
        result = {
            'features': features,
            'threat_probability': threat_prob,
            'confidence': confidence,
            'physics': {'M': M, 'phi': phi, 'r_s': r_s, 'r': r},
            'enhanced_threat': enhanced_threat,
            'decision': decision,
            'timestamp': time.time()
        }
        
        self.attack_history.append(result)
        return result
    
    def enable_quantum_mode(self):
        """Enable quantum-enhanced analysis"""
        self.quantum_defense = QuantumEnhancedDefense(self)
        return self.quantum_defense
    
    def quantum_analyze_attack(self, attack_data):
        """Quantum-enhanced attack analysis"""
        if hasattr(self, 'quantum_defense'):
            return self.quantum_defense.analyze_with_quantum_enhancement(attack_data)
        else:
            return self.analyze_attack(attack_data)

def generate_synthetic_attacks(num_attacks=75):
    """Generate synthetic attacks for testing"""
    attacks = []
    attack_types = ['sql_injection', 'xss', 'ddos', 'buffer_overflow', 'path_traversal', 'benign']
    
    payloads = {
        'sql_injection': ["' OR 1=1--", "'; DROP TABLE users--", "UNION SELECT * FROM admin"],
        'xss': ["<script>alert('xss')</script>", "javascript:alert(1)", "<img onerror=alert(1)>"],
        'ddos': ["flood_request", "syn_flood_packet", "udp_amplification"],
        'buffer_overflow': ["A" * 1000, "%n%n%n%n", "\\x90\\x90\\x90\\x90"],
        'path_traversal': ["../../../etc/passwd", "..\\..\\boot.ini", "%2e%2e%2f"],
        'benign': ["normal_request", "user_login", "file_download"]
    }
    
    ip_ranges = {
        'internal': ['192.168.1.{}', '10.0.0.{}'],
        'external': ['203.0.113.{}', '198.51.100.{}', '185.220.101.{}']
    }
    
    for i in range(num_attacks):
        attack_type = random.choice(attack_types)
        is_malicious = attack_type != 'benign'
        
        # Choose IP range based on attack type
        if is_malicious and random.random() > 0.3:
            ip_template = random.choice(ip_ranges['external'])
        else:
            ip_template = random.choice(ip_ranges['internal'])
        
        source_ip = ip_template.format(random.randint(1, 254))
        
        attack = {
            'type': attack_type,
            'payload': random.choice(payloads[attack_type]),
            'source_ip': source_ip,
            'request_size': random.randint(100, 5000),
            'response_time': random.uniform(0.01, 2.0),
            'frequency': random.randint(1, 100),
            'target_layer': random.randint(1, 7),
            'num_techniques': random.randint(1, 5),
            'label': 1 if is_malicious else 0
        }
        
        attacks.append(attack)
    
    return attacks