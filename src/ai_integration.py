import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean, cityblock, minkowski, cosine
from scipy.stats import multivariate_normal
import json
import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class AttackType(Enum):
    SQL_INJECTION = 0
    XSS = 1
    DDOS = 2
    MALWARE = 3
    PHISHING = 4
    BRUTE_FORCE = 5
    PRIVILEGE_ESCALATION = 6
    DATA_EXFILTRATION = 7
    RANSOMWARE = 8
    APT = 9
    ZERO_DAY = 10
    SOCIAL_ENGINEERING = 11
    MAN_IN_MIDDLE = 12
    DNS_POISONING = 13
    BUFFER_OVERFLOW = 14
    ADVERSARIAL_ML = 15
    IOT_ATTACK = 16
    SUPPLY_CHAIN = 17
    INSIDER_THREAT = 18
    NORMAL = 19

@dataclass
class AttackFeatures:
    packet_size: float
    frequency: float
    port_diversity: float
    payload_entropy: float
    connection_duration: float
    geo_anomaly: float
    time_pattern: float
    protocol_anomaly: float

class LightweightAttackClassifier:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _build_model(self):
        """Lightweight CNN for 20 attack types"""
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(8,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(20, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train_quick(self, X_train, y_train):
        """Quick training with synthetic data"""
        try:
            X_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_scaled, y_train, epochs=5, batch_size=16, verbose=0)  # Reduced epochs and batch size
            self.is_trained = True
        except Exception as e:
            print(f"Warning: Classifier training failed: {e}")
            self.is_trained = False
    
    def predict_attack(self, features: AttackFeatures) -> Tuple[AttackType, float]:
        if not self.is_trained:
            return AttackType.NORMAL, 0.5
        
        feature_vector = np.array([[
            features.packet_size, features.frequency, features.port_diversity,
            features.payload_entropy, features.connection_duration, features.geo_anomaly,
            features.time_pattern, features.protocol_anomaly
        ]])
        
        X_scaled = self.scaler.transform(feature_vector)
        prediction = self.model.predict(X_scaled, verbose=0)
        
        attack_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][attack_idx])
        
        return AttackType(attack_idx), confidence

class AnomalyAutoencoder:
    def __init__(self, input_dim=8):
        self.input_dim = input_dim
        self.model = self._build_autoencoder()
        self.threshold = 0.1
        self.scaler = StandardScaler()
    
    def _build_autoencoder(self):
        """Lightweight autoencoder for anomaly detection"""
        input_layer = keras.layers.Input(shape=(self.input_dim,))
        encoded = keras.layers.Dense(4, activation='relu')(input_layer)
        encoded = keras.layers.Dense(2, activation='relu')(encoded)
        decoded = keras.layers.Dense(4, activation='relu')(encoded)
        decoded = keras.layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def train_on_normal(self, normal_data):
        """Train on normal traffic patterns"""
        try:
            X_scaled = self.scaler.fit_transform(normal_data)
            self.model.fit(X_scaled, X_scaled, epochs=20, batch_size=16, verbose=0)  # Reduced epochs and batch size
            
            # Set threshold based on reconstruction error
            reconstructed = self.model.predict(X_scaled, verbose=0)
            mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
            self.threshold = np.percentile(mse, 95)
        except Exception as e:
            print(f"Warning: Anomaly detector training failed: {e}")
            self.threshold = 0.1
    
    def detect_anomaly(self, features: AttackFeatures) -> Tuple[bool, float]:
        feature_vector = np.array([[
            features.packet_size, features.frequency, features.port_diversity,
            features.payload_entropy, features.connection_duration, features.geo_anomaly,
            features.time_pattern, features.protocol_anomaly
        ]])
        
        X_scaled = self.scaler.transform(feature_vector)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructed, 2))
        
        is_anomaly = mse > self.threshold
        anomaly_score = float(mse / self.threshold)
        
        return is_anomaly, anomaly_score

class MaliciousTrafficGenerator:
    def __init__(self):
        self.attack_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self):
        return {
            AttackType.SQL_INJECTION: {
                'packet_size': (200, 800), 'frequency': (0.1, 0.3),
                'payload_entropy': (0.7, 0.9), 'port_diversity': (0.1, 0.2)
            },
            AttackType.DDOS: {
                'packet_size': (64, 128), 'frequency': (0.8, 1.0),
                'payload_entropy': (0.1, 0.3), 'port_diversity': (0.9, 1.0)
            },
            AttackType.ADVERSARIAL_ML: {
                'packet_size': (500, 1200), 'frequency': (0.2, 0.4),
                'payload_entropy': (0.8, 1.0), 'port_diversity': (0.3, 0.5)
            }
        }
    
    def generate_attack_traffic(self, attack_type: AttackType, num_samples: int = 100) -> List[AttackFeatures]:
        if attack_type not in self.attack_patterns:
            attack_type = AttackType.NORMAL
        
        pattern = self.attack_patterns.get(attack_type, {
            'packet_size': (100, 500), 'frequency': (0.1, 0.5),
            'payload_entropy': (0.3, 0.7), 'port_diversity': (0.2, 0.6)
        })
        
        samples = []
        for _ in range(num_samples):
            features = AttackFeatures(
                packet_size=np.random.uniform(*pattern['packet_size']),
                frequency=np.random.uniform(*pattern['frequency']),
                port_diversity=np.random.uniform(*pattern['port_diversity']),
                payload_entropy=np.random.uniform(*pattern['payload_entropy']),
                connection_duration=np.random.uniform(0.1, 10.0),
                geo_anomaly=np.random.uniform(0.0, 1.0),
                time_pattern=np.random.uniform(0.0, 1.0),
                protocol_anomaly=np.random.uniform(0.0, 1.0)
            )
            samples.append(features)
        
        return samples

class DistanceBasedDetector:
    def __init__(self):
        self.normal_baseline = None
        self.covariance_matrix = None
        self.scaler = StandardScaler()
    
    def set_baseline(self, normal_features: List[AttackFeatures]):
        """Establish normal behavior baseline"""
        feature_matrix = np.array([[f.packet_size, f.frequency, f.port_diversity, f.payload_entropy,
                                   f.connection_duration, f.geo_anomaly, f.time_pattern, f.protocol_anomaly] 
                                  for f in normal_features])
        scaled_matrix = self.scaler.fit_transform(feature_matrix)
        self.normal_baseline = np.mean(scaled_matrix, axis=0)
        self.covariance_matrix = np.cov(scaled_matrix.T)
    
    def euclidean_anomaly_score(self, features: AttackFeatures) -> float:
        """Euclidean: detects 'how far' an event is from normal"""
        if self.normal_baseline is None:
            return 0.5
        
        feature_vector = np.array([features.packet_size, features.frequency, features.port_diversity,
                                  features.payload_entropy, features.connection_duration, features.geo_anomaly,
                                  features.time_pattern, features.protocol_anomaly])
        scaled_vector = self.scaler.transform([feature_vector])[0]
        return euclidean(scaled_vector, self.normal_baseline)
    
    def manhattan_anomaly_score(self, features: AttackFeatures) -> float:
        """Manhattan: resistant to outliers, captures differences 'line by line'"""
        if self.normal_baseline is None:
            return 0.5
        
        feature_vector = np.array([features.packet_size, features.frequency, features.port_diversity,
                                  features.payload_entropy, features.connection_duration, features.geo_anomaly,
                                  features.time_pattern, features.protocol_anomaly])
        scaled_vector = self.scaler.transform([feature_vector])[0]
        return cityblock(scaled_vector, self.normal_baseline)
    
    def minkowski_anomaly_score(self, features: AttackFeatures, p: float = 3) -> float:
        """Minkowski: regulate curvature, altering sensitivity"""
        if self.normal_baseline is None:
            return 0.5
        
        feature_vector = np.array([features.packet_size, features.frequency, features.port_diversity,
                                  features.payload_entropy, features.connection_duration, features.geo_anomaly,
                                  features.time_pattern, features.protocol_anomaly])
        scaled_vector = self.scaler.transform([feature_vector])[0]
        return minkowski(scaled_vector, self.normal_baseline, p)
    
    def cosine_anomaly_score(self, features: AttackFeatures) -> float:
        """Cosine: measures angle between patterns, great for similar behaviors"""
        if self.normal_baseline is None:
            return 0.5
        
        feature_vector = np.array([features.packet_size, features.frequency, features.port_diversity,
                                  features.payload_entropy, features.connection_duration, features.geo_anomaly,
                                  features.time_pattern, features.protocol_anomaly])
        scaled_vector = self.scaler.transform([feature_vector])[0]
        return cosine(scaled_vector, self.normal_baseline)
    
    def mahalanobis_anomaly_score(self, features: AttackFeatures) -> float:
        """Mahalanobis: powerful anomaly detection using system distribution"""
        if self.normal_baseline is None or self.covariance_matrix is None:
            return 0.5
        
        feature_vector = np.array([features.packet_size, features.frequency, features.port_diversity,
                                  features.payload_entropy, features.connection_duration, features.geo_anomaly,
                                  features.time_pattern, features.protocol_anomaly])
        scaled_vector = self.scaler.transform([feature_vector])[0]
        diff = scaled_vector - self.normal_baseline
        return np.sqrt(diff.T @ np.linalg.pinv(self.covariance_matrix) @ diff)
    
    def detect_with_distance(self, features: AttackFeatures) -> Dict[str, float]:
        """Compare all distance metrics for comprehensive pattern detection"""
        return {
            'euclidean_distance': self.euclidean_anomaly_score(features),
            'manhattan_distance': self.manhattan_anomaly_score(features),
            'minkowski_distance': self.minkowski_anomaly_score(features),
            'cosine_distance': self.cosine_anomaly_score(features),
            'mahalanobis_distance': self.mahalanobis_anomaly_score(features)
        }

class EventHorizonRecovery:
    def __init__(self, uncertainty_threshold=0.3):
        self.uncertainty_threshold = uncertainty_threshold
        self.lost_information = []
        self.recovery_model = self._build_recovery_model()
        self.hawking_radiation = []
    
    def _build_recovery_model(self):
        """Quantum-inspired recovery network"""
        model = keras.Sequential([
            keras.layers.Dense(16, activation='tanh', input_shape=(8,)),
            keras.layers.Dense(8, activation='sigmoid'),
            keras.layers.Dense(8, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def detect_lost_information(self, original: AttackFeatures, reconstructed: np.ndarray) -> Dict[str, float]:
        """Identify missing/corrupted information in reconstruction"""
        original_vector = np.array([original.packet_size, original.frequency, original.port_diversity,
                                   original.payload_entropy, original.connection_duration, original.geo_anomaly,
                                   original.time_pattern, original.protocol_anomaly])
        
        information_loss = np.abs(original_vector - reconstructed.flatten())
        uncertainty_mask = information_loss > self.uncertainty_threshold
        
        lost_bits = {
            'packet_size_loss': information_loss[0] if uncertainty_mask[0] else 0,
            'frequency_loss': information_loss[1] if uncertainty_mask[1] else 0,
            'entropy_loss': information_loss[3] if uncertainty_mask[3] else 0,
            'total_information_loss': np.sum(information_loss[uncertainty_mask])
        }
        
        if lost_bits['total_information_loss'] > 0:
            self.lost_information.append({
                'timestamp': datetime.datetime.now(),
                'original': original_vector,
                'reconstructed': reconstructed.flatten(),
                'loss_vector': information_loss
            })
        
        return lost_bits
    
    def hawking_radiation_recovery(self, steps=5) -> List[Dict[str, float]]:
        """Gradual information recovery like virtual particles escaping horizon"""
        if not self.lost_information:
            return []
        
        recovered_particles = []
        latest_loss = self.lost_information[-1]
        
        for step in range(steps):
            escape_probability = np.exp(-step * 0.5)
            prediction_input = latest_loss['reconstructed'].reshape(1, -1)
            recovered_bits = self.recovery_model.predict(prediction_input, verbose=0).flatten()
            
            uncertainty_factor = 1.0 - (step / steps)
            recovered_with_uncertainty = recovered_bits + np.random.normal(0, uncertainty_factor * 0.1, 8)
            
            particle = {
                'step': step,
                'escape_probability': escape_probability,
                'recovered_information': np.sum(np.abs(recovered_with_uncertainty - latest_loss['reconstructed'])),
                'uncertainty_level': uncertainty_factor,
                'virtual_particle': recovered_with_uncertainty.tolist()
            }
            
            recovered_particles.append(particle)
            self.hawking_radiation.append(particle)
        
        return recovered_particles
    
    def train_recovery_oracle(self, training_pairs: List[Tuple[AttackFeatures, np.ndarray]]):
        """Train the information recovery predictor"""
        X_train = []
        y_train = []
        
        for original, corrupted in training_pairs:
            original_vector = np.array([original.packet_size, original.frequency, original.port_diversity,
                                       original.payload_entropy, original.connection_duration, original.geo_anomaly,
                                       original.time_pattern, original.protocol_anomaly])
            X_train.append(corrupted.flatten())
            y_train.append(original_vector)
        
        self.recovery_model.fit(np.array(X_train), np.array(y_train), epochs=20, verbose=0)

class CyberEinsteinLogger:
    def __init__(self):
        self.insights = []
        self.pattern_memory = {}
        self.event_horizon = EventHorizonRecovery()
    
    def analyze_and_log(self, attack_data: Dict, classification: AttackType, 
                       confidence: float, anomaly_score: float, 
                       original_features: AttackFeatures = None, 
                       reconstructed_data: np.ndarray = None) -> str:
        """AI-generated contextual logging with Event Horizon Recovery"""
        timestamp = datetime.datetime.now().isoformat()
        
        # Event Horizon: Detect lost information and recover virtual particles
        information_recovery = {}
        if original_features and reconstructed_data is not None:
            lost_info = self.event_horizon.detect_lost_information(original_features, reconstructed_data)
            if lost_info['total_information_loss'] > 0:
                hawking_particles = self.event_horizon.hawking_radiation_recovery()
                information_recovery = {
                    'information_loss_detected': lost_info,
                    'virtual_particles_escaped': len(hawking_particles),
                    'recovered_bits': sum(p['recovered_information'] for p in hawking_particles)
                }
        
        # Generate Einstein-style insight
        insight = self._generate_insight(attack_data, classification, confidence, anomaly_score, information_recovery)
        
        log_entry = {
            'timestamp': timestamp,
            'attack_type': classification.name,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'insight': insight,
            'threat_level': self._assess_threat_level(confidence, anomaly_score),
            'recommended_action': self._recommend_action(classification, confidence),
            'event_horizon_recovery': information_recovery
        }
        
        self.insights.append(log_entry)
        return self._format_log_entry(log_entry)
    
    def _generate_insight(self, attack_data: Dict, classification: AttackType, 
                         confidence: float, anomaly_score: float, 
                         recovery_info: Dict = None) -> str:
        """Generate contextual AI insights with Event Horizon context"""
        base_patterns = {
            AttackType.SQL_INJECTION: "Database query manipulation detected with {:.1%} certainty",
            AttackType.DDOS: "Coordinated traffic flood overwhelming system resources",
            AttackType.ADVERSARIAL_ML: "AI model poisoning attempt with sophisticated evasion techniques",
            AttackType.RANSOMWARE: "Encryption-based extortion pattern identified",
            AttackType.APT: "Advanced persistent threat exhibiting stealth characteristics"
        }
        
        base_insight = base_patterns.get(classification, f"{classification.name} pattern detected")
        
        # Add Event Horizon recovery context
        if recovery_info and recovery_info.get('virtual_particles_escaped', 0) > 0:
            base_insight += f" - {recovery_info['virtual_particles_escaped']} virtual particles escaped event horizon, {recovery_info['recovered_bits']:.2f} bits recovered"
        
        # Add contextual details
        if anomaly_score > 2.0:
            base_insight += " - HIGHLY ANOMALOUS behavior exceeding normal variance"
        elif confidence > 0.9:
            base_insight += " - High confidence classification with clear attack signatures"
        
        # Add temporal context
        source_ip = attack_data.get('source_ip', 'unknown')
        if source_ip in self.pattern_memory:
            base_insight += f" - Repeat offender from {source_ip}"
        
        return base_insight.format(confidence)
    
    def _assess_threat_level(self, confidence: float, anomaly_score: float) -> str:
        combined_score = confidence * anomaly_score
        if combined_score > 1.5:
            return "CRITICAL"
        elif combined_score > 0.8:
            return "HIGH"
        elif combined_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _recommend_action(self, classification: AttackType, confidence: float) -> str:
        if confidence > 0.8:
            actions = {
                AttackType.SQL_INJECTION: "Block source IP, sanitize database inputs",
                AttackType.DDOS: "Activate rate limiting, engage DDoS mitigation",
                AttackType.ADVERSARIAL_ML: "Isolate ML models, retrain with adversarial examples",
                AttackType.RANSOMWARE: "Immediate isolation, activate backup protocols"
            }
            return actions.get(classification, "Monitor closely, prepare countermeasures")
        else:
            return "Continue monitoring, gather additional evidence"
    
    def _format_log_entry(self, entry: Dict) -> str:
        return (f"[{entry['timestamp']}] 🧠 CYBER EINSTEIN: {entry['insight']} "
                f"| Threat: {entry['threat_level']} | Action: {entry['recommended_action']}")

class IntegratedAIDefense:
    def __init__(self):
        self.classifier = LightweightAttackClassifier()
        self.anomaly_detector = AnomalyAutoencoder()
        self.traffic_generator = MaliciousTrafficGenerator()
        self.logger = CyberEinsteinLogger()
        self._initialize_models()
    
    def _initialize_models(self):
        """Quick initialization with synthetic data"""
        try:
            # Generate minimal training data
            normal_data = []
            attack_data = []
            labels = []
            
            # Only train on a few key attack types to speed up initialization
            key_types = [AttackType.SQL_INJECTION, AttackType.DDOS, AttackType.ADVERSARIAL_ML, AttackType.NORMAL]
            
            for attack_type in key_types:
                samples = self.traffic_generator.generate_attack_traffic(attack_type, 10)  # Reduced samples
                for sample in samples:
                    feature_vector = [
                        sample.packet_size, sample.frequency, sample.port_diversity,
                        sample.payload_entropy, sample.connection_duration, sample.geo_anomaly,
                        sample.time_pattern, sample.protocol_anomaly
                    ]
                    attack_data.append(feature_vector)
                    labels.append(attack_type.value)
                    
                    if attack_type == AttackType.NORMAL:
                        normal_data.append(feature_vector)
            
            # Train models with reduced data
            if len(attack_data) > 0:
                self.classifier.train_quick(np.array(attack_data), np.array(labels))
            if len(normal_data) > 0:
                self.anomaly_detector.train_on_normal(np.array(normal_data))
        except Exception as e:
            print(f"Warning: AI model initialization failed: {e}")
            # Continue without AI models
    
    def analyze_traffic(self, attack_data: Dict) -> Dict:
        """Complete AI analysis pipeline"""
        # Extract features
        features = AttackFeatures(
            packet_size=attack_data.get('packet_size', 500),
            frequency=attack_data.get('frequency', 0.3),
            port_diversity=attack_data.get('port_diversity', 0.4),
            payload_entropy=attack_data.get('payload_entropy', 0.6),
            connection_duration=attack_data.get('connection_duration', 2.0),
            geo_anomaly=attack_data.get('geo_anomaly', 0.2),
            time_pattern=attack_data.get('time_pattern', 0.3),
            protocol_anomaly=attack_data.get('protocol_anomaly', 0.1)
        )
        
        # AI Analysis
        attack_type, confidence = self.classifier.predict_attack(features)
        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(features)
        
        # Generate intelligent log
        log_message = self.logger.analyze_and_log(attack_data, attack_type, confidence, anomaly_score)
        
        return {
            'ai_classification': attack_type.name,
            'confidence': round(confidence, 4),
            'is_anomaly': is_anomaly,
            'anomaly_score': round(anomaly_score, 4),
            'ai_insight': log_message,
            'threat_assessment': self.logger._assess_threat_level(confidence, anomaly_score)
        }

def demo_ai_integration():
    print("🤖 AI-Powered Cyber Defense Integration")
    print("=" * 60)
    
    ai_defense = IntegratedAIDefense()
    
    test_scenarios = [
        {
            'name': 'Suspicious SQL Traffic',
            'packet_size': 650, 'frequency': 0.25, 'port_diversity': 0.15,
            'payload_entropy': 0.85, 'connection_duration': 1.5,
            'geo_anomaly': 0.3, 'time_pattern': 0.4, 'protocol_anomaly': 0.2,
            'source_ip': '192.168.1.100'
        },
        {
            'name': 'DDoS Attack Pattern',
            'packet_size': 80, 'frequency': 0.95, 'port_diversity': 0.95,
            'payload_entropy': 0.2, 'connection_duration': 0.1,
            'geo_anomaly': 0.8, 'time_pattern': 0.9, 'protocol_anomaly': 0.7,
            'source_ip': '10.0.0.50'
        },
        {
            'name': 'Adversarial ML Attack',
            'packet_size': 900, 'frequency': 0.35, 'port_diversity': 0.4,
            'payload_entropy': 0.95, 'connection_duration': 3.0,
            'geo_anomaly': 0.6, 'time_pattern': 0.2, 'protocol_anomaly': 0.8,
            'source_ip': '172.16.0.20'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🔍 Analyzing: {scenario['name']}")
        result = ai_defense.analyze_traffic(scenario)
        
        print(f"   🎯 AI Classification: {result['ai_classification']} ({result['confidence']:.1%})")
        print(f"   🚨 Anomaly Detection: {'YES' if result['is_anomaly'] else 'NO'} (score: {result['anomaly_score']:.2f})")
        print(f"   ⚡ Threat Level: {result['threat_assessment']}")
        print(f"   🧠 {result['ai_insight']}")
    
    print(f"\n{'='*60}")
    print("🎲 Generating Malicious Traffic Samples...")
    
    generator = MaliciousTrafficGenerator()
    ddos_samples = generator.generate_attack_traffic(AttackType.DDOS, 3)
    
    for i, sample in enumerate(ddos_samples):
        print(f"\n   Sample {i+1}: DDoS Pattern")
        print(f"   • Packet Size: {sample.packet_size:.0f}")
        print(f"   • Frequency: {sample.frequency:.2f}")
        print(f"   • Entropy: {sample.payload_entropy:.2f}")

if __name__ == '__main__':
    demo_ai_integration()