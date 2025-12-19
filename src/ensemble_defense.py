import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import pickle
import os
from adaptive_mass import AdaptiveMassCalculator, DefenseMode

class EnsembleDefenseSystem:
    def __init__(self, w_ai=0.6, w_physics=0.4, calibration_method='platt'):
        self.w_ai = w_ai
        self.w_physics = w_physics
        self.calibration_method = calibration_method
        
        # Physics component
        self.mass_calculator = AdaptiveMassCalculator(mode=DefenseMode.BALANCED)
        
        # AI component
        self.ai_classifier = None
        self.calibrated_classifier = None
        self.is_trained = False
        
        # Calibration
        self.calibrator = None
        
    def _calculate_physics_score(self, attack_data):
        """Calculate normalized physics-based score from Schwarzschild metric"""
        result = self.mass_calculator.analyze_attack_with_history(attack_data)
        M_total = result['M_total']
        
        # Calculate Schwarzschild radius and phi
        r = attack_data.get('distance_to_core', 0.5)
        r_s = 2 * M_total  # Simplified: G=c=1
        
        if r <= r_s:
            phi_score = 0.0  # Inside event horizon
        else:
            phi = 1 - (2 * M_total) / r
            phi_score = max(0, phi)  # Normalize to [0,1]
        
        # Convert to threat probability (invert phi)
        physics_prob = 1 - phi_score
        
        return {
            'physics_prob': physics_prob,
            'M_total': M_total,
            'phi': phi_score,
            'r_schwarzschild': r_s,
            'inside_horizon': r <= r_s
        }
    
    def _extract_ai_features(self, attack_data):
        """Extract features for AI classifier"""
        return np.array([
            attack_data.get('threat_score', 0.5),
            len(attack_data.get('type', '')),
            attack_data.get('num_techniques', 1),
            attack_data.get('mutations', 0),
            attack_data.get('adversarial_noise', 0),
            attack_data.get('target_layer', 3),
            1.0 - attack_data.get('distance_to_core', 0.5)
        ]).reshape(1, -1)
    
    def train_ai_component(self, training_data):
        """Train AI classifier with calibration"""
        X = []
        y = []
        
        for data in training_data:
            features = self._extract_ai_features(data).flatten()
            X.append(features)
            y.append(1 if data.get('is_attack', False) else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train base classifier
        self.ai_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ai_classifier.fit(X, y)
        
        # Apply calibration with minimal CV
        cv_folds = min(3, len(np.unique(y)) * 2, len(y) // 2)
        cv_folds = max(2, cv_folds)
        
        if self.calibration_method == 'platt':
            self.calibrated_classifier = CalibratedClassifierCV(
                self.ai_classifier, method='sigmoid', cv=cv_folds
            )
        else:  # isotonic
            self.calibrated_classifier = CalibratedClassifierCV(
                self.ai_classifier, method='isotonic', cv=cv_folds
            )
        
        self.calibrated_classifier.fit(X, y)
        self.is_trained = True
        
        return {'accuracy': self.calibrated_classifier.score(X, y)}
    
    def _get_ai_probability(self, attack_data):
        """Get calibrated AI probability"""
        if not self.is_trained:
            return 0.5  # Default probability
        
        features = self._extract_ai_features(attack_data)
        prob = self.calibrated_classifier.predict_proba(features)[0][1]
        return prob
    
    def ensemble_score(self, attack_data):
        """Combine AI + Physics with ensemble scoring"""
        # Get AI probability (calibrated)
        ai_prob = self._get_ai_probability(attack_data)
        
        # Get physics probability
        physics_result = self._calculate_physics_score(attack_data)
        physics_prob = physics_result['physics_prob']
        
        # Ensemble combination
        combined_score = self.w_ai * ai_prob + self.w_physics * physics_prob
        
        # Calculate confidence based on agreement
        agreement = 1 - abs(ai_prob - physics_prob)
        confidence = min(0.95, 0.5 + 0.5 * agreement)
        
        return {
            'ensemble_score': combined_score,
            'ai_probability': ai_prob,
            'physics_probability': physics_prob,
            'confidence': confidence,
            'agreement': agreement,
            'physics_details': physics_result,
            'weights': {'w_ai': self.w_ai, 'w_physics': self.w_physics}
        }
    
    def classify_with_confidence(self, attack_data, threshold=0.5):
        """Classification with confidence intervals"""
        result = self.ensemble_score(attack_data)
        
        is_threat = result['ensemble_score'] > threshold
        confidence = result['confidence']
        
        # Adjust threshold based on confidence
        if confidence < 0.7:
            # Low confidence - be more conservative
            adjusted_threshold = threshold + 0.1
            is_threat = result['ensemble_score'] > adjusted_threshold
        
        return {
            'is_threat': is_threat,
            'threat_probability': result['ensemble_score'],
            'confidence': confidence,
            'threshold_used': threshold if confidence >= 0.7 else threshold + 0.1,
            'recommendation': self._get_recommendation(result),
            'ensemble_details': result
        }
    
    def _get_recommendation(self, ensemble_result):
        """Get action recommendation based on ensemble result"""
        score = ensemble_result['ensemble_score']
        confidence = ensemble_result['confidence']
        
        if score > 0.8 and confidence > 0.8:
            return "BLOCK_IMMEDIATE"
        elif score > 0.6 and confidence > 0.7:
            return "MONITOR_CLOSELY"
        elif score > 0.4:
            return "LOG_AND_WATCH"
        else:
            return "ALLOW"
    
    def update_weights(self, feedback_data):
        """Adaptive weight adjustment based on feedback"""
        ai_correct = sum(1 for f in feedback_data if f['ai_correct'])
        physics_correct = sum(1 for f in feedback_data if f['physics_correct'])
        total = len(feedback_data)
        
        if total > 0:
            ai_accuracy = ai_correct / total
            physics_accuracy = physics_correct / total
            
            # Normalize weights based on performance
            total_accuracy = ai_accuracy + physics_accuracy
            if total_accuracy > 0:
                self.w_ai = ai_accuracy / total_accuracy
                self.w_physics = physics_accuracy / total_accuracy
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'ai_classifier': self.ai_classifier,
            'calibrated_classifier': self.calibrated_classifier,
            'w_ai': self.w_ai,
            'w_physics': self.w_physics,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load trained model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.ai_classifier = model_data['ai_classifier']
            self.calibrated_classifier = model_data['calibrated_classifier']
            self.w_ai = model_data['w_ai']
            self.w_physics = model_data['w_physics']
            self.is_trained = model_data['is_trained']
            return True
        return False

def demo_ensemble_system():
    """Demo ensemble scoring with calibration"""
    print("=== Ensemble Defense: AI + Physics with Calibration ===\n")
    
    # Create ensemble system
    ensemble = EnsembleDefenseSystem(w_ai=0.6, w_physics=0.4)
    
    # Generate training data
    training_data = [
        # Attacks
        {'type': 'sql_injection', 'threat_score': 0.9, 'num_techniques': 3, 'mutations': 5, 
         'adversarial_noise': 2, 'target_layer': 0, 'distance_to_core': 0.1, 'is_attack': True},
        {'type': 'ddos', 'threat_score': 0.95, 'num_techniques': 2, 'mutations': 0,
         'adversarial_noise': 0, 'target_layer': 1, 'distance_to_core': 0.2, 'is_attack': True},
        {'type': 'xss', 'threat_score': 0.7, 'num_techniques': 2, 'mutations': 3,
         'adversarial_noise': 1, 'target_layer': 2, 'distance_to_core': 0.4, 'is_attack': True},
        {'type': 'buffer_overflow', 'threat_score': 0.85, 'num_techniques': 4, 'mutations': 2,
         'adversarial_noise': 1, 'target_layer': 0, 'distance_to_core': 0.15, 'is_attack': True},
        {'type': 'path_traversal', 'threat_score': 0.75, 'num_techniques': 2, 'mutations': 1,
         'adversarial_noise': 0, 'target_layer': 1, 'distance_to_core': 0.3, 'is_attack': True},
        {'type': 'adversarial_ml', 'threat_score': 0.8, 'num_techniques': 3, 'mutations': 4,
         'adversarial_noise': 5, 'target_layer': 0, 'distance_to_core': 0.2, 'is_attack': True},
        
        # Normal traffic
        {'type': 'normal', 'threat_score': 0.1, 'num_techniques': 0, 'mutations': 0,
         'adversarial_noise': 0, 'target_layer': 5, 'distance_to_core': 0.9, 'is_attack': False},
        {'type': 'normal', 'threat_score': 0.2, 'num_techniques': 0, 'mutations': 0,
         'adversarial_noise': 0, 'target_layer': 4, 'distance_to_core': 0.8, 'is_attack': False},
        {'type': 'normal', 'threat_score': 0.15, 'num_techniques': 0, 'mutations': 0,
         'adversarial_noise': 0, 'target_layer': 6, 'distance_to_core': 0.95, 'is_attack': False},
        {'type': 'normal', 'threat_score': 0.05, 'num_techniques': 0, 'mutations': 0,
         'adversarial_noise': 0, 'target_layer': 7, 'distance_to_core': 0.98, 'is_attack': False},
        {'type': 'normal', 'threat_score': 0.25, 'num_techniques': 1, 'mutations': 0,
         'adversarial_noise': 0, 'target_layer': 3, 'distance_to_core': 0.7, 'is_attack': False},
        {'type': 'normal', 'threat_score': 0.3, 'num_techniques': 0, 'mutations': 0,
         'adversarial_noise': 0, 'target_layer': 4, 'distance_to_core': 0.75, 'is_attack': False}
    ]
    
    # Train the system
    print("Training ensemble system...")
    train_result = ensemble.train_ai_component(training_data)
    print(f"Training accuracy: {train_result['accuracy']:.3f}\n")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Sophisticated SQL Injection',
            'type': 'sql_injection',
            'threat_score': 0.85,
            'source_ip': '192.168.1.100',
            'target_layer': 0,
            'distance_to_core': 0.15,
            'num_techniques': 4,
            'mutations': 6,
            'adversarial_noise': 3
        },
        {
            'name': 'Suspicious but Uncertain',
            'type': 'unknown',
            'threat_score': 0.55,
            'source_ip': '10.0.0.50',
            'target_layer': 2,
            'distance_to_core': 0.45,
            'num_techniques': 1,
            'mutations': 1,
            'adversarial_noise': 0
        },
        {
            'name': 'Normal Traffic',
            'type': 'normal',
            'threat_score': 0.15,
            'source_ip': '172.16.0.10',
            'target_layer': 4,
            'distance_to_core': 0.85,
            'num_techniques': 0,
            'mutations': 0,
            'adversarial_noise': 0
        }
    ]
    
    print("Analyzing test scenarios...\n")
    
    for scenario in test_scenarios:
        result = ensemble.classify_with_confidence(scenario)
        
        print(f"Scenario: {scenario['name']}")
        print(f"  AI Probability: {result['ensemble_details']['ai_probability']:.3f}")
        print(f"  Physics Probability: {result['ensemble_details']['physics_probability']:.3f}")
        print(f"  Ensemble Score: {result['threat_probability']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Agreement: {result['ensemble_details']['agreement']:.3f}")
        print(f"  Classification: {'THREAT' if result['is_threat'] else 'SAFE'}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Threshold Used: {result['threshold_used']:.3f}")
        
        physics = result['ensemble_details']['physics_details']
        print(f"  Physics Details: M={physics['M_total']:.3f}, phi={physics['phi']:.3f}")
        print()

if __name__ == '__main__':
    demo_ensemble_system()