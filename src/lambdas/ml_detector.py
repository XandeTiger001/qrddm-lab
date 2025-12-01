import json
import pickle
import os
import sys
from datetime import datetime
from collections import Counter
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trisense_engine import enhance_ml_prediction

class MLThreatDetector:
    def __init__(self):
        self.threat_threshold = 0.7
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'threat_model.pkl')
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except:
            self.model = None
    
    def extract_features(self, request_data):
        body = str(request_data.get('body', ''))
        features = {
            'request_rate': request_data.get('rate', 0),
            'payload_size': len(body),
            'suspicious_patterns': self._detect_patterns(body),
            'entropy': self._calculate_entropy(body)
        }
        return features
    
    def _detect_patterns(self, body):
        body_lower = body.lower()
        patterns = ['select', 'drop', 'script', '../', 'union', 'exec']
        return sum(1 for p in patterns if p in body_lower)
    
    def _calculate_entropy(self, data):
        if not data:
            return 0
        counts = Counter(data)
        entropy = 0
        for count in counts.values():
            p_x = count / len(data)
            entropy += - p_x * math.log2(p_x)
        return entropy
    
    def predict_threat(self, features):
        if self.model:
            feature_vector = [[
                features['request_rate'],
                features['payload_size'],
                features['suspicious_patterns'],
                features['entropy']
            ]]
            return self.model.predict_proba(feature_vector)[0][1]
        else:
            score = (
                features['suspicious_patterns'] * 0.4 +
                min(features['request_rate'] / 100, 1) * 0.3 +
                min(features['payload_size'] / 10000, 1) * 0.2 +
                min(features['entropy'] / 8, 1) * 0.1
            )
            return min(score, 1.0)

def lambda_handler(event, context):
    detector = MLThreatDetector()
    
    request_data = {
        'body': event.get('body', ''),
        'headers': event.get('headers', {}),
        'rate': event.get('requestContext', {}).get('requestRate', 1)
    }
    
    features = detector.extract_features(request_data)
    threat_score = detector.predict_threat(features)
    
    is_threat = threat_score >= detector.threat_threshold
    
    # Enhanced with TriSense Logic Engine
    enhanced_result = enhance_ml_prediction(
        payload=request_data['body'],
        rate=features['request_rate'],
        size=features['payload_size'],
        ml_score=threat_score
    )
    
    response = {
        'timestamp': datetime.utcnow().isoformat(),
        'threat_detected': enhanced_result['threat_detected'],
        'threat_score': round(enhanced_result['threat_score'], 3),
        'trisense_state': enhanced_result['trisense_state'],
        'trisense_meaning': enhanced_result['trisense_details']['state_meaning'],
        'recommendation': enhanced_result['trisense_details']['recommendation'],
        'confidence': round(enhanced_result['trisense_details']['confidence'], 3),
        'features': features,
        'original_ml_score': round(threat_score, 3),
        'action': 'REDIRECT' if enhanced_result['threat_detected'] else 'ALLOW',
        'model_used': f"TriSense + {'RandomForest' if detector.model else 'Heuristic'}"
    }
    
    return {
        'statusCode': 403 if enhanced_result['threat_detected'] else 200,
        'body': json.dumps(response),
        'headers': {'Content-Type': 'application/json'}
    }
