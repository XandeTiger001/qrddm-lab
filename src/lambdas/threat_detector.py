import json
import numpy as np
from datetime import datetime

class EventHorizonDetector:
    def __init__(self):
        self.threat_threshold = 0.7
        
    def extract_features(self, request_data):
        """Extract features from incoming request"""
        features = {
            'request_rate': request_data.get('rate', 0),
            'payload_size': len(str(request_data.get('body', ''))),
            'suspicious_patterns': self._detect_patterns(request_data),
            'entropy': self._calculate_entropy(str(request_data.get('body', '')))

        }
        return features
    
    def _detect_patterns(self, data):
        """Detect SQL injection, XSS, etc."""
        body = str(data.get('body', '')).lower()
        patterns = ['select', 'drop', 'script', '../', 'union', 'exec']
        return sum(1 for p in patterns if p in body)
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy for anomaly detection"""
        if not data:
            return 0
        from collections import Counter
        import math
        counts = Counter(data)
        entropy = 0
        for count in counts.values():
            p_x = count / len(data)
            entropy += - p_x * math.log2(p_x)
        return entropy
    
    def predict_threat(self, features):
        """Simplified ML prediction (replace with trained model)"""
        score = (
            features['suspicious_patterns'] * 0.4 +
            min(features['request_rate'] / 100, 1) * 0.3 +
            min(features['payload_size'] / 10000, 1) * 0.2 +
            min(features['entropy'] / 8, 1) * 0.1
        )
        return min(score, 1.0)

def lambda_handler(event, context):
    detector = EventHorizonDetector()
    
    request_data = {
        'body': event.get('body', ''),
        'headers': event.get('headers', {}),
        'rate': event.get('requestContext', {}).get('requestRate', 1)
    }
    
    features = detector.extract_features(request_data)
    threat_score = detector.predict_threat(features)
    
    is_threat = threat_score >= detector.threat_threshold
    
    response = {
        'timestamp': datetime.utcnow().isoformat(),
        'threat_detected': is_threat,
        'threat_score': round(threat_score, 3),
        'features': features,
        'action': 'REDIRECT' if is_threat else 'ALLOW'
    }
    
    return {
        'statusCode': 403 if is_threat else 200,
        'body': json.dumps(response),
        'headers': {'Content-Type': 'application/json'}
    }
