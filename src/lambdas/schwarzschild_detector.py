import json
import sys
import os
from datetime import datetime
from collections import Counter
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schwarzschild_defense import SchwarzschildDefense, ThreatLevel

class SchwarzschildThreatDetector:
    def __init__(self):
        self.defense = SchwarzschildDefense(D=1.5, c=1.0, T_shadow=0.5, T_crit=0.2)
    
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
    
    def calculate_threat_score(self, features):
        """Score heurístico simples"""
        score = (
            features['suspicious_patterns'] * 0.4 +
            min(features['request_rate'] / 100, 1) * 0.3 +
            min(features['payload_size'] / 10000, 1) * 0.2 +
            min(features['entropy'] / 8, 1) * 0.1
        )
        return min(score, 1.0)

def lambda_handler(event, context):
    detector = SchwarzschildThreatDetector()
    
    request_data = {
        'body': event.get('body', ''),
        'headers': event.get('headers', {}),
        'rate': event.get('requestContext', {}).get('requestRate', 1)
    }
    
    features = detector.extract_features(request_data)
    threat_score = detector.calculate_threat_score(features)
    
    # Analysis Schwarzschild
    schwarzschild_result = detector.defense.analyze_event(
        threat_score=threat_score,
        payload_size=features['payload_size'],
        request_rate=features['request_rate'],
        time_to_impact=1.0,
        network_hops=1
    )
    
    is_threat = schwarzschild_result['classification']['value'] >= 0
    
    response = {
        'timestamp': datetime.utcnow().isoformat(),
        'threat_detected': is_threat,
        'threat_score': round(threat_score, 3),
        'features': features,
        'schwarzschild': schwarzschild_result['metrics'],
        'classification': schwarzschild_result['classification'],
        'action': schwarzschild_result['action']
    }
    
    return {
        'statusCode': 403 if is_threat else 200,
        'body': json.dumps(response),
        'headers': {'Content-Type': 'application/json'}
    }
