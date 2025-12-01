"""
TriSense Logic Engine - Quantum-Inspired Cybersecurity Assessment

Implements ternary logic for nuanced threat detection:
-1 = Shadow (incomplete data, noise, suspicious but not confirmed)
 0 = Normality (baseline behavior, no threat indicators)
+1 = Evidence (confirmed pattern, AI confident in prediction)

Inspired by quantum superposition - threats exist in uncertain states
until sufficient evidence collapses them into definitive classifications.
"""

import re
import math
from typing import Dict, Any, Tuple

class TriSenseEngine:
    def __init__(self):
        self.shadow_threshold = 0.3
        self.evidence_threshold = 0.7
        
    def analyze_payload_entropy(self, payload: str) -> int:
        """Entropy-based pattern detection"""
        if len(payload) < 10:
            return -1  # Shadow: insufficient data
        
        entropy = self._calculate_entropy(payload)
        if entropy > 4.5:
            return 1   # Evidence: high entropy indicates obfuscation
        elif entropy < 2.0:
            return 1   # Evidence: suspiciously low entropy (padding attacks)
        return 0       # Normality: normal entropy range
    
    def analyze_injection_patterns(self, payload: str) -> int:
        """SQL/XSS injection pattern detection"""
        high_confidence = [r"(?i)(union\s+select|drop\s+table|or\s+1\s*=\s*1)",
                          r"<script[^>]*>.*?</script>",
                          r"javascript\s*:",
                          r"(?i)(exec|system|cmd)\s*\("]
        
        suspicious = [r"['\";]", r"--", r"/\*.*?\*/", r"<[^>]+>"]
        
        for pattern in high_confidence:
            if re.search(pattern, payload):
                return 1  # Evidence: confirmed attack pattern
        
        shadow_count = sum(1 for pattern in suspicious if re.search(pattern, payload))
        if shadow_count >= 2:
            return -1  # Shadow: multiple suspicious indicators
        elif shadow_count == 1:
            return -1  # Shadow: single suspicious indicator
        
        return 0  # Normality: no injection patterns
    
    def analyze_request_velocity(self, rate: int, size: int) -> int:
        """Request rate and size anomaly detection"""
        if rate > 20 and size < 50:
            return 1   # Evidence: rapid small requests (brute force)
        elif rate > 15:
            return -1  # Shadow: elevated request rate
        elif size > 100:
            return -1  # Shadow: unusually large payload
        return 0       # Normality: normal request patterns
    
    def analyze_path_traversal(self, payload: str) -> int:
        """Directory traversal detection"""
        traversal_patterns = [r"\.\./", r"\.\.\\", r"etc/passwd", r"windows/system32"]
        
        for pattern in traversal_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                return 1  # Evidence: confirmed traversal attempt
        
        if ".." in payload or "system" in payload.lower():
            return -1  # Shadow: potential traversal indicators
        
        return 0  # Normality: no traversal patterns
    
    def synthesize_assessment(self, payload: str, rate: int, size: int) -> Tuple[int, Dict[str, Any]]:
        """Main TriSense assessment combining all analyses"""
        assessments = {
            'entropy': self.analyze_payload_entropy(payload),
            'injection': self.analyze_injection_patterns(payload),
            'velocity': self.analyze_request_velocity(rate, size),
            'traversal': self.analyze_path_traversal(payload)
        }
        
        evidence_count = sum(1 for v in assessments.values() if v == 1)
        shadow_count = sum(1 for v in assessments.values() if v == -1)
        
        # TriSense state transition rules
        if evidence_count >= 2:
            final_state = 1  # Evidence: multiple confirmations
        elif evidence_count == 1 and shadow_count == 0:
            final_state = 1  # Evidence: single strong indicator
        elif shadow_count >= 2:
            final_state = -1  # Shadow: multiple uncertainties
        elif evidence_count == 1 and shadow_count >= 1:
            final_state = -1  # Shadow: conflicting signals
        else:
            final_state = 0  # Normality: no significant indicators
        
        confidence = self._calculate_confidence(assessments, final_state)
        
        return final_state, {
            'assessments': assessments,
            'confidence': confidence,
            'state_meaning': self._get_state_meaning(final_state),
            'recommendation': self._get_recommendation(final_state, confidence)
        }
    
    def _calculate_entropy(self, data: str) -> float:
        """Shannon entropy calculation"""
        if not data:
            return 0
        
        freq = {}
        for char in data:
            freq[char] = freq.get(char, 0) + 1
        
        entropy = 0
        length = len(data)
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_confidence(self, assessments: Dict[str, int], final_state: int) -> float:
        """Calculate confidence based on assessment consensus"""
        matching = sum(1 for v in assessments.values() if v == final_state)
        total = len(assessments)
        return matching / total
    
    def _get_state_meaning(self, state: int) -> str:
        """Human-readable state interpretation"""
        meanings = {
            -1: "Shadow: Suspicious patterns detected, requires investigation",
            0: "Normality: Baseline behavior, no threat indicators",
            1: "Evidence: High confidence threat detected, immediate action required"
        }
        return meanings[state]
    
    def _get_recommendation(self, state: int, confidence: float) -> str:
        """Action recommendations based on TriSense state"""
        if state == 1:
            return "BLOCK and redirect to honeypot for analysis"
        elif state == -1:
            if confidence > 0.6:
                return "MONITOR closely, consider rate limiting"
            else:
                return "LOG for pattern analysis, continue monitoring"
        else:
            return "ALLOW with standard logging"

# Integration function for existing ML pipeline
def enhance_ml_prediction(payload: str, rate: int, size: int, ml_score: float) -> Dict[str, Any]:
    """Enhance binary ML predictions with TriSense ternary logic"""
    engine = TriSenseEngine()
    trisense_state, details = engine.synthesize_assessment(payload, rate, size)
    
    # Fusion of ML binary prediction with TriSense ternary logic
    if trisense_state == 1 and ml_score > 0.5:
        final_threat = True
        confidence = min(0.95, ml_score + details['confidence'] * 0.3)
    elif trisense_state == -1:
        final_threat = ml_score > 0.3  # Lower threshold for shadow cases
        confidence = ml_score * 0.8  # Reduced confidence for uncertain cases
    else:
        final_threat = ml_score > 0.7  # Higher threshold for normal cases
        confidence = ml_score
    
    return {
        'threat_detected': final_threat,
        'threat_score': confidence,
        'trisense_state': trisense_state,
        'trisense_details': details,
        'ml_score': ml_score,
        'fusion_logic': 'TriSense-Enhanced ML Detection'
    }