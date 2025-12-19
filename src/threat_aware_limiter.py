from action_limits import TieredActionLimiter, ActionLevel
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

class ThreatAwareLimiter(TieredActionLimiter):
    """Enhanced action limiter that considers threat scores"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.threat_scores = defaultdict(list)
        self.high_threat_threshold = 0.7
        self.critical_threat_threshold = 0.9
    
    def _clean_old_records(self, source_ip):
        """Remove records outside the time window"""
        super()._clean_old_records(source_ip)
        cutoff = datetime.utcnow() - timedelta(minutes=self.config['window_minutes'])
        self.threat_scores[source_ip] = [
            (ts, score) for ts, score in self.threat_scores[source_ip] if ts > cutoff
        ]
    
    def _calculate_threat_weighted_count(self, source_ip):
        """Calculate request count weighted by threat scores"""
        self._clean_old_records(source_ip)
        
        threat_data = self.threat_scores[source_ip]
        if not threat_data:
            return 0
        
        # Weight requests by threat score
        weighted_count = sum(score for _, score in threat_data)
        return weighted_count
    
    def _determine_action_level(self, source_ip, request_count, threat_score=0.5):
        """Enhanced action level determination with threat awareness"""
        multiplier = self._calculate_escalation_multiplier(source_ip)
        weighted_count = self._calculate_threat_weighted_count(source_ip)
        
        # Immediate escalation for high-threat requests
        if threat_score >= self.critical_threat_threshold:
            if weighted_count >= 2:  # Multiple critical threats
                return ActionLevel.BLOCK
            elif weighted_count >= 1:
                return ActionLevel.REDIRECT
            else:
                return ActionLevel.THROTTLE
        
        elif threat_score >= self.high_threat_threshold:
            if weighted_count >= 3:
                return ActionLevel.REDIRECT
            elif weighted_count >= 2:
                return ActionLevel.THROTTLE
            else:
                return ActionLevel.MONITOR
        
        # Standard thresholds for lower threat scores
        thresholds = {
            ActionLevel.MONITOR: self.config['monitor_threshold'],
            ActionLevel.THROTTLE: max(3, self.config['throttle_threshold'] / multiplier),
            ActionLevel.REDIRECT: max(5, self.config['redirect_threshold'] / multiplier),
            ActionLevel.BLOCK: max(8, self.config['block_threshold'] / multiplier)
        }
        
        effective_count = max(request_count, weighted_count)
        
        if effective_count >= thresholds[ActionLevel.BLOCK]:
            return ActionLevel.BLOCK
        elif effective_count >= thresholds[ActionLevel.REDIRECT]:
            return ActionLevel.REDIRECT
        elif effective_count >= thresholds[ActionLevel.THROTTLE]:
            return ActionLevel.THROTTLE
        else:
            return ActionLevel.MONITOR
    
    def evaluate_request(self, source_ip, request_data=None):
        """Enhanced evaluation with threat score consideration"""
        now = datetime.utcnow()
        threat_score = request_data.get('threat_score', 0.5) if request_data else 0.5
        
        # Store request with timestamp
        self.request_counts[source_ip].append(now)
        self.threat_scores[source_ip].append((now, threat_score))
        
        request_count = self._get_request_count(source_ip)
        new_level = self._determine_action_level(source_ip, request_count, threat_score)
        
        # Record level escalation
        if new_level.value > self.current_levels[source_ip].value:
            self.violation_history[source_ip].append((now, new_level))
            self.current_levels[source_ip] = new_level
        
        return self._create_action_response(source_ip, new_level, request_count, threat_score)
    
    def _create_action_response(self, source_ip, level, request_count, threat_score=0.5):
        """Enhanced response creation with threat score info"""
        response = super()._create_action_response(source_ip, level, request_count)
        response['threat_score'] = threat_score
        response['weighted_count'] = self._calculate_threat_weighted_count(source_ip)
        
        # Adjust actions based on threat level
        if level == ActionLevel.THROTTLE and threat_score >= self.high_threat_threshold:
            response['delay_seconds'] = min(response.get('delay_seconds', 1) * 2, 60)
        
        elif level == ActionLevel.REDIRECT and threat_score >= self.critical_threat_threshold:
            # Skip honeypot for critical threats, go straight to block
            if self.honeypot_redirects[source_ip] >= 1:
                response['allowed'] = False
                response['action'] = 'block'
                response['block_duration'] = 600  # 10 minutes for critical threats
        
        return response

def demo_threat_aware_limiter():
    """Demonstrate threat-aware limiting"""
    print("Threat-Aware Action Limiter Demo")
    print("=" * 40)
    
    limiter = ThreatAwareLimiter({
        'monitor_threshold': 3,
        'throttle_threshold': 5,
        'redirect_threshold': 8,
        'block_threshold': 12,
        'window_minutes': 2,
        'escalation_factor': 1.5
    })
    
    # Test scenarios
    scenarios = [
        ("Low threat requests", "192.168.1.10", [(0.2, 'normal'), (0.3, 'scan'), (0.1, 'normal')]),
        ("Medium threat buildup", "192.168.1.20", [(0.4, 'probe'), (0.6, 'exploit'), (0.5, 'probe')]),
        ("High threat attack", "192.168.1.30", [(0.8, 'sql_injection'), (0.9, 'sql_injection')]),
        ("Critical threat", "192.168.1.40", [(0.95, 'advanced_attack')])
    ]
    
    for scenario_name, ip, requests in scenarios:
        print(f"\n{scenario_name} ({ip}):")
        
        for i, (threat_score, attack_type) in enumerate(requests):
            request_data = {
                'threat_score': threat_score,
                'type': attack_type,
                'source_ip': ip
            }
            
            response = limiter.evaluate_request(ip, request_data)
            
            print(f"  Request {i+1}: threat={threat_score:.2f} -> {response['action_level']:8s}")
            print(f"            action={response['action']}, weighted_count={response['weighted_count']:.1f}")
            
            if not response['allowed']:
                print(f"            BLOCKED/REDIRECTED")
                break

if __name__ == '__main__':
    demo_threat_aware_limiter()