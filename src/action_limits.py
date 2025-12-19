from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

class ActionLevel(Enum):
    MONITOR = 1
    THROTTLE = 2
    REDIRECT = 3
    BLOCK = 4

class TieredActionLimiter:
    def __init__(self, config=None):
        self.config = config or {
            'monitor_threshold': 10,
            'throttle_threshold': 20,
            'redirect_threshold': 50,
            'block_threshold': 100,
            'window_minutes': 5,
            'escalation_factor': 2.0
        }
        
        self.request_counts = defaultdict(list)
        self.violation_history = defaultdict(list)
        self.current_levels = defaultdict(lambda: ActionLevel.MONITOR)
        self.honeypot_redirects = defaultdict(int)
        
    def _clean_old_records(self, source_ip):
        """Remove records outside the time window"""
        cutoff = datetime.utcnow() - timedelta(minutes=self.config['window_minutes'])
        self.request_counts[source_ip] = [
            ts for ts in self.request_counts[source_ip] if ts > cutoff
        ]
        self.violation_history[source_ip] = [
            (ts, level) for ts, level in self.violation_history[source_ip] if ts > cutoff
        ]
    
    def _get_request_count(self, source_ip):
        """Get current request count for IP"""
        self._clean_old_records(source_ip)
        return len(self.request_counts[source_ip])
    
    def _calculate_escalation_multiplier(self, source_ip):
        """Calculate escalation based on violation history"""
        violations = len(self.violation_history[source_ip])
        return self.config['escalation_factor'] ** min(violations, 3)
    
    def _determine_action_level(self, source_ip, request_count):
        """Determine appropriate action level"""
        multiplier = self._calculate_escalation_multiplier(source_ip)
        
        thresholds = {
            ActionLevel.MONITOR: self.config['monitor_threshold'],
            ActionLevel.THROTTLE: self.config['throttle_threshold'] / multiplier,
            ActionLevel.REDIRECT: self.config['redirect_threshold'] / multiplier,
            ActionLevel.BLOCK: self.config['block_threshold'] / multiplier
        }
        
        if request_count >= thresholds[ActionLevel.BLOCK]:
            return ActionLevel.BLOCK
        elif request_count >= thresholds[ActionLevel.REDIRECT]:
            return ActionLevel.REDIRECT
        elif request_count >= thresholds[ActionLevel.THROTTLE]:
            return ActionLevel.THROTTLE
        else:
            return ActionLevel.MONITOR
    
    def evaluate_request(self, source_ip, request_data=None):
        """Evaluate request and return action decision"""
        now = datetime.utcnow()
        self.request_counts[source_ip].append(now)
        
        request_count = self._get_request_count(source_ip)
        new_level = self._determine_action_level(source_ip, request_count)
        
        # Record level escalation
        if new_level.value > self.current_levels[source_ip].value:
            self.violation_history[source_ip].append((now, new_level))
            self.current_levels[source_ip] = new_level
        
        return self._create_action_response(source_ip, new_level, request_count)
    
    def _create_action_response(self, source_ip, level, request_count):
        """Create appropriate response based on action level"""
        response = {
            'source_ip': source_ip,
            'action_level': level.name,
            'request_count': request_count,
            'timestamp': datetime.utcnow().isoformat(),
            'allowed': True,
            'action': None
        }
        
        if level == ActionLevel.MONITOR:
            response['action'] = 'log_only'
            
        elif level == ActionLevel.THROTTLE:
            response['allowed'] = True
            response['action'] = 'delay_response'
            response['delay_seconds'] = min(2 ** (request_count // 10), 30)
            
        elif level == ActionLevel.REDIRECT:
            # Only redirect if not already redirected multiple times
            if self.honeypot_redirects[source_ip] < 3:
                response['allowed'] = False
                response['action'] = 'redirect_to_honeypot'
                response['redirect_url'] = f'/honeypot/{source_ip}'
                self.honeypot_redirects[source_ip] += 1
            else:
                # Escalate to block after multiple redirects
                response['allowed'] = False
                response['action'] = 'block'
                
        elif level == ActionLevel.BLOCK:
            response['allowed'] = False
            response['action'] = 'block'
            response['block_duration'] = min(300 * (request_count // 50), 3600)  # Max 1 hour
        
        return response
    
    def get_status_summary(self):
        """Get current system status"""
        summary = {
            'total_monitored_ips': len(self.current_levels),
            'levels': {level.name: 0 for level in ActionLevel},
            'top_violators': []
        }
        
        # Count IPs by level
        for ip, level in self.current_levels.items():
            summary['levels'][level.name] += 1
        
        # Get top violators
        violator_scores = []
        for ip in self.current_levels:
            count = self._get_request_count(ip)
            violations = len(self.violation_history[ip])
            score = count + violations * 10
            violator_scores.append((ip, score, self.current_levels[ip].name))
        
        summary['top_violators'] = sorted(violator_scores, key=lambda x: x[1], reverse=True)[:5]
        
        return summary

class ActionLimitMiddleware:
    """Middleware for integrating with web frameworks"""
    
    def __init__(self, limiter=None):
        self.limiter = limiter or TieredActionLimiter()
        self.logger = logging.getLogger(__name__)
    
    def process_request(self, request):
        """Process incoming request through action limits"""
        source_ip = self._get_client_ip(request)
        request_data = {
            'method': getattr(request, 'method', 'GET'),
            'path': getattr(request, 'path', '/'),
            'user_agent': getattr(request, 'headers', {}).get('User-Agent', '')
        }
        
        response = self.limiter.evaluate_request(source_ip, request_data)
        
        # Log the decision
        self.logger.info(f"Action limit decision: {response}")
        
        return response
    
    def _get_client_ip(self, request):
        """Extract client IP from request"""
        # Try various headers for real IP
        headers_to_check = [
            'HTTP_X_FORWARDED_FOR',
            'HTTP_X_REAL_IP',
            'HTTP_X_FORWARDED',
            'HTTP_FORWARDED_FOR',
            'HTTP_FORWARDED',
            'REMOTE_ADDR'
        ]
        
        for header in headers_to_check:
            ip = getattr(request, header, None) or getattr(request, 'META', {}).get(header)
            if ip:
                # Handle comma-separated IPs (X-Forwarded-For)
                return ip.split(',')[0].strip()
        
        return getattr(request, 'remote_addr', '127.0.0.1')

def demo_tiered_limits():
    """Demonstrate the tiered action limits system"""
    print("Tiered Action Limits with Graceful Degradation\n")
    print("=" * 60)
    
    limiter = TieredActionLimiter({
        'monitor_threshold': 5,
        'throttle_threshold': 10,
        'redirect_threshold': 15,
        'block_threshold': 25,
        'window_minutes': 1,
        'escalation_factor': 1.5
    })
    
    # Simulate attack progression
    attacker_ip = "192.168.1.100"
    
    print(f"\nSimulating attack from {attacker_ip}:\n")
    
    for i in range(30):
        response = limiter.evaluate_request(attacker_ip)
        
        if i in [4, 9, 14, 24]:  # Show key transition points
            print(f"Request {i+1:2d}: {response['action_level']:8s} -> {response['action']}")
            if not response['allowed']:
                print(f"           Request blocked/redirected")
        elif i < 5:
            print(f"Request {i+1:2d}: {response['action_level']:8s} -> {response['action']}")
    
    print(f"\nFinal Status:")
    summary = limiter.get_status_summary()
    print(f"  Current level: {limiter.current_levels[attacker_ip].name}")
    print(f"  Honeypot redirects: {limiter.honeypot_redirects[attacker_ip]}")
    print(f"  Violation history: {len(limiter.violation_history[attacker_ip])}")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    demo_tiered_limits()