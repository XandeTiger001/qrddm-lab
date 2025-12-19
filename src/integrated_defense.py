from threat_aware_limiter import ThreatAwareLimiter
from action_limits import ActionLevel
from rate_limiter import RateLimiter
from honeypot import HoneypotHandler
from defense_layers import LayeredDefenseSystem
import time
import threading
from datetime import datetime

class IntegratedDefenseSystem:
    """Unified defense system combining all protection layers"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.action_limiter = ThreatAwareLimiter(self.config['action_limits'])
        self.rate_limiter = RateLimiter(**self.config['rate_limiting'])
        self.defense_layers = LayeredDefenseSystem(self.config['defense_layers'])
        
        # Tracking
        self.active_blocks = {}
        self.honeypot_sessions = {}
        
    def _default_config(self):
        return {
            'action_limits': {
                'monitor_threshold': 2,
                'throttle_threshold': 4,
                'redirect_threshold': 6,
                'block_threshold': 10,
                'window_minutes': 2,
                'escalation_factor': 1.5
            },
            'rate_limiting': {
                'max_requests': 20,
                'window_seconds': 60,
                'debounce_seconds': 10
            },
            'defense_layers': [
                (10.0, 2.0, 1.5),  # Outer monitoring layer
                (7.5, 2.5, 1.8),   # Throttling layer
                (5.0, 3.0, 2.0),   # Redirect layer
                (2.5, 3.5, 2.2),   # Pre-block layer
                (1.0, 4.0, 2.5)    # Core block layer
            ]
        }
    
    def process_request(self, source_ip, request_data=None):
        """Main request processing with tiered defense"""
        
        # Step 1: Check if already blocked
        if self._is_blocked(source_ip):
            return self._create_response('BLOCKED', 'IP is currently blocked')
        
        # Step 2: Rate limiting check
        allowed, reason = self.rate_limiter.is_allowed(source_ip)
        if not allowed:
            return self._escalate_to_action_limits(source_ip, request_data, reason)
        
        # Step 3: Action limits evaluation
        action_response = self.action_limiter.evaluate_request(source_ip, request_data)
        
        # Step 4: Apply quantum defense analysis
        threat_mass = self._calculate_threat_mass(source_ip, action_response)
        defense_result = self._apply_quantum_defense(threat_mass)
        
        # Step 5: Execute appropriate action
        return self._execute_action(source_ip, action_response, defense_result)
    
    def _escalate_to_action_limits(self, source_ip, request_data, reason):
        """Escalate rate limit violations to action limits"""
        # Force multiple evaluations to escalate quickly
        for _ in range(3):
            self.action_limiter.evaluate_request(source_ip, request_data)
        
        action_response = self.action_limiter.evaluate_request(source_ip, request_data)
        return self._execute_action(source_ip, action_response, None)
    
    def _calculate_threat_mass(self, source_ip, action_response):
        """Calculate threat mass based on action level and history"""
        base_mass = {
            'MONITOR': 0.5,
            'THROTTLE': 1.5,
            'REDIRECT': 3.0,
            'BLOCK': 5.0
        }
        
        mass = base_mass.get(action_response['action_level'], 1.0)
        
        # Increase mass based on request frequency
        if action_response['request_count'] > 50:
            mass *= 2.0
        elif action_response['request_count'] > 20:
            mass *= 1.5
        
        return mass
    
    def _apply_quantum_defense(self, threat_mass):
        """Apply quantum defense layers"""
        if threat_mass < 1.0:
            return {'action': 'allow', 'processing_time': 1.0}
        
        results, final_mass, state, processing_time = self.defense_layers.simulate_attack(
            threat_mass, mode='deterministic'
        )
        
        return {
            'action': 'allow' if state == 1 else 'escalate' if state == 0 else 'block',
            'final_mass': final_mass,
            'processing_time': processing_time,
            'layers_engaged': len(results)
        }
    
    def _execute_action(self, source_ip, action_response, defense_result):
        """Execute the determined action"""
        
        if not action_response['allowed']:
            if action_response['action'] == 'redirect_to_honeypot':
                return self._redirect_to_honeypot(source_ip, action_response)
            elif action_response['action'] == 'block':
                return self._apply_block(source_ip, action_response)
        
        elif action_response['action'] == 'delay_response':
            return self._apply_throttling(source_ip, action_response)
        
        # Default: allow with monitoring
        return self._create_response('ALLOWED', 'Request processed normally', {
            'action_level': action_response['action_level'],
            'quantum_defense': defense_result
        })
    
    def _redirect_to_honeypot(self, source_ip, action_response):
        """Redirect to honeypot instead of blocking"""
        self.honeypot_sessions[source_ip] = {
            'start_time': datetime.utcnow(),
            'redirect_count': self.action_limiter.honeypot_redirects[source_ip]
        }
        
        return self._create_response('REDIRECT', 'Redirecting to honeypot', {
            'redirect_url': action_response.get('redirect_url', '/honeypot'),
            'session_id': f"hp_{source_ip}_{int(time.time())}"
        })
    
    def _apply_throttling(self, source_ip, action_response):
        """Apply throttling delay"""
        delay = action_response.get('delay_seconds', 1)
        
        # Non-blocking delay simulation
        threading.Timer(delay, lambda: None).start()
        
        return self._create_response('THROTTLED', f'Response delayed by {delay}s', {
            'delay_applied': delay,
            'action_level': action_response['action_level']
        })
    
    def _apply_block(self, source_ip, action_response):
        """Apply blocking action"""
        duration = action_response.get('block_duration', 300)
        self.active_blocks[source_ip] = {
            'start_time': datetime.utcnow(),
            'duration': duration,
            'reason': 'Exceeded action limits'
        }
        
        return self._create_response('BLOCKED', f'IP blocked for {duration}s', {
            'block_duration': duration,
            'unblock_time': (datetime.utcnow().timestamp() + duration)
        })
    
    def _is_blocked(self, source_ip):
        """Check if IP is currently blocked"""
        if source_ip not in self.active_blocks:
            return False
        
        block_info = self.active_blocks[source_ip]
        elapsed = (datetime.utcnow() - block_info['start_time']).total_seconds()
        
        if elapsed >= block_info['duration']:
            del self.active_blocks[source_ip]
            return False
        
        return True
    
    def _create_response(self, status, message, details=None):
        """Create standardized response"""
        return {
            'status': status,
            'message': message,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        }
    
    def get_system_status(self):
        """Get comprehensive system status"""
        action_status = self.action_limiter.get_status_summary()
        
        return {
            'action_limits': action_status,
            'active_blocks': len(self.active_blocks),
            'honeypot_sessions': len(self.honeypot_sessions),
            'system_health': 'operational'
        }

def demo_integrated_defense():
    """Demonstrate integrated defense system"""
    print("Integrated Defense System Demo\n")
    print("=" * 50)
    
    defense = IntegratedDefenseSystem()
    
    # Simulate various attack scenarios
    scenarios = [
        ("Normal user", "192.168.1.10", 3),
        ("Suspicious user", "192.168.1.20", 15),
        ("Aggressive attacker", "192.168.1.30", 35),
        ("Persistent attacker", "192.168.1.40", 60)
    ]
    
    for name, ip, request_count in scenarios:
        print(f"\n{name} ({ip}) - {request_count} requests:")
        
        for i in range(request_count):
            response = defense.process_request(ip, {
                'method': 'GET',
                'path': f'/api/data/{i}',
                'user_agent': 'TestClient/1.0'
            })
            
            if i in [2, 9, 19, 34] or response['status'] != 'ALLOWED':
                print(f"  Request {i+1:2d}: {response['status']:10s} - {response['message']}")
                
                if response['status'] == 'BLOCKED':
                    break
    
    print(f"\nFinal System Status:")
    status = defense.get_system_status()
    print(f"  Active blocks: {status['active_blocks']}")
    print(f"  Honeypot sessions: {status['honeypot_sessions']}")
    print(f"  Monitored IPs: {status['action_limits']['total_monitored_ips']}")
    
    print("\n" + "=" * 50)

if __name__ == '__main__':
    demo_integrated_defense()