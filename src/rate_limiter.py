from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_requests=10, window_seconds=60, debounce_seconds=5):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.debounce = timedelta(seconds=debounce_seconds)
        self.requests = defaultdict(list)
        self.last_block = defaultdict(lambda: None)
    
    def is_allowed(self, source_ip):
        now = datetime.utcnow()
        self.requests[source_ip] = [ts for ts in self.requests[source_ip] if now - ts < self.window]
        if len(self.requests[source_ip]) >= self.max_requests:
            return False, 'RATE_LIMIT_EXCEEDED'
        self.requests[source_ip].append(now)
        return True, 'OK'
    
    def should_debounce(self, source_ip):
        last = self.last_block[source_ip]
        if last is None:
            return False
        return (datetime.utcnow() - last) < self.debounce
    
    def record_block(self, source_ip):
        self.last_block[source_ip] = datetime.utcnow()
