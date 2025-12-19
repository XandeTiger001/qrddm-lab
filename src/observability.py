import json
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

class ProductionLogger:
    def __init__(self, log_file='logs/cyber_defense.log'):
        self.logger = logging.getLogger('cyber_defense')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_decision(self, timestamp, features, M_total, phi, r_s, decision, action, confidence, source_ip):
        """Log complete decision with all physics parameters"""
        log_entry = {
            'timestamp': timestamp,
            'source_ip': source_ip,
            'features': features,
            'M_total': M_total,
            'phi': phi,
            'r_s': r_s,
            'decision': decision,
            'action': action,
            'confidence': confidence
        }
        self.logger.info(json.dumps(log_entry))

class MetricsCollector:
    def __init__(self, window_minutes=5):
        self.window_minutes = window_minutes
        self.decisions = deque()
        self.metrics_history = defaultdict(list)
        
    def record_decision(self, timestamp, y_true, y_pred, confidence, latency, source_ip):
        """Record decision for metrics calculation"""
        self.decisions.append({
            'timestamp': timestamp,
            'y_true': y_true,
            'y_pred': y_pred,
            'confidence': confidence,
            'latency': latency,
            'source_ip': source_ip
        })
        
        # Clean old records
        cutoff = timestamp - timedelta(minutes=self.window_minutes)
        while self.decisions and self.decisions[0]['timestamp'] < cutoff:
            self.decisions.popleft()
    
    def calculate_window_metrics(self):
        """Calculate metrics for current window"""
        if not self.decisions:
            return {}
        
        decisions = list(self.decisions)
        y_true = [d['y_true'] for d in decisions]
        y_pred = [d['y_pred'] for d in decisions]
        
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        avg_latency = np.mean([d['latency'] for d in decisions])
        avg_confidence = np.mean([d['confidence'] for d in decisions])
        
        return {
            'timestamp': datetime.utcnow(),
            'window_minutes': self.window_minutes,
            'total_decisions': len(decisions),
            'precision': precision,
            'recall': recall,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'avg_latency_ms': avg_latency * 1000,
            'avg_confidence': avg_confidence,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }

class CloudWatchExporter:
    def __init__(self, namespace='CyberDefense'):
        self.namespace = namespace
        self.metrics_buffer = []
    
    def put_metric(self, metric_name, value, unit='Count', dimensions=None):
        """Add metric to buffer for batch export"""
        self.metrics_buffer.append({
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Dimensions': dimensions or [],
            'Timestamp': datetime.utcnow()
        })
    
    def export_metrics(self, metrics_data):
        """Export metrics to CloudWatch format"""
        self.put_metric('Precision', metrics_data['precision'], 'Percent')
        self.put_metric('Recall', metrics_data['recall'], 'Percent')
        self.put_metric('FalsePositiveRate', metrics_data['false_positive_rate'], 'Percent')
        self.put_metric('FalseNegativeRate', metrics_data['false_negative_rate'], 'Percent')
        self.put_metric('AvgLatency', metrics_data['avg_latency_ms'], 'Milliseconds')
        self.put_metric('AvgConfidence', metrics_data['avg_confidence'], 'Percent')
        self.put_metric('TotalDecisions', metrics_data['total_decisions'], 'Count')
        
        # Simulate CloudWatch export
        print(f"CloudWatch Export: {len(self.metrics_buffer)} metrics")
        self.metrics_buffer.clear()

class GrafanaExporter:
    def __init__(self, endpoint='http://localhost:3000'):
        self.endpoint = endpoint
    
    def export_metrics(self, metrics_data):
        """Export metrics in Grafana format"""
        grafana_metrics = {
            'cyber_defense_precision': metrics_data['precision'],
            'cyber_defense_recall': metrics_data['recall'],
            'cyber_defense_fpr': metrics_data['false_positive_rate'],
            'cyber_defense_fnr': metrics_data['false_negative_rate'],
            'cyber_defense_latency': metrics_data['avg_latency_ms'],
            'cyber_defense_confidence': metrics_data['avg_confidence'],
            'cyber_defense_decisions': metrics_data['total_decisions']
        }
        
        # Simulate Grafana export
        print(f"Grafana Export: {len(grafana_metrics)} metrics to {self.endpoint}")
        return grafana_metrics

class AlertManager:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {
            'max_fpr': 0.05,
            'min_recall': 0.80,
            'max_latency_ms': 10.0,
            'min_confidence': 0.70
        }
        self.alerts_sent = defaultdict(lambda: datetime.min)
        self.alert_cooldown = timedelta(minutes=5)
    
    def check_alerts(self, metrics_data):
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        now = datetime.utcnow()
        
        # False Positive Rate alert
        if metrics_data['false_positive_rate'] > self.thresholds['max_fpr']:
            if now - self.alerts_sent['high_fpr'] > self.alert_cooldown:
                alerts.append({
                    'type': 'HIGH_FALSE_POSITIVE_RATE',
                    'severity': 'WARNING',
                    'message': f"FPR {metrics_data['false_positive_rate']:.3f} exceeds threshold {self.thresholds['max_fpr']}",
                    'value': metrics_data['false_positive_rate']
                })
                self.alerts_sent['high_fpr'] = now
        
        # Recall alert
        if metrics_data['recall'] < self.thresholds['min_recall']:
            if now - self.alerts_sent['low_recall'] > self.alert_cooldown:
                alerts.append({
                    'type': 'LOW_RECALL',
                    'severity': 'CRITICAL',
                    'message': f"Recall {metrics_data['recall']:.3f} below threshold {self.thresholds['min_recall']}",
                    'value': metrics_data['recall']
                })
                self.alerts_sent['low_recall'] = now
        
        # Latency alert
        if metrics_data['avg_latency_ms'] > self.thresholds['max_latency_ms']:
            if now - self.alerts_sent['high_latency'] > self.alert_cooldown:
                alerts.append({
                    'type': 'HIGH_LATENCY',
                    'severity': 'WARNING',
                    'message': f"Latency {metrics_data['avg_latency_ms']:.2f}ms exceeds threshold {self.thresholds['max_latency_ms']}ms",
                    'value': metrics_data['avg_latency_ms']
                })
                self.alerts_sent['high_latency'] = now
        
        return alerts

class ProductionObservability:
    def __init__(self, window_minutes=5):
        self.logger = ProductionLogger()
        self.metrics = MetricsCollector(window_minutes)
        self.cloudwatch = CloudWatchExporter()
        self.grafana = GrafanaExporter()
        self.alerts = AlertManager()
        
    def log_defense_decision(self, source_ip, features, physics_params, decision_data, ground_truth=None):
        """Complete logging of defense decision"""
        timestamp = datetime.utcnow()
        
        # Extract physics parameters
        M_total = physics_params.get('M_total', 0)
        phi = physics_params.get('phi', 1.0)
        r_s = physics_params.get('r_s', 0)
        
        # Log detailed decision
        self.logger.log_decision(
            timestamp=timestamp.isoformat(),
            features=features,
            M_total=M_total,
            phi=phi,
            r_s=r_s,
            decision=decision_data['status'],
            action=decision_data.get('action', 'none'),
            confidence=decision_data.get('confidence', 0.5),
            source_ip=source_ip
        )
        
        # Record metrics if ground truth available
        if ground_truth is not None:
            y_pred = 1 if decision_data['status'] in ['BLOCKED', 'REDIRECT', 'THROTTLED'] else 0
            self.metrics.record_decision(
                timestamp=timestamp,
                y_true=ground_truth,
                y_pred=y_pred,
                confidence=decision_data.get('confidence', 0.5),
                latency=decision_data.get('latency', 0.001),
                source_ip=source_ip
            )
    
    def export_metrics(self):
        """Export current window metrics to dashboards"""
        metrics_data = self.metrics.calculate_window_metrics()
        
        if not metrics_data:
            return None
        
        # Export to monitoring systems
        self.cloudwatch.export_metrics(metrics_data)
        grafana_data = self.grafana.export_metrics(metrics_data)
        
        # Check for alerts
        alerts = self.alerts.check_alerts(metrics_data)
        
        return {
            'metrics': metrics_data,
            'grafana_data': grafana_data,
            'alerts': alerts
        }

def demo_observability():
    """Demonstrate production observability"""
    print("Production Observability Demo")
    print("=" * 40)
    
    obs = ProductionObservability(window_minutes=1)
    
    # Simulate defense decisions
    scenarios = [
        # Normal traffic
        ('192.168.1.10', {'S': 0.1, 'F': 0.2, 'V': 0.1, 'D': 0.8, 'C': 0.1}, 
         {'M_total': 0.5, 'phi': 0.95, 'r_s': 0.1}, 
         {'status': 'ALLOWED', 'confidence': 0.9, 'latency': 0.001}, 0),
        
        # Detected attack
        ('192.168.1.20', {'S': 0.9, 'F': 0.8, 'V': 0.7, 'D': 0.3, 'C': 0.8}, 
         {'M_total': 3.2, 'phi': 0.2, 'r_s': 2.1}, 
         {'status': 'BLOCKED', 'confidence': 0.95, 'latency': 0.002}, 1),
        
        # False positive
        ('192.168.1.30', {'S': 0.3, 'F': 0.4, 'V': 0.2, 'D': 0.7, 'C': 0.2}, 
         {'M_total': 1.1, 'phi': 0.8, 'r_s': 0.5}, 
         {'status': 'THROTTLED', 'confidence': 0.6, 'latency': 0.003}, 0),
        
        # Missed attack
        ('192.168.1.40', {'S': 0.7, 'F': 0.3, 'V': 0.4, 'D': 0.6, 'C': 0.5}, 
         {'M_total': 1.8, 'phi': 0.7, 'r_s': 0.8}, 
         {'status': 'ALLOWED', 'confidence': 0.4, 'latency': 0.001}, 1)
    ]
    
    for source_ip, features, physics, decision, ground_truth in scenarios:
        obs.log_defense_decision(source_ip, features, physics, decision, ground_truth)
        time.sleep(0.1)
    
    # Export metrics and check alerts
    print("\nExporting metrics...")
    result = obs.export_metrics()
    
    if result:
        metrics = result['metrics']
        print(f"\nWindow Metrics ({metrics['window_minutes']} min):")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.3f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.3f}")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
        
        if result['alerts']:
            print(f"\nALERTS ({len(result['alerts'])}):")
            for alert in result['alerts']:
                print(f"  {alert['severity']}: {alert['message']}")

if __name__ == '__main__':
    demo_observability()