# Complete System - Cyber Event Horizon

## Run the Complete System

```bash
python src/cyber_event_horizon.py
```

## System Architecture

**4 Modules Working as Living Organism:**

1. **Core** - Ternary field with digital relativity
2. **Detection** - Transforms real data into M(r)
3. **Response** - Defense actions based on field state
4. **Visualization** - Real-time rendering

## Quick Start

```python
from cyber_event_horizon import CyberEventHorizon

# Initialize
system = CyberEventHorizon(grid_size=50, k=2.0)

# Process event
event = {
    'source_ip': '10.0.0.50',
    'endpoint': '/admin',
    'anomaly_score': 0.9,
    'error_rate': 0.8,
    'payload_size': 5000,
    'type': 'sql_injection',
    'target_layer': 1
}

result = system.process_event(event)
print(result['response']['action'])  # BLOCK, ISOLATE, etc.

# Visualize
system.visualize(save_path='field.png')
```

## Event Data Format

```python
{
    'source_ip': str,           # Source IP address
    'endpoint': str,            # Target endpoint
    'anomaly_score': float,     # 0.0-1.0
    'error_rate': float,        # 0.0-1.0
    'payload_size': int,        # Bytes
    'type': str,                # Attack type
    'target_layer': int,        # 0=core, 5=edge
    'obfuscation_level': int,   # 0-10
    'encoding_tricks': int,     # 0-10
    'complexity': int           # 0-10
}
```

## Defense Actions

- **BLOCK**: Request rejected (403)
- **ISOLATE**: Source quarantined (429)
- **MONITOR**: Allowed with logging (200)
- **REVERT**: False positive, allowed (200)
- **ALLOW**: Normal operation (200)

## Field States

- `-1`: Threat (red)
- `0`: Neutral (yellow)
- `+1`: Protected (green)

See ARCHITECTURE.md for complete technical details.
