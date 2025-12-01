# Cyber Event Horizon - System Architecture

## Overview

A cybersecurity defense system inspired by General Relativity that transforms attacks into geometric curvature in a digital field.

## System Components

### 1. Core - Ternary Digital Field

**File**: `src/core/ternary_field.py`

2D grid where each cell has ternary state:
- `-1`: Threat (attack confirmed)
- `0`: Neutral (uncertain)
- `+1`: Protected (defense dominant)

**Digital Relativity Metric**:
```
g(r) = 1 - k·M(r)/(r+1)
```

Where:
- `r`: Distance to critical core
- `M(r)`: Attack mass from real data
- `k`: Field sensitivity (default: 2.0)

**Evolution Rules**:
- `g(r) > 0.7` → State = +1 (Protected)
- `0.3 < g(r) ≤ 0.7` → State = 0 (Neutral)
- `g(r) ≤ 0.3` → State = -1 (Threat)

### 2. Detection Module

**File**: `src/detection/detector.py`

Transforms real event data into M(r) components:

**M(r) = α·S + β·F + γ·V + δ·D + ε·C**

Components:
- **S** (Intensity): Anomaly score, error rate, payload size
- **F** (Frequency): Similar events in time window
- **V** (Variation): Rate of change in attack pattern
- **D** (Density): Proximity to critical component
- **C** (Noise): Obfuscation, encoding tricks, complexity

**Input**: Event data (logs, API calls, behaviors)
**Output**: Position (x,y) and M(r) value

### 3. Response Module

**File**: `src/response/defender.py`

Defense actions based on field state:

| Field State | Region State | Action |
|-------------|--------------|--------|
| -1 | -1 | BLOCK |
| -1 | 0 | MONITOR |
| -1 | +1 | REVERT |
| 0/+1 | -1 | ISOLATE |
| 0/+1 | 0/+1 | ALLOW |

**Actions**:
- **BLOCK**: Reject request (403)
- **ISOLATE**: Quarantine source (429)
- **MONITOR**: Allow with logging (200)
- **REVERT**: Allow, likely false positive (200)
- **ALLOW**: Normal operation (200)

### 4. Visualization

**File**: `src/visualization/renderer.py`

Real-time rendering:
- Ternary state field (color-coded)
- M(r) attack mass heatmap
- Distance to core visualization
- Statistics overlay

## Main System Integration

**File**: `src/cyber_event_horizon.py`

Combines all modules into living organism:

```python
system = CyberEventHorizon(grid_size=50, k=2.0)

# Process event
result = system.process_event(event_data)

# Execute cycle
cycle_result = system.cycle()

# Visualize
system.visualize(save_path='output.png')
```

## Data Flow

```
Real Event → Detection → M(r) → Core Field → Response → Action
     ↓                                           ↓
  Logs/API                                   Block/Allow
```

## Usage

```bash
# Run complete system demo
python src/cyber_event_horizon.py

# Run tests
python tests/test_ternary_field.py
```

## Key Features

- **Mathematical Core**: Digital Relativity metric
- **Real Data Integration**: Transforms logs into geometry
- **Adaptive Response**: Field-based defense decisions
- **Visual Feedback**: Real-time system state rendering
- **Elegant & New**: Physics-inspired cybersecurity
