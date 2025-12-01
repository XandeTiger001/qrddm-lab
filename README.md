# Cyber Event Horizon 🛡️

A cybersecurity system inspired by theoretical physics that detects and neutralizes threats before they reach your infrastructure.

## Concept

Like a black hole's event horizon, this system creates a boundary where malicious traffic is detected and absorbed before causing damage. Uses ML-based anomaly detection to predict threats.

## Architecture

```
Incoming Traffic → Threat Detector (ML) → Decision Point
                                          ├─ Safe → Allow
                                          └─ Threat → Redirector (Neutralize)
```

## Local Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the ML model**:
   ```bash
   python src/train_model.py
   ```

3. **Run the ML-enhanced server**:
   ```bash
   python src/ml_server.py
   ```

4. **Run the honeypot** (optional, in another terminal):
   ```bash
   python src/honeypot.py
   ```

5. **Test with simulated attacks** (in another terminal):
   ```bash
   python tests/test_attacks.py
   python tests/test_honeypot.py
   ```

## How It Works

- **ML Threat Detector**: Random Forest classifier trained on attack patterns
- **Feature Extraction**: Request rate, payload size, suspicious patterns, Shannon entropy
- **Event Horizon**: Requests with threat score ≥ 0.7 are redirected
- **Honeypot**: Fake endpoints (/admin, /login, /api/users) capture attacker behavior
- **Threat Intelligence**: Logs all attacks for continuous model improvement

## AWS Deployment (Future)

- Deploy `threat_detector.py` and `redirector.py` as Lambda functions
- Configure API Gateway to route through threat detector
- Use S3 for threat logs and ML model storage
- Add CloudWatch for monitoring

## Features Detected

- SQL Injection patterns
- XSS attempts
- Path traversal
- Abnormal payload sizes
- High entropy (encrypted/obfuscated payloads)
- Request rate anomalies

## 🌌 Adaptive Schwarzschild Metric

The system uses an analogy with the Schwarzschild metric of general relativity:

```
ds² = -(1 - 2GM(r)/c²r)c²dt² + (1 - 2GM(r)/c²r)⁻¹dr² + r²dΩ²
```

### Adaptive M(r) - Attack Mass

```
M(r) = α·S + β·F + γ·V + δ·D + ε·C
```

**Components:**
- **S** (Severity): Severity of the attack (SQL injection, DDoS, etc.)
- **F** (Frequency): Frequency of similar attacks
- **V** (Velocity): Speed ​​of change of the attack
- **D** (Distance): Proximity to the critical component
- **C** (Complexity): Complexity (techniques, mutations, adversarial noise)

**Adaptive Coefficients (AI adjusts):**
- **α, β, γ, δ, ε**: Sensitivity knobs
- PARANOID Mode: α=1.5, δ=1.5 (high sensitivity)
- ECONOMY Mode: γ=0.5, ε=0.5 (low cost)
- STUDY Mode: β=1.5 (frequency focus)

**Other Metrics:**
- **r**: Distance to the critical core
- **G**: Overall defense sensitivity
- **c**: Propagation speed (normalized = 1)
- **Φ(r)**: Stability indicator = 1 - 2GM(r)/(c²r)
- **dΩ²**: Angular dispersion (modules) (affected)

**Ternary Classification:**
- Φ ≥ 0.5 → SAFE (-1)
- 0.2 < Φ < 0.5 → MONITOR (0)
- Φ ≤ 0.2 → CRITICAL (+1)

**Event Horizon:** r_s = 2DM/c²

When r ≤ r_s, the attack is within the horizon → immediate blocking.

### Using the Schwarzschild System

```bash

# 2D ternary field simulation
python src/ternary_field_simulation.py

# Adaptive M(r) with AI
python src/adaptive_mass.py

# Simulating attacks with physical analysis
python src/schwarzschild_defense.py

# Server with Schwarzschild metrics
python src/schwarzschild_server.py

# Tests
python tests/test_schwarzschild.py
python tests/test_ternary_field.py

# Visualization
python visualize_schwarzschild.py

```

### System Intuition

**Digital Field Curvature:**

- M(r) ↑ → digital space curves → more energy to defend
- r ↓ (attack close to (core) → curvature explodes → critical alert
- Strong attacks have more "mass" → curve the field more

**Defense Modes:**
```python
Paranoid: α=1.5, δ=1.5 # Maximum protection
Balanced: all = 1.0 # Equilibrium
Economy: γ=0.5, ε=0.5 # Resource economy
Study: β=1.5 # Pattern analysis
```

## 🟢 Ternary Field Simulation

The system simulates digital space as a 2D grid with ternary states:

**Fundamental States:**
- **-1**: Active threat (attack confirmed)
- **0**: Neutral/Unknown (noise, uncertainty)
- **+1**: Protected/Stable (dominant defense)

**Digital Field** g(r):**
```
g(r) = 1 - k·M(r)/r
```

**Evolution Rules:**
```
g(r) > 0.7 → State +1 (PROTECTED)
0.3 < g(r) ≤ 0.7 → State 0 (NEUTRAL)
g(r) ≤ 0.3 → State -1 (THREAT)
```

**M(r) in the Grid:**
- **S**: Attack intensity on the cell
- **F**: Neighbors in state -1
- **V**: Variation between cycles
- **D**: 1/r (proximity to the core)
- **C**: Noise/obfuscation level

**Visualization:**
- 50x50 grid with critical core in the center
- Injected attacks propagate through the field
- System evolves towards stability or collapse
- Animation shows temporal evolution
