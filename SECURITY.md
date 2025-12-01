# Risks and Mitigations - Schwarzschild System

## Identified Risks

### 1. Scale and Units
**Problem**: Arbitrary values of M, D, c can cause numerical explosion.

**Mitigation**:
- Logarithmic normalization: `np.log1p()` for payload_size and request_rate
- M Clamp: `[0.01, 100.0]`
- r clamp: `[0.1, 100.0]`
- Φ Clamp: `[-10, 10]`
- C clamp: `[0, 1e6]`

### 2. Adversarial Inputs
**Issue**: Attacker manipulates fields to force negative Φ and cause DoS.

**Mitigation**:
- Rate limiting: 20 req/min per IP
- Debouncing: 5s between blocks
- Cross validation: Φ + ML score (majority vote)
- Minimum confidence: 0.7 for automatic blocking

### 3. Numerical Instability
**Issue**: Extreme values break visualizations and rules.

**Mitigation**:
- Protection against division by zero: `max(denominator, 1e-6)`
- Limits on all metrics
- Explicit conversion to float: `float(M)`

### 4. Human Interpretation
**Problem**: Operators don't understand why something was blocked.

**Mitigation**:
- `explanation` field with readable reason
- Logs structured with M, r, Φ, action
- Dashboard with visual metrics

### 5. Automation Without Review
**Problem**: Immediate blocking causes false positives.

**Mitigation**:
- Human-in-the-loop: confidence < 0.7 → `ALERT_ANALYST`
- First redirect to honeypot, then blocking
- Adaptive Baseline dynamically adjusts thresholds

## Practical Recommendations

### Signal Composition
```python
phi_vote = classify_by_phi(phi_value)
ml_vote = classify_by_ml(ml_score)
agreement = (phi_vote == ml_vote)
confidence = 0.9 if agreement else 0.5
```

### Adaptive Thresholds
```python
baseline = np.median(event_history)
std = np.std(event_history)
T_shadow = baseline + 0.5 * std
T_crit = baseline - 0.5 * std
```

### Rate Limiting
- 20 requests/min per IP
- 5s debounce between blocks
- 60s sliding window

### Alerts and Explainability
```json
{ 
"explanation": { 
"summary": "Attack with energy M=5.23 at range r=0.5", 
"metric": "Φ=-0.123 indicator indicates CRITICAL level", 
"reason": "Attack too close to the core" 
}
}
```

## Decision Flow

```
Request → Rate Limit → Debounce → Extract Features 
↓
Calculate M, r, Φ 
↓
ML Score + Φ Vote → Confidence 
↓
Confidence > 0.7? 
├─ Yes → Auto Action (BLOCK/REDIRECT) 
└─ No → ALERT_ANALYST (Human-in-the-loop)
```

## Monitoring

- False positive/negative rate
- Distribution of Φ over time
- Automatic adjustment of thresholds
- Logs of all decisions with explanation
