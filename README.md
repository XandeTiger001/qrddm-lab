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

## 🌌 Métrica de Schwarzschild Adaptativa

O sistema usa uma analogia com a métrica de Schwarzschild da relatividade geral:

```
ds² = -(1 - 2GM(r)/c²r)c²dt² + (1 - 2GM(r)/c²r)⁻¹dr² + r²dΩ²
```

### M(r) Adaptativo - Massa do Ataque

```
M(r) = α·S + β·F + γ·V + δ·D + ε·C
```

**Componentes:**
- **S** (Severity): Severidade do ataque (SQL injection, DDoS, etc.)
- **F** (Frequency): Frequência de ataques similares
- **V** (Velocity): Velocidade de mudança do ataque
- **D** (Distance): Proximidade ao componente crítico
- **C** (Complexity): Complexidade (técnicas, mutações, ruído adversarial)

**Coeficientes Adaptativos (IA ajusta):**
- **α, β, γ, δ, ε**: Botões de sensibilidade
- Modo PARANOID: α=1.5, δ=1.5 (alta sensibilidade)
- Modo ECONOMY: γ=0.5, ε=0.5 (baixo custo)
- Modo STUDY: β=1.5 (foco em frequência)

**Outras Métricas:**
- **r**: Distância ao núcleo crítico
- **G**: Sensibilidade geral de defesa
- **c**: Velocidade de propagação (normalizado = 1)
- **Φ(r)**: Indicador de estabilidade = 1 - 2GM(r)/(c²r)
- **dΩ²**: Dispersão angular (módulos afetados)

**Classificação Ternária:**
- Φ ≥ 0.5 → SAFE (-1)
- 0.2 < Φ < 0.5 → MONITOR (0)
- Φ ≤ 0.2 → CRITICAL (+1)

**Horizonte de Eventos:** r_s = 2DM/c²

Quando r ≤ r_s, o ataque está dentro do horizonte → bloqueio imediato.

### Usar Sistema Schwarzschild

```bash
# Simulação de campo ternário 2D
python src/ternary_field_simulation.py

# M(r) adaptativo com IA
python src/adaptive_mass.py

# Simular ataques com análise física
python src/schwarzschild_defense.py

# Servidor com métrica Schwarzschild
python src/schwarzschild_server.py

# Testes
python tests/test_schwarzschild.py
python tests/test_ternary_field.py

# Visualização
python visualize_schwarzschild.py
```

### Intuição do Sistema

**Curvatura do Campo Digital:**
- M(r) ↑ → espaço digital se curva → mais energia para defender
- r ↓ (ataque próximo ao núcleo) → curvatura explode → alerta crítico
- Ataques fortes têm mais "massa" → curvam mais o campo

**Modos de Defesa:**
```python
Paranoid: α=1.5, δ=1.5  # Máxima proteção
Balanced: todos = 1.0    # Equilíbrio
Economy: γ=0.5, ε=0.5   # Economia de recursos
Study: β=1.5            # Análise de padrões
```

## 🟢 Simulação de Campo Ternário

O sistema simula o espaço digital como um grid 2D com estados ternários:

**Estados Fundamentais:**
- **-1**: Ameaça ativa (ataque confirmado)
- **0**: Neutro/Desconhecido (ruído, incerteza)
- **+1**: Protegido/Estável (defesa dominante)

**Campo Digital g(r):**
```
g(r) = 1 - k·M(r)/r
```

**Regras de Evolução:**
```
g(r) > 0.7  → Estado +1 (PROTEGIDO)
0.3 < g(r) ≤ 0.7 → Estado 0 (NEUTRO)
g(r) ≤ 0.3  → Estado -1 (AMEAÇA)
```

**M(r) no Grid:**
- **S**: Intensidade do ataque na célula
- **F**: Vizinhos em estado -1
- **V**: Variação entre ciclos
- **D**: 1/r (proximidade ao núcleo)
- **C**: Nível de ruído/ofuscação

**Visualização:**
- Grid 50x50 com núcleo crítico no centro
- Ataques injetados se propagam pelo campo
- Sistema evolui até estabilidade ou colapso
- Animação mostra evolução temporal
