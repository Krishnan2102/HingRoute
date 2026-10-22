# HingRoute: Romanized Hinglish Banking Support Triage Service

A lightweight, CPU-efficient triage service designed to sanitize, classify, and route customer support queries written in Romanized Hinglish. Built as a fast front-door filter, it enforces data privacy and deterministic policy-based escalation for high-risk banking intents before queries reach downstream agents or human queues.

---

## Architecture Overview

```
                                 [ Incoming Request ]
                                          │
                                          ▼
                            [ Pydantic Input Validation ]
                                          │
                                          ▼
                            [ Regex PII Redaction Engine ]
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
              [ SQLite Request Audit ]       [ Scikit-Learn Pipeline ]
                   (Local Logging)          (Char n-gram TF-IDF + Logistic Reg)
                                                           │
                                                           ▼
                                              [ Intent & Confidence Score ]
                                                           │
                                                           ▼
                                            [ Deterministic Risk Routing ]
                                            (Policy Override & Escalation)
                                                           │
                                                           ▼
                                                 [ JSON Triage Action ]

```

* **Classification Engine:** Scikit-Learn TF-IDF vectorizer (character-level n-grams, range 2–5) paired with a Logistic Regression classifier calibrated on Romanized Hinglish syntax. Character n-grams provide resilience against phonetic spelling variants and typos without requiring heavy transformer inference infrastructure.
* **API Framework:** FastAPI application serving endpoints with strict Pydantic schemas.
* **Data Sanitization:** Regex-driven PII redaction covering standard payment card structures and mobile numbers before audit persistence.
* **Local Persistence:** Structured SQLite query logging recording timestamps, sanitized queries, predicted intents, and routing dispositions.
* **Containerization:** Self-contained Docker image using `python:3.11-slim` running a Uvicorn ASGI server.

---

## Risk Routing Logic

Statistical model confidence alone is insufficient for high-consequence banking operations. The triage layer enforces strict rule overrides: whenever an intent falls into a designated high-risk category, the service mandates human intervention regardless of the model's confidence score.

### Policy Escalation Classes

* `lost_or_stolen_card`
* `compromised_card`
* `extra_charge_on_statement`

### Routing Actions

| Scenario | Action | Reason |
| --- | --- | --- |
| Any Risk Intent detected | `escalate` | `policy_risk` |
| Low model confidence ($< 0.60$) | `escalate` | `low_confidence` |
| Standard intent with high confidence | `auto_reply` | `confident` |

---

## Performance & Verification

### Classification Accuracy

Evaluated on a 12-class Romanized Hinglish banking dataset derived from Banking77:

| Model Architecture | Macro-F1 | Inference Execution (Raw Model) |
| --- | --- | --- |
| **TF-IDF (Char n-grams) + Logistic Regression** | **0.9171** | **~0.17 ms** |
| MuRIL (Multilingual BERT Baseline) | 0.9155 | ~14.80 ms (CPU) |

The lightweight linear model matches the transformer baseline in macro-F1 while operating at two orders of magnitude lower compute overhead.

### Load Test Benchmarks (Locust)

A 60-second headless load test was executed against the containerized service simulating 20 concurrent clients (`locust -f locustfile.py --headless -u 20 -r 5 -t 60s --host http://localhost:8000`):

| Metric | Measured Value |
| --- | --- |
| **Total Requests Completed** | 3,428 |
| **Throughput (Simulated Client Ceiling)** | ~57.1 req/s |
| **Median End-to-End Latency (p50)** | 23 ms |
| **p95 End-to-End Latency** | 110 ms |
| **p99 End-to-End Latency** | 220 ms |
| **Max Recorded Latency** | 670 ms |

> **Note on Latency Profile:** Recorded percentiles reflect end-to-end request cycles (HTTP deserialization, Pydantic validation, regex masking, inference, and synchronous SQLite write). Further decoupling the disk write path via background workers is planned for subsequent iterations.

---

## Repository Structure

```
hingroute/
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI application, routing, and PII masking
├── models/
│   ├── config.json             # Intent labels and risk-class definitions
│   ├── templates.json          # Response templates for automated replies
│   └── tfidf_pipeline.pkl      # Serialized Scikit-Learn vectorizer & classifier
├── tests/
│   └── test_api.py             # Integration test suite (Pytest + TestClient)
├── Dockerfile                  # Slim production deployment container
├── requirements-server.txt     # Minimal serving dependencies
├── locustfile.py               # Performance benchmarking script
└── README.md

```

---

## Quickstart

### 1. Build and Run Container

```bash
# Build the minimal image
docker build -t hingroute-api .

# Run the container exposing port 8000
docker run -p 8000:8000 hingroute-api

```

### 2. Verify Health

```bash
curl http://localhost:8000/health

```

```json
{
  "status": "healthy"
}

```

### 3. Sample Requests

#### Risk Intent (Automatic Escalation)

```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": "mera debit card kho gaya hai please block kardo"}'

```

```json
{
  "intent": "lost_or_stolen_card",
  "confidence": 0.9002,
  "action": "escalate",
  "reason": "policy_risk",
  "latency_ms": 26.7
}

```

#### Standard Query (Confident Auto-Reply)

```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": "mera card kab tak deliver hoga tracking status bata do"}'

```

```json
{
  "intent": "card_arrival",
  "confidence": 0.8841,
  "action": "auto_reply",
  "reason": "confident",
  "latency_ms": 24.1
}

```

---

## Test Suite Execution

Run unit and integration tests across endpoints, validation schemas, and escalation logic:

```bash
python -m pytest tests/ -v

```

---

## Known Boundaries & Roadmap

* **Synchronous I/O Decoupling:** Database writes currently share the synchronous request cycle; migrating to an asynchronous broker or background event queue will align API latency with raw model capability.
* **PII Expansion:** Masking currently targets basic credit card structures and 10-digit mobile formats; future updates will broaden coverage across international formats, UPI virtual payment addresses, and alphanumeric identifiers.
* **Phase 2 Expansion:** Upgrading the triage front-door to interface with an agentic multi-tier resolution framework, including tool-assisted balance/block execution and local policy retrieval.
