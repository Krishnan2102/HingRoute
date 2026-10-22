import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_risk_intent_escalation(client):
    payload = {"message": "mera debit card kho gaya hai please block kardo"}
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "lost_or_stolen_card"
    assert data["action"] == "escalate"
    assert data["reason"] == "policy_risk"

def test_normal_confident_intent(client):
    payload = {"message": "mera card abhi tak deliver nahi hua kab tak aayega"}
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["confidence"] > 0.0
    assert "action" in data
    assert "latency_ms" in data

def test_validation_empty_string(client):
    response = client.post("/triage", json={"message": ""})
    assert response.status_code == 422

def test_validation_oversized_string(client):
    long_msg = "paisa " * 110
    response = client.post("/triage", json={"message": long_msg})
    assert response.status_code == 422