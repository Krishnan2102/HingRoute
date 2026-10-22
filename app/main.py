import time
import json
import sqlite3
import re
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

def init_db():
    conn=sqlite3.connect("triage_logs.db")
    cursor=conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    masked_text TEXT,
    intent TEXT,
    confidence REAL,
    action TEXT,
    reason TEXT,
    latency_ms REAL
    )
    """)
    conn.commit()
    conn.close()

init_db()

class TriageRequest(BaseModel):
    message: str=Field(min_length=1,max_length=500)

class TriageResponse(BaseModel):
    intent:str
    confidence:float
    action:str
    reason:str
    latency_ms:float

@asynccontextmanager
async def lifespan(app:FastAPI):
    with open("models/config.json","r") as f:
        app.state.config=json.load(f)

    app.state.model=joblib.load("models/tfidf_pipeline.pkl")
    yield

app=FastAPI(title="HingRoute Triage API", lifespan=lifespan)

def mask_pii(text:str)-> str:
    return re.sub(r'\d{6,}','[MASKED]',text)

def decide_action(intent:str,confidence:float,config:dict):
    if intent in config["risk_intents"] or intent=="other":
        return "escalate","policy_risk"
    if confidence>=config["tau"]:
        return "auto_reply","confident"
    return "escalate","low_confidence"

@app.get("/health")
def health_check():
    return {"status":"healthy","model":"TF-IDF+LR"}

@app.post("/triage",response_model=TriageResponse)
def triage(req: TriageRequest):
    t0=time.perf_counter()
    masked_message=mask_pii(req.message)

    pipeline=app.state.model
    config= app.state.config

    probs=pipeline.predict_proba([req.message])[0]
    max_prob=max(probs)
    class_idx=list(probs).index(max_prob)
    intent=pipeline.classes_[class_idx]

    action,reason=decide_action(intent,max_prob,config)
    latency_ms=(time.perf_counter()-t0)*1000

    conn=sqlite3.connect("triage_logs.db")
    cursor=conn.cursor()
    cursor.execute(
        """INSERT INTO logs (masked_text, intent, confidence, action, reason, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?)""",
           (masked_message,intent,float(max_prob),action,reason,float(latency_ms))
    )
    conn.commit()
    conn.close()

    return TriageResponse(
        intent=intent,
        confidence=float(max_prob),
        action=action,
        reason=reason,
        latency_ms=latency_ms
    )

