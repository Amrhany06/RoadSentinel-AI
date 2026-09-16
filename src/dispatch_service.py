"""Mock CAD Emergency Dispatch microservice for RoadSentinel AI.

Stands in for municipal emergency dispatch CAD (Computer Aided Dispatch).
Uses SQLite with SQLAlchemy for persistent ticket management.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, Form
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
engine = create_engine("sqlite:///dispatch_tickets.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    severity_tier = Column(Integer)
    explanation = Column(String)
    frame_path = Column(String)
    status = Column(String, default="LOGGED")  # LOGGED, DISPATCHED, EN_ROUTE, RESOLVED
    units_dispatched = Column(String, default="Highway Patrol")
    predicted_eta_minutes = Column(Float, default=8.5)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(engine)
app = FastAPI(title="RoadSentinel Mock Dispatch CAD API")


def create_ticket(
    severity_tier: int,
    explanation: str,
    frame_path: str = "",
    predicted_eta_minutes: float = 8.5,
) -> Dict[str, Any]:
    """Create a persistent emergency ticket in SQLite CAD database."""
    # Determine appropriate emergency units based on severity tier
    if severity_tier >= 5:
        units = "Heavy Rescue, Trauma EMS Unit, Fire Engine, Highway Patrol (Level 5)"
    elif severity_tier == 4:
        units = "Advanced EMS Ambulance, Police Interceptor, Tow Truck (Level 4)"
    elif severity_tier == 3:
        units = "Basic Life Support EMS, Traffic Incident Response (Level 3)"
    else:
        units = "Roadway Service Patrol / Tow Unit (Advisory)"

    db = SessionLocal()
    try:
        ticket = Ticket(
            severity_tier=severity_tier,
            explanation=explanation,
            frame_path=frame_path,
            status="DISPATCHED" if severity_tier >= 4 else "LOGGED",
            units_dispatched=units,
            predicted_eta_minutes=round(predicted_eta_minutes, 1),
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        return {
            "ticket_id": ticket.id,
            "severity_tier": ticket.severity_tier,
            "status": ticket.status,
            "units_dispatched": ticket.units_dispatched,
            "predicted_eta_minutes": ticket.predicted_eta_minutes,
            "created_at": ticket.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
    finally:
        db.close()


def update_ticket_status(ticket_id: int, new_status: str) -> bool:
    """Update status of a dispatch ticket."""
    db = SessionLocal()
    try:
        t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if t:
            t.status = new_status
            db.commit()
            return True
        return False
    finally:
        db.close()


def list_tickets() -> List[Dict[str, Any]]:
    """Retrieve all tickets in CAD database."""
    db = SessionLocal()
    try:
        tickets = db.query(Ticket).order_by(Ticket.id.desc()).all()
        return [
            {
                "ID": t.id,
                "Tier": f"Tier {t.severity_tier}",
                "Status": t.status,
                "ETA (min)": t.predicted_eta_minutes,
                "Units": t.units_dispatched,
                "Explanation": t.explanation,
                "Created At": t.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for t in tickets
        ]
    finally:
        db.close()


def get_tickets_df() -> pd.DataFrame:
    """Return all tickets as a pandas DataFrame for Streamlit CAD view."""
    records = list_tickets()
    if not records:
        return pd.DataFrame(columns=["ID", "Tier", "Status", "ETA (min)", "Units", "Explanation", "Created At"])
    return pd.DataFrame(records)


@app.post("/tickets")
def create_ticket_endpoint(
    severity_tier: int = Form(...),
    explanation: str = Form(...),
    frame_path: str = Form(""),
    eta_minutes: float = Form(8.5),
):
    return create_ticket(severity_tier, explanation, frame_path, eta_minutes)


@app.get("/tickets")
def list_tickets_endpoint():
    return list_tickets()
