from fastapi import FastAPI, Request
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from agents import AgentWorkflow  # your workflow module

# ---------------------------
# Load env
# ---------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------------------------
# DB setup
# ---------------------------
engine = create_engine(DATABASE_URL, echo=False, future=True)

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI()

workflow = AgentWorkflow()

# ---------------------------
# Create employees table
# ---------------------------
def create_table():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS employees (
                id SERIAL PRIMARY KEY,
                name TEXT,
                date DATE,
                login_time TEXT,
                logout_time TEXT,
                working_hours FLOAT
            )
        """))
        conn.commit()

create_table()

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
async def home():
    return {"message": "FastAPI server is running!"}


@app.post("/ingest")
async def ingest_chatlio(request: Request):
    try:
        payload = await request.json()
        print("✅ Received Zapier data:")
        print(json.dumps(payload, indent=2))

        # Map Slack/Zapier payload → employee fields
        name = payload.get("User Real Name") or payload.get("user", {}).get("name")
        text_status = payload.get("Text", "").lower()  # "logged in" / "logged out"
        timestamp = payload.get("Ts Time")

        # Convert timestamp to date & time
        date_val = None
        time_val = None
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date_val = dt.date()
                time_val = dt.strftime("%H:%M")
            except Exception:
                pass

        login_time = time_val if "in" in text_status else None
        logout_time = time_val if "out" in text_status else None

        # Run AI workflow
        results = await workflow.execute_workflow(payload)

        # Insert into DB
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO employees (name, date, login_time, logout_time, working_hours)
                    VALUES (:name, :date, :login, :logout, :hours)
                """),
                {
                    "name": name,
                    "date": date_val,
                    "login": login_time,
                    "logout": logout_time,
                    "hours": None  # can be computed later
                }
            )
            conn.commit()

        return {
            "status": "success",
            "received": True,
            "mapped_employee": {
                "name": name,
                "date": str(date_val),
                "login_time": login_time,
                "logout_time": logout_time
            },
            "workflow_result": results,
            "current": results
        }

    except SQLAlchemyError as db_err:
        print("❌ Database error:", str(db_err))
        return {"status": "error", "message": f"Database error: {str(db_err)}", "current": None}

    except Exception as e:
        print("❌ Error processing request:", str(e))
        return {"status": "error", "message": str(e), "current": None}
