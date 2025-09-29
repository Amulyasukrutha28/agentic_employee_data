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
# ---------------------------
# DB setup
# ---------------------------
engine = create_engine(DATABASE_URL, echo=False, future=True)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

workflow = AgentWorkflow()

# ---------------------------
# Create employees table
# ---------------------------
def create_table():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS employees (
                id SERIAL PRIMARY KEY,
                name TEXT,
                date DATE,
                login_time TIME,
                logout_time TIME,
                working_hours FLOAT
            )
        """))

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
        print("✅ Received payload:")
        print(json.dumps(payload, indent=2))

        # Extract employee info
        user_data = payload.get("user", {})
        name = (
            user_data.get("real_name")
            or user_data.get("name")
            or user_data.get("profile", {}).get("first_name")
            or "Unknown"
        )

        text_status = (payload.get("text") or payload.get("raw_text") or "").lower()
        timestamp = payload.get("ts_time")

        date_val, time_val = None, None
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date_val = dt.date()
                time_val = dt.time()
            except Exception as e:
                print("⚠️ Timestamp parse error:", str(e))

        # Determine login/logout
        login_time = time_val if "in" in text_status else None
        logout_time = time_val if "out" in text_status else None

        # Compute working hours if both login and logout exist
        working_hours = None
        if login_time and logout_time:
            delta = datetime.combine(date_val, logout_time) - datetime.combine(date_val, login_time)
            working_hours = round(delta.total_seconds() / 3600, 2)

        # Run AI workflow
        workflow_result = await workflow.execute_workflow(payload)

        # Save to DB
        with engine.begin() as conn:
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
                    "hours": working_hours
                }
            )

        return {
            "status": "success",
            "received": True,
            "mapped_employee": {
                "name": name,
                "date": str(date_val) if date_val else None,
                "login_time": str(login_time) if login_time else None,
                "logout_time": str(logout_time) if logout_time else None,
                "working_hours": working_hours
            },
            "workflow_result": workflow_result
        }

    except SQLAlchemyError as db_err:
        print("❌ Database error:", str(db_err))
        return {"status": "error", "message": f"Database error: {str(db_err)}"}

    except Exception as e:
        print("❌ Error processing request:", str(e))
        return {"status": "error", "message": str(e)}
