from fastapi import FastAPI, Request
import json
import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------------------------
# Initialize database engine
# ---------------------------
engine = create_engine(DATABASE_URL, echo=False, future=True)

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI()

# ---------------------------
# Import your agent workflow
# ---------------------------
from agentic_workflow import AgentWorkflow  # Ensure this is your workflow module
workflow = AgentWorkflow()


# ---------------------------
# Create employees table if not exists
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
        # Get payload
        payload = await request.json()
        print("✅ Received Chatlio data:")
        print(json.dumps(payload, indent=2))

        # Run your agentic workflow
        results = await workflow.execute_workflow(payload)

        # Insert each employee record into the database
        with engine.begin() as conn:
            for record in payload:
                name = record.get('Employee Name') or record.get('employee') or record.get('name')
                date = record.get('Date') or record.get('date')
                login_time = record.get('Log in') or record.get('login')
                logout_time = record.get('Log out') or record.get('logout')
                working_hours = record.get('Working hours') or record.get('workingHours')

                conn.execute(
                    text("""
                        INSERT INTO employees (name, date, login_time, logout_time, working_hours)
                        VALUES (:name, :date, :login, :logout, :hours)
                    """),
                    {"name": name, "date": date, "login": login_time, "logout": logout_time, "hours": working_hours}
                )
            conn.commit()

        return {"status": "success", "received": True, "workflow_result": results}

    except SQLAlchemyError as db_err:
        print("❌ Database error:", str(db_err))
        return {"status": "error", "message": f"Database error: {str(db_err)}"}

    except Exception as e:
        print("❌ Error processing request:", str(e))
        return {"status": "error", "message": str(e)}

