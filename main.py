from fastapi import FastAPI, Request
import json
import asyncio

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "FastAPI server is running!"}

@app.post("/ingest")
async def ingest_chatlio(request: Request):
    try:
        payload = await request.json()
        print("✅ Received Chatlio data:")
        print(json.dumps(payload, indent=2))

        # Example: Here you can call your agent workflow asynchronously
        # asyncio.create_task(run_agentic_workflow(payload))

        return {"status": "success", "received": True}

    except Exception as e:
        print("❌ Error processing request:", str(e))
        return {"status": "error", "message": str(e)}
