from fastapi import FastAPI, Request
import asyncio
import json

app = FastAPI()

@app.post("/ingest")
async def ingest_chatlio(request: Request):
    payload = await request.json()
    print("Received Chatlio data:", json.dumps(payload, indent=2))
    
    # Here you can call your agentic workflow
    # asyncio.create_task(run_agentic_workflow(payload))

    return {"status": "success"}
