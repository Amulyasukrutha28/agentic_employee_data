from fastapi import FastAPI, Request
import json
import asyncio

from agents import AgentWorkflow  # Make sure to import your workflow class

app = FastAPI()

# Initialize the agentic workflow
workflow = AgentWorkflow()

@app.get("/")
async def home():
    return {"message": "FastAPI server is running!"}

@app.post("/ingest")
async def ingest_chatlio(request: Request):
    try:
        payload = await request.json()
        results = await workflow.execute_workflow(payload)
        
        print("✅ Received Chatlio data:")
        print(json.dumps(payload, indent=2))

        # Save results for dashboard or further analysis
        with open('analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return {"status": "success", "received": True}

    except Exception as e:
        print("❌ Error processing request:", str(e))
        return {"status": "error", "message": str(e)}

