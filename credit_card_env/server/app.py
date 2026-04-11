from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, HTTPException, Body, Query, Request

from credit_card_env.models import Action, ResetRequest, Reward
from credit_card_env.server.environment import CreditCardRewardEnvironment

app = FastAPI(title="credit-card-optimizer")
environment = CreditCardRewardEnvironment()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/reset", response_model=Reward)
async def reset(
    task_id: Optional[str] = Query(None), 
    request: Optional[ResetRequest] = Body(None),
    raw_request: Request = None
) -> Reward:
    try:
        # 1. Try to get task_id from Query Param (?task_id=easy)
        # 2. Try to get task_id from JSON Body ({"task_id": "easy"})
        # 3. If all fails, default to "easy"
        
        final_id = "easy"
        if task_id:
            final_id = task_id
        elif request and request.task_id:
            final_id = request.task_id
            
        print(f"Resetting with task_id: {final_id}") # This will show in your HF logs
        return environment.reset(final_id)
    except Exception as e:
        # We catch everything and return "easy" instead of a 400 error
        print(f"Reset Error: {e}. Defaulting to easy.")
        return environment.reset("easy")

@app.post("/step", response_model=Reward)
def step(request: Action) -> Reward:
    try:
        return environment.step(request.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()