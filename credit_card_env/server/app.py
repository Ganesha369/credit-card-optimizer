from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, HTTPException, Body, Query

from credit_card_env.models import Action, ResetRequest, Reward
from credit_card_env.server.environment import CreditCardRewardEnvironment

app = FastAPI(
    title="credit-card-optimizer",
    version="1.0.0",
    description="OpenEnv-compatible credit card optimization environment.",
)

# Use the instance name 'environment' as per your original code
environment = CreditCardRewardEnvironment()

@app.get("/")
def root() -> dict[str, str | int]:
    return {"name": "credit-card-optimizer", "version": "1.0.0", "port": 7860}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/reset", response_model=Reward)
def reset(
    task_id: Optional[str] = Query(None), 
    request: Optional[ResetRequest] = Body(None)
) -> Reward:
    """
    Flexible reset route: Handles task_id from URL query params or JSON body.
    This fixes the '400 Bad Request' seen during validator startup.
    """
    try:
        # 1. Check if task_id was sent in the URL (?task_id=easy)
        # 2. If not, check if it was sent in the JSON body
        # 3. Default to "easy" if both are missing
        final_task_id = task_id or (request.task_id if request else "easy")
        
        return environment.reset(final_task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/step", response_model=Reward)
def step(request: Action) -> Reward:
    try:
        return environment.step(request.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# --- THE VALIDATOR FIX ---
def main():
    """
    Main entry point for the validator to start the server.
    Ensures the 'multi-mode deployment' check passes.
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()