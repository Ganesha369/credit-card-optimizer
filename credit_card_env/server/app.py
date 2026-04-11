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

# Use the instance name 'environment' used in your folder structure
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
    Handles environment reset. This version is 'multilingual'—it can read
    the task_id from a URL query string or a JSON body, defaulting to 'easy'.
    """
    try:
        # Priority 1: Check URL Query (?task_id=...)
        # Priority 2: Check JSON Body ({"task_id": "..."})
        # Priority 3: Default to "easy"
        final_id = "easy"
        if task_id:
            final_id = task_id
        elif request and request.task_id:
            final_id = request.task_id
            
        return environment.reset(final_id)
    except Exception as e:
        # Catch-all to prevent 400 errors during the startup handshake
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=Reward)
def step(request: Action) -> Reward:
    """
    Executes one step in the environment based on the agent's action.
    """
    try:
        return environment.step(request.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- VALIDATOR ENTRY POINT ---
def main():
    """
    Main entry point for the validator. 
    Crucial for passing the 'Multi-mode deployment' check.
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()