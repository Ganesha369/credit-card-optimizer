from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, HTTPException, Body

from credit_card_env.models import Action, ResetRequest, Reward
from credit_card_env.server.environment import CreditCardRewardEnvironment

app = FastAPI(
    title="credit-card-optimizer",
    version="1.0.0",
    description="OpenEnv-compatible credit card optimization environment.",
)

environment = CreditCardRewardEnvironment()

@app.get("/")
def root() -> dict[str, str | int]:
    return {"name": "credit-card-optimizer", "version": "1.0.0", "port": 7860}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/reset", response_model=Reward)
def reset(request: Optional[ResetRequest] = Body(None)) -> Reward:
    try:
        task_id = request.task_id if request and request.task_id else "easy"
        return environment.reset(task_id)
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
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()