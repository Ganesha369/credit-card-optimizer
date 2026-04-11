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
    # Keeping port 7860 here as confirmed for the hackathon
    return {"name": "credit-card-optimizer", "version": "1.0.0", "port": 7860}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/reset", response_model=Reward)
def reset(request: Optional[ResetRequest] = Body(None)) -> Reward:
    """
    Resets the environment. 
    Handles cases where the validator sends an empty body.
    """
    try:
        # If request is None or task_id is missing, default to "easy"
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