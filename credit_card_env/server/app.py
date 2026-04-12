from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query

from credit_card_env.models import Action, ResetRequest, Reward
from credit_card_env.server.environment import (
    DEFAULT_TASK_ID,
    TASK_CONFIG,
    CreditCardRewardEnvironment,
)

app = FastAPI(title="credit-card-optimizer")
env = CreditCardRewardEnvironment()


def _normalize_task_id(task_id: Any) -> str:
    if task_id is None:
        return DEFAULT_TASK_ID

    normalized = str(task_id).strip().lower()
    if not normalized or normalized not in TASK_CONFIG:
        return DEFAULT_TASK_ID
    return normalized


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "running"}


@app.post("/reset", response_model=Reward)
def reset(
    task_id: str | None = Query(default=None),
    request: dict[str, Any] | ResetRequest | None = Body(default=None),
) -> Reward:
    try:
        body_task_id: Any = None
        if isinstance(request, ResetRequest):
            body_task_id = request.task_id
        elif isinstance(request, dict):
            body_task_id = request.get("task_id")

        requested_task_id = task_id if task_id is not None else body_task_id
        reward_obj = env.reset(_normalize_task_id(requested_task_id))
    except Exception:
        reward_obj = env.reset(DEFAULT_TASK_ID)

    print(f"[START] task={env.task_id} env=credit_card_env model=custom", flush=True)
    return reward_obj


@app.post("/step", response_model=Reward)
def step(request: Action) -> Reward:
    try:
        reward_obj = env.step(request.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    print(
        f"[STEP] step={env.step_index} action={request.action} reward={reward_obj.reward:.2f} done={str(reward_obj.done).lower()} error=null",
        flush=True,
    )
    return reward_obj


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
