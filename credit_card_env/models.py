from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


Difficulty = Literal["easy", "medium", "hard"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Transaction(StrictModel):
    amount: float = Field(..., gt=0)
    category: str


class Card(StrictModel):
    index: int = Field(..., ge=0, le=3)
    name: str
    cashback_rates: dict[str, float]
    annual_fee: float = Field(default=0.0, ge=0)


class Observation(StrictModel):
    task_id: Difficulty
    difficulty: Difficulty
    description: str
    step_index: int = Field(..., ge=0)
    num_steps: int = Field(..., ge=1)
    transaction: Transaction
    cards: list[Card] = Field(min_length=4, max_length=4)
    total_reward: float = 0.0


class Action(StrictModel):
    action: int = Field(..., ge=0, le=3)


class Reward(StrictModel):
    observation: Observation
    reward: float = 0.0
    score: float = 0.0
    done: bool


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="easy")
