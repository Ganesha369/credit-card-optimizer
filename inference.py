from __future__ import annotations

import os
from typing import List, Optional

from credit_card_env.server.environment import CreditCardRewardEnvironment

MODEL_NAME = os.getenv("MODEL_NAME", "custom")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = os.getenv("BENCHMARK", "credit_card_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def choose_best_card(observation) -> int:
    transaction = observation.transaction
    best_index = 0
    best_value = float("-inf")

    for card in observation.cards:
        rate = card.cashback_rates.get(transaction.category, card.cashback_rates.get("other", 0.0))
        cashback_value = transaction.amount * rate
        if cashback_value > best_value:
            best_value = cashback_value
            best_index = card.index

    return best_index


def main() -> None:
    env = CreditCardRewardEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(TASK_NAME)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = choose_best_card(result.observation)
            result = env.step(action)

            reward = round(float(result.reward), 2)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=str(action),
                reward=reward,
                done=bool(result.done),
                error=None,
            )

            if result.done:
                break

        score = max(0.0, min(round(float(result.score), 2), 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        log_step(
            step=max(steps_taken, 1),
            action="null",
            reward=0.0,
            done=True,
            error=str(exc),
        )
        success = False
        score = 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
