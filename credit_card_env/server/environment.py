from __future__ import annotations

import random
from typing import Final

from credit_card_env.models import Card, Observation, Reward, Transaction

TASK_CONFIG: Final[dict[str, dict[str, int | str]]] = {
    "easy": {
        "num_steps": 1,
        "description": "Single transaction: pick the card with the highest cashback for the active category.",
    },
    "medium": {
        "num_steps": 3,
        "description": "Multiple transactions: maximize cumulative cashback across overlapping card strengths.",
    },
    "hard": {
        "num_steps": 5,
        "description": "Longer horizon: balance category rewards against higher-fee premium cards.",
    },
}

CARD_LIBRARY: Final[dict[str, list[Card]]] = {
    "easy": [
        Card(index=0, name="card0", cashback_rates={"grocery": 0.03, "dining": 0.01, "travel": 0.01, "other": 0.01}, annual_fee=0.0),
        Card(index=1, name="card1", cashback_rates={"grocery": 0.01, "dining": 0.02, "travel": 0.01, "other": 0.01}, annual_fee=0.0),
        Card(index=2, name="card2", cashback_rates={"grocery": 0.015, "dining": 0.015, "travel": 0.015, "other": 0.015}, annual_fee=0.0),
        Card(index=3, name="card3", cashback_rates={"grocery": 0.005, "dining": 0.005, "travel": 0.04, "other": 0.005}, annual_fee=0.0),
    ],
    "medium": [
        Card(index=0, name="card0", cashback_rates={"grocery": 0.03, "dining": 0.01, "travel": 0.015, "fuel": 0.01, "other": 0.01}, annual_fee=0.0),
        Card(index=1, name="card1", cashback_rates={"grocery": 0.01, "dining": 0.02, "travel": 0.02, "fuel": 0.015, "other": 0.01}, annual_fee=0.0),
        Card(index=2, name="card2", cashback_rates={"grocery": 0.015, "dining": 0.03, "travel": 0.01, "fuel": 0.01, "other": 0.01}, annual_fee=0.0),
        Card(index=3, name="card3", cashback_rates={"grocery": 0.01, "dining": 0.01, "travel": 0.035, "fuel": 0.025, "other": 0.01}, annual_fee=0.0),
    ],
    "hard": [
        Card(index=0, name="card0", cashback_rates={"grocery": 0.025, "dining": 0.02, "travel": 0.03, "fuel": 0.015, "online": 0.02, "other": 0.01}, annual_fee=999.0),
        Card(index=1, name="card1", cashback_rates={"grocery": 0.02, "dining": 0.03, "travel": 0.02, "fuel": 0.01, "online": 0.015, "other": 0.01}, annual_fee=199.0),
        Card(index=2, name="card2", cashback_rates={"grocery": 0.015, "dining": 0.015, "travel": 0.04, "fuel": 0.03, "online": 0.015, "other": 0.01}, annual_fee=499.0),
        Card(index=3, name="card3", cashback_rates={"grocery": 0.03, "dining": 0.015, "travel": 0.015, "fuel": 0.015, "online": 0.035, "other": 0.012}, annual_fee=0.0),
    ],
}

TRANSACTION_POOLS: Final[dict[str, list[Transaction]]] = {
    "easy": [
        Transaction(amount=1250.0, category="grocery"),
        Transaction(amount=860.0, category="dining"),
        Transaction(amount=4200.0, category="travel"),
    ],
    "medium": [
        Transaction(amount=1800.0, category="grocery"),
        Transaction(amount=950.0, category="dining"),
        Transaction(amount=3200.0, category="travel"),
        Transaction(amount=2200.0, category="fuel"),
    ],
    "hard": [
        Transaction(amount=2450.0, category="grocery"),
        Transaction(amount=1350.0, category="dining"),
        Transaction(amount=7800.0, category="travel"),
        Transaction(amount=2600.0, category="fuel"),
        Transaction(amount=4100.0, category="online"),
    ],
}

DEFAULT_TASK_ID: Final[str] = "easy"


class CreditCardRewardEnvironment:
    def __init__(self) -> None:
        self._rng = random.Random(7)
        self.task_id = DEFAULT_TASK_ID
        self.step_index = 0
        self.total_reward = 0.0
        self.done = False
        self.current_transaction = TRANSACTION_POOLS[DEFAULT_TASK_ID][0]

    def reset(self, task_id: str | None = "easy") -> Reward:
        self.task_id = self._normalize_task_id(task_id)
        self.step_index = 0
        self.total_reward = 0.0
        self.done = False
        self.current_transaction = self._sample_transaction()
        return self._build_response(reward=0.0)

    def step(self, action: int) -> Reward:
        if self.done:
            raise ValueError("Episode already completed. Call /reset before /step.")
        if not 0 <= action <= 3:
            raise ValueError("Action must be between 0 and 3.")

        cards = CARD_LIBRARY[self.task_id]
        best_index = self._best_card_index(self.current_transaction)
        selected_value = self._cashback_value(cards[action], self.current_transaction)
        best_value = self._cashback_value(cards[best_index], self.current_transaction)

        raw_reward = 0.0 if best_value <= 0 else selected_value / best_value
        reward = max(0.0, min(round(raw_reward, 2), 1.0))

        self.total_reward = round(self.total_reward + reward, 2)
        self.step_index += 1
        self.done = self.step_index >= int(TASK_CONFIG[self.task_id]["num_steps"])

        if not self.done:
            self.current_transaction = self._sample_transaction()

        return self._build_response(reward=reward)

    def _normalize_task_id(self, task_id: str | None) -> str:
        if task_id is None:
            return DEFAULT_TASK_ID

        normalized = str(task_id).strip().lower()
        if not normalized or normalized not in TASK_CONFIG:
            return DEFAULT_TASK_ID
        return normalized

    def _current_score(self) -> float:
        max_steps = int(TASK_CONFIG[self.task_id]["num_steps"])
        if max_steps <= 0:
            return 0.0
        return max(0.0, min(round(self.total_reward / max_steps, 2), 1.0))

    def _current_observation(self) -> Observation:
        config = TASK_CONFIG[self.task_id]
        return Observation(
            task_id=self.task_id,
            difficulty=self.task_id,
            description=str(config["description"]),
            step_index=self.step_index,
            num_steps=int(config["num_steps"]),
            transaction=self.current_transaction,
            cards=CARD_LIBRARY[self.task_id],
            total_reward=round(self.total_reward, 2),
        )

    def _build_response(self, reward: float) -> Reward:
        final_reward = max(0.0, min(round(float(reward), 2), 1.0))
        return Reward(
            observation=self._current_observation(),
            reward=final_reward,
            score=self._current_score(),
            done=self.done,
        )

    def _sample_transaction(self) -> Transaction:
        return self._rng.choice(TRANSACTION_POOLS[self.task_id])

    @staticmethod
    def _cashback_value(card: Card, transaction: Transaction) -> float:
        rate = card.cashback_rates.get(transaction.category, card.cashback_rates.get("other", 0.0))
        return transaction.amount * rate

    def _best_card_index(self, transaction: Transaction) -> int:
        values = [self._cashback_value(card, transaction) for card in CARD_LIBRARY[self.task_id]]
        return max(range(len(values)), key=values.__getitem__)
