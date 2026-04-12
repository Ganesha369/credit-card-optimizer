from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Transaction:
    amount: float
    category: str

@dataclass
class Card:
    index: int
    name: str
    cashback_rates: dict[str, float]

@dataclass
class Observation:
    task_id: str
    difficulty: str
    description: str
    step_index: int
    num_steps: int
    transaction: Transaction
    cards: list[Card]
    total_reward: float

@dataclass
class Reward:
    observation: Observation
    reward: float
    score: float
    done: bool

# --- MANDATORY TASK CONFIGURATION ---
TASK_CONFIG = {
    "easy": {
        "num_steps": 3,
        "description": "Basic categories with clear best choices."
    },
    "medium": {
        "num_steps": 5,
        "description": "Moderate complexity with varied transaction amounts."
    },
    "hard": {
        "num_steps": 10,
        "description": "High complexity with many steps and overlapping categories."
    }
}

DEFAULT_TASK_ID = "easy"

# --- CARD LIBRARIES FOR EACH DIFFICULTY ---
CARD_LIBRARY = {
    "easy": [
        Card(index=0, name="Foodie", cashback_rates={"food": 0.05, "other": 0.01}),
        Card(index=1, name="GasPro", cashback_rates={"fuel": 0.05, "other": 0.01}),
        Card(index=2, name="ShopMax", cashback_rates={"shopping": 0.05, "other": 0.01}),
        Card(index=3, name="Basic", cashback_rates={"other": 0.02}),
    ],
    "medium": [
        Card(index=0, name="Traveler", cashback_rates={"travel": 0.06, "food": 0.02}),
        Card(index=1, name="Retailer", cashback_rates={"shopping": 0.05, "fuel": 0.02}),
        Card(index=2, name="Global", cashback_rates={"travel": 0.03, "other": 0.02}),
        Card(index=3, name="Utility", cashback_rates={"fuel": 0.04, "other": 0.01}),
    ],
    "hard": [
        Card(index=0, name="Elite Travel", cashback_rates={"travel": 0.07, "dining": 0.03}),
        Card(index=1, name="Premium Dining", cashback_rates={"dining": 0.07, "shopping": 0.02}),
        Card(index=2, name="Hardcore Shop", cashback_rates={"shopping": 0.06, "travel": 0.01}),
        Card(index=3, name="AllRound", cashback_rates={"other": 0.03}),
    ]
}

# --- TRANSACTION POOLS ---
TRANSACTION_POOLS = {
    "easy": [
        Transaction(amount=100.0, category="food"),
        Transaction(amount=50.0, category="fuel")
    ],
    "medium": [
        Transaction(amount=200.0, category="shopping"),
        Transaction(amount=500.0, category="travel"),
        Transaction(amount=150.0, category="food")
    ],
    "hard": [
        Transaction(amount=1000.0, category="travel"),
        Transaction(amount=400.0, category="dining"),
        Transaction(amount=80.0, category="fuel"),
        Transaction(amount=300.0, category="shopping")
    ]
}

class CreditCardRewardEnvironment:
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self.task_id = DEFAULT_TASK_ID
        self.step_index = 0
        self.total_reward = 0.0
        self.done = False
        self.current_transaction = None

    def reset(self, task_id: str | None = None) -> Reward:
        self.task_id = self._normalize_task_id(task_id)
        self.step_index = 0
        self.total_reward = 0.0
        self.done = False
        self.current_transaction = self._sample_transaction()
        return self._build_response(reward=0.0)

    def step(self, action: int) -> Reward:
        if self.done:
            raise ValueError("Episode already completed. Call /reset before /step.")
        
        # Action validation
        action = max(0, min(int(action), 3))

        cards = CARD_LIBRARY[self.task_id]
        best_index = self._best_card_index(self.current_transaction)
        
        selected_value = self._cashback_value(cards[action], self.current_transaction)
        best_value = self._cashback_value(cards[best_index], self.current_transaction)

        # Reward calculation: Perfect choice = 1.0 reward
        raw_reward = 1.0 if (best_value <= 0 and selected_value <= 0) else (selected_value / best_value if best_value > 0 else 0.0)
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
        return normalized if normalized in TASK_CONFIG else DEFAULT_TASK_ID

    def _current_score(self) -> float:
        max_steps = int(TASK_CONFIG[self.task_id]["num_steps"])
        if max_steps <= 0:
            return 0.0
        # Normalizes total reward by the number of steps
        return max(0.01, min(round(self.total_reward / max_steps, 2), 0.99))

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
        return Reward(
            observation=self._current_observation(),
            reward=max(0.0, min(float(reward), 1.0)),
            score=self._current_score(),
            done=self.done,
        )

    def _sample_transaction(self) -> Transaction:
        pool = TRANSACTION_POOLS[self.task_id]
        return self._rng.choice(pool)

    @staticmethod
    def _cashback_value(card: Card, transaction: Transaction) -> float:
        rate = card.cashback_rates.get(transaction.category, card.cashback_rates.get("other", 0.0))
        return transaction.amount * rate

    def _best_card_index(self, transaction: Transaction) -> int:
       values = [self._cashback_value(card, transaction) for card in CARD_LIBRARY[self.task_id]]
    # This finds the index of the highest value in the list
       return int(np.argmax(values))