from __future__ import annotations
import os
import traceback
from typing import Any
import httpx
from openai import OpenAI
from dotenv import load_dotenv # Added to load your .env file
from credit_card_env.client import CreditCardEnvClient

# Load the .env file automatically
load_dotenv()

# Mandatory Environment Variables
# These now use the values from your .env or fall back to defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") 
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://localhost:7860")
TASK_IDS = ("easy", "medium", "hard")

def create_http_client() -> httpx.Client:
    return httpx.Client(timeout=60.0)

def create_openai_client() -> OpenAI:
    if not HF_TOKEN:
        # If this hits, load_dotenv() failed or HF_TOKEN is missing from .env
        raise RuntimeError("HF_TOKEN is missing! Please check your .env file.")
    
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        http_client=create_http_client(),
    )

def build_prompt(transaction: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    return (
        "Analyze the transaction and available credit cards. "
        "Choose the single best card index (0, 1, 2, or 3). "
        "Return ONLY the integer.\n"
        f"Transaction: {transaction}\n"
        f"Cards: {cards}"
    )

def choose_card(client: OpenAI, observation: dict[str, Any]) -> int:
    obs = observation.get("observation", observation)
    transaction = obs.get("transaction", obs)
    cards = obs.get("cards", [])

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0, # Keep at 0 for consistency!
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are an expert Credit Card Reward Optimizer. "
                    "Your goal is to maximize cashback. "
                    "1. Analyze the transaction category (e.g., Food, Travel, Shopping). "
                    "2. Compare the cashback rates for each of the 4 cards (indices 0-3). "
                    "3. Select the card with the highest percentage for this specific transaction. "
                    "Output ONLY the integer index (0, 1, 2, or 3)."
                )
            },
            {"role": "user", "content": build_prompt(transaction, cards)},
        ],
    )

    content = (completion.choices[0].message.content or "").strip()
    # Robust extraction to ensure we always get a valid integer
    digits = [char for char in content if char in "0123"]
    return int(digits[0]) if digits else 0

def log_step(task_id: str, step_num: int, action: int, reward: float, done: bool, error: str) -> None:
    done_str = "true" if done else "false"
    print(
        f"[STEP] task={task_id} step={step_num} action={action} reward={reward:.2f} done={done_str} error={error}",
        flush=True,
    )

def run_task(task_id: str, env_client: CreditCardEnvClient, client: OpenAI) -> float:
    observation = env_client.reset(task_id)
    step_num = 1
    total_reward = 0.0

    while True:
        action, reward, done, error = -1, 0.0, False, "none"
        try:
            action = choose_card(client, observation)
            result = env_client.step(action)
            observation = result.get("observation", result)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            total_reward += reward
        except Exception as exc:
            done, error = True, str(exc).replace(" ", "_")
            log_step(task_id, step_num, action, reward, done, error)
            break 

        log_step(task_id, step_num, action, reward, done, error)
        if done:
            break
        step_num += 1
    return total_reward

def main() -> None:
    print("[START]", flush=True)
    total_reward, success = 0.0, True
    try:
        openai_client = create_openai_client()
        env_client = CreditCardEnvClient(base_url=SERVER_BASE_URL)

        for task_id in TASK_IDS:
            total_reward += run_task(task_id, env_client, openai_client)
    except Exception:
        success = False
    finally:
        success_str = "true" if success else "false"
        print(f"[END] success={success_str} total_reward={total_reward:.2f}", flush=True)

if __name__ == "__main__":
    main()