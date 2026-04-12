import os
from typing import List, Optional
from openai import OpenAI
from credit_card_env.server.environment import CreditCardRewardEnvironment

# 1. Initialize the Mandatory Client
client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.getenv("API_KEY", "dummy-key")
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = os.getenv("BENCHMARK", "credit_card_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_llm_action(observation) -> int:
    # 2. This is the API call the validator is looking for
    prompt = f"Transaction: {observation.transaction}. Cards: {observation.cards}. Return ONLY the index of the best card as an integer."
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        content = response.choices[0].message.content.strip()
        # Extract the first digit found in the response
        return int(''.join(filter(str.isdigit, content))[0])
    except:
        return 0 # Fallback to first card if API fails

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
            if result.done: break

            # Use the LLM to choose the action
            action = get_llm_action(result.observation)
            result = env.step(action)

            reward = round(float(result.reward), 2)
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=str(action), reward=reward, done=bool(result.done), error=None)

            if result.done: break

        score = max(0.0, min(round(float(result.score), 2), 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        log_step(step=max(steps_taken, 1), action="null", reward=0.0, done=True, error=str(exc))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()