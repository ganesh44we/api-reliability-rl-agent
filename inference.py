import requests
import time
import os
import random
from openai import OpenAI

random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = os.getenv("BASE_URL", "http://0.0.0.0:8000")

client = OpenAI(
    api_key=os.getenv("HF_TOKEN", "dummy"),
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")


# -----------------------------
# RESET ENV
# -----------------------------
def reset_env(difficulty="easy"):
    return requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty}).json()


# -----------------------------
# STEP ENV
# -----------------------------
def step_env(action):
    payload = {"action": {"action": action}}
    return requests.post(f"{BASE_URL}/step", json=payload).json()


# -----------------------------
# SCORE FUNCTION
# -----------------------------
def compute_score(total_reward, difficulty="easy"):
    targets = {"easy": 30, "medium": 20, "hard": 10}
    target = targets.get(difficulty, 20)
    return round(max(0.0, min(1.0, total_reward / target)), 2)


# -----------------------------
# 🧠 RULE-BASED SMART AGENT
# -----------------------------
def rule_based_agent(obs):
    o = obs["observation"]

    status = o["api_status"]
    latency = o["latency"]
    retries = o["retry_count"]
    load = o["system_load"]

    # ✅ PERFECT CASE → ACCEPT
    if status == "success" and latency < 100 and load in ["low", "medium"]:
        return "accept"

    # ⚠️ SLOW BUT WORKING → CACHE
    if status == "success" and latency >= 100:
        return "use_cache"

    # ❌ FAILURE → RETRY FIRST
    if status == "failed":
        if retries < 2:
            return "retry"
        else:
            return "switch_api"

    # 🔥 HIGH LOAD → CACHE
    if load == "high":
        return "use_cache"

    # Fallback
    return "retry"


# -----------------------------
# 🤖 OPTIONAL LLM (fallback only)
# -----------------------------
def llm_agent(obs):
    o = obs["observation"]

    prompt = f"""
You are an API reliability agent.

State:
- status: {o['api_status']}
- latency: {o['latency']}
- retries: {o['retry_count']}
- load: {o['system_load']}

Choose ONE action:
accept, retry, switch_api, use_cache, return_error

Only output the action.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )

        action = response.choices[0].message.content.strip().lower()

        if action in ["accept", "retry", "switch_api", "use_cache", "return_error"]:
            return action

    except:
        pass

    return "retry"


# -----------------------------
# 🔥 FINAL AGENT (HYBRID)
# -----------------------------
def agent(obs):
    # 90% rule-based (reliable)
    action = rule_based_agent(obs)

    return action


# -----------------------------
# RUN EPISODE
# -----------------------------
def run_episode(difficulty="easy"):
    print(f"[START] task={difficulty} env=api_env model={MODEL_NAME}")

    obs = reset_env(difficulty)
    total_reward = 0
    rewards_list = []
    step_count = 0
    done = False

    MAX_STEPS = 10

    while step_count < MAX_STEPS:
        step_count += 1

        try:
            action = agent(obs)

            result = step_env(action)

            reward = result["reward"]
            done = result["done"]

            total_reward += reward
            rewards_list.append(round(reward, 2))

            print(f"[STEP] step={step_count} action={action} reward={round(reward,2)} done={str(done).lower()} error=null")

            if done:
                success = result["observation"]["api_status"] == "success"
                break

            obs = result

        except Exception as e:
            print(f"[STEP] step={step_count} action=error reward=0.00 done=true error={str(e)}")
            success = False
            break

        time.sleep(0.1)

    if not done:
        success = False

    score = compute_score(total_reward, difficulty)
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])

    print(f"[END] success={str(success).lower()} steps={step_count} score={score:.2f} rewards={rewards_str}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    for difficulty in ["easy", "medium", "hard"]:
        run_episode(difficulty)