import gradio as gr
import requests
import os
import random

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

ACTIONS = ["accept", "retry", "switch_api", "use_cache", "return_error"]

Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.6


# -----------------------------
# ENV
# -----------------------------
def reset_env(difficulty):
    return requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty}).json()

def step_env(action):
    return requests.post(
        f"{BASE_URL}/step",
        json={"action": {"action": action}}
    ).json()


# -----------------------------
# STATE
# -----------------------------
def get_state(obs):
    return (
        obs["api_status"],
        int(obs["latency"] // 50),
        obs["system_load"]
    )


# -----------------------------
# AGENT
# -----------------------------
def agent(obs):
    state = get_state(obs)

    if state not in Q or random.random() < epsilon:
        return random.choice(ACTIONS)

    return max(Q[state], key=Q[state].get)


# -----------------------------
# UPDATE
# -----------------------------
def update_q(obs, action, reward, next_obs):
    reward = max(min(reward, 10), -10)

    s = get_state(obs)
    ns = get_state(next_obs)

    if s not in Q:
        Q[s] = {a: 0 for a in ACTIONS}
    if ns not in Q:
        Q[ns] = {a: 0 for a in ACTIONS}

    Q[s][action] += alpha * (
        reward + gamma * max(Q[ns].values()) - Q[s][action]
    )


# -----------------------------
# LABEL
# -----------------------------
def get_label(reward):
    if reward >= 8:
        return "GOOD"
    elif reward >= 1:
        return "OKAY"
    else:
        return "BAD"


def compute_score(reward):
    return 1 if reward >= 8 else (0.5 if reward >= 1 else 0)


# -----------------------------
# MAIN (🔥 FAST RL + SUGGESTION)
# -----------------------------
def run_step(difficulty):

    # reset env
    obs = reset_env(difficulty)["observation"]

    # 🔥 suggested best action
    state = get_state(obs)

    if state in Q:
        best_action = max(Q[state], key=Q[state].get)
    else:
        best_action = "exploring"

    # 🔥 agent action
    action = agent(obs)

    # take step
    res = step_env(action)
    next_obs = res["observation"]
    reward = res["reward"]

    # update learning
    update_q(obs, action, reward, next_obs)

    label = get_label(reward)

    return (
        best_action,
        action,
        next_obs["api_status"],
        round(next_obs["latency"], 2),
        next_obs["retry_count"],
        round(next_obs["api_cost"], 3),
        next_obs["system_load"],
        round(reward, 2),
        compute_score(reward),
        f"{label} - Agent decision",
        True,
        len(Q)
    )


# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Smart API RL Agent")
    gr.Markdown("Suggests best action + learns from outcomes")

    difficulty = gr.Dropdown(["easy", "medium", "hard"], value="easy")

    run_btn = gr.Button("Run")

    best_action_box = gr.Textbox(label="💡 Suggested Best Action")
    taken_action_box = gr.Textbox(label="🤖 Agent Action")

    api_status = gr.Textbox(label="API Status")
    latency = gr.Number(label="Latency")
    retry = gr.Number(label="Retries")

    cost = gr.Number(label="Cost")
    load = gr.Textbox(label="Load")

    reward = gr.Number(label="Reward")
    score = gr.Number(label="Score")

    explanation = gr.Textbox(label="Decision")
    done = gr.Checkbox()

    states = gr.Number(label="States Learned")

    run_btn.click(
        run_step,
        [difficulty],
        [
            best_action_box,
            taken_action_box,
            api_status,
            latency,
            retry,
            cost,
            load,
            reward,
            score,
            explanation,
            done,
            states
        ]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)