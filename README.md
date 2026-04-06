

#  Cost-Aware API Reliability RL Environment

## 🧠 Overview

This project implements a **real-world reinforcement learning (RL) environment** that simulates API reliability challenges in backend systems.

Agents must make intelligent decisions under uncertainty to balance:

*  Success rate
*  Latency
*  Cost

This models real-world scenarios like microservice failures, API retries, and fallback strategies.

---

## 🎯 Objective

Enable agents to learn optimal strategies for handling unreliable APIs using the OpenEnv framework.

---

##  State Space (Observation)

| Feature       | Description                      |
| ------------- | -------------------------------- |
| `api_status`  | success / slow / failed          |
| `latency`     | Response time in ms              |
| `retry_count` | Number of retries performed      |
| `api_cost`    | Cost of API usage                |
| `system_load` | low / medium / high              |
| `message`     | Human-readable feedback per step |

---

## ⚡ Action Space

| Action         | Description                                       |
| -------------- | ------------------------------------------------- |
| `accept`       | Accept the current response as-is                 |
| `retry`        | Retry the same API (increments retry_count)       |
| `switch_api`   | Switch between API A (cheap) and API B (reliable) |
| `use_cache`    | Use cached response — fast, free, 80% hit rate    |
| `return_error` | Stop and return failure immediately               |

---

##  Environment Dynamics

* **API A** → cheaper but less reliable (fail_prob +0.1)
* **API B** → more reliable but higher cost (fail_prob −0.1)
* Failures persist across steps (**temporal memory**)
* Repeated retries increase failure probability (**cascading effect**)
* Switching API resets retry count and sets load to medium
* High system load adds +0.1 to failure probability

---

## 🏆 Reward Function

| Component              | Value                         |
| ---------------------- | ----------------------------- |
| Successful response    | +8                            |
| Failed response        | −8                            |
| Latency penalty        | −0.02 × latency (ms)          |
| Cost penalty           | −5 × api_cost                 |
| Retry penalty          | −2 × retry_count              |
| Correct decision bonus | +5 (right action for context) |
| Wrong decision penalty | −5 (wrong action for context) |
| Repeated action penalty| −2                            |

**Context bonuses:**
- Previous status `success` + latency < 120ms → `accept` gives +5, anything else −5
- Previous status `failed` → `retry` or `switch_api` gives +4, else −4
- Previous status `slow` → `use_cache` or `switch_api` gives +3, else −2

---

##  Tasks (Difficulty Levels)

| Task   | Fail Probability | Description                           |
| ------ | ---------------- | ------------------------------------- |
| Easy   | ~30%             | Low failure, good for initial learning |
| Medium | ~50%             | Moderate failures, balanced challenge  |
| Hard   | ~70%             | High failure + high load cascades      |

Episodes end after **5 steps** or **5 retries** (whichever comes first).

---

## 🤖 Agents

### Q-Learning Agent (rl-agent space)

The primary agent uses **tabular Q-learning** with ε-greedy exploration:

* **State representation:** `(api_status, latency // 50, system_load)`
* **Learning rate (α):** 0.1
* **Discount factor (γ):** 0.9
* **Exploration rate (ε):** 0.6 — 60% random, 40% greedy from Q-table
* **Q-update (Bellman equation):**

```
Q[s][a] += α × (reward + γ × max(Q[s']) − Q[s][a])
```

The agent also shows the **suggested best action** from the current Q-table alongside the action it actually takes — so you can watch it learn in real time.

### Rule-Based Agent (inference.py)

A deterministic heuristic agent used for reproducible baseline scoring:

| Condition                             | Action    |
| ------------------------------------- | --------- |
| success + latency < 100 + low/med load| accept    |
| success + latency ≥ 100               | use_cache |
| failed + retries < 2                  | retry     |
| failed + retries ≥ 2                  | switch_api|
| high load                             | use_cache |

### LLM Agent (rl-env space)

Uses **Qwen/Qwen2.5-7B-Instruct** via HF router to generate a one-line natural language explanation for each decision — adds interpretability to every step.

---

## 📊 Evaluation & Scoring

Scores are computed per episode using difficulty-normalized total reward:

```python
def compute_score(total_reward, difficulty):
    targets = {"easy": 30, "medium": 20, "hard": 10}
    target = targets.get(difficulty, 20)
    return round(max(0.0, min(1.0, total_reward / target)), 2)
```

* Score of **1.0** = perfect episode
* Score of **0.0** = agent failed completely
* Partial scores reflect partial progress

---

## 🧪 Inference Script

Run baseline agents across all difficulty levels with reproducible results:

```bash
python inference.py
```

Seeded with `random.seed(42)` for reproducibility. Runs easy → medium → hard and prints per-step rewards and final score.

---

## 🌐 API Endpoints

### Reset Environment

```
POST /reset
Body: { "difficulty": "easy" | "medium" | "hard" }
```

### Take a Step

```
POST /step
Body: { "action": { "action": "accept" | "retry" | "switch_api" | "use_cache" | "return_error" } }
```

### Get Current State

```
GET /state
```

---

## 🤖 AI Integration

The **rl-env** space uses an OpenAI-compatible API (Qwen2.5-7B via Hugging Face router) to generate real-time explanations for each agent decision, improving interpretability and debuggability.

---

##  Live Demos

| Space                        | URL                                                      |
| ---------------------------- | -------------------------------------------------------- |
| RL Agent (Q-learning)        | https://rahilahmed1945-api-reliability-rl-agent.hf.space |
| RL Environment (manual + LLM)| https://rahilahmed1945-api-reliability-rl-env.hf.space   |

---

## 🛠️ Tech Stack

* **OpenEnv** — RL environment framework
* **FastAPI** — backend API server
* **Gradio** — interactive UI
* **Docker** — containerised deployment
* **Hugging Face Spaces** — live hosting
* **Qwen2.5-7B-Instruct** — LLM for decision explanation

---

## 📦 Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI backend (terminal 1)
uvicorn server.app:app --reload --port 8000

# Start the Gradio frontend (terminal 2)
python app.py
```

Then open http://localhost:7860 for the UI and http://localhost:8000/docs for the API.

---

## 🐳 Docker Setup

```bash
docker build -t api-env .
docker run -p 7860:7860 -p 8000:8000 api-env
```

---

## 📁 Project Structure

```
api-reliability-rl-agent/
├── app.py                  # Gradio UI + Q-learning agent
├── inference.py            # Baseline rule-based + LLM agents
├── models.py               # Typed Action / Observation / State models
├── openenv.yaml            # OpenEnv compliance manifest
├── requirements.txt        # Dependencies
├── Dockerfile              # Container config
└── server/
    ├── app.py              # FastAPI app (OpenEnv create_fastapi_app)
    └── environment.py      # Core RL environment logic
```

---

## 🧠 OpenEnv Compliance

*  Typed `Action` / `Observation` / `State` models (Pydantic)
*  `step()`, `reset()`, `state` property implemented
*  `openenv.yaml` included
*  Dockerized deployment
*  HF Space live and running
*  Minimum 3 difficulty tasks with 0.0–1.0 scoring
*  Baseline inference script with reproducible scores (`random.seed(42)`)

---

## 🎯 Key Highlights

* Real-world API reliability simulation (not a game or toy)
* Dual API model — API A (cheap/unreliable) vs API B (reliable/costly)
* Temporal memory — failures and retries compound across steps
* Multi-component reward with partial progress signals
* Q-learning agent that visibly improves over steps
* LLM-based interpretability for every decision
* Fully OpenEnv-compliant and Dockerized

---

## 👥 Team

| Name             | Email                                                        | 
| ---------------- | ------------------------------------------------------------ | 
| Ganesh Rayapati  | [ganeshrayapati44@gmail.com](mailto:ganeshrayapati44@gmail.com) |
| Rahil Ahmed      | [rahilahmed1305@gmail.com](mailto:rahilahmed1305@gmail.com)  | 
| PALETI SAI TARUN | [saitarunpaleti@gmail.com](mailto:saitarunpaleti@gmail.com)  | 

---

*Built for the OpenEnv × Scaler Meta PyTorch Hackathon*
