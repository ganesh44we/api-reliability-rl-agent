import random
import uuid

from openenv.core.env_server import Environment
from models import APIAction, APIObservation, APIState


class APIEnvironment(Environment):

    def __init__(self):
        self._state = APIState()
        self.current_status = "success"
        self.latency = 100
        self.cost = 0.01
        self.system_load = "low"
        self.difficulty = "easy"
        self.last_action = None

    def reset(self, seed=None, episode_id=None, difficulty="easy", **kwargs):
        # ❌ DO NOT FIX RANDOMNESS
        # Remove seeding completely for real randomness

        self.difficulty = difficulty

        self._state = APIState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_api="A",
            retry_count=0,
            done=False
        )

        # 🔥 RANDOMIZE EVERY RESET
        self.system_load = random.choice(["low", "medium", "high"])

        # 🔥 simulate AFTER setting load
        status, latency, cost = self.simulate_api()

        self.current_status = status
        self.latency = latency
        self.cost = cost

        return APIObservation(
            done=False,
            reward=None,
            api_status=self.current_status,
            latency=self.latency,
            retry_count=self._state.retry_count,
            api_cost=self.cost,
            system_load=self.system_load,
            message="Environment reset"
        )

    def simulate_api(self):
        if self.difficulty == "easy":
            fail_prob = 0.3
        elif self.difficulty == "medium":
            fail_prob = 0.5
        else:
            fail_prob = 0.7

        if self.system_load == "high":
            fail_prob += 0.1

        if self._state.current_api == "A":
            fail_prob += 0.1
        else:
            fail_prob -= 0.1

        fail_prob = max(0.0, min(1.0, fail_prob))

        rand = random.random()

        if rand < fail_prob:
            return "failed", random.uniform(300, 800), random.uniform(0.01, 0.1)
        elif rand < fail_prob + 0.1:
            return "slow", random.uniform(200, 500), random.uniform(0.01, 0.1)
        else:
            return "success", random.uniform(50, 200), random.uniform(0.01, 0.1)

    def step(self, action: APIAction, timeout_s=None, **kwargs):
        self._state.step_count += 1

        prev_status = self.current_status
        prev_latency = self.latency

        # ---------------- ACTIONS ----------------
        if action.action == "accept":
            pass

        elif action.action == "retry":
            self._state.retry_count += 1
            self.current_status, self.latency, self.cost = self.simulate_api()

        elif action.action == "switch_api":
            self._state.current_api = "B" if self._state.current_api == "A" else "A"
            self.system_load = "medium"
            self._state.retry_count = max(0, self._state.retry_count - 1)
            self.current_status, self.latency, self.cost = self.simulate_api()

        elif action.action == "use_cache":
            self.current_status = "success" if random.random() < 0.8 else "failed"
            self.latency = random.uniform(10, 50)
            self.cost = 0.0

        elif action.action == "return_error":
            self.current_status = "failed"
            self.latency = 0
            self.cost = 0.0

        else:
            self.current_status, self.latency, self.cost = self.simulate_api()

        # ---------------- REWARD ----------------
        reward = 0

        if self.current_status == "success":
            reward += 8
        else:
            reward -= 8

        reward -= 0.02 * self.latency
        reward -= 5 * self.cost
        reward -= 2 * self._state.retry_count

        # 🔥 Decision quality
        if prev_status == "success" and prev_latency < 120:
            reward += 5 if action.action == "accept" else -5

        if prev_status == "failed":
            reward += 4 if action.action in ["retry", "switch_api"] else -4

        if prev_status == "slow":
            reward += 3 if action.action in ["use_cache", "switch_api"] else -2

        if self.last_action == action.action:
            reward -= 2

        self.last_action = action.action

        done = self._state.step_count >= 5 or self._state.retry_count >= 5

        return APIObservation(
            done=done,
            reward=reward,
            api_status=self.current_status,
            latency=self.latency,
            retry_count=self._state.retry_count,
            api_cost=self.cost,
            system_load=self.system_load,
            message=f"Action taken: {action.action}"
        )

    # -----------------------------
    # ✅ REQUIRED FOR OPENENV
    # -----------------------------
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value