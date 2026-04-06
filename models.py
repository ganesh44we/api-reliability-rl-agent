from typing import Literal
from openenv.core.env_server import Action, Observation, State


# =========================
# ACTION (what agent does)
# =========================
class APIAction(Action):
    action: Literal["accept", "retry", "switch_api", "use_cache", "return_error"]


# =========================
# OBSERVATION (what agent sees)
# =========================
class APIObservation(Observation):
    api_status: str              # success / slow / failed
    latency: float              # ms
    retry_count: int
    api_cost: float
    system_load: str            # low / medium / high
    message: str                # explanation / feedback


# =========================
# STATE (internal memory)
# =========================
class APIState(State):
    current_api: str = "A"
    step_count: int = 0
    retry_count: int = 0
    done: bool = False