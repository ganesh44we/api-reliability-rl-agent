from openenv.core.env_server import create_fastapi_app
from server.environment import APIEnvironment
from models import APIAction, APIObservation

app = create_fastapi_app(
    APIEnvironment,
    action_cls=APIAction,
    observation_cls=APIObservation
)