from gym_optimal_intrusion_response.dao.env.env_config import EnvConfig


class AttackerObservationState:

    def __init__(self, env_config : EnvConfig):
        self.env_config = env_config
