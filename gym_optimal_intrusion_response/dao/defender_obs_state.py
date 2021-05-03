import gym
import numpy as np
from gym_optimal_intrusion_response.dao.env_config import EnvConfig


class DefenderObservationState:

    def __init__(self, env_config : EnvConfig):
        self.env_config = env_config
        self.num_alerts = 0
        self.num_failed_logins = 0