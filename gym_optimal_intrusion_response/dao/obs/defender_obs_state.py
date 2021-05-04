import numpy as np
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig


class DefenderObservationState:

    def __init__(self, env_config : EnvConfig):
        self.env_config = env_config
        self.num_alerts = 0
        self.num_failed_logins = 0


    def get_defender_observation(self):
        obs = np.zeros(2).tolist()
        obs[0] = self.num_alerts
        obs[1] = self.num_failed_logins
        return np.array(obs)