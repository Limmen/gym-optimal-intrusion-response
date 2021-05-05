import numpy as np
import math
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
from gym_optimal_intrusion_response.logic.defender_dynamics.defender_dynamics import DefenderDynamics
import gym_optimal_intrusion_response.constants.constants as constants


class DefenderObservationState:

    def __init__(self, env_config : EnvConfig):
        self.env_config = env_config
        self.num_alerts = 0
        self.num_failed_logins = 0
        self.f1_a = DefenderDynamics.f1_a()
        self.f1_b = DefenderDynamics.f1_b()
        self.f2_a = DefenderDynamics.f2_a()
        self.f2_b = DefenderDynamics.f2_b()

    def get_defender_observation(self):
        obs = np.zeros(2).tolist()
        obs[0] = self.num_alerts
        obs[1] = self.num_failed_logins
        return np.array(obs)

    def update_state(self, intrusion_in_progress: bool = False):
        if intrusion_in_progress:
            new_alerts = int(round(self.f1_a.rvs(size=1)[0]))
            new_logins = int(round(self.f2_a.rvs(size=1)[0]))
        else:
            new_alerts = int(round(self.f1_b.rvs(size=1)[0]))
            new_logins = int(round(self.f2_b.rvs(size=1)[0]))
        self.num_alerts = self.num_alerts + new_alerts
        self.num_failed_logins = self.num_failed_logins + new_logins
        if self.env_config.use_state_limits:
            self.num_alerts = min(constants.DP.MAX_ALERTS, self.num_alerts)
            self.num_failed_logins = min(constants.DP.MAX_LOGINS, self.num_failed_logins)


    def probability_of_intrusion(self, t: int):
        ttc = DefenderDynamics.ttc(self.num_alerts, self.num_failed_logins, constants.DP.MAX_ALERTS)
        hp = DefenderDynamics.hack_prob(ttc, t)
        return hp