import numpy as np
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
from gym_optimal_intrusion_response.logic.defender_dynamics.defender_dynamics import DefenderDynamics
import gym_optimal_intrusion_response.constants.constants as constants
from gym_optimal_intrusion_response.dao.dp.dp_setup import DPSetup


class DefenderObservationState:

    def __init__(self, env_config : EnvConfig):
        self.env_config = env_config
        self.num_alerts = 0
        self.num_failed_logins = 0
        self.ttc = constants.DP.MAX_TTC-1
        # self.ttc = 10
        self.f1_a = DefenderDynamics.f1_a()
        self.f1_b = DefenderDynamics.f1_b()
        self.f2_a = DefenderDynamics.f2_a()
        self.f2_b = DefenderDynamics.f2_b()

    def get_defender_observation(self, t, dp_setup: DPSetup = None):
        if not self.env_config.dp:
            obs = np.zeros(3).tolist()
            obs[0] = t
            obs[1] = self.num_alerts
            obs[2] = self.num_failed_logins
        else:
            obs = np.zeros(2).tolist()
            obs[0] = t
            obs[1] = self.ttc
        return np.array(obs)

    def update_state(self, t, intrusion_in_progress: bool = False, dp_setup: DPSetup = None):
        if not self.env_config.dp:
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
        else:
            print("updating TTC")
            state = (t, self.ttc)
            state_id = dp_setup.state_to_id[state]
            next_state_id = np.random.choice(dp_setup.state_ids, p=dp_setup.T[state_id][0])
            next_state = dp_setup.id_to_state[next_state_id]
            (t2, ttc2) = next_state
            print("old ttc:{}, new ttc:{}, state id:{}, new state id:{}, prob:{}".format(
                self.ttc, ttc2, state_id, next_state_id, dp_setup.T[state_id][0][next_state_id]))
            self.ttc = ttc2


    def probability_of_intrusion(self, t: int):
        ttc = DefenderDynamics.ttc(self.num_alerts, self.num_failed_logins, constants.DP.MAX_ALERTS)
        print("ttc:{}".format(ttc))
        hp = DefenderDynamics.hack_prob(ttc)
        return hp