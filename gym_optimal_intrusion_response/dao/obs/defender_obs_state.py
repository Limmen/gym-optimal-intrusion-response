import numpy as np
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
from gym_optimal_intrusion_response.logic.defender_dynamics.defender_dynamics import DefenderDynamics
import gym_optimal_intrusion_response.constants.constants as constants
from gym_optimal_intrusion_response.dao.dp.dp_setup import DPSetup
from gym_pycr_ctf.dao.defender_dynamics.defender_dynamics_model import DefenderDynamicsModel


class DefenderObservationState:

    def __init__(self, env_config : EnvConfig):
        self.env_config = env_config
        self.num_alerts = 0
        self.num_failed_logins = 0
        self.ttc = constants.DP.MAX_TTC - 1
        self.f1_a = DefenderDynamics.f1_a()
        self.f1_b = DefenderDynamics.f1_b()
        self.f2_a = DefenderDynamics.f2_a()
        self.f2_b = DefenderDynamics.f2_b()
        self.reset()


    def reset(self):
        self.num_alerts = 0
        self.num_failed_logins = 0
        self.ttc = constants.DP.MAX_TTC - 1

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

    def update_state(self, t, intrusion_in_progress: bool = False, dp_setup: DPSetup = None,
                     attacker_action: int = None, defender_dynamics_model : DefenderDynamicsModel = None):
        if not self.env_config.dp and not self.env_config.traces:
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
        elif self.env_config.dp:
            state = (t, self.ttc)
            state_id = dp_setup.state_to_id[state]
            next_state_id = np.random.choice(dp_setup.state_ids, p=dp_setup.T[state_id][0])
            next_state = dp_setup.id_to_state[next_state_id]
            done = False
            if next_state == "terminal":
                done = True
            else:
                (t2, ttc2) = next_state
                # print("t1:{}, t2:{} old ttc:{}, new ttc:{}, state id:{}, new state id:{}, prob:{}".format(t, t2,
                #     self.ttc, ttc2, state_id, next_state_id, dp_setup.T[state_id][0][next_state_id]))
                self.ttc = ttc2
            return done
        elif self.env_config.traces:
            num_new_alerts = 0
            num_new_severe_alerts = 0
            num_new_warning_alerts = 0
            logged_in_ips_str, done, intrusion_in_progress = self.env_config.action_to_state[(attacker_action, t)]
            attacker_action_id = self.env_config.attack_idx_to_id[attacker_action]

            if (attacker_action_id, logged_in_ips_str) in \
                    defender_dynamics_model.norm_num_new_severe_alerts:
                num_new_severe_alerts = \
                    defender_dynamics_model.norm_num_new_severe_alerts[
                        (attacker_action_id, logged_in_ips_str)].rvs()

            if (attacker_action_id, logged_in_ips_str) in \
                    defender_dynamics_model.norm_num_new_warning_alerts:
                num_new_warning_alerts = \
                    defender_dynamics_model.norm_num_new_warning_alerts[(attacker_action_id, logged_in_ips_str)].rvs()

            num_new_alerts = num_new_severe_alerts + num_new_warning_alerts
            num_new_failed_login_attempts = 0
            for k,v in defender_dynamics_model.machines_dynamics_model.items():
                ip = k
                m_dynamics = v
                if (attacker_action_id, logged_in_ips_str) in m_dynamics.norm_num_new_failed_login_attempts:
                    num_new_failed_login_attempts = m_dynamics.norm_num_new_failed_login_attempts[
                        (attacker_action_id, logged_in_ips_str)].rvs()
            self.num_alerts = self.num_alerts + num_new_alerts
            self.num_failed_logins = self.num_failed_logins + num_new_failed_login_attempts
            return done



    def probability_of_intrusion(self, t: int):
        ttc = DefenderDynamics.ttc(self.num_alerts, self.num_failed_logins, constants.DP.MAX_ALERTS)
        print("ttc:{}".format(ttc))
        hp = DefenderDynamics.hack_prob(ttc)
        return hp