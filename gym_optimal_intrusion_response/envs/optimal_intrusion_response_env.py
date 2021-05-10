from typing import Tuple
import gym
import numpy as np
import inspect
from abc import ABC
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
from gym_optimal_intrusion_response.dao.game.env_state import EnvState
from gym_optimal_intrusion_response.logic.transition_operator import TransitionOperator

class OptimalIntrusionResponseEnv(gym.Env, ABC):
    """
    TODO
    """

    def __init__(self, env_config : EnvConfig):
        self.env_config = env_config
        self.env_state = EnvState(env_config=env_config)
        self.attacker_observation_space = self.env_state.attacker_observation_space
        self.defender_observation_space = self.env_state.defender_observation_space
        self.attacker_action_space = self.env_state.attacker_action_space
        self.defender_action_space = self.env_state.defender_action_space
        self.time_step = 0

    # -------- API ------------

    def step(self, action_id: Tuple[int, int]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[int, int], bool, dict]:

        if isinstance(action_id, int) or isinstance(action_id, np.int64):
            action_id = (action_id, None)
            print("[WARNING]: This is a multi-agent environment where the input should be "
                  "(attacker_action, defender_action)")

        attack_action_id, defense_action_id = action_id
        if attack_action_id is None:
            attack_action_id = self.env_config.attacker_static_opponent.action(env=self, t=self.env_state.t)

        if defense_action_id is None:
            defense_action_id = self.env_config.defender_static_opponent.action(env=self)

        attack_action_id = int(attack_action_id)
        defense_action_id = int(defense_action_id)

        attacker_reward, defender_reward, done, defender_info = self.step_defender(defense_action_id)

        info = {}
        info["flags"] = 0
        info["caught_attacker"] = 0
        info["early_stopped"] = 0
        info["snort_severe_baseline_reward"] = 0
        info["snort_warning_baseline_reward"] = 0
        info["snort_critical_baseline_reward"] = 0
        info["var_log_baseline_reward"] = 0
        info["successful_intrusion"] = False
        info["attacker_cost"] = 0
        info["attacker_cost_norm"] = 0
        info["attacker_alerts"] = 0
        info["attacker_alerts_norm"] = 0
        info["flags"] = 0

        if not done and not self.env_config.dp and not self.env_config.traces:
            attacker_reward, defender_reward_2, done, info = self.step_attacker(attack_action_id)
            defender_reward = defender_reward + defender_reward_2
        elif self.env_config.traces:
            logged_in_ips_str, done2, intrusion_in_progress = self.env_config.action_to_state[(attack_action_id,
                                                                                              self.env_state.t)]
            self.env_state.intrusion_in_progress = intrusion_in_progress
            if done2:
                self.env_state.target_compromised = True
                done = True
                defender_reward = self.env_config.defender_target_compromised_reward

        d3 = TransitionOperator.update_defender_state(self.env_state, attacker_action=attack_action_id)
        if not done:
            done = d3

        # Merge infos
        if info is None:
            info = defender_info
        else:
            if defender_info is not None:
                for k, v in defender_info.items():
                    info[k] = v

        defender_obs = self.env_state.get_defender_observation().flatten()
        attacker_obs = self.env_state.get_attacker_observation().flatten()
        self.time_step += 1

        return (attacker_obs, defender_obs), (attacker_reward, defender_reward), done, info

    def step_defender(self, defender_action_id : int) -> Tuple[int, int, bool, dict]:
        s, attacker_reward, defender_reward, done, info = TransitionOperator.transition_defender(
            defender_action_id=defender_action_id, env_state=self.env_state, env_config=self.env_config)
        self.env_state = s
        return attacker_reward, defender_reward, done, info

    def step_attacker(self, attacker_action_id : int) -> Tuple[int, int, bool, dict]:
        s, attacker_reward, defender_reward, done = TransitionOperator.transition_attacker(
            attacker_action_id=attacker_action_id, env_state=self.env_state, env_config=self.env_config)
        self.env_state = s
        return attacker_reward, defender_reward, done, {}

    @staticmethod
    def is_attack_action_legal(a_id: int, env_config: EnvConfig, env_state: EnvState) -> bool:
        node_id = EnvState.get_attacked_node(a_id, env_config)
        attribute_id = EnvState.get_attacked_attribute(a_id, env_config)
        if not env_state.attacker_reachable(node_id):
            return False
        if env_state.nodes[node_id].compromised:
            return False
        if attribute_id == env_config.recon_attribute and env_state.nodes[node_id].recon_done:
            return False
        if attribute_id != env_config.recon_attribute and env_state.nodes[node_id].attack_attributes[attribute_id] >= env_config.max_attribute_value:
            return False
        return True

    @staticmethod
    def is_defense_action_legal(d_id : int) -> bool:
        return True

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.env_state.reset()
        defender_obs = self.env_state.get_defender_observation().flatten()
        attacker_obs = self.env_state.get_attacker_observation().flatten()
        return attacker_obs, defender_obs


