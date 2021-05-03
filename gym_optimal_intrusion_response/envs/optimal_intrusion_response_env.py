from typing import Tuple
import gym
import numpy as np
from abc import ABC
from gym_optimal_intrusion_response.dao.env_config import EnvConfig
from gym_optimal_intrusion_response.dao.env_state import EnvState
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
            attack_action_id = self.env_config.attacker_static_opponent.action(env=self)

        attack_action_id = int(attack_action_id)
        defense_action_id = int(defense_action_id)

        attacker_reward, defender_reward, done, defender_info = self.step_defender(defense_action_id)
        defender_info["flags"] = 0

        info = {}
        if not done:
            attacker_reward, defender_reward_2, done, info = self.step_attacker(attack_action_id)
            defender_reward = defender_reward + defender_reward_2

        # Merge infos
        if info is None:
            info = defender_info
        else:
            if defender_info is not None:
                for k, v in defender_info.items():
                    if k not in info:
                        info[k] = v

        defender_obs = self.env_state.get_defender_observation()
        attacker_obs = self.env_state.get_attacker_observation()
        self.time_step += 1

        return (attacker_obs, defender_obs), (attacker_reward, defender_reward), done, info

    def step_defender(self, defender_action_id : int) -> Tuple[int, int, bool, dict]:
        s, attacker_reward, defender_reward, done = TransitionOperator.transition_defender(
            defender_action_id=defender_action_id, env_state=self.env_state, env_config=self.env_config)
        self.env_state = s
        return attacker_reward, defender_reward, done, {}

    def step_attacker(self, attacker_action_id : int) -> Tuple[int, int, bool, dict]:
        s, attacker_reward, defender_reward, done = TransitionOperator.transition_attacker(
            attacker_action_id=attacker_action_id, env_state=self.env_state, env_config=self.env_config)
        self.env_state = s
        return attacker_reward, defender_reward, done, {}

    def is_attack_action_legal(self, a_id: int) -> bool:
        return True

    def is_defense_action_legal(self, d_id : int) -> bool:
        return True

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.env_state.reset()
        defender_obs = self.env_state.get_defender_observation()
        attacker_obs = self.env_state.get_attacker_observation()
        return attacker_obs, defender_obs


