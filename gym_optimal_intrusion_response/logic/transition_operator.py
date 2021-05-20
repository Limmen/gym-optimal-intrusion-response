from typing import Tuple
import numpy as np
from gym_optimal_intrusion_response.dao.game.env_state import EnvState
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
import gym_optimal_intrusion_response.constants.constants as constants
import math

class TransitionOperator:

    @staticmethod
    def transition_attacker(attacker_action_id : int, env_state : EnvState, env_config : EnvConfig) \
            -> Tuple[EnvState, int, int, bool]:
        node_id = EnvState.get_attacked_node(attacker_action_id, env_config)
        attribute_id = EnvState.get_attacked_attribute(attacker_action_id, env_config)
        node = env_state.nodes[node_id]
        if attribute_id == env_config.recon_attribute:
            node.recon()
        else:
            node.attack(attribute_id)
        attacker_reward, defender_reward = TransitionOperator.attacker_reward_fun(node, env_config)
        done = (node.target_component and node.compromised)
        s = env_state
        return s, attacker_reward, defender_reward, done

    @staticmethod
    def transition_defender(defender_action_id: int, env_state: EnvState, env_config: EnvConfig) \
            -> Tuple[EnvState, int, int, bool, dict]:
        s = env_state
        if defender_action_id == constants.ACTIONS.STOPPING_ACTION:
            s.stopped = True
        attacker_reward, defender_reward, info = TransitionOperator.defender_reward_fun(s, env_config, defender_action_id)
        done = s.stopped
        s.t+=1
        if env_config.dp and s.t >= constants.DP.MAX_TIMESTEPS:
            done = True
            if defender_action_id != 1:
                info["successful_intrusion"]= 1
        if env_config.traces and s.t >= constants.TRACES.MAX_TIMESTEPS:
            done = True
            if defender_action_id != 1:
                info["successful_intrusion"]= 1
        return s, attacker_reward, defender_reward, done, info

    @staticmethod
    def attacker_reward_fun(node, env_config : EnvConfig) -> Tuple[int, int]:
        if node.target_component and node.compromised:
            return env_config.attacker_target_compromised_reward, env_config.defender_target_compromised_reward
        else:
            return 0,0

    @staticmethod
    def update_defender_state(env_state: EnvState, attacker_action, t) -> bool:
        intrusion_in_progress = any(list(map(lambda x: x.compromised, env_state.nodes)))
        done = env_state.defender_observation_state.update_state(t, intrusion_in_progress=intrusion_in_progress,
                                                          dp_setup=env_state.dp_setup, attacker_action=attacker_action,
                                                                 defender_dynamics_model=env_state.dynamics_model)
        return done

    @staticmethod
    def defender_reward_fun(env_state: EnvState, env_config: EnvConfig, defender_action : int) -> Tuple[int, int, dict]:
        info = {}
        if not env_state.env_config.dp and not env_state.env_config.traces:
            intrusion_in_progress = any(list(map(lambda x: x.compromised, env_state.nodes)))
            if intrusion_in_progress and env_state.stopped:
                env_state.caught = True
                info["caught_attacker"] = 1
                return env_config.attacker_intrusion_prevention_reward, env_config.defender_intrusion_prevention_reward, info
            elif intrusion_in_progress and not env_state.stopped:
                return 0,0,info
            elif not intrusion_in_progress and env_state.stopped:
                info["early_stopped"] = 1
                return env_config.attacker_early_stopping_reward, env_config.defender_early_stopping_reward, info
            elif not env_state.stopped and not intrusion_in_progress:
                return 0,0,info
        elif env_state.env_config.dp:
            info = {}
            state_id = env_state.dp_setup.state_to_id[(env_state.t, env_state.defender_observation_state.ttc)]
            r = env_state.dp_setup.R[state_id][defender_action]
            hp = env_state.dp_setup.HP[state_id]
            if defender_action == 1:
                if np.random.rand() < hp:
                    info["caught_attacker"] = 1
                else:
                    info["early_stopped"] = 1
            return 0, r, info
        elif env_config.traces:
            # print("t:{}, intrusion_t:{}".format(env_state.t, env_state.intrusion_t))
            intrusion_in_progress = env_state.intrusion_in_progress
            if intrusion_in_progress and env_state.stopped:
                env_state.caught = True
                info["caught_attacker"] = 1
                r = env_config.defender_intrusion_prevention_reward / max(1, (math.pow((env_state.t - env_state.intrusion_t), 1.05)))
                return env_config.attacker_intrusion_prevention_reward, r, info
            elif intrusion_in_progress and not env_state.stopped:
                return env_config.attacker_target_compromised_reward, \
                       env_config.defender_target_compromised_reward + env_config.defender_continue_reward, info
            elif not intrusion_in_progress and env_state.stopped:
                info["early_stopped"] = 1
                return env_config.attacker_early_stopping_reward, env_config.defender_early_stopping_reward, info
            elif not env_state.stopped and not intrusion_in_progress:
                return env_config.attacker_continue_reward, env_config.defender_continue_reward, info
