from typing import Tuple
from gym_optimal_intrusion_response.dao.game.env_state import EnvState
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
import gym_optimal_intrusion_response.constants.constants as constants

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
        TransitionOperator.update_defender_state(env_state)
        attacker_reward, defender_reward = TransitionOperator.attacker_reward_fun(node, env_config)
        done = (node.target_component and node.compromised)
        s = env_state
        return s, attacker_reward, defender_reward, done

    @staticmethod
    def transition_defender(defender_action_id: int, env_state: EnvState, env_config: EnvConfig) \
            -> Tuple[EnvState, int, int, bool]:
        s = env_state
        if defender_action_id == constants.ACTIONS.STOPPING_ACTION:
            s.stopped = True
        attacker_reward, defender_reward = TransitionOperator.defender_reward_fun(env_state, env_config)
        done = s.stopped
        s.t+=1
        return s, attacker_reward, defender_reward, done

    @staticmethod
    def attacker_reward_fun(node, env_config : EnvConfig) -> Tuple[int, int]:
        if node.target_component and node.compromised:
            return env_config.attacker_target_compromised_reward, env_config.defender_target_compromised_reward
        else:
            return 0,0

    @staticmethod
    def update_defender_state(env_state: EnvState):
        intrusion_in_progress = any(list(map(lambda x: x.compromised, env_state.nodes)))
        env_state.defender_observation_state.update_state(env_state.t, intrusion_in_progress=intrusion_in_progress,
                                                          dp_setup=env_state.dp_setup)

    @staticmethod
    def defender_reward_fun(env_state: EnvState, env_config: EnvConfig) -> Tuple[int, int]:
        intrusion_in_progress = any(list(map(lambda x: x.compromised, env_state.nodes)))
        if intrusion_in_progress and env_state.stopped:
            env_state.caught = True
            return env_config.attacker_intrusion_prevention_reward, env_config.defender_intrusion_prevention_reward
        elif intrusion_in_progress and not env_state.stopped:
            return 0,0
        elif not intrusion_in_progress and env_state.stopped:
            return env_config.attacker_early_stopping_reward, env_config.defender_early_stopping_reward
        elif not env_state.stopped and not intrusion_in_progress:
            return 0,0