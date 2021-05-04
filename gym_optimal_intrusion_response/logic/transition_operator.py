from typing import Tuple
from gym_optimal_intrusion_response.dao.env.env_state import EnvState
from gym_optimal_intrusion_response.dao.env.env_config import EnvConfig


class TransitionOperator:

    @staticmethod
    def transition_attacker(attacker_action_id : int, env_state : EnvState, env_config : EnvConfig) \
            -> Tuple[EnvState, int, int, bool]:
        node_id = TransitionOperator.get_attacked_node(attacker_action_id, env_config)
        attribute_id = TransitionOperator.get_attacked_attribute(attacker_action_id, env_config)
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
            -> Tuple[EnvState, int, int, bool]:
        s = env_state
        if defender_action_id == 1:
            s.stopped = True
        attacker_reward, defender_reward = TransitionOperator.defender_reward_fun(env_state, env_config)
        done = s.stopped
        return s, attacker_reward, defender_reward, done

    @staticmethod
    def get_attacked_node(attacker_action_id : int, env_config: EnvConfig) -> int:
        return attacker_action_id % env_config.num_attributes

    @staticmethod
    def get_attacked_attribute(attacker_action_id : int, env_config: EnvConfig):
        return attacker_action_id//env_config.num_nodes


    @staticmethod
    def attacker_reward_fun(node, env_config : EnvConfig) -> Tuple[int, int]:
        if node.target_component:
            return env_config.attacker_target_compromised_reward, env_config.defender_target_compromised_reward
        else:
            return 0,0

    @staticmethod
    def defender_reward_fun(env_state: EnvState, env_config: EnvConfig) -> Tuple[int, int]:
        intrusion_in_progress = any(list(map(lambda x: x.compromised, env_state.nodes)))
        if intrusion_in_progress and env_state.stopped:
            return env_config.attacker_intrusion_prevention_reward, env_config.defender_intrusion_prevention_reward
        elif intrusion_in_progress and not env_state.stopped:
            return 0,0
        elif not intrusion_in_progress and env_state.stopped:
            return env_config.attacker_early_stopping_reward, env_config.defender_early_stopping_reward
        elif not env_state.stopped and not intrusion_in_progress:
            return 0,0