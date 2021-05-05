import numpy as np
from gym_optimal_intrusion_response.logic.static_opponents.static_attacker import StaticAttacker
from gym_optimal_intrusion_response.logic.static_opponents.static_defender import StaticDefender


class EnvConfig:

    def __init__(self, attacker_static_opponent : StaticAttacker,
                 defender_static_opponent: StaticDefender,
                 adjacency_matrix,
                 initial_reachable,
                 num_nodes: int = 4,
                 num_attributes : int = 4,
                 initial_attack_attributes = None, initial_defense_attributes = None,
                 max_attribute_value : int = 100, recon_attribute : int = 4,
                 attacker_target_compromised_reward : int = 1,
                 defender_target_compromised_reward: int = -1,
                 defender_early_stopping_reward : int = -1,
                 attacker_early_stopping_reward: int = -1,
                 defender_intrusion_prevented_reward : int = 1,
                 attacker_intrusion_prevented_reward: int = 1,
                 target_id : int = 4,
                 use_state_limits : bool = True
                 ):
        self.num_nodes = num_nodes
        self.num_attributes = num_attributes
        self.attacker_static_opponent = attacker_static_opponent
        self.defender_static_opponent = defender_static_opponent
        self.initial_attack_attributes = initial_attack_attributes
        self.initial_defense_attributes = initial_defense_attributes
        if initial_attack_attributes is None:
            self.initial_attack_attributes = np.zeros(self.num_nodes, self.num_attributes)
        if initial_defense_attributes is None:
            self.initial_defense_attributes = np.zeros(self.num_nodes, self.num_attributes)
        self.max_attribute_value = max_attribute_value
        self.recon_attribute = recon_attribute
        self.attacker_target_compromised_reward : int = attacker_target_compromised_reward
        self.defender_target_compromised_reward : int = defender_target_compromised_reward
        self.attacker_early_stopping_reward = attacker_early_stopping_reward
        self.defender_early_stopping_reward = defender_early_stopping_reward
        self.defender_intrusion_prevention_reward = defender_intrusion_prevented_reward
        self.attacker_intrusion_prevention_reward = attacker_intrusion_prevented_reward
        self.attacker_num_actions = self.num_nodes*self.num_attributes
        self.defender_num_actions = 2
        self.target_id = target_id
        self.adjacency_matrix = adjacency_matrix
        self.initial_reachable = initial_reachable
        self.use_state_limits = use_state_limits

