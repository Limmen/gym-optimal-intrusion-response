import numpy as np
from gym_optimal_intrusion_response.envs.optimal_intrusion_response_env import OptimalIntrusionResponseEnv
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
from gym_optimal_intrusion_response.logic.static_opponents.random_attacker import RandomAttacker
from gym_optimal_intrusion_response.logic.static_opponents.random_defender import RandomDefender


class OptimalIntrusionResponseEnvV1(OptimalIntrusionResponseEnv):

    def __init__(self):
        num_nodes = 4
        num_attributes = 4
        random_attacker = RandomAttacker(num_actions=(num_nodes*num_attributes))
        random_defender = RandomDefender(num_actions=2, stopping_probability=0.005)
        initial_defense_attributes = [
            [3, 4, 0, 1],
            [7, 8, 5, 2],
            [4, 1, 6, 6],
            [5, 5, 5, 1]
        ]
        adjacency_matrix = [
            [0,1,0,0],
            [1,0,1,0],
            [0, 1, 0, 1],
            [0, 0, 0, 1]
        ]

        initial_reachable = [0,1]

        env_config = EnvConfig(attacker_static_opponent=random_attacker,
                               defender_static_opponent=random_defender,
                               adjacency_matrix=adjacency_matrix,
                               initial_reachable=initial_reachable,
                               num_nodes = num_nodes,
                               num_attributes=num_attributes,
                               initial_attack_attributes=np.zeros((num_nodes, num_attributes)),
                               initial_defense_attributes=initial_defense_attributes,
                               max_attribute_value=10,
                               recon_attribute = 4,
                               attacker_target_compromised_reward=1,
                               defender_target_compromised_reward = -1,
                               defender_early_stopping_reward = -1,
                               attacker_early_stopping_reward=0,
                               defender_intrusion_prevented_reward=1,
                               attacker_intrusion_prevented_reward=-1,
                               target_id=3
                               )
        super().__init__(env_config=env_config)