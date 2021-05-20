import numpy as np
from gym_optimal_intrusion_response.envs.optimal_intrusion_response_env import OptimalIntrusionResponseEnv
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
from gym_optimal_intrusion_response.logic.static_opponents.random_defender import RandomDefender
from gym_optimal_intrusion_response.logic.static_opponents.custom_attacker import CustomAttacker


class OptimalIntrusionResponseEnvV2(OptimalIntrusionResponseEnv):
    """
    Version 2 of the optimal intrusion response environment.
    """

    def __init__(self):
        num_nodes = 4
        num_attributes = 4
        # random_attacker = RandomAttacker(num_actions=(num_nodes*num_attributes))

        custom_attacker = CustomAttacker(
            num_actions=(num_nodes*num_attributes),
            strategy=[99, 33, 104, 105, 106, 1, 104, 105,
                      106, 70, 104, 105, 107, 99, 165, 104, 105, 106,
                      200, 104, 105, 106, 58, 104, 105, 331,
                      105, 99, 266, 104, 105, 106, 99, 113, 104, 105],
            continue_prob=0.8
        )

        attack_idx_to_id = {}
        attack_idx_to_id[372] = 85
        attack_idx_to_id[99] = 19
        attack_idx_to_id[100] = 20
        attack_idx_to_id[33] = 11
        attack_idx_to_id[104] = 38
        attack_idx_to_id[105] = 39
        attack_idx_to_id[106] = 51
        attack_idx_to_id[1] = 10
        attack_idx_to_id[70] = 12
        attack_idx_to_id[107] = 52
        attack_idx_to_id[165] = 54
        attack_idx_to_id[200] = 55
        attack_idx_to_id[58] = 11
        attack_idx_to_id[331] = 59
        attack_idx_to_id[266] = 57
        attack_idx_to_id[113] = 53

        action_to_state = {}
        action_to_state[(99, 1)] = ("172.18.9.191", False, True)
        action_to_state[(100, 1)] = ("172.18.9.191", False, True)
        action_to_state[(33, 2)] = ("172.18.9.191", False, True)
        action_to_state[(104, 3)] = ("172.18.9.191_172.18.9.2_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(105, 4)] = ("172.18.9.191_172.18.9.2_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(106, 5)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(1, 6)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(104, 7)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=0_root=1_172.18.9.3_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(105, 8)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=0_root=1_172.18.9.3_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(106, 9)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=0_root=1_172.18.9.3_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(70, 10)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=0_root=1_172.18.9.3_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(104, 11)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=0_root=1_172.18.9.3_tools=1_backdoor=0_root=1_172.18.9.79_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(105, 12)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=0_root=1_172.18.9.3_tools=1_backdoor=0_root=1_172.18.9.79_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(107, 13)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.79_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(99, 14)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.79_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(99, 14)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.79_tools=0_backdoor=0_root=1",False, True)
        action_to_state[(165, 15)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.79_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(104, 16)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=0_backdoor=1_root=1_172.18.9.79_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(105, 17)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=0_backdoor=1_root=1_172.18.9.79_tools=0_backdoor=0_root=1", False, True)
        action_to_state[(106, 18)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(200, 19)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(104, 20)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.74_tools=0_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(105, 21)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.74_tools=0_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(106, 22)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(58, 23)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(104, 24)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=0_backdoor=0_root=0_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(105, 25)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=0_backdoor=0_root=0_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(331, 26)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=0_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(105, 27)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=0_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(99, 28)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=0_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(99, 28)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=0_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(266, 29)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=0_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(104, 30)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=0_backdoor=1_root=1_172.18.9.62_tools=0_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(105, 31)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=0_backdoor=1_root=1_172.18.9.62_tools=0_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(106, 32)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=1_backdoor=1_root=1_172.18.9.62_tools=1_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(99, 33)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=1_backdoor=1_root=1_172.18.9.62_tools=1_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(99, 33)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=1_backdoor=1_root=1_172.18.9.62_tools=1_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1",False, True)
        action_to_state[(113, 34)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=1_backdoor=1_root=1_172.18.9.62_tools=1_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1", False, True)
        action_to_state[(104, 35)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=1_backdoor=1_root=1_172.18.9.62_tools=1_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1_172.18.9.7_tools=0_backdoor=1_root=1", False, True)
        action_to_state[(105, 36)] = ("172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=1_backdoor=1_root=1_172.18.9.62_tools=1_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1_172.18.9.7_tools=0_backdoor=1_root=1", True, True)
        #172.18.9.191_172.18.9.2_tools=1_backdoor=1_root=1_172.18.9.3_tools=1_backdoor=1_root=1_172.18.9.54_tools=1_backdoor=1_root=1_172.18.9.61_tools=1_backdoor=1_root=1_172.18.9.62_tools=1_backdoor=1_root=1_172.18.9.74_tools=1_backdoor=1_root=1_172.18.9.79_tools=1_backdoor=0_root=1
        for i in range(1000):
            action_to_state[(372, i)] = ("172.18.9.191", False, False)


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

        env_config = EnvConfig(attacker_static_opponent=custom_attacker,
                               defender_static_opponent=random_defender,
                               adjacency_matrix=adjacency_matrix,
                               initial_reachable=initial_reachable,
                               num_nodes = num_nodes,
                               num_attributes=num_attributes,
                               initial_attack_attributes=np.zeros((num_nodes, num_attributes)),
                               initial_defense_attributes=initial_defense_attributes,
                               max_attribute_value=10,
                               recon_attribute = 4,
                               attacker_target_compromised_reward=100,
                               defender_target_compromised_reward = -100,
                               defender_early_stopping_reward = -100,
                               attacker_early_stopping_reward=0,
                               defender_intrusion_prevented_reward=100,
                               attacker_intrusion_prevented_reward=-100,
                               defender_continue_reward=10,
                               attacker_continue_reward=0,
                               target_id=3,
                               dp=False,
                               dp_load=False,
                               traces=True,
                               action_to_state=action_to_state,
                               attack_idx_to_id=attack_idx_to_id,
                               save_dynamics_model_dir = "/home/kim/workspace/gym-optimal-intrusion-response/traces/",
                               dynamics_model_name = "traces.json"
                               )
        super().__init__(env_config=env_config)