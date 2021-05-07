import gym
import numpy as np
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
from gym_optimal_intrusion_response.dao.obs.attacker_obs_state import AttackerObservationState
from gym_optimal_intrusion_response.dao.obs.defender_obs_state import DefenderObservationState
from gym_optimal_intrusion_response.dao.game.node import Node
from gym_optimal_intrusion_response.logic.defender_dynamics.dp import DP
from gym_optimal_intrusion_response.dao.dp.dp_setup import DPSetup


class EnvState:

    def __init__(self, env_config : EnvConfig):
        self.env_config = env_config
        self.attacker_observation_space = None
        self.defender_observation_space = None
        self.attacker_action_space = None
        self.defender_action_space = None
        self.setup_spaces(self.env_config)
        self.attacker_observation_state = AttackerObservationState(env_config=env_config)
        self.defender_observation_state = DefenderObservationState(env_config=env_config)
        self.nodes = []
        self.initialize_nodes()
        self.stopped = False
        self.caught = False
        self.t=0
        self.dp_setup = None
        if self.env_config.dp:
            self.dp_setup = self.setup_dp()

    def setup_spaces(self, env_config: EnvConfig):
        self.attacker_observation_space = gym.spaces.Box(
            low=0, high=1000, dtype=np.float32, shape=(env_config.num_nodes * (env_config.num_attributes+2),))
        if self.env_config.dp:
            self.defender_observation_space = gym.spaces.Box(
                low=0, high=1000, dtype=np.float32, shape=(2,))
        else:
            self.defender_observation_space = gym.spaces.Box(
                low=0, high=1000, dtype=np.float32, shape=(3,))
        self.attacker_action_space = gym.spaces.Discrete(env_config.num_nodes * (env_config.num_attributes+1))
        self.defender_action_space = gym.spaces.Discrete(2)

    def get_defender_observation(self):
        return self.defender_observation_state.get_defender_observation(t=self.t, dp_setup=self.dp_setup)

    def get_attacker_observation(self):
        return self.attacker_observation_state.get_attacker_observation(self.nodes)

    def reset(self):
        self.initialize_nodes()
        self.stopped = False
        self.caught = False
        self.defender_observation_state.reset()
        self.t=0

    def initialize_nodes(self):
        nodes = []
        for i in range(self.env_config.num_nodes):
            target = (i == self.env_config.target_id)
            node = Node(initial_defense_attributes=self.env_config.initial_defense_attributes[i],
                        initial_attack_attributes=self.env_config.initial_attack_attributes[i],
                        num_attributes=self.env_config.num_attributes,
                        max_attribute_value=self.env_config.max_attribute_value,
                        target_component=target)
            nodes.append(node)
        self.nodes = nodes

    def attacker_reachable(self, node_id):
        if node_id >= self.env_config.num_nodes:
            return False
        if node_id in self.env_config.initial_reachable:
            return True
        for i in range(len(self.nodes)):
            if self.nodes[i].compromised:
                if self.env_config.adjacency_matrix[i][node_id] == 1:
                    return True
        return False

    def setup_dp(self) -> DPSetup:
        print("Setup DP")
        dp_setup = None
        if self.env_config.dp_load:
            HP = DP.load_HP_table()
            R = DP.load_R_table()
            T = DP.load_transition_kernel()
            next_state_lookahead = DP.load_next_states_lookahead_table()
            state_to_id = DP.load_state_to_id()
            id_to_state = DP.load_id_to_state()
            dp_setup = DPSetup(HP=HP, R=R,T=T,next_state_lookahead=next_state_lookahead,
                               state_to_id=state_to_id, id_to_state=id_to_state)
        else:
            state_to_id, id_to_state = DP.state_to_id_dict()
            ttc_to_alerts_logins, alerts_logins_to_ttc = DP.ttc_to_alerts_table()
            HP, R = DP.hp_and_ttc_and_r(DP.num_states(), DP.num_actions(), id_to_state)
            T = DP.transition_kernel(id_to_state, DP.num_states(), DP.num_actions(), HP, ttc_to_alerts_logins,
                                     alerts_logins_to_ttc, state_to_id)
            next_state_lookahead = DP.next_states_lookahead_table(DP.num_states(), DP.num_actions(), T, id_to_state)
            dp_setup = DPSetup(HP=HP, R=R, T=T, next_state_lookahead=next_state_lookahead,
                               state_to_id=state_to_id, id_to_state=id_to_state)
        return dp_setup

    @staticmethod
    def get_attacked_node(attacker_action_id: int, env_config: EnvConfig) -> int:
        return attacker_action_id // (env_config.num_attributes + 1)

    @staticmethod
    def get_attacked_attribute(attacker_action_id: int, env_config: EnvConfig):
        return attacker_action_id % (env_config.num_attributes + 1)


    def __str__(self):
        return ",".join(list(map(lambda x: str(x), self.nodes)))

