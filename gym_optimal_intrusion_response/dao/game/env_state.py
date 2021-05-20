import gym
import numpy as np
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
from gym_optimal_intrusion_response.dao.obs.attacker_obs_state import AttackerObservationState
from gym_optimal_intrusion_response.dao.obs.defender_obs_state import DefenderObservationState
from gym_optimal_intrusion_response.dao.game.node import Node
from gym_optimal_intrusion_response.logic.defender_dynamics.dp import DP
from gym_optimal_intrusion_response.dao.dp.dp_setup import DPSetup
from gym_pycr_ctf.dao.defender_dynamics.defender_dynamics_model import DefenderDynamicsModel


class EnvState:
    """
    DTO with the environment state
    """

    def __init__(self, env_config : EnvConfig):
        """
        Class constructor, initializes the state

        :param env_config: the environment configuration
        """
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
        self.intrusion_in_progress = False
        self.target_compromised = False
        self.intrusion_t = 0
        self.t=0
        self.dp_setup = None
        self.dynamics_model = None
        if self.env_config.dp:
            self.dp_setup = self.setup_dp()
        elif self.env_config.traces:
            self.dynamics_model = self.setup_dynamics_model()

    def setup_spaces(self, env_config: EnvConfig) -> None:
        """
        Setup of the action and observation spaces

        :param env_config: the environment configuration
        :return: None
        """
        self.attacker_observation_space = gym.spaces.Box(
            low=0, high=1000, dtype=np.float32, shape=(env_config.num_nodes * (env_config.num_attributes+2),))
        if self.env_config.dp:
            self.defender_observation_space = gym.spaces.Box(
                low=0, high=1000, dtype=np.float32, shape=(2,))
        elif not self.env_config.dp and self.env_config.traces:
            self.defender_observation_space = gym.spaces.Box(
                low=0, high=1000, dtype=np.float32, shape=(4,))
            # self.defender_observation_space = gym.spaces.Box(
            #     low=0, high=1000, dtype=np.float32, shape=(5,))
        else:
            self.defender_observation_space = gym.spaces.Box(
                low=0, high=1000, dtype=np.float32, shape=(3,))
        self.attacker_action_space = gym.spaces.Discrete(env_config.num_nodes * (env_config.num_attributes+1))
        self.defender_action_space = gym.spaces.Discrete(2)

    def get_defender_observation(self) -> np.ndarray:
        """
        :return: the latest defender observation
        """
        return self.defender_observation_state.get_defender_observation(t=self.t, dp_setup=self.dp_setup)

    def get_attacker_observation(self) -> np.ndarray:
        """
        :return: the latest attacker observation
        """
        return self.attacker_observation_state.get_attacker_observation(self.nodes)

    def reset(self) -> None:
        """
        Resets the state

        :return: None
        """
        self.initialize_nodes()
        self.stopped = False
        self.caught = False
        self.intrusion_in_progress = False
        self.target_compromised = False
        self.defender_observation_state.reset()
        self.t=1
        self.intrusion_t = -1

    def initialize_nodes(self) -> None:
        """
        Utility function for initializing the node states

        :return: None
        """
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

    def attacker_reachable(self, node_id) -> bool:
        """
        Utility function for checking if a node id is reachable for the  attacker or not

        :param node_id: the id of the node to check
        :return: True if reachable otherwise False
        """
        if node_id >= self.env_config.num_nodes:
            return False
        if node_id in self.env_config.initial_reachable:
            return True
        for i in range(len(self.nodes)):
            if self.nodes[i].compromised:
                if self.env_config.adjacency_matrix[i][node_id] == 1:
                    return True
        return False

    def setup_dynamics_model(self) -> None:
        """
        Utility function for loading and setting up the dynamics model based on system traces

        :return: None
        """
        if self.env_config.traces:
            defender_dynamics_model = DefenderDynamicsModel()
            new_model = DefenderDynamicsModel()
            if self.env_config.save_dynamics_model_dir is not None:
                print("loading dynamics model")
                defender_dynamics_model.read_model(self.env_config.save_dynamics_model_dir,
                                                   model_name=self.env_config.dynamics_model_name)
                defender_dynamics_model.normalize()
                print("model loaded")
                return defender_dynamics_model
        return None

    def setup_dp(self) -> DPSetup:
        """
        Utility function for setting up the D.P parameters

        :return: the DPsetup DTO
        """
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
        """
        Utility function for getting the id of the node of an attack

        :param attacker_action_id: the attack id
        :param env_config: the environment config
        :return: the node id
        """
        return attacker_action_id // (env_config.num_attributes + 1)

    @staticmethod
    def get_attacked_attribute(attacker_action_id: int, env_config: EnvConfig) -> int:
        """
        Utility function for getting the attribute idx of an attack

        :param attacker_action_id: the attack id
        :param env_config: the environment config
        :return: the attribute id
        """
        return attacker_action_id % (env_config.num_attributes + 1)


    def __str__(self):
        """
        :return: a string representation of the object
        """
        return ",".join(list(map(lambda x: str(x), self.nodes)))


