import gym
import numpy as np
from gym_optimal_intrusion_response.dao.env_config import EnvConfig
from gym_optimal_intrusion_response.dao.attacker_obs_state import AttackerObservationState
from gym_optimal_intrusion_response.dao.defender_obs_state import DefenderObservationState
from gym_optimal_intrusion_response.dao.node import Node


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


    def setup_spaces(self, env_config: EnvConfig):
        self.attacker_observation_space = gym.spaces.Box(
            low=0, high=1000, dtype=np.float32, shape=(env_config.num_nodes * env_config.num_attributes,))
        self.defender_observation_space = gym.spaces.Box(
            low=0, high=1000, dtype=np.float32, shape=(2,))
        self.attacker_action_space = gym.spaces.Discrete(env_config.num_nodes * env_config.num_attributes)
        self.defender_action_space = gym.spaces.Discrete(2)

    def get_defender_observation(self):
        return np.zeros(self.defender_observation_space.shape)

    def get_attacker_observation(self):
        return np.zeros(self.attacker_observation_space.shape)

    def reset(self):
        self.initialize_nodes()
        self.stopped = False

    def initialize_nodes(self):
        nodes = []
        for i in range(self.env_config.num_nodes):
            node = Node(initial_defense_attributes=self.env_config.initial_defense_attributes[i],
                        initial_attack_attributes=self.env_config.initial_attack_attributes[i],
                        num_attributes=self.env_config.num_attributes,
                        max_attribute_value=self.env_config.max_attribute_value)
            nodes.append(node)
        self.nodes = nodes
