from typing import List
import numpy as np
from gym_optimal_intrusion_response.dao.game.env_config import EnvConfig
from gym_optimal_intrusion_response.dao.game.node import Node

class AttackerObservationState:

    def __init__(self, env_config : EnvConfig):
        self.env_config = env_config


    def get_attacker_observation(self, nodes : List[Node]):
        obs = np.zeros((self.env_config.num_nodes, self.env_config.num_attributes + 2)).tolist()
        for i in range(len(nodes)):
            if nodes[i].recon_done:
                obs[i][0:self.env_config.num_attributes] = (np.array(nodes[i].attack_attributes)-np.array(nodes[i].defense_attributes)).tolist()
                obs[i][self.env_config.num_attributes] = 1
            else:
                obs[i][0:self.env_config.num_attributes] = nodes[i].attack_attributes
                obs[i][self.env_config.num_attributes] = 0
            obs[i][self.env_config.num_attributes+1] = int(nodes[i].compromised)
        return np.array(obs)
