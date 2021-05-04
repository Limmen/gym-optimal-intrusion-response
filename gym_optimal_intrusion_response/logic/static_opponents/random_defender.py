import numpy as np
from gym_optimal_intrusion_response.logic.static_opponents.static_defender import StaticDefender


class RandomDefender(StaticDefender):

    def __init__(self, num_actions : int, stopping_probability : float = 0.5):
        super().__init__(num_actions=num_actions, stopping_probability=stopping_probability)

    def action(self, env) -> int:
        if np.random.rand() < self.stopping_probability:
            action = 1
        else:
            action = 0
        return action

