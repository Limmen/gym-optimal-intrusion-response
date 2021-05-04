import numpy as np
from gym_optimal_intrusion_response.logic.static_opponents.static_defender import StaticDefender


class RandomDefender(StaticDefender):

    def __init__(self, num_actions : int):
        super().__init__(num_actions=num_actions)

    def action(self, env) -> int:
        legal_actions = list(filter(lambda x: env.is_defense_action_legal(x), self.actions))
        action = np.random.choice(legal_actions)
        return action

