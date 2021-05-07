import numpy as np
from gym_optimal_intrusion_response.logic.static_opponents.static_attacker import StaticAttacker


class RandomAttacker(StaticAttacker):

    def __init__(self, num_actions : int):
        super().__init__(num_actions=num_actions)

    def action(self, env) -> int:
        legal_actions = list(filter(lambda x: env.is_attack_action_legal(x, env.env_config, env.env_state), self.actions))
        if len(legal_actions) == 0:
            print("no legal actions")
            print("stopped:{}, caught:{}".format(env.env_state.stopped, env.env_state.caught))
            print("nodes: {}".format(str(env.env_state)))
        action = np.random.choice(legal_actions)
        return action

