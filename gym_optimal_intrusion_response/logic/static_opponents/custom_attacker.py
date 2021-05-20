from typing import List
from abc import ABC, abstractmethod
import numpy as np
from gym_optimal_intrusion_response.logic.static_opponents.static_attacker import StaticAttacker

class CustomAttacker(StaticAttacker):

    def __init__(self, num_actions : int, strategy : List[int], continue_prob: float = 0.8):
        super().__init__(num_actions=num_actions)
        self.strategy = strategy
        self.continue_prob = continue_prob
        self.t = 0
        self.startup_phase = True

    def action(self, env, t=None) -> int:
        if t == 0:
            self.t = 0
            self.startup_phase = True
        print("t")
        if t < 7:
            print("372")
            return 372, self.t
        else:
            self.startup_phase = False
            self.t += 1
            return self.strategy[self.t - 1], self.t
        # if self.startup_phase and np.random.rand() < self.continue_prob:
        #     return 372, 0
        # else:
        #     self.t += 1
        #     self.startup_phase = False
        #     return self.strategy[self.t-1], self.t

    def reset(self):
        self.t = 0
        self.startup_phase = True