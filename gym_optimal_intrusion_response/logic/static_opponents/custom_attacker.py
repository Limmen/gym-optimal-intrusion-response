from typing import List
from abc import ABC, abstractmethod
import numpy as np
from gym_optimal_intrusion_response.logic.static_opponents.static_attacker import StaticAttacker

class CustomAttacker(StaticAttacker):

    def __init__(self, num_actions : int, strategy : List[int]):
        super().__init__(num_actions=num_actions)
        self.strategy = strategy

    def action(self, env, t) -> int:
        return self.strategy[t]