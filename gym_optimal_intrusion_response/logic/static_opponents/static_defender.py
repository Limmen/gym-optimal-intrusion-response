from abc import ABC, abstractmethod
import numpy as np


class StaticDefender(ABC):

    def __init__(self, num_actions : int, stopping_probability : float):
        self.num_actions = num_actions
        self.actions = np.array(list(range(num_actions)))
        self.stopping_probability = stopping_probability

    @abstractmethod
    def action(self, env) -> int:
        pass