from abc import ABC, abstractmethod
import numpy as np


class StaticDefender(ABC):

    def __init__(self, num_actions : int):
        self.num_actions = num_actions
        self.actions = np.array(list(range(num_actions)))

    @abstractmethod
    def action(self, env) -> int:
        pass