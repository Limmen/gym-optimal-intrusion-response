"""
Abstract Train Agent
"""
import numpy as np
import logging
import random
import torch
from abc import ABC, abstractmethod
from gym_optimal_intrusion_response.agents.config.agent_config import AgentConfig
from gym_optimal_intrusion_response.dao.experiment.experiment_result import ExperimentResult
from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode

class TrainAgent(ABC):
    """
    Abstract Train Agent
    """
    def __init__(self, env, attacker_config: AgentConfig,
                 defender_config: AgentConfig,
                 train_mode : TrainMode = TrainMode.TRAIN_ATTACKER):
        self.env = env
        self.attacker_config = attacker_config
        self.defender_config = defender_config
        self.train_result = ExperimentResult()
        self.eval_result = ExperimentResult()
        self.train_mode=train_mode
        if self.attacker_config is None:
            self.attacker_config = self.defender_config
        if self.attacker_config.logger is None:
            self.attacker_config.logger = logging.getLogger('Train Agent - Attacker')

        if self.defender_config is None:
            self.defender_config = self.attacker_config
        if self.defender_config.logger is None:
            self.defender_config.logger = logging.getLogger('Train Agent - Defender')

        random.seed(self.attacker_config.random_seed)
        np.random.seed(self.attacker_config.random_seed)
        torch.manual_seed(self.attacker_config.random_seed)

    @abstractmethod
    def train(self) -> ExperimentResult:
        pass

    @abstractmethod
    def eval(self, log=True) -> ExperimentResult:
        pass