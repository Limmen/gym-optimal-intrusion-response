from typing import Tuple
import gym
from gym_optimal_intrusion_response.dao.experiment.client_config import ClientConfig
from gym_optimal_intrusion_response.dao.agent.agent_type import AgentType
from gym_optimal_intrusion_response.dao.experiment.experiment_result import ExperimentResult
from gym_optimal_intrusion_response.agents.policy_gradient.ppo_baseline.ppo_baseline_agent import PPOBaselineAgent
from gym_optimal_intrusion_response.dao.experiment.runner_mode import RunnerMode
from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode


class Runner:
    """
    Utility class for running experiments
    """

    @staticmethod
    def run(config: ClientConfig) -> Tuple[ExperimentResult, ExperimentResult]:
        """
        Runs an experiment with the given configuration

        :param config: the client configuration
        :return: the experiment results
        """
        if config.mode == RunnerMode.TRAIN_ATTACKER.value or config.mode == RunnerMode.TRAIN_DEFENDER.value \
                or config.mode == RunnerMode.SELF_PLAY.value:
            return Runner.train(config)
        else:
            raise AssertionError("Runner mode not recognized: {}".format(config.mode))

    @staticmethod
    def train(config: ClientConfig) -> Tuple[ExperimentResult, ExperimentResult]:
        """
        Starts a training process with the given configuration

        :param config: the client configuration
        :return: the training results
        """
        env = Runner.regular_env_creation(config=config)
        if config.agent_type == AgentType.PPO_BASELINE.value:
            if config.defender_agent_config is not None:
                config.defender_agent_config.random_seed = config.random_seed
            if config.attacker_agent_config is not None:
                config.attacker_agent_config.random_seed = config.random_seed
            agent = PPOBaselineAgent(env,
                                     attacker_agent_config=config.defender_agent_config,
                                     defender_agent_config=config.defender_agent_config,
                                     train_mode=TrainMode(config.train_mode))
        else:
            raise AssertionError("Train agent type not recognized: {}".format(config.agent_type))
        agent.train()
        train_result = agent.train_result
        eval_result = agent.eval_result
        env.cleanup()
        env.close()
        return train_result, eval_result

    @staticmethod
    def regular_env_creation(config: ClientConfig):
        """
        Creates the environment

        :param config: the client configuration
        :return: the created env
        """
        env = gym.make(config.env_name, traces_dir=config.traces_dir, traces_filename=config.traces_filename)
        return env