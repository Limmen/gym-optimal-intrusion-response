"""
Starts a defender agent that can be controlled manually
"""

from gym_optimal_intrusion_response.agents.manual.manual_defender_agent import ManualDefenderAgent
import gym


def manual_control(env_name : str) -> None:
    """
    Starts the manual control process

    :param env_name: the name of the environment
    :return: None
    """
    env = gym.make(env_name)
    ManualDefenderAgent(env=env)


def start() -> None:
    """
    Starts the manual control process
    :return: None
    """
    manual_control("optimal-intrusion-response-v3")


# Script entrypoint
if __name__ == '__main__':
    start()