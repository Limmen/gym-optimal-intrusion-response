"""
Starts an attacker agent that can be controlled manually
"""

from gym_optimal_intrusion_response.agents.manual.manual_attacker_agent import ManualAttackerAgent
import gym


def manual_control(env_name : str) -> None:
    """
    Creates the environment and the agent

    :param env_name: the name of the environment
    :return: None
    """
    env = gym.make(env_name)
    ManualAttackerAgent(env=env)


def start() -> None:
    """
    Starts the manual control process

    :return: None
    """
    manual_control("optimal-intrusion-response-v1")


# Script entrypoint
if __name__ == '__main__':
    start()