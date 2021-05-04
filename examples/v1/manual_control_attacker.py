from gym_optimal_intrusion_response.agents.manual.manual_attacker_agent import ManualAttackerAgent
import gym

def manual_control(env_name : str):
    env = gym.make(env_name)
    ManualAttackerAgent(env=env)

def test_all():
    manual_control("optimal-intrusion-response-v1")

if __name__ == '__main__':
    test_all()