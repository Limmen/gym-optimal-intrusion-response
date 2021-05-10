from gym_optimal_intrusion_response.agents.manual.manual_defender_agent import ManualDefenderAgent
import gym

def manual_control(env_name : str):
    env = gym.make(env_name)
    ManualDefenderAgent(env=env)

def test_all():
    manual_control("optimal-intrusion-response-v2")

if __name__ == '__main__':
    test_all()