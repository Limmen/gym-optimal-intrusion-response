from gym_optimal_intrusion_response.envs.derived_envs.optimal_intrusion_response_env_v1 import OptimalIntrusionResponseEnvV1
import gym
import time
import numpy as np
import sys

def test_env(env_name : str, num_steps : int):
    env = gym.make(env_name)
    env.reset()

    num_attacker_actions = env.env_config.attacker_num_actions
    num_defender_actions = env.env_config.defender_num_actions
    attacker_actions = np.array(list(range(num_attacker_actions)))
    defender_actions = np.array(list(range(num_defender_actions)))
    print("num attacker actions:{}, num defender actions:{}".format(num_attacker_actions, num_defender_actions))
    for i in range(num_steps):
        legal_attacker_actions = list(filter(lambda x: env.is_attack_action_legal(x), attacker_actions))
        legal_defender_actions = list(filter(lambda x: env.is_defense_action_legal(x), defender_actions))
        attacker_action = np.random.choice(legal_attacker_actions)
        defender_action = np.random.choice(legal_defender_actions)
        action = (attacker_action, defender_action)
        obs, reward, done, info = env.step(action)
        # print("step:{}, reward:{}, action:{}".format(i, reward, action))
        if done:
            print("done: {}".format(reward))
            env.reset()
            #sys.exit()
        #time.sleep(0.001)
    env.reset()
    env.close()

def test_all():
    test_env("optimal-intrusion-response-v1", num_steps=1000000000)
    #test_env("pycr-ctf-level-1-sim-base-v1", num_steps=1000000000)

if __name__ == '__main__':
    test_all()