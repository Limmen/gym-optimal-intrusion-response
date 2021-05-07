"""
Manual defender agent
"""
from gym_optimal_intrusion_response.envs.optimal_intrusion_response_env import OptimalIntrusionResponseEnv
import numpy as np

class ManualDefenderAgent:
    """
    Class representing a manual defender agent
    """

    def __init__(self, env: OptimalIntrusionResponseEnv):
        self.env = env
        num_actions = env.defender_action_space.n
        actions = np.array(list(range(num_actions)))
        cumulative_reward = 0
        latest_obs = None
        latest_rew = None
        latest_obs = env.reset()
        history = []
        for i in range(env.env_config.num_nodes*env.env_config.num_attributes):
            print("action:{}, attribute:{}, node:{}".format(i, i % env.env_config.num_attributes, i // env.env_config.num_attributes))
        while True:
            raw_input = input(">")
            raw_input = raw_input.strip()
            legal_actions = list(
                filter(lambda x: env.is_defense_action_legal(x), actions))
            if raw_input == "help":
                print("Enter an action id to execute the action, "
                      "press R to reset, press S to print the state, press A to print the actions, "
                      "press L to print the legal actions, press x to select a random legal action,"
                      "press H to print the history of actions")
            elif raw_input == "R":
                latest_obs = env.reset()
                cumulative_reward = 0
            elif raw_input == "S":
                print(str(env.env_state.attacker_obs_state))
            elif raw_input == "L":
                print(legal_actions)
            elif raw_input == "x":
                a = np.random.choice(legal_actions)
                _, _, done, _ = self.env.step(a)
                history.append(a)
                if done:
                    print("done:{}".format(done))
            elif raw_input == "H":
                print(history)
            elif raw_input == "O":
                print(latest_obs)
                print(latest_obs[0].shape)
                print(latest_obs[1].shape)
            elif raw_input == "U":
                print(latest_rew)
            elif raw_input == "P":
                print(cumulative_reward)
            else:
                actions_str = raw_input.split(",")
                digits_only = any(any(char.isdigit() for char in x) for x in actions_str)
                attacker_action = None
                if not digits_only:
                    print("Invalid action. Actions must be integers.")
                else:
                    actions = list(map(lambda x: int(x), actions_str))
                    for a in actions:
                        if env.is_defense_action_legal(a):
                            action = (attacker_action, a)
                            latest_obs, latest_rew, done, _ = self.env.step(action)
                            attacker_rew, defender_rew = latest_rew
                            cumulative_reward += defender_rew
                            history.append(a)
                            if done:
                                target_compromised = list(filter(lambda x: x.target_component, env.env_state.nodes))[0].compromised
                                print("done:{}, attacker_caught:{}, stopped:{}, target_compromised:{} rew:{}".format(
                                    done, env.env_state.caught,
                                    env.env_state.stopped,
                                    target_compromised, latest_rew
                                ))
                        else:
                            print("action:{} is illegal".format(a))
