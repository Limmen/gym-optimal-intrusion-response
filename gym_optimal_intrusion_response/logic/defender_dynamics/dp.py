from typing import Tuple
import math
from gym_optimal_intrusion_response.logic.defender_dynamics.defender_dynamics import DefenderDynamics
import json
import pickle
import gym_optimal_intrusion_response.constants.constants as constants
import numpy as np


class DP:

    @staticmethod
    def num_states() -> int:
        """
        :return: the number of states
        """
        return constants.DP.MAX_TTC * constants.DP.MAX_TIMESTEPS + 1

    @staticmethod
    def num_actions() -> int:
        """
        :return: the number of actions
        """
        return 2

    @staticmethod
    def actions() -> dict:
        """
        :return: action lookup dictp
        """
        return {"continue": 0, "stop": 1}

    @staticmethod
    def hp_and_ttc_and_r(n_states: int, n_actions: int, id_to_state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-computes the HP, TTC, and R

        :param n_states: number of states
        :param n_actions: number of actions
        :param id_to_state: lookup dict to convert between ids and states
        :return: the hack probability matrix and the reward matrix
        """
        HP = np.zeros(n_states)
        R = np.zeros((n_states, n_actions))
        for i in range(n_states):
            s1 = id_to_state[i]
            if s1 == "terminal":
                HP[i] = 0
                R[i][0] = 0
                R[i][1] = 0
            else:
                t1, x1 = s1
                hp = DefenderDynamics.hack_prob(x1, t1)
                HP[i] = hp
                for j in range(n_actions):
                    r = DP.reward_fun(i, j, id_to_state)
                    R[i][j] = r
        DP.save_numpy(HP, "hp_table.npy")
        DP.save_numpy(R, "reward_fun.npy")
        return HP, R

    @staticmethod
    def ttc_to_alerts_table() -> Tuple[dict, dict]:
        """
        Utility function for computing a lookup table between TTC and alerts

        :return: (ttc->alerts/logins,alerts/logins->ttc)
        """
        alerts_logins_to_ttc = {}
        ttc_to_alerts_logins = {}
        for i in range(constants.DP.MAX_ALERTS):
            for j in range(constants.DP.MAX_LOGINS):
                ttc = int(round(DefenderDynamics.ttc(i, j, constants.DP.MAX_ALERTS)))
                alerts_logins_to_ttc[(i, j)] = ttc
                if ttc not in ttc_to_alerts_logins:
                    ttc_to_alerts_logins[ttc] = [(i, j)]
                else:
                    ttc_to_alerts_logins[ttc] = ttc_to_alerts_logins[ttc] + [(i, j)]
        avg_ttc_to_alerts_logins = {}
        for k, v in ttc_to_alerts_logins.items():
            alerts = list(map(lambda x: x[0], v))
            logins = list(map(lambda x: x[1], v))
            avg_ttc_to_alerts_logins[k] = (np.mean(np.array(alerts)), np.mean(np.array(logins)))

        for i in range(constants.DP.MAX_TTC):
            if i not in avg_ttc_to_alerts_logins or avg_ttc_to_alerts_logins[i] == 0:
                if i - 1 in avg_ttc_to_alerts_logins:
                    avg_ttc_to_alerts_logins[i] = avg_ttc_to_alerts_logins[i - 1]
                elif i + 1 in avg_ttc_to_alerts_logins:
                    avg_ttc_to_alerts_logins[i] = avg_ttc_to_alerts_logins[i + 1]
                else:
                    print("total miss")

        DP.save_pickle(avg_ttc_to_alerts_logins, "ttc_to_alerts_logins.pkl")
        DP.save_pickle(alerts_logins_to_ttc, "logins_alerts_to_tcc.pkl")
        return avg_ttc_to_alerts_logins, alerts_logins_to_ttc

    @staticmethod
    def transition_kernel(id_to_state: dict, num_states: int, num_actions: int, HP: np.ndarray,
                          ttc_to_alerts_logins: dict,
                          alerts_logins_to_ttc: dict, state_to_id: dict) \
            -> np.ndarray:
        """

        Precomputes the transition kernel

        :param id_to_state: dict to convert between ids and states
        :param num_states: the number of states
        :param num_actions: the number of actions
        :param HP: the hack probability
        :param ttc_to_alerts_logins: a lookup dict between TTC and alerts and logins
        :param alerts_logins_to_ttc: a lookup dict between alerts/logins and TTC
        :param state_to_id: state to id dict
        :return: the transition kernel
        """
        f1_a = DefenderDynamics.f1_a()
        f1_b = DefenderDynamics.f1_b()
        f2_a = DefenderDynamics.f2_a()
        f2_b = DefenderDynamics.f2_b()
        T = np.zeros((num_states, num_actions, num_states))
        state_ids = list(range(num_states))
        sorted_state_ids = state_ids

        for i in range(num_states):
            s1 = id_to_state[i]
            if s1 == "terminal":
                T[i][0][i] = 1
                T[i][1][i] = 1
            else:
                t1, x1 = s1
                if t1 == constants.DP.MAX_TIMESTEPS - 1:
                    T[i][0][state_to_id["terminal"]] = 1
                    T[i][1][state_to_id["terminal"]] = 1
                else:
                    print("{}/{}".format(i, num_states))
                    feasible_states = sorted_state_ids
                    for j in feasible_states:
                        s2 = id_to_state[j]

                        if x1 == constants.DP.MIN_TTC:
                            T[i][1][state_to_id["terminal"]] = 1
                            if s2 == "terminal" and t1 < constants.DP.MAX_TIMESTEPS - 1:
                                T[i][0][j] = 0
                            else:
                                t2, x2 = s2
                                if t2 == t1 + 1 and x2 == x1:
                                    print("min reached")
                                    T[i][0][j] = 1
                        else:
                            for k in range(num_actions):
                                if k == constants.ACTIONS.STOPPING_ACTION:
                                    if s2 == "terminal":
                                        T[i][k][j] = 1
                                    else:
                                        T[i][k][j] = 0
                                else:
                                    if s2 == "terminal" and t1 < constants.DP.MAX_TIMESTEPS - 1:
                                        T[i][k][j] = 0
                                    else:
                                        t2, x2 = s2
                                        if t2 != (t1 + 1) or x2 >= x1:
                                            T[i][k][j] = 0
                                        else:
                                            hp = HP[j]
                                            prob = 0.0
                                            if x1 in ttc_to_alerts_logins and x2 in ttc_to_alerts_logins:
                                                (alerts1, logins1) = ttc_to_alerts_logins[x1]
                                                (alerts2, logins2) = ttc_to_alerts_logins[x2]
                                                alerts_delta = max(alerts2 - alerts1, 0.0)
                                                logins_delta = max(logins2 - logins1, 0.0)
                                                p = hp * (f1_a.pmf(alerts_delta) * f2_a.pmf(logins_delta)) + (1 - hp) \
                                                    * (f1_b.pmf(alerts_delta) * f2_b.pmf(logins_delta))
                                                prob = prob + p
                                                p = prob
                                                if p > 0 and k == 1:
                                                    print(
                                                        "setting positive probability despite stopping:{}, next state:{}".format(
                                                            p, j))
                                                T[i][k][j] = p
                                            else:
                                                print("total miss")
                    if sum(T[i][0]) == 0:
                        # t1, x1 = s1
                        new_state1 = (t1 + 1, x1 - 1)
                        s_prime_id1 = state_to_id[new_state1]
                        s_prime_id2 = None
                        if x1 > 5:
                            new_state2 = (t1 + 1, x1 - 5)
                            s_prime_id2 = state_to_id[new_state2]
                        elif x1 > 4:
                            new_state2 = (t1 + 1, x1 - 4)
                            s_prime_id2 = state_to_id[new_state2]
                        elif x1 > 3:
                            new_state2 = (t1 + 1, x1 - 3)
                            s_prime_id2 = state_to_id[new_state2]
                        elif x1 > 2:
                            new_state2 = (t1 + 1, x1 - 2)
                            s_prime_id2 = state_to_id[new_state2]
                        if s_prime_id2 is not None:
                            hp = HP[i]
                            T[i][0][s_prime_id1] = 1 - hp
                            T[i][0][s_prime_id2] = hp
                        else:
                            T[i][0][s_prime_id1] = 1
            if sum(T[i][0]) != 0:
                T[i][0] = T[i][0] / sum(T[i][0])
            if sum(T[i][1]) != 0:
                T[i][1] = T[i][1] / sum(T[i][1])
        DP.save_numpy(T, "transition_kernel.npy")
        return T

    @staticmethod
    def state_to_id_dict() -> Tuple[dict, dict]:
        """
        Utility function for creating a lookup dict to convert between state ids and states and back

        :return: (state_to_id lookup, id_to_state lookup)
        """
        state_to_id = {}
        id_to_state = {}
        id = 1
        state_to_id["terminal"] = 0
        id_to_state[0] = "terminal"
        for t in range(constants.DP.MAX_TIMESTEPS):
            for x in range(constants.DP.MAX_TTC):
                state_to_id[(t, x)] = id
                id_to_state[id] = (t, x)
                id += 1
        DP.save_pickle(state_to_id, "state_to_id.pkl")
        DP.save_pickle(id_to_state, "id_to_state.pkl")
        return state_to_id, id_to_state

    @staticmethod
    def reward_fun(state: np.ndarray, action: int, id_to_state: dict) -> float:
        """
        The reward function

        :param state: the current state
        :param action: the current action
        :param id_to_state: the id to state lookup dict
        :return: the reward
        """
        s1 = id_to_state[state]
        t1, x1 = s1
        hp = DefenderDynamics.hack_prob(x1, t1)
        if t1 == constants.DP.MAX_TIMESTEPS and action != constants.ACTIONS.STOPPING_ACTION:
            return hp * constants.DP.ATTACK_REWARD
        else:
            if action == constants.ACTIONS.STOPPING_ACTION:
                r = hp * 50 * (max(math.pow(x1, 1), 1)) + (1 - hp) * (constants.DP.EARLY_STOPPING_REWARD)
                return r
            else:
                return hp * constants.DP.ATTACK_REWARD + (1 - hp) * (constants.DP.SERVICE_REWARD)

    @staticmethod
    def one_step_lookahead(state, V, num_actions, num_states, T, discount_factor, R, next_state_lookahead, id_to_state) \
            -> np.ndarray:
        """
        Performs a one-step lookahead for value iteration

        :param state: the current state
        :param V: the current value function
        :param num_actions: the number of actions
        :param num_states: the number of states
        :param T: the transition kernel
        :param discount_factor: the discount factor
        :param R: the table with rewards
        :param next_state_lookahead: the next state lookahead table
        :param id_to_state: the id to state lookeahead table
        :return: an array with lookahead values
        """
        A = np.zeros(num_actions)
        for a in range(num_actions):
            for next_state in next_state_lookahead[str(state)]:
                reward = R[state][a]
                prob = T[state][a][next_state]
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    @staticmethod
    def next_states_lookahead_table(n_states: int, n_actions: int, T: np.ndarray, id_to_state: dict) -> dict:
        """
        Precomputes a table with lookeahead values

        :param n_states: the number of states
        :param n_actions: the number of actions
        :param T: the transition kernel
        :param id_to_state: the id to state lookup table
        :return: a dict with next state lookups
        """
        next_state_lookahead = {}
        for i in range(n_states):
            next_states = []
            print("{}/{}".format(i, n_states))
            for j in range(n_states):
                for k in range(n_actions):
                    if T[i][k][j] > 0.0 and j not in next_states:
                        next_states.append(j)
            if len(next_states) == 0:
                print("state:{}, {}, has no next state".format(i, id_to_state[i]))
            next_state_lookahead[i] = next_states
        DP.save_json(next_state_lookahead, "next_state_lookahead.json")
        return next_state_lookahead

    @staticmethod
    def value_iteration(T: np.ndarray, num_states: int, num_actions: int, state_to_id: dict,
                        id_to_state: dict, HP: np.ndarray, R: np.ndarray,
                        next_state_lookahead: dict,
                        theta=0.0001, discount_factor=1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        An implementation of the Value Iteration algorithm

        :param T: the transition kernel T
        :param num_states: the number of states
        :param num_actions: the number of actions
        :param state_to_id: the state-to-id lookup table
        :param id_to_state: the id-to-state lookup table
        :param HP: the table with hack probabilities
        :param R: the table with rewards
        :param next_state_lookahead: the next-state-lookahead table
        :param theta: convergence threshold
        :param discount_factor: the discount factor
        :return: (greedy policy, value function)
        """
        V = np.zeros(num_states)

        while True:
            # Stopping condition
            delta = 0
            # Update each state...
            for s in range(num_states):
                # Do a one-step lookahead to find the best action
                A = DP.one_step_lookahead(s, V, num_actions, num_states, T, discount_factor, R, next_state_lookahead,
                                       id_to_state)
                best_action_value = np.max(A)
                # Calculate delta across all states seen so far
                delta = max(delta, np.abs(best_action_value - V[s]))
                # Update the value function. Ref: Sutton book eq. 4.10.
                V[s] = best_action_value

            print("delta:{}".format(delta))
            # Check if we can stop
            if delta < theta:
                break

        # Create a deterministic policy using the optimal value function
        policy = np.zeros([num_states, num_actions * 2])
        for s in range(num_states):
            # One step lookahead to find the best action for this state
            A = DP.one_step_lookahead(s, V, num_actions, num_states, T, discount_factor, R, next_state_lookahead,
                                   id_to_state)
            best_action = np.argmax(A)
            # Always take the best action
            policy[s, best_action] = 1.0
            policy[s][2] = A[0]
            policy[s][3] = A[1]

        DP.save_numpy(V, "value_fun.npy")
        DP.save_numpy(policy, "policy.npy")
        return policy, V

    @staticmethod
    def compute_thresholds(V: np.ndarray, T: np.ndarray, R: np.ndarray, n_states: int, next_state_lookahead: dict,
                           id_to_state: dict, HP: np.ndarray) -> np.ndarray:
        """
        Utility function for computing stopping thresholds

        :param V: the value function
        :param T: the transiton kernel
        :param R: the reward function
        :param n_states: the number of states
        :param next_state_lookahead: the lookup table for the next state
        :param id_to_state: the id-to-state lookup table
        :param HP: the hack probabilities
        :return: the computed thresholds
        """
        thresholds = np.zeros(n_states)
        for i in range(n_states):
            s1 = id_to_state[i]
            w = 0
            hp = HP[i]
            w = R[i][0]
            for next_state in next_state_lookahead[str(i)]:
                prob = T[i][0][next_state]
                w = w + prob * V[next_state]
            if s1 != "terminal":
                (t1, x1) = s1
                alpha = -(w + 10 - 10 * hp) / (50 * hp)
            else:
                alpha = 0
            thresholds[i] = alpha
        DP.save_numpy(thresholds, "thresholds.npy")
        return thresholds

    @staticmethod
    def save_numpy(arr: np.ndarray, filename: str) -> None:
        """
        Utility function for saving a numpy array

        :param arr: the HP table to save
        :return: None
        """
        print("saving {}...".format(filename))
        np.save(filename, arr)
        print("{} saved".format(filename))


    @staticmethod
    def load_numpy(filename) -> np.ndarray:
        """
        Utility function for loading a numpy array

        :param filename: the name of the file to load
        :return: the loaded array
        """
        print("loading {}..".format(filename))
        with open(filename, 'rb') as f:
            arr = np.load(f)
            print("{} loaded:{}".format(filename, arr.shape))
            return arr

    @staticmethod
    def save_json(d: dict, file_name: str) -> None:
        """
        Utility function for saving a dict into json format

        :param d:
        :return: None
        """
        print("Saving {}".format(file_name))
        with open(file_name, 'w') as fp:
            json.dump(d, fp)
        print("{} saved".format(file_name))

    @staticmethod
    def save_pickle(obj, filename) -> None:
        """
        Utility function for saving an object with pickle

        :param obj: the obj to save
        :param filename: the name of the file to save
        :return: None
        """
        print("Saving {} table".format(filename))
        with open(filename, 'wb') as fp:
            pickle.dump(obj, fp)
        print("{} saved".format(filename))

    @staticmethod
    def load_pickle(filename) -> object:
        """
        Utility function for loading an object saved as pickle

        :param filename: the filename to load
        :return: the loaded object
        """
        print("Loading {}".format(filename))
        with open(filename, 'rb') as fp:
            obj = pickle.load(fp)
        print("{} loaded".format(filename))

        return obj

    @staticmethod
    def load_json(filename) -> dict:
        """
        Loads a dict saved as json

        :param filename: name of the file to load
        :return: the loaded file
        """
        print("Loading {}".format(filename))
        with open(filename, 'r') as fp:
            d = json.load(fp)
        print("{} loaded:{}".format(filename, len(d)))
        return d
