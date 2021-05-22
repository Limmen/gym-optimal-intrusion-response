"""
Value Iteration Dynamic Programming to Compute the Optimal Policy
"""
from gym_optimal_intrusion_response.logic.defender_dynamics.dp import DP
from typing import Tuple
import math
from gym_optimal_intrusion_response.logic.defender_dynamics.defender_dynamics import DefenderDynamics
import json
import pickle
import gym_optimal_intrusion_response.constants.constants as constants
import numpy as np
from gym_pycr_ctf.dao.defender_dynamics.defender_dynamics_model import DefenderDynamicsModel
from scipy.stats import poisson
import math


def num_states_2() -> int:
    """
    :return: the number of states
    """
    # count = 0
    # for t in range(constants.DP.MAX_TIMESTEPS):
    #     for i in range(min(f1A.a, f1B.a), max(f1A.b, f1B.b)):
    #         if f1A.pmf(i) + f1B.pmf(i) > 0:
    #             for j in range(min(f2A.a, f2B.a), max(f2A.b, f2B.b)):
    #                 if f2A.pmf(j) + f2B.pmf(j) > 0:
    #                     count +=1
    # count +=1
    # return count
    return constants.DP2.MAX_SEVERE_ALERTS * constants.DP2.MAX_WARNING_ALERTS * constants.DP.MAX_TIMESTEPS + 1


def hack_prob2(t, x, y, z):
    """
    The hack probability

    :param ttc_val: the ttc
    :param t: the time-step
    :return:
    """
    hp = 1/max(1, (40-(x+y+2*t)))
    return hp

def reward_fun2(state: np.ndarray, action: int, id_to_state: dict) -> float:
    """
    The reward function

    :param state: the current state
    :param action: the current action
    :param id_to_state: the id to state lookup dict
    :return: the reward
    """
    s1 = id_to_state[state]
    t1, x1, y1, z1 = s1
    hp = hack_prob2(t1, x1, y1, z1)
    if t1 == constants.DP2.MAX_TIMESTEPS and action != constants.ACTIONS.STOPPING_ACTION:
        return hp * constants.DP2.ATTACK_REWARD
    else:
        if action == constants.ACTIONS.STOPPING_ACTION:
            r = (hp * constants.DP2.STOPPING_REWARD)/(math.pow(max(1, t1-(1/hp)), 1.05)) + (1 - hp) * (constants.DP2.EARLY_STOPPING_REWARD)
            return r
        else:
            return hp * constants.DP2.ATTACK_REWARD + (1 - hp) * (constants.DP2.SERVICE_REWARD)

def state_to_id_dict2() -> Tuple[dict, dict]:
    """
    Utility function for creating a lookup dict to convert between state ids and states and back

    :return: (state_to_id lookup, id_to_state lookup)
    """
    state_to_id = {}
    id_to_state = {}
    id = 1
    state_to_id["terminal"] = 0
    id_to_state[0] = "terminal"

    for t in range(constants.DP2.MAX_TIMESTEPS):
        for x in range(constants.DP2.MAX_SEVERE_ALERTS):
            for y in range(constants.DP2.MAX_WARNING_ALERTS):
                state_to_id[(t, x, y, 0)] = id
                id_to_state[id] = (t, x, y, 0)
                id += 1
    DP.save_pickle(state_to_id, "state_to_id.pkl")
    DP.save_pickle(id_to_state, "id_to_state.pkl")
    return state_to_id, id_to_state

def transition_kernel(id_to_state: dict, num_states: int, num_actions: int, HP: np.ndarray,
                      state_to_id: dict, f1_a, f2_a, f1_b, f2_b) \
        -> np.ndarray:
    """

    Precomputes the transition kernel

    :param id_to_state: dict to convert between ids and states
    :param num_states: the number of states
    :param num_actions: the number of actions
    :param HP: the hack probability
    :param state_to_id: state to id dict
    :return: the transition kernel
    """
    T = np.zeros((num_states, num_actions, num_states))
    state_ids = list(range(num_states))
    sorted_state_ids = state_ids

    for i in range(num_states):
        s1 = id_to_state[i]
        if s1 == "terminal":
            T[i][0][i] = 1
            T[i][1][i] = 1
        else:
            t1, x1, y1, z1 = s1
            if t1 == constants.DP2.MAX_TIMESTEPS - 1:
                T[i][0][state_to_id["terminal"]] = 1
                T[i][1][state_to_id["terminal"]] = 1
            elif x1== constants.DP2.MAX_SEVERE_ALERTS and y1 == constants.DP2.MAX_WARNING_ALERTS and z1 == 0:
                T[i][0][i] = 1
                T[i][1][state_to_id["terminal"]] = 1
            else:
                print("{}/{}".format(i, num_states))
                feasible_states = sorted_state_ids
                for j in feasible_states:
                    s2 = id_to_state[j]
                    if s2 != "terminal":
                        t2, x2, y2, z2 = s2
                    for k in range(num_actions):
                        if k == constants.ACTIONS.STOPPING_ACTION:
                            if s2 == "terminal":
                                T[i][k][j] = 1
                            else:
                                T[i][k][j] = 0
                        else:
                            if s2 != "terminal":
                                #state_to_id
                                if t2 != t1+1 or x2 < x1 or y2 < y1 or z2 != 0:
                                    T[i][k][j] = 0
                                else:
                                    hp = HP[j]
                                    prob = hp*(f1_a.pmf(x2-x1)*f2_a.pmf(y2-y1)) + (1-hp)*(f1_b.pmf(x2-x1)*f2_b.pmf(y2-y1))
                                    # print("prob:{}, {}, {}".format(prob, x2-x1, y2-y1))
                                    T[i][k][j] = prob
        if sum(T[i][0]) != 0:
            T[i][0] = T[i][0] / sum(T[i][0])
        if sum(T[i][1]) != 0:
            T[i][1] = T[i][1] / sum(T[i][1])

    DP.save_numpy(T, "transition_kernel.npy")
    return T


def hp_and_r(n_states: int, n_actions: int, id_to_state: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-computes the HP and R

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
            t1, x1, y1, z1 = s1
            hp = hack_prob2(t1, x1, y1, z1)
            HP[i] = hp
            for j in range(n_actions):
                r = reward_fun2(i, j, id_to_state)
                R[i][j] = r
    DP.save_numpy(HP, "hp_table.npy")
    DP.save_numpy(R, "reward_fun.npy")
    return HP, R

def compute_thresholds2(V: np.ndarray, T: np.ndarray, R: np.ndarray, n_states: int, next_state_lookahead: dict,
                       id_to_state: dict, HP: np.ndarray, f1A, f2A, f1B, f2B) -> np.ndarray:
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
            t1, x1, y1, z1 = s1
            alpha = w
            # alpha = w/t1
            # alpha = -(w + 10 - 10 * hp) / (50 * hp)
        else:
            alpha = 0
        thresholds[i] = alpha
    DP.save_numpy(thresholds, "thresholds.npy")
    return thresholds


def compute_thresholds3(V: np.ndarray, T: np.ndarray, R: np.ndarray, n_states: int, next_state_lookahead: dict,
                       id_to_state: dict, HP: np.ndarray, f1A, f2A, f1B, f2B) -> np.ndarray:
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
        print("thresholds {}/{}".format(i, n_states))
        s1 = id_to_state[i]
        w = 0
        hp = HP[i]
        w = R[i][0]
        b = 0
        a = 0
        b = constants.DP2.ATTACK_REWARD - constants.DP2.SERVICE_REWARD
        a = constants.DP2.SERVICE_REWARD
        for next_state in next_state_lookahead[str(i)]:
            prob = T[i][0][next_state]
            w = w + prob * V[next_state]
            if s1 != "terminal":
                t1, x1, y1, z1 = s1
                b_temp = f1A.pmf(x1)*f2A.pmf(y1) - f1B.pmf(x1)*f2B.pmf(y1)
                b_temp = b_temp*V[next_state]
                b = b + b_temp
                a_temp = f1B.pmf(x1)*f2B.pmf(y1)
                a_temp = a_temp * V[next_state]
                a_temp = a + a_temp
        if s1 != "terminal":
            t1, x1, y1, z1 = s1
            g = (x1 + y1)
            alpha = ((constants.DP2.EARLY_STOPPING_REWARD + b - constants.DP2.STOPPING_REWARD)/(constants.DP2.EARLY_STOPPING_REWARD - a))

            # alpha = w
            # alpha = -(w + 10 - 10 * hp) / (50 * hp)
        else:
            alpha = 0
        thresholds[i] = alpha
    DP.save_numpy(thresholds, "thresholds.npy")
    return thresholds

# Script entrypoint
if __name__ == '__main__':
    # save_dynamics_model_dir = "/Users/kimham/workspace/gym-optimal-intrusion-response/traces/"
    # defender_dynamics_model = DefenderDynamicsModel()
    # defender_dynamics_model.read_model(save_dynamics_model_dir, model_name="traces.json")
    # defender_dynamics_model.normalize()

    # f1B = defender_dynamics_model.norm_num_new_severe_alerts[(85, '172.18.9.191')]
    # f2B = defender_dynamics_model.norm_num_new_warning_alerts[(85, '172.18.9.191')]
    # f1A = defender_dynamics_model.norm_num_new_severe_alerts[(20, '172.18.9.191')]
    # f2A = defender_dynamics_model.norm_num_new_warning_alerts[(20, '172.18.9.191')]
    f1B = poisson(mu=2)
    f2B = poisson(mu=3)
    f1A = poisson(mu=11)
    f2A = poisson(mu=9)
    # print("num states:{}, num actions:{}".format(num_states_2(), DP.num_actions()))
    state_to_id, id_to_state = state_to_id_dict2()


    HP, R = hp_and_r(num_states_2(), DP.num_actions(), id_to_state)
    T = transition_kernel(id_to_state, num_states_2(), DP.num_actions(), HP, state_to_id, f1A, f2A, f1B, f2B)
    next_state_lookahead = DP.next_states_lookahead_table(num_states_2(), DP.num_actions(), T, id_to_state)


    HP = DP.load_numpy("hp_table.npy")
    R = DP.load_numpy("reward_fun.npy")
    T = DP.load_numpy("transition_kernel.npy")
    next_state_lookahead = DP.load_json("next_state_lookahead.json")


    policy, V = DP.value_iteration(T, num_states_2(), DP.num_actions(), state_to_id, id_to_state, HP, R,
                                next_state_lookahead, theta=0.0001, discount_factor=1.0)


    V = DP.load_numpy("value_fun.npy")
    policy = DP.load_numpy("policy.npy")
    thresholds = compute_thresholds2(V, T, R, num_states_2(), next_state_lookahead, id_to_state, HP, f1A, f2A, f1B, f2B)
    for i in range(len(V)):
        s = id_to_state[i]
        if s != "terminal":
            t1, x1, y1, z1 = s
            print("t:{}, x:{}, y:{}, z:{}, V[s]:{}, actions:{}, threshold:{}".format(t1, x1, y1, z1, V[i], policy[i],
                                                                                     thresholds[i]))

    # thresholds = compute_thresholds3(V, T, R, num_states_2(), next_state_lookahead, id_to_state, HP, f1A, f2A, f1B, f2B)