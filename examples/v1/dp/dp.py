import math
import numpy as np
import gym_optimal_intrusion_response.constants.constants as constants
from tempfile import TemporaryFile
from gym_optimal_intrusion_response.logic.defender_dynamics.defender_dynamics import DefenderDynamics
import json
import pickle
import sys


def num_states():
    return constants.DP.MAX_TTC * constants.DP.MAX_TIMESTEPS + 1


def num_actions():
    return 2


def actions():
    return {"continue": 0, "stop": 1}


def hp_and_ttc_and_r(n_states, n_actions, id_to_state):
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
                r = reward_fun(i, j, id_to_state)
                R[i][j] = r
    save_hp_table(HP)
    save_R_table(R)
    return HP, R


def ttc_to_alerts_table():
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

    save_tcc_to_alerts_logins(avg_ttc_to_alerts_logins, alerts_logins_to_ttc)
    return avg_ttc_to_alerts_logins, alerts_logins_to_ttc


def transition_kernel(id_to_state, num_states, num_actions, HP, ttc_to_alerts_logins, alerts_logins_to_ttc):
    f1_a = DefenderDynamics.f1_a()
    f1_b = DefenderDynamics.f1_b()
    f2_a = DefenderDynamics.f2_a()
    f2_b = DefenderDynamics.f2_b()
    T = np.zeros((num_states, num_actions, num_states))
    state_ids = list(range(num_states))
    # sorted_state_ids = sorted(state_ids[1:], key=lambda x: id_to_state[x][0], reverse=False)
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
        # if sum(T[i][0]) > 1.0:
        #     print("0")
        #     print(sum(T[i][0]))
        # if sum(T[i][1]) > 1.0:
        #     print("1")
        #     print(sum(T[i][1]))
        # assert sum(T[i][0]) == 1.0
        # assert sum(T[i][1]) == 1.0
        # print("after norm:{}".format(sum(T[i][0])))
        # print("after norm:{}".format(sum(T[i][1])))
    save_transition_kernel(T)
    # load_transition_kernel()
    return T


def save_hp_table(HP):
    print("saving HP table...")
    np.save('hp_table.npy', HP)
    print("HP table saved")


def save_ttc_table(TTC):
    print("saving TTC table...")
    np.save('ttc_table.npy', TTC)
    print("TTC table saved")


def save_R_table(R):
    print("saving R table...")
    np.save('reward_fun.npy', R)
    print("R table saved")


def save_transition_kernel(T):
    print("saving transition kernel..")
    np.save('transition_kernel.npy', T)
    print("kernel saved")


def save_value_function(V):
    print("saving value function..")
    np.save('value_fun.npy', V)
    print("value function saved")


def save_policy(policy):
    print("saving policy...")
    np.save('policy.npy', policy)
    print("policy saved")


def save_thresholds(thresholds):
    print("saving thresholds...")
    np.save('thresholds.npy', thresholds)
    print("thresholds saved")


def load_transition_kernel():
    print("loading transition kernel..")
    with open('transition_kernel.npy', 'rb') as f:
        T = np.load(f)
        print("kernel loaded:{}".format(T.shape))
        return T


def load_value_fun():
    print("loading value function..")
    with open('value_fun.npy', 'rb') as f:
        V = np.load(f)
        print("value function loaded:{}".format(V.shape))
        return V


def load_policy():
    print("loading policy..")
    with open('policy.npy', 'rb') as f:
        policy = np.load(f)
        print("policy loaded:{}".format(policy.shape))
        return policy


def load_thresholds():
    print("loading thresholds..")
    with open('thresholds.npy', 'rb') as f:
        thresholds = np.load(f)
        print("thresholds loaded:{}".format(thresholds.shape))
        return thresholds


def load_HP_table():
    print("loading HP table..")
    with open('hp_table.npy', 'rb') as f:
        HP = np.load(f)
        print("hp table loaded:{}".format(HP.shape))
        return HP


def load_TTC_table():
    print("loading TTC table..")
    with open('ttc_table.npy', 'rb') as f:
        TTC = np.load(f)
        print("TTC table loaded:{}".format(TTC.shape))
        return TTC


def load_R_table():
    print("loading R table..")
    with open('reward_fun.npy', 'rb') as f:
        R = np.load(f)
        print("R table loaded:{}".format(R.shape))
        return R


def state_to_id_dict():
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
    save_state_to_id(state_to_id)
    save_id_to_state(id_to_state)
    return state_to_id, id_to_state


def reward_fun(state, action, id_to_state):
    s1 = id_to_state[state]
    t1, x1 = s1
    # ttc = DefenderDynamics.ttc(x1, y1, constants.DP.MAX_ALERTS)
    hp = DefenderDynamics.hack_prob(x1, t1)
    if t1 == constants.DP.MAX_TIMESTEPS and action != constants.ACTIONS.STOPPING_ACTION:
        return hp * constants.DP.ATTACK_REWARD
    else:
        if action == constants.ACTIONS.STOPPING_ACTION:
            # print("hp:{}, stopping reward:{}".format(hp, hp * (100) + (1 - hp) * (-100)))
            # return hp * (math.pow(x1, 2))  + (1 - hp) * (-math.pow((constants.DP.MAX_TIMESTEPS + 1 - t1), 2))
            # return hp * (math.pow(x1, 2)) + (1 - hp) * (( t1/2 - constants.DP.MAX_TIMESTEPS))
            # return hp * (math.pow(x1, 2))*50 + (1 - hp) * ((-constants.DP.MAX_TIMESTEPS))
            # return hp * ((math.pow(x1, 1)) +(t1-100)) + (1 - hp) * (t1-100)
            # return hp * (100*(math.pow(x1, 1))) + (1 - hp) * (t1 - 100)
            # return hp * ((math.pow(x1, 1))) + (1 - hp) * (t1 - 100)
            # return 100*hp * (math.pow(x1, 2)) + (1 - hp) * (10*(t1 - constants.DP.MAX_TIMESTEPS))
            p1 = hp * 100*(math.pow(x1, 1))
            p2 = (1 - hp) * (constants.DP.EARLY_STOPPING_REWARD)
            r = hp * 50 * (max(math.pow(x1, 1), 1)) + (1 - hp) * (constants.DP.EARLY_STOPPING_REWARD)
            if r > 0:
                print("positive stopping r:{}, {}, hp:{}, p1:{}, p2:{}".format(r, (t1, x1), hp, p1, p2))
            return r
        else:
            return hp * constants.DP.ATTACK_REWARD + (1 - hp) * (constants.DP.SERVICE_REWARD)


def one_step_lookahead(state, V, num_actions, num_states, T, discount_factor, R, next_state_lookahead, id_to_state):
    A = np.zeros(num_actions)
    for a in range(num_actions):
        for next_state in next_state_lookahead[str(state)]:
            reward = R[state][a]
            prob = T[state][a][next_state]
            A[a] += prob * (reward + discount_factor * V[next_state])
    # print("return A:{}".format(A))
    return A


def next_states_lookahead_table(n_states, n_actions, T, id_to_state):
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
    save_next_states_lookahead_table(next_state_lookahead)
    return next_state_lookahead


def save_next_states_lookahead_table(next_state_lookahead):
    print("Saving next state lookahead table")
    with open("next_state_lookahead.json", 'w') as fp:
        json.dump(next_state_lookahead, fp)
    print("Next state lookahead saved")


def save_state_to_id(state_to_id):
    print("Saving state_to_id table")
    with open("state_to_id.json", 'wb') as fp:
        pickle.dump(state_to_id, fp)
    print("state_to_id dict saved")


def save_id_to_state(id_to_state):
    print("Saving id_to_state table")
    with open("id_to_state.json", 'wb') as fp:
        pickle.dump(id_to_state, fp)
    print("id_to_state dict saved")


def save_tcc_to_alerts_logins(ttc_to_alerts_logins, logins_alerts_to_tcc):
    print("Saving ttc_to_alerts_logins table")
    with open("ttc_to_alerts_logins.json", 'wb') as fp:
        pickle.dump(ttc_to_alerts_logins, fp)
    print("ttc_to_alerts_logins dict saved")

    print("Saving logins_alerts_to_tcc table")
    with open("logins_alerts_to_tcc.json", 'wb') as fp:
        pickle.dump(logins_alerts_to_tcc, fp)
    print("logins_alerts_to_tcc dict saved")


def load_tcc_to_alerts_logins():
    print("Loading ttc_to_alerts_logins table")
    with open("ttc_to_alerts_logins.json", 'rb') as fp:
        ttc_to_alerts_logins = pickle.load(fp)
    print("ttc_to_alerts_logins dict loaded")

    print("Loading logins_alerts_to_tcc table")
    with open("logins_alerts_to_tcc.json", 'rb') as fp:
        logins_alerts_to_tcc = pickle.load(fp)
    print("logins_alerts_to_tcc dict loaded")
    return ttc_to_alerts_logins, logins_alerts_to_tcc


def load_next_states_lookahead_table():
    print("Loading next state lookahead table")
    with open("next_state_lookahead.json", 'r') as fp:
        next_state_lookahead = json.load(fp)
    print("Next state lookahead loaded:{}".format(len(next_state_lookahead)))
    return next_state_lookahead


def load_state_to_id():
    print("Loading state_to_id table")
    with open("state_to_id.json", 'rb') as fp:
        state_to_id = pickle.load(fp)
    print("state_to_id loaded:{}".format(len(state_to_id)))
    return state_to_id


def load_id_to_state():
    print("Loading id_to_state table")
    with open("id_to_state.json", 'rb') as fp:
        id_to_state = pickle.load(fp)
    print("id_to_state loaded:{}".format(len(id_to_state)))
    return id_to_state


def value_iteration(T, num_states, num_actions, state_to_id, id_to_state, HP, R, next_state_lookahead,
                    theta=0.0001, discount_factor=1.0):
    V = np.zeros(num_states)

    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(num_states):
            # print("{}/{}".format(s, num_states))
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, num_actions, num_states, T, discount_factor, R, next_state_lookahead,
                                   id_to_state)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            if np.abs(best_action_value - V[s]) == 1.0 and s == 400:
                s1 = id_to_state[s]
                print("s1:{}".format(s1))
                print("delta:{}".format(delta))
                print(A)
                print("s:{}".format(s))
                print("V(s):{}".format(V[s]))
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
        A = one_step_lookahead(s, V, num_actions, num_states, T, discount_factor, R, next_state_lookahead, id_to_state)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
        policy[s][2] = A[0]
        policy[s][3] = A[1]

    save_value_function(V)
    save_policy(policy)
    return policy, V


def compute_thresholds(V, T, R, n_states, next_state_lookahead):
    thresholds = np.zeros(n_states)
    for i in range(n_states):
        w = 0
        for next_state in next_state_lookahead[str(i)]:
            prob = T[i][0][next_state]
            w = w + prob * V[next_state]
        alpha = math.sqrt((10) / (w + 5))
        thresholds[i] = alpha
    save_thresholds(thresholds)
    return thresholds


def compute_thresholds_2(V, T, R, n_states, next_state_lookahead, id_to_state, HP):
    thresholds = np.zeros(n_states)
    for i in range(n_states):
        s1 = id_to_state[i]
        w = 0
        hp = HP[i]
        w = R[i][0]
        for next_state in next_state_lookahead[str(i)]:
            prob = T[i][0][next_state]
            w = w + prob * V[next_state]
            # if s1 != "terminal":
            #     (t1, x1) = s1
            #     if t1 == 2:
            #         print("2 prob :{}".format(prob))
            #         print("2 val :{}".format(V[next_state]))
        if s1 != "terminal":
            (t1, x1) = s1
            if t1 == 99:
                print("99 w:{}".format(w))

            if t1 == 2:
                print("2 w:{}".format(w))
            # alpha = (math.sqrt(100)) / (math.sqrt(w+120 - t1))
            # alpha = (math.sqrt(t1)) / (math.sqrt(w +180 + 2*t1))
            # alpha = (2*(-100))/(w+2*t1-110)
            # alpha = (-100)/(-110 + w + t1)
            # alpha = t1/(w+2*t1)
            # alpha = 2*(-50+t1)/(-99-w-t1)
            # alpha = (100-t1)/(135 - t1+w)
            # alpha = (t1-100) / (t1-w-99)
            # alpha = 100/(99+w)
            # alpha=(t1-100)/(t1-w-99)
            # alpha = (101-t1) / (150 + w -t1)
            # alpha = 101/(100+w)
            # alpha = (101 - t1) / (100 + w - t1)
            # alpha =-((w+100-100*hp)/(hp))
            alpha = -(w+ 10-10*hp)/(50*hp)

            # alpha = - (math.sqrt((w + 100 - 100 * hp) / (hp)))
            # alpha = - (math.sqrt((w + 100 - 100 * hp + hp*t1 - t1) / (hp)))
            # alpha = -((w + 100 - 100 * hp + hp * t1 - t1) / (hp))

            # alpha = 100/(99+w)
            # alpha = (1) / (101 - w - t1)
            # alpha = (math.sqrt(1000-10*t1)) / (math.sqrt(1200 + w-10*t1))
            # alpha = (math.sqrt(10001 -10*t1)) / max(1, (math.sqrt(1010 + w - 10 * t1)))
            # alpha = 100/(1+w)
            # alpha = (math.sqrt(50)) / (math.sqrt(50 + w - t1/2))
            # alpha = (math.sqrt(50)) / (math.sqrt(50 + w - t1 / 2))
            # alpha = 100/(max(1,w-t1))
            # alpha = (100-t1/2) / (200 + w - t1)
            # print("t1:{}, w:{}, alpha:{}".format(t1, w, alpha))
            # alpha = (math.sqrt(101)) / (math.sqrt(200 + w - t1))
            # alpha = math.sqrt((math.pow(100,2) - 200*t1 + math.pow(t1,2))/(w+math.pow(100, 2) - 1 - 200*t1 + math.pow(t1, 2)))
            # alpha = (math.sqrt(200 - t1/2)) / (math.sqrt(198 + 2*w - t1))
        else:
            alpha = 0
        thresholds[i] = alpha
    save_thresholds(thresholds)
    return thresholds


def verify_transition_kernel(T, n_states):
    # print(sum(T[0][1][:]))
    # t = []
    # for i in range(n_states):
    #     if T[0][1][i] > 0:
    #         t.append(i)
    # print(T[0][1][:])
    # print(t)
    # print(T[0][1][0])
    for i in range(n_states):
        # if sum(T[i][0]) > 1.0:
        #     print("overloaded T:{}, a:{}".format(sum(T[i][0]), 0))
        if sum(T[i][1]) > 1.01:
            print("overloaded T:{}, a:{}".format(sum(T[i][1]), 1))


if __name__ == '__main__':
    print("num states:{}, num actions:{}".format(num_states(), num_actions()))
    state_to_id, id_to_state = state_to_id_dict()
    # ttc_to_alerts_logins, alerts_logins_to_ttc = ttc_to_alerts_table()
    ttc_to_alerts_logins, alerts_logins_to_ttc = load_tcc_to_alerts_logins()
    HP, R = hp_and_ttc_and_r(num_states(), num_actions(), id_to_state)
    T = transition_kernel(id_to_state, num_states(), num_actions(), HP, ttc_to_alerts_logins, alerts_logins_to_ttc)
    next_state_lookahead = next_states_lookahead_table(num_states(), num_actions(), T, id_to_state)
    HP = load_HP_table()
    R = load_R_table()
    T = load_transition_kernel()
    next_state_lookahead = load_next_states_lookahead_table()
    policy, V = value_iteration(T, num_states(), num_actions(), state_to_id, id_to_state, HP, R,
                                next_state_lookahead, theta=0.0001, discount_factor=1.0)
    V = load_value_fun()
    policy = load_policy()
    thresholds = compute_thresholds_2(V, T, R, num_states(), next_state_lookahead, id_to_state, HP)
    for i in range(len(V)):
        s = id_to_state[i]
        if s != "terminal":
            t1, x1 = s
            print("t:{}, ttc:{}, V[s]:{}, actions:{}, threshold:{}".format(t1, x1, V[i], policy[i], thresholds[i]))
    # # thresholds = compute_thresholds(V, T, R, num_states(), next_state_lookahead)
    # thresholds = load_thresholds()
    # print(thresholds)
    # print(max(thresholds))
    # for i in range(400):
    #     print(thresholds[i])
    #
    #
    # print(V[0])
    # print(V[-1])
    # print(V[-2])
    # print(V[-100])
    # print(V[-200])
    # print(V[-300])
    # print(V[1])
    # print(V[2])
    # print(V[50])
    # for i in range(5000):
    #     s = id_to_state[i]
    #     if s != "terminal":
    #         t1, x1, y1 = s
    #         print("state:{}, value:{}, hp:{}, ttc:{}, r:{}, action:{}".format((t1,x1,y1), V[i], HP[i], TTC[i], (R[i][0], R[i][1]), policy[i]))
