import math
import numpy as np
import gym_optimal_intrusion_response.constants.constants as constants
from tempfile import TemporaryFile
from gym_optimal_intrusion_response.logic.defender_dynamics.defender_dynamics import DefenderDynamics
import json
import pickle
import sys

def num_states():
    return constants.DP.MAX_ALERTS*constants.DP.MAX_TIMESTEPS*constants.DP.MAX_LOGINS + 1

def num_actions():
    return 2

def actions():
    return {"continue": 0, "stop": 1}


def hp_and_ttc_and_r(n_states, n_actions, id_to_state):
    HP = np.zeros(n_states)
    TTC = np.zeros(n_states)
    R = np.zeros((n_states, n_actions))
    for i in range(n_states):
        s1 = id_to_state[i]
        if s1 == "terminal":
            HP[i] = 0
            TTC[i] = 0
            R[i][0] = 0
            R[i][1] = 0
        else:
            t1, x1, y1 = s1
            ttc = DefenderDynamics.ttc(x1, y1, constants.DP.MAX_ALERTS)
            hp = DefenderDynamics.hack_prob(ttc, t1)
            HP[i] = hp
            TTC[i] = ttc
            for j in range(n_actions):
                r = reward_fun(i, j, id_to_state)
                R[i][j] = r
    save_hp_table(HP)
    save_ttc_table(TTC)
    save_R_table(R)
    return HP, TTC, R

def transition_kernel(id_to_state, num_states, num_actions, HP, TTC):
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
        else :
            t1, x1, y1 = s1
            if t1 == constants.DP.MAX_TIMESTEPS - 1:
                T[i][0][state_to_id["terminal"]] = 1
                T[i][1][state_to_id["terminal"]] = 1
            else:
                print("{}/{}".format(i, num_states))
                # feasible_states = list(filter(lambda x: id_to_state[x][0] == t1+1 and id_to_state[x][1] >= x1 and id_to_state[x][2] >= y1, state_ids))
                # feasible_states = sorted_state_ids[t1*(constants.DP.MAX_ALERTS*constants.DP.MAX_LOGINS*num_actions):]
                feasible_states = sorted_state_ids
                for j in feasible_states:
                    s2 = id_to_state[j]

                    if x1 == constants.DP.MAX_ALERTS - 1 and y1 == constants.DP.MAX_LOGINS - 1:
                        T[i][1][state_to_id["terminal"]] = 1
                        if s2 == "terminal" and t1 < constants.DP.MAX_TIMESTEPS - 1:
                            T[i][0][j] = 0
                        else:
                            t2, x2, y2 = s2
                            if t2 == t1+1 and x2==x1 and y2==y1:
                                print("max reached")
                                T[i][0][j] = 1
                    else:
                        for k in range(num_actions):
                            if k == constants.ACTIONS.STOPPING_ACTION:
                                if s2 == "terminal":
                                    T[i][k][j] = 1
                                else:
                                    T[i][k][j] = 0
                            else:
                                if s2 == "terminal" and t1 < constants.DP.MAX_TIMESTEPS-1:
                                    T[i][k][j] = 0
                                else:
                                    t2, x2, y2 = s2
                                    if t2 != (t1+1) or x2 < x1 or y2<y1:
                                        T[i][k][j] = 0
                                    else:
                                        hp = HP[j]
                                        x_delta = x2 - x1
                                        y_delta = y2 - y1
                                        p = hp*(f1_a.pmf(x_delta)*f2_a.pmf(y_delta)) + (1-hp)*(f1_b.pmf(x_delta)*f2_b.pmf(y_delta))
                                        if p > 0 and k == 1:
                                            print("setting positive probability despite stopping:{}, nexgt state:{}".format(p, j))
                                        T[i][k][j] = p
        if sum(T[i][0]) == 0:
            print("transition probabilities are zero! :{}".format(i))
        if sum(T[i][1]) == 0:
            print("transition probabilities are zero!: {}".format(i))
        T[i][0] = T[i][0]/sum(T[i][0])
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
    #load_transition_kernel()
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
    state_to_id["terminal"] =0
    id_to_state[0] = "terminal"
    for t in range(constants.DP.MAX_TIMESTEPS):
        for x in range(constants.DP.MAX_ALERTS):
            for y in range(constants.DP.MAX_LOGINS):
                state_to_id[(t, x, y)] = id
                id_to_state[id] = (t, x, y)
                id+= 1
    save_state_to_id(state_to_id)
    save_id_to_state(id_to_state)
    return state_to_id, id_to_state

def reward_fun(state, action, id_to_state):
    s1 = id_to_state[state]
    t1, x1, y1 = s1
    ttc = DefenderDynamics.ttc(x1, y1, constants.DP.MAX_ALERTS)
    hp = DefenderDynamics.hack_prob(ttc, t1)
    if action == constants.ACTIONS.STOPPING_ACTION:
        # print("hp:{}, stopping reward:{}".format(hp, hp * (100) + (1 - hp) * (-100)))
        return hp * (100) + (1 - hp) * (-100)
    else:
        return hp*(-100) + (1-hp)*(-1)


def one_step_lookahead(state, V, num_actions, num_states, T, discount_factor, R, next_state_lookahead, id_to_state):
    A = np.zeros(num_actions)
    for a in range(num_actions):
        for next_state in next_state_lookahead[str(state)]:
            reward = R[state][a]
            prob = T[state][a][next_state]
            if reward > 0 and next_state != 0 and prob > 0:
                s1 = id_to_state[state]
                print("positive reward but next state is not terminal:{}, current state:{}, action:{}, current state type:{}, prob:{}".format(next_state, state, a, s1, prob))
            # print("prob:{}".format(prob))
            # print("reward:{}".format(reward))
            # print("V[next_state]:{}".format(V[next_state]))
            if math.isnan(V[next_state]):
                print(V[next_state])
                sys.exit(0)
            upd = prob * (reward + discount_factor * V[next_state])
            if math.isnan(upd):
                print("upd is nan")
                print("prob:{}".format(prob))
                print("reward:{}".format(reward))
                print("V[next_state]:{}".format(V[next_state]))
                sys.exit(0)

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
        if len(next_states)== 0:
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


def value_iteration(T, num_states, num_actions, state_to_id, id_to_state, HP, TTC, R, next_state_lookahead,
                    theta=0.0001, discount_factor=1.0):
    V = np.zeros(num_states)

    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(num_states):
            # print("{}/{}".format(s, num_states))
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, num_actions, num_states, T, discount_factor, R, next_state_lookahead, id_to_state)
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
    policy = np.zeros([num_states, num_actions*2])
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

def compute_thresholds(V, T, R, n_states,  next_state_lookahead):
    thresholds = np.zeros(n_states)
    for i in range(n_states):
        w = 0
        for next_state in next_state_lookahead[str(i)]:
            prob = T[i][0][next_state]
            w = w + prob*V[next_state]
        alpha = (w + 100)/200
        # if R[i][1] == 0:
        #     alpha = 1
        # else:
        #     alpha = w/R[i][1]
        # alpha = min(1, alpha)
        # alpha = max(0, alpha)
        thresholds[i] = alpha
    save_thresholds(thresholds)
    return thresholds

def compute_thresholds_2(V, T, R, n_states,  next_state_lookahead, id_to_state):
    thresholds = np.zeros(n_states)
    for i in range(n_states):
        w = 0
        for next_state in next_state_lookahead[str(i)]:
            prob = T[i][0][next_state]
            w = w + prob*V[next_state]
        s = id_to_state[i]
        if s != "terminal":
            t1, x1, y1 = s
            alpha = (100*t1)/(w+100)
            # alpha = (w + 100) / 200
        else:
            alpha = 200
        # ttc = TTC[i]
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
    # HP, TTC, R = hp_and_ttc_and_r(num_states(), num_actions(), id_to_state)
    # T = transition_kernel(id_to_state, num_states(), num_actions(), HP, TTC)
    # next_state_lookahead = next_states_lookahead_table(num_states(), num_actions(), T, id_to_state)
    # HP = load_HP_table()
    TTC = load_TTC_table()
    # R = load_R_table()
    # T = load_transition_kernel()
    # # verify_transition_kernel(T, num_states())
    # next_state_lookahead = load_next_states_lookahead_table()
    # policy, V = value_iteration(T, num_states(), num_actions(), state_to_id, id_to_state, HP, TTC, R,
    #                             next_state_lookahead, theta=0.0001, discount_factor=1.0)
    # V = load_value_fun()
    # policy = load_policy()
    # thresholds = compute_thresholds_2(V, T, R, num_states(), next_state_lookahead, id_to_state)
    thresholds = load_thresholds()
    print(thresholds)
    print(max(thresholds))
    for i in range(2000):
        print(thresholds[i])
    # print(V[0])
    # print(V[1])
    # print(V[2])
    # print(V[50])
    # for i in range(5000):
    #     s = id_to_state[i]
    #     if s != "terminal":
    #         t1, x1, y1 = s
    #         print("state:{}, value:{}, hp:{}, ttc:{}, r:{}, action:{}".format((t1,x1,y1), V[i], HP[i], TTC[i], (R[i][0], R[i][1]), policy[i]))
