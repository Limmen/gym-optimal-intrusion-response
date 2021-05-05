import math
import numpy as np
import gym_optimal_intrusion_response.constants.constants as constants
from tempfile import TemporaryFile
from gym_optimal_intrusion_response.logic.defender_dynamics.defender_dynamics import DefenderDynamics

def num_states():
    return constants.DP.MAX_ALERTS*constants.DP.MAX_TIMESTEPS*constants.DP.MAX_LOGINS

def num_actions():
    return 2

def actions():
    return {"continue": 0, "stop": 1}


def transition_kernel(id_to_state, num_states, num_actions):
    f1_a = DefenderDynamics.f1_a()
    f1_b = DefenderDynamics.f1_b()
    f2_a = DefenderDynamics.f2_a()
    f2_b = DefenderDynamics.f2_b()
    T = np.zeros((num_states, num_states, num_actions))
    state_ids = list(range(num_states))
    #state_ids_with_steps = list(map(lambda x: (x, id_to_state[x][0]), state_ids))
    #sorted_state_ids = sorted(state_ids_with_steps, key=lambda x: x[1], reverse=False)
    sorted_state_ids = sorted(state_ids, key=lambda x: id_to_state[x][0], reverse=False)

    for i in range(num_states):
        s1 = id_to_state[i]
        t1, x1, y1 = s1
        print("{}/{}".format(i, num_states))
        # feasible_states = list(filter(lambda x: id_to_state[x][0] == t1+1 and id_to_state[x][1] >= x1 and id_to_state[x][2] >= y1, state_ids))
        feasible_states = sorted_state_ids[t1*1500:]
        for j in feasible_states:
        # for j in range(num_states):
            s2 = id_to_state[j]
            for k in range(num_actions):
                t2, x2, y2 = s2
                if t2 != (t1+1) or x2 < x1 or y2<y1:
                    T[i][j][k] = 0
                else:
                    ttc = DefenderDynamics.ttc(x1, y1, 200)
                    hp = DefenderDynamics.hack_prob(ttc, t1)
                    x_delta = x2 - x1
                    y_delta = y2 - y1
                    p = hp*(f1_a.pmf(x_delta)*f2_a.pmf(y_delta)) + (1-hp)*(f1_b.pmf(x_delta)*f2_b.pmf(y_delta))
                    T[i][j][k] = p
    save_transition_kernel(T)
    #load_transition_kernel()
    return T

def save_transition_kernel(T):
    print("saving transition kernel..")
    np.save('transition_kernel.npy', T)
    print("kernel saved")

def load_transition_kernel():
    print("loading transition kernel..")
    with open('transition_kernel.npy', 'rb') as f:
        T = np.load(f)
        print("kernel loaded:{}".format(T.shape))
        return T

def state_to_id_dict():
    state_to_id = {}
    id_to_state = {}
    id = 0
    for t in range(constants.DP.MAX_TIMESTEPS):
        for x in range(constants.DP.MAX_ALERTS):
            for y in range(constants.DP.MAX_LOGINS):
                state_to_id[(t, x, y)] = id
                id_to_state[id] = (t, x, y)
                id+= 1
    return state_to_id, id_to_state

def reward_fun(state, action, state_to_id, id_to_state):
    s1 = id_to_state[state]
    t1, x1, y1 = s1


def one_step_lookahead(state, V, num_actions, num_states, T, discount_factor):
    A = np.zeros(num_actions)
    for a in range(num_actions):
        for next_state in range(num_states):
            reward = reward_fun(state, a, state_to_id, id_to_state)
            prob = T[state][next_state][a]
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A

def value_iteration(T, num_states, num_actions, state_to_id, id_to_state, theta=0.0001, discount_factor=1.0):
    V = np.zeros(num_states)

    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(num_states):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, num_actions, num_states, T, discount_factor)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value

        print("theta:{}".format(delta))
        # Check if we can stop
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([num_states, num_actions])
    for s in range(num_states):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V, num_actions, num_states, T, discount_factor)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, V

if __name__ == '__main__':
    print("num states:{}, num actions:{}".format(num_states(), num_actions()))
    state_to_id, id_to_state = state_to_id_dict()
    print(len(state_to_id))
    # T = transition_kernel(id_to_state, num_states(), num_actions())
    T = load_transition_kernel()
    print(T.shape)
    policy, V = value_iteration(T, num_states(), num_actions(), state_to_id, id_to_state, theta=0.0001, discount_factor=1.0)