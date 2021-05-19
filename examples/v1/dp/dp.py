"""
Value Iteration Dynamic Programming to Compute the Optimal Policy
"""
from gym_optimal_intrusion_response.logic.defender_dynamics.dp import DP


# Script entrypoint
if __name__ == '__main__':
    print("num states:{}, num actions:{}".format(DP.num_states(), DP.num_actions()))
    state_to_id, id_to_state = DP.state_to_id_dict()
    # ttc_to_alerts_logins, alerts_logins_to_ttc = ttc_to_alerts_table()
    ttc_to_alerts_logins = DP.load_pickle(filename="ttc_to_alerts_logins.pkl")
    alerts_logins_to_ttc = DP.load_pickle(filename="logins_alerts_to_tcc.pkl")
    HP, R = DP.hp_and_ttc_and_r(DP.num_states(), DP.num_actions(), id_to_state)
    T = DP.transition_kernel(id_to_state, DP.num_states(), DP.num_actions(), HP, ttc_to_alerts_logins, alerts_logins_to_ttc,
                          state_to_id)
    next_state_lookahead = DP.next_states_lookahead_table(DP.num_states(), DP.num_actions(), T, id_to_state)
    HP = DP.load_numpy("hp_table.npy")
    R = DP.load_numpy("reward_fun.npy")
    T = DP.load_numpy("transition_kernel.npy")
    next_state_lookahead = DP.load_json("next_state_lookahead.json")
    policy, V = DP.value_iteration(T, DP.num_states(), DP.num_actions(), state_to_id, id_to_state, HP, R,
                                next_state_lookahead, theta=0.0001, discount_factor=1.0)
    V = DP.load_numpy("value_fun.npy")
    policy = DP.load_numpy("policy.npy")
    thresholds = DP.compute_thresholds(V, T, R, DP.num_states(), next_state_lookahead, id_to_state, HP)
    for i in range(len(V)):
        s = id_to_state[i]
        if s != "terminal":
            t1, x1 = s
            print("t:{}, ttc:{}, V[s]:{}, actions:{}, threshold:{}".format(t1, x1, V[i], policy[i], thresholds[i]))