"""
Utility script for plotting a dynamics model
"""

from gym_optimal_intrusion_response.util.plots import plot_dynamics_model

def plot() -> None:
    """
    Plots a dynamics model

    :return: None
    """
    # path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/new_defender_dynamics_model_2.json"
    # path = "/home/kim/workspace/gym-optimal-intrusion-response/traces/traces.json"
    path="/Users/kimham/workspace/gym-optimal-intrusion-response/traces/traces.json"
    defender_dynamics_model = plot_dynamics_model.read_model(path)
    a_counts = []
    for attack_id_str, v1 in defender_dynamics_model.num_new_warning_alerts.items():
        c = 0
        for logged_in_ips, v2 in v1.items():
            temp = 0
            for num_alerts_str, count in v2.items():
                temp = temp + count
            c += temp
        a_counts.append((attack_id_str, c))
    print(a_counts)
    print(len(a_counts))
    # print(max(a_counts))



# Script entrypoint
if __name__ == '__main__':
    plot()