"""
Utility script for plotting a dynamics model
"""

from gym_optimal_intrusion_response.util.plots import plot_dynamics_model

def plot() -> None:
    """
    Plots a dynamics model

    :return: None
    """
    path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/new_defender_dynamics_model_2.json"
    defender_dynamics_model = plot_dynamics_model.read_model(path)
    plot_dynamics_model.plot_ids_infra_and_one_machine(defender_dynamics_model)


# Script entrypoint
if __name__ == '__main__':
    plot()