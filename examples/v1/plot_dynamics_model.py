from gym_optimal_intrusion_response.util.plots import plot_dynamics_model

def plot():
    print("reading model")
    # path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v1/new_defender_dynamics_model.json"
    path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/new_defender_dynamics_model_2.json"
    defender_dynamics_model = plot_dynamics_model.read_model(path)
    print("model read")
    #plot_dynamics_model.plot_all(defender_dynamics_model)
    plot_dynamics_model.plot_ids_infra_and_one_machine(defender_dynamics_model)
    # plot_dynamics_model.plot_ids_dynamics_2(defender_dynamics_model)

if __name__ == '__main__':
    plot()