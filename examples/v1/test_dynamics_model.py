"""
Utility script for testing a dynamics model
"""
from gym_pycr_ctf.dao.defender_dynamics.defender_dynamics_model import DefenderDynamicsModel


def test_model() -> None:
    """
    Loads a dynamics model and prints some statistics

    :return: None
    """
    # save_dynamics_model_dir = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v1/"
    save_dynamics_model_dir = "/home/kim/workspace/gym-optimal-intrusion-response/traces/"


    defender_dynamics_model = DefenderDynamicsModel()
    new_model = DefenderDynamicsModel()
    if save_dynamics_model_dir is not None:
        print("loading dynamics model")
        defender_dynamics_model.read_model(save_dynamics_model_dir, model_name="traces.json")
        new_model.read_model(save_dynamics_model_dir, model_name="traces.json")
        print("model loaded")

    print("normalizing model counts")
    new_model.normalize()
    print("model counts normalized")

    print("85:")
    print(new_model.norm_num_new_alerts[(85, '172.18.9.191')].mean())
    print(new_model.norm_num_new_alerts[(85, '172.18.9.191')].std())
    print(new_model.norm_num_new_alerts[(85, '172.18.9.191')].var())

    print("19:")
    print(new_model.norm_num_new_alerts[(19, '172.18.9.191')].mean())
    print(new_model.norm_num_new_alerts[(19, '172.18.9.191')].std())
    print(new_model.norm_num_new_alerts[(19, '172.18.9.191')].var())

    print("20:")
    print(new_model.norm_num_new_alerts[(20, '172.18.9.191')].mean())
    print(new_model.norm_num_new_alerts[(20, '172.18.9.191')].std())
    print(new_model.norm_num_new_alerts[(20, '172.18.9.191')].var())

    print("11:")
    print(new_model.norm_num_new_alerts[(11, '172.18.9.191')].mean())
    print(new_model.norm_num_new_alerts[(11, '172.18.9.191')].std())
    print(new_model.norm_num_new_alerts[(11, '172.18.9.191')].var())


# Script entrypoint
if __name__ == '__main__':
    test_model()