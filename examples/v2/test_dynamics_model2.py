from gym_pycr_ctf.envs.derived_envs.level1.simulation.pycr_ctf_level1_sim_env import PyCRCTFLevel1Sim1Env
from gym_pycr_ctf.envs.derived_envs.level1.emulation.pycr_ctf_level1_emulation_env import PyCRCTFLevel1Emulation1Env
from gym_pycr_ctf.dao.network.emulation_config import EmulationConfig
from gym_pycr_ctf.dao.defender_dynamics.defender_dynamics_model import DefenderDynamicsModel
import gym_pycr_ctf.constants.constants as constants
import gym
import time
import numpy as np
import os
import json

def test_env(env_name : str, num_steps : int):
    # emulation_config = EmulationConfig(server_ip="172.31.212.91", agent_ip="172.18.4.191",
    #                                agent_username="agent", agent_pw="agent", server_connection=True,
    #                                server_private_key_file="/Users/kimham/.ssh/pycr_id_rsa",
    #                                server_username="kim")
    emulation_config = EmulationConfig(server_ip="172.31.212.92", agent_ip="172.18.9.191",
                                   agent_username="agent", agent_pw="agent", server_connection=True,
                                   server_private_key_file="/home/kim/.ssh/id_rsa",
                                   server_username="kim")
    # emulation_config = EmulationConfig(agent_ip="172.18.9.191", agent_username="agent", agent_pw="agent",
    #                                  server_connection=False)
    # save_dynamics_model_dir = "/home/kim/storage/workspace/pycr/python-envs/minigames/network_intrusion/ctf/" \
    #                                            "gym-pycr-ctf/examples/difficulty_level_9/hello_world/"
    save_dynamics_model_dir = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v2"
    #env = gym.make(env_name, env_config=None, emulation_config=emulation_config)

    defender_dynamics_model = DefenderDynamicsModel()
    new_model = DefenderDynamicsModel()
    if save_dynamics_model_dir is not None:
        print("loading dynamics model")
        defender_dynamics_model.read_model(save_dynamics_model_dir, model_name="new_defender_dynamics_model_2.json")
        new_model.read_model(save_dynamics_model_dir, model_name="new_defender_dynamics_model_2.json")
        print("model loaded")
        # if os.path.exists(load_dir):
        #     env.env_config.network_conf = \
        #         env.env_config.network_conf.load(load_dir)


    for k,v in defender_dynamics_model.num_new_alerts.items():
        for k2,v2 in v.items():
            for k3,v3 in v2.items():
                if int(k) == 19 and int(k3) > 0:
                    val = int(int(k3)/20 + np.random.randint(0, 10))
                    if str(20) not in new_model.num_new_alerts:
                        new_model.num_new_alerts[str(20)] = {}
                    if k2 not in new_model.num_new_alerts[str(20)]:
                        new_model.num_new_alerts[str(20)][k2] = {}
                    new_model.num_new_alerts[str(20)][k2][str(val)] = int(new_model.num_new_alerts[str(19)][k2][k3])
                    # if k3 < 150:
                    #     new_model.num_new_alerts[str(19)][k2][str(val)] = int(new_model.num_new_alerts[str(19)][k2][k3])

    for k, v in defender_dynamics_model.num_new_priority.items():
        for k2, v2 in v.items():
            # print("action:{}, state:{}".format(k, k2))
            for k3, v3 in v2.items():
                if int(k) == 19 and int(k3) > 0:
                    val = int(int(k3) /20 + np.random.randint(0, 10))
                    if str(20) not in new_model.num_new_priority:
                        new_model.num_new_priority[str(20)] = {}
                    if k2 not in new_model.num_new_priority[str(20)]:
                        new_model.num_new_priority[str(20)][k2] = {}
                    new_model.num_new_priority[str(20)][k2][str(val)] = int(new_model.num_new_priority[str(19)][k2][k3])

    for k, v in defender_dynamics_model.num_new_severe_alerts.items():
        for k2, v2 in v.items():
            # print("action:{}, state:{}".format(k, k2))
            for k3, v3 in v2.items():
                if int(k) == 19:
                    del new_model.num_new_severe_alerts[k][k2][k3]

    for k, v in defender_dynamics_model.num_new_severe_alerts.items():
        for k2, v2 in v.items():
            # print("action:{}, state:{}".format(k, k2))
            for k3, v3 in v2.items():
                if int(k) == 20:
                    # val = int(int(k3) + 0 + np.random.randint(0, 10))
                    val = int(k3) + 200
                    if str(19) not in new_model.num_new_severe_alerts:
                        new_model.num_new_severe_alerts[str(19)] = {}
                    if k2 not in new_model.num_new_severe_alerts[str(19)]:
                        new_model.num_new_severe_alerts[str(19)][k2] = {}
                    new_model.num_new_severe_alerts[str(19)][k2][str(val)] = int(new_model.num_new_severe_alerts[str(20)][k2][k3])


    for k, v in defender_dynamics_model.num_new_warning_alerts.items():
        for k2, v2 in v.items():
            # print("action:{}, state:{}".format(k, k2))
            for k3, v3 in v2.items():
                if int(k) == 19:
                    del new_model.num_new_warning_alerts[k][k2][k3]

    for k, v in defender_dynamics_model.num_new_warning_alerts.items():
        for k2, v2 in v.items():
            # print("action:{}, state:{}".format(k, k2))
            for k3, v3 in v2.items():
                if int(k) == 20:
                    # val = int(int(k3) + 0 + np.random.randint(0, 10))
                    val = int(k3) + 0
                    if str(19) not in new_model.num_new_warning_alerts:
                        new_model.num_new_warning_alerts[str(19)] = {}
                    if k2 not in new_model.num_new_warning_alerts[str(19)]:
                        new_model.num_new_warning_alerts[str(19)][k2] = {}
                    new_model.num_new_warning_alerts[str(19)][k2][str(val)] = int(new_model.num_new_warning_alerts[str(20)][k2][k3])

    print("normalizing model counts")
    new_model.normalize()
    print("model counts normalized")
    # print(defender_dynamics_model.norm_num_new_alerts)
    print("85:")
    print(new_model.norm_num_new_severe_alerts[(85, '172.18.9.191')].mean())
    print(new_model.norm_num_new_severe_alerts[(85, '172.18.9.191')].std())
    print(new_model.norm_num_new_severe_alerts[(85, '172.18.9.191')].var())

    print("19:")
    print(new_model.norm_num_new_severe_alerts[(19, '172.18.9.191')].mean())
    print(new_model.norm_num_new_severe_alerts[(19, '172.18.9.191')].std())
    print(new_model.norm_num_new_severe_alerts[(19, '172.18.9.191')].var())
    print("19-2:")
    print(new_model.norm_num_new_warning_alerts[(19, '172.18.9.191')].mean())
    print(new_model.norm_num_new_warning_alerts[(19, '172.18.9.191')].std())
    print(new_model.norm_num_new_warning_alerts[(19, '172.18.9.191')].var())

    print("20:")
    print(new_model.norm_num_new_severe_alerts[(20, '172.18.9.191')].mean())
    print(new_model.norm_num_new_severe_alerts[(20, '172.18.9.191')].std())
    print(new_model.norm_num_new_severe_alerts[(20, '172.18.9.191')].var())

    print("20 2:")
    print(new_model.norm_num_new_warning_alerts[(20, '172.18.9.191')].mean())
    print(new_model.norm_num_new_warning_alerts[(20, '172.18.9.191')].std())
    print(new_model.norm_num_new_warning_alerts[(20, '172.18.9.191')].var())

    print("11:")
    print(new_model.norm_num_new_severe_alerts[(11, '172.18.9.191')].mean())
    print(new_model.norm_num_new_severe_alerts[(11, '172.18.9.191')].std())
    print(new_model.norm_num_new_severe_alerts[(11, '172.18.9.191')].var())

    # for k,v in new_model.machines_dynamics_model.items():
    #     if k in ips:
    #         print(k)
    #         print(v.norm_num_new_failed_login_attempts[(85, "172.18.9.191")].mean())

    # print("12:")
    # print(defender_dynamics_model.norm_num_new_alerts[(12, '172.18.9.191')].mean())
    # print(defender_dynamics_model.norm_num_new_alerts[(12, '172.18.9.191')].std())
    # print(defender_dynamics_model.norm_num_new_alerts[(12, '172.18.9.191')].var())


    # env.reset()
    # env.close()
    d = new_model.to_dict()
    with open("new_defender_dynamics_model_21.json", 'w') as fp:
        json.dump(d, fp)


def test_all():
    test_env("pycr-ctf-level-9-emulation-v5", num_steps=1000000000)

if __name__ == '__main__':
    test_all()