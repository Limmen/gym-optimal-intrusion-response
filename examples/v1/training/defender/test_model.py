from gym_optimal_intrusion_response.envs.derived_envs.optimal_intrusion_response_env_v1 import \
    OptimalIntrusionResponseEnvV1
import gym
from gym_optimal_intrusion_response.agents.policy_gradient.ppo_baseline.impl.ppo.ppo import PPO
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import gym_optimal_intrusion_response.constants.constants as constants
from gym_optimal_intrusion_response.logic.defender_dynamics.dp import DP


def initialize_model(env, load_path, device, agent_config) -> None:
    """
    Initialize models

    :return: None
    """
    # Initialize models
    model = PPO.load(path=load_path, env=env, load_path=load_path, device=device,
                     agent_config=agent_config)
    return model


def plot_policy():
    env = gym.make("optimal-intrusion-response-v1")
    load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v1/training/defender/results_backup/data/1620424642.9181776_0_3075_policy_network.zip"
    load_path_2 = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v1/training/defender/results_backup2/data/1620469740.5786622_0_1850_policy_network.zip"
    model = initialize_model(env, load_path, "cuda:0", None)
    model2 = initialize_model(env, load_path_2, "cuda:0", None)

    thresholds = DP.load_thresholds()
    id_to_state = DP.load_id_to_state()
    policy = DP.load_policy()

    ts_to_thresholds = {}
    for i in range(len(thresholds)):
        s = id_to_state[i]
        if s != "terminal":
            t1, x1 = s
            if x1 > 0.0 and policy[i][1] == 1:
                if t1 in ts_to_thresholds:
                    ts_to_thresholds[t1] = ts_to_thresholds[t1] + [thresholds[i]]
                else:
                    ts_to_thresholds[t1] = [thresholds[i]]
            else:
                if t1 in ts_to_thresholds:
                    if ts_to_thresholds[t1][0] < thresholds[i]:
                        ts_to_thresholds[t1][0] = thresholds[i]
                else:
                    ts_to_thresholds[t1] = [thresholds[i]]
    timesteps = []
    thresholds = []
    for i in range(constants.DP.MAX_TIMESTEPS):
        avg_threshold = np.max(np.array(ts_to_thresholds[i]))
        timesteps.append(i)
        thresholds.append(avg_threshold)

    fontsize = 8.5
    labelsize = 8
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 2.4
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams.update({'font.size': 6.5})
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(5, 4.5))

    x = []
    y = []
    y2 = []
    y3 = []
    t = 11
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
        state = np.array([t, i])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y3.append(val)

    data = np.array([y2, y3])
    means_y2 = np.mean(tuple(data), axis=0)
    stds_y2 = np.std(tuple(data), axis=0, ddof=1)

    ax[0][0].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color="r", alpha=1)

    ax[0][0].fill_between(x[0:], means_y2 - stds_y2, means_y2 + stds_y2, alpha=0.35, color="r")

    ax[0][0].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[0][0].set_xlim(-10, 10)

    ax[0][0].set_title(r"$t=11$", fontsize=fontsize)
    # ax[0][0].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # ax[0][0].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[0][0].xaxis.get_label()
    ylab = ax[0][0].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[0][0].spines['right'].set_color((.8, .8, .8))
    ax[0][0].spines['top'].set_color((.8, .8, .8))

    x = []
    y = []
    y2 = []
    y3 = []
    t = 12
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
        state = np.array([t, i])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                   deterministic=False,
                                                                   attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y3.append(val)

    data = np.array([y2, y3])
    means_y2 = np.mean(tuple(data), axis=0)
    stds_y2 = np.std(tuple(data), axis=0, ddof=1)

    ax[0][1].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color="r", alpha=1)

    ax[0][1].fill_between(x[0:], means_y2 - stds_y2, means_y2 + stds_y2, alpha=0.35, color="r")
    ax[0][1].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[0][1].set_xlim(-10, 10)

    ax[0][1].set_title(r"$t=12$", fontsize=fontsize)
    # ax[0][1].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # ax[0][1].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[0][1].xaxis.get_label()
    ylab = ax[0][1].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[0][1].spines['right'].set_color((.8, .8, .8))
    ax[0][1].spines['top'].set_color((.8, .8, .8))

    x = []
    y = []
    y2 = []
    t = 13
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
        state = np.array([t, i])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

    data = np.array([y2, y3])
    means_y2 = np.mean(tuple(data), axis=0)
    stds_y2 = np.std(tuple(data), axis=0, ddof=1)

    ax[0][2].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color="r", alpha=1)

    ax[0][2].fill_between(x[0:], means_y2 - stds_y2, means_y2 + stds_y2, alpha=0.35, color="r")
    ax[0][2].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[0][2].set_xlim(-10, 10)

    ax[0][2].set_title(r"$t=13$", fontsize=fontsize)
    # ax[0][2].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # ax[0][2].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[0][2].xaxis.get_label()
    ylab = ax[0][2].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[0][2].spines['right'].set_color((.8, .8, .8))
    ax[0][2].spines['top'].set_color((.8, .8, .8))

    x = []
    y = []
    y2 = []
    y3 = []
    t = 14
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
        state = np.array([t, i])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                   deterministic=False,
                                                                   attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y3.append(val)

    data = np.array([y2, y3])
    means_y2 = np.mean(tuple(data), axis=0)
    stds_y2 = np.std(tuple(data), axis=0, ddof=1)

    ax[1][0].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color="r", alpha=1)

    ax[1][0].fill_between(x[0:], means_y2 - stds_y2, means_y2 + stds_y2, alpha=0.35, color="r")
    ax[1][0].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[1][0].set_xlim(-10, 10)

    ax[1][0].set_title(r"$t=14$", fontsize=fontsize)
    # ax[1][0].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # ax[1][0].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[1][0].xaxis.get_label()
    ylab = ax[1][0].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[1][0].spines['right'].set_color((.8, .8, .8))
    ax[1][0].spines['top'].set_color((.8, .8, .8))

    x = []
    y = []
    y2 = []
    y3 = []
    t = 15
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
        state = np.array([t, i])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                   deterministic=False,
                                                                   attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y3.append(val)

    data = np.array([y2, y3])
    means_y2 = np.mean(tuple(data), axis=0)
    stds_y2 = np.std(tuple(data), axis=0, ddof=1)

    ax[1][1].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color="r", alpha=1)

    ax[1][1].fill_between(x[0:], means_y2 - stds_y2, means_y2 + stds_y2, alpha=0.35, color="r")
    ax[1][1].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[1][1].set_xlim(-10, 10)

    ax[1][1].set_title(r"$t=15$", fontsize=fontsize)
    # ax[1][1].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # ax[1][1].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[1][1].xaxis.get_label()
    ylab = ax[1][1].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[1][1].spines['right'].set_color((.8, .8, .8))
    ax[1][1].spines['top'].set_color((.8, .8, .8))

    x = []
    y = []
    y2 = []
    y3 = []
    t = 16
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
        state = np.array([t, i])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                   deterministic=False,
                                                                   attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y3.append(val)

    data = np.array([y2, y3])
    means_y2 = np.mean(tuple(data), axis=0)
    stds_y2 = np.std(tuple(data), axis=0, ddof=1)

    ax[1][2].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color="r", alpha=1)

    ax[1][2].fill_between(x[0:], means_y2 - stds_y2, means_y2 + stds_y2, alpha=0.35, color="r")
    ax[1][2].plot(x[0:],
                  y[0:], label=r"$Optimal \pi^{*}",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[1][2].set_xlim(-10, 10)

    ax[1][2].set_title(r"$t=16$", fontsize=fontsize)
    # ax[1][2].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # ax[1][2].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[1][2].xaxis.get_label()
    ylab = ax[1][2].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[1][2].spines['right'].set_color((.8, .8, .8))
    ax[1][2].spines['top'].set_color((.8, .8, .8))

    x = []
    y = []
    y2 = []
    y3 = []
    t = 17
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
        state = np.array([t, i])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                   deterministic=False,
                                                                   attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y3.append(val)

    data = np.array([y2, y3])
    means_y2 = np.mean(tuple(data), axis=0)
    stds_y2 = np.std(tuple(data), axis=0, ddof=1)

    ax[2][0].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color="r", alpha=1)

    ax[2][0].fill_between(x[0:], means_y2 - stds_y2, means_y2 + stds_y2, alpha=0.35, color="r")
    ax[2][0].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[2][0].set_xlim(-10, 10)

    ax[2][0].set_title(r"$t=17$", fontsize=fontsize)
    ax[2][0].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # ax[2][0].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[2][0].xaxis.get_label()
    ylab = ax[2][0].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[2][0].spines['right'].set_color((.8, .8, .8))
    ax[2][0].spines['top'].set_color((.8, .8, .8))

    x = []
    y = []
    y2 = []
    y3 = []
    t = 18
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
        state = np.array([t, i])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                   deterministic=False,
                                                                   attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y3.append(val)

    data = np.array([y2, y3])
    means_y2 = np.mean(tuple(data), axis=0)
    stds_y2 = np.std(tuple(data), axis=0, ddof=1)

    ax[2][1].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color="r", alpha=1)

    ax[2][1].fill_between(x[0:], means_y2 - stds_y2, means_y2 + stds_y2, alpha=0.35, color="r")
    ax[2][1].plot(x[0:],
                  y[0:], label=r"Optimal  $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[2][1].set_xlim(-10, 10)

    ax[2][1].set_title(r"$t=18$", fontsize=fontsize)
    ax[2][1].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # ax[2][1].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[2][1].xaxis.get_label()
    ylab = ax[2][1].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[2][1].spines['right'].set_color((.8, .8, .8))
    ax[2][1].spines['top'].set_color((.8, .8, .8))

    x = []
    y = []
    y2 = []
    y3 = []
    t = 19
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
        state = np.array([t, i])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                   deterministic=False,
                                                                   attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y3.append(val)

    data = np.array([y2, y3])
    means_y2 = np.mean(tuple(data), axis=0)
    stds_y2 = np.std(tuple(data), axis=0, ddof=1)

    ax[2][2].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color="r", alpha=1)

    ax[2][2].fill_between(x[0:], means_y2 - stds_y2, means_y2 + stds_y2, alpha=0.35, color="r")
    ax[2][2].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[2][2].set_xlim(-10, 10)
    # ax[2][2].set_ylim(0, 1)

    ax[2][2].set_title(r"$t=19$", fontsize=fontsize)
    ax[2][2].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # ax[2][2].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[2][2].xaxis.get_label()
    ylab = ax[2][2].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[2][2].spines['right'].set_color((.8, .8, .8))
    ax[2][2].spines['top'].set_color((.8, .8, .8))

    ax[2][2].set_yticks([])
    ax[2][1].set_yticks([])
    ax[1][2].set_yticks([])
    ax[1][1].set_yticks([])
    ax[0][2].set_yticks([])
    ax[0][1].set_yticks([])

    ax[0][0].set_ylabel(r"$\mathbb{P}[\text{stop}|c]$", fontsize=labelsize)
    ax[1][0].set_ylabel(r"$\mathbb{P}[\text{stop}|c]$", fontsize=labelsize)
    ax[2][0].set_ylabel(r"$\mathbb{P}[\text{stop}|c]$", fontsize=labelsize)

    ax[0][0].set_xticks([])
    ax[0][1].set_xticks([])
    ax[0][2].set_xticks([])
    ax[1][0].set_xticks([])
    ax[1][1].set_xticks([])
    ax[1][2].set_xticks([])

    handles, labels = ax[2][2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.51, 0.066),
               ncol=2, fancybox=True, shadow=True, fontsize=fontsize)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.115)
    # plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("opt_learned_policies" + ".png", format="png", dpi=600)
    fig.savefig("opt_learned_policies" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)


if __name__ == '__main__':
    plot_policy()
