from gym_optimal_intrusion_response.envs.derived_envs.optimal_intrusion_response_env_v1 import \
    OptimalIntrusionResponseEnvV1
from gym_optimal_intrusion_response.agents.policy_gradient.ppo_baseline.impl.ppo.ppo import PPO
import torch
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os


def initialize_model(env, load_path, device, agent_config) -> None:
    """
    Initialize models

    :return: None
    """
    # Initialize models
    model = PPO.load(path=load_path, env=env, load_path=load_path, device=device,
                     agent_config=agent_config, map_location='cpu')
    return model

def plot_policy() -> None:
    """
    Plots the policy

    :return: None
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # env = gym.make("optimal-intrusion-response-v2")
    env = None
    # v2_load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/training/v2_results/results_1/data/1621504581.9928813_72899_400_policy_network.zip"
    # v3_load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/training/v3_results/results_1/data/1621513319.4798174_299_300_policy_network.zip"
    v2_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/backup_pycr_cnsm_21_22_may/v2_results/results_1/data/1621504581.9928813_72899_400_policy_network.zip"
    v3_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/backup_pycr_cnsm_21_22_may/v3_results/results_1/data/1621513319.4798174_299_300_policy_network.zip"
    model = initialize_model(env, v2_load_path, "cpu", None)

    fontsize = 10.5
    labelsize = 9.5
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 2.4
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams.update({'font.size': 6.5})
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(5.2, 3.4))
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]

    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    sev_alerts = [0,209,329,389,389,389,458,470,470,470,614,614,614,614,614]
    warn_alerts = [0,9,209,209, 210,212,312,312,312,314,514,514,514,514,514]
    logins = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    y = []
    y2 = []
    t = 1
    opt_t = 2
    for i in np.arange(0,15):
        # x.append(i)
        if i < opt_t-1:
            y.append(0)
        else:
            y.append(1)
        state = np.array([t, sev_alerts[i], warn_alerts[i], logins[i]])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)
    means_y2=y2

    ax[0][0].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color=colors[0], alpha=1)

    ax[0][0].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[0][0].set_xlim(0, 15)

    ax[0][0].set_title(r"$i_t=1$", fontsize=fontsize)

    # tweak the axis labels
    xlab = ax[0][0].xaxis.get_label()
    ylab = ax[0][0].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[0][0].spines['right'].set_color((.8, .8, .8))
    ax[0][0].spines['top'].set_color((.8, .8, .8))

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sev_alerts = [0, 0, 208, 274, 274, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330]
    warn_alerts = [0, 0, 15, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215]
    logins = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    y = []
    y2 = []
    t = 2
    opt_t = 3
    for i in np.arange(0, 15):

        if i < opt_t-1:
            y.append(0)
        else:
            y.append(1)
        state = np.array([t, sev_alerts[i], warn_alerts[i], logins[i]])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)
    means_y2 = y2

    ax[0][1].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$ vs \textsc{NoisyAttacker}",
                  ls='-', color=colors[0], alpha=1)

    ax[0][1].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[0][1].set_xlim(0, 15)

    ax[0][1].set_title(r"$i_t=2$", fontsize=fontsize)

    # tweak the axis labels
    xlab = ax[0][1].xaxis.get_label()
    ylab = ax[0][1].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[0][1].spines['right'].set_color((.8, .8, .8))
    ax[0][1].spines['top'].set_color((.8, .8, .8))

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sev_alerts = [0, 0, 0, 216, 356, 356, 416, 416, 416, 416, 416, 416, 416, 416, 416]
    warn_alerts = [0, 10, 20, 32, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232]
    logins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = []
    y2 = []
    t = 3
    opt_t = 4
    for i in np.arange(0, 15):

        if i < opt_t-1:
            y.append(0)
        else:
            y.append(1)
        state = np.array([t, sev_alerts[i], warn_alerts[i], logins[i]])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)
    means_y2 = y2

    ax[0][2].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color=colors[0], alpha=1)

    ax[0][2].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[0][2].set_xlim(0, 15)

    ax[0][2].set_title(r"$i_t=3$", fontsize=fontsize)

    # tweak the axis labels
    xlab = ax[0][2].xaxis.get_label()
    ylab = ax[0][2].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[0][2].spines['right'].set_color((.8, .8, .8))
    ax[0][2].spines['top'].set_color((.8, .8, .8))

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sev_alerts = [0, 0, 0, 0, 216, 336, 349, 409, 409, 409, 409, 409, 409, 409, 409]
    warn_alerts = [0, 12, 12, 12, 18, 88, 88, 93, 93, 93, 93, 93, 93, 93, 93]
    logins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = []
    y2 = []
    t = 4
    opt_t = 5
    for i in np.arange(0, 15):

        if i < opt_t-1:
            y.append(0)
        else:
            y.append(1)
        state = np.array([t, sev_alerts[i], warn_alerts[i], logins[i]])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)
    means_y2 = y2

    ax[0][3].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color=colors[0], alpha=1)

    ax[0][3].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[0][3].set_xlim(0, 15)

    ax[0][3].set_title(r"$i_t=4$", fontsize=fontsize)

    # tweak the axis labels
    xlab = ax[0][3].xaxis.get_label()
    ylab = ax[0][3].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[0][3].spines['right'].set_color((.8, .8, .8))
    ax[0][3].spines['top'].set_color((.8, .8, .8))

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sev_alerts = [0, 0, 2, 4, 5, 239, 305, 305, 325, 325, 325, 325, 325, 325, 325]
    warn_alerts = [0, 6, 11, 11, 11, 49, 118, 118, 120, 120, 120, 120, 120, 120, 120]
    logins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = []
    y2 = []
    t = 5
    opt_t = 6
    for i in np.arange(0, 15):

        if i < opt_t - 1:
            y.append(0)
        else:
            y.append(1)
        state = np.array([t, sev_alerts[i], warn_alerts[i], logins[i]])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)
    means_y2 = y2

    ax[1][0].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color=colors[0], alpha=1)

    ax[1][0].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[1][0].set_xlim(0, 15)

    ax[1][0].set_title(r"$i_t=5$", fontsize=fontsize)
    ax[1][0].set_xlabel(r"Time-step $t$", fontsize=labelsize)

    # tweak the axis labels
    xlab = ax[1][0].xaxis.get_label()
    ylab = ax[1][0].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[1][0].spines['right'].set_color((.8, .8, .8))
    ax[1][0].spines['top'].set_color((.8, .8, .8))

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sev_alerts = [0, 0, 0, 0, 0, 1, 215, 305, 305, 305, 305, 305, 305, 305, 305]
    warn_alerts = [0, 1, 2, 2, 3, 5, 10, 210, 210, 210, 210, 210, 210, 210, 210]
    logins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = []
    y2 = []
    t = 6
    opt_t = 7
    for i in np.arange(0, 15):

        if i < opt_t-1:
            y.append(0)
        else:
            y.append(1)
        state = np.array([t, sev_alerts[i], warn_alerts[i], logins[i]])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)
    means_y2 = y2

    ax[1][1].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color=colors[0], alpha=1)

    ax[1][1].plot(x[0:],
                  y[0:], label=r"$Optimal \pi^{*}",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[1][1].set_xlim(0, 15)

    ax[1][1].set_title(r"$i_t=6$", fontsize=fontsize)
    ax[1][1].set_xlabel(r"Time-step $t$", fontsize=labelsize)

    # tweak the axis labels
    xlab = ax[1][1].xaxis.get_label()
    ylab = ax[1][1].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[1][1].spines['right'].set_color((.8, .8, .8))
    ax[1][1].spines['top'].set_color((.8, .8, .8))

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sev_alerts = [0, 0, 0, 0, 1, 1, 3, 214, 364, 364, 364, 364, 364, 364, 364]
    warn_alerts = [0, 2, 2, 3, 15, 15, 19, 32, 232, 232, 232, 232, 232, 232, 232]
    logins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = []
    y2 = []
    t = 7
    opt_t = 8
    for i in np.arange(0, 15):

        if i < opt_t-1:
            y.append(0)
        else:
            y.append(1)
        state = np.array([t, sev_alerts[i], warn_alerts[i], logins[i]])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)
    means_y2 = y2

    ax[1][2].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color=colors[0], alpha=1)

    ax[1][2].plot(x[0:],
                  y[0:], label=r"Optimal $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[1][2].set_xlim(0, 15)

    ax[1][2].set_title(r"$i_t=7$", fontsize=fontsize)
    ax[1][2].set_xlabel(r"Time-step $t$", fontsize=labelsize)
    # ax[2][0].grid('on')
    # ax[0][0].set_xlim(-1, 1)

    # tweak the axis labels
    xlab = ax[1][2].xaxis.get_label()
    ylab = ax[1][2].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[1][2].spines['right'].set_color((.8, .8, .8))
    ax[1][2].spines['top'].set_color((.8, .8, .8))

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sev_alerts = [0, 0, 0, 0, 1, 1, 3, 4, 214, 364, 364, 364, 364, 364, 364]
    warn_alerts = [0, 2, 2, 3, 15, 15, 19, 32, 32, 232, 232, 232, 232, 232, 232]
    logins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = []
    y2 = []
    t = 8
    opt_t = 9
    for i in np.arange(0, 15):

        if i < opt_t-1:
            y.append(0)
        else:
            y.append(1)
        state = np.array([t, sev_alerts[i], warn_alerts[i], logins[i]])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)
    means_y2 = y2

    ax[1][3].plot(x[0:],
                  means_y2[0:], label=r"Learned $\pi_{\theta}$",
                  ls='-', color=colors[0], alpha=1)

    ax[1][3].plot(x[0:],
                  y[0:], label=r"Optimal  $\pi^{*}$",
                  ls='-', color="black", alpha=1, linestyle="dashed")
    ax[1][3].set_xlim(0, 15)

    ax[1][3].set_title(r"$i_t=8$", fontsize=fontsize)
    ax[1][3].set_xlabel(r"Time-step $t$", fontsize=labelsize)

    # tweak the axis labels
    xlab = ax[1][3].xaxis.get_label()
    ylab = ax[1][3].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[1][3].spines['right'].set_color((.8, .8, .8))
    ax[1][3].spines['top'].set_color((.8, .8, .8))

    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # sev_alerts = [0, 209, 329, 389, 389, 389, 458, 470, 470, 470, 614, 614, 614, 614, 614]
    # warn_alerts = [0, 9, 209, 209, 210, 212, 312, 312, 312, 314, 514, 514, 514, 514, 514]
    # logins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # y = []
    # y2 = []
    # t = 8
    # opt_t = 2
    # for i in np.arange(0, 15):
    #
    #     if i < opt_t-1:
    #         y.append(0)
    #     else:
    #         y.append(1)
    #     state = np.array([t, sev_alerts[i], warn_alerts[i], logins[i]])
    #     actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
    #                                                               deterministic=False,
    #                                                               attacker=True, env=env, filter_illegal=False)
    #     if actions.item() == 1:
    #         val = math.exp(log_prob.item())
    #     else:
    #         val = 1 - math.exp(log_prob.item())
    #     y2.append(val)
    # means_y2 = y2
    #
    # ax[2][2].plot(x[0:],
    #               means_y2[0:], label=r"Learned $\pi_{\theta}$",
    #               ls='-', color=colors[0], alpha=1)
    #
    # ax[2][2].plot(x[0:],
    #               y[0:], label=r"Optimal $\pi^{*}$",
    #               ls='-', color="black", alpha=1, linestyle="dashed")
    # ax[2][2].set_xlim(0, 15)
    # # ax[2][2].set_ylim(0, 1)
    #
    # ax[2][2].set_title(r"$t=9$", fontsize=fontsize)
    # ax[2][2].set_xlabel(r"TTC $c$", fontsize=labelsize)
    # # ax[2][2].grid('on')
    # # ax[0][0].set_xlim(-1, 1)
    #
    # # tweak the axis labels
    # xlab = ax[2][2].xaxis.get_label()
    # ylab = ax[2][2].yaxis.get_label()
    #
    # xlab.set_size(labelsize)
    # ylab.set_size(labelsize)
    #
    # # change the color of the top and right spines to opaque gray
    # ax[2][2].spines['right'].set_color((.8, .8, .8))
    # ax[2][2].spines['top'].set_color((.8, .8, .8))
    #
    # ax[2][2].set_yticks([])
    # ax[2][1].set_yticks([])
    ax[1][3].set_yticks([])
    ax[1][2].set_yticks([])
    ax[1][1].set_yticks([])
    ax[0][2].set_yticks([])
    ax[0][1].set_yticks([])
    ax[0][3].set_yticks([])
    #
    ax[0][0].set_ylabel(r"$\mathbb{P}[\text{stop}|o_t]$", fontsize=labelsize)
    ax[1][0].set_ylabel(r"$\mathbb{P}[\text{stop}|o_t]$", fontsize=labelsize)
    # ax[2][0].set_ylabel(r"$\mathbb{P}[\text{stop}|c]$", fontsize=labelsize)
    #
    ax[0][0].set_xticks([])
    ax[0][1].set_xticks([])
    ax[0][2].set_xticks([])
    ax[0][3].set_xticks([])
    # ax[1][0].set_xticks([])
    # ax[1][1].set_xticks([])
    # ax[1][2].set_xticks([])

    handles, labels = ax[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.51, 0.096),
               ncol=2, fancybox=True, shadow=True, fontsize=labelsize)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.15, bottom=0.17)
    # plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("opt_learned_policies" + ".png", format="png", dpi=600)
    fig.savefig("opt_learned_policies" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)

# Script entrypoint
if __name__ == '__main__':
    plot_policy()