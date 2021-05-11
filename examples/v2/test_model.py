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


def plot_alerts_threshold():
    env = gym.make("optimal-intrusion-response-v2")
    load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v2/training/defender/ppo_baseline/results/data/1620725214.4430697_0_100_policy_network.zip"
    load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v2/training/defender/ppo_baseline/results/data/1620751383.982535_0_100_policy_network.zip"
    # load_path_2 = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v1/training/defender/results_backup2/data/1620469740.5786622_0_1850_policy_network.zip"
    model = initialize_model(env, load_path, "cuda:0", None)
    # model2 = initialize_model(env, load_path_2, "cuda:0", None)
    num_alerts = np.arange(0, 1000, 1)
    x = []
    y = []
    vals = []
    # for i in range(200):
    #     print("{}/{}".format(i, 200))
    #     for j in range(25):
    #         for k in range(25):
    #             actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([[i,j,k]])).to("cuda:0"),
    #                                                                       deterministic=False,
    #                                                                       attacker=False, env=env, filter_illegal=False)
    #             if actions.item() == 1:
    #                 val = math.exp(log_prob.item())
    #             else:
    #                 val = 1 - math.exp(log_prob.item())
    #             if val > 0.5:
    #                 print("val:{}, alerts:{}, t:{}, logins:{}".format(val, i, j, k))
    #             vals.append(val)
    #
    # print("max val:{}".format(max(vals)))



    for i in range(len(num_alerts)):
        state = np.array([num_alerts[i], 0])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        x.append(i)
        y.append(val)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.8
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams.update({'font.size': 10})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3.2))

    # ylims = (0, 920)

    # Plot Avg Eval rewards Gensim
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax.plot(x,
            y, label=r"$\pi_{\theta}$ simulation",
            ls='-', color=colors[0])
    ax.fill_between(x, y, np.zeros(len(y)),
                    alpha=0.35, color=colors[0])

    # if plot_opt:
    ax.plot(x,
            [0.5] * len(x), label=r"0.5",
            color="black",
            linestyle="dashed")

    ax.set_title(r"$\pi_{\theta^D}(\text{stop}|a)$", fontsize=12.5)
    ax.set_xlabel(r"\# Alerts $a$", fontsize=11.5)
    # ax.set_ylabel(r"$\mathbb{P}[\text{stop}|w]$", fontsize=12)
    ax.set_xlim(0, len(x))
    ax.set_ylim(0, 1.1)
    # ax.set_ylim(ylim_rew)

    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    xlab.set_size(11.5)
    ylab.set_size(11.5)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
    #           ncol=2, fancybox=True, shadow=True)
    # ax.legend(loc="lower right")
    # ax.xaxis.label.set_size(13.5)
    # ax.yaxis.label.set_size(13.5)

    ttl = ax.title
    ttl.set_position([.5, 1.05])

    fig.tight_layout()
    plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0)
    # fig.savefig("threshold_alerts" + ".png", format="png", dpi=600)
    # fig.savefig("threshold_alerts" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)



if __name__ == '__main__':
    plot_alerts_threshold()
