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
    env = gym.make("optimal-intrusion-response-v3")
    # load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/training/defender/ppo_baseline/results/data/1620736046.5410578_0_50_policy_network.zip"
    load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/training/backup_results/data/1620736795.3647537_0_200_policy_network.zip"
    # load_path_2 = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v2/training/defender/ppo_baseline/results/data/1620749768.5217767_0_100_policy_network.zip"
    # load_path_2 = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v2/training/defender/ppo_baseline/results/data/1620751383.982535_0_100_policy_network.zip"
    load_path_2 = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v2/training/defender/ppo_baseline/results/data/1620752333.0477703_0_275_policy_network.zip"
    # load_path = load_path_2
    model = initialize_model(env, load_path, "cuda:0", None)
    model2 = initialize_model(env, load_path_2, "cuda:0", None)
    num_alerts = np.arange(0, 400, 1)
    x = []
    y = []
    y2 = []

    for i in range(len(num_alerts)):
        # state = np.array([num_alerts[i], 0])
        # state = np.array([num_alerts[i], num_alerts[i], num_alerts[i], num_alerts[i]*2 ,0])
        state = np.array([num_alerts[i], num_alerts[i], 0])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                 deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        x.append(i*2)
        y.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                  deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        y2.append(val)

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
            y, label=r"$\pi_{\theta}$ vs $B_1$",
            ls='-', color=colors[0])
    ax.fill_between(x, y, y2,
                    alpha=0.35, color=colors[0])

    ax.plot(x,
            y2, label=r"$\pi_{\theta}$ vs $B_2$",
            ls='-', color="r")
    ax.fill_between(x, y2, np.zeros(len(y2)),
                    alpha=0.35, color="r")

    # if plot_opt:
    ax.plot(x,
            [0.5] * len(x), label=r"0.5",
            color="black",
            linestyle="dashed")

    ax.set_title(r"$\pi_{\theta}(\text{stop}|a)$", fontsize=12.5)
    ax.set_xlabel(r"\# Alerts $a$", fontsize=11.5)
    # ax.set_ylabel(r"$\mathbb{P}[\text{stop}|w]$", fontsize=12)
    ax.set_xlim(0, len(x)*2)
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

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=2, fancybox=True, shadow=True)
    ax.legend(loc="lower right")
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


def plot_3d():
    env = gym.make("optimal-intrusion-response-v3")
    # load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/training/defender/ppo_baseline/results/data/1620736046.5410578_0_50_policy_network.zip"
    load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/training/backup_results/data/1620736795.3647537_0_200_policy_network.zip"
    # load_path_2 = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v2/training/defender/ppo_baseline/results/data/1620749768.5217767_0_100_policy_network.zip"
    # load_path_2 = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v2/training/defender/ppo_baseline/results/data/1620751383.982535_0_100_policy_network.zip"
    load_path_2 = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v2/training/defender/ppo_baseline/results/data/1620752333.0477703_0_275_policy_network.zip"
    # load_path = load_path_2
    model = initialize_model(env, load_path, "cuda:0", None)
    model2 = initialize_model(env, load_path_2, "cuda:0", None)


    # num_severe_alerts_recent = np.arange(0, 200, 1)
    num_severe_alerts_recent = np.arange(200, 0, -1)
    num_severe_alerts_total = np.arange(0, 200, 1)
    # num_severe_alerts_total = np.arange(200, 0, -1)
    sev, warn = np.meshgrid(num_severe_alerts_recent, num_severe_alerts_total)
    action_val = action_pred_core_state_severe_warning(sev, warn, model, env)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 8
    plt.rcParams['xtick.major.pad'] = 0.0
    plt.rcParams['ytick.major.pad'] = 0.0
    plt.rcParams['axes.labelpad'] = 0.0
    plt.rcParams['axes.linewidth'] = 0.05
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'figure.autolayout': True})

    #
    #fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'}, figsize=(4.5,3))

    ax.plot_surface(sev, warn, action_val, rstride=12, cstride=12,
                    cmap='cividis')

    ax.set_title(r"$\pi_{\theta^D}(\text{stop} | w_a, s_a)$", fontsize=12.5)
    ax.set_xlabel(r"warn alerts $w_a$")
    ax.set_ylabel(r"sev alerts $s_a$")
    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = 0
    ax.set_xticks(np.arange(0, 200 + 1, 50))
    ax.set_yticks(np.arange(0, 200+1, 50))
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_size(11.5)
    ylab.set_size(11.5)
    ax.tick_params(axis='both', which='major', labelsize=10, length=2.2, width=0.6)
    ax.tick_params(axis='both', which='minor', labelsize=10, length=2.2, width=0.6)
    # ax.set_ylim(200, 0)
    ax.set_xlim(200, 0)
    #plt.subplots_adjust(wspace=150, hspace=150, top=200, bottom=0.0)
    fig.tight_layout()
    #plt.subplots_adjust(wspace=150, hspace=150, top=200, bottom=0.0)
    fig.subplots_adjust(bottom=0.9)
    #plt.autoscale()
    #plt.subplots_adjust(bottom=0.55)
    plt.show()
    # fig.savefig("alerts_stopping" + ".png", format="png", dpi=600)
    # fig.savefig("alerts_stopping" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)



def action_pred_core_state_severe_warning(severe_alerts, warning_alerts, model, env):
    z = []
    for i in range(len(severe_alerts)):
        z1 = []
        for j in range(len(severe_alerts[i])):
            state = np.array([severe_alerts[i][j],warning_alerts[i][j], 0])
            actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cuda:0"),
                                                                      deterministic=False,
                                                                      attacker=True, env=env, filter_illegal=False)
            if actions.item() == 1:
                val = math.exp(log_prob.item())
            else:
                val = 1 - math.exp(log_prob.item())
            z1.append(val)
        z.append(z1)
    z = np.array(z)
    return z

if __name__ == '__main__':
    # env = gym.make("optimal-intrusion-response-v3")
    # load_path = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v3/training/defender/ppo_baseline/results/data/1620736046.5410578_0_50_policy_network.zip"
    # # load_path_2 = "/home/kim/workspace/gym-optimal-intrusion-response/examples/v1/training/defender/results_backup2/data/1620469740.5786622_0_1850_policy_network.zip"
    # model = initialize_model(env, load_path, "cuda:0", None)
    plot_3d()
    # plot_alerts_threshold()
