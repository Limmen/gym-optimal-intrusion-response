"""
Utility scripts for plotting
"""

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


def plot_alerts_threshold() -> None:
    """
    Plots alerts thresholds

    :return: None
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # env = gym.make("optimal-intrusion-response-v3")
    env = None
    # v3_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/v3_results/v3/results_1/data/1620801616.3180108_0_4575_policy_network.zip"
    # v2_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/v2_results/v2/results_2/data/1620760014.3964121_0_1150_policy_network.zip"
    v3_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/12_may/rl/v3_results/v3/results_1/data/1620801616.3180108_0_4575_policy_network.zip"
    v2_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/12_may/rl/v2_results/v2/results_2/data/1620760014.3964121_0_1150_policy_network.zip"
    model = initialize_model(env, v3_load_path, "cpu", None)
    model2 = initialize_model(env, v2_load_path, "cpu", None)
    num_alerts = np.arange(0, 200, 1)
    x = []
    y = []
    y2 = []

    for i in range(len(num_alerts)):
        state = np.array([num_alerts[i], num_alerts[i], 0])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                 deterministic=False,
                                                                  attacker=True, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        x.append(i*2)
        y.append(val)

        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
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
    plt.rcParams['axes.titlepad'] = 2
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 2
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams.update({'font.size': 8})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.85))

    # ylims = (0, 920)

    # Plot Avg Eval rewards Gensim
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax.plot(x,
            y, label=r"$\pi_{\theta}$ vs \textsc{StealthyAttacker}",
            ls='-', color=colors[0])
    ax.fill_between(x, y, y2,
                    alpha=0.35, color=colors[0])

    ax.plot(x,
            y2, label=r"$\pi_{\theta}$ vs \textsc{NoisyAttacker}",
            ls='-', color="r")
    ax.fill_between(x, y2, np.zeros(len(y2)),
                    alpha=0.35, color="r")

    # if plot_opt:
    ax.plot(x, [0.5] * len(x), label=r"0.5", color="black", linestyle="dashed")

    ax.set_title(r"$\pi_{\theta}(\text{stop}|a)$", fontsize=13.5)
    ax.set_xlabel(r"\# Alerts $a$", fontsize=12)
    ax.set_xlim(0, len(x)*2)
    ax.set_ylim(0, 1.1)

    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    xlab.set_size(12)
    ylab.set_size(12)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
    #           ncol=2, fancybox=True, shadow=True, fontsize=13.5)
    ax.legend(loc="lower right", fontsize=12)

    ttl = ax.title
    ttl.set_position([.5, 1.05])

    ax.tick_params(axis='both', which='major', labelsize=12, length=2.2, width=0.6)
    ax.tick_params(axis='both', which='minor', labelsize=12, length=2.2, width=0.6)

    fig.tight_layout()
    # plt.show()
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0.2)
    fig.savefig("threshold_alerts" + ".png", format="png", dpi=600)
    fig.savefig("threshold_alerts_22" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)


def plot_3d() -> None:
    """
    3d plot of empirical thresholds

    :return: None
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # env = gym.make("optimal-intrusion-response-v3")
    # v3_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/v3_results/v3/results_1/data/1620801616.3180108_0_4575_policy_network.zip"
    # v2_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/v2_results/v2/results_2/data/1620760014.3964121_0_1150_policy_network.zip"
    v3_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/12_may/rl/v3_results/v3/results_1/data/1620801616.3180108_0_4575_policy_network.zip"
    v2_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/12_may/rl/v2_results/v2/results_2/data/1620760014.3964121_0_1150_policy_network.zip"
    env=None
    model = initialize_model(env, v3_load_path, "cpu", None)
    num_severe_alerts_recent = np.arange(200, 0, -1)
    num_severe_alerts_total = np.arange(0, 200, 1)
    sev, warn = np.meshgrid(num_severe_alerts_recent, num_severe_alerts_total)
    action_val = action_pred_core_state_severe_warning(sev, warn, model, env)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 5
    plt.rcParams['xtick.major.pad'] = 0.05
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.2
    plt.rcParams['axes.linewidth'] = 0.2
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'}, figsize=(4.15, 2.9))
    ax.plot_surface(sev, warn, action_val, cmap='Blues', linewidth=0.3,
                    alpha=0.8, edgecolor='k', rstride=12, cstride=12)
    ax.set_title(r"$\pi_{\theta}(\text{stop} | w_a, s_a)$ vs \textsc{StealthyAttacker}", fontsize=13.5)
    # ax.set_xlabel(r"warn alerts $w_a$")
    # ax.set_ylabel(r"sev alerts $s_a$")
    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = 0
    ax.set_xticks(np.arange(0, 200 + 1, 50))
    ax.set_yticks(np.arange(0, 200+1, 50))
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_size(12)
    ylab.set_size(12)
    ax.tick_params(axis='both', which='major', labelsize=12, length=2.2, width=0.6)
    ax.tick_params(axis='both', which='minor', labelsize=12, length=2.2, width=0.6)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.55)
    # plt.show()
    # fig.savefig("alerts_stopping" + ".png", format="png", dpi=600)
    fig.savefig("alerts_stopping" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)


def plot_3d_2() -> None:
    """
    3d plot of empirical thresholds

    :return: None
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # env = gym.make("optimal-intrusion-response-v3")
    env=None
    # v2_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/v2_results/v2/results_2/data/1620760014.3964121_0_1150_policy_network.zip"
    v3_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/12_may/rl/v3_results/v3/results_1/data/1620801616.3180108_0_4575_policy_network.zip"
    v2_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/12_may/rl/v2_results/v2/results_2/data/1620760014.3964121_0_1150_policy_network.zip"
    model2 = initialize_model(env, v2_load_path, "cpu", None)
    num_severe_alerts_recent = np.arange(200, 0, -1)
    num_severe_alerts_total = np.arange(0, 200, 1)
    sev, warn = np.meshgrid(num_severe_alerts_recent, num_severe_alerts_total)
    action_val_2 = action_pred_core_state_severe_warning(sev, warn, model2, env)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 5
    plt.rcParams['xtick.major.pad'] = 0.05
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.2
    plt.rcParams['axes.linewidth'] = 0.2
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'}, figsize=(4.15, 2.9))
    ax.plot_surface(sev, warn, action_val_2, cmap='Reds', linewidth=0.3,
                    alpha=0.8, edgecolor='k', rstride=12, cstride=12)
    ax.set_title(r"$\pi_{\theta}(\text{stop} | w_a, s_a)$ vs \textsc{NoisyAttacker}", fontsize=13.5)
    # ax.set_xlabel(r"warn alerts $w_a$")
    # ax.set_ylabel(r"sev alerts $s_a$")
    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = 0
    ax.set_xticks(np.arange(0, 200 + 1, 50))
    ax.set_yticks(np.arange(0, 200+1, 50))
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_size(12)
    ylab.set_size(12)
    ax.tick_params(axis='both', which='major', labelsize=12, length=2.2, width=0.6)
    ax.tick_params(axis='both', which='minor', labelsize=12, length=2.2, width=0.6)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.55)
    fig.savefig("alerts_stopping_2" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)



def plot_logins_threshold() -> None:
    """
    Plots logins thresholds

    :return: None
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # env = gym.make("optimal-intrusion-response-v3")
    env = None
    v2_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/backup_pycr_cnsm_21_22_may/v2_results/results_1/data/1621504581.9928813_72899_400_policy_network.zip"
    v3_load_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/backup_pycr_cnsm_21_22_may/v3_results/results_1/data/1621513319.4798174_299_300_policy_network.zip"
    model = initialize_model(env, v2_load_path, "cpu", None)
    model2 = initialize_model(env, v3_load_path, "cpu", None)
    num_logins = np.arange(0, 100, 1)
    x = []
    y = []
    y2 = []


    for i in range(len(num_logins)):
        state = np.array([0, 0, 0, num_logins[i]])
        actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                 deterministic=False,
                                                                  attacker=False, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        x.append(i)
        y.append(val)

        state = np.array([0, 0, 0, num_logins[i]])
        actions, values, log_prob = model2.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
                                                                  deterministic=False,
                                                                  attacker=False, env=env, filter_illegal=False)
        if actions.item() == 1:
            val = math.exp(log_prob.item())
        else:
            val = 1 - math.exp(log_prob.item())
        if val > 0.2:
            val = 0.0
        y2.append(val)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 2
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 2
    plt.rcParams['axes.linewidth'] = 0.1
    fontsize=16.5
    labelsize=15
    plt.rcParams.update({'font.size': fontsize})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.2, 2.95))

    # ylims = (0, 920)

    # Plot Avg Eval rewards Gensim
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax.plot(x,
            y2, label=r"$\pi_{\theta}$ vs \textsc{StealthyAttacker}",
            ls='-', color=colors[0])
    ax.fill_between(x, y2, y,
                    alpha=0.35, color=colors[0])
    #
    ax.plot(x,
            y, label=r"$\pi_{\theta}$ vs \textsc{NoisyAttacker}",
            ls='-', color="r")
    # ax.fill_between(x, y2, np.zeros(len(y2)),
    #                 alpha=0.35, color="r")

    # if plot_opt:
    # ax.plot(x,
    #         [0.5] * len(x), label=r"0.5",
    #         color="black",
    #         linestyle="dashed")

    ax.set_title(r"$\pi_{\theta}(\text{stop}|l)$", fontsize=fontsize)
    ax.set_xlabel(r"\# Login attempts $l$", fontsize=fontsize)
    ax.set_xlim(0, len(x))
    ax.set_ylim(0, 0.15)

    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
    #           ncol=2, fancybox=True, shadow=True, fontsize=12)
    ax.legend(loc="lower right", fontsize=12)

    # ttl = ax.title
    # ttl.set_position([.5, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax.tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)

    fig.tight_layout()
    # plt.show()
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0.2)
    fig.savefig("logins_thresholds" + ".png", format="png", dpi=600)
    fig.savefig("logins_thresholds" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)


def action_pred_core_state_severe_warning(severe_alerts, warning_alerts, model, env):
    """
    Utility function for empirical threshold plots

    :param severe_alerts: number of severe alerts
    :param warning_alerts: number of warning alerts
    :param model: model to predict with
    :param env: the env
    :return: the predicted thresholds
    """
    z = []
    for i in range(len(severe_alerts)):
        z1 = []
        for j in range(len(severe_alerts[i])):
            state = np.array([severe_alerts[i][j],warning_alerts[i][j], 0])
            actions, values, log_prob = model.defender_policy.forward(torch.tensor(np.array([state])).to("cpu"),
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

# script entrypoint
if __name__ == '__main__':
    # model = initialize_model(env, load_path, "cuda:0", None)
    # plot_3d()
    # plot_logins_threshold()
    # plot_3d_2()
    # plot_alerts_threshold()
    plot_logins_threshold()
