"""
Plotting functions
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import gym_optimal_intrusion_response.constants.constants as constants
from gym_optimal_intrusion_response.logic.defender_dynamics.dp import DP
import math


def plot_thresholds() -> None:
    """
    Plots the thresholds

    :return: None
    """
    thresholds = DP.load_numpy("thresholds.npy")
    V = DP.load_numpy("value_fun.npy")
    state_to_id = DP.load_pickle("state_to_id.pkl")
    id_to_state = DP.load_pickle("id_to_state.pkl")
    policy = DP.load_numpy("policy.npy")

    # for i in range(len(thresholds[1:])):
    #     pass
    min_threshold = -100
    ts_to_thresholds = {}
    for i in range(len(thresholds)):
        s = id_to_state[i]
        if s != "terminal":
            t1, x1, y1, z1 = s
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
    x = []
    y = []
    for i in range(constants.DP2.MAX_TIMESTEPS):
        s = id_to_state[i]
        t1 = 100
        if s != "terminal":
            t1, x1, y1, z1 = s
        avg_threshold = np.max(np.array(ts_to_thresholds[i]))
        print("avg threshold:{}".format(avg_threshold))
        avg_threshold = avg_threshold-(i)
        print("avg threshold prime:{}".format(avg_threshold))

        if i == constants.DP2.MAX_TIMESTEPS-1:
            print(ts_to_thresholds[i])
        x.append(i)
        y.append(avg_threshold)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.8
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams.update({'font.size': 10})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3.2))

    # ylims = (0, 920)

    # Plot Avg Eval rewards Gensim
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax.plot(x[1:],
            y[1:], label=r"$\pi_{\theta}$ simulation",
            ls='-', color=colors[0])
    lower_bound = np.zeros(len(y[1:]))
    lower_bound.fill(min(y[1:]))
    ax.fill_between(x[1:], y[1:], lower_bound,
                    alpha=0.35, color=colors[0])

    ax.set_title(r"Stopping thresholds $\alpha_t$", fontsize=12.5)
    ax.set_xlabel(r"\# Time-step $t$", fontsize=11.5)

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xlim(x[1], len(x)-1)
    ax.set_ylim(min(y[1:]), max(y[1:])+45)
    a = ax.get_xticks().tolist()
    a[-1] = r'$T$'
    ax.set_xticklabels(a)
    ax.set_xticks([1.0, 5.0, 10.0, 15.0, 19.0])

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

    ttl = ax.title
    ttl.set_position([.5, 1.05])

    fig.tight_layout()
    # plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("threshold_alerts" + ".png", format="png", dpi=600)
    fig.savefig("threshold_alerts" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)


if __name__ == '__main__':
    plot_thresholds()