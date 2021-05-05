import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym_optimal_intrusion_response.logic.defender_dynamics.defender_dynamics import DefenderDynamics
import gym_optimal_intrusion_response.constants.constants as constants

def plot_alers_logins_distributions():
    f1_a = DefenderDynamics.f1_a()
    f1_b = DefenderDynamics.f1_b()
    f2_a = DefenderDynamics.f2_a()
    f2_b = DefenderDynamics.f2_b()

    fontsize = 6.5
    labelsize=6

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.8
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams.update({'font.size': fontsize})


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4.2, 1.85))
    x1 = np.arange(0, constants.DP.MAX_ALERTS, 1)
    x1 = np.array(list(filter(lambda k: f1_a.pmf(k) > 0.01, x1.tolist())))
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[0].plot(x1, f1_a.pmf(x1), 'ro', ms=2.5, mec=colors[0], color=colors[0])
    ax[0].vlines(x1, 0, f1_a.pmf(x1), label=r"$f_1^A$",
            ls='-', color=colors[0], lw=1)
    x2 = np.arange(0, constants.DP.MAX_ALERTS, 1)
    x2 = np.array(list(filter(lambda k: f1_b.pmf(k) > 0.01, x2.tolist())))
    ax[0].plot(x2, f1_b.pmf(x2), 'ro', ms=2.5, mec="darkred", color="darkred")
    ax[0].vlines(x2, 0, f1_b.pmf(x2), label=r"$f_1^B$",
                  ls='-', color="darkred", lw=1)

    ax[0].set_title(r"Alerts PMFs $f_1^{A}(x), f_1^{B}(x)$", fontsize=fontsize)
    ax[0].set_xlabel(r"\# Alerts $x$", fontsize=labelsize)
    # ax.set_ylabel(r"$\mathbb{P}[\text{stop}|w]$", fontsize=12)
    #ax.set_xlim(0, len(x))
    #ax[0].set_ylim(0, 0.3)
    # ax[0][0].set_xlim(6, 18)
    ax[0].grid('on')
    xlab = ax[0].xaxis.get_label()
    ylab = ax[0].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(labelsize)
    ax[0].spines['right'].set_color((.8, .8, .8))
    ax[0].spines['top'].set_color((.8, .8, .8))
    ax[0].legend(loc='right', ncol=1)

    x3 = np.arange(0, constants.DP.MAX_LOGINS, 1)
    x3 = np.array(list(filter(lambda k: f2_a.pmf(k) > 0.01, x3.tolist())))
    ax[1].plot(x3, f2_a.pmf(x3), 'ro', ms=2.5, mec=colors[0], color=colors[0])
    ax[1].vlines(x3, 0, f2_a.pmf(x3), label=r"$f_2^A$",
                 ls='-', color=colors[0], lw=1)

    x4 = np.arange(0, constants.DP.MAX_LOGINS, 1)
    x4 = np.array(list(filter(lambda k: f2_b.pmf(k) > 0.01, x4.tolist())))
    ax[1].plot(x4, f2_b.pmf(x4), 'ro', ms=2.5, mec="darkred", color="darkred")
    ax[1].vlines(x4, 0, f2_b.pmf(x4), label=r"$f_2^B$",
                 ls='-', color="darkred", lw=1)

    ax[1].set_title(r"Login Attempts PMFs $f_2^{A}(y), f_2^{B}(y)$", fontsize=fontsize)
    ax[1].set_xlabel(r"\# Login attempts $y$", fontsize=labelsize)
    # ax.set_ylabel(r"$\mathbb{P}[\text{stop}|w]$", fontsize=12)
    # ax.set_xlim(0, len(x))
    #ax[1].set_ylim(0, 1.1)
    #ax[0][1].set_xlim(6, 18)
    ax[1].grid('on')
    xlab = ax[1].xaxis.get_label()
    ylab = ax[1].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(labelsize)
    ax[1].spines['right'].set_color((.8, .8, .8))
    ax[1].spines['top'].set_color((.8, .8, .8))
    ax[1].legend(loc='right', ncol=1)

    # ax = fig.add_subplot(2, 2, 3, projection='3d')

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
    #           ncol=2, fancybox=True, shadow=True)
    # ax.legend(loc="lower right")
    # ax.xaxis.label.set_size(13.5)
    # ax.yaxis.label.set_size(13.5)

    # ttl = ax[0].title
    # ttl.set_position([.5, 1.05])

    fig.tight_layout()
    # plt.show()
    plt.subplots_adjust(wspace=0.18)
    fig.savefig("alerts_logins_distributions" + ".png", format="png", dpi=600)
    fig.savefig("alerts_logins_distributions" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)

def ttc_plot_helper(v, s, max_v):
    z = []
    for i in range(len(v)):
        z1 = []
        for j in range(len(v[i])):
            #prob = intrusion_prob_dist.cdf([alerts[i][j], logins[i][j]])
            val = DefenderDynamics.ttc(v[i][j], s[i,j], max_v)
            z1.append(val)
        z.append(z1)
    z = np.array(z)
    return z


def hack_prob_helper(ttc_val, t):
    z = []
    for i in range(len(ttc_val)):
        z1 = []
        for j in range(len(ttc_val[i])):
            val = DefenderDynamics.hack_prob(ttc_val[i][j], t[i][j])
            z1.append(val)
        z.append(z1)
    z = np.array(z)
    return z


def plot_ttc():
    v = np.arange(1, constants.DP.MAX_ALERTS, 1)
    s = np.arange(1, constants.DP.MAX_LOGINS, 1)
    x,y = np.meshgrid(v,  s)
    z = ttc_plot_helper(x,y,constants.DP.MAX_ALERTS)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 15
    plt.rcParams['xtick.major.pad'] = 0.05
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.2
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams.update({'font.size': 6.5})
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'})
    ax.plot_surface(x, y, z, cmap='viridis_r', linewidth=0.3,
                    alpha=0.8, edgecolor='k')

    ax.set_title(r"$TTC(a,l," + str(constants.DP.MAX_ALERTS) + ")$", fontsize=14)
    ax.set_xlabel(r"IDS alerts $a$")
    ax.set_ylabel(r"Login attempts $l$")
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_size(12)
    ylab.set_size(12)
    # ax.tick_params(axis='both', which='major', labelsize=10, length=2.2, width=0.6)
    # ax.tick_params(axis='both', which='minor', labelsize=10, length=2.2, width=0.6)
    #ax.set_yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0])
    #ax.set_yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #plt.yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.ylim(1.0, 0.0)
    plt.ylim(constants.DP.MAX_LOGINS, 0.0)
    fig.tight_layout()
    #plt.show()
    #plt.subplots_adjust(wspace=0, hspace=0, top=0.2)
    fig.savefig("ttc_alerts_logins" + ".png", format="png", dpi=600)
    fig.savefig("ttc_alerts_logins" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)

def plot_hp():
    t = np.arange(1, constants.DP.MAX_TIMESTEPS,1)
    z = np.arange(1, constants.DP.MAX_TTC, 1)
    print("z:{}".format(len(z)))
    print("z:{}".format(z.shape))
    x2,y2 = np.meshgrid(z, t)
    z2 = hack_prob_helper(x2, y2)


    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 15
    plt.rcParams['xtick.major.pad'] = 0.05
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.2
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams.update({'font.size': 6.5})
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'})
    ax.plot_surface(x2, y2, z2, cmap='viridis_r', linewidth=0.3,
                    alpha=0.8, edgecolor='k')

    ax.set_title(r"Probability of intrusion $\phi(x_t, y_t, t)$", fontsize=14)
    ax.set_xlabel(r"$TTC(x_t, y_t)$")
    ax.set_ylabel(r"Time $t$")
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_size(12)
    ylab.set_size(12)
    # ax.tick_params(axis='both', which='major', labelsize=10, length=2.2, width=0.6)
    # ax.tick_params(axis='both', which='minor', labelsize=10, length=2.2, width=0.6)
    #ax.set_yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0])
    #ax.set_yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #plt.yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #plt.ylim(1.0, 0.0)
    plt.xlim(constants.DP.MAX_TTC, 0.0)
    fig.tight_layout()
    #plt.show()
    #plt.subplots_adjust(wspace=0, hspace=0, top=0.2)
    fig.savefig("intrusion_prob" + ".png", format="png", dpi=600)
    fig.savefig("intrusion_prob" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)


if __name__ == '__main__':
    #test()
    # plot_ttc()
    # plot_hp()
    plot_alers_logins_distributions()
    #plot_intrusion_dist()