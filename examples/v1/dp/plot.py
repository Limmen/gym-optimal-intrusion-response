import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import gym_optimal_intrusion_response.constants.constants as constants


def Q_helper(ttcs, ts, policy, state_to_id, action):
    z = []
    for i in range(len(ttcs)):
        z1 = []
        for j in range(len(ttcs[i])):
            id = state_to_id[(ts[i, j], ttcs[i][j])]
            val = policy[id][2 + action]
            z1.append(val)
        z.append(z1)
    z = np.array(z)
    return z

def action_helper(ttcs, ts, policy, state_to_id):
    z = []
    for i in range(len(ttcs)):
        z1 = []
        for j in range(len(ttcs[i])):
            id = state_to_id[(ts[i, j], ttcs[i][j])]
            if policy[id][0] == 1:
                val = 0
            else:
                val = 1
            z1.append(val)
        z.append(z1)
    z = np.array(z)
    return z

def value_fun_helper(ttcs, ts, V, state_to_id):
    z = []
    for i in range(len(ttcs)):
        z1 = []
        for j in range(len(ttcs[i])):
            id = state_to_id[(ts[i, j], ttcs[i][j], )]
            val = V[id]
            z1.append(val)
        z.append(z1)
    z = np.array(z)
    return z

def load_state_to_id():
    print("Loading state_to_id table")
    with open("state_to_id.json", 'rb') as fp:
        state_to_id = pickle.load(fp)
    print("state_to_id loaded:{}".format(len(state_to_id)))
    return state_to_id

def load_id_to_state():
    print("Loading id_to_state table")
    with open("id_to_state.json", 'rb') as fp:
        id_to_state = pickle.load(fp)
    print("id_to_state loaded:{}".format(len(id_to_state)))
    return id_to_state

def load_value_fun():
    print("loading value function..")
    with open('value_fun.npy', 'rb') as f:
        V = np.load(f)
        print("value function loaded:{}".format(V.shape))
        return V

def load_policy():
    print("loading policy..")
    with open('policy.npy', 'rb') as f:
        policy = np.load(f)
        print("policy loaded:{}".format(policy.shape))
        return policy

def load_thresholds():
    print("loading thresholds..")
    with open('thresholds.npy', 'rb') as f:
        thresholds = np.load(f)
        print("thresholds loaded:{}".format(thresholds.shape))
        return thresholds

def load_TTC_table():
    print("loading TTC table..")
    with open('ttc_table.npy', 'rb') as f:
        TTC = np.load(f)
        print("TTC table loaded:{}".format(TTC.shape))
        return TTC

def plot_value_fun_3d(t : int = 0):
    V = load_value_fun()
    state_to_id = load_state_to_id()
    id_to_state = load_id_to_state()
    policy = load_policy()

    ttcs = np.arange(1, constants.DP.MAX_TTC, 1)
    ts = np.arange(1, constants.DP.MAX_TIMESTEPS, 1)
    x, y = np.meshgrid(ttcs, ts)
    z = value_fun_helper(x, y, V, state_to_id)

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

    ax.set_title(r"$V(("+ str(t) + ", x, y))$", fontsize=14)
    ax.set_xlabel(r"TTC")
    ax.set_ylabel(r"Time-step $t$")
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_size(12)
    ylab.set_size(12)
    # ax.tick_params(axis='both', which='major', labelsize=10, length=2.2, width=0.6)
    # ax.tick_params(axis='both', which='minor', labelsize=10, length=2.2, width=0.6)
    # ax.set_yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0])
    # ax.set_yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.ylim(1.0, 0.0)
    #plt.ylim(constants.DP.MAX_LOGINS, 0.0)
    fig.tight_layout()
    plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0, top=0.2)
    # fig.savefig("value_fun_3d_t_" +str(t) + ".png", format="png", dpi=600)
    # fig.savefig("value_fun_3d_t_" +str(t) + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)

def plot_policy(t=0):
    policy = load_policy()
    state_to_id = load_state_to_id()
    id_to_state = load_id_to_state()

    ttcs = np.arange(1, constants.DP.MAX_TTC, 1)
    ts = np.arange(1, constants.DP.MAX_TIMESTEPS, 1)
    x, y = np.meshgrid(ttcs, ts)
    # z = value_fun_helper(x, y, t, V, state_to_id)
    z = action_helper(x, y, policy, state_to_id)
    # z = Q_helper(x, y, policy, state_to_id, 0)
    # z2 = Q_helper(x, y, policy, state_to_id, 1)

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
    ax.plot_surface(x, y, z, cmap='Blues', linewidth=0.3,
                    alpha=0.8, edgecolor='k')
    # ax.plot_surface(x, y, z2, cmap='Reds', linewidth=0.3,
    #                 alpha=0.8, edgecolor='k')

    ax.set_title(r"$Q((" + str(t) + ", x, y), a)$", fontsize=14)
    ax.set_xlabel(r"TTC")
    ax.set_ylabel(r"Time-steps")
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_size(12)
    ylab.set_size(12)
    # ax.tick_params(axis='both', which='major', labelsize=10, length=2.2, width=0.6)
    # ax.tick_params(axis='both', which='minor', labelsize=10, length=2.2, width=0.6)
    # ax.set_yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0])
    # ax.set_yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.yticks([1, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.ylim(1.0, 0.0)
    # plt.ylim(constants.DP.MAX_LOGINS, 0.0)
    fig.tight_layout()
    plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0, top=0.2)
    # fig.savefig("policy_fun_3d_t_" + str(t) + ".png", format="png", dpi=600)
    # fig.savefig("policy_fun_3d_t_" + str(t) + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)
    # stopping_times = []
    # for i in range(policy.shape[0]):
    #     s = id_to_state[i]
    #     if s != "terminal":
    #         t1, x1, y1 = s
    #         if policy[i][1] == 1:
    #             stopping_times.append(t1)
    # print(stopping_times)

def plot_thresholds():
    # TTC = load_TTC_table()
    thresholds = load_thresholds()
    id_to_state = load_id_to_state()
    policy = load_policy()

    # for i in range(len(thresholds[1:])):
    #     pass
    min_threshold = -100
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
    x = []
    y = []
    for i in range(constants.DP.MAX_TIMESTEPS):
        # print(ts_to_thresholds[i])
        # avg_threshold = np.mean(np.array(ts_to_thresholds[i]))
        # avg_threshold = max(min_threshold, np.max(np.array(ts_to_thresholds[i])))
        avg_threshold = np.max(np.array(ts_to_thresholds[i]))
        # if i in ts_to_thresholds:
        #     avg_threshold = max(min_threshold, np.max(np.array(ts_to_thresholds[i])))
        # else:
        #     avg_threshold =min_threshold
        if i == constants.DP.MAX_TIMESTEPS-1:
            print(ts_to_thresholds[i])
        x.append(i)
        y.append(avg_threshold)
    print("len:{}".format(len(y)))
    # print(y[-1])
    # print(y[-2])
    # print(y[-3])
    # y[-2] = y[-3] + (y[-1] - y[-3])/1.2
    # y[-2] = 90

    print(y[-1])
    print(y[-2])
    print(y[-3])
    print(y[-4])
    print(y[-5])



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
    lower_bound = np.zeros(len(y))
    lower_bound.fill(min(y))
    print(min(y))
    print(lower_bound)
    ax.fill_between(x, y, lower_bound,
                    alpha=0.35, color=colors[0])

    # if plot_opt:
    # ax.plot(x,
    #         [0.5] * len(x), label=r"0.5",
    #         color="black",
    #         linestyle="dashed")

    ax.set_title(r"Stopping thresholds $\alpha_t$", fontsize=12.5)
    ax.set_xlabel(r"\# Time-step $t$", fontsize=11.5)

    ax.axhline(y=0, color='k', linewidth=0.5)
    # ax.axvline(x=0, color='k')
    # ax.set_ylabel(r"TTC $c$", fontsize=12)
    #ax.set_xlim(0, 92)
    #ax.set_ylim(-40, 100)
    # ax.set_ylim((0,5))

    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[-1] = r'$T$'

    a = ax.get_xticks().tolist()
    # a[-2] = r'$T$'
    # for i in range(len(a)):
    #     print(a[i])
    #     a[i] = 35*i
    a[-2] = r'$T$'
    ax.set_xticklabels(a)

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


def policy_plot_2():
    # TTC = load_TTC_table()
    thresholds = load_thresholds()
    id_to_state = load_id_to_state()
    policy = load_policy()

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

    fontsize = 6.5
    labelsize = 5
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.8
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams.update({'font.size': 6.5})

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(5, 4.5))

    x = []
    y = []
    t = 11
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[0][0].plot(x[0:],
            y[0:], label=r"$\pi_{\theta}$ simulation",
            ls='-', color=colors[0])

    ax[0][0].set_title(r"Optimal policy $\pi^{*}, t=11$", fontsize=fontsize)
    ax[0][0].set_xlabel(r"TTC $c$", fontsize=labelsize)
    ax[0][0].grid('on')
    ax[0][0].set_xlim(-1, 1)

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
    t = 12
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[0][1].plot(x[0:],
                  y[0:], label=r"$\pi_{\theta}$ simulation",
                  ls='-', color=colors[0])

    ax[0][1].set_title(r"Optimal policy $\pi^{*}, t=12$", fontsize=fontsize)
    ax[0][1].set_xlabel(r"TTC $c$", fontsize=labelsize)
    ax[0][1].grid('on')
    ax[0][1].set_xlim(-1, 1)

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
    t = 13
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[0][2].plot(x[0:],
                  y[0:], label=r"$\pi_{\theta}$ simulation",
                  ls='-', color=colors[0])

    ax[0][2].set_title(r"Optimal policy $\pi^{*}, t=13$", fontsize=fontsize)
    ax[0][2].set_xlabel(r"TTC $c$", fontsize=labelsize)
    ax[0][2].grid('on')
    ax[0][2].set_xlim(-1, 1)

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
    t = 14
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[1][0].plot(x[0:],
                  y[0:], label=r"$\pi_{\theta}$ simulation",
                  ls='-', color=colors[0])

    ax[1][0].set_title(r"Optimal policy $\pi^{*}, t=14$", fontsize=fontsize)
    ax[1][0].set_xlabel(r"TTC $c$", fontsize=labelsize)
    ax[1][0].grid('on')

    ax[1][0].set_xlim(-1, 1)

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
    t = 15
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[1][1].plot(x[0:],
                  y[0:], label=r"$\pi_{\theta}$ simulation",
                  ls='-', color=colors[0])

    ax[1][1].set_title(r"Optimal policy $\pi^{*}, t=15$", fontsize=fontsize)
    ax[1][1].set_xlabel(r"TTC $c$", fontsize=labelsize)
    ax[1][1].grid('on')

    ax[1][1].set_xlim(-1, 1)

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
    t = 16
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[1][2].plot(x[0:],
                  y[0:], label=r"$\pi_{\theta}$ simulation",
                  ls='-', color=colors[0])

    ax[1][2].set_title(r"Optimal policy $\pi^{*}, t=16$", fontsize=fontsize)
    ax[1][2].set_xlabel(r"TTC $c$", fontsize=labelsize)
    ax[1][2].grid('on')

    ax[1][2].set_xlim(-1, 1)

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
    t = 17
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[2][0].plot(x[0:],
                  y[0:], label=r"$\pi_{\theta}$ simulation",
                  ls='-', color=colors[0])

    ax[2][0].set_title(r"Optimal policy $\pi^{*}, t=17$", fontsize=fontsize)
    ax[2][0].set_xlabel(r"TTC $c$", fontsize=labelsize)
    ax[2][0].grid('on')

    ax[2][0].set_xlim(-1, 1)

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
    t = 18
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[2][1].plot(x[0:],
                  y[0:], label=r"$\pi_{\theta}$ simulation",
                  ls='-', color=colors[0])

    ax[2][1].set_title(r"Optimal policy $\pi^{*}, t=18$", fontsize=fontsize)
    ax[2][1].set_xlabel(r"TTC $c$", fontsize=labelsize)
    ax[2][1].grid('on')

    ax[2][1].set_xlim(-1, 1)

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
    t = 19
    print("thresh:{}".format(thresholds[t]))
    for i in np.arange(-10, constants.DP.MAX_TTC, 0.05):
        x.append(i)
        if i < thresholds[t]:
            y.append(1)
        else:
            y.append(0)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    ax[2][2].plot(x[0:],
                  y[0:], label=r"$\pi_{\theta}$ simulation",
                  ls='-', color=colors[0])

    ax[2][2].set_title(r"Optimal policy $\pi^{*}, t=19$", fontsize=fontsize)
    ax[2][2].set_xlabel(r"TTC $c$", fontsize=labelsize)
    ax[2][2].grid('on')

    ax[2][2].set_xlim(-1, 3)

    # tweak the axis labels
    xlab = ax[2][2].xaxis.get_label()
    ylab = ax[2][2].yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax[2][2].spines['right'].set_color((.8, .8, .8))
    ax[2][2].spines['top'].set_color((.8, .8, .8))


    fig.tight_layout()
    plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0)
    # fig.savefig("threshold_alerts" + ".png", format="png", dpi=600)
    # fig.savefig("threshold_alerts" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)



if __name__ == '__main__':
    policy_plot_2()
    # plot_thresholds()
    # plot_policy(t=0)
    # plot_policy(t=1)
    # plot_policy(t=2)
    # plot_policy(t=3)
    # plot_policy(t=4)
    # plot_policy(t=5)
    # plot_value_fun_3d(t=0)
    # plot_value_fun_3d(t=1)
    # plot_value_fun_3d(t=2)
    # plot_value_fun_3d(t=3)
    # plot_value_fun_3d(t=4)
    # plot_value_fun_3d(t=5)
    # plot_value_fun_3d(t=10)
    # plot_value_fun_3d(t=15)
    # plot_value_fun_3d(t=20)
    # plot_value_fun_3d(t=25)