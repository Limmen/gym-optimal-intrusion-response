import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import gym_optimal_intrusion_response.constants.constants as constants


def Q_helper(alerts, logins, t, policy, state_to_id, action):
    z = []
    for i in range(len(alerts)):
        z1 = []
        for j in range(len(alerts[i])):
            id = state_to_id[(t, alerts[i][j], logins[i, j])]
            val = policy[id][2 + action]
            z1.append(val)
        z.append(z1)
    z = np.array(z)
    return z

def value_fun_helper(alerts, logins, t, V, state_to_id):
    z = []
    for i in range(len(alerts)):
        z1 = []
        for j in range(len(alerts[i])):
            id = state_to_id[(t, alerts[i][j], logins[i, j])]
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

def plot_value_fun_3d(t : int = 0):
    V = load_value_fun()
    state_to_id = load_state_to_id()
    id_to_state = load_id_to_state()
    policy = load_policy()

    alerts = np.arange(1, constants.DP.MAX_ALERTS, 1)
    logins = np.arange(1, constants.DP.MAX_LOGINS, 1)
    x, y = np.meshgrid(alerts, logins)
    z = value_fun_helper(x, y, t, V, state_to_id)

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
    ax.set_xlabel(r"IDS alerts $x$")
    ax.set_ylabel(r"Login attempts $y$")
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
    plt.ylim(constants.DP.MAX_LOGINS, 0.0)
    fig.tight_layout()
    # plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0, top=0.2)
    fig.savefig("value_fun_3d_t_" +str(t) + ".png", format="png", dpi=600)
    fig.savefig("value_fun_3d_t_" +str(t) + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)

def plot_policy(t=0):
    policy = load_policy()
    state_to_id = load_state_to_id()
    id_to_state = load_id_to_state()

    alerts = np.arange(1, constants.DP.MAX_ALERTS, 1)
    logins = np.arange(1, constants.DP.MAX_LOGINS, 1)
    x, y = np.meshgrid(alerts, logins)
    # z = value_fun_helper(x, y, t, V, state_to_id)
    z = Q_helper(x, y, t, policy, state_to_id, 0)
    z2 = Q_helper(x, y, t, policy, state_to_id, 1)

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
    ax.plot_surface(x, y, z2, cmap='Reds', linewidth=0.3,
                    alpha=0.8, edgecolor='k')

    ax.set_title(r"$Q((" + str(t) + ", x, y), a)$", fontsize=14)
    ax.set_xlabel(r"IDS alerts $x$")
    ax.set_ylabel(r"Login attempts $y$")
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
    plt.ylim(constants.DP.MAX_LOGINS, 0.0)
    fig.tight_layout()
    # plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0, top=0.2)
    fig.savefig("policy_fun_3d_t_" + str(t) + ".png", format="png", dpi=600)
    fig.savefig("policy_fun_3d_t_" + str(t) + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # plt.close(fig)
    # stopping_times = []
    # for i in range(policy.shape[0]):
    #     s = id_to_state[i]
    #     if s != "terminal":
    #         t1, x1, y1 = s
    #         if policy[i][1] == 1:
    #             stopping_times.append(t1)
    # print(stopping_times)

if __name__ == '__main__':
    plot_policy(t=0)
    plot_policy(t=1)
    plot_policy(t=2)
    plot_policy(t=3)
    plot_policy(t=4)
    plot_policy(t=5)
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