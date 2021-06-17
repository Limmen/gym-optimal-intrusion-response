from typing import Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_flags_int_r_steps_costs_alerts(
        avg_train_rewards_data_v1, avg_train_rewards_means_v1,
        avg_train_rewards_stds_v1,
        avg_train_steps_data_v1, avg_train_steps_means_v1, avg_train_steps_stds_v1,
        avg_train_caught_frac_data_v1, avg_train_caught_frac_means_v1,
        avg_train_caught_frac_stds_v1,
        avg_train_early_stopping_frac_data_v1, avg_train_early_stopping_means_v1,
        avg_train_early_stopping_stds_v1, avg_train_intrusion_frac_data_v1,
        avg_train_intrusion_means_v1,
        avg_train_intrusion_stds_v1,
        fontsize : int = 6.5, figsize: Tuple[int,int] =  (3.75, 3.4),
        title_fontsize=8, lw=0.5, wspace=0.02, hspace=0.3, top=0.9,
        labelsize=6, markevery=10, optimal_reward = 95, sample_step = 1,
        eval_only=False, plot_opt = False, iterations_per_step : int = 1, optimal_int = 1.0,
        optimal_flag = 1.0, file_name = "test", markersize=5, bottom=0.02):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.8
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams.update({'font.size': fontsize})

    # plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=figsize)


    # Plot flags

    ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1[::sample_step])))) * sample_step * iterations_per_step,
            avg_train_rewards_means_v1[::sample_step], label=r"$\pi_{\theta}$ simulation",
            marker="s", ls='-', color="r", markevery=markevery, markersize=markersize, lw=lw)
    ax[0].fill_between(
        np.array(list(range(len(avg_train_rewards_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_rewards_means_v1[::sample_step] - avg_train_rewards_stds_v1[::sample_step],
        avg_train_rewards_means_v1[::sample_step] + avg_train_rewards_stds_v1[::sample_step],
        alpha=0.35, color="r", lw=lw)

    ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1)))) * iterations_per_step,
                  [165] * len(avg_train_rewards_means_v1), label=r"Optimal $\pi^{*}$",
                  color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
               ([113] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
               color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")

    ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
               ([71] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
               color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    # ax[0][0].plot(np.array(list(range(len(avg_train_flags_means_v1)))) * iterations_per_step,
    #         [optimal_flag*100] * len(avg_train_flags_means_v1), label=r"upper bound",
    #         color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[0].grid('on')
    # ax[0][0].set_xlabel("", fontsize=labelsize)
    #ax[0][0].set_ylabel(r"\% Flags captured", fontsize=labelsize)
    ax[0].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[0].xaxis.get_label()
    ylab = ax[0].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[0].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    # ax[0].set_ylim(0, 50)
    ax[0].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[0].set_title(r"Reward per episode", fontsize=fontsize)


    # % intrusions

    # ax[1].plot(
    #     np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
    #     avg_train_caught_frac_means_v1[::sample_step], label=r"$\mathbb{P}[detected]$ $\pi_{\theta}$ emulation",
    #     marker="p", ls='-', color="#599ad3",
    #     markevery=markevery, markersize=markersize, lw=lw)
    # ax[1].fill_between(
    #     np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
    #     avg_train_caught_frac_means_v1[::sample_step] - avg_train_rewards_stds_v1[::sample_step],
    #     avg_train_caught_frac_means_v1[::sample_step] + avg_train_rewards_stds_v1[::sample_step],
    #     alpha=0.35, color="#599ad3")

    ax[1].plot(np.array(list(range(len(avg_train_steps_means_v1[::sample_step]))))*sample_step* iterations_per_step,
            avg_train_steps_means_v1[::sample_step], label=r"$\mathbb{P}[detected]$ $\pi_{\theta}$ simulation",
            marker="s", ls='-', color="r",
            markevery=markevery, markersize=markersize, lw=lw)
    ax[1].fill_between(np.array(list(range(len(avg_train_steps_means_v1[::sample_step]))))*sample_step* iterations_per_step,
                    avg_train_steps_means_v1[::sample_step] - avg_train_steps_stds_v1[::sample_step],
                    avg_train_steps_means_v1[::sample_step] + avg_train_steps_stds_v1[::sample_step],
                    alpha=0.35, color="r")

    ax[1].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
               ([8] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
               color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")

    ax[1].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
               ([6] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
               color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    ax[1].grid('on')
    # ax[0][0].set_xlabel("", fontsize=labelsize)
    #ax[0][1].set_ylabel(r"$\mathbb{P}[\text{detected}]$", fontsize=labelsize)
    ax[1].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[1].xaxis.get_label()
    ylab = ax[1].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[1].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    ax[1].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[1].set_title(r"Episode length (steps)", fontsize=fontsize)

    ax[2].plot(np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
            avg_train_caught_frac_means_v1[::sample_step], label=r"Learned $\pi_{\theta}$",
            marker="s", ls='-', color="r",
            markevery=markevery, markersize=markersize, lw=lw)
    ax[2].fill_between(
        np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_caught_frac_means_v1[::sample_step] - avg_train_caught_frac_stds_v1[::sample_step],
        avg_train_caught_frac_means_v1[::sample_step] + avg_train_caught_frac_stds_v1[::sample_step],
        alpha=0.35, color="r")

    ax[2].grid('on')
    #ax[0][2].set_ylabel(r"Reward", fontsize=labelsize)
    ax[2].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[2].xaxis.get_label()
    ylab = ax[2].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[2].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[2].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    # ax[2].set_ylim(-100, 110)
    # ax[2].set_ylim(0, 1)
    ax[2].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[2].set_title(r"$\mathbb{P}[\text{attacker detected}]$", fontsize=fontsize)

    ax[2].plot(np.array(list(range(len(avg_train_rewards_means_v1)))) * iterations_per_step,
               [0.84] * len(avg_train_rewards_means_v1), label=r"Optimal $\pi^{*}$",
               color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[2].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
               ([0.6] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
               color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")

    ax[2].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
               ([0.01] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
               color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    ax[3].plot(
        np.array(list(range(len(avg_train_early_stopping_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v1[::sample_step], label=r"Defender $\pi_{\theta^D}$ simulation",
        marker="s", ls='-', color="r",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[3].fill_between(
        np.array(list(range(len(avg_train_early_stopping_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v1[::sample_step] - avg_train_early_stopping_stds_v1[::sample_step],
        avg_train_early_stopping_means_v1[::sample_step] + avg_train_early_stopping_stds_v1[::sample_step],
        alpha=0.35, color="r")

    ax[3].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
               ([0.4] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
               color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")

    ax[3].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
               ([0.99] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
               color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    ax[3].grid('on')
    # ax[0][2].set_ylabel(r"Reward", fontsize=labelsize)
    ax[3].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[3].xaxis.get_label()
    ylab = ax[3].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[3].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[3].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    # ax[2].set_ylim(-100, 110)
    # ax[3].set_ylim(0, 1)
    ax[3].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[3].set_title(r"$\mathbb{P}[\text{early stopping}]$", fontsize=fontsize)

    ax[3].plot(np.array(list(range(len(avg_train_rewards_means_v1)))) * iterations_per_step,
               [0.16] * len(avg_train_rewards_means_v1), label=r"Optimal $\pi^{*}$",
               color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, 0.165),
               ncol=5, fancybox=True, shadow=True)

    fig.tight_layout()
    #fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom)
    fig.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom)
    fig.savefig(file_name + ".png", format="png", dpi=600)
    fig.savefig(file_name + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)




def plot_flags_int_r_steps_costs_alerts_two_versions(
        avg_train_rewards_data_v1, avg_train_rewards_means_v1,
        avg_train_rewards_stds_v1,
        avg_train_steps_data_v1, avg_train_steps_means_v1, avg_train_steps_stds_v1,
        avg_train_caught_frac_data_v1, avg_train_caught_frac_means_v1,
        avg_train_caught_frac_stds_v1,
        avg_train_early_stopping_frac_data_v1, avg_train_early_stopping_means_v1,
        avg_train_early_stopping_stds_v1, avg_train_intrusion_frac_data_v1,
        avg_train_intrusion_means_v1,
        avg_train_intrusion_stds_v1,

        avg_train_rewards_data_v2, avg_train_rewards_means_v2,
        avg_train_rewards_stds_v2,
        avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2,
        avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2,
        avg_train_caught_frac_stds_v2,
        avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2,
        avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2,
        avg_train_intrusion_means_v2,
        avg_train_intrusion_stds_v2,
        optimal_rewards_v3_data, optimal_rewards_v3_means, optimal_rewards_v3_stds,
        optimal_steps_v3_data, optimal_steps_v3_means, optimal_steps_v3_stds,
        optimal_steps_v2_data, optimal_steps_v2_means, optimal_steps_v2_stds,
        optimal_rewards_v2_data, optimal_rewards_v2_means, optimal_rewards_v2_stds,

        t_baseline_rewards_data, t_baseline_rewards_means, t_baseline_rewards_stds,
        t_baseline_early_stopping_data, t_baseline_early_stopping_means, t_baseline_early_stopping_stds,
        t_baseline_caught_data, t_baseline_caught_means, t_baseline_caught_stds,

        a_baseline_rewards_data, a_baseline_rewards_means, a_baseline_rewards_stds,
        a_baseline_early_stopping_data, a_baseline_early_stopping_means, a_baseline_early_stopping_stds,
        a_baseline_caught_data, a_baseline_caught_means, a_baseline_caught_stds,

        a_baseline_steps_data, a_baseline_steps_means, a_baseline_steps_stds,

        t_baseline_i_steps_data, t_baseline_i_steps_means, t_baseline_i_steps_stds,
        a_baseline_i_steps_data, a_baseline_i_steps_means, a_baseline_i_steps_stds,

        avg_train_i_steps_data_v3, avg_train_i_steps_means_v3, avg_train_i_steps_stds_v3,
        avg_train_i_steps_data_v2, avg_train_i_steps_means_v2, avg_train_i_steps_stds_v2,

        fontsize : int = 6.5, figsize: Tuple[int,int] =  (3.75, 3.4),
        title_fontsize=8, lw=0.5, wspace=0.02, hspace=0.3, top=0.9,
        labelsize=6, markevery=10, optimal_reward = 95, sample_step = 1,
        eval_only=False, plot_opt = False, iterations_per_step : int = 1, optimal_int = 1.0,
        optimal_flag = 1.0, file_name = "test", markersize=5, bottom=0.02):

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.8
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams.update({'font.size': fontsize})

    # plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=figsize)


    # Plot flags

    ax[0].plot(
        np.array(list(range(len(avg_train_rewards_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_rewards_means_v2[::sample_step], label=r"$\pi_{\theta}$ simulation",
        marker="s", ls='-', color="r", markevery=markevery, markersize=markersize, lw=lw)
    ax[0].fill_between(
        np.array(list(range(len(avg_train_rewards_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_rewards_means_v2[::sample_step] - avg_train_rewards_stds_v2[::sample_step],
        avg_train_rewards_means_v2[::sample_step] + avg_train_rewards_stds_v2[::sample_step],
        alpha=0.35, color="r", lw=lw)

    ax[0].plot(
        np.array(list(range(len(avg_train_rewards_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_rewards_means_v1[::sample_step], label=r"$\pi_{\theta}$ simulation",
        marker="o", ls='-', color="#599ad3", markevery=markevery, markersize=markersize, lw=lw)
    ax[0].fill_between(
        np.array(list(range(len(avg_train_rewards_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_rewards_means_v1[::sample_step] - avg_train_rewards_stds_v1[::sample_step],
        avg_train_rewards_means_v1[::sample_step] + avg_train_rewards_stds_v1[::sample_step],
        alpha=0.35, color="#599ad3", lw=lw)

    ax[0].plot(
        np.array(list(range(len(t_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_rewards_means[::sample_step], label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)
    ax[0].fill_between(
        np.array(list(range(len(t_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_rewards_means[::sample_step] - t_baseline_rewards_stds[::sample_step],
        t_baseline_rewards_means[::sample_step] + t_baseline_rewards_stds[::sample_step],
        alpha=0.35, color="#f9a65a", lw=lw)

    print("t:{}".format(t_baseline_rewards_means))
    print("a:{}".format(a_baseline_rewards_means))

    ax[0].plot(
        np.array(list(range(len(a_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_rewards_means[::sample_step], label=r"$a=1$ baseline",
        marker="h", ls='-', color="#E7298A", markevery=markevery, markersize=markersize, lw=lw)
    ax[0].fill_between(
        np.array(list(range(len(a_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_rewards_means[::sample_step] - a_baseline_rewards_stds[::sample_step],
        a_baseline_rewards_means[::sample_step] + a_baseline_rewards_stds[::sample_step],
        alpha=0.35, color="#E7298A", lw=lw)


    ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1[::sample_step]))))* sample_step * iterations_per_step,
                  optimal_rewards_v2_means[::sample_step], label=r"Optimal $\pi^{*}$ vs \textsc{NoisyAttacker}",
                  color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[0].fill_between(
        np.array(list(range(len(avg_train_rewards_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        optimal_rewards_v2_means[::sample_step] - optimal_rewards_v2_stds[::sample_step],
        optimal_rewards_v2_means[::sample_step] + optimal_rewards_v2_stds[::sample_step],
        alpha=0.35, color="black")


    # ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([113] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
    #            color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")
    #
    # ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([71] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
    #            color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    # ax[0][0].plot(np.array(list(range(len(avg_train_flags_means_v1)))) * iterations_per_step,
    #         [optimal_flag*100] * len(avg_train_flags_means_v1), label=r"upper bound",
    #         color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[0].grid('on')
    # ax[0][0].set_xlabel("", fontsize=labelsize)
    #ax[0][0].set_ylabel(r"\% Flags captured", fontsize=labelsize)
    ax[0].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[0].xaxis.get_label()
    ylab = ax[0].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[0].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    ax[0].set_ylim(-130, 200)
    ax[0].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[0].set_title(r"Reward per episode", fontsize=fontsize)


    # % intrusions

    # ax[1].plot(
    #     np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
    #     avg_train_caught_frac_means_v1[::sample_step], label=r"$\mathbb{P}[detected]$ $\pi_{\theta}$ emulation",
    #     marker="p", ls='-', color="#599ad3",
    #     markevery=markevery, markersize=markersize, lw=lw)
    # ax[1].fill_between(
    #     np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
    #     avg_train_caught_frac_means_v1[::sample_step] - avg_train_rewards_stds_v1[::sample_step],
    #     avg_train_caught_frac_means_v1[::sample_step] + avg_train_rewards_stds_v1[::sample_step],
    #     alpha=0.35, color="#599ad3")

    ax[1].plot(np.array(list(range(len(avg_train_steps_means_v2[::sample_step])))) * sample_step * iterations_per_step,
               avg_train_steps_means_v2[::sample_step], label=r"$\mathbb{P}[detected]$ $\pi_{\theta}$ simulation",
               marker="s", ls='-', color="r",
               markevery=markevery, markersize=markersize, lw=lw)
    ax[1].fill_between(
        np.array(list(range(len(avg_train_steps_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_steps_means_v2[::sample_step] - avg_train_steps_stds_v2[::sample_step],
        avg_train_steps_means_v2[::sample_step] + avg_train_steps_stds_v2[::sample_step],
        alpha=0.35, color="r")

    ax[1].plot(np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
               avg_train_steps_means_v1[::sample_step], label=r"$\mathbb{P}[detected]$ $\pi_{\theta}$ simulation",
               marker="o", ls='-', color="#599ad3",
               markevery=markevery, markersize=markersize, lw=lw)
    ax[1].fill_between(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_steps_means_v1[::sample_step] - avg_train_steps_stds_v1[::sample_step],
        avg_train_steps_means_v1[::sample_step] + avg_train_steps_stds_v1[::sample_step],
        alpha=0.35, color="#599ad3")

    ax[1].plot(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        [6]*len(optimal_steps_v2_means[::sample_step]), label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)

    ax[1].plot(np.array(list(range(len(a_baseline_steps_means[::sample_step])))) * sample_step * iterations_per_step,
               a_baseline_steps_means[::sample_step], label=r"$a=1$ baseline",
               marker="h", ls='-', color="#E7298A",
               markevery=markevery, markersize=markersize, lw=lw)
    ax[1].fill_between(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_steps_means[::sample_step] - a_baseline_steps_stds[::sample_step],
        a_baseline_steps_means[::sample_step] + a_baseline_steps_stds[::sample_step],
        alpha=0.35, color="#E7298A")


    # ax[1].fill_between(
    #     np.array(list(range(len(t_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
    #     t_baseline_rewards_means[::sample_step] - t_baseline_rewards_stds[::sample_step],
    #     t_baseline_rewards_means[::sample_step] + t_baseline_rewards_stds[::sample_step],
    #     alpha=0.35, color="#f9a65a", lw=lw)

    ax[1].plot(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        optimal_steps_v2_means[::sample_step], label=r"Optimal $\pi^{*}$",
        color="black", linestyle="dashed", markersize=markersize, dashes=(0.5, 0.5), lw=lw, markevery=100)
    ax[1].fill_between(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        optimal_steps_v2_means[::sample_step] - optimal_steps_v2_stds[::sample_step],
        optimal_steps_v2_means[::sample_step] + optimal_steps_v2_stds[::sample_step],
        alpha=0.35, color="black")

    # ax[1].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([8] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
    #            color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")
    #
    # ax[1].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([6] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
    #            color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    ax[1].grid('on')
    # ax[0][0].set_xlabel("", fontsize=labelsize)
    #ax[0][1].set_ylabel(r"$\mathbb{P}[\text{detected}]$", fontsize=labelsize)
    ax[1].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[1].xaxis.get_label()
    ylab = ax[1].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[1].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    ax[1].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[1].set_title(r"Episode length (steps)", fontsize=fontsize)

    ax[2].plot(
        np.array(list(range(len(avg_train_caught_frac_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_caught_frac_means_v2[::sample_step], label=r"Learned $\pi_{\theta}$ vs \textsc{NoisyAttacker}",
        marker="s", ls='-', color="r",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[2].fill_between(
        np.array(list(range(len(avg_train_caught_frac_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_caught_frac_means_v2[::sample_step] - avg_train_caught_frac_stds_v2[::sample_step],
        avg_train_caught_frac_means_v2[::sample_step] + avg_train_caught_frac_stds_v2[::sample_step],
        alpha=0.35, color="r")

    ax[2].plot(
        np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_caught_frac_means_v1[::sample_step], label=r"Learned $\pi_{\theta}$ vs \textsc{StealthyAttacker}",
        marker="o", ls='-', color="#599ad3",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[2].fill_between(
        np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_caught_frac_means_v1[::sample_step] - avg_train_caught_frac_stds_v1[::sample_step],
        avg_train_caught_frac_means_v1[::sample_step] + avg_train_caught_frac_stds_v1[::sample_step],
        alpha=0.35, color="#599ad3")

    ax[2].plot(
        np.array(list(range(len(t_baseline_caught_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_caught_means[::sample_step], label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)
    ax[2].fill_between(
        np.array(list(range(len(t_baseline_caught_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_caught_means[::sample_step] - t_baseline_caught_stds[::sample_step],
        t_baseline_caught_means[::sample_step] + t_baseline_caught_stds[::sample_step],
        alpha=0.35, color="#f9a65a", lw=lw)

    ax[2].plot(
        np.array(list(range(len(a_baseline_caught_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_caught_means[::sample_step], label=r"$a=1$ baseline",
        marker="h", ls='-', color="#E7298A", markevery=markevery, markersize=markersize, lw=lw)
    ax[2].fill_between(
        np.array(list(range(len(a_baseline_caught_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_caught_means[::sample_step] - a_baseline_caught_stds[::sample_step],
        a_baseline_caught_means[::sample_step] + a_baseline_caught_stds[::sample_step],
        alpha=0.35, color="#E7298A", lw=lw)

    ax[2].grid('on')
    #ax[0][2].set_ylabel(r"Reward", fontsize=labelsize)
    ax[2].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[2].xaxis.get_label()
    ylab = ax[2].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[2].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[2].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    # ax[2].set_ylim(-100, 110)
    # ax[2].set_ylim(0, 1)
    ax[2].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[2].set_title(r"$\mathbb{P}[\text{intrusion interrupted}]$", fontsize=fontsize)

    ax[2].plot(np.array(list(range(len(avg_train_rewards_means_v1)))) * iterations_per_step,
               [1.00] * len(avg_train_rewards_means_v1), label=r"Upper bound $\pi^{*}$",
               color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)
    #label=r"Optimal $\pi^{*}$"

    # ax[2].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([0.6] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
    #            color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")
    #
    # ax[2].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([0.01] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
    #            color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    ax[3].plot(
        np.array(
            list(range(len(avg_train_early_stopping_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v2[::sample_step], label=r"Defender $\pi_{\theta^D}$ simulation",
        marker="s", ls='-', color="r",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[3].fill_between(
        np.array(
            list(range(len(avg_train_early_stopping_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v2[::sample_step] - avg_train_early_stopping_stds_v2[::sample_step],
        avg_train_early_stopping_means_v2[::sample_step] + avg_train_early_stopping_stds_v2[::sample_step],
        alpha=0.35, color="r")

    ax[3].plot(
        np.array(
            list(range(len(avg_train_early_stopping_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v1[::sample_step], label=r"Defender $\pi_{\theta^D}$ simulation",
        marker="o", ls='-', color="#599ad3",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[3].fill_between(
        np.array(
            list(range(len(avg_train_early_stopping_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v1[::sample_step] - avg_train_early_stopping_stds_v1[::sample_step],
        avg_train_early_stopping_means_v1[::sample_step] + avg_train_early_stopping_stds_v1[::sample_step],
        alpha=0.35, color="#599ad3")

    ax[3].plot(
        np.array(list(range(len(t_baseline_early_stopping_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_early_stopping_means[::sample_step], label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)
    ax[3].fill_between(
        np.array(list(range(len(t_baseline_early_stopping_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_early_stopping_means[::sample_step] - t_baseline_early_stopping_stds[::sample_step],
        t_baseline_early_stopping_means[::sample_step] + t_baseline_early_stopping_stds[::sample_step],
        alpha=0.35, color="#f9a65a", lw=lw)

    ax[3].plot(
        np.array(list(range(len(a_baseline_early_stopping_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_early_stopping_means[::sample_step], label=r"$t=6$ baseline",
        marker="h", ls='-', color="#E7298A", markevery=markevery, markersize=markersize, lw=lw)
    ax[3].fill_between(
        np.array(list(range(len(a_baseline_early_stopping_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_early_stopping_means[::sample_step] - a_baseline_early_stopping_stds[::sample_step],
        a_baseline_early_stopping_means[::sample_step] + a_baseline_early_stopping_stds[::sample_step],
        alpha=0.35, color="#E7298A", lw=lw)
    ax[3].plot(np.array(list(range(len(avg_train_rewards_means_v1)))) * iterations_per_step,
               [0.0] * len(avg_train_rewards_means_v1), label=r"Optimal $\pi^{*}$",
               color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[3].grid('on')
    # ax[0][2].set_ylabel(r"Reward", fontsize=labelsize)
    ax[3].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[3].xaxis.get_label()
    ylab = ax[3].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[3].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[3].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    # ax[2].set_ylim(-100, 110)
    ax[3].set_ylim(0, 1)
    ax[3].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[3].set_title(r"$\mathbb{P}[\text{early stopping}]$", fontsize=fontsize)

    ax[4].plot(
        np.array(
            list(range(len(avg_train_i_steps_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_i_steps_means_v2[::sample_step], label=r"Defender $\pi_{\theta^D}$ simulation",
        marker="s", ls='-', color="r",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[4].fill_between(
        np.array(
            list(range(len(avg_train_i_steps_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_i_steps_means_v2[::sample_step] - avg_train_early_stopping_stds_v2[::sample_step],
        avg_train_i_steps_means_v2[::sample_step] + avg_train_i_steps_stds_v2[::sample_step],
        alpha=0.35, color="r")

    ax[4].plot(
        np.array(
            list(range(len(avg_train_i_steps_means_v3[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_i_steps_means_v3[::sample_step], label=r"Defender $\pi_{\theta^D}$ simulation",
        marker="o", ls='-', color="#599ad3",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[4].fill_between(
        np.array(
            list(range(len(avg_train_i_steps_means_v3[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_i_steps_means_v3[::sample_step] - avg_train_i_steps_stds_v3[::sample_step],
        avg_train_i_steps_means_v3[::sample_step] + avg_train_i_steps_stds_v3[::sample_step],
        alpha=0.35, color="#599ad3")

    ax[4].plot(
        np.array(list(range(len(t_baseline_i_steps_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_i_steps_means[::sample_step], label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)
    ax[4].fill_between(
        np.array(list(range(len(t_baseline_i_steps_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_i_steps_means[::sample_step] - t_baseline_i_steps_stds[::sample_step],
        t_baseline_i_steps_means[::sample_step] + t_baseline_i_steps_stds[::sample_step],
        alpha=0.35, color="#f9a65a", lw=lw)

    ax[4].plot(
        np.array(list(range(len(a_baseline_i_steps_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_i_steps_means[::sample_step], label=r"$t=6$ baseline",
        marker="h", ls='-', color="#E7298A", markevery=markevery, markersize=markersize, lw=lw)
    ax[4].fill_between(
        np.array(list(range(len(a_baseline_i_steps_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_i_steps_means[::sample_step] - a_baseline_i_steps_stds[::sample_step],
        a_baseline_i_steps_means[::sample_step] + a_baseline_i_steps_stds[::sample_step],
        alpha=0.35, color="#E7298A", lw=lw)

    ax[4].plot(np.array(list(range(len(avg_train_rewards_means_v1)))) * iterations_per_step,
               [1.0] * len(avg_train_rewards_means_v1), label=r"Optimal $\pi^{*}$",
               color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[4].grid('on')
    # ax[0][2].set_ylabel(r"Reward", fontsize=labelsize)
    ax[4].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[4].xaxis.get_label()
    ylab = ax[4].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[4].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[4].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    # ax[2].set_ylim(-100, 110)
    ax[4].set_ylim(1, 4)
    ax[4].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[4].set_title(r"Uninterrupted intrusion $t$", fontsize=fontsize)


    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.505, 0.165),
               ncol=5, fancybox=True, shadow=True)

    fig.tight_layout()
    #fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom)
    fig.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom)
    fig.savefig(file_name + ".png", format="png", dpi=600)
    fig.savefig(file_name + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)



def plot_flags_int_r_steps_costs_alerts_two_versions(
        avg_train_rewards_data_v1, avg_train_rewards_means_v1,
        avg_train_rewards_stds_v1,
        avg_train_steps_data_v1, avg_train_steps_means_v1, avg_train_steps_stds_v1,
        avg_train_caught_frac_data_v1, avg_train_caught_frac_means_v1,
        avg_train_caught_frac_stds_v1,
        avg_train_early_stopping_frac_data_v1, avg_train_early_stopping_means_v1,
        avg_train_early_stopping_stds_v1, avg_train_intrusion_frac_data_v1,
        avg_train_intrusion_means_v1,
        avg_train_intrusion_stds_v1,

        avg_train_rewards_data_v2, avg_train_rewards_means_v2,
        avg_train_rewards_stds_v2,
        avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2,
        avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2,
        avg_train_caught_frac_stds_v2,
        avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2,
        avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2,
        avg_train_intrusion_means_v2,
        avg_train_intrusion_stds_v2,
        optimal_rewards_v3_data, optimal_rewards_v3_means, optimal_rewards_v3_stds,
        optimal_steps_v3_data, optimal_steps_v3_means, optimal_steps_v3_stds,
        optimal_steps_v2_data, optimal_steps_v2_means, optimal_steps_v2_stds,
        optimal_rewards_v2_data, optimal_rewards_v2_means, optimal_rewards_v2_stds,

        t_baseline_rewards_data, t_baseline_rewards_means, t_baseline_rewards_stds,
        t_baseline_early_stopping_data, t_baseline_early_stopping_means, t_baseline_early_stopping_stds,
        t_baseline_caught_data, t_baseline_caught_means, t_baseline_caught_stds,

        a_baseline_rewards_data, a_baseline_rewards_means, a_baseline_rewards_stds,
        a_baseline_early_stopping_data, a_baseline_early_stopping_means, a_baseline_early_stopping_stds,
        a_baseline_caught_data, a_baseline_caught_means, a_baseline_caught_stds,

        a_baseline_steps_data, a_baseline_steps_means, a_baseline_steps_stds,

        t_baseline_i_steps_data, t_baseline_i_steps_means, t_baseline_i_steps_stds,
        a_baseline_i_steps_data, a_baseline_i_steps_means, a_baseline_i_steps_stds,

        avg_train_i_steps_data_v3, avg_train_i_steps_means_v3, avg_train_i_steps_stds_v3,
        avg_train_i_steps_data_v2, avg_train_i_steps_means_v2, avg_train_i_steps_stds_v2,

        fontsize : int = 6.5, figsize: Tuple[int,int] =  (3.75, 3.4),
        title_fontsize=8, lw=0.5, wspace=0.02, hspace=0.3, top=0.9,
        labelsize=6, markevery=10, optimal_reward = 95, sample_step = 1,
        eval_only=False, plot_opt = False, iterations_per_step : int = 1, optimal_int = 1.0,
        optimal_flag = 1.0, file_name = "test", markersize=5, bottom=0.02):

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.8
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams.update({'font.size': fontsize})

    # plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=figsize)


    # Plot flags

    ax[0].plot(
        np.array(list(range(len(avg_train_rewards_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_rewards_means_v2[::sample_step], label=r"$\pi_{\theta}$ simulation",
        marker="s", ls='-', color="r", markevery=markevery, markersize=markersize, lw=lw)
    ax[0].fill_between(
        np.array(list(range(len(avg_train_rewards_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_rewards_means_v2[::sample_step] - avg_train_rewards_stds_v2[::sample_step],
        avg_train_rewards_means_v2[::sample_step] + avg_train_rewards_stds_v2[::sample_step],
        alpha=0.35, color="r", lw=lw)

    ax[0].plot(
        np.array(list(range(len(avg_train_rewards_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_rewards_means_v1[::sample_step], label=r"$\pi_{\theta}$ simulation",
        marker="o", ls='-', color="#599ad3", markevery=markevery, markersize=markersize, lw=lw)
    ax[0].fill_between(
        np.array(list(range(len(avg_train_rewards_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_rewards_means_v1[::sample_step] - avg_train_rewards_stds_v1[::sample_step],
        avg_train_rewards_means_v1[::sample_step] + avg_train_rewards_stds_v1[::sample_step],
        alpha=0.35, color="#599ad3", lw=lw)

    ax[0].plot(
        np.array(list(range(len(t_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_rewards_means[::sample_step], label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)
    ax[0].fill_between(
        np.array(list(range(len(t_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_rewards_means[::sample_step] - t_baseline_rewards_stds[::sample_step],
        t_baseline_rewards_means[::sample_step] + t_baseline_rewards_stds[::sample_step],
        alpha=0.35, color="#f9a65a", lw=lw)

    print("t:{}".format(t_baseline_rewards_means))
    print("a:{}".format(a_baseline_rewards_means))

    ax[0].plot(
        np.array(list(range(len(a_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_rewards_means[::sample_step], label=r"$a=1$ baseline",
        marker="h", ls='-', color="#E7298A", markevery=markevery, markersize=markersize, lw=lw)
    ax[0].fill_between(
        np.array(list(range(len(a_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_rewards_means[::sample_step] - a_baseline_rewards_stds[::sample_step],
        a_baseline_rewards_means[::sample_step] + a_baseline_rewards_stds[::sample_step],
        alpha=0.35, color="#E7298A", lw=lw)


    ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1[::sample_step]))))* sample_step * iterations_per_step,
                  optimal_rewards_v2_means[::sample_step], label=r"Optimal $\pi^{*}$ vs \textsc{NoisyAttacker}",
                  color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[0].fill_between(
        np.array(list(range(len(avg_train_rewards_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        optimal_rewards_v2_means[::sample_step] - optimal_rewards_v2_stds[::sample_step],
        optimal_rewards_v2_means[::sample_step] + optimal_rewards_v2_stds[::sample_step],
        alpha=0.35, color="black")


    # ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([113] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
    #            color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")
    #
    # ax[0].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([71] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
    #            color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    # ax[0][0].plot(np.array(list(range(len(avg_train_flags_means_v1)))) * iterations_per_step,
    #         [optimal_flag*100] * len(avg_train_flags_means_v1), label=r"upper bound",
    #         color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[0].grid('on')
    # ax[0][0].set_xlabel("", fontsize=labelsize)
    #ax[0][0].set_ylabel(r"\% Flags captured", fontsize=labelsize)
    ax[0].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[0].xaxis.get_label()
    ylab = ax[0].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[0].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    ax[0].set_ylim(-130, 200)
    ax[0].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[0].set_title(r"Reward per episode", fontsize=fontsize)


    # % intrusions

    # ax[1].plot(
    #     np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
    #     avg_train_caught_frac_means_v1[::sample_step], label=r"$\mathbb{P}[detected]$ $\pi_{\theta}$ emulation",
    #     marker="p", ls='-', color="#599ad3",
    #     markevery=markevery, markersize=markersize, lw=lw)
    # ax[1].fill_between(
    #     np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
    #     avg_train_caught_frac_means_v1[::sample_step] - avg_train_rewards_stds_v1[::sample_step],
    #     avg_train_caught_frac_means_v1[::sample_step] + avg_train_rewards_stds_v1[::sample_step],
    #     alpha=0.35, color="#599ad3")

    ax[1].plot(np.array(list(range(len(avg_train_steps_means_v2[::sample_step])))) * sample_step * iterations_per_step,
               avg_train_steps_means_v2[::sample_step], label=r"$\mathbb{P}[detected]$ $\pi_{\theta}$ simulation",
               marker="s", ls='-', color="r",
               markevery=markevery, markersize=markersize, lw=lw)
    ax[1].fill_between(
        np.array(list(range(len(avg_train_steps_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_steps_means_v2[::sample_step] - avg_train_steps_stds_v2[::sample_step],
        avg_train_steps_means_v2[::sample_step] + avg_train_steps_stds_v2[::sample_step],
        alpha=0.35, color="r")

    ax[1].plot(np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
               avg_train_steps_means_v1[::sample_step], label=r"$\mathbb{P}[detected]$ $\pi_{\theta}$ simulation",
               marker="o", ls='-', color="#599ad3",
               markevery=markevery, markersize=markersize, lw=lw)
    ax[1].fill_between(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_steps_means_v1[::sample_step] - avg_train_steps_stds_v1[::sample_step],
        avg_train_steps_means_v1[::sample_step] + avg_train_steps_stds_v1[::sample_step],
        alpha=0.35, color="#599ad3")

    ax[1].plot(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        [6]*len(optimal_steps_v2_means[::sample_step]), label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)

    ax[1].plot(np.array(list(range(len(a_baseline_steps_means[::sample_step])))) * sample_step * iterations_per_step,
               a_baseline_steps_means[::sample_step], label=r"$a=1$ baseline",
               marker="h", ls='-', color="#E7298A",
               markevery=markevery, markersize=markersize, lw=lw)
    ax[1].fill_between(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_steps_means[::sample_step] - a_baseline_steps_stds[::sample_step],
        a_baseline_steps_means[::sample_step] + a_baseline_steps_stds[::sample_step],
        alpha=0.35, color="#E7298A")


    # ax[1].fill_between(
    #     np.array(list(range(len(t_baseline_rewards_means[::sample_step])))) * sample_step * iterations_per_step,
    #     t_baseline_rewards_means[::sample_step] - t_baseline_rewards_stds[::sample_step],
    #     t_baseline_rewards_means[::sample_step] + t_baseline_rewards_stds[::sample_step],
    #     alpha=0.35, color="#f9a65a", lw=lw)

    ax[1].plot(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        optimal_steps_v2_means[::sample_step], label=r"Optimal $\pi^{*}$",
        color="black", linestyle="dashed", markersize=markersize, dashes=(0.5, 0.5), lw=lw, markevery=100)
    ax[1].fill_between(
        np.array(list(range(len(avg_train_steps_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        optimal_steps_v2_means[::sample_step] - optimal_steps_v2_stds[::sample_step],
        optimal_steps_v2_means[::sample_step] + optimal_steps_v2_stds[::sample_step],
        alpha=0.35, color="black")

    # ax[1].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([8] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
    #            color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")
    #
    # ax[1].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([6] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
    #            color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    ax[1].grid('on')
    # ax[0][0].set_xlabel("", fontsize=labelsize)
    #ax[0][1].set_ylabel(r"$\mathbb{P}[\text{detected}]$", fontsize=labelsize)
    ax[1].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[1].xaxis.get_label()
    ylab = ax[1].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[1].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    ax[1].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[1].set_title(r"Episode length (steps)", fontsize=fontsize)

    ax[2].plot(
        np.array(list(range(len(avg_train_caught_frac_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_caught_frac_means_v2[::sample_step], label=r"Learned $\pi_{\theta}$ vs \textsc{NoisyAttacker}",
        marker="s", ls='-', color="r",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[2].fill_between(
        np.array(list(range(len(avg_train_caught_frac_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_caught_frac_means_v2[::sample_step] - avg_train_caught_frac_stds_v2[::sample_step],
        avg_train_caught_frac_means_v2[::sample_step] + avg_train_caught_frac_stds_v2[::sample_step],
        alpha=0.35, color="r")

    ax[2].plot(
        np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_caught_frac_means_v1[::sample_step], label=r"Learned $\pi_{\theta}$ vs \textsc{StealthyAttacker}",
        marker="o", ls='-', color="#599ad3",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[2].fill_between(
        np.array(list(range(len(avg_train_caught_frac_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_caught_frac_means_v1[::sample_step] - avg_train_caught_frac_stds_v1[::sample_step],
        avg_train_caught_frac_means_v1[::sample_step] + avg_train_caught_frac_stds_v1[::sample_step],
        alpha=0.35, color="#599ad3")

    ax[2].plot(
        np.array(list(range(len(t_baseline_caught_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_caught_means[::sample_step], label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)
    ax[2].fill_between(
        np.array(list(range(len(t_baseline_caught_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_caught_means[::sample_step] - t_baseline_caught_stds[::sample_step],
        t_baseline_caught_means[::sample_step] + t_baseline_caught_stds[::sample_step],
        alpha=0.35, color="#f9a65a", lw=lw)

    ax[2].plot(
        np.array(list(range(len(a_baseline_caught_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_caught_means[::sample_step], label=r"$(x+y)\geq 1$ baseline",
        marker="h", ls='-', color="#E7298A", markevery=markevery, markersize=markersize, lw=lw)
    ax[2].fill_between(
        np.array(list(range(len(a_baseline_caught_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_caught_means[::sample_step] - a_baseline_caught_stds[::sample_step],
        a_baseline_caught_means[::sample_step] + a_baseline_caught_stds[::sample_step],
        alpha=0.35, color="#E7298A", lw=lw)

    ax[2].grid('on')
    #ax[0][2].set_ylabel(r"Reward", fontsize=labelsize)
    ax[2].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[2].xaxis.get_label()
    ylab = ax[2].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[2].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[2].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    # ax[2].set_ylim(-100, 110)
    # ax[2].set_ylim(0, 1)
    ax[2].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[2].set_title(r"$\mathbb{P}[\text{intrusion interrupted}]$", fontsize=fontsize)

    ax[2].plot(np.array(list(range(len(avg_train_rewards_means_v1)))) * iterations_per_step,
               [1.00] * len(avg_train_rewards_means_v1), label=r"Upper bound $\pi^{*}$",
               color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)
    #label=r"Optimal $\pi^{*}$"

    # ax[2].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([0.6] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_0$",
    #            color="#599ad3", markersize=markersize, lw=lw, markevery=markevery, marker="d")
    #
    # ax[2].plot(np.array(list(range(len(avg_train_rewards_means_v1))))[::sample_step] * iterations_per_step,
    #            ([0.01] * len(avg_train_rewards_means_v1))[::sample_step], label=r"$TTC_5$",
    #            color="#f9a65a", markersize=markersize, lw=lw, markevery=markevery, marker="h")

    ax[3].plot(
        np.array(
            list(range(len(avg_train_early_stopping_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v2[::sample_step], label=r"Defender $\pi_{\theta^D}$ simulation",
        marker="s", ls='-', color="r",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[3].fill_between(
        np.array(
            list(range(len(avg_train_early_stopping_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v2[::sample_step] - avg_train_early_stopping_stds_v2[::sample_step],
        avg_train_early_stopping_means_v2[::sample_step] + avg_train_early_stopping_stds_v2[::sample_step],
        alpha=0.35, color="r")

    ax[3].plot(
        np.array(
            list(range(len(avg_train_early_stopping_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v1[::sample_step], label=r"Defender $\pi_{\theta^D}$ simulation",
        marker="o", ls='-', color="#599ad3",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[3].fill_between(
        np.array(
            list(range(len(avg_train_early_stopping_means_v1[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_early_stopping_means_v1[::sample_step] - avg_train_early_stopping_stds_v1[::sample_step],
        avg_train_early_stopping_means_v1[::sample_step] + avg_train_early_stopping_stds_v1[::sample_step],
        alpha=0.35, color="#599ad3")

    ax[3].plot(
        np.array(list(range(len(t_baseline_early_stopping_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_early_stopping_means[::sample_step], label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)
    ax[3].fill_between(
        np.array(list(range(len(t_baseline_early_stopping_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_early_stopping_means[::sample_step] - t_baseline_early_stopping_stds[::sample_step],
        t_baseline_early_stopping_means[::sample_step] + t_baseline_early_stopping_stds[::sample_step],
        alpha=0.35, color="#f9a65a", lw=lw)

    ax[3].plot(
        np.array(list(range(len(a_baseline_early_stopping_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_early_stopping_means[::sample_step], label=r"$t=6$ baseline",
        marker="h", ls='-', color="#E7298A", markevery=markevery, markersize=markersize, lw=lw)
    ax[3].fill_between(
        np.array(list(range(len(a_baseline_early_stopping_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_early_stopping_means[::sample_step] - a_baseline_early_stopping_stds[::sample_step],
        a_baseline_early_stopping_means[::sample_step] + a_baseline_early_stopping_stds[::sample_step],
        alpha=0.35, color="#E7298A", lw=lw)
    ax[3].plot(np.array(list(range(len(avg_train_rewards_means_v1)))) * iterations_per_step,
               [0.0] * len(avg_train_rewards_means_v1), label=r"Optimal $\pi^{*}$",
               color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[3].grid('on')
    # ax[0][2].set_ylabel(r"Reward", fontsize=labelsize)
    ax[3].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[3].xaxis.get_label()
    ylab = ax[3].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[3].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[3].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    # ax[2].set_ylim(-100, 110)
    ax[3].set_ylim(0, 1)
    ax[3].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[3].set_title(r"$\mathbb{P}[\text{early stopping}]$", fontsize=fontsize)

    ax[4].plot(
        np.array(
            list(range(len(avg_train_i_steps_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_i_steps_means_v2[::sample_step], label=r"Defender $\pi_{\theta^D}$ simulation",
        marker="s", ls='-', color="r",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[4].fill_between(
        np.array(
            list(range(len(avg_train_i_steps_means_v2[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_i_steps_means_v2[::sample_step] - avg_train_early_stopping_stds_v2[::sample_step],
        avg_train_i_steps_means_v2[::sample_step] + avg_train_i_steps_stds_v2[::sample_step],
        alpha=0.35, color="r")

    ax[4].plot(
        np.array(
            list(range(len(avg_train_i_steps_means_v3[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_i_steps_means_v3[::sample_step], label=r"Defender $\pi_{\theta^D}$ simulation",
        marker="o", ls='-', color="#599ad3",
        markevery=markevery, markersize=markersize, lw=lw)
    ax[4].fill_between(
        np.array(
            list(range(len(avg_train_i_steps_means_v3[::sample_step])))) * sample_step * iterations_per_step,
        avg_train_i_steps_means_v3[::sample_step] - avg_train_i_steps_stds_v3[::sample_step],
        avg_train_i_steps_means_v3[::sample_step] + avg_train_i_steps_stds_v3[::sample_step],
        alpha=0.35, color="#599ad3")

    ax[4].plot(
        np.array(list(range(len(t_baseline_i_steps_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_i_steps_means[::sample_step], label=r"$t=6$ baseline",
        marker="d", ls='-', color="#f9a65a", markevery=markevery, markersize=markersize, lw=lw)
    ax[4].fill_between(
        np.array(list(range(len(t_baseline_i_steps_means[::sample_step])))) * sample_step * iterations_per_step,
        t_baseline_i_steps_means[::sample_step] - t_baseline_i_steps_stds[::sample_step],
        t_baseline_i_steps_means[::sample_step] + t_baseline_i_steps_stds[::sample_step],
        alpha=0.35, color="#f9a65a", lw=lw)

    ax[4].plot(
        np.array(list(range(len(a_baseline_i_steps_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_i_steps_means[::sample_step], label=r"$t=6$ baseline",
        marker="h", ls='-', color="#E7298A", markevery=markevery, markersize=markersize, lw=lw)
    ax[4].fill_between(
        np.array(list(range(len(a_baseline_i_steps_means[::sample_step])))) * sample_step * iterations_per_step,
        a_baseline_i_steps_means[::sample_step] - a_baseline_i_steps_stds[::sample_step],
        a_baseline_i_steps_means[::sample_step] + a_baseline_i_steps_stds[::sample_step],
        alpha=0.35, color="#E7298A", lw=lw)

    ax[4].plot(np.array(list(range(len(avg_train_rewards_means_v1)))) * iterations_per_step,
               [1.0] * len(avg_train_rewards_means_v1), label=r"Optimal $\pi^{*}$",
               color="black", linestyle="dashed", markersize=markersize, dashes=(4, 2), lw=lw)

    ax[4].grid('on')
    # ax[0][2].set_ylabel(r"Reward", fontsize=labelsize)
    ax[4].set_xlabel(r"\# policy updates", fontsize=labelsize)
    xlab = ax[4].xaxis.get_label()
    ylab = ax[4].yaxis.get_label()
    xlab.set_size(labelsize)
    ylab.set_size(fontsize)
    ax[4].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax[4].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
    # ax[2].set_ylim(-100, 110)
    ax[4].set_ylim(1, 4)
    ax[4].set_xlim(0, len(avg_train_rewards_means_v1[::sample_step]) * sample_step * iterations_per_step)
    ax[4].set_title(r"Uninterrupted intrusion $t$", fontsize=fontsize)


    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.505, 0.165),
               ncol=5, fancybox=True, shadow=True, handletextpad=0.4, labelspacing=0.5, columnspacing=0.65)

    fig.tight_layout()
    #fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom)
    fig.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom)
    fig.savefig(file_name + ".png", format="png", dpi=600)
    fig.savefig(file_name + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)