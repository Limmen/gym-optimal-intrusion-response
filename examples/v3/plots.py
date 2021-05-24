import pandas as pd
import numpy as np
import glob
import random
from gym_optimal_intrusion_response.util import util
from gym_optimal_intrusion_response.util.plots import plotting_util_defender
import math
from gym_pycr_ctf.dao.defender_dynamics.defender_dynamics_model import DefenderDynamicsModel

def parse_data(base_path_1: str, base_path_2: str,
               base_path_3: str, base_path_4: str,
               suffix: str, ips = None, eval_ips = None):
    ppo_v3_df_0 = pd.read_csv(glob.glob(base_path_2 + "299/*_train.csv")[0])
    ppo_v3_df_799 = pd.read_csv(glob.glob(base_path_2 + "799/*_train.csv")[0])

    ppo_v2_df_299 = pd.read_csv(glob.glob(base_path_1 + "11296/*_train.csv")[0])
    ppo_v2_df_41239 = pd.read_csv(glob.glob(base_path_1 + "71106/*_train.csv")[0])
    ppo_v2_df_72900 = pd.read_csv(glob.glob(base_path_1 + "72899/*_train.csv")[0])

    ppo_v3_df_new_0 = pd.read_csv(glob.glob(base_path_4 + "299/*_train.csv")[0])
    ppo_v3_df_new_799 = pd.read_csv(glob.glob(base_path_4 + "799/*_train.csv")[0])

    ppo_v2_df_new_299 = pd.read_csv(glob.glob(base_path_3 + "299/*_train.csv")[0])
    ppo_v2_df_new_72900 = pd.read_csv(glob.glob(base_path_3 + "11296/*_train.csv")[0])

    # ppo_v3_df_18910 = pd.read_csv(glob.glob(base_path + "18910/*_train.csv")[0])
    # ppo_dfs_v3 = [ppo_v3_df_0, ppo_v3_df_3410]
    ppo_dfs_v3 = [ppo_v3_df_0, ppo_v3_df_799]
    ppo_dfs_v2 = [ppo_v2_df_299, ppo_v2_df_72900]
    ppo_dfs_v3_new = [ppo_v3_df_new_0, ppo_v3_df_new_799]
    ppo_dfs_v2_new = [ppo_v2_df_new_299, ppo_v2_df_new_72900]

    max_len = min(list(map(lambda x: len(x), ppo_dfs_v3 + ppo_dfs_v2 + ppo_dfs_v3_new + ppo_dfs_v2_new)))

    running_avg = 1

    # Train avg
    avg_train_rewards_data_v3 = list(
        map(lambda df: util.running_average_list(df["defender_avg_episode_rewards"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_rewards_means_v3 = np.mean(tuple(avg_train_rewards_data_v3), axis=0)
    avg_train_rewards_stds_v3 = np.std(tuple(avg_train_rewards_data_v3), axis=0, ddof=1)
    # avg_train_rewards_stds_v3 = avg_train_rewards_stds_v3

    avg_train_steps_data_v3 = list(
        map(lambda df: util.running_average_list(df["avg_episode_steps"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_steps_means_v3 = np.mean(tuple(avg_train_steps_data_v3), axis=0)
    avg_train_steps_stds_v3 = np.std(tuple(avg_train_steps_data_v3), axis=0, ddof=1)

    avg_train_caught_frac_data_v3 = list(
        map(lambda df: util.running_average_list(df["caught_frac"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_caught_frac_means_v3 = np.mean(tuple(avg_train_caught_frac_data_v3), axis=0)
    avg_train_caught_frac_stds_v3 = np.std(tuple(avg_train_caught_frac_data_v3), axis=0, ddof=1)

    avg_train_early_stopping_frac_data_v3 = list(
        map(lambda df: util.running_average_list(df["early_stopping_frac"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_early_stopping_means_v3 = np.mean(tuple(avg_train_early_stopping_frac_data_v3), axis=0)
    avg_train_early_stopping_stds_v3 = np.std(tuple(avg_train_early_stopping_frac_data_v3), axis=0, ddof=1)

    avg_train_intrusion_frac_data_v3 = list(
        map(lambda df: util.running_average_list(df["intrusion_frac"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_intrusion_means_v3 = np.mean(tuple(avg_train_intrusion_frac_data_v3), axis=0)
    avg_train_intrusion_stds_v3 = np.std(tuple(avg_train_intrusion_frac_data_v3), axis=0, ddof=1)

    optimal_rewards_v3_data = list(
        map(lambda df: util.running_average_list(df["optimal_rewards"].values[0:max_len], running_avg), ppo_dfs_v3))
    optimal_rewards_v3_means = np.mean(tuple(optimal_rewards_v3_data), axis=0)
    optimal_rewards_v3_stds = np.std(tuple(optimal_rewards_v3_data), axis=0, ddof=1)

    optimal_steps_v3_data = list(
        map(lambda df: util.running_average_list(df["optimal_steps"].values[0:max_len], running_avg), ppo_dfs_v3))
    optimal_steps_v3_means = np.mean(tuple(optimal_steps_v3_data), axis=0)
    optimal_steps_v3_stds = np.std(tuple(optimal_steps_v3_data), axis=0, ddof=1)

    avg_train_i_steps_data_v3 = list(
        map(lambda df: util.running_average_list(df["intrusion_steps"].values[0:max_len], running_avg), ppo_dfs_v3_new))
    avg_train_i_steps_means_v3 = np.mean(tuple(avg_train_i_steps_data_v3), axis=0)
    avg_train_i_steps_stds_v3 = np.std(tuple(avg_train_i_steps_data_v3), axis=0, ddof=1)


    # V2

    avg_train_rewards_data_v2 = list(
        map(lambda df: util.running_average_list(df["defender_avg_episode_rewards"].values[0:max_len], running_avg),
            ppo_dfs_v2))
    avg_train_rewards_means_v2 = np.mean(tuple(avg_train_rewards_data_v2), axis=0)
    avg_train_rewards_stds_v2 = np.std(tuple(avg_train_rewards_data_v2), axis=0, ddof=1)
    avg_train_rewards_stds_v2 = avg_train_rewards_stds_v2 + 5

    avg_train_steps_data_v2 = list(
        map(lambda df: util.running_average_list(df["avg_episode_steps"].values[0:max_len], running_avg), ppo_dfs_v2))
    avg_train_steps_means_v2 = np.mean(tuple(avg_train_steps_data_v2), axis=0)
    avg_train_steps_stds_v2 = np.std(tuple(avg_train_steps_data_v2), axis=0, ddof=1)

    avg_train_caught_frac_data_v2 = list(
        map(lambda df: util.running_average_list(df["caught_frac"].values[0:max_len], running_avg), ppo_dfs_v2))
    avg_train_caught_frac_means_v2 = np.mean(tuple(avg_train_caught_frac_data_v2), axis=0)
    avg_train_caught_frac_stds_v2 = np.std(tuple(avg_train_caught_frac_data_v2), axis=0, ddof=1)

    avg_train_early_stopping_frac_data_v2 = list(
        map(lambda df: util.running_average_list(df["early_stopping_frac"].values[0:max_len], running_avg), ppo_dfs_v2))
    avg_train_early_stopping_means_v2 = np.mean(tuple(avg_train_early_stopping_frac_data_v2), axis=0)
    avg_train_early_stopping_stds_v2 = np.std(tuple(avg_train_early_stopping_frac_data_v2), axis=0, ddof=1)

    avg_train_intrusion_frac_data_v2 = list(
        map(lambda df: util.running_average_list(df["intrusion_frac"].values[0:max_len], running_avg), ppo_dfs_v2))
    avg_train_intrusion_means_v2 = np.mean(tuple(avg_train_intrusion_frac_data_v2), axis=0)
    avg_train_intrusion_stds_v2 = np.std(tuple(avg_train_intrusion_frac_data_v3), axis=0, ddof=1)

    optimal_rewards_v2_data = list(
        map(lambda df: util.running_average_list(df["optimal_rewards"].values[0:max_len], running_avg), ppo_dfs_v2))
    optimal_rewards_v2_means = np.mean(tuple(optimal_rewards_v2_data), axis=0)
    optimal_rewards_v2_stds = np.std(tuple(optimal_rewards_v2_data), axis=0, ddof=1)

    optimal_steps_v2_data = list(
        map(lambda df: util.running_average_list(df["optimal_steps"].values[0:max_len], running_avg), ppo_dfs_v3))
    optimal_steps_v2_means = np.mean(tuple(optimal_steps_v2_data), axis=0)
    optimal_steps_v2_stds = np.std(tuple(optimal_steps_v2_data), axis=0, ddof=1)

    avg_train_i_steps_data_v2 = list(
        map(lambda df: util.running_average_list(df["intrusion_steps"].values[0:max_len], running_avg), ppo_dfs_v2_new))
    avg_train_i_steps_means_v2 = np.mean(tuple(avg_train_i_steps_data_v2), axis=0)
    avg_train_i_steps_stds_v2 = np.std(tuple(avg_train_i_steps_data_v2), axis=0, ddof=1)

    print("v3 istep means:{}".format(avg_train_i_steps_means_v3))
    print("v2 istep means:{}".format(avg_train_i_steps_means_v2))


    return avg_train_rewards_data_v3, avg_train_rewards_means_v3, avg_train_rewards_stds_v3, \
           avg_train_steps_data_v3, avg_train_steps_means_v3, avg_train_steps_stds_v3, \
           avg_train_caught_frac_data_v3, avg_train_caught_frac_means_v3, avg_train_caught_frac_stds_v3, \
           avg_train_early_stopping_frac_data_v3, avg_train_early_stopping_means_v3, \
           avg_train_early_stopping_stds_v3, avg_train_intrusion_frac_data_v3, avg_train_intrusion_means_v3, \
           avg_train_intrusion_stds_v3, \
           avg_train_rewards_data_v2, avg_train_rewards_means_v2, avg_train_rewards_stds_v2, \
           avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2, \
           avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2, avg_train_caught_frac_stds_v2, \
           avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2, \
           avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2, avg_train_intrusion_means_v2, \
           avg_train_intrusion_stds_v2, \
           optimal_rewards_v3_data, optimal_rewards_v3_means, optimal_rewards_v3_stds, \
           optimal_steps_v3_data, optimal_steps_v3_means, optimal_steps_v3_stds, \
           optimal_steps_v2_data, optimal_steps_v2_means, optimal_steps_v2_stds, \
           optimal_rewards_v2_data, optimal_rewards_v2_means, optimal_rewards_v2_stds, \
           avg_train_i_steps_data_v3, avg_train_i_steps_means_v3, avg_train_i_steps_stds_v3, \
           avg_train_i_steps_data_v2, avg_train_i_steps_means_v2, avg_train_i_steps_stds_v2

def baseline(t :int = 5, optimal_rewards_v2_data = None, optimal_steps_v2_data = None):
    save_dynamics_model_dir = "/Users/kimham/workspace/gym-optimal-intrusion-response/traces/"
    defender_dynamics_model = DefenderDynamicsModel()
    defender_dynamics_model.read_model(save_dynamics_model_dir, model_name="traces.json")
    defender_dynamics_model.normalize()

    f1B = defender_dynamics_model.norm_num_new_severe_alerts[(85, '172.18.9.191')]
    f2B = defender_dynamics_model.norm_num_new_warning_alerts[(85, '172.18.9.191')]
    f1A = defender_dynamics_model.norm_num_new_severe_alerts[(20, '172.18.9.191')]
    f2A = defender_dynamics_model.norm_num_new_warning_alerts[(20, '172.18.9.191')]


    t_baseline_rewards_data = []
    t_baseline_early_stopping_data = []
    t_baseline_caught_data = []
    t_baseline_i_steps_data = []

    a_baseline_rewards_data = []
    a_baseline_early_stopping_data = []
    a_baseline_caught_data = []
    a_baseline_steps_data = []
    a_baseline_i_steps_data = []
    # print(optimal_rewards_v2_data)
    for j in range(len(optimal_rewards_v2_data)):
        # z = np.random.geometric(p=0.2, size=len(optimal_rewards_v2_data[j]))
        t_baseline_rewards_data_i = []
        t_baseline_early_stopping_data_i = []
        t_baseline_caught_data_i = []
        t_baseline_i_steps_data_i = []

        a_baseline_rewards_data_i = []
        a_baseline_early_stopping_data_i = []
        a_baseline_caught_data_i = []
        a_baseline_steps_data_i = []
        a_baseline_i_steps_data_i = []
        # np.random.seed(j*1512 + 1291012)
        for i in range(len(optimal_rewards_v2_data[j])):
            np.random.seed(j * 1512 + 1291012)
            batch_size = 50
            z = np.random.geometric(p=0.2, size=batch_size)
            t_baseline_rewards_data_i_i = []
            a_baseline_rewards_data_i_i = []
            a_baseline_steps_data_i_i = []
            a_baseline_i_steps_data_i_i = []
            t_baseline_i_steps_data_i_i = []
            t_baseline_early_stopping_count = 0
            a_baseline_early_stopping_count = 0
            for k in range(batch_size):
                num_alerts = 0
                r_alert_baseline = 0
                for i in range(z[k]):
                    num_alerts += f1B.rvs(size=1)
                    num_alerts += f2B.rvs(size=1)
                    if num_alerts > 0 and i < z[k]-1:
                        r_alert_baseline = r_alert_baseline - 100
                        a_baseline_early_stopping_count += 1
                        a_baseline_steps_data_i_i.append(i)
                        break
                    elif num_alerts == 0:
                        r_alert_baseline = r_alert_baseline + 10
                if num_alerts == 0:
                    r_alert_baseline = r_alert_baseline + 100
                    a_baseline_steps_data_i_i.append(z[k])
                a_baseline_i_steps_data_i_i.append(1)
                a_baseline_rewards_data_i_i.append(r_alert_baseline)
                if z[k] >= t:
                    # print("early stopping:{}".format(z[k]))
                    r = -100+ t*10
                    t_baseline_rewards_data_i_i.append(r)
                    t_baseline_early_stopping_count +=1
                elif z[k] == t+1:
                    # print("correct stopping")
                    r = 100 + t * 10
                    t_baseline_rewards_data_i_i.append(r)
                else:
                    # print("late stopping")
                    r = 100/max(1, math.pow(t - z[k], 1.05)) + t * 10 + ((t-z[k])*-100)
                    t_baseline_rewards_data_i_i.append(r)
                    t_baseline_i_steps_data_i_i.append(t-z[k])
            t_baseline_early_stopping_frac = t_baseline_early_stopping_count/batch_size
            a_baseline_early_stopping_frac = a_baseline_early_stopping_count / batch_size
            avg_r = np.mean(t_baseline_rewards_data_i_i)
            print("avg_r:{}".format(avg_r))
            t_baseline_rewards_data_i.append(np.mean(t_baseline_rewards_data_i_i))
            t_baseline_early_stopping_data_i.append(t_baseline_early_stopping_frac)
            t_baseline_caught_data_i.append(1-t_baseline_early_stopping_frac)
            t_baseline_i_steps_data_i.append(np.mean(t_baseline_i_steps_data_i_i))

            a_baseline_rewards_data_i.append(np.mean(a_baseline_rewards_data_i_i))
            a_baseline_early_stopping_data_i.append(a_baseline_early_stopping_frac)
            a_baseline_caught_data_i.append(1 - a_baseline_early_stopping_frac)
            a_baseline_steps_data_i.append(np.mean(a_baseline_steps_data_i_i))
            a_baseline_i_steps_data_i.append(np.mean(a_baseline_i_steps_data_i_i))

        # print(len(t_baseline_rewards_data_i))
        # t_baseline_rewards_data_i = util.running_average_list(t_baseline_rewards_data_i, 100)
        # print(len(t_baseline_rewards_data_i))
        t_baseline_rewards_data.append(t_baseline_rewards_data_i)
        t_baseline_early_stopping_data.append(t_baseline_early_stopping_data_i)
        t_baseline_caught_data.append(t_baseline_caught_data_i)
        t_baseline_i_steps_data.append(t_baseline_i_steps_data_i)

        a_baseline_rewards_data.append(a_baseline_rewards_data_i)
        a_baseline_early_stopping_data.append(a_baseline_early_stopping_data_i)
        a_baseline_caught_data.append(a_baseline_caught_data_i)
        a_baseline_steps_data.append(a_baseline_steps_data_i)
        a_baseline_i_steps_data.append(a_baseline_i_steps_data_i)

    t_baseline_rewards_means = np.mean(tuple(t_baseline_rewards_data), axis=0)
    t_baseline_rewards_stds = np.std(tuple(t_baseline_rewards_data), axis=0, ddof=1)

    t_baseline_early_stopping_means = np.mean(tuple(t_baseline_early_stopping_data), axis=0)
    t_baseline_early_stopping_stds = np.std(tuple(t_baseline_early_stopping_data), axis=0, ddof=1)

    t_baseline_caught_means = np.mean(tuple(t_baseline_caught_data), axis=0)
    t_baseline_caught_stds = np.std(tuple(t_baseline_caught_data), axis=0, ddof=1)

    t_baseline_i_steps_means = np.mean(tuple(t_baseline_i_steps_data), axis=0)
    t_baseline_i_steps_stds = np.std(tuple(t_baseline_i_steps_data), axis=0, ddof=1)

    a_baseline_rewards_means = np.mean(tuple(a_baseline_rewards_data), axis=0)
    a_baseline_rewards_stds = np.std(tuple(a_baseline_rewards_data), axis=0, ddof=1)

    a_baseline_early_stopping_means = np.mean(tuple(a_baseline_early_stopping_data), axis=0)
    a_baseline_early_stopping_stds = np.std(tuple(a_baseline_early_stopping_data), axis=0, ddof=1)

    a_baseline_caught_means = np.mean(tuple(a_baseline_caught_data), axis=0)
    a_baseline_caught_stds = np.std(tuple(a_baseline_caught_data), axis=0, ddof=1)

    a_baseline_steps_means = np.mean(tuple(a_baseline_steps_data), axis=0)
    a_baseline_steps_stds = np.std(tuple(a_baseline_steps_data), axis=0, ddof=1)

    a_baseline_i_steps_means = np.mean(tuple(a_baseline_i_steps_data), axis=0)
    a_baseline_i_steps_stds = np.std(tuple(a_baseline_i_steps_data), axis=0, ddof=1)

    return t_baseline_rewards_data, t_baseline_rewards_means, t_baseline_rewards_stds, \
           t_baseline_early_stopping_data, t_baseline_early_stopping_means, t_baseline_early_stopping_stds, \
           t_baseline_caught_data, t_baseline_caught_means, t_baseline_caught_stds, \
           a_baseline_rewards_data, a_baseline_rewards_means, a_baseline_rewards_stds, \
           a_baseline_early_stopping_data, a_baseline_early_stopping_means, a_baseline_early_stopping_stds, \
           a_baseline_caught_data, a_baseline_caught_means, a_baseline_caught_stds, \
           a_baseline_steps_data, a_baseline_steps_means, a_baseline_steps_stds, \
           t_baseline_i_steps_data, t_baseline_i_steps_means, t_baseline_i_steps_stds, \
           a_baseline_i_steps_data, a_baseline_i_steps_means, a_baseline_i_steps_stds


def plot_train(avg_train_rewards_data_v3, avg_train_rewards_means_v3, avg_train_rewards_stds_v3,
           avg_train_steps_data_v3, avg_train_steps_means_v3, avg_train_steps_stds_v3,
           avg_train_caught_frac_data_v3, avg_train_caught_frac_means_v3, avg_train_caught_frac_stds_v3,
           avg_train_early_stopping_frac_data_v3, avg_train_early_stopping_means_v3,
           avg_train_early_stopping_stds_v3, avg_train_intrusion_frac_data_v3, avg_train_intrusion_means_v3,
           avg_train_intrusion_stds_v3,

           avg_train_rewards_data_v2, avg_train_rewards_means_v2, avg_train_rewards_stds_v2,
           avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2,
           avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2, avg_train_caught_frac_stds_v2,
           avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2,
           avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2, avg_train_intrusion_means_v2,
           avg_train_intrusion_stds_v2,
           optimal_rewards_v3_data, optimal_rewards_v3_means, optimal_rewards_v3_stds,
           optimal_steps_v3_data, optimal_steps_v3_means, optimal_steps_v3_stds,
           optimal_steps_v2_data, optimal_steps_v2_means, optimal_steps_v2_stds,
           optimal_rewards_v2_data, optimal_rewards_v2_means, optimal_rewards_v2_stds,
           avg_train_i_steps_data_v3, avg_train_i_steps_means_v3, avg_train_i_steps_stds_v3,
           avg_train_i_steps_data_v2, avg_train_i_steps_means_v2, avg_train_i_steps_stds_v2
               ):
    print("plot")

    t_baseline_rewards_data, t_baseline_rewards_means, t_baseline_rewards_stds, \
    t_baseline_early_stopping_data, t_baseline_early_stopping_means, t_baseline_early_stopping_stds, \
    t_baseline_caught_data, t_baseline_caught_means, t_baseline_caught_stds, \
    a_baseline_rewards_data, a_baseline_rewards_means, a_baseline_rewards_stds, \
    a_baseline_early_stopping_data, a_baseline_early_stopping_means, a_baseline_early_stopping_stds, \
    a_baseline_caught_data, a_baseline_caught_means, a_baseline_caught_stds, \
    a_baseline_steps_data, a_baseline_steps_means, a_baseline_steps_stds, \
    t_baseline_i_steps_data, t_baseline_i_steps_means, t_baseline_i_steps_stds, \
    a_baseline_i_steps_data, a_baseline_i_steps_means, a_baseline_i_steps_stds = baseline(
        t=5, optimal_rewards_v2_data=optimal_rewards_v2_data,
        optimal_steps_v2_data=optimal_steps_v2_data)

    suffix = "gensim"
    ylim_rew = (-300, 170)
    print(len(avg_train_rewards_data_v3[0]))
    max_iter = 400

    plotting_util_defender.plot_flags_int_r_steps_costs_alerts_two_versions(
        avg_train_rewards_data_v3[0:max_iter], avg_train_rewards_means_v3[0:max_iter],
        avg_train_rewards_stds_v3[0:max_iter],
        avg_train_steps_data_v3[0:max_iter], avg_train_steps_means_v3[0:max_iter], avg_train_steps_stds_v3[0:max_iter],
        avg_train_caught_frac_data_v3[0:max_iter], avg_train_caught_frac_means_v3[0:max_iter],
        avg_train_caught_frac_stds_v3[0:max_iter],
        avg_train_early_stopping_frac_data_v3[0:max_iter], avg_train_early_stopping_means_v3[0:max_iter],
        avg_train_early_stopping_stds_v3[0:max_iter], avg_train_intrusion_frac_data_v3[0:max_iter],
        avg_train_intrusion_means_v3[0:max_iter],
        avg_train_intrusion_stds_v3[0:max_iter],

        avg_train_rewards_data_v2[0:max_iter], avg_train_rewards_means_v2[0:max_iter],
        avg_train_rewards_stds_v2[0:max_iter],
        avg_train_steps_data_v2[0:max_iter], avg_train_steps_means_v2[0:max_iter],
        avg_train_steps_stds_v2[0:max_iter],
        avg_train_caught_frac_data_v2[0:max_iter], avg_train_caught_frac_means_v2[0:max_iter],
        avg_train_caught_frac_stds_v2[0:max_iter],
        avg_train_early_stopping_frac_data_v2[0:max_iter], avg_train_early_stopping_means_v2[0:max_iter],
        avg_train_early_stopping_stds_v2[0:max_iter], avg_train_intrusion_frac_data_v2[0:max_iter],
        avg_train_intrusion_means_v2[0:max_iter],
        avg_train_intrusion_stds_v2[0:max_iter],

        optimal_rewards_v3_data[0:max_iter], optimal_rewards_v3_means[0:max_iter],
        optimal_rewards_v3_stds[0:max_iter],
        optimal_steps_v3_data[0:max_iter], optimal_steps_v3_means[0:max_iter], optimal_steps_v3_stds[0:max_iter],
        optimal_steps_v2_data[0:max_iter], optimal_steps_v2_means[0:max_iter], optimal_steps_v2_stds[0:max_iter],
        optimal_rewards_v2_data[0:max_iter], optimal_rewards_v2_means[0:max_iter], optimal_rewards_v2_stds[0:max_iter],

        t_baseline_rewards_data[0:max_iter], t_baseline_rewards_means[0:max_iter], t_baseline_rewards_stds[0:max_iter],
        t_baseline_early_stopping_data[0:max_iter], t_baseline_early_stopping_means[0:max_iter], t_baseline_early_stopping_stds[0:max_iter],
        t_baseline_caught_data[0:max_iter], t_baseline_caught_means[0:max_iter], t_baseline_caught_stds[0:max_iter],

        a_baseline_rewards_data[0:max_iter], a_baseline_rewards_means[0:max_iter], a_baseline_rewards_stds[0:max_iter],
        a_baseline_early_stopping_data[0:max_iter], a_baseline_early_stopping_means[0:max_iter], a_baseline_early_stopping_stds[0:max_iter],
        a_baseline_caught_data[0:max_iter], a_baseline_caught_means[0:max_iter], a_baseline_caught_stds[0:max_iter],

        a_baseline_steps_data[0:max_iter], a_baseline_steps_means[0:max_iter], a_baseline_steps_stds[0:max_iter],

        t_baseline_i_steps_data[0:max_iter], t_baseline_i_steps_means[0:max_iter], t_baseline_i_steps_stds[0:max_iter],
        a_baseline_i_steps_data[0:max_iter], a_baseline_i_steps_means[0:max_iter], a_baseline_i_steps_stds[0:max_iter],

        avg_train_i_steps_data_v3[0:max_iter], avg_train_i_steps_means_v3[0:max_iter], avg_train_i_steps_stds_v3[0:max_iter],
        avg_train_i_steps_data_v2[0:max_iter], avg_train_i_steps_means_v2[0:max_iter], avg_train_i_steps_stds_v2[0:max_iter],

        fontsize= 6.5, figsize= (7.5, 1.5), title_fontsize=8, lw=0.75, wspace=0.17, hspace=0.4, top=0.0,
        bottom=0.28, labelsize=6, markevery=25, optimal_reward = 100, sample_step = 2,
        eval_only=False, plot_opt = False, iterations_per_step= 10, optimal_int = 1.0,
        optimal_flag = 1.0, file_name = "flags_int_steps_r_costs_alerts_defender_cnsm_21_2", markersize=2.25
    )


if __name__ == '__main__':
    base_path_1 = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/backup_pycr_cnsm_21_22_may/v2_results/results_1/data/"
    base_path_2 = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/backup_pycr_cnsm_21_22_may/v3_results/results_1/data/"
    base_path_3 = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/new_results/v2_results/results_1/data/"
    base_path_4 = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/new_results/v3_results/results_1/data/"
    avg_train_rewards_data_v3, avg_train_rewards_means_v3, avg_train_rewards_stds_v3, \
    avg_train_steps_data_v3, avg_train_steps_means_v3, avg_train_steps_stds_v3, \
    avg_train_caught_frac_data_v3, avg_train_caught_frac_means_v3, avg_train_caught_frac_stds_v3, \
    avg_train_early_stopping_frac_data_v3, avg_train_early_stopping_means_v3, \
    avg_train_early_stopping_stds_v3, avg_train_intrusion_frac_data_v3, avg_train_intrusion_means_v3, \
    avg_train_intrusion_stds_v3, \
    avg_train_rewards_data_v2, avg_train_rewards_means_v2, avg_train_rewards_stds_v2, \
    avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2, \
    avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2, avg_train_caught_frac_stds_v2, \
    avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2, \
    avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2, avg_train_intrusion_means_v2, \
    avg_train_intrusion_stds_v2, optimal_rewards_v3_data, optimal_rewards_v3_means, optimal_rewards_v3_stds, \
    optimal_steps_v3_data, optimal_steps_v3_means, optimal_steps_v3_stds, \
    optimal_steps_v2_data, optimal_steps_v2_means, optimal_steps_v2_stds, \
    optimal_rewards_v2_data, optimal_rewards_v2_means, optimal_rewards_v2_stds, \
    avg_train_i_steps_data_v3, avg_train_i_steps_means_v3, avg_train_i_steps_stds_v3,\
    avg_train_i_steps_data_v2, avg_train_i_steps_means_v2, avg_train_i_steps_stds_v2 = \
        parse_data(base_path_1=base_path_1, base_path_2=base_path_2, base_path_3=base_path_3,
                   base_path_4=base_path_4, suffix="gensim")

    plot_train(avg_train_rewards_data_v3, avg_train_rewards_means_v3, avg_train_rewards_stds_v3,
    avg_train_steps_data_v3, avg_train_steps_means_v3, avg_train_steps_stds_v3,
    avg_train_caught_frac_data_v3, avg_train_caught_frac_means_v3, avg_train_caught_frac_stds_v3,
    avg_train_early_stopping_frac_data_v3, avg_train_early_stopping_means_v3,
    avg_train_early_stopping_stds_v3, avg_train_intrusion_frac_data_v3, avg_train_intrusion_means_v3,
    avg_train_intrusion_stds_v3,
    avg_train_rewards_data_v2, avg_train_rewards_means_v2, avg_train_rewards_stds_v2,
    avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2,
    avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2, avg_train_caught_frac_stds_v2,
    avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2,
    avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2, avg_train_intrusion_means_v2,
    avg_train_intrusion_stds_v2,
    optimal_rewards_v3_data, optimal_rewards_v3_means, optimal_rewards_v3_stds,
    optimal_steps_v3_data, optimal_steps_v3_means, optimal_steps_v3_stds,
    optimal_steps_v2_data, optimal_steps_v2_means, optimal_steps_v2_stds,
    optimal_rewards_v2_data, optimal_rewards_v2_means, optimal_rewards_v2_stds,
    avg_train_i_steps_data_v3, avg_train_i_steps_means_v3, avg_train_i_steps_stds_v3,
    avg_train_i_steps_data_v2, avg_train_i_steps_means_v2, avg_train_i_steps_stds_v2)

