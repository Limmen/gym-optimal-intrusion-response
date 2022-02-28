import random
from itertools import count, tee, islice
import math
import random
import time
import numpy as np
import sys
import csv
import tnsm_21_revision.stopping_pomdp as stopping_pomdp
import tnsm_21_revision.hsvi as hsvi

# A simple function that returns its argument
identity = lambda x: x


def set_seed(seed: float) -> None:
    """
    Deterministic seed config

    :param seed: random seed for the PRNG
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)


def initial_theta(L: int) -> np.ndarray:
    theta_0 = []
    upper_bound = 3
    for k in range(L):
        upper_bound = np.random.uniform(-3, upper_bound)
        theta_0.append(upper_bound)
    theta_0 = np.array(theta_0)
    return theta_0


def sample_next_state(T: np.ndarray, s: int, a: int, S: np.ndarray) -> int:
    state_probs = []
    for s_prime in S:
        state_probs.append(T[a][s][s_prime])
    # print(f"state probs:{state_probs}")
    s_prime = np.random.choice(np.arange(0, len(S)), p=state_probs)
    return s_prime


def sample_next_observation(Z: np.ndarray, s_prime: int, a: int, O: np.ndarray) -> int:
    observation_probs = []
    for o in O:
        observation_probs.append(Z[a][s_prime][o])
    o = np.random.choice(np.arange(0, len(O)), p=observation_probs)
    return o

def sigmoid(x, upper_bound):
    return 1/(1 + math.exp(-x))
    # return 1/(1 + math.exp(-x))*(upper_bound)


def get_thresholds(theta):
    upper_bound = 1
    thresholds = []
    for k in range(len(theta)):
        upper_bound = sigmoid(theta[k], upper_bound=upper_bound)
        thresholds.append(upper_bound)
    return thresholds


def save_csv_files(times, values, optimal_values, thetas, early_stopping_probs,
                   intrusion_prevented_probs, episode_lenghts, intrusion_lengths,
                   snort_rewards, snort_intrusion_prevented_probs, snort_early_stopping_probs,
                   snort_ep_lens, optimal_ep_lens, snort_intrusion_lens, file_name, L):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        labels = ["t", "J", "J_opt", "early_stopping_prob", "intrusion_prevented_prob", "T", "intrusion_length",
                  "snort_R", "snort_intrusion_prevented_prob", "snort_early_stopping_prob", "snort_ep_len",
                  "optimal_ep_len", "snort_intrusion_len",]
        for l in range(L):
            labels.append(f"alpha_{l}")
        writer.writerow(labels)
        for i in range(len(times)):
            row = [times[i], values[i], optimal_values[i], early_stopping_probs[i], intrusion_prevented_probs[i],
                   episode_lenghts[i], intrusion_lengths[i], snort_rewards[i], snort_intrusion_prevented_probs[i],
                   snort_early_stopping_probs[i], snort_ep_lens[i], optimal_ep_lens[i], snort_intrusion_lens[i]]
            for l in range(L):
                row.append(round(thetas[i][l],5))
            writer.writerow(row)

def SPSA(t0, iterations,alpha, gamma, c, a, delta, b0, s0, Z, A, O, S, eval_batch_size, L, start, discount_factor = 1, constraint=identity):
    theta = t0
    thetas = []
    values = []
    times = []
    optimal_values = []
    early_stopping_probs = []
    intrusion_prevented_probs = []
    episode_lengths = []
    intrusion_lengths = []
    snort_rewards = []
    snort_ep_lens = []
    snort_intrusion_prevented_probs = []
    snort_early_stopping_probs = []
    snort_intrusion_lens = []
    optimal_ep_lens = []
    J, J_opt, early_stopping_prob, intrusion_prevented_prob, avg_ep_len, avg_intrusion_len, avg_snort_reward, \
    snort_early_stopping_prob, snort_intrusion_prevented_prob, avg_snort_ep_len, avg_snort_intrusion_len, \
    avg_optimal_ep_len = \
        eval_theta(theta=theta, b0=b0, L=L, s0=s0, Z=Z, O=O, S=S, batch_size = eval_batch_size,
                         sigmoid_transform=True, gamma=discount_factor)
    J = round(J, 5)
    J_opt = round(J_opt, 5)
    values.append(J)
    optimal_values.append(J_opt)
    t = round((time.time()-start)/60.0, 3)
    times.append(t)
    thetas.append(get_thresholds(theta=theta))
    early_stopping_probs.append(early_stopping_prob)
    intrusion_prevented_probs.append(intrusion_prevented_prob)
    episode_lengths.append(avg_ep_len)
    intrusion_lengths.append(avg_intrusion_len)
    snort_rewards.append(avg_snort_reward)
    snort_early_stopping_probs.append(snort_early_stopping_prob)
    snort_intrusion_prevented_probs.append(snort_intrusion_prevented_prob)
    snort_ep_lens.append(avg_snort_ep_len)
    optimal_ep_lens.append(avg_optimal_ep_len)
    snort_intrusion_lens.append(avg_snort_intrusion_len)
    print(f"i:{0}, theta1:{get_thresholds(theta=theta)}, J:{J}, J_Opt:{J_opt}, early_stopping: {early_stopping_prob}, "
          f"ep len:{avg_ep_len}, intrusion len:{avg_intrusion_len}"
          f"t:{t}, snort_R:{avg_snort_reward}, snort_ep_len:{avg_snort_ep_len}, snort_es:{snort_early_stopping_prob}, "
          f"snort_intrusion_len:{avg_snort_intrusion_len}"
          f"optimal_ep_len:{avg_optimal_ep_len}")

    for i in range(iterations):
        ak = standard_ak(a=a, A=A, alpha=alpha, k=i)
        ck = standard_ck(c=c, gamma=gamma, k=i)
        # Get estimated gradient
        gk = estimate_gk(theta, delta, ck, b0, s0, Z, O, S, eval_batch_size, L)

        # Adjust theta using SA
        theta = [t + ak * gkk for t, gkk in zip(theta, gk)]

        # Constrain
        theta = constraint(theta)
        # print(theta)
        for j in range(len(theta)-1):
            if theta[j+1] > theta[j]:
                # print("sec upd")
                if random.random() < 0.8:
                    theta[j] = theta[j+1] + random.random()/10
        J, J_opt, early_stopping_prob, intrusion_prevented_prob, avg_ep_len, avg_intrusion_len, \
        avg_snort_reward, snort_early_stopping_prob, snort_intrusion_prevented_prob, \
        avg_snort_ep_len, avg_snort_intrusion_len, avg_optimal_ep_len = \
            eval_theta(theta=theta, b0=b0, L=L, s0=s0, Z=Z, O=O, S=S, batch_size = eval_batch_size,
                       gamma=discount_factor, sigmoid_transform=True)
        J = round(J, 5)
        J_opt = round(J_opt, 5)
        t = round((time.time()-start)/60.0, 3)
        times.append(t)
        values.append(J)
        optimal_values.append(J_opt)
        thetas.append(get_thresholds(theta=theta))
        early_stopping_probs.append(early_stopping_prob)
        episode_lengths.append(avg_ep_len)
        intrusion_prevented_probs.append(intrusion_prevented_prob)
        intrusion_lengths.append(avg_intrusion_len)
        snort_rewards.append(avg_snort_reward)
        snort_intrusion_prevented_probs.append(snort_intrusion_prevented_prob)
        snort_early_stopping_probs.append(snort_early_stopping_prob)
        snort_ep_lens.append(avg_snort_ep_len)
        optimal_ep_lens.append(avg_optimal_ep_len)
        snort_intrusion_lens.append(avg_snort_intrusion_len)
        print(f"i:{i}, theta:{get_thresholds(theta=theta)}, J:{J}, J_Opt:{J_opt}, "
              f"early_stopping: {early_stopping_prob}, t:{t}, int len:{avg_intrusion_len}, ep len:{avg_ep_len},"
              f"snort_rew:{avg_snort_reward}, snort_es:{snort_early_stopping_prob}, snort_ep_len:{avg_snort_ep_len}, "
              f"avg_snort_intrusio_len:{avg_snort_intrusion_len}, avg_optimal_len:{avg_optimal_ep_len}")

    return thetas, values, times, optimal_values, early_stopping_probs, intrusion_prevented_probs, \
           episode_lengths, intrusion_lengths, snort_rewards, snort_intrusion_prevented_probs, \
           snort_early_stopping_probs, snort_ep_lens, optimal_ep_lens, snort_intrusion_lens


def estimate_gk(theta, delta, ck, b0, s0, Z, O, S, eval_batch_size, L):
    '''Helper function to estimate gk from SPSA'''
    # Generate Delta vector
    delta_k = delta()

    # Get the two perturbed values of theta
    # list comprehensions like this are quite nice
    ta = [t + ck * dk for t, dk in zip(theta, delta_k)]
    tb = [t - ck * dk for t, dk in zip(theta, delta_k)]

    # Calculate g_k(theta_k)
    ya, _, _, _, _, _, _, _, _, _, _, _ = eval_theta(theta=ta, b0=b0, L=L, s0=s0, Z=Z, O=O, S=S, batch_size = eval_batch_size, sigmoid_transform=True)
    yb, _, _, _, _, _, _, _, _, _, _, _ = eval_theta(theta=tb, b0=b0, L=L, s0=s0, Z=Z, O=O, S=S, batch_size = eval_batch_size, sigmoid_transform=True)
    ya = round(ya, 5)
    yb = round(yb, 5)
    gk = [(ya-yb) / (2*ck*dk) for dk in delta_k]

    return gk


def standard_ak(a, A, alpha, k):
    '''Create a generator for values of a_k in the standard form.'''
    # Parentheses makes this an iterator comprehension
    # count() is an infinite iterator as 0, 1, 2, ...
    return a / (k + 1 + A) ** alpha


def standard_ck(c, gamma, k):
    '''Create a generator for values of c_k in the standard form.'''
    return  c / (k + 1) ** gamma


class Bernoulli:
    '''
    Bernoulli Perturbation distributions.
    p is the dimension
    +/- r are the alternate values
    '''
    def __init__(self, r=1, p=2):
        self.p = p
        self.r = r

    def __call__(self):
        return [random.choice((-self.r, self.r)) for _ in range(self.p)]


def run_spsa(iterations=1000, replications=40):
    L = 3
    s0 = stopping_pomdp.initial_state()
    b0 = stopping_pomdp.initial_belief()
    O_novice, O_experienced, O_expert, Z_novice, Z_experienced, Z_expert = stopping_pomdp.observation_tensor_and_space("/home/kim/tnsm_21_revision/tnsm_21_revision/")
    A, _ = stopping_pomdp.actions()
    S, _ = stopping_pomdp.states()
    discount = 1

    p = 1
    c=1
    gamma=.101
    alpha = .602
    A=100
    a=1

    delta = Bernoulli(p=p)
    eval_batch_size = 100

    # seeds = [0, 399, 999]
    seeds = [0, 399, 999]

    L = 1
    Z = Z_novice
    O = O_novice
    theta0 = [-4]
    for seed in seeds:
        set_seed(seed)
        start = time.time()
        thetas, values, times, optimal_values, early_stopping_probs, intrusion_prevented_probs, \
        episode_lengths, intrusion_lengths, snort_rewards, snort_intrusion_prevented_probs, snort_early_stopping_probs, \
        snort_ep_lens, optimal_ep_lens, snort_intrusion_lens \
            = SPSA(iterations=iterations, t0=theta0, delta=delta, b0=b0, s0=s0, Z=Z, A=A, O=O, S=S,
                   eval_batch_size=eval_batch_size, L=L,discount_factor=discount, start=start, alpha=alpha, gamma=gamma, c=c, a=a)
        save_csv_files(times=times, values=values, optimal_values=optimal_values, thetas=thetas,
                       early_stopping_probs = early_stopping_probs,
                       intrusion_prevented_probs = intrusion_prevented_probs,
                       episode_lenghts=episode_lengths,
                       intrusion_lengths=intrusion_lengths,
                       snort_rewards = snort_rewards,
                       snort_intrusion_prevented_probs=snort_intrusion_prevented_probs,
                       snort_early_stopping_probs=snort_early_stopping_probs,
                       snort_ep_lens=snort_ep_lens, optimal_ep_lens=optimal_ep_lens,
                       snort_intrusion_lens=snort_intrusion_lens,
                       file_name=f"novice_spsa_{seed}.csv" , L=L)

    L = 2
    Z = Z_experienced
    O = O_experienced
    theta0 = [-4, -4]
    for seed in seeds:
        set_seed(seed)
        start = time.time()
        thetas, values, times, optimal_values, early_stopping_probs, intrusion_prevented_probs, \
        episode_lengths, intrusion_lengths, snort_rewards, snort_intrusion_prevented_probs, snort_early_stopping_probs, \
        snort_ep_lens, optimal_ep_lens, snort_intrusion_lens \
            = SPSA(iterations=iterations, t0=theta0, delta=delta, b0=b0, s0=s0, Z=Z, A=A, O=O, S=S,
                   eval_batch_size=eval_batch_size, L=L,discount_factor=discount, start=start, alpha=alpha, gamma=gamma, c=c, a=a)
        save_csv_files(times=times, values=values, optimal_values=optimal_values, thetas=thetas,
                       early_stopping_probs = early_stopping_probs,
                       intrusion_prevented_probs = intrusion_prevented_probs,
                       episode_lenghts=episode_lengths,
                       intrusion_lengths=intrusion_lengths,
                       snort_rewards = snort_rewards,
                       snort_intrusion_prevented_probs=snort_intrusion_prevented_probs,
                       snort_early_stopping_probs=snort_early_stopping_probs,
                       snort_ep_lens=snort_ep_lens, optimal_ep_lens=optimal_ep_lens,
                       snort_intrusion_lens=snort_intrusion_lens,
                       file_name=f"experienced_spsa_{seed}.csv" , L=L)


    Z = Z_expert
    O = O_expert
    theta0 = [-4, -4, -4]
    for seed in seeds:
        set_seed(seed)
        start = time.time()
        thetas, values, times, optimal_values, early_stopping_probs, intrusion_prevented_probs, \
        episode_lengths, intrusion_lengths, snort_rewards, snort_intrusion_prevented_probs, snort_early_stopping_probs, \
        snort_ep_lens, optimal_ep_lens, snort_intrusion_lens \
            = SPSA(iterations=iterations, t0=theta0, delta=delta, b0=b0, s0=s0, Z=Z, A=A, O=O, S=S,
                   eval_batch_size=eval_batch_size, L=L,discount_factor=discount, start=start, alpha=alpha, gamma=gamma, c=c, a=a)
        save_csv_files(times=times, values=values, optimal_values=optimal_values, thetas=thetas,
                       early_stopping_probs = early_stopping_probs,
                       intrusion_prevented_probs = intrusion_prevented_probs,
                       episode_lenghts=episode_lengths,
                       intrusion_lengths=intrusion_lengths,
                       snort_rewards = snort_rewards,
                       snort_intrusion_prevented_probs=snort_intrusion_prevented_probs,
                       snort_early_stopping_probs=snort_early_stopping_probs,
                       snort_ep_lens=snort_ep_lens, optimal_ep_lens=optimal_ep_lens,
                       snort_intrusion_lens=snort_intrusion_lens,
                       file_name=f"expert_spsa_{seed}.csv" , L=L)

    return values

if __name__ == '__main__':
    run_spsa(iterations=300, replications=1)
