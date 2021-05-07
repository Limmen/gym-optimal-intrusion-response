import math
from scipy.stats import poisson
import gym_optimal_intrusion_response.constants.constants as constants

class DefenderDynamics:

    @staticmethod
    def p1(v, m, k):
        return (1 - math.exp(-(v * m) / k))

    @staticmethod
    def ttc(v, s, k):
        if v == 0:
            return constants.DP.MAX_TTC
        s =s / constants.DP.MAX_LOGINS
        c1 = 1
        c2 = 5.8
        c3 = 32.42
        t1 = c1
        P1 = 1 - math.exp(-v * (DefenderDynamics.m_fun(s) / k))
        t2 = c2 * DefenderDynamics.E(v, s)
        t3 = (1 / DefenderDynamics.f_fun(s) - 0.5) * c3 + c2
        u = math.pow(1 - DefenderDynamics.f_fun(s), v)
        res = t1 * P1 + t2 * (1 - P1) * (1 - u) + t3 * u * (1 - P1)
        return res

    # @staticmethod
    # def hack_prob(ttc_val, t):
    #     min(0.5 * (t / ttc_val), 1.0)
    #     return min(0.5 * (t / ttc_val), 1.0)

    @staticmethod
    def hack_prob(ttc_val, t):
        # ttc_1 = min(25, ttc_val)
        ttc_1 = max(1, ttc_val)
        # hp = math.log((25 - ttc_1)/25)
        # hp = (1/math.pow(ttc_1, 2))
        hp = (1 / math.pow(ttc_1, 1))
        # hp = (1 / math.pow(0.55*ttc_1 + 0.45*(constants.DP.MAX_TIMESTEPS/2.2- t/2.2), 1))
        # hp = (1 / (0.9 * max(1, ttc_1) + 0.1 * max(1, ((constants.DP.MAX_TIMESTEPS) - (t)))))
        hp = (1 / (0.995 * math.pow(max(1, ttc_1),2) + 0.005 * math.pow(max(1, ((constants.DP.MAX_TIMESTEPS) - (t))), 2)))
        # hp = (25 - ttc_1)/25
        return hp

    @staticmethod
    def f_fun(s):
        return 0.145 * math.pow(2.6, 2 * s + 0.07) - 0.1

    @staticmethod
    def m_fun(s):
        return 83 * math.pow(3.5, (4 * s) / 2.7) - 82

    @staticmethod
    def E(v, s):
        temp = DefenderDynamics.xi(math.floor(DefenderDynamics.f_fun(s) * v), v) * (
                math.ceil(DefenderDynamics.f_fun(s) * v) - DefenderDynamics.f_fun(s) * v) \
               + DefenderDynamics.xi(math.ceil(DefenderDynamics.f_fun(s) * v), v) * (
                       1 - math.ceil(DefenderDynamics.f_fun(s) * v) + DefenderDynamics.f_fun(s) * v)
        return temp

    @staticmethod
    def xi(a, v):
        s = 0
        for t in range(math.ceil((v * (1 - a / v)) + 1)):
            x = t * (math.factorial(v - t + 1) / (math.factorial(v - a - t + 1) * (v - t + 1)))
            s = s + x

        res = a / v + ((a * (math.factorial(v - a))) / math.factorial(v)) * (s)
        return res


    @staticmethod
    def f1_a():
        return poisson(mu=3)

    @staticmethod
    def f2_a():
        return poisson(mu=0.5)

    @staticmethod
    def f1_b():
        return poisson(mu=1)

    @staticmethod
    def f2_b():
        return poisson(mu=0.25)
