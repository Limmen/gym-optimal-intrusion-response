import math
from scipy.stats import poisson
import gym_optimal_intrusion_response.constants.constants as constants


class DefenderDynamics:
    """
    Logic functions for the defender dynamics
    """

    @staticmethod
    def p1(v, m, k):
        """
        p1 parameter for the TTC
        :param v: v
        :param m: m
        :param k: k
        :return: p1
        """
        return (1 - math.exp(-(v * m) / k))

    @staticmethod
    def ttc(v, s, k):
        """
        TTC computation
        :param v: v
        :param s: s
        :param k: k
        :return: ttc
        """
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

    @staticmethod
    def hack_prob(ttc_val, t):
        """
        The hack probability

        :param ttc_val: the ttc
        :param t: the time-step
        :return:
        """
        ttc_1 = max(1, ttc_val)
        hp = (1 / (0.99 * math.pow(max(1, ttc_1),3) + 0.01 * math.pow(max(1, ((constants.DP.MAX_TIMESTEPS-1) - (t))), 2)))
        return hp

    @staticmethod
    def f_fun(s):
        """
        The f function for the TTC computation

        :param s: s
        :return: f(s)
        """
        return 0.145 * math.pow(2.6, 2 * s + 0.07) - 0.1

    @staticmethod
    def m_fun(s):
        """
        The m function for the TTC computation
        :param s: s
        :return: m(s)
        """
        return 83 * math.pow(3.5, (4 * s) / 2.7) - 82

    @staticmethod
    def E(v, s):
        """
        The E function for the TTC computation
        :param v: v
        :param s: s
        :return: E(v,s)
        """
        temp = DefenderDynamics.xi(math.floor(DefenderDynamics.f_fun(s) * v), v) * (
                math.ceil(DefenderDynamics.f_fun(s) * v) - DefenderDynamics.f_fun(s) * v) \
               + DefenderDynamics.xi(math.ceil(DefenderDynamics.f_fun(s) * v), v) * (
                       1 - math.ceil(DefenderDynamics.f_fun(s) * v) + DefenderDynamics.f_fun(s) * v)
        return temp

    @staticmethod
    def xi(a, v):
        """
        The xi function for the TTC computation

        :param a: a
        :param v: v
        :return: xi(a,v)
        """
        s = 0
        for t in range(math.ceil((v * (1 - a / v)) + 1)):
            x = t * (math.factorial(v - t + 1) / (math.factorial(v - a - t + 1) * (v - t + 1)))
            s = s + x

        res = a / v + ((a * (math.factorial(v - a))) / math.factorial(v)) * (s)
        return res


    @staticmethod
    def f1_a():
        """
        :return: f1_a() in the model
        """
        return poisson(mu=3)

    @staticmethod
    def f2_a():
        """
        :return: f2_a() in the model
        """
        return poisson(mu=0.5)

    @staticmethod
    def f1_b():
        """
        :return: f1_b() in the model
        """
        return poisson(mu=1)

    @staticmethod
    def f2_b():
        """
        :return: f2_b() in the model
        """
        return poisson(mu=0.25)
