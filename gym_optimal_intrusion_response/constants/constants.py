"""
Constants for the environments
"""

class DP2:
    MAX_SEVERE_ALERTS = 20
    MAX_WARNING_ALERTS = 20
    MAX_TIMESTEPS = 20
    MAX_LOGINS = 0
    SERVICE_REWARD = 10
    ATTACK_REWARD = -100
    EARLY_STOPPING_REWARD = -100
    STOPPING_REWARD = 100

class DP:
    MAX_ALERTS = 100
    MAX_LOGINS = 100
    MAX_TIMESTEPS = 20
    # MAX_TTC = 200
    MAX_TTC = 20
    MIN_TTC = 0
    SERVICE_REWARD = 10
    ATTACK_REWARD = -65
    EARLY_STOPPING_REWARD = -10

class TRACES:
    SERVICE_REWARD = 10
    ATTACK_REWARD = -100
    EARLY_STOPPING_REWARD = -100
    MAX_TIMESTEPS = 100


class ACTIONS:
    STOPPING_ACTION = 1