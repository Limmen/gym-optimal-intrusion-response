
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


class ACTIONS:
    STOPPING_ACTION = 1