
class DP:
    MAX_ALERTS = 100
    MAX_LOGINS = 100
    MAX_TIMESTEPS = 50
    # MAX_TTC = 200
    MAX_TTC = 20
    MIN_TTC = 0
    SERVICE_REWARD = 1
    ATTACK_REWARD = -100
    EARLY_STOPPING_REWARD = -1


class ACTIONS:
    STOPPING_ACTION = 1