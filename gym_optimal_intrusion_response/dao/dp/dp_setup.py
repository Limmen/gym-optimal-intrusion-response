import numpy as np
class DPSetup:

    def __init__(self, HP, R, T, next_state_lookahead, state_to_id, id_to_state):
        self.HP = HP
        self.R = R
        self.T = T
        self.next_state_lookahead = next_state_lookahead
        self.state_to_id = state_to_id
        self.id_to_state = id_to_state
        self.state_ids = np.array(list(range(T.shape[0])))

