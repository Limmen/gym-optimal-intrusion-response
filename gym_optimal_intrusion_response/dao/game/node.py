import numpy as np


class Node:

    def __init__(self, initial_defense_attributes = None, initial_attack_attributes = None,
                 num_attributes : int = 1, max_attribute_value :int = 100, target_component : bool = False):
        self.initial_defense_attributes = initial_defense_attributes
        self.initial_attack_attributes = initial_attack_attributes
        if self.initial_defense_attributes is None:
            self.initial_defense_attributes = np.zeros(num_attributes)
        if self.initial_attack_attributes is None:
            self.initial_attack_attributes = np.zeros(num_attributes)
        self.num_attributes = num_attributes
        self.max_attribute_value = max_attribute_value
        self.compromised = False
        self.attack_attributes = list(np.zeros(num_attributes))
        self.defense_attributes = list(np.zeros(num_attributes))
        self.initialize_attributes()
        self.recon_done = False
        self.target_component = target_component


    def attack(self, attribute_id):
        if self.attack_attributes[attribute_id] < self.max_attribute_value:
            self.attack_attributes[attribute_id] += 1
        if self.attack_attributes[attribute_id] > self.defense_attributes[attribute_id]:
            self.compromised = True

    def recon(self):
        self.recon_done = True


    def initialize_attributes(self):
        for i in range(self.num_attributes):
            self.attack_attributes[i] = self.initial_attack_attributes[i]
            self.defense_attributes[i] = self.initial_defense_attributes[i]
