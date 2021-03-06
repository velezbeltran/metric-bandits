"""
Convention
----------
last dimension of action is the +1 similarity / -1 dissimilarity.
"""

import uuid
from metric_bandits.envs.base_env import BaseEnv
from metric_bandits.data.wine import WINE
from metric_bandits.constants.constants import SEED
import numpy as np

class WineEnv(BaseEnv):
    def __init__(self, algo, T, batch_size, context="linear"):
        """
        Initializes the environment

        Args:
            batch_size: size of the batch to use for training
        """
        # Set dateset here 
        data = WINE
        super().__init__(data, algo, T)
        self.batch_size = batch_size
        self.idx = {}
        self.init_data()
        self.rewards = []
        self.context_type = context

    def next_actions(self):
        """
        Returns a list of next available actions which are represented
        as a vector of (data_1, data_2, similar(+1)/dissimilar(-1)). 
        """
        cur_idx = self.t % int(len(self.data) * 0.8 - 1) 
        # choose the next batch of images to use for training
        b_idxs = self.idx[self.mode][cur_idx : cur_idx + self.batch_size]
        batch = [self.data[i] for i in b_idxs]

        # store stuff to interpret returned action
        self.current_actions = {}
        self.proposed_distances = {}
        self.real_distances = {}
        # produce the actions
        for data_x, label_x in batch:
            for data_y, label_y in batch:
                if not np.array_equal(data_x, data_y):
                    for prop_distance in [+1,-1]:
                        context_full = self.make_context(self, data_x, data_y, prop_distance, _type=self.context_type)
                        id = str(uuid.uuid4())
                        self.current_actions[id] = context_full
                        self.real_distances[id] = 1 if label_x == label_y else -1
                        self.proposed_distances[id] = prop_distance
        
        return self.current_actions

    def make_context(self, data_x, data_y, prop_distance, _type="linear"):
        if _type == "linear":
            context_partial = np.concatenate(
                            [data_x, 
                            data_y]
                            )
            context_full = np.concatenate([
                            context_partial,
                            [prop_distance]]
                            )
            return context_full

        if _type == "quadratic":
            context_partial = [x*y for x,y in list(zip(data_x,data_y))]
            
            context_full = np.concatenate([
                            context_partial,
                            [prop_distance]]
                            )
            return context_full



    def step(self, action):
        """
        Returns the reward for the action taken
        """

        real_distance = self.real_distances[action]
        prop_distance = self.proposed_distances[action]
        reward = 1 if real_distance == prop_distance else 0
        self.t += 1
        return reward


        
    def init_data(self):
        """
        Initializes the data loader
        """
        train_len = int(len(self.data) * 0.8)
        permutation = np.random.RandomState(seed=SEED).permutation(len(self.data))

        self.idx = {}
        self.idx["train"] = permutation[:train_len]
        self.idx["test"] = permutation[train_len:]

    def reset(self):
        """
        Resets the environment
        """
        self.t = 0
        self.algo.reset()

    def update(self, r):
        """
        Updates the environment
        """
        self.rewards.append(r)
        self.cum_regrets.append((1 - r) + self.cum_regrets[-1])