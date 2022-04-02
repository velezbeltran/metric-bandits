"""
Convention
----------
last dimension of action is the proposed distance.
"""

import torch

from metric_bandits.constants.constants import SEED
from metric_bandits.data.mnist import MNIST
from metric_bandits.envs.base_env import BaseEnv


class MNISTEnv(BaseEnv):
    def __init__(self, algo, T, batch_size, persistence):
        """
        Initializes the environment

        Args:
            batch_size: size of the batch to use for training
            persistence: how many rounds to keep the same dataset for
        """
        # set seed
        torch.manual_seed(SEED)

        super().__init__(MNIST, algo, T)
        self.persistence = persistence
        self.batch_size = batch_size
        self.idx = {}
        self.init_data()

    def next_actions(self):
        """
        Returns a list of next available actions which are represented
        as a vector of (img, img, prop_distance). Every `persistence`
        a new set of actions is returned
        """
        self.real_distances = {}  # keeps track of distances to obtain reward

        # Get new set of actions
        if self.t % self.persistence == 0:
            cur_idx = self.t // self.persistence
            b_idxs = self.idx[self.mode][cur_idx : cur_idx + self.batch_size]
            batch = [self.data[i] for i in b_idxs]

            self.current_actions = []
            for imgx, labelx in batch:
                for imgy, labely in batch:
                    for prop_distance in range(10):
                        if not torch.eq(imgx, imgy).all():
                            context_partial = torch.cat(
                                (imgx.flatten(), imgy.flatten())
                            )
                            context_full = torch.cat(
                                (context_partial, torch.tensor([prop_distance]))
                            )
                            self.current_actions.append(context_full)
                            self.real_distances[context_partial] = abs(labelx - labely)

        return self.current_actions

    def step(self, action):
        """
        Returns the reward for the action taken
        """
        return 1.0
        context_partial, prop_distance = action[:-1], action[-1]
        real_distance = self.real_distances[context_partial]
        reward = 1 - (real_distance - prop_distance) / 10
        self.t += 1
        return reward

    def init_data(self):
        """
        Initializes the data loader
        """
        train_len = int(len(self.data) * 0.8)
        permutaion = torch.randperm(len(self.data))

        self.idx = {}
        self.idx["train"] = permutaion[:train_len]
        self.idx["test"] = permutaion[train_len:]
