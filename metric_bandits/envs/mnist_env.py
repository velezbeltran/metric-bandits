"""
Convention
----------
last dimension of action is the proposed distance.
"""

import uuid

import torch

from metric_bandits.constants.constants import SEED
from metric_bandits.data.mnist import MNIST, make_pca_mnist
from metric_bandits.envs.base_env import BaseEnv


class MNISTEnv(BaseEnv):
    def __init__(self, algo, T, batch_size, persistence, pca=False):
        """
        Initializes the environment

        Args:
            batch_size: size of the batch to use for training
            persistence: how many rounds to keep the same dataset for
        """
        # set seed
        torch.manual_seed(SEED)
        data = MNIST if not pca else make_pca_mnist(MNIST)
        super().__init__(data, algo, T)
        self.persistence = persistence
        self.batch_size = batch_size
        self.idx = {}
        self.init_data()
        self.rewards = []

    def next_actions(self):
        """
        Returns a list of next available actions which are represented
        as a vector of (img, img, prop_distance). Every `persistence`
        a new set of actions is returned
        """

        # Get new set of actions
        if self.t % self.persistence == 0:
            cur_idx = self.t // self.persistence

            # choose the next batch of images to use for training
            b_idxs = self.idx[self.mode][cur_idx : cur_idx + self.batch_size]
            batch = [self.data[i] for i in b_idxs]

            # store stuff to interpret returned action
            self.current_actions = {}
            self.proposed_distances = {}
            self.real_distances = {}

            # produce the actions
            for imgx, labelx in batch:
                for imgy, labely in batch:
                    for prop_distance in range(10):
                        if not torch.eq(imgx, imgy).all():
                            context_partial = torch.cat(
                                (imgx.flatten(), imgy.flatten())
                            )
                            context_full = self.make_full_vector(
                                context_partial, prop_distance
                            )
                            id = str(uuid.uuid4())
                            self.current_actions[id] = context_full
                            self.real_distances[id] = abs(labelx - labely)
                            self.proposed_distances[id] = prop_distance

        return self.current_actions

    def step(self, action):
        """
        Returns the reward for the action taken
        """
        real_distance = self.real_distances[action]
        prop_distance = self.proposed_distances[action]
        reward = 1 - abs((real_distance - prop_distance) / 10)
        self.t += 1
        return reward

    def make_full_vector(self, vector, prop_distance):
        """
        Makes the vector full by adding the distance to the context
        """
        number_arms = 10  # 10 possible distances
        v_size = len(vector)
        full_v = torch.zeros(v_size * number_arms)
        full_v[prop_distance * v_size : (prop_distance + 1) * v_size] = vector
        return full_v

    def init_data(self):
        """
        Initializes the data loader
        """
        train_len = int(len(self.data) * 0.8)
        permutaion = torch.randperm(len(self.data))

        self.idx = {}
        self.idx["train"] = permutaion[:train_len]
        self.idx["test"] = permutaion[train_len:]

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