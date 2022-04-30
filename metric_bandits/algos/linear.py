"""
Implements the epsilon greedy strategy for the linear case.
"""

import random
from collections import defaultdict

import torch

from metric_bandits.algos.base import BaseAlgo
from metric_bandits.utils.math import sherman_morrison


class Linear(BaseAlgo):
    def __init__(
        self,
        input_dim,
        explore_param,
        reg=1.0,
        verbose=True,
    ):
        """
        If active is true, the model forgets completely about regret and just takes actions
        with the aim of maximizing information gain.
        """
        super().__init__()
        self.explore_param = explore_param
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.input_dim = input_dim
        self.reg = reg

        # state of the algorithm
        self.Z_inv = None
        self.t = 0
        self.train_t = 0

        # parameters to keep track of
        self.last_action = None
        self.rewards = []

    def choose_action(self, actions):
        """
        actions is a list-like object dictionary and contains the available actions
        """
        greedy = random.random() > self.explore_param

        if not greedy:
            items = list(actions.keys())
            random.shuffle(items)
            return items[0][0]

        self.ucb_vals = defaultdict(int)
        for action in actions:
            ctx = actions[action][:, :-1]  # last element is the action
            val = self.theta.T @ ctx
            self.ucb_val[action] += val

        # return the key with the highest value
        self.last_action = max(self.ucb_estimate, key=self.ucb_estimate.get)
        self.last_context = actions[self.last_action][:, :-1]
        return self.last_action

    def update(self, reward):
        """
        Updates the model
        """
        self.rewards.append(reward)

        # update our confidence matrix
        self.Z_inv = sherman_morrison(self.Z_inv, self.last_context)
        self.b = self.b + self.last_context * reward

        # decide whether to train the model
        self.t += 1

    @property
    def theta(self):
        return self.Z_inv @ self.b

    def reset(self):
        """
        Resets the model
        """
        self.Z_inv = self.reg * torch.eye(
            self.input_dim**2, requires_grad=False, device=self.device
        )
        self.b = torch.zeros(
            (self.input_dim**2, 1), requires_grad=False, device=self.device
        )
