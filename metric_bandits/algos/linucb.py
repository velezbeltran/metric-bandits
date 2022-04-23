"""
This is an implementation of LinUCB: UCB with linear hypothesis
"""

from math import sqrt
from metric_bandits.algos.base import BaseAlgo
import numpy as np
from metric_bandits.utils.math import square_matrix_norm, sherman_morrison

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinUCB(BaseAlgo):
    def __init__(
        self,
        matrix_tuning_parameter=1,
        ucb_parameter=1
    ):
        # Tuning parameter, set  to 1 by default.
        self._lambda = matrix_tuning_parameter
        self.beta = ucb_parameter

        # Starting state:
        self.A = None
        self.A_inv = None
        self.b = None
        self.t = 0
        self.theta = None
        self.dim = 0

         # parameters to keep track of
        self.last_action = None
        self.rewards = []
        self.contexts_played = []

    def choose_action(self, actions):
        """
        actions is a list-like object dictionary and contains the available actions
        """
        self.ucb_estimate = {}

        # At step 0, initialise all variables
        if self.t == 0:
            # Get the dimension of (x,x',a) vector
            self.dim = len(list(actions.values())[0])
            self.A = torch.eye(self.dim)
            self.A_inv = self.A
            self.b = torch.zeros(self.dim)

        
        self.theta = torch.matmul(self.A_inv, self.b)

        # Loop through all available action, context pairs
        for action in actions:
            self.ucb_estimate[action] = self.get_ucb_estimate(actions[action])
        # return the key with the highest value
        self.last_action = max(self.ucb_estimate, key=self.ucb_estimate.get)
        self.contexts_played.append(actions[self.last_action])
        
        return self.last_action


    def get_ucb_estimate(self, context):
        # Dont have self.theta.T here.
        mean = torch.dot(self.theta, context)
        upper_dev = self.beta * sqrt(square_matrix_norm(self.A_inv, context))

        return mean + upper_dev

    def update(self, reward):
        """
        Updates the model
        reward is 1 if metric was correct, else 0
        """
        self.rewards.append(reward)

        # Unsqueezing here to fix dim error in downstream sherman morrison update.
        latest_context = self.contexts_played[-1]

        # Update the model params:
        self.A = self.A + latest_context @ latest_context.T
        self.A_inv = sherman_morrison(self.A_inv, latest_context.unsqueeze(-1))
        self.b = self.b + torch.mul(latest_context, reward)
        self.t += 1

    def reset(self):
        """
        Resets the model
        """
        self.A = None
        self.b = None
        self.dim = None
        print("Reset model.")



