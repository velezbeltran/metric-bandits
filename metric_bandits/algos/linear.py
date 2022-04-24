"""
Implements the epsilon greedy strategy for the linear case.
"""

from turtle import pos
from metric_bandits.algos.base import BaseAlgo
import random
import torch
from collections import defaultdict

import numpy as np
from metric_bandits.utils.math import square_matrix_norm, sherman_morrison

class Linear(BaseAlgo):
    def __init__(
        self,
        matrix_tuning_parameter=1,
        explore_param=1,
        verbose=False
    ):
        # Tuning parameter, set  to 1 by default.
        self.explore_param = explore_param
        self.verbose = verbose
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
        Returns the action to be taken
        """

        if self.t == 0:
            # Get the dimension of (x,x',a) vector
            self.dim = len(list(actions.values())[0])
            self.A = torch.eye(self.dim)
            self.A_inv = self.A
            self.b = torch.zeros(self.dim)

    
        self.theta = torch.matmul(self.A_inv, self.b)


        greedy = random.random() > self.explore_param

        if greedy:
            self.vals = {}
            for action in actions:
                val = self.theta @ actions[action]
                self.vals[action] = val
            self.last_action = max(self.vals, key=self.vals.get)
        else:
            self.last_action = random.choice(list(actions.keys()))
        self.contexts_played.append(actions[self.last_action])
        return self.last_action            

    def update(self, reward):
        """
        Updates the algorithm
        """
        
        self.rewards.append(reward)

        # decide whether to train the model

        # Unsqueezing here to fix dim error in downstream sherman morrison update.
        latest_context = self.contexts_played[-1]

        # Update the model params:
        self.A = self.A + latest_context @ latest_context.T
        self.A_inv = sherman_morrison(self.A_inv, latest_context.unsqueeze(-1))
        self.b = self.b + torch.mul(latest_context, reward)
        self.t += 1

    def estimate(self, actions):
        """
        Given an array of (x,y,a) contexts for a select x and y, produces  +1 or -1
        representing a similarity estimate of the current model
        """
        self.vals = {}
        for action in actions:
            val = self.theta @ actions[action]
            self.vals[action] = val

        return max(self.vals, key=self.vals.get)[-1]


    def reset(self):
        """
        Resets the model
        """
        self.A = None
        self.b = None
        self.dim = None
        print("Reset model.")