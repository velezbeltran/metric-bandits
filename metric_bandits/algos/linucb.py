"""
This is an implementation of LinUCB: UCB with linear hypothesis
"""

from math import sqrt
import random
from collections import defaultdict
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
        explore_param=1,
        active=False
    ):
        # Tuning parameter, set  to 1 by default.
        self._lambda = matrix_tuning_parameter
        self.explore_param = explore_param

        # Starting state:
        self.A = None
        self.A_inv = None
        self.b = None
        self.t = 0
        self.theta = None
        self.dim = 0
        self.active=active

         # parameters to keep track of
        self.last_action = None
        self.rewards = []
        self.contexts_played = []


    def choose_action(self, actions):
        """
        actions is a list-like object dictionary and contains the available actions
        """
        if self.active:
            return self.choose_action_active(actions)
        else:
            return self.choose_action_default(actions)

    def choose_action_default(self, actions):
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

    def choose_action_active(self, actions):

        # At step 0, initialise all variables
        if self.t == 0:
            # Get the dimension of (x,x',a) vector
            self.dim = len(list(actions.values())[0])
            self.A = torch.eye(self.dim)
            self.A_inv = self.A
            self.b = torch.zeros(self.dim)

        self.theta = torch.matmul(self.A_inv, self.b)

        self.mean_estimate = defaultdict(list)
        self.ucb_estimate = defaultdict(int)
        self.unique_contexts = defaultdict(list)

        # Keep track of relevant values per unique context

        # We shuffle in case we have an optimist reward of zero
        items = list(actions.items())
        random.shuffle(items)
        for action, ctxt in items:
            ctxt = str(ctxt[:-1])
            mean = self.get_mean_estimate(actions[action])
            self.mean_estimate[ctxt].append(mean)
            self.ucb_estimate[ctxt] += self.get_opt_estimate(actions[action])
            self.unique_contexts[ctxt].append(action)

        # Choose to make a desicion on the pair with the highes opt value
        self.last_context = max(self.ucb_estimate, key=self.ucb_estimate.get)
        # choose the action with the highest value
        ctxt_val = self.mean_estimate[self.last_context]
        argmax = 0 if ctxt_val[0] > ctxt_val[1] else 1
        self.last_action = self.unique_contexts[self.last_context][argmax]
        self.contexts_played.append(actions[self.last_action])
        return self.last_action

    
    def get_mean_estimate(self, context):
        mean = torch.dot(self.theta, context)
        return mean

    def get_opt_estimate(self, context):
        upper_dev = self.explore_param * sqrt(square_matrix_norm(self.A_inv, context))
        return upper_dev
        

    def get_ucb_estimate(self, context):
        # Dont have self.theta.T here.
        mean = torch.dot(self.theta, context)
        upper_dev = self.explore_param * sqrt(square_matrix_norm(self.A_inv, context))

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



