"""
This is an implementation of LinUCB: UCB with linear hypothesis
"""

from math import sqrt
from metric_bandits.algos.base import BaseAlgo
import numpy as np
from metric_bandits.utils.math import _inv, square_matrix_norm

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
            self.A = np.identity(self.dim)
            self.b = np.zeros(self.dim)
        
        self.theta = _inv(self.A) @ self.b

        # Loop through all available action, context pairs
        for action in actions:
            self.ucb_estimate[action] = self.get_ucb_estimate(actions[action])
        
        # return the key with the highest value
        self.last_action = max(self.ucb_estimate, key=self.ucb_estimate.get)
        self.contexts_played.append(actions[self.last_action])
        
        return self.last_action


    def get_ucb_estimate(self, context):
        mean = self.theta.T @ context
        upper_dev = self.beta * sqrt(square_matrix_norm(_inv(self.A), context))

        return mean + upper_dev

    def update(self, reward):
        """
        Updates the model
        reward is 1 if metric was correct, else 0
        """
        self.rewards.append(reward)

        latest_context = self.contexts_played[-1]

        # Update the model params:
        self.A = self.A + latest_context @ latest_context.T
        self.b = self.b + latest_context * reward

        self.t += 1

    def reset(self):
        """
        Resets the model
        """
        self.A = None
        self.b = None
        self.dim = None
        print("Reset model.")



