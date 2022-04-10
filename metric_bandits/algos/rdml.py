"""
The implementation of  RDML - regularised distance metric learning, 
follows the online-reg algorithm described in 
the paper:
`https://papers.nips.cc/paper/2009/file/a666587afda6e89aec274a3657558a27-Paper.pdf`
"""

# import statements here
from metric_bandits.algos.base import BaseAlgo
from metric_bandits.utils.math import square_matrix_norm

import numpy as np 

class RDML(BaseAlgo):
    def __init__(
        self,
        learning_rate
    ):
        self.learning_rate = learning_rate
        self.metric = None
        self.last_action = None
        self.contexts_played = []
        self.rewards = []

    def choose_action(self, actions):
        """
        actions is a list-like object dictionary and contains the available actions.

        Note that the actions are of the form (data_1, data_2, similar/dissimilar).
        """
        for index, _action in actions.items():
            half = len(_action)//2
            x_1, x_2, y_act = np.array(_action[:half]), np.array(_action[half:-1]),int(_action[-1])
            x_1 = x_1.reshape((len(x_1), 1))
            x_2 = x_2.reshape((len(x_2), 1))
            if self.metric is None:
                self.metric = np.identity(len(x_1))

            y_hat = 1 if 1 - square_matrix_norm(self.metric, x_1 - x_2) ** 2 > 0 else -1

            if y_hat == y_act:
                self.contexts_played.append(_action)
                self.last_action = index
                break
        
        return self.last_action

    def update(self, reward):
        """
        Updates the model
        """
        self.rewards.append(reward)

        # If reward is 1, then the identification was correct
        # metric stays the same.
        if reward == 1:
            self.metric = self.metric
        
        # Else we were wrong. We update the matrix 
        else:
            _lambda = self.learning_rate
        
            # If y_act = +1, means y was -1. _lambda stays the same
            # If y_act = -1, means y was +1, then we update update lambda via the following
            half = len(self.contexts_played[-1])//2
            context = self.contexts_played[-1]
            x_1, x_2, y_act = np.array(context[:half]), np.array(context[half:-1]), context[-1]
            x_1 = x_1.reshape((len(x_1), 1))
            x_2 = x_2.reshape((len(x_2), 1))
            x_diff = x_1 - x_2
            if y_act == -1:

                if self.metric is None:
                    self.metric = np.identity(len(x_1))

                _lambda = min(
                            self.learning_rate,
                             1 / square_matrix_norm(np.linalg.pinv(self.metric), x_diff)
                             )
            # Now update the metric:
            self.metric = self.metric - _lambda * (- y_act) * x_diff @ x_diff.T


    def reset(self):
        """
        Resets the model
        """
        self.metric = None
        print("Reset model.")
