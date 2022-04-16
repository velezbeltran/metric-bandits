"""
Convention
----------
last dimension of action is the proposed distance.
"""


import torch

from metric_bandits.constants.data import TEST_NUM, TRAIN_NUM
from metric_bandits.data.moons import MOONS
from metric_bandits.envs.base_env import BaseEnv


class MoonsEnv(BaseEnv):
    def __init__(
        self,
        algo,
        T,
        batch_size,
        persistence,
        eval_freq=1000,
        to_eval=["knn, embedding"],
    ):
        """
        Initializes the environment

        Args:
            batch_size: size of the batch to use for training
            persistence: how many rounds to keep the same dataset for
        """
        # set seed
        data = MOONS
        # center and scale
        super().__init__(
            data=data, algo=algo, T=T, eval_freq=eval_freq, to_eval=to_eval
        )
        self.persistence = persistence
        self.batch_size = batch_size
        self.idx = {}
        self.init_data()
        self.rewards = []
        self.granularity = [i for i in range(10)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If algorithm is LinUCB, change settings:

    def next_actions(self):
        raise NotImplementedError

    def step(self, action):
        """
        Returns the reward for the action taken
        """
        raise NotImplementedError

    def init_data(self):
        """
        Initializes the data and creates the pretty version
        """
        X, Y = self.data
        train_len = int(len(X) * 0.8)
        perm = torch.randperm(len(X))

        self.idx = {}
        self.idx["train"] = perm[:train_len]
        self.idx["test"] = perm[train_len:]

        # create pretty version (i.e compatible with sklarn)
        self.X_train = X[self.idx["train"]].numpy()[:TRAIN_NUM]
        self.Y_train = Y[self.idx["train"]].numpy()[:TRAIN_NUM]
        self.X_test = X[self.idx["test"]].numpy()[:TEST_NUM]
        self.Y_test = Y[self.idx["test"]].numpy()[:TEST_NUM]
        self.nice_data_available = True

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


class MoonsSimEnv(MoonsEnv):
    def __init__(
        self,
        algo,
        T,
        batch_size,
        persistence,
        eval_freq=1000,
        possible_actions=[-1, 1],
        to_eval=["knn, embedding"],
    ):
        """
        Mnist environment

        Args:
            batch_size: size of the batch to use for training
            persistence: how many rounds to keep the same dataset for
        """
        super().__init__(
            algo, T, batch_size, persistence, eval_freq=eval_freq, to_eval=to_eval
        )
        self.possible_actions = possible_actions

    def next_actions(self):
        """
        Returns a list of next available actions which are represented
        as a vector of (img, img, prop_distance). Every `persistence` num steps
        a new set of actions is returned
        """

        # Get new set of actions
        if self.t % self.persistence == 0:
            cur_idx = self.t // self.persistence

            X, y = self.data
            idx = self.idx[self.mode][
                cur_idx * self.batch_size : (cur_idx + 1) * self.batch_size
            ]
            bX, by = X[idx], y[idx]

            # store stuff to interpret returned action
            self.current_actions = {}
            self.real_label = {}

            # produce the actions
            for i in range(len(bX)):
                for j in range(i + 1, len(bX)):
                    for a in self.possible_actions:
                        imgx, labelx = bX[i], by[i]
                        imgy, labely = bX[j], by[j]
                        imgx, imgy = imgx.flatten(), imgy.flatten()
                        context_partial = torch.cat((imgx, imgy, torch.tensor([a])))
                        context_partial = context_partial.to(self.device).float()
                        self.current_actions[context_partial] = context_partial
                        self.real_label[context_partial] = 2 * int(labelx == labely) - 1

        return self.current_actions

    def step(self, action):
        """
        Returns the reward for the action taken
        """
        prop_sim = self.current_actions[action][-1]
        real_sim = self.real_label[action]
        reward = prop_sim * real_sim
        self.t += 1
        return reward

    def update(self, r):
        """
        Updates the environment
        """
        self.rewards.append(r)
        self.cum_regrets.append((1 - r) + self.cum_regrets[-1])
