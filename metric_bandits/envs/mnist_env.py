"""
Convention
----------
last dimension of action is the proposed distance.
"""


import itertools
import uuid
from collections import defaultdict

import torch

from metric_bandits.algos.linucb import LinUCB
from metric_bandits.constants.data import TEST_NUM, TRAIN_NUM
from metric_bandits.data.mnist import MNIST, make_pca_mnist
from metric_bandits.envs.base_env import BaseEnv
from metric_bandits.utils.math import accumulate


class MNISTEnv(BaseEnv):
    def __init__(
        self,
        algo,
        T,
        batch_size,
        persistence,
        pca_dims=None,
        context=None,
        eval_freq=1000,
        to_eval=None,
        possible_actions=None,
    ):
        """
        Initializes the environment

        Args:
            batch_size: size of the batch to use for training
            persistence: how many rounds to keep the same dataset for
        """
        # set seed
        data = MNIST if not pca_dims else make_pca_mnist(MNIST, pca_dims)
        # center and scale
        super().__init__(data, algo, T, eval_freq, to_eval=to_eval)
        self.persistence = persistence
        self.batch_size = batch_size
        self.idx = {}
        self.init_data()
        self.rewards = []
        self.granularity = [i for i in range(10)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context = "linear" if context is None else context
        print(self.context)

        # If algorithm is LinUCB, change settings:
        if isinstance(self.algo, LinUCB):
            self.granularity = [-1, 1]

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
        train_len = int(len(self.data) * 0.8)
        permutaion = torch.randperm(len(self.data))

        self.idx = {}
        self.idx["train"] = permutaion[:train_len]
        self.idx["test"] = permutaion[train_len:]

        # create pretty version (i.e compatible with sklarn)
        X = defaultdict(list)
        Y = defaultdict(list)
        for mode in ["train", "test"]:
            for i in range(len(self.idx[mode])):
                X[mode].append(self.data[self.idx[mode][i]][0])
                Y[mode].append(self.data[self.idx[mode][i]][1])

        self.X_train = torch.vstack(X["train"]).numpy()[:TRAIN_NUM]
        self.Y_train = torch.tensor(Y["train"]).numpy()[:TRAIN_NUM]
        self.X_test = torch.vstack(X["test"]).numpy()[:TEST_NUM]
        self.Y_test = torch.tensor(Y["test"]).numpy()[:TEST_NUM]

        print(self.X_train.shape)
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

    @property
    def prob_no_pairs(self):
        """
        probability that in a batch size there are no pairs
        """
        numerator = list(
            accumulate(range(10, 10 - self.batch_size, -1), lambda x, y: x * y)
        )[-1]
        denominator = 10**self.batch_size
        return numerator / denominator


class MNISTNumDistEnv(MNISTEnv):
    def __init__(self, algo, T, batch_size, persistence, pca_dims=None):
        """
        Environment for doing MNIST using numerical distances
        """
        super().__init__(algo, T, batch_size, persistence, pca_dims)

    def next_actions(self):
        """
        Returns a list of next available actions which are represented
        as a vector of (img, img, prop_distance). Every `persistence` num steps
        a new set of actions is returned
        """

        # Get new set of actions
        if self.t % self.persistence == 0:
            cur_idx = (self.t // self.persistence) % int(len(self.data) * 0.8 - 1)

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
                    for prop_distance in self.granularity:
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
        reward = 1 - abs((real_distance - prop_distance) / len(self.granularity))
        if isinstance(self.algo, LinUCB):
            reward = 1 if real_distance == prop_distance else 0
        self.t += 1
        return reward

    def make_full_vector(self, vector, prop_distance):
        """
        Makes the vector full by adding the distance to the context
        """
        number_arms = len(self.granularity)  # 10 possible distances for neural UCB
        v_size = len(vector)
        full_v = torch.zeros(v_size * number_arms)

        # temporary fix to accomodate for [-1,+1] similarity measures
        ind_prop = prop_distance

        # If LinUCB using [-1,+1], change index of -1 to 0:
        if isinstance(self.algo, LinUCB):
            if prop_distance == -1:
                ind_prop = 0

        full_v[ind_prop * v_size : (ind_prop + 1) * v_size] = vector
        return full_v


class MNISTSimEnv(MNISTEnv):
    def __init__(
        self,
        algo,
        T,
        batch_size,
        persistence,
        context=None,
        pca_dims=None,
        eval_freq=1000,
        possible_actions=[-1, 1],
        to_eval=[],
    ):
        """
        Mnist environment

        Args:
            batch_size: size of the batch to use for training
            persistence: how many rounds to keep the same dataset for
        """
        super().__init__(
            algo,
            T,
            batch_size,
            persistence,
            pca_dims,
            context,
            eval_freq=eval_freq,
            to_eval=to_eval,
        )
        assert batch_size < 11, "batch size must be less than 10"
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

            # choose the next batch of images to use for training
            b_idxs = self.idx[self.mode][cur_idx : cur_idx + self.batch_size]
            batch = [self.data[i] for i in b_idxs]

            # store stuff to interpret returned action
            self.current_actions = {}
            self.real_label = {}

            # produce the actions
            for i in range(len(batch)):
                for j in range(i + 1, len(batch)):
                    for a in self.possible_actions:
                        imgx, labelx = batch[i]
                        imgy, labely = batch[j]
                        imgx, imgy = imgx.flatten(), imgy.flatten()
                        context_partial = self.make_context(imgx, imgy, a, self.context)
                        context_partial = context_partial.to(self.device)
                        self.current_actions[context_partial] = context_partial
                        self.real_label[context_partial] = 2 * int(labelx == labely) - 1
        return self.current_actions

    def make_context(self, imgx, imgy, a, context):
        if context == "linear":
            context_partial = torch.cat((imgx, imgy, torch.tensor([a])))

        elif context == "quadratic":
            context_partial = torch.tensor(
                [i * j for i, j in list(itertools.product(imgx, imgy))]
            )
            context_partial = torch.cat((context_partial, torch.tensor([a])))
        else:
            raise NotImplementedError

        return context_partial

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
        self.rewards.append(r.item())
        self.cum_regrets.append((self.prob_no_pairs - r.item()) + self.cum_regrets[-1])
