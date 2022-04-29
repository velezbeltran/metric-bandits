"""
Contains the code for creating an environment for an abstract environment
for exploration
"""
import random
from collections import defaultdict

import numpy as np
import torch as torch
from tqdm import tqdm

from metric_bandits.utils.eval import eval_knn, eval_linear
from metric_bandits.utils.math import cross_terms


class BaseEnv:
    """
    Class for creating an environment for exploration

    A subclass should initialize self.X_train and self.Y_train
    as a numpy array format.

    Parameters:
    -----------
    to_eval: list
        list of strings with the names of criteria to evaluate e.g
        ['knn','linear', 'embedding']
    """

    def __init__(self, data, algo, T, eval_freq=1000, to_eval=[]):
        """
        Initializes the environment
        """
        self.data = data  # dataset
        self.algo = algo  # algo to use for exploration
        self.T = T  # Total number of rounds
        self.t = 0  # current round
        self.mode = "train"  # mode of the environment (train/test)
        self.eval_freq = eval_freq  # How of to call self.eval
        self.to_eval = to_eval  # What to evaluate

        # Nice way of of storing the data
        # should be initialized for evaluation as standard
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None
        self.nice_data_available = False
        self.idx = {}

        self.cum_regrets = [0]  # keeps track of the regret per round
        self.rewards = []  # keeps track of the rewards per round
        self.eval_metrics = defaultdict(
            list
        )  # keeps track of the evaluation metrics per round

    def update(self, r):
        """
        Updates the environment
        """
        raise NotImplementedError

    def next_actions(self):
        """
        Returns the next available set of available actions
        Should be returned as a list.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Returns the reward for the action taken
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the environment
        """
        raise NotImplementedError

    def train(self):
        """
        Trains the algorithm
        """
        self.mode = "train"
        pbar = tqdm(total=self.T)
        for _ in range(self.T):
            actions = self.next_actions()
            action = self.algo.choose_action(actions)
            r = self.step(action)
            self.algo.update(r)
            self.update(r)

            # print the regret nicely
            pbar.set_description(f"Regret/time: {self.cum_regrets[-1]/self.t:.2f}")
            pbar.update(1)

            if (self.t + 1) % self.eval_freq == 0:
                self.eval()

    def eval(self):
        """
        Evaluates the algorithm by looking at the quality of the embeddings
        and of a KNN predictor. The results are stored in `self.eval_metrics`.
        """

        # if the algorithm has a metric use it to test KNN
        if not self.nice_data_available:
            raise Exception("No nice data available so can't do evals")

        for eval_func_name in self.to_eval:
            getattr(self, "eval_" + eval_func_name)()

        for k, v in self.eval_metrics.items():
            if k != "embedding":
                print(f"{k}: {v[-1]:.4f}")

    def eval_knn(self):
        if not hasattr(self.algo, "metric"):
            raise Exception("Algorithm does not have a metric so can't evaluate knn")

        metric = self.algo.metric
        acc = eval_knn(self.X_train, self.Y_train, self.X_test, self.Y_test, metric)
        self.eval_metrics["knn_acc"].append(acc)

    def eval_linear(self):
        if not hasattr(self.algo, "embed"):
            raise Exception(
                "Algorithm does not have embedding so can't evaluate linear predictor"
            )
        embed = self.algo.embed
        X_train = embed(torch.tensor(self.X_train).to(self.device).float())
        X_train = X_train.detach().cpu().numpy()
        X_test = embed(torch.tensor(self.X_test).to(self.device).float())
        X_test = X_test.detach().cpu().numpy()
        acc = eval_linear(X_train, self.Y_train, X_test, self.Y_test)
        # save the embedding
        self.eval_metrics["linear"].append(acc)

    def eval_embedding(self):
        if not hasattr(self.algo, "embed"):
            raise Exception(
                "Algorithm does not have embedding so can't evaluate embedding"
            )
        embed = self.algo.embed
        X_tensor = torch.tensor(self.X_test).to(self.device, dtype=torch.float)
        X_embed = embed(X_tensor).detach().cpu().numpy()
        self.eval_metrics["embedding"].append((X_embed, self.Y_test))

    def eval_l2_loss_embed(self):
        if not hasattr(self.algo, "embed"):
            raise Exception(
                "Algorithm does not have embedding so can't evaluate l2 loss"
            )

        # get the true similarity
        if not hasattr(self, "_similarity_labels"):
            self._similarity_labels = np.zeros((len(self.Y_test), len(self.Y_test)))
            for i in range(len(self.Y_test)):
                for j in range(len(self.Y_test)):
                    self._similarity_labels[i, j] = (
                        int(self.Y_test[i] == self.Y_test[j]) * 2 - 1
                    )

        embed = self.algo.embed
        X_tensor = torch.tensor(self.X_test).to(self.device, dtype=torch.float)
        X_embed = embed(X_tensor).detach().cpu().numpy()

        similarity = np.dot(X_embed, X_embed.T)
        loss = np.square(similarity - self._similarity_labels)
        loss = loss[np.triu_indices(len(self.Y_test), 1)]
        loss = np.sum(loss) / ((len(self.Y_test) ** 2 - len(self.Y_test)) / 2)
        self.eval_metrics["l2_loss_embed"].append(loss)

    def eval_l2_loss_linear(self):
        """
        Evaluates the l2 loss of the linear similarity metric
        """
        # random permutation with torch
        perm = torch.randperm(len(self.Y_test))
        X_test = torch.tensor(self.X_test).to(self.device, dtype=torch.float)

        X_left = X_test
        X_right = X_test[perm]
        Y_left = torch.tensor(self.Y_test).to(self.device)
        Y_right = torch.tensor(self.Y_test).to(self.device)[perm]
        Y_true = 2 * (Y_left == Y_right) - 1

        # get features
        feats = cross_terms(X_left, X_right)
        pred = feats @ self.algo.theta
        loss = torch.mean((pred - Y_true) ** 2)
        self.eval_metrics["l2_loss_linear"].append(loss.item())

    def eval_square_loss(self):
        if not hasattr(self.algo, "estimate"):
            raise Exception(
                "Algorithm does not have a estimator function so can't evaluate square loss"
            )

        if not hasattr(self, "make_context") or not hasattr(self, "context"):
            raise Exception(
                "Environment does not have a context function or variable so can't evaluate square loss"
            )

        estimate = self.algo.estimate

        # choose the next batch of images to use for training
        b_idxs = self.idx["test"]
        X, y = self.data
        test_set = list(zip(X[b_idxs], y[b_idxs]))
        random.shuffle(test_set)
        bX, by = zip(*test_set)
        # batch = [self.data[i] for i in b_idxs]

        # store stuff to interpret returned action

        self.real_label = []
        self.pred_label = []

        # produce the actions
        for i in range(100):
            for j in range(i + 1, 100):
                self.test_actions = {}
                for a in self.possible_actions:
                    imgx, labelx = bX[i], by[i]
                    imgy, labely = bX[j], by[j]
                    imgx, imgy = imgx.flatten(), imgy.flatten()
                    context_partial = self.make_context(imgx, imgy, a, self.context)

                    context_partial = context_partial.float()
                    self.test_actions[context_partial] = context_partial
                self.real_label.append(2 * int(labelx == labely) - 1)
                self.pred_label.append(self.algo.estimate(self.test_actions))
        # print("Produced Estimates")

        # evaluate empirical estimate:
        matches = [
            1 if self.pred_label[i] == self.real_label[i] else 0
            for i in range(len(self.pred_label))
        ]
        acc = sum(matches) / len(matches)
        # print(acc)
        self.eval_metrics["square_loss"].append(acc)

    @property
    def mode(self):
        """
        Returns the mode of the environment
        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        """
        Sets the mode of the environment
        """
        self._mode = mode
        self.algo.mode = mode
