"""
Contains the code for creating an environment for an abstract environment
for exploration
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import torch as torch
from tqdm import tqdm

from metric_bandits.utils.eval import eval_knn, eval_linear

# make pretty plots ggplot
plt.style.use("ggplot")
plt.gca().set_aspect("equal", adjustable="box")


class BaseEnv:
    """
    Class for creating an environment for exploration
    """

    def __init__(self, data, algo, T, eval_freq=1000, to_eval=["knn, embedding"]):
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
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None
        self.nice_data_available = False

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
        print("Evaluating...")

        # if the algorithm has a metric use it to test KNN
        if not self.nice_data_available:
            return None

        if hasattr(self.algo, "metric") and "knn" in self.to_eval:
            metric = self.algo.metric
            acc = eval_knn(self.X_train, self.Y_train, self.X_test, self.Y_test, metric)
            self.eval_metrics["knn_acc"].append(acc)

        if hasattr(self.algo, "embed") and "linear" in self.to_eval:
            embed = self.algo.embed
            X_train = embed(torch.tensor(self.X_train).to(self.device).float())
            X_train = X_train.detach().cpu().numpy()
            X_test = embed(torch.tensor(self.X_test).to(self.device).float())
            X_test = X_test.detach().cpu().numpy()

            acc = eval_linear(X_train, self.Y_train, X_test, self.Y_test)

            # plot the embedding
            self.eval_metrics["linear_acc"].append(acc)

        # if the algorithm has an embedding associated with it
        # use it to visualize the embedding
        if hasattr(self.algo, "embed") and "embedding" in self.to_eval:
            embed = self.algo.embed
            X_tensor = torch.tensor(self.X_test).to(self.device, dtype=torch.float)
            X_embed = embed(X_tensor).detach().cpu().numpy()
            self.eval_metrics["embedding"].append((X_embed, self.Y_test))

        for k, v in self.eval_metrics.items():
            if k != "embedding":
                print(f"{k}: {v[-1]:.4f}")

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
