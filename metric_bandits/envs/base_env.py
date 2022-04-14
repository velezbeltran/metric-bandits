"""
Contains the code for creating an environment for an abstract environment
for exploration
"""
from tqdm import tqdm

from metric_bandits.utils.eval import eval_knn


class BaseEnv:
    """
    Class for creating an environment for exploration
    """

    def __init__(self, data, algo, T, eval_freq=1000):
        """
        Initializes the environment
        """
        self.data = data  # dataset
        self.algo = algo  # algo to use for exploration
        self.T = T  # Total number of rounds
        self.t = 0  # current round
        self.mode = "train"  # mode of the environment (train/test)
        self.eval_freq = eval_freq  # How of to call self.eval

        # Nice way of of storing the data
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None
        self.nice_data_available = False

        self.cum_regrets = [0]  # keeps track of the regret per round
        self.rewards = []  # keeps track of the rewards per round
        self.eval_metrics = []  # keeps track of the evaluation metrics per round

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
        for _ in (pbar := tqdm(range(self.T))):
            actions = self.next_actions()
            action = self.algo.choose_action(actions)
            r = self.step(action)
            self.algo.update(r)
            self.update(r)
            # print the regret nicely
            pbar.set_description(f"Regret/time: {self.cum_regrets[-1]/self.t:.2f}")
            if (self.t + 1) % self.eval_freq == 0:
                self.eval()

    def eval(self):
        eval_metric = {}
        print("Evaluating...")
        if hasattr(self.algo, "metric") and self.nice_data_available:
            metric = self.algo.metric
            acc = eval_knn(self.X_train, self.Y_train, self.X_test, self.Y_test, metric)
            eval_metric["knn acc"] = acc

        for k, v in eval_metric.items():
            print(f"{k}: {v}")
        self.eval_metrics.append(eval_metric)

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
