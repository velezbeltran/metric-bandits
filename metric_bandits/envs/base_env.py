"""
Contains the code for creating an environment for an abstract environment
for exploration
"""
from tqdm import tqdm


class BaseEnv:
    """
    Class for creating an environment for exploration
    """

    def __init__(self, data, algo, T):
        """
        Initializes the environment
        """
        self.data = data  # dataset
        self.algo = algo  # algo to use for exploration
        self.T = T  # Total number of rounds
        self.t = 0  # current round
        self.mode = "train"  # mode of the environment (train/test)

        self.regret = [0]  # keeps track of the regret per round

    def update(self, r):
        """
        Updates the environment
        """
        self.regret.append(r + self.regret[-1])

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
