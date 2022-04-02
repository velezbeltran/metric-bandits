"""
Contains the code for creating an environment for an abstract environment 
for exploration 
"""


class BaseEnv:
    """
    Class for creating an environment for exploration
    """
    
    def __init__(self, data, algo, T):
        """
        Initializes the environment
        """
        self.data = data # dataset
        self.algo = algo # algo to use for exploration
        self.T = T # Total number of rounds
        self.t = 0 # current round
        self.mode = 'train' # mode of the environment (train/test)

    def set_mode(self, mode):
        """
        Sets the mode of the environment
        """
        self.mode = mode

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
        raise NotImplementedError



