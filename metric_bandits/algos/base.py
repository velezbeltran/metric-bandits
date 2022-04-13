"""
Contains the base class for the algorithms
"""
import os
import pickle as pkl
from datetime import datetime

from metric_bandits.constants.paths import MODELS


class BaseAlgo:
    def __init__(self, run_id=None):
        self.run_id = run_id
        self.name = "BaseAlgo"
        return None

    def choose_action(self, actions):
        """
        Returns the action to be taken
        """
        raise NotImplementedError

    def update(self, r):
        """
        Updates the algorithm
        """
        raise NotImplementedError

    def save(self):
        """
        Saves the algorithm to the folder in models folder with the same name
        as the algorithm class. The file name is the run_id and if not set the
        current time.
        """
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d%H%M")

        dir_path = os.path.join(MODELS, type(self).__name__)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        path = os.path.join(dir_path, self.run_id + ".pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def load(self, run_id):
        """
        Loads the algorithm with the given run_id from the folder in models
        folder with the same name as the algorithm class.
        """
        dir_path = os.path.join(MODELS, self.__name__)
        path = os.path.join(dir_path, run_id + ".pkl")
        with open(path, "rb") as f:
            return pkl.load(f)
