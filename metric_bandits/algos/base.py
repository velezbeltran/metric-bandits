"""
Contains the base class for the algorithms
"""


class BaseAlgo:
    def __init__(self):
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
