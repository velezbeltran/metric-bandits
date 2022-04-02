
import torch
from torch.utils.data import DataLoader

from metric_bandits.envs.base_env import BaseEnv
from metric_bandits.data.mnist import MNIST
from metric_bandits.constants.constants import SEED

class MNISTEnv(BaseEnv):
    def __init__(self, algo, T, batch_size, persistence): 
        """
        Initializes the environment

        Args:
            batch_size: size of the batch to use for training
            persistence: how many rounds to keep the same dataset for
        """
        # set seed
        torch.manual_seed(SEED)

        super().__init__(MNIST, algo, T)
        self.persistence = persistence
        self.batch_size = batch_size
        self.idx = {}
        self.init_data()

    def next_actions(self):
        """
        Returns a list of next available actions which are represented 
        as a vector of (img, img, prop_distance). Every `persistence`
        a new set of actions is returned
        """
        self.real_distances = {} # keeps track of real distances

        # Get new set of actions 
        if self.t % persistence == 0:
            cur_idx = self.t // persistence
            b_idxs = self.idx[self.mode][cur_idx:cur_idx+self.batch_size]
            batch = [self.data[i] for i in b_idxs]

            self.current_actions = []
            for imgx, labelx in batch:
                for imgy, labely in batch:
                    for prop_distance in range(10):
                        if not torch.eq(imgx,imgy).all():
                            self.current_actions.append((imgx, imgy, prop_distance))
                            self.real_distances[(imgx, imgy)] = abs(labelx - labely)
        return self.current_actions

    def step(self, action):
        """
        Returns the reward for the action taken
        """
        imgx, imgy, prop_distance = action
        real_distance = self.real_distances[(imgx, imgy)]
        reward = 1 - (real_distance - prop_distance)/10
        self.t += 1
        return reward
            
    def init_data(self):
        """
        Initializes the data loader
        """
        train_len = int(len(self.data) * 0.8)
        permutaion = torch.randperm(len(self.data))

        self.idx = {}
        self.idx['train'] = permutaion[:train_len]
        self.idx['test'] = permutaion[train_len:]


algo = None
T = 100
batch_size = 2
persistence = 10

env = MNISTEnv(algo, T, batch_size, persistence)
print([action[-1] for action in env.next_actions()])

