"""
This script does a very simple test just to see whether neural UCB works on the similarity
based environment.
"""

from metric_bandits.algos.neural_ucb import NeuralUCB
from metric_bandits.envs.mnist_env import MNISTSimEnv
from metric_bandits.nets.siamese import SiameseNet

# Constants for the neural network
pca_dims = 20
input_dim = pca_dims
depth = 4
hidden_dim = 20
out_dim = 5
dropout = 0.4

# Constants for the environment
T = 1000
batch_size = 20
persistence = 5

# Constnats for UCB
reg = 0.1
step_size = 0.3
num_steps = 20
train_freq = 50
explore_param = 1


# set up the enviromenent and model
model = SiameseNet(input_dim, hidden_dim, depth, out_dim, dropout)
algo = NeuralUCB(model, reg, step_size, num_steps, train_freq, explore_param)
env = MNISTSimEnv(algo, T, batch_size, persistence, pca_dims=20)
env.reset()
env.train()
