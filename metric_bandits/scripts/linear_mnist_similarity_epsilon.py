from metric_bandits.algos.linear import Linear
from metric_bandits.envs.mnist_env import MNISTSimEnv

# Constants for the environment
T = 10000
batch_size = 2
persistence = 1
context = "quadratic"

# set up the enviromenent and model
algo = Linear(explore_param=0.1)
env = MNISTSimEnv(algo, T, batch_size, persistence, pca_dims=20, context=context)
env.reset()

print("Running contextual LinUCB on MNIST dataset")

# Run the algorithm
env.train()
print("trained")