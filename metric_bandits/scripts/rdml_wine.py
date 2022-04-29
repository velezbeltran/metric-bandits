from metric_bandits.algos.rdml import RDML
from metric_bandits.envs.wine_env import WineEnv

# Constants for the environment
T = 10000
batch_size = 2


# set up the enviromenent and model
algo = RDML(learning_rate=1)

print("Running Regularised Distance Metric Learning...")

env = WineEnv(algo, T, batch_size)
env.reset()

# Run the algorithm
env.train()
print("trained")
