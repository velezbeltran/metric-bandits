from metric_bandits.algos.neural_ucb import NeuralUCB
from metric_bandits.envs.mnist_env import MNISTEnv

# Constants for the neural network
context_dim = 2 * 10 * 10  # (28x28 image + 1 for bias
depth = 1
hidden_dim = 20
dropout = 0
regularization = 0.1
step_size = 0.1
num_steps = 100

# Constants for the environment
T = 1000
batch_size = 2
persistence = 10

# set up the enviromenent and model
algo = NeuralUCB(
    context_dim=context_dim,
    hidden_dim=hidden_dim,
    depth=depth,
    dropout=dropout,
    regularization=regularization,
    step_size=step_size,
    num_steps=num_steps,
)
print(algo.model.num_params)
env = MNISTEnv(algo, T, batch_size, persistence, pca=True)
env.reset()
# Run the algorithm
env.train()
print("trained")
# evaluate the performace
