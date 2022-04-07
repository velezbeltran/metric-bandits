"""
The implementation of neural UCB follows very closely what is described in
the paper. Name of variables are chosen so as to agree with the paper.
`https://arxiv.org/pdf/1911.04462.pdf`
s T, regularization parameter λ, exploration parameter ν, confidence parameter δ, norm
parameter S, step size η, number of gradient descent steps J, network width m, network depth L.
"""
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from metric_bandits.algos.base import BaseAlgo
from metric_bandits.utils.math import sherman_morrison


class NeuralUCB(BaseAlgo):
    def __init__(
        self,
        context_dim,
        hidden_dim,
        depth,
        dropout,
        regularization,
        step_size,
        num_steps,
        train_freq=100,
        exploration_param=0.1,
    ):
        self.regularization = regularization
        self.step_size = step_size
        self.num_steps = num_steps
        self.depth = depth
        self.exploration_param = exploration_param
        self.train_freq = train_freq

        # Set up model and optimizer
        self.model = BaseNN(context_dim, hidden_dim, depth, dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=step_size)

        # state of the algorithm
        self.Z_inv = None
        self.t = 0

        # parameters to keep track of
        self.last_action = None
        self.rewards = []
        self.contexts_played = []

    def choose_action(self, actions):
        """
        actions is a list-like object dictionary and contains the available actions
        """
        self.ucb_val_grads, self.ucb_estimate = {}, {}
        for action in actions:
            val, grad = self.get_val_grad(actions[action])
            self.ucb_val_grads[action] = (val, grad)
            self.ucb_estimate[action] = val + self.optimist_reward(grad)

        # return the key with the highest value
        self.last_action = max(self.ucb_estimate, key=self.ucb_estimate.get)
        self.contexts_played.append(actions[self.last_action])
        return self.last_action

    def update(self, reward):
        """
        Updates the model
        """
        self.rewards.append(reward)

        # update our confidence matrix
        prev_grad = self.ucb_val_grads[self.last_action][1] / sqrt(
            self.model.num_params
        )
        self.Z_inv = sherman_morrison(self.Z_inv, prev_grad)

        # decide whether to train the model
        self.t += 1
        if self.t % self.train_freq == 0:
            self.train()

    def get_val_grad(self, x):
        """
        Returns the predicted value and the gradient of the neural network
        as a one dimensional column vector.
        """
        self.model.zero_grad()
        val = self.model(x)
        grad = torch.autograd.grad(val, self.model.parameters())
        g = torch.cat([g.flatten() for g in grad])
        return val, g.unsqueeze(-1)

    def optimist_reward(self, grad):
        with torch.no_grad():
            val = self.exploration_param * torch.sqrt(
                grad.T @ self.Z_inv @ grad / self.model.num_params
            )
            return val

    def train(self):
        """
        Trains the model
        """
        inputs = torch.stack(self.contexts_played)
        tgts = torch.tensor(self.rewards).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(inputs, tgts)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.train_freq, shuffle=True
        )

        for epoch in (pbar := tqdm(range(self.num_steps))):
            for x, y in loader:
                assert x.shape[0] == self.train_freq
                assert y.shape[0] == self.train_freq
                self.optimizer.zero_grad()
                val = self.model(x)
                loss = F.mse_loss(val, y, reduction="mean")
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f"Loss: {loss.item():.4f}")

    def reset(self):
        """
        Resets the model
        """
        self.Z_inv = torch.eye(self.model.num_params)
        print("Reset model")


class BaseNN(nn.Module):
    """
    Implements an extremely simple neural network with initialization as explained in
    the paper.

    Parameters
    ----------
    context_dim : int
        Dimension of the context vector
    hidden_dim : int
        Dimension of the hidden layer
    depth : int
        Depth of the network does not include the last layer.
        The last layer is always a linear layer.
    Dropout : float
        Dropout probability.
    """

    def __init__(self, context_dim, hidden_dim, depth, dropout):
        assert depth > 0, "Depth must be greater than 0"

        super(BaseNN, self).__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.activation = F.relu

        layers = [
            nn.Linear(
                context_dim,
                hidden_dim,
            )
        ]
        for i in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.ModuleList(layers)
        self.init_weights()

    def forward(self, x):
        """
        Forward pass of the neural network
        """
        x = self.make_batch(x)
        for i in range(self.depth):
            x = self.layers[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout)
        x = self.layers[-1](x)
        return x

    def make_batch(self, x):
        """
        Makes x into a batch if it doesn't already have batch dimension
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return x

    def init_weights(self):
        """
        Initializes the weights of the neural network as done in the
        paper
        """
        # TODO: Implement this
        return None

    @property
    def num_params(self):
        """
        Returns the number of parameters in the model
        """
        if not hasattr(self, "_num_params"):
            self._num_params = sum(p.numel() for p in self.parameters())
        return self._num_params
