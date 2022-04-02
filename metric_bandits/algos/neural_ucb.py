"""
The implementation of neural UCB follows very closely what is described in
the paper. Name of variables are chosen so as to agree with the paper.
`https://arxiv.org/pdf/1911.04462.pdf`
s T, regularization parameter λ, exploration parameter ν, confidence parameter δ, norm
parameter S, step size η, number of gradient descent steps J, network width m, network depth L.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNN(nn.Module):
    """
    Implements an extremely simple neural network with initialization as explained in
    the paper.
    """

    def __init__(self, context_dim, hidden_dim, depth, dropout):
        assert depth > 0, "Depth must be greater than 0"

        super(BaseNN, self).__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.activation = F.relu

        layers = [nn.Linear(context_dim, hidden_dim)]
        for i in range(depth - 1):
            layers.append(nn.Linear(context_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.ModuleList(layers)
        self.init_weights()

    def forward(self, x):
        """
        Forward pass of the neural network
        """
        for i in range(self.depth):
            x = self.layers[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout)
        x = self.layers[-1](x)
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


class NeuralUCB:
    def __init__(
        self,
        context_dim,
        hidden_dim,
        depth,
        dropout,
        regularization,
        step_size,
        num_steps,
    ):
        self.regularization = regularization
        self.step_size = step_size
        self.num_steps = num_steps
        self.depth = depth
        self.model = BaseNN(context_dim, hidden_dim, depth, dropout)

    def choose_action(self, actions):
        """
        actions is a dictionary and contains the available actions
        """
        ucb_values = {}
        ucb_grads = {}
        for action in actions:
            val, grad = self.get_val_grad(actions[action])

    def get_val_grad(self, x):
        """
        Returns the predicted value and the gradient of the neural network
        """
        self.model.zero_grad()
        val = self.model(x)
        grad = torch.autograd.grad(val, self.model.parameters())
        print(val, grad)

    def reset(self):
        """
        Resets the model
        """
        self._Z = torch.eye(self.model.num_params)

    @property
    def Z(self):
        """
        Returns the norm parameter
        """
        return None
