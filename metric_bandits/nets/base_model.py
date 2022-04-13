import torch.nn as nn
import torch.nn.functional as F


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
