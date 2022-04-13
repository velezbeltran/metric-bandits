"""
Implements a siamese neural network that computes a similarity metric
between two objects.
"""
import torch
import torch.nn.functional as F

from metric_bandits.nets.base import BaseNN


class SiameseNN(BaseNN):
    """
    Implements a siamese neural network that computes a similarity metric
    between two objects. Structure is ver much like the base network

    In particular if we have two context vectors `x` and `y` and a
    similarity value `s` (which is either 0 or 1) then we should have

    >>> val = torch.concat([x, y, s], dim=1)
    >>> r = model(val)

    where `r` is going to be  f(x).T f(y)* s if f$is the neural network.
    """

    def __init__(self, context_dim, hidden_dim, depth, dropout):
        super(SiameseNN, self).__init__(context_dim, hidden_dim, depth, dropout)

    def forward(self, x):
        """
        We assume that the input is a single vector of size 2 * context_dim + 1
        where the last element is the similarity value (0 or 1) and the rest
        contains the two context vectors `x` and `y` one after the other.

        x: torch.Tensor
            Shape (batch_size, 2 * context_dim + 1)
        """
        v1 = x[:, : self.context_dim]
        v2 = x[:, self.context_dim : 2 * self.context_dim]
        s = x[:, [-1]]

        x = torch.cat([v1, v1], dim=0)
        for i in range(self.depth):
            x = self.layers[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout)
        x = self.layers[-1](x)

        v1 = x[: self.context_dim]
        v2 = x[self.context_dim :]

        sim = torch.sum(v1 * v2, axis=1, keepdim=True) * s
        return sim
