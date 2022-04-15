"""
Implements a siamese neural network that computes a similarity metric
between two objects.
"""
import torch
import torch.nn.functional as F

from metric_bandits.nets.base import BaseNN
from metric_bandits.utils.nn import make_batch


class SiameseNet(BaseNN):
    """
    Implements a siamese neural network that computes a similarity metric
    between two objects. Structure is ver much like the base network

    In particular if we have two context vectors `x` and `y` and a
    similarity value `s` (which is either 0 or 1) then we should have

    >>> val = torch.concat([x, y, s], dim=1)
    >>> r = model(val)

    where `r` is going to be  f(x).T f(y)* s if f$is the neural network.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        depth,
        dropout,
        normalize=False,
        batch_norm=False,
    ):
        super(SiameseNet, self).__init__(
            input_dim, hidden_dim, out_dim, depth, dropout, batch_norm
        )
        self.normalize = normalize

    def forward(self, x):
        """
        We assume that the input is a single vector of size 2 * context_dim + 1
        where the last element is the similarity value (0 or 1) and the rest
        contains the two context vectors `x` and `y` one after the other.

        x: torch.Tensor
            Shape (batch_size, 2 * context_dim + 1)
        """
        x = x.to(self.device)
        x = make_batch(x)
        v1 = x[:, : self.context_dim]
        v2 = x[:, self.context_dim : 2 * self.context_dim]
        s = x[:, [-1]]
        pred_sim, etc = self.predict_similarity(v1, v2)
        return pred_sim * s, etc

    def embed(self, x):
        """
        Embeds the context vector `x` into the network.

        x: torch.Tensor
            Shape (batch_size, context_dim)
        """
        x = x.to(self.device)
        x = make_batch(x)
        for i in range(self.depth):
            x = self.layers[i](x)
            if self.batch_norm:
                x = self.bn_layers[i](x)

            x = self.activation(x)
            x = F.dropout(x, p=self.dropout)
        x = self.layers[-1](x)

        if self.normalize:
            x = x / torch.norm(x, dim=1, keepdim=True)

        return x

    def predict_similarity(self, x, y):
        """
        Computes the similarity between `x` and `y`

        x: torch.Tensor
            Shape (batch_size, context_dim)
        y: torch.Tensor
            Shape (batch_size, context_dim)
        """
        x = make_batch(x)
        y = make_batch(y)

        v1 = self.embed(x)
        v2 = self.embed(y)

        sim = torch.sum(v1 * v2, axis=1, keepdim=True)
        return sim, (v1, v2)
