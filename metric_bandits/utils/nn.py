"""
Utilities related to neural networks.
"""
import torch


def make_batch(x):
    """
    Makes x into a batch if it doesn't already have batch dimension
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return x


def to_tensor(x):
    """
    Converts x into a torch.Tensor if it isn't already
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x


def make_metric(model):
    """
    Takes a model and using its embedding function returns a metric.
    Assumes that x, y are numpy arrays.
    """

    def metric(x, y, **kwargs):
        x = make_batch(to_tensor(x)).float()
        y = make_batch(to_tensor(y)).float()

        sim, _ = model.predict_similarity(x, y)
        return -sim.item()

    return metric
