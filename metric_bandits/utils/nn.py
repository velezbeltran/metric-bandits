"""
Utilities related to neural networks.
"""


def make_batch(x):
    """
    Makes x into a batch if it doesn't already have batch dimension
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return x
