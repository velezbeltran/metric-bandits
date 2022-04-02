"""
General utilities related to math operations.
"""


def sherman_morrison(Z_inv, g):
    """
    Uses a sherman morrison update formula
    to update the inverse of a matrix A. I.e computes efficiently
    (A + vv^\top)^{-1}
    """
    qf = g.T @ Z_inv @ g
    return Z_inv - (Z_inv @ g @ g.T @ Z_inv) / (1 + qf)
