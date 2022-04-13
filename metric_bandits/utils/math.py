"""
General utilities related to math operations.
"""
import numpy as np

def sherman_morrison(Z_inv, g):
    """
    Uses a sherman morrison update formula
    to update the inverse of a matrix A. I.e computes efficiently
    (A + vv^\top)^{-1}
    """
    qf = g.T @ Z_inv @ g
    return Z_inv - (Z_inv @ g @ g.T @ Z_inv) / (1 + qf)

def square_matrix_norm(A, x):
    """
    Computes the projection of vector x in the matrix norm of A
    """
    return x.T @ A @ x

def _inv(A):
    return np.linalg.pinv(A)
