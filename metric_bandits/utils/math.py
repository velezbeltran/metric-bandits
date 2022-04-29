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


def square_matrix_norm(A, x):
    """
    Computes the projection of vector x in the matrix norm of A
    """
    return x.T @ A @ x


def get_argmax(values, f):
    """
    Returns the index of the maximum value in a list. according to f
    """
    values = [f(v) for v in values]
    return values.index(max(values))


def cross_terms(X_left, X_right):
    """
    Returns the cross terms of A and B
    where A and B are tensors of shape n x m
    """
    cross_terms = X_left[:, :, None] @ X_right[:, None, :]
    # make into size n x (m x m)
    cross_terms = cross_terms.reshape(cross_terms.shape[0], -1)
    return cross_terms
