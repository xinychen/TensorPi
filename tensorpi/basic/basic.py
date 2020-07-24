"""
Tensor Computation Foundations - basic.py
"""
# Created by Xinyu Chen, July 2020
__all__ = ['ten2mat']

def ten2mat(X, mode):
    """
    Return tensor unfolding (i.e., matrix) along certain mode.
    Parameters
    ----------
    X : ndarray
        N-dimensional array/tensor of size (n1,n2,...,nd) to tensor unfolding.
    mode : int
        Integer in {1,2,...,d}.
        By definition, if ``mode = 1``, the returned matrix is the mode-1 
        tensor unfolding of X, the returned matrix is the mode-2 tensor 
        unfolding of X if ``mode = 2``, and the returned matrix is the mode-N
        tensor unfolding of X if ``mode = d``.
    Returns
    -------
    Y : 2-dimensional array
        of size::
            (n1,n2n3...nd) if ``mode = 1``;
            (n2,n1n3...nd) if ``mode = 2``;
            ...    ...
    See Also
    --------
    mat2ten : Fold matrix to tensor along certain mode.
    Examples
    --------
    >>> from tensorpi.basics import ten2mat
    >>> X = np.array([[[1, 2, 3, 4], [3, 4, 5, 6]], 
    >>>              [[5, 6, 7, 8], [7, 8, 9, 10]], 
    >>>              [[9, 10, 11, 12], [11, 12, 13, 14]]])
    >>> print('tensor size:')
    >>> print(X.shape)
    >>> print(X[:, :, 0])
    >>> print(X[:, :, 1])
    >>> print(X[:, :, 2])
    >>> print(X[:, :, 3])
    >>> print('mode-1 tensor unfolding:')
    >>> print(ten2mat(X, 0))
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')