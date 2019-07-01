from __future__ import division
import numpy as np
from scipy import linalg


def T_mtx_sep(b, vec_len, num_vec, size_c):
    """
    The (1D) convolution matrix associated with b * c,
    where each COLUMN of b is annihilated by c.
    :param b: the uniformly sampled sinusoids. Here we assume b is in a 2D shape already.
    :param vec_len: length of each vector that should satisfy the annihilation constraint
    :param num_vec: total number of such vectors that should be annihilated by the same filter,
            whose coefficients are given by 'c'
    :param size_c: size of the 1D annihilating filter (usually: num_dirc + 1)
    :return:
    """
    b = np.reshape(b, (vec_len, num_vec), order='F')
    anni_out_sz0 = vec_len - size_c + 1
    Tmtx = np.zeros((num_vec * anni_out_sz0, size_c), dtype=b.dtype)
    for loop in range(num_vec):
        Tmtx[loop * anni_out_sz0:(loop + 1) * anni_out_sz0, :] = \
            linalg.toeplitz(b[size_c - 1::, loop], b[size_c - 1::-1, loop])

    return Tmtx


def R_mtx_sep(coef, vec_len, num_vec):
    """
    the convolution matrix associated with the annihilating filter coefficients c.
    :param coef: 1D annihilating filter coefficients
    :param vec_len: length of each set of uniformly sampled sinusoids
    :param num_vec: number of vectors that can be annihilated.
    :return:
    """
    size_c = coef.size
    anni_out_sz0 = vec_len - size_c + 1
    col = np.zeros(anni_out_sz0, dtype=coef.dtype)
    col[0] = coef[-1]
    row = np.zeros(vec_len, dtype=coef.dtype)
    row[:size_c] = coef[::-1]
    return linalg.block_diag(*([linalg.toeplitz(col, row)] * num_vec))
    # return np.kron(np.eye(num_vec), linalg.toeplitz(col, row))
