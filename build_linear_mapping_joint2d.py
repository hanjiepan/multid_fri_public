from __future__ import division

import numpy as np
from scipy import linalg

from utils_2d import periodic_sinc


def convmtx2_valid(H, M, N):
    """
    2d convolution matrix with the boundary condition 'valid', i.e., only filter
    within the given data block.
    :param H: 2d filter
    :param M: input signal dimension is M x N
    :param N: input signal dimension is M x N
    :return:
    """
    T = convmtx2(H, M, N)
    s_H0, s_H1 = H.shape
    input_large_flag = np.all(np.array([M, N]) >= np.array(H.shape))
    H_large_flag = np.all(np.array([M, N]) <= np.array(H.shape))
    assert input_large_flag or H_large_flag
    if input_large_flag:
        S = np.pad(np.ones((M - s_H0 + 1, N - s_H1 + 1), dtype=bool),
                   ((s_H0 - 1, s_H0 - 1), (s_H1 - 1, s_H1 - 1)),
                   'constant', constant_values=False)
    else:
        S = np.pad(np.ones((s_H0 - M + 1, s_H1 - N + 1), dtype=bool),
                   ((M - 1, M - 1), (N - 1, N - 1)),
                   'constant', constant_values=False)
    T = T[S.flatten('F'), :]
    return T


def convmtx2(H, M, N):
    """
    build 2d convolution matrix
    :param H: 2d filter
    :param M: input signal dimension is M x N
    :param N: input signal dimension is M x N
    :return:
    """
    P, Q = H.shape
    blockHeight = int(M + P - 1)
    blockWidth = int(M)
    blockNonZeros = int(P * M)
    totalNonZeros = int(Q * N * blockNonZeros)

    THeight = int((N + Q - 1) * blockHeight)
    TWidth = int(N * blockWidth)

    Tvals = np.empty((totalNonZeros, 1), dtype=H.dtype)
    Trows = np.empty((totalNonZeros, 1), dtype=int)
    Tcols = np.empty((totalNonZeros, 1), dtype=int)

    c = np.dot(np.diag(np.arange(1, M + 1)), np.ones((M, P), dtype=float))
    r = np.repeat(np.reshape(c + np.arange(0, P)[np.newaxis], (-1, 1), order='F'), N, axis=1)
    c = np.repeat(c.flatten('F')[:, np.newaxis], N, axis=1)

    colOffsets = np.arange(N) * M
    colOffsets = np.reshape(np.repeat(colOffsets[np.newaxis], M * P, axis=0) + c, (-1, 1), order='F') - 1

    rowOffsets = np.arange(N) * blockHeight
    rowOffsets = np.reshape(np.repeat(rowOffsets[np.newaxis], M * P, axis=0) + r, (-1, 1), order='F') - 1

    N_blockNonZeros = N * blockNonZeros
    for k in range(Q):
        val = np.reshape(np.tile((H[:, k]).flatten(), (M, 1)), (-1, 1), order='F')
        first = int(k * N_blockNonZeros)
        last = int(first + N_blockNonZeros)
        Trows[first:last] = rowOffsets
        Tcols[first:last] = colOffsets
        Tvals[first:last] = np.tile(val, (N, 1))
        rowOffsets += blockHeight

    T = np.zeros((THeight, TWidth), dtype=H.dtype)
    T[Trows, Tcols] = Tvals
    return T


def R_mtx_joint(c1, c2, shape):
    """
    build the right-dual matrix associated with 2D filters c1 and c2
    :param c1: the first filter that the uniformly sampled sinusoid should be annihilated
    :param c2: the second filter that the uniformly sampled sinusoid should be annihilated
    :param shape: a tuple of the shape of the uniformly sampled sinusoid (i.e., b)
    :return:
    """
    L0, L1 = shape
    R_loop_row = convmtx2_valid(c1, L0, L1)
    R_loop_col = convmtx2_valid(c2, L0, L1)
    return np.vstack((R_loop_row, R_loop_col))


def T_mtx_joint(b, shape_c1, shape_c2):
    """
    The convolution matrix associated with b * c1 and b * c2 (jointly)
    :param b: the uniformly sampled sinusoid.
            Here we assume b is in a 2D shape already (instead of a vector form)
    :param shape_c1: shape of the first 2D filter
    :param shape_c2: shape of the second 2D filter
    :return:
    """
    return linalg.block_diag(
        convmtx2_valid(b, shape_c1[0], shape_c1[1]),
        convmtx2_valid(b, shape_c2[0], shape_c2[1])
    )


def planar_sel_coef_subset_one_filter(shape_coef, num_non_zero,
                                      max_num_same_x=1, max_num_same_y=1):
    """
    Select subsets of the 2D filters with total number of entries at
    least num_non_zero that should be zero. In the end, the 2D filters
    only have num_non_zero entries of non-zero values.
    :param shape_coef: a tuple of size 2 for the shape of filter
    :param num_non_zero: number of non-zero entries in the 2D filters.
            Typically num_dirac + 1, where num_dirac is the number of Dirac deltas.
    :param max_num_same_x: maximum number of Dirac deltas that have the
            same horizontal locations. This will impose the minimum dimension
            of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the
            same vertical locations This will impose the minimum dimension
            of the annihilating filter used.
    :return:
    """
    # the selection indices that corresponds to the part where the
    # coefficients are DIFFERENT from zero
    num_coef = 1
    for dim_loop in shape_coef:
        num_coef *= dim_loop

    more = True
    mask = np.zeros(num_coef, dtype=int)
    while more:
        mask = np.zeros(num_coef, dtype=int)
        non_zero_ind = np.random.permutation(num_coef)[:num_non_zero]
        mask[non_zero_ind] = 1
        mask = np.reshape(mask, shape_coef, order='F')
        cord_0, cord_1 = np.nonzero(mask)

        more = (np.max(cord_1) - np.min(cord_1) + 1 < max_num_same_y + 1) or \
               (np.max(cord_0) - np.min(cord_0) + 1 < max_num_same_x + 1) or \
               np.any(np.all(1 - mask, axis=0)) or np.any(np.all(1 - mask, axis=1))

    subset_idx = (1 - mask).ravel(order='F').nonzero()[0]
    S = np.eye(num_coef)[subset_idx, :]
    return S


def planar_sel_coef_subset_complement(shape_coef, num_non_zero,
                                      max_num_same_x=1, max_num_same_y=1):
    """
    Select subsets of the 2D filters with total number of entries at
    least num_non_zero that are DIFFERENT from zero. In the end, the 2D filters
    only have num_non_zero entries of non-zero values.
    :param shape_coef: a tuple of size 2 for the shape of 2D annihilating filter
    :param num_non_zero: number of non-zero entries in the 2D filters.
            Typically num_dirac + 1, where num_dirac is the number of Dirac deltas.
    :param max_num_same_x: maximum number of Dirac deltas that have the
            same horizontal locations. This will impose the minimum dimension
            of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the
            same vertical locations This will impose the minimum dimension
            of the annihilating filter used.
    :return:
    """
    # the selection indices that corresponds to the part where the
    # coefficients are DIFFERENT from zero
    num_coef = 1
    for dim_loop in shape_coef:
        num_coef *= dim_loop

    more = True
    while more:
        mask = np.zeros(num_coef, dtype=int)
        non_zero_ind1 = np.random.permutation(num_coef)[:num_non_zero]
        mask[non_zero_ind1] = 1
        mask = np.reshape(mask, shape_coef, order='F')
        cord_0, cord_1 = np.nonzero(mask)

        more = (np.max(cord_1) - np.min(cord_1) + 1 < max_num_same_y + 1) or \
               (np.max(cord_0) - np.min(cord_0) + 1 < max_num_same_x + 1) or \
               np.any(np.all(1 - mask, axis=0))

    subset_idx = mask.ravel(order='F').nonzero()[0]
    S_0 = np.eye(num_coef)[subset_idx, :]

    if S_0.shape[0] == 0:
        S = np.zeros((0, num_coef * 2))
    else:
        S = linalg.block_diag(S_0, S_0)

    return S


def planar_sel_coef_subset(shape_coef1, shape_coef2, num_non_zero,
                           max_num_same_x=1, max_num_same_y=1):
    """
    Select subsets of the 2D filters with total number of entries at
    least num_non_zero that should be zero. In the end, the 2D filters
    only have num_non_zero entries of non-zero values.
    :param shape_coef1: a tuple of size 2 for the shape of filter 1
    :param shape_coef2: a tuple of size 2 for the shape of filter 2
    :param num_non_zero: number of non-zero entries in the 2D filters.
            Typically num_dirac + 1, where num_dirac is the number of Dirac deltas.
    :param max_num_same_x: maximum number of Dirac deltas that have the
            same horizontal locations. This will impose the minimum dimension
            of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the
            same vertical locations This will impose the minimum dimension
            of the annihilating filter used.
    :return:
    """
    # the selection indices that corresponds to the part where the
    # coefficients are DIFFERENT from zero
    num_coef1 = 1
    for dim_loop in shape_coef1:
        num_coef1 *= dim_loop

    num_coef2 = 1
    for dim_loop in shape_coef2:
        num_coef2 *= dim_loop

    more = True

    while more:
        mask1 = np.zeros(num_coef1, dtype=int)
        non_zero_ind1 = np.random.permutation(num_coef1)[:num_non_zero]
        mask1[non_zero_ind1] = 1
        mask1 = np.reshape(mask1, shape_coef1, order='F')
        cord1_0, cord1_1 = np.nonzero(mask1)

        more = (np.max(cord1_1) - np.min(cord1_1) + 1 < max_num_same_y + 1) or \
               (np.max(cord1_0) - np.min(cord1_0) + 1 < max_num_same_x + 1) or \
               np.any(np.all(1 - mask1, axis=0))

    mask2 = mask1

    subset_idx_row = (1 - mask1).ravel(order='F').nonzero()[0]
    subset_idx_col = (1 - mask2).ravel(order='F').nonzero()[0]

    S_row = np.eye(num_coef1)[subset_idx_row, :]
    S_col = np.eye(num_coef2)[subset_idx_col, :]

    if S_row.shape[0] == 0 and S_col.shape[0] == 0:
        S = np.zeros((0, num_coef1 + num_coef2))
    elif S_row.shape[0] == 0 and S_col.shape[0] != 0:
        S = np.column_stack((np.zeros((S_col.shape[0], num_coef1)), S_col))
    elif S_row.shape[0] != 0 and S_col.shape[0] == 0:
        S = np.column_stack((S_row, np.zeros((S_row.shape[0], num_coef2))))
    else:
        S = linalg.block_diag(S_row, S_col)

    return S


def planar_amp_mtx(xk, yk, x_samp_loc, y_samp_loc, bandwidth, taus):
    """
    Build the linear mapping that relates the Dirac amplitudes to
    the ideal low-pass filtered samples.
    :param xk: Dirac locations (x-axis)
    :param yk: Dirac locations (y-axis)
    :param x_samp_loc: sampling locations (x-axis)
    :param y_samp_loc: sampling locations (y-axis)
    :param bandwidth: a tuple of size 2 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 2 for the periods of the Dirac stream along x and y axis
    :return:
    """
    Bx, By = bandwidth
    taux, tauy = taus
    # reshape to use broadcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    x_samp_loc = np.reshape(x_samp_loc, (-1, 1), order='F')
    y_samp_loc = np.reshape(y_samp_loc, (-1, 1), order='F')

    mtx_amp = periodic_sinc(np.pi * Bx * (x_samp_loc - xk), Bx * taux) * \
              periodic_sinc(np.pi * By * (y_samp_loc - yk), By * tauy)

    return mtx_amp


def compute_effective_num_eq_2d(shape1, shape2):
    """
    compute the effective number of equations in 2D joint annihilation
    :param shape1: a tuple for the shape of the first filter
    :param shape2: a tuple for the shape of the first filter
    :return:
    """
    shape1, shape2 = np.array(shape1), np.array(shape2)
    shape_out = shape1 + shape2 - 1
    return (np.prod(shape_out) - 1) - (np.prod(shape1) - 1) - (np.prod(shape2) - 1)
