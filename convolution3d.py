"""
routines to build 3D convolutions
"""
from __future__ import division
import numpy as np
from scipy.sparse import spdiags
from build_linear_mapping_joint2d import convmtx2_valid


def convmtx3_valid(H, shape_input):
    """
    3D convolution matrix with the boundary condition 'valid', i.e., only filter within
    the given data block.
    :param H: 3D filter
    :param shape_input: a tuple of size 3 for the input signal shape
    :return:
    """
    # sanity check
    # either H or the the input should be large enough in order to move the filter block
    # around within the data block
    input_large_flag = np.all(np.array(shape_input) >= np.array(H.shape))
    H_large_flag = np.all(np.array(shape_input) <= np.array(H.shape))
    assert input_large_flag or H_large_flag
    k, l, m = shape_input
    s_H0, s_H1, s_H2 = H.shape
    # full output shape
    S_sz0 = int(np.abs(k - s_H0) + 1)
    S_sz1 = int(np.abs(l - s_H1) + 1)
    S_sz2 = int(np.abs(m - s_H2) + 1)

    data_type = H.dtype
    T = np.zeros((S_sz0 * S_sz1 * S_sz2, k * l * m), dtype=data_type)

    # CAUTION: note the difference in scipy's spdiags and Matlab's version.
    diag_entries = np.ones(max(m, S_sz2))
    if H_large_flag:
        for loop in range(s_H2):
            H_loop = H[:, :, loop]
            T_2d_loop = convmtx2_valid(H_loop, k, l)
            outer_diag = (spdiags(diag_entries, m - loop - 1, S_sz2, m).toarray()).astype(data_type)
            T += np.kron(outer_diag, T_2d_loop)
    else:
        for loop in range(s_H2):
            H_loop = H[:, :, loop]
            T_2d_loop = convmtx2_valid(H_loop, k, l)
            outer_diag = (spdiags(diag_entries, s_H2 - loop - 1, S_sz2, m).toarray()).astype(data_type)
            T += np.kron(outer_diag, T_2d_loop)

    return T


if __name__ == '__main__':
    '''
    test cases for the 3D convolution matrices
    '''
    from scipy import linalg
    # case I: H has the larger size compared with the input size.
    # Here, H is in fact the data (to be filtered) and the input is the 3D filter
    data_blk = np.random.randn(3, 3, 3)
    coef_blk = np.random.randn(2, 2, 2)
    # data_blk = np.reshape(np.arange(27), (3, 3, 3), order='F')
    # coef_blk = np.reshape(np.arange(8), (2, 2, 2), order='F')
    Tmtx = convmtx3_valid(data_blk, coef_blk.shape)
    res1 = Tmtx.dot(coef_blk.flatten('F'))
    print(res1)

    # case II: H has the smaller size compared with the input size.
    # Here, H is the actual filter while the input is the data to be filtered
    Rmtx = convmtx3_valid(coef_blk, data_blk.shape)
    res2 = Rmtx.dot(data_blk.flatten('F'))
    print(res2)

    print(linalg.norm(res1 - res2))
