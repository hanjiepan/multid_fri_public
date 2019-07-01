"""
a collection of routines that build matrices used in the 3D joint annihilation
"""
from __future__ import division
import numpy as np
from scipy import linalg
from utils_2d import periodic_sinc, nd_distance
from convolution3d import convmtx3_valid
from build_linear_mapping_joint2d import compute_effective_num_eq_2d


def build_G_mtx_idealLP(samp_loc, bandwidth, taus, shape_b=None,
                        input_ordering='F', output_ordering='F', normalize=True):
    """
    Build the linear mapping from the uniformly sampled sinuosoids to the ideally lowpass
    filtered samples.
    :param samp_loc: a 3-COLUMN matrix that contains sampling locations (x, y, z)
    :param bandwidth: a tuple of size 3 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 3 for the periods of the Dirac stream along x, y and z axis
    :param shape_b: shape of the uniformly sampled sinusoids
    :param input_ordering: ordering convention used for the input of the linear mapping
    :param output_ordering:ordering convention used for the output of the linear mapping
    :return:
    """
    Bx, By, Bz = bandwidth
    taux, tauy, tauz = taus
    x_samp_loc, y_samp_loc, z_samp_loc = samp_loc[:, 0], samp_loc[:, 1], samp_loc[:, 2]
    if shape_b is None:
        # use round here because of finite accuracy leads to rounding error
        shape_b = (int(np.round(By * tauy)), int(np.round(Bx * taux)), int(np.round(Bz * tauz)))
        # assert shape_b[0] // 2 * 2 + 1 == shape_b[0]
        # assert shape_b[1] // 2 * 2 + 1 == shape_b[1]
        # assert shape_b[2] // 2 * 2 + 1 == shape_b[2]

    shape_b0, shape_b1, shape_b2 = shape_b
    if np.mod(shape_b0, 2) == 0:
        vec_y = np.arange(-shape_b0 // 2, (shape_b0 + 1) // 2)
    else:
        vec_y = np.arange(-(shape_b0 - 1) // 2, (shape_b0 + 1) // 2)
    if np.mod(shape_b1, 2) == 0:
        vec_x = np.arange(-shape_b1 // 2, (shape_b1 + 1) // 2)
    else:
        vec_x = np.arange(-(shape_b1 - 1) // 2, (shape_b1 + 1) // 2)
    if np.mod(shape_b2, 2) == 0:
        vec_z = np.arange(-shape_b2 // 2, (shape_b2 + 1) // 2)
    else:
        vec_z = np.arange(-(shape_b2 - 1) // 2, (shape_b2 + 1) // 2)

    # build the linear mapping that links the uniformly sampled sinusoids
    # to the ideally lowpass filtered samples
    m_grid_samp, n_grid_samp, p_grid_samp = np.meshgrid(vec_x, vec_y, vec_z)

    # reshape to use broadcasting
    m_grid_samp = np.reshape(m_grid_samp, (1, -1), order=input_ordering)
    n_grid_samp = np.reshape(n_grid_samp, (1, -1), order=input_ordering)
    p_grid_samp = np.reshape(p_grid_samp, (1, -1), order=input_ordering)
    x_samp_loc = np.reshape(x_samp_loc, (-1, 1), order=output_ordering)
    y_samp_loc = np.reshape(y_samp_loc, (-1, 1), order=output_ordering)
    z_samp_loc = np.reshape(z_samp_loc, (-1, 1), order=output_ordering)

    if normalize:
        G_mtx = np.exp(1j * 2 * np.pi / taux * x_samp_loc * m_grid_samp +
                       1j * 2 * np.pi / tauy * y_samp_loc * n_grid_samp +
                       1j * 2 * np.pi / tauz * z_samp_loc * p_grid_samp) / (Bx * By * Bz)
    else:
        G_mtx = np.exp(1j * 2 * np.pi / taux * x_samp_loc * m_grid_samp +
                       1j * 2 * np.pi / tauy * y_samp_loc * n_grid_samp +
                       1j * 2 * np.pi / tauz * z_samp_loc * p_grid_samp)

    return G_mtx


def R_mtx_joint3d(c1, c2, c3, shape_in):
    """
    build the right-dual matrix associated with 3D filters c1, c2 and c3
    :param c1: the first filter that the uniformly sampled sinusoid should be annihilated
    :param c2: the second filter that the uniformly sampled sinusoid should be annihilated
    :param c3: the third filter that the uniformly sampled sinusoid should be annihilated
    :param shape_in: a tuple of size 3 for the shape of the uniformly sampled sinusoid (i.e., b)
    :return:
    """
    return np.vstack([convmtx3_valid(c_loop, shape_in) for c_loop in [c1, c2, c3]])


def T_mtx_joint3d(b, shape_c1, shape_c2, shape_c3):
    """
    The convolution matrix associated with b * c2, b * c2 and b * c3 (jointly)
    :param b: the uniformly sampled sinusoid.
            Here we assume b is in a 3D shape already (instead of a vector form)
    :param shape_c1: shape of the first 2D filter
    :param shape_c2: shape of the second 2D filter
    :param shape_c3: shape of the third 2D filter
    :return:
    """
    return linalg.block_diag(
        convmtx3_valid(b, shape_c1),
        convmtx3_valid(b, shape_c2),
        convmtx3_valid(b, shape_c3)
    )


def cubical_sel_coef_subset_complement(shape_coef, num_non_zero):
    """
    Select subsets of the 3D filters with total number of entries at least num_non_zero
    that should be zero. In the end, the 3D filters only have num_non_zero entries of
    non-zero values
    :param shape_coef: a tuple of size 3 for the shape of filter 1
    :param shape_coef2: a tuple of size 3 for the shape of filter 2
    :param shape_coef3: a tuple of size 3 for the shape of filter 3
    :param num_non_zero: number of non-zero entries in the 2D filters.
            Typically num_dirac + 1, where num_dirac is the number of Dirac deltas.
    :param max_num_same_x: maximum number of Dirac deltas that have the
            same x locations. This will impose the minimum dimension
            of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the
            same y locations This will impose the minimum dimension
            of the annihilating filter used.
    :param max_num_same_z: maximum number of Dirac deltas that have the
            same z locations This will impose the minimum dimension
            of the annihilating filter used.
    :return:
    """
    # TODO: include extra constraints to cope with shared x, y, z cases.
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

        more = np.any(np.all(1 - mask, axis=(0, 2)))

    subset_idx = mask.ravel(order='F').nonzero()[0]

    S_0 = np.eye(num_coef)[subset_idx, :]

    if S_0.shape[0] == 0:
        S = np.zeros((0, num_coef * 3))
    else:
        S = linalg.block_diag(S_0, S_0, S_0)

    return S


def cubical_sel_coef_subset(shape_coef1, shape_coef2, shape_coef3, num_non_zero,
                            max_num_same_x=1, max_num_same_y=1, max_num_same_z=1):
    """
    Select subsets of the 3D filters with total number of entries at least num_non_zero
    that should be zero. In the end, the 3D filters only have num_non_zero entries of
    non-zero values
    :param shape_coef1: a tuple of size 3 for the shape of filter 1
    :param shape_coef2: a tuple of size 3 for the shape of filter 2
    :param shape_coef3: a tuple of size 3 for the shape of filter 3
    :param num_non_zero: number of non-zero entries in the 2D filters.
            Typically num_dirac + 1, where num_dirac is the number of Dirac deltas.
    :param max_num_same_x: maximum number of Dirac deltas that have the
            same x locations. This will impose the minimum dimension
            of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the
            same y locations This will impose the minimum dimension
            of the annihilating filter used.
    :param max_num_same_z: maximum number of Dirac deltas that have the
            same z locations This will impose the minimum dimension
            of the annihilating filter used.
    :return:
    """
    # TODO: include extra constraints to cope with shared x, y, z cases.
    # the selection indices that corresponds to the part where the
    # coefficients are DIFFERENT from zero
    num_coef1 = 1
    for dim_loop in shape_coef1:
        num_coef1 *= dim_loop

    num_coef2 = 1
    for dim_loop in shape_coef2:
        num_coef2 *= dim_loop

    num_coef3 = 1
    for dim_loop in shape_coef3:
        num_coef3 *= dim_loop

    '''for mask1'''
    more = True
    while more:
        mask1 = np.zeros(num_coef1, dtype=int)
        non_zero_ind1 = np.random.permutation(num_coef1)[:num_non_zero]
        mask1[non_zero_ind1] = 1
        mask1 = np.reshape(mask1, shape_coef1, order='F')
        # cord1_0, cord1_1, cord1_2 = np.nonzero(mask1)

        more = np.any(np.all(1 - mask1, axis=(0, 2)))
        # (np.max(cord1_1) - np.min(cord1_1) + 1 < max_num_same_y + 1)

    mask2 = mask3 = mask1

    subset_idx_row = (1 - mask1).ravel(order='F').nonzero()[0]
    subset_idx_col = (1 - mask2).ravel(order='F').nonzero()[0]
    subset_idx_depth = (1 - mask3).ravel(order='F').nonzero()[0]

    S_row = np.eye(num_coef1)[subset_idx_row, :]
    S_col = np.eye(num_coef2)[subset_idx_col, :]
    S_depth = np.eye(num_coef3)[subset_idx_depth, :]

    if S_row.shape[0] == 0 and S_col.shape[0] == 0 and S_depth.shape[0] == 0:
        S = np.zeros((0, num_coef1 + num_coef2 + num_coef3))
    elif S_row.shape[0] == 0 and S_col.shape[0] == 0 and S_depth.shape[0] != 0:
        S = np.column_stack((
            np.zeros((S_depth.shape[0], num_coef1 + num_coef2)),
            S_depth
        ))
    elif S_row.shape[0] == 0 and S_col.shape[0] != 0 and S_depth.shape[0] == 0:
        S = np.column_stack((
            np.zeros((S_col.shape[0], num_coef1)),
            S_col,
            np.zeros((S_col.shape[0], num_coef3))
        ))
    elif S_row.shape[0] != 0 and S_col.shape[0] == 0 and S_depth.shape[0] == 0:
        S = np.column_stack((
            S_row,
            np.zeros((S_row.shape[0], num_coef2 + num_coef3))
        ))
    elif S_row.shape[0] == 0 and S_col.shape[0] != 0 and S_depth.shape[0] != 0:
        S = np.column_stack((
            np.zeros((S_col.shape[0] + S_depth.shape[0], num_coef1)),
            linalg.block_diag(S_col, S_depth)
        ))
    elif S_row.shape[0] != 0 and S_col.shape[0] == 0 and S_depth.shape[0] != 0:
        S = linalg.block_diag(
            S_row,
            np.column_stack((
                np.zeros((S_depth.shape[0], num_coef2)),
                S_depth
            ))
        )
    elif S_row.shape[0] != 0 and S_col.shape[0] != 0 and S_depth.shape[0] == 0:
        S = np.column_stack((
            linalg.block_diag(S_row, S_col),
            np.zeros((S_row.shape[0] + S_col.shape[0], num_coef3))
        ))
    else:
        S = linalg.block_diag(S_row, S_col, S_depth)

    return S


def cubical_amp_mtx(dirac_locs, samp_locs, bandwidth, taus):
    """
    Build the linear mapping that relates the Dirac amplitudes to
    the ideal low-pass filtered samples (for 3D cases)
    :param dirac_locs: a num_dirac by 3 matrix with the columns corresponds to
            the Dirac x, y and z locations. num_dirac is the number of Dirac.
    :param samp_locs: an N by 3 matrix with the columns corresponds to the 3D
            sampling locations along the x, y and z axis. N is the total number of samples.
    :param bandwidth: a tuple of size 3 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 3 for the periods of the Dirac stream along x, y and z axis.
    :return:
    """
    Bx, By, Bz = bandwidth
    taux, tauy, tauz = taus
    xk, yk, zk = dirac_locs[:, 0], dirac_locs[:, 1], dirac_locs[:, 2]
    x_samp_loc, y_samp_loc, z_samp_loc = samp_locs[:, 0], samp_locs[:, 1], samp_locs[:, 2]

    # reshape to use broadcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    zk = np.reshape(zk, (1, -1), order='F')

    x_samp_loc = np.reshape(x_samp_loc, (-1, 1), order='F')
    y_samp_loc = np.reshape(y_samp_loc, (-1, 1), order='F')
    z_samp_loc = np.reshape(z_samp_loc, (-1, 1), order='F')

    mtx_amp = periodic_sinc(np.pi * Bx * (x_samp_loc - xk), Bx * taux) * \
              periodic_sinc(np.pi * By * (y_samp_loc - yk), By * tauy) * \
              periodic_sinc(np.pi * Bz * (z_samp_loc - zk), Bz * tauz)

    return mtx_amp


def compute_effective_num_eq_3d(shape1, shape2, shape3):
    """
    compute the effective number of equations in 3D joint annihilation
    :param shape1: a tuple for the shape of the first filter
    :param shape2: a tuple for the shape of the second filter
    :param shape3: a tuple for the shape of the third filter
    :return:
    """
    shape1, shape2, shape3 = np.array(shape1), np.array(shape2), np.array(shape3)
    # shape for the convolution among all three filters
    shape_out = shape1 + shape2 + shape3 - 2
    return (np.prod(shape_out) - 1) - \
           compute_effective_num_eq_2d(shape1, shape2) - \
           compute_effective_num_eq_2d(shape1, shape3) - \
           compute_effective_num_eq_2d(shape2, shape3) - \
           (np.prod(shape1) - 1) - (np.prod(shape2) - 1) - (np.prod(shape3) - 1)


def check_linear_dependency(m1, m2, m3, gen_val=True):
    """
    check if the three masks are linearly dependent
    :param m1: mask 1
    :param m2: mask 2
    :param m3: mask 3
    :return:
    """
    shape_common = (max(m1.shape[0], m2.shape[0], m3.shape[0]),
                    max(m1.shape[1], m2.shape[1], m3.shape[1]),
                    max(m1.shape[2], m2.shape[2], m3.shape[2]))
    pad_sz_m1 = ((0, shape_common[0] - m1.shape[0]),
                 (0, shape_common[1] - m1.shape[1]),
                 (0, shape_common[2] - m1.shape[2]))
    pad_sz_m2 = ((0, shape_common[0] - m2.shape[0]),
                 (0, shape_common[1] - m2.shape[1]),
                 (0, shape_common[2] - m2.shape[2]))
    pad_sz_m3 = ((0, shape_common[0] - m3.shape[0]),
                 (0, shape_common[1] - m3.shape[1]),
                 (0, shape_common[2] - m3.shape[2]))
    m1_pad = np.pad(m1, pad_sz_m1, mode='constant', constant_values=0)
    m2_pad = np.pad(m2, pad_sz_m2, mode='constant', constant_values=0)
    m3_pad = np.pad(m3, pad_sz_m3, mode='constant', constant_values=0)
    if gen_val:
        # generate random values to fill in the non-zero entries
        m1_rnd = m1_pad * np.random.randn(*shape_common)
        m2_rnd = m2_pad * np.random.randn(*shape_common)
        m3_rnd = m3_pad * np.random.randn(*shape_common)
    else:
        m1_rnd = m1_pad
        m2_rnd = m2_pad
        m3_rnd = m3_pad

    m12 = np.column_stack((m1_rnd.flatten('F'), m2_rnd.flatten('F')))
    coef = linalg.lstsq(m12, m3_rnd.flatten('F'))[0]
    if linalg.norm(np.dot(m12, coef) - m3_rnd.flatten('F')) < 1e-8:
        return True
    else:
        return False


if __name__ == '__main__':
    '''
    test cases for the convolution building functions
    '''
    shape_coef1 = (2, 2, 2)
    shape_coef2 = (2, 3, 2)
    shape_coef3 = (2, 2, 3)
    num_non_zero = 7 + 1
    S = cubical_sel_coef_subset(shape_coef1, shape_coef2, shape_coef3, num_non_zero)
    print(S.shape)
