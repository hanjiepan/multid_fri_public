"""
Alternative implementation of the FRI reconstruction algorithm (2D) for the speed
consideration.
"""
from __future__ import division
import numpy as np
from scipy import linalg
from build_linear_mapping_joint2d import planar_sel_coef_subset, R_mtx_joint, \
    T_mtx_joint, convmtx2_valid, compute_effective_num_eq_2d


def dirac_recon_joint_alg_fast(G, measurement, num_dirac, shape_b,
                               flatten_order='F',
                               noise_level=0,
                               max_ini=100, stop_cri='mse', max_inner_iter=20,
                               max_num_same_x=1, max_num_same_y=1):
    """
    ALGORITHM that reconstructs 2D Dirac deltas jointly
        min     |a - Gb|^2
        s.t.    c_1 * b = 0
                c_2 * b = 0

    This is a faster implementation, which involves using nested inverses and
    reusing intermediate results, etc.

    The new formulation exploit the fact that c_1 and c_2 are linearly indepdnet.
    Hence, the effective number of unknowns are less than the total size of the two filters.

    :param G: the linear mapping that links the unknown uniformly sampled
            sinusoids to the given measurements
    :param measurement: the given measurements of the 2D Dirac deltas
    :param num_dirac: number of Dirac deltas
    :param shape_b: shape of the (2D) uniformly sampled sinusoids
    :param flatten_order: flatten order to be used. This is related to how G is build.
            If the dimension 0 of G is 'C' ordered, then flattern_order = 'C';
            otherwise, flattern_order = 'F'.
    :param noise_level: noise level present in the given measurements
    :param max_ini: maximum number of random initializations
    :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
    :param max_inner_iter: maximum number of inner iterations for each random initializations
    :param max_num_same_x: maximum number of Dirac deltas that have the same horizontal locations.
            This will impose the minimum dimension of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the same vertical locations
            This will impose the minimum dimension of the annihilating filter used.
    :return:
    """
    check_finite = False  # use False for faster speed
    compute_mse = (stop_cri == 'mse')
    measurement = measurement.flatten(flatten_order)
    num_non_zero = num_dirac + 2

    # choose the shapes of the 2D annihilating filters (as square as possible)
    # total number of entries should be at least num_dirac + 1
    shape_c_0 = int(np.ceil(np.sqrt(num_non_zero)))
    shape_c_1 = int(np.ceil(num_non_zero / shape_c_0))

    # sanity check
    assert shape_c_0 * shape_c_1 >= num_non_zero

    # in case of common roots, the filter has to satisfy a certain minimum dimension
    shape_c_1 = max(shape_c_1, max_num_same_y + 1)
    shape_c_0 = int(np.ceil(num_non_zero / shape_c_1))
    shape_c_0 = max(shape_c_0, max_num_same_x + 1)

    shape_c = (shape_c_0, shape_c_1)

    # total number of coefficients
    num_coef = shape_c_0 * shape_c_1

    # determine the effective row rank of the joint annihilation right-dual matrix
    c1_test = np.random.randn(shape_c_0, shape_c_1) + \
              1j * np.random.randn(shape_c_0, shape_c_1)
    c2_test = np.random.randn(shape_c_0, shape_c_1) + \
              1j * np.random.randn(shape_c_0, shape_c_1)
    R_test = R_mtx_joint(c1_test, c2_test, shape_b)
    try:
        s_test = linalg.svd(R_test, compute_uv=False)
        shape_Tb0_effective = min(R_test.shape) - np.where(np.abs(s_test) < 1e-12)[0].size
    except ValueError:
        # the effective number of equations as predicted by the derivation
        shape_Tb0_effective = \
            min(max(np.prod(shape_b) - compute_effective_num_eq_2d(shape_c, shape_c),
                    num_coef - 1 + num_coef - 1),
                R_test.shape[0])

    # sizes of various matrices / vectors
    sz_coef = num_coef * 2 - 1  # -1 because of linear independence
    sz_S0 = num_coef * 2 - 2 * num_non_zero

    # pre-compute a few things
    GtG = np.dot(G.conj().T, G)
    lu_GtG = linalg.lu_factor(GtG, check_finite=check_finite)
    beta, lsq_err = linalg.lstsq(G, measurement)[:2]
    beta_reshaped = np.reshape(beta, shape_b, order='F')
    # a block diagonal matrix
    Tbeta0 = T_mtx_joint(beta_reshaped, shape_c, shape_c)

    # use one block of Tbeta0 to do QR decomposition
    Tbeta_one_blk = convmtx2_valid(beta_reshaped, shape_c[0], shape_c[1])
    Qtilde_full = linalg.qr(Tbeta_one_blk.conj().T, mode='economic', pivoting=False)[0]
    Qtilde1 = Qtilde_full
    Qtilde2 = Qtilde_full[:, 1:]
    Qtilde_mtx = linalg.block_diag(Qtilde1, Qtilde2)
    Tbeta0_Qtilde = np.dot(Tbeta0, Qtilde_mtx)

    # initializations
    min_error = np.inf
    rhs = np.concatenate((np.zeros(sz_coef + sz_S0, dtype=complex),
                          np.append(np.ones(2, dtype=complex), 0)))

    c1_opt = None
    c2_opt = None

    # iterations over different random initializations of the annihilating filter coefficients
    ini = 0
    while ini < max_ini:
        ini += 1
        c1 = np.random.randn(shape_c_0, shape_c_1) + \
             1j * np.random.randn(shape_c_0, shape_c_1)
        c2 = np.random.randn(shape_c_0, shape_c_1) + \
             1j * np.random.randn(shape_c_0, shape_c_1)

        # build a selection matrix that chooses a subset of c1 and c2 to ZERO OUT
        S = np.dot(planar_sel_coef_subset((shape_c_0, shape_c_1),
                                          (shape_c_0, shape_c_1),
                                          num_non_zero=num_non_zero,
                                          max_num_same_x=max_num_same_x,
                                          max_num_same_y=max_num_same_y),
                   Qtilde_mtx)
        S_H = S.conj().T  # S is real valued

        # the initializations of the annihilating filter coefficients
        Gamma0 = np.column_stack((
            linalg.block_diag(
                np.dot(Qtilde1.T, c1.flatten('F'))[:, np.newaxis],
                np.dot(Qtilde2.T, c2.flatten('F'))[:, np.newaxis]
            ),
            np.concatenate((
                np.dot(Qtilde1.T, c2.flatten('F')),
                np.dot(Qtilde2.T, c1.flatten('F'))
            ))[:, np.newaxis]
        ))

        mtx_S_row = np.hstack((S, np.zeros((sz_S0, sz_S0 + 3), dtype=complex)))

        # last row in mtx_loop
        mtx_last_row = np.hstack((Gamma0.T, np.zeros((3, sz_S0 + 3), dtype=complex)))

        R_loop = R_mtx_joint(c1, c2, shape_b)
        # use QR decomposition to extract effective lines of equations
        Q_H = linalg.qr(R_loop, mode='economic',
                        pivoting=False)[0][:, :shape_Tb0_effective].conj().T
        R_loop = np.dot(Q_H, R_loop)
        Tbeta_loop = np.dot(Q_H, Tbeta0_Qtilde)

        # inner loop for each random initialization
        for inner in range(max_inner_iter):
            if inner == 0:
                R_GtGinv_Rh = \
                    np.dot(R_loop,
                           linalg.lu_solve(lu_GtG, R_loop.conj().T,
                                           check_finite=check_finite)
                           )
                mtx_loop = \
                    np.vstack((
                        np.hstack((
                            np.dot(Tbeta_loop.conj().T,
                                   linalg.solve(R_GtGinv_Rh, Tbeta_loop,
                                                check_finite=check_finite)
                                   ),
                            S_H, Gamma0.conj()
                        )),
                        mtx_S_row,
                        mtx_last_row
                    ))
            else:
                mtx_loop[:sz_coef, :sz_coef] = Tbetah_R_GtGinv_Rh_inv_Tbeta

            # solve annihilating filter coefficients
            try:
                gamma = linalg.solve(mtx_loop, rhs)[:sz_coef]
                coef = np.dot(Qtilde_mtx, gamma)
            except linalg.LinAlgError:
                break
            c1 = np.reshape(coef[:num_coef], shape_c, order='F')
            c2 = np.reshape(coef[num_coef:], shape_c, order='F')

            # update the right-dual matrix R and T based on the new coefficients
            R_loop = R_mtx_joint(c1, c2, shape_b)
            # use QR decomposition to extract effective lines of equations
            Q_H = linalg.qr(R_loop, mode='economic',
                            pivoting=False)[0][:, :shape_Tb0_effective].conj().T
            R_loop = np.dot(Q_H, R_loop)
            Tbeta_loop = np.dot(Q_H, Tbeta0_Qtilde)

            # evaluate fitting error without computing b
            R_GtGinv_Rh = np.dot(R_loop,
                                 linalg.lu_solve(lu_GtG, R_loop.conj().T,
                                                 check_finite=check_finite))
            Tbetah_R_GtGinv_Rh_inv_Tbeta = \
                np.dot(
                    Tbeta_loop.conj().T,
                    linalg.solve(R_GtGinv_Rh, Tbeta_loop, check_finite=check_finite)
                )
            Tbeta_c = np.dot(Tbeta_loop, gamma)

            if inner == 0:
                mtx_error = np.row_stack((
                    np.column_stack((
                        R_GtGinv_Rh,
                        np.zeros((shape_Tb0_effective, 1), dtype=complex)
                    )),
                    np.append(Tbeta_c.conj()[np.newaxis, :], -1)
                ))
                rhs_error = np.append(Tbeta_c, 0)
            else:
                mtx_error[:shape_Tb0_effective, :shape_Tb0_effective] = R_GtGinv_Rh
                mtx_error[-1, :shape_Tb0_effective] = Tbeta_c.conj()
                rhs_error[:-1] = Tbeta_c

            l_rho = linalg.solve(mtx_error, rhs_error, check_finite=check_finite)
            # the error computed does not contain the constant offset |a - G beta|^2
            error_loop = l_rho[-1].real + lsq_err

            if 0 < error_loop < min_error:
                # check that the number of non-zero entries are
                # indeed num_dirac + 1 (could be less)
                c1[np.abs(c1) < 1e-2 * np.max(np.abs(c1))] = 0
                c2[np.abs(c2) < 1e-2 * np.max(np.abs(c2))] = 0
                nnz_cond = \
                    np.sum(1 - np.isclose(np.abs(c1), 0).astype(int)) == num_non_zero and \
                    np.sum(1 - np.isclose(np.abs(c2), 0).astype(int)) == num_non_zero
                cord1_0, cord1_1 = np.nonzero(c1)
                cord2_0, cord2_1 = np.nonzero(c2)
                min_order_cond = \
                    (np.max(cord2_0) - np.min(cord2_0) + 1 >= max_num_same_x + 1) and \
                    (np.max(cord1_1) - np.min(cord1_1) + 1 >= max_num_same_y + 1)
                if nnz_cond and min_order_cond:
                    min_error = error_loop
                    c1_opt = c1
                    c2_opt = c2
                    l_opt = l_rho[:-1]

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

        if c1_opt is None or c2_opt is None:
            max_ini += 1

    # compute b_opt at the end
    R_opt = R_mtx_joint(c1_opt, c2_opt, shape_b)
    # use QR decomposition to extract effective lines of equations
    Q_H = linalg.qr(R_opt, mode='economic',
                    pivoting=False)[0][:, :shape_Tb0_effective].conj().T
    R_opt = np.dot(Q_H, R_opt)
    b_opt = beta - linalg.lu_solve(lu_GtG, np.dot(R_opt.conj().T, l_opt),
                                   check_finite=check_finite)

    print('fitting SNR {:.2f}'.format(10 * np.log10(linalg.norm(measurement) ** 2 / min_error)))
    return c1_opt, c2_opt, min_error, b_opt, ini
