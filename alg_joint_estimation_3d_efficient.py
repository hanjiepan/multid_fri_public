"""
Alternative implementation of the FRI reconstruction algorithm (3D) for the speed
consideration.
"""
from __future__ import division
import numpy as np
from scipy import linalg
from convolution3d import convmtx3_valid
from build_linear_mapping_joint3d import cubical_sel_coef_subset, \
    cubical_sel_coef_subset_complement, R_mtx_joint3d, T_mtx_joint3d, \
    compute_effective_num_eq_3d


def dirac_recon_joint_alg_fast(G, a, num_dirac, shape_b, noise_level=0,
                               max_ini=100, stop_cri='mse', max_inner_iter=20,
                               max_num_same_x=1, max_num_same_y=1, max_num_same_z=1,
                               refine_coef=False):
    """
    ALGORITHM that reconstructs 2D Dirac deltas jointly
        min     |a - Gb|^2
        s.t.    c_1 * b = 0
                c_2 * b = 0
                c_3 * b = 0

    This is an optimzied version for speed consideration. For instance, we try to
    reuse intermediate results and pre-compute a few matrices, etc.

    :param G: the linear mapping that links the unknown uniformly sampled
            sinusoids to the given measurements
    :param a: the given measurements of the 3D Dirac deltas
    :param num_dirac: number of Dirac deltas
    :param shape_b: shape of the (3D) uniformly sampled sinusoids
    :param noise_level: noise level present in the given measurements
    :param max_ini: maximum number of random initializations
    :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
    :param max_inner_iter: maximum number of inner iterations for each random initializations
    :param max_num_same_x: maximum number of Dirac deltas that have the same horizontal locations.
            This will impose the minimum dimension of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the same vertical locations
            This will impose the minimum dimension of the annihilating filter used.
    :param max_num_same_z: maximum number of Dirac deltas that have the same depth locations
            This will impose the minimum dimension of the annihilating filter used.
    :return:
    """
    check_finite = False  # use False for faster speed
    compute_mse = (stop_cri == 'mse')
    a = a.flatten('F')
    num_non_zero = num_dirac + 3

    shape_c_0 = int(np.ceil(num_non_zero ** (1. / 3)))
    shape_c_1 = max(int(np.ceil((num_non_zero / shape_c_0) ** 0.5)), 2)
    shape_c_2 = max(int(np.ceil((num_non_zero / (shape_c_0 * shape_c_1)))), 2)

    # sanity check
    assert shape_c_0 * shape_c_1 * shape_c_2 >= num_non_zero

    shape_c = (shape_c_0, shape_c_1, shape_c_2)
    # total number of coefficients in c1 and c2
    num_coef = shape_c_0 * shape_c_1 * shape_c_2

    # determine the effective row rank of the joint annihilation right-dual matrix
    c1_test = np.random.randn(*shape_c) + 1j * np.random.randn(*shape_c)
    c2_test = np.random.randn(*shape_c) + 1j * np.random.randn(*shape_c)
    c3_test = np.random.randn(*shape_c) + 1j * np.random.randn(*shape_c)
    R_test = R_mtx_joint3d(c1_test, c2_test, c3_test, shape_b)
    try:
        s_test = linalg.svd(R_test, compute_uv=False)
        shape_Tb0_effective = min(R_test.shape) - np.where(np.abs(s_test) < 1e-12)[0].size
    except ValueError:
        # the effective number of equations as predicted by the derivation
        shape_Tb0_effective = \
            min(max((num_coef - 1) * 3,
                    np.prod(shape_b) - compute_effective_num_eq_3d(shape_c, shape_c, shape_c)),
                R_test.shape[0])
    if shape_Tb0_effective == R_test.shape[0]:
        need_QR = False
    else:
        need_QR = True

    # print('Tb0: {}'.format(shape_Tb0_effective))
    # print('R_sz0: {}'.format(R_test.shape[0]))
    # print('need QR: {}'.format(need_QR))

    # sizes of various matrices / vectors
    sz_coef = 3 * num_non_zero

    # pre-compute a few things
    # we use LU decomposition so that later we can use lu_solve, which is much faster
    GtG = np.dot(G.conj().T, G)
    lu_GtG = linalg.lu_factor(GtG, check_finite=check_finite)
    beta = linalg.lstsq(G, a)[0]
    Tbeta0 = T_mtx_joint3d(np.reshape(beta, shape_b, order='F'),
                           shape_c, shape_c, shape_c)
    if not need_QR:
        Tbeta_loop = Tbeta0

    # initializations
    min_error = np.inf
    rhs = np.concatenate((np.zeros(sz_coef, dtype=complex),
                          np.ones(3, dtype=complex)))
    c1_opt = None
    c2_opt = None
    c3_opt = None

    # iterations over different random initializations of the annihilating filter coefficients
    ini = 0
    while ini < max_ini:
        ini += 1
        c1 = np.random.randn(*shape_c) + 1j * np.random.randn(*shape_c)
        c2 = np.random.randn(*shape_c) + 1j * np.random.randn(*shape_c)
        c3 = np.random.randn(*shape_c) + 1j * np.random.randn(*shape_c)
        # the initializations of the annihilating filter coefficients
        Gamma0 = linalg.block_diag(c1.flatten('F')[:, np.newaxis],
                               c2.flatten('F')[:, np.newaxis],
                               c3.flatten('F')[:, np.newaxis])

        # build a selection matrix that chooses a subset of c1 and c2 to ZERO OUT
        S_complement = cubical_sel_coef_subset_complement(shape_c, num_non_zero=num_non_zero)
        S_H = S_complement.conj().T
        S_Gamma0 = np.dot(S_complement, Gamma0)

        # last row in mtx_loop
        mtx_last_row = np.hstack((
            S_Gamma0.conj().T, np.zeros((3, 3), dtype=complex)
        ))

        R_loop = R_mtx_joint3d(c1, c2, c3, shape_b)
        if need_QR:
            # use QR decomposition to extract effective lines of equations
            Q_H = linalg.qr(R_loop, mode='economic',
                            pivoting=False)[0][:, :shape_Tb0_effective].conj().T
            R_loop = np.dot(Q_H, R_loop)
            Tbeta_loop = np.dot(Q_H, Tbeta0)
            # Q_full, U_full = linalg.qr(R_loop, mode='economic', pivoting=False)
            # R_loop = U_full[:shape_Tb0_effective, :]
            # Tbeta_loop = np.dot(Q_full[:, :shape_Tb0_effective].conj().T, Tbeta0)

        # inner loop for each random initialization
        Tbetah_R_GtGinv_Rh_inv_Tbeta = None
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
                            np.dot(
                                S_complement,
                                np.dot(
                                    np.dot(Tbeta_loop.conj().T,
                                           linalg.solve(R_GtGinv_Rh, Tbeta_loop,
                                                        check_finite=check_finite)
                                           ),
                                    S_H)
                            ),
                            S_Gamma0
                        )),
                        mtx_last_row
                    ))
            else:
                mtx_loop[:sz_coef, :sz_coef] = Tbetah_R_GtGinv_Rh_inv_Tbeta

            # solve annihilating filter coefficients
            try:
                coef = np.dot(S_H, linalg.solve(mtx_loop, rhs)[:sz_coef])
            except linalg.LinAlgError:
                break
            c1 = np.reshape(coef[:num_coef], shape_c, order='F')
            c2 = np.reshape(coef[num_coef:num_coef + num_coef], shape_c, order='F')
            c3 = np.reshape(coef[num_coef + num_coef:], shape_c, order='F')

            # update the right-dual matrix R and T based on the new coefficients
            R_loop = R_mtx_joint3d(c1, c2, c3, shape_b)
            if need_QR:
                # use QR decomposition to extract effective lines of equations
                Q_H = linalg.qr(R_loop, mode='economic',
                                pivoting=False)[0][:, :shape_Tb0_effective].conj().T
                R_loop = np.dot(Q_H, R_loop)
                Tbeta_loop = np.dot(Q_H, Tbeta0)
                # Q_full, U_full = linalg.qr(R_loop, mode='economic', pivoting=False)
                # R_loop = U_full[:shape_Tb0_effective, :]
                # Tbeta_loop = np.dot(Q_full[:, :shape_Tb0_effective].conj().T, Tbeta0)

            # evaluate fitting error without computing b
            '''implementation I, which involves a two-layer nested matrix inverses'''
            # Tbetah_R_GtGinv_Rh_inv_Tbeta = \
            #     np.dot(Tbeta_loop.conj().T,
            #            linalg.solve(
            #                np.dot(R_loop,
            #                       linalg.lu_solve(lu_GtG, R_loop.conj().T,
            #                                       check_finite=check_finite)),
            #                Tbeta_loop, check_finite=check_finite)
            #            )
            # # the actual error is this value + |a - G beta|^2, which is a constant
            # error_loop = \
            #     np.real(np.dot(coef.conj().T,
            #                    np.dot(Tbetah_R_GtGinv_Rh_inv_Tbeta, coef)))

            '''implementation II, which only involves G^h G inverse and 
            not too much extra computational cost compared with implementation I'''
            R_GtGinv_Rh = np.dot(R_loop,
                                 linalg.lu_solve(lu_GtG, R_loop.conj().T,
                                                 check_finite=check_finite)
                                 )
            try:
                Tbetah_R_GtGinv_Rh_inv_Tbeta = \
                    np.dot(
                        S_complement,
                        np.dot(
                            np.dot(
                                Tbeta_loop.conj().T,
                                linalg.solve(R_GtGinv_Rh, Tbeta_loop, check_finite=check_finite)
                            ),
                            S_H
                        )
                    )
            except linalg.LinAlgError:
                break

            Tbeta_c = np.dot(Tbeta_loop, coef)
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
            error_loop = l_rho[-1].real

            if 0 < error_loop < min_error:
                # check that the number of non-zero entries are
                # indeed num_dirac + 1 (could be less)
                c1[np.abs(c1) < 1e-2 * np.max(np.abs(c1))] = 0
                c2[np.abs(c2) < 1e-2 * np.max(np.abs(c2))] = 0
                c3[np.abs(c3) < 1e-2 * np.max(np.abs(c3))] = 0
                nnz_cond = \
                    np.sum(1 - np.isclose(np.abs(c1), 0).astype(int)) == num_non_zero and \
                    np.sum(1 - np.isclose(np.abs(c2), 0).astype(int)) == num_non_zero and \
                    np.sum(1 - np.isclose(np.abs(c3), 0).astype(int)) == num_non_zero
                # TODO: add the checks for cases when certain number of Dirac share the x, y, z coordinates
                if nnz_cond:
                    min_error = error_loop
                    c1_opt = c1
                    c2_opt = c2
                    c3_opt = c3
                    S_complement_opt = S_complement
                    l_opt = l_rho[:-1]

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

        if c1_opt is None or c2_opt is None or c3_opt is None:
            max_ini += 1

    # compute b_opt at the end
    R_opt = R_mtx_joint3d(c1_opt, c2_opt, c3_opt, shape_b)
    if need_QR:
        # use QR decomposition to extract effective lines of equations
        Q_H = linalg.qr(R_opt, mode='economic',
                        pivoting=False)[0][:, :shape_Tb0_effective].conj().T
        R_opt = np.dot(Q_H, R_opt)
        # R_opt = linalg.qr(R_opt, mode='economic',
        #                   pivoting=False)[1][:shape_Tb0_effective, :]
    '''use with implementation I'''
    # mtx_brecon = np.vstack((
    #     np.hstack((GtG, R_opt.conj().T)),
    #     np.hstack((R_opt, np.zeros((shape_Tb0_effective, shape_Tb0_effective))))
    # ))
    # b_opt = \
    #     linalg.solve(mtx_brecon,
    #                  np.concatenate((Gt_a,
    #                                  np.zeros(shape_Tb0_effective,
    #                                           dtype=complex)))
    #                  )[:sz_R1]
    '''use with implementation II'''
    b_opt = beta - linalg.lu_solve(lu_GtG, np.dot(R_opt.conj().T, l_opt),
                                   check_finite=check_finite)
    # use denoised FRI data b to estimate c as the final refinement
    if refine_coef:
        S_blk_H = S_complement_opt[
                  :S_complement_opt.shape[0] // 3,
                  :S_complement_opt.shape[1] // 3].conj().T
        Tb = np.dot(convmtx3_valid(np.reshape(b_opt, shape_b, order='F'), shape_c), S_blk_H)
        V = linalg.svd(Tb, compute_uv=True)[2].conj().T
        c1_opt, c2_opt, c3_opt = \
            np.reshape(np.dot(S_blk_H, V[:, -3]), shape_c, order='F'), \
            np.reshape(np.dot(S_blk_H, V[:, -2]), shape_c, order='F'), \
            np.reshape(np.dot(S_blk_H, V[:, -1]), shape_c, order='F')

    return c1_opt, c2_opt, c3_opt, min_error, b_opt, ini


def dirac_recon_joint_alg_fast_newform(G, a, num_dirac, shape_b, noise_level=0,
                                       max_ini=100, stop_cri='mse', max_inner_iter=20,
                                       max_num_same_x=1, max_num_same_y=1, max_num_same_z=1):
    """
    ALGORITHM that reconstructs 2D Dirac deltas jointly
        min     |a - Gb|^2
        s.t.    c_1 * b = 0
                c_2 * b = 0
                c_3 * b = 0

    This is an optimzied version for speed consideration. For instance, we try to
    reuse intermediate results and pre-compute a few matrices, etc.

    :param G: the linear mapping that links the unknown uniformly sampled
            sinusoids to the given measurements
    :param a: the given measurements of the 3D Dirac deltas
    :param num_dirac: number of Dirac deltas
    :param shape_b: shape of the (3D) uniformly sampled sinusoids
    :param noise_level: noise level present in the given measurements
    :param max_ini: maximum number of random initializations
    :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
    :param max_inner_iter: maximum number of inner iterations for each random initializations
    :param max_num_same_x: maximum number of Dirac deltas that have the same horizontal locations.
            This will impose the minimum dimension of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the same vertical locations
            This will impose the minimum dimension of the annihilating filter used.
    :param max_num_same_z: maximum number of Dirac deltas that have the same depth locations
            This will impose the minimum dimension of the annihilating filter used.
    :return:
    """
    check_finite = False  # use False for faster speed
    compute_mse = (stop_cri == 'mse')
    a = a.flatten('F')
    num_non_zero = num_dirac + 3

    shape_c1_0 = int(np.ceil(num_non_zero ** (1. / 3)))
    shape_c1_1 = max(int(np.ceil((num_non_zero / shape_c1_0) ** 0.5)), 2)
    shape_c1_2 = max(int(np.ceil((num_non_zero / (shape_c1_0 * shape_c1_1)))), 2)

    # sanity check
    assert shape_c1_0 * shape_c1_1 * shape_c1_2 >= num_non_zero

    shape_c3_0, shape_c3_1, shape_c3_2 = \
        shape_c2_0, shape_c2_1, shape_c2_2 = shape_c1_0, shape_c1_1, shape_c1_2

    shape_c1 = (shape_c1_0, shape_c1_1, shape_c1_2)
    shape_c2 = (shape_c2_0, shape_c2_1, shape_c2_2)
    shape_c3 = (shape_c3_0, shape_c3_1, shape_c3_2)

    # # check if there will be sufficient number of effective number of equations
    # update_shape = True
    # dict_keys = ['shape_c1', 'shape_c2', 'shape_c3']
    # shapes = {
    #     'shape_c1': list(shape_c1),
    #     'shape_c2': list(shape_c2),
    #     'shape_c3': list(shape_c3)
    # }
    # shapes_update_eq = [1, 0, 2]
    # shapes_update_neq = [2, 1, 0]
    # exp_count = 0
    # while update_shape:
    #     if compute_effective_num_eq_3d(shapes['shape_c1'],
    #                                    shapes['shape_c2'],
    #                                    shapes['shape_c3']) < num_dirac:
    #         shape_loop = shapes[dict_keys[exp_count]]
    #         if shape_loop[0] == shape_loop[1] == shape_loop[2]:
    #             shapes[dict_keys[exp_count]][shapes_update_eq[exp_count]] += 1
    #         else:
    #             shapes[dict_keys[exp_count]][shapes_update_neq[exp_count]] += 1
    #
    #         exp_count += 1
    #         exp_count = np.mod(exp_count, 3)
    #         update_shape = True
    #     else:
    #         update_shape = False
    #
    # shape_c1 = tuple(shapes['shape_c1'])
    # shape_c2 = tuple(shapes['shape_c2'])
    # shape_c3 = tuple(shapes['shape_c3'])
    # shape_c1_0, shape_c1_1, shape_c1_2 = shape_c1
    # shape_c2_0, shape_c2_1, shape_c2_2 = shape_c2
    # shape_c3_0, shape_c3_1, shape_c3_2 = shape_c3

    # total number of coefficients in c1 and c2
    num_coef1 = shape_c1_0 * shape_c1_1 * shape_c1_2
    num_coef2 = shape_c2_0 * shape_c2_1 * shape_c2_2
    num_coef3 = shape_c3_0 * shape_c3_1 * shape_c3_2

    # determine the effective row rank of the joint annihilation right-dual matrix
    c1_test = np.random.randn(*shape_c1) + 1j * np.random.randn(*shape_c1)
    c2_test = np.random.randn(*shape_c2) + 1j * np.random.randn(*shape_c2)
    c3_test = np.random.randn(*shape_c3) + 1j * np.random.randn(*shape_c3)
    R_test = R_mtx_joint3d(c1_test, c2_test, c3_test, shape_b)
    try:
        s_test = linalg.svd(R_test, compute_uv=False)
        shape_Tb0_effective = min(R_test.shape) - np.where(np.abs(s_test) < 1e-12)[0].size
    except ValueError:
        # the effective number of equations as predicted by the derivation
        shape_Tb0_effective = \
            min(max(num_coef1 - 1 + num_coef2 - 1 + num_coef3 - 1,
                    np.prod(shape_b) - compute_effective_num_eq_3d(shape_c1, shape_c2, shape_c3)),
                R_test.shape[0])
    # assert shape_Tb0_effective == shape_Tb0_effective_thm  # just to make sure

    # sizes of various matrices / vectors
    sz_coef = num_coef1 + num_coef2 + num_coef3 - 3  # -3 because of linear independence
    sz_S0 = num_coef1 + num_coef2 + num_coef3 - 3 * num_non_zero

    # pre-compute a few things
    # we use LU decomposition so that later we can use lu_solve, which is much faster
    GtG = np.dot(G.conj().T, G)
    lu_GtG = linalg.lu_factor(GtG, check_finite=check_finite)
    beta = linalg.lstsq(G, a)[0]
    Tbeta0 = T_mtx_joint3d(np.reshape(beta, shape_b, order='F'),
                           shape_c1, shape_c2, shape_c3)
    # use one block of Tbeta0 to do QR decomposition
    Tbeta_one_blk = convmtx3_valid(np.reshape(beta, shape_b, order='F'), shape_c1)
    Qtilde_full = linalg.qr(Tbeta_one_blk.conj().T, mode='economic', pivoting=False)[0]
    Qtilde1 = Qtilde_full
    Qtilde2 = Qtilde_full[:, 1:]
    Qtilde3 = Qtilde_full[:, 2:]
    Qtilde_mtx = linalg.block_diag(Qtilde1, Qtilde2, Qtilde3)
    Tbeta0_Qtilde = np.dot(Tbeta0, Qtilde_mtx)

    # initializations
    min_error = np.inf
    rhs = np.concatenate((
        np.zeros(sz_coef + sz_S0, dtype=complex),
        np.concatenate((np.ones(3, dtype=complex),
                        np.zeros(3, dtype=complex)))
    ))
    c1_opt = None
    c2_opt = None
    c3_opt = None

    # iterations over different random initializations of the annihilating filter coefficients
    ini = 0
    while ini < max_ini:
        ini += 1
        c1 = np.random.randn(*shape_c1) + 1j * np.random.randn(*shape_c1)
        c2 = np.random.randn(*shape_c2) + 1j * np.random.randn(*shape_c2)
        c3 = np.random.randn(*shape_c3) + 1j * np.random.randn(*shape_c3)
        # the initializations of the annihilating filter coefficients
        Gamma0 = np.dot(
            Qtilde_mtx.T,
            np.column_stack((
                linalg.block_diag(c1.flatten('F')[:, np.newaxis],
                                  c2.flatten('F')[:, np.newaxis],
                                  c3.flatten('F')[:, np.newaxis]),
                np.concatenate((
                    c2.flatten('F'), c1.flatten('F'), np.zeros(num_coef3)
                )),
                np.concatenate((
                    c3.flatten('F'), np.zeros(num_coef2), c1.flatten('F')
                )),
                np.concatenate((
                    np.zeros(num_coef1), c3.flatten('F'), c2.flatten('F')
                ))
            ))
        )

        # build a selection matrix that chooses a subset of c1 and c2 to ZERO OUT
        S = np.dot(cubical_sel_coef_subset(shape_c1, shape_c2, shape_c3,
                                           num_non_zero=num_non_zero,
                                           max_num_same_x=max_num_same_x,
                                           max_num_same_y=max_num_same_y,
                                           max_num_same_z=max_num_same_z),
                   Qtilde_mtx)
        S_H = S.conj().T
        mtx_S_row = np.hstack((S, np.zeros((sz_S0, sz_S0 + 6), dtype=complex)))

        # last row in mtx_loop
        mtx_last_row = np.hstack((
            Gamma0.T, np.zeros((6, sz_S0 + 6), dtype=complex)
        ))

        R_loop = R_mtx_joint3d(c1, c2, c3, shape_b)
        # use QR decomposition to extract effective lines of equations
        Q_H = linalg.qr(R_loop, mode='economic',
                        pivoting=False)[0][:, :shape_Tb0_effective].conj().T
        R_loop = np.dot(Q_H, R_loop)
        Tbeta_loop = np.dot(Q_H, Tbeta0_Qtilde)

        # inner loop for each random initialization
        Tbetah_R_GtGinv_Rh_inv_Tbeta = None
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
            c1 = np.reshape(coef[:num_coef1], shape_c1, order='F')
            c2 = np.reshape(coef[num_coef1:num_coef1 + num_coef2], shape_c2, order='F')
            c3 = np.reshape(coef[num_coef1 + num_coef2:], shape_c3, order='F')

            # update the right-dual matrix R and T based on the new coefficients
            R_loop = R_mtx_joint3d(c1, c2, c3, shape_b)
            # use QR decomposition to extract effective lines of equations
            Q_H = linalg.qr(R_loop, mode='economic',
                            pivoting=False)[0][:, :shape_Tb0_effective].conj().T
            R_loop = np.dot(Q_H, R_loop)
            Tbeta_loop = np.dot(Q_H, Tbeta0_Qtilde)

            # evaluate fitting error without computing b
            '''implementation I, which involves a two-layer nested matrix inverses'''
            # Tbetah_R_GtGinv_Rh_inv_Tbeta = \
            #     np.dot(Tbeta_loop.conj().T,
            #            linalg.solve(
            #                np.dot(R_loop,
            #                       linalg.lu_solve(lu_GtG, R_loop.conj().T,
            #                                       check_finite=check_finite)),
            #                Tbeta_loop, check_finite=check_finite)
            #            )
            # # the actual error is this value + |a - G beta|^2, which is a constant
            # error_loop = \
            #     np.real(np.dot(coef.conj().T,
            #                    np.dot(Tbetah_R_GtGinv_Rh_inv_Tbeta, coef)))

            '''implementation II, which only involves G^h G inverse and 
            not too much extra computational cost compared with implementation I'''
            R_GtGinv_Rh = np.dot(R_loop,
                                 linalg.lu_solve(lu_GtG, R_loop.conj().T,
                                                 check_finite=check_finite)
                                 )
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
            error_loop = l_rho[-1].real

            if 0 < error_loop < min_error:
                # check that the number of non-zero entries are
                # indeed num_dirac + 1 (could be less)
                c1[np.abs(c1) < 1e-2 * np.max(np.abs(c1))] = 0
                c2[np.abs(c2) < 1e-2 * np.max(np.abs(c2))] = 0
                c3[np.abs(c3) < 1e-2 * np.max(np.abs(c3))] = 0
                nnz_cond = \
                    np.sum(1 - np.isclose(np.abs(c1), 0).astype(int)) == num_non_zero and \
                    np.sum(1 - np.isclose(np.abs(c2), 0).astype(int)) == num_non_zero and \
                    np.sum(1 - np.isclose(np.abs(c3), 0).astype(int)) == num_non_zero
                # TODO: add the checks for cases when certain number of Dirac share the x, y, z coordinates
                if nnz_cond:
                    min_error = error_loop
                    c1_opt = c1
                    c2_opt = c2
                    c3_opt = c3
                    l_opt = l_rho[:-1]

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

        if c1_opt is None or c2_opt is None or c3_opt is None:
            max_ini += 1

    # compute b_opt at the end
    R_opt = R_mtx_joint3d(c1_opt, c2_opt, c3_opt, shape_b)
    # use QR decomposition to extract effective lines of equations
    Q_H = linalg.qr(R_opt, mode='economic',
                    pivoting=False)[0][:, :shape_Tb0_effective].conj().T
    R_opt = np.dot(Q_H, R_opt)
    '''use with implementation I'''
    # mtx_brecon = np.vstack((
    #     np.hstack((GtG, R_opt.conj().T)),
    #     np.hstack((R_opt, np.zeros((shape_Tb0_effective, shape_Tb0_effective))))
    # ))
    # b_opt = \
    #     linalg.solve(mtx_brecon,
    #                  np.concatenate((Gt_a,
    #                                  np.zeros(shape_Tb0_effective,
    #                                           dtype=complex)))
    #                  )[:sz_R1]
    '''use with implementation II'''
    b_opt = beta - linalg.lu_solve(lu_GtG, np.dot(R_opt.conj().T, l_opt),
                                   check_finite=check_finite)
    return c1_opt, c2_opt, c3_opt, min_error, b_opt, ini
