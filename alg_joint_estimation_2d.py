from __future__ import division

import numpy as np
from scipy import linalg
from scipy.optimize import nnls

from alg_joint_estimation_2d_efficient import dirac_recon_joint_alg_fast
from build_linear_mapping_joint2d import planar_sel_coef_subset, R_mtx_joint, \
    T_mtx_joint, planar_amp_mtx, compute_effective_num_eq_2d, convmtx2_valid
from poly_common_roots_2d import find_roots_2d


def dirac_recon_joint_interface(a, num_dirac, samp_loc, bandwidth, taus, noise_level=0,
                                max_ini=100, stop_cri='max_iter', max_inner_iter=20,
                                strategy='both', max_num_same_x=1, max_num_same_y=1,
                                use_fast_alg=False):
    """
    INTERFACE that reconstructs 2D Dirac deltas jointly
    :param a: the given measurements of the 2D Dirac deltas
    :param num_dirac: number of Dirac deltas
    :param samp_loc: a 2-COLUMN matrix that contains sampling locations (x, y)
    :param bandwidth: a tuple of size 2 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 2 for the periods of the Dirac stream along x and y axis
    :param noise_level: noise level present in the given measurements
    :param max_ini: maximum number of random initializations
    :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
    :param max_inner_iter: maximum number of inner iterations for each random initializations
    :param strategy: either 'aggregate', which reconstructs amplitudes for
            all xk_recon and yk_recon and select the ones that have the largest amplitudes;
            Or 'one_by_one', which computes the fitting error by leaving out one of the
            reconstructed xk_recon and yk_recon each time. The ones that contributes least
            are eliminated.
    :param max_num_same_x: maximum number of Dirac deltas that have the same horizontal locations.
            This will impose the minimum dimension of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the same vertical locations
            This will impose the minimum dimension of the annihilating filter used.
    :param use_fast_alg: whether to use a faster implementation but may not be as stable as the
            exact implmentation.
    :return:
    """
    Bx, By = bandwidth
    taux, tauy = taus
    x_samp_loc, y_samp_loc = samp_loc[:, 0], samp_loc[:, 1]
    shape_b = (int(By * tauy), int(Bx * taux))
    assert shape_b[0] // 2 * 2 + 1 == shape_b[0]
    assert shape_b[1] // 2 * 2 + 1 == shape_b[1]

    freq_limit_y, freq_limit_x = shape_b[0] // 2, shape_b[1] // 2

    # build the linear mapping that links the uniformly sampled sinusoids
    # to the ideally lowpass filtered samples
    m_grid_samp, n_grid_samp = np.meshgrid(np.arange(-freq_limit_x, freq_limit_x + 1),
                                           np.arange(-freq_limit_y, freq_limit_y + 1))
    # reshape to use broadcasting
    m_grid_samp = np.reshape(m_grid_samp, (1, -1), order='F')
    n_grid_samp = np.reshape(n_grid_samp, (1, -1), order='F')
    x_samp_loc = np.reshape(x_samp_loc, (-1, 1), order='F')
    y_samp_loc = np.reshape(y_samp_loc, (-1, 1), order='F')

    mtx_fri2samp = np.exp(1j * 2 * np.pi / taux * m_grid_samp * x_samp_loc +
                          1j * 2 * np.pi / tauy * n_grid_samp * y_samp_loc) / (Bx * By)

    # apply joint FRI estimation
    if use_fast_alg:
        c1_opt, c2_opt = \
            dirac_recon_joint_alg_fast(mtx_fri2samp, a, num_dirac, shape_b,
                                       noise_level=noise_level, max_ini=max_ini,
                                       stop_cri=stop_cri, max_inner_iter=max_inner_iter,
                                       max_num_same_x=max_num_same_x,
                                       max_num_same_y=max_num_same_y)[:2]
    else:
        c1_opt, c2_opt = \
            dirac_recon_joint_alg(mtx_fri2samp, a, num_dirac, shape_b,
                                  noise_level=noise_level, max_ini=max_ini,
                                  stop_cri=stop_cri, max_inner_iter=max_inner_iter,
                                  max_num_same_x=max_num_same_x,
                                  max_num_same_y=max_num_same_y)[:2]

    # retrieve Dirac parameters
    xk_recon, yk_recon, amp_recon = \
        planar_extract_innovation(a, num_dirac, c1_opt, c2_opt, x_samp_loc, y_samp_loc,
                                  bandwidth, taus, strategy=strategy)

    return xk_recon, yk_recon, amp_recon


def dirac_recon_joint_alg(G, measurement, num_dirac, shape_b,
                          flatten_order='F',
                          num_band=1, noise_level=0,
                          max_ini=100, stop_cri='mse', max_inner_iter=20,
                          max_num_same_x=1, max_num_same_y=1):
    """
    ALGORITHM that reconstructs 2D Dirac deltas jointly
        min     |a - Gb|^2
        s.t.    c_1 * b = 0
                c_2 * b = 0

    This is the exact form that we have in the paper without any alternations for
    performance considerations, e.g., reusing intermediate results, etc.

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
    compute_mse = (stop_cri == 'mse')
    measurement = measurement.flatten(flatten_order)
    num_non_zero = num_dirac + 2

    # choose the shapes of the 2D annihilating filters (as square as possible)
    # total number of entries should be at least num_dirac + 1
    shape_c_0 = int(np.ceil(np.sqrt(num_non_zero)))
    shape_c_1 = int(np.ceil(num_non_zero / shape_c_0))
    if shape_c_0 > shape_c_1:
        shape_c_1, shape_c_0 = shape_c_0, shape_c_1

    # sanity check
    assert shape_c_0 * shape_c_1 >= num_non_zero

    # in case of common roots, the filter has to satisfy a certain minimum dimension
    shape_c_1 = max(shape_c_1, max_num_same_y + 1)
    shape_c_0 = int(np.ceil(num_non_zero / shape_c_1))
    shape_c_0 = max(shape_c_0, max_num_same_x + 1)

    shape_c = (shape_c_0, shape_c_1)

    # total number of coefficients in c1 and c2
    num_coef = shape_c_0 * shape_c_1

    if num_band > 1:
        def func_build_R(coef1, coef2, shape_in):
            R_mtx_band = R_mtx_joint(coef1, coef2, shape_in)
            return linalg.block_diag(*[R_mtx_band for _ in range(num_band)])
    else:
        def func_build_R(coef1, coef2, shape_in):
            return R_mtx_joint(coef1, coef2, shape_in)

    # determine the effective row rank of the joint annihilation right-dual matrix
    c1_test = np.random.randn(shape_c_0, shape_c_1) + \
              1j * np.random.randn(shape_c_0, shape_c_1)
    c2_test = np.random.randn(shape_c_0, shape_c_1) + \
              1j * np.random.randn(shape_c_0, shape_c_1)
    R_test = func_build_R(c1_test, c2_test, shape_b)
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
    sz_R1 = np.prod(shape_b) * num_band

    # a few indices that are fixed
    idx_bg0_Tb = sz_coef
    idx_end0_Tb = sz_coef + shape_Tb0_effective
    idx_bg1_Tb = 0
    idx_end1_Tb = sz_coef

    idx_bg0_TbH = 0
    idx_end0_TbH = sz_coef
    idx_bg1_TbH = sz_coef
    idx_end1_TbH = sz_coef + shape_Tb0_effective

    idx_bg0_Rc = sz_coef
    idx_end0_Rc = sz_coef + shape_Tb0_effective
    idx_bg1_Rc = sz_coef + shape_Tb0_effective
    idx_end1_Rc = sz_coef + shape_Tb0_effective + sz_R1

    idx_bg0_RcH = sz_coef + shape_Tb0_effective
    idx_end0_RcH = sz_coef + shape_Tb0_effective + sz_R1
    idx_bg1_RcH = sz_coef
    idx_end1_RcH = sz_coef + shape_Tb0_effective

    # pre-compute a few things
    GtG = np.dot(G.conj().T, G)
    Gt_a = np.dot(G.conj().T, measurement)
    try:
        beta = linalg.lstsq(G, measurement)[0]
    except np.linalg.linalg.LinAlgError:
        beta = linalg.solve(GtG, Gt_a)
    beta_reshaped = np.reshape(beta, (shape_b[0], shape_b[1], num_band), order='F')
    Tbeta0 = np.vstack([
        T_mtx_joint(beta_reshaped[:, :, band_count], shape_c, shape_c)
        for band_count in range(num_band)
    ])
    # QR-decomposition of Tbeta0.T
    Tbeta_band = np.vstack([convmtx2_valid(beta_reshaped[:, :, band_count], shape_c[0], shape_c[1])
                            for band_count in range(num_band)])
    Qtilde_full = linalg.qr(Tbeta_band.conj().T, mode='economic', pivoting=False)[0]
    Qtilde1 = Qtilde_full
    Qtilde2 = Qtilde_full[:, 1:]
    Qtilde_mtx = linalg.block_diag(Qtilde1, Qtilde2)
    Tbeta0_Qtilde = np.dot(Tbeta0, Qtilde_mtx)

    # initializations
    min_error = np.inf
    rhs = np.concatenate((np.zeros(sz_coef + shape_Tb0_effective + sz_R1 + sz_S0, dtype=float),
                          np.append(np.ones(2, dtype=float), 0)))

    rhs_bl = np.concatenate((Gt_a, np.zeros(shape_Tb0_effective, dtype=Gt_a.dtype)))
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
        S_H = S.conj().T

        # the initializations of the annihilating filter coefficients
        c0 = np.column_stack((
            linalg.block_diag(
                np.dot(Qtilde1.T, c1.flatten('F'))[:, np.newaxis],
                np.dot(Qtilde2.T, c2.flatten('F'))[:, np.newaxis]
            ),
            np.concatenate((
                np.dot(Qtilde1.T, c2.flatten('F')),
                np.dot(Qtilde2.T, c1.flatten('F'))
            ))[:, np.newaxis]
        ))

        mtx_S_row = np.hstack((S, np.zeros((sz_S0, shape_Tb0_effective + sz_R1 + sz_S0 + 3))))

        # last row in mtx_loop
        mtx_last_row = np.hstack((c0.T,
                                  np.zeros((3, shape_Tb0_effective + sz_R1 + sz_S0 + 3))))

        R_loop = func_build_R(c1, c2, shape_b)
        # use QR decomposition to extract effective lines of equations
        Q_H = linalg.qr(R_loop, mode='economic',
                        pivoting=False)[0][:, :shape_Tb0_effective].conj().T
        R_loop = np.dot(Q_H, R_loop)
        Tbeta_loop = np.dot(Q_H, Tbeta0_Qtilde)

        # inner loop for each random initialization
        for inner in range(max_inner_iter):
            if inner == 0:
                mtx_loop = np.vstack((
                    np.hstack((np.zeros((sz_coef, sz_coef)),
                               Tbeta_loop.conj().T,
                               np.zeros((sz_coef, sz_R1)), S_H, c0.conj())),
                    np.hstack((Tbeta_loop,
                               np.zeros((shape_Tb0_effective, shape_Tb0_effective)),
                               -R_loop,
                               np.zeros((shape_Tb0_effective, 3 + sz_S0)))),
                    np.hstack((np.zeros((sz_R1, sz_coef)),
                               -R_loop.conj().T,
                               GtG,
                               np.zeros((sz_R1, 3 + sz_S0)))),
                    mtx_S_row,
                    mtx_last_row
                ))
            else:
                mtx_loop[idx_bg0_Tb:idx_end0_Tb, idx_bg1_Tb:idx_end1_Tb] = Tbeta_loop
                mtx_loop[idx_bg0_TbH:idx_end0_TbH, idx_bg1_TbH:idx_end1_TbH] = Tbeta_loop.conj().T
                mtx_loop[idx_bg0_Rc:idx_end0_Rc, idx_bg1_Rc:idx_end1_Rc] = -R_loop
                mtx_loop[idx_bg0_RcH:idx_end0_RcH, idx_bg1_RcH:idx_end1_RcH] = -R_loop.conj().T

            # solve annihilating filter coefficients
            try:
                coef = np.dot(Qtilde_mtx, linalg.solve(mtx_loop, rhs)[:sz_coef])
            except linalg.LinAlgError:
                break
            c1 = np.reshape(coef[:num_coef], shape_c, order='F')
            c2 = np.reshape(coef[num_coef:], shape_c, order='F')

            # update the right-dual matrix R and T based on the new coefficients
            R_loop = func_build_R(c1, c2, shape_b)
            # use QR decomposition to extract effective lines of equations
            Q_H = linalg.qr(R_loop, mode='economic',
                            pivoting=False)[0][:, :shape_Tb0_effective].conj().T
            R_loop = np.dot(Q_H, R_loop)
            Tbeta_loop = np.dot(Q_H, Tbeta0_Qtilde)

            # reconstruct b
            if inner == 0:
                mtx_brecon = np.vstack((
                    np.hstack((GtG, R_loop.conj().T)),
                    np.hstack((R_loop, np.zeros((shape_Tb0_effective, shape_Tb0_effective))))
                ))
            else:
                mtx_brecon[:sz_R1, sz_R1:] = R_loop.conj().T
                mtx_brecon[sz_R1:, :sz_R1] = R_loop

            try:
                b_recon = linalg.solve(mtx_brecon, rhs_bl)[:sz_R1]
            except linalg.LinAlgError:
                break

            # compute fitting error
            error_loop = linalg.norm(measurement - np.dot(G, b_recon))

            if 0 <= error_loop < min_error:
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
                    b_opt = b_recon
                    c1_opt = c1
                    c2_opt = c2

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

        if c1_opt is None or c2_opt is None:
            max_ini += 1

    print('fitting SNR {:.2f}'.format(20 * np.log10(linalg.norm(measurement) / min_error)))
    return c1_opt, c2_opt, min_error, b_opt, ini


def planar_extract_innovation(a, num_dirac, c1, c2, x_samp_loc, y_samp_loc,
                              bandwidth, taus, strategy='both', around_zero=False):
    """
    Reconstruct Dirac parameters from the annihilating filter coefficients c1 and c2
    :param a: the given measurements
    :param num_dirac: number of Dirac deltas to extract
    :param c1: the first annihilating filter
    :param c2: the second annihilating filter
    :param x_samp_loc: sampling locations (x-axis)
    :param y_samp_loc: sampling locations (y-axis)
    :param bandwidth: a tuple of size 2 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 2 for the periods of the Dirac stream along x and y axis
    :param strategy: either 'aggregate', which reconstructs amplitudes for
            all xk_recon and yk_recon and select the ones that have the largest amplitudes;
            Or 'one_by_one', which computes the fitting error by leaving out one of the
            reconstructed xk_recon and yk_recon each time. The ones that contributes least
            are eliminated.
    :param around_zero: whether to center the interval around zero or not.
    :return:
    """
    tau_x, tau_y = taus
    # root finding
    z1, z2 = find_roots_2d(c1, c2)
    z1 = z1.squeeze()
    z2 = z2.squeeze()
    # the roots should be on a unit circle (for the Dirac cases)
    valid_idx = ~np.bitwise_or(np.isclose(z1, 0), np.isclose(z2, 0))
    if z1.size > 1 and z2.size > 1:
        z1 = z1[valid_idx]
        z2 = z2[valid_idx]
        z1 /= np.abs(z1)
        z2 /= np.abs(z2)

    xk_recon = np.mod(np.angle(z1) * tau_x / (-2 * np.pi), tau_x)
    yk_recon = np.mod(np.angle(z2) * tau_y / (-2 * np.pi), tau_y)

    xk_recon = xk_recon.flatten()
    yk_recon = yk_recon.flatten()

    remove_nan = np.bitwise_or(np.isnan(xk_recon), np.isnan(yk_recon))
    xk_recon = xk_recon[~remove_nan]
    yk_recon = yk_recon[~remove_nan]

    if around_zero:
        # put Dirac locations within one period center around zero: -tau/2 to tau/2
        xk_recon[xk_recon > 0.5 * tau_x] -= tau_x
        yk_recon[yk_recon > 0.5 * tau_y] -= tau_y

    # select reliable Dirac reconstructions
    total_num_dirac = xk_recon.size

    if total_num_dirac <= num_dirac:
        xk_reliable, yk_reliable = xk_recon, yk_recon
        amp_mtx = planar_amp_mtx(xk_reliable, yk_reliable,
                                 x_samp_loc, y_samp_loc,
                                 bandwidth, taus)
        if total_num_dirac > 0:
            amp_reliable = linalg.lstsq(amp_mtx, a)[0]
        else:
            amp_reliable = np.array([])
    else:
        xk_reliable, yk_reliable, amp_reliable = \
            planar_select_reliable_recon(a, xk_recon, yk_recon, num_dirac,
                                         x_samp_loc, y_samp_loc, bandwidth, taus,
                                         strategy=strategy)

    return xk_reliable, yk_reliable, amp_reliable


def planar_select_reliable_recon(a, xk_recon, yk_recon, num_dirac_sel,
                                 x_samp_loc, y_samp_loc, bandwidth, taus,
                                 strategy='both'):
    """
    Select reliable reconstructions
    :param a: the given measurements
    :param xk_recon: reconstructed Dirac locations (x-axis)
    :param yk_recon: reconstructed Dirac locations (y-axis)
    :param num_dirac_sel: number of Dirac deltas to be extracted (as reliable reconstruction)
    :param x_samp_loc: sampling locations (x-axis)
    :param y_samp_loc: sampling locations (y-axis)
    :param bandwidth: a tuple of size 2 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 2 for the periods of the Dirac stream along x and y axis
    :param strategy: either 'aggregate', which reconstructs amplitudes for
            all xk_recon and yk_recon and select the ones that have the largest amplitudes;
            Or 'one_by_one', which computes the fitting error by leaving out one of the
            reconstructed xk_recon and yk_recon each time. The ones that contributes least
            are eliminated.
    :return:
    """
    a = a.flatten('F')
    # build the linear mapping from the Dirac amplitudes to the measurements
    # for all xk_recon and yk_recon
    amp_mtx_all = planar_amp_mtx(xk_recon, yk_recon,
                                 x_samp_loc, y_samp_loc,
                                 bandwidth, taus)
    if strategy == 'aggregate':
        # find the amplitudes for all Dirac deltas
        amp_recon_all = compute_fitting_error(a, amp_mtx_all)[1]
        amp_sort_idx = np.argsort(np.abs(amp_recon_all))[-num_dirac_sel:]
        # select the reliable reconstruction based on the amplitudes
        xk_reliable = xk_recon[amp_sort_idx]
        yk_reliable = yk_recon[amp_sort_idx]

        amp_mtx_reliable = planar_amp_mtx(xk_reliable, yk_reliable,
                                          x_samp_loc, y_samp_loc,
                                          bandwidth, taus)
        amp_reliable = linalg.lstsq(amp_mtx_reliable, a)[0]
    elif strategy == 'one_by_one':
        if xk_recon.size > 1:
            # number of Dirac to be removed
            num_removal = xk_recon.size - num_dirac_sel
            mask_all = (
                    np.ones((xk_recon.size, xk_recon.size), dtype=int) -
                    np.eye(xk_recon.size, dtype=int)
            ).astype(bool)

            # compute leave-one-out error
            leave_one_out_error = [
                compute_fitting_error(a, amp_mtx_all[:, mask_all[removal_ind, :]])[0]
                for removal_ind in range(mask_all.shape[0])]
            idx_opt = np.argsort(np.asarray(leave_one_out_error))[num_removal:]
        else:
            idx_opt = 0

        xk_reliable = xk_recon[idx_opt]
        yk_reliable = yk_recon[idx_opt]

        amp_mtx_reliable = planar_amp_mtx(xk_reliable, yk_reliable,
                                          x_samp_loc, y_samp_loc,
                                          bandwidth, taus)
        amp_reliable = linalg.lstsq(amp_mtx_reliable, a)[0]
    elif strategy == 'both':
        '''aggregate'''
        # find the amplitudes for all Dirac deltas
        amp_recon_all = compute_fitting_error(a, amp_mtx_all)[1]
        amp_sort_idx = np.argsort(np.abs(amp_recon_all))[-num_dirac_sel:]
        # select the reliable reconstruction based on the amplitudes
        xk_reliable_agg = xk_recon[amp_sort_idx]
        yk_reliable_agg = yk_recon[amp_sort_idx]

        amp_mtx_reliable_agg = planar_amp_mtx(xk_reliable_agg, yk_reliable_agg,
                                              x_samp_loc, y_samp_loc,
                                              bandwidth, taus)
        amp_reliable_agg, fitting_error_agg = linalg.lstsq(amp_mtx_reliable_agg, a)[:2]

        '''one by one strategy'''
        if xk_recon.size > 1:
            # number of Dirac to be removed
            num_removal = xk_recon.size - num_dirac_sel
            mask_all = (
                    np.ones((xk_recon.size, xk_recon.size), dtype=int) -
                    np.eye(xk_recon.size, dtype=int)
            ).astype(bool)

            # compute leave-one-out error
            leave_one_out_error = [
                compute_fitting_error(a, amp_mtx_all[:, mask_all[removal_ind, :]])[0]
                for removal_ind in range(mask_all.shape[0])]
            idx_opt = np.argsort(np.asarray(leave_one_out_error))[num_removal:]
        else:
            idx_opt = 0

        xk_reliable_1by1 = xk_recon[idx_opt]
        yk_reliable_1by1 = yk_recon[idx_opt]

        amp_mtx_reliable_1by1 = planar_amp_mtx(xk_reliable_1by1, yk_reliable_1by1,
                                               x_samp_loc, y_samp_loc,
                                               bandwidth, taus)
        amp_reliable_1by1, fitting_error_1by1 = linalg.lstsq(amp_mtx_reliable_1by1, a)[:2]

        if fitting_error_agg < fitting_error_1by1:
            xk_reliable = xk_reliable_agg
            yk_reliable = yk_reliable_agg
            amp_reliable = amp_reliable_agg
        else:
            xk_reliable = xk_reliable_1by1
            yk_reliable = yk_reliable_1by1
            amp_reliable = amp_reliable_1by1
    else:
        RuntimeError('Unknown strategy: {}'.format(strategy))

    return xk_reliable, yk_reliable, amp_reliable


def compute_fitting_error(a, amp_mtx, non_negative_weights=False):
    """
    compute the fitting error for a given set of Dirac locations.
    :param a: the given measurements
    :param amp_mtx: the amplitude matrix
    :return:
    """
    if non_negative_weights:
        # use non-negative least square
        amplitude, error = nnls(amp_mtx, a)
    else:
        # use least square
        amplitude = linalg.lstsq(amp_mtx, a)[0]
        error = linalg.norm(np.dot(amp_mtx, amplitude) - a)
    return error, amplitude
