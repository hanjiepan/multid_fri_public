import numpy as np
from scipy import linalg
from build_linear_mapping_joint3d import cubical_sel_coef_subset, \
    R_mtx_joint3d, T_mtx_joint3d, cubical_amp_mtx, compute_effective_num_eq_3d, \
    build_G_mtx_idealLP
from alg_joint_estimation_2d import compute_fitting_error
from poly_common_roots_3d import find_roots_3d
from alg_joint_estimation_3d_efficient import dirac_recon_joint_alg_fast


def dirac_recon_joint_ideal_interface(a, num_dirac, samp_loc, bandwidth, taus, noise_level=0,
                                      max_ini=100, stop_cri='max_iter', max_inner_iter=20,
                                      strategy='both', max_num_same_x=1, max_num_same_y=1,
                                      max_num_same_z=1, use_fast_alg=False):
    """
    INTERFACE that reconstructs 3D Dirac deltas jointly
    :param a: the given measurements of the 3D Dirac deltas
    :param num_dirac: number of Dirac deltas
    :param samp_loc: a 3-COLUMN matrix that contains sampling locations (x, y, z)
    :param bandwidth: a tuple of size 3 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 3 for the periods of the Dirac stream along x, y and z axis
    :param noise_level: noise level present in the given measurements
    :param max_ini: maximum number of random initializations
    :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
    :param max_inner_iter: maximum number of inner iterations for each random initializations
    :param strategy: either 'aggregate', which reconstructs amplitudes for
            all xk_recon, yk_recon and zk_recon and select the ones that have the largest amplitudes;
            Or 'one_by_one', which computes the fitting error by leaving out one of the
            reconstructed xk_recon, yk_recon and zk_recon each time. The ones that contributes least
            are eliminated.
    :param max_num_same_x: maximum number of Dirac deltas that have the same horizontal locations.
            This will impose the minimum dimension of the annihilating filter used.
    :param max_num_same_y: maximum number of Dirac deltas that have the same vertical locations
            This will impose the minimum dimension of the annihilating filter used.
    :param max_num_same_z: maximum number of Dirac deltas that have the same depth locations
            This will impose the minimum dimension of the annihilating filter used.
    :param use_fast_alg: whether to use a faster implementation but may not be as stable as the
            exact implmentation.
    :return:
    """
    Bx, By, Bz = bandwidth
    taux, tauy, tauz = taus
    shape_b = (int(By * tauy), int(Bx * taux), int(Bz * tauz))
    assert shape_b[0] // 2 * 2 + 1 == shape_b[0]
    assert shape_b[1] // 2 * 2 + 1 == shape_b[1]
    assert shape_b[2] // 2 * 2 + 1 == shape_b[2]

    # build the linear mapping that links the uniformly sampled sinusoids
    # to the ideally lowpass filtered samples
    mtx_fri2samp = build_G_mtx_idealLP(samp_loc, bandwidth, taus, shape_b)

    # apply joint FRI estimation
    if use_fast_alg:
        # a faster implementation
        c1_opt, c2_opt, c3_opt = \
            dirac_recon_joint_alg_fast(mtx_fri2samp, a, num_dirac, shape_b,
                                       noise_level=noise_level, max_ini=max_ini,
                                       stop_cri=stop_cri, max_inner_iter=max_inner_iter,
                                       max_num_same_x=max_num_same_x,
                                       max_num_same_y=max_num_same_y,
                                       max_num_same_z=max_num_same_z)[:3]
    else:
        c1_opt, c2_opt, c3_opt = \
            dirac_recon_joint_alg(mtx_fri2samp, a, num_dirac, shape_b,
                                  noise_level=noise_level, max_ini=max_ini,
                                  stop_cri=stop_cri, max_inner_iter=max_inner_iter,
                                  max_num_same_x=max_num_same_x,
                                  max_num_same_y=max_num_same_y,
                                  max_num_same_z=max_num_same_z)[:3]

    # retrieve Dirac parameters
    # function handel to build the mapping from Dirac amplitudes to given samples
    def func_amp2samp(dirac_loc, samp_loc):
        return cubical_amp_mtx(dirac_loc, samp_loc, bandwidth, taus)

    dirac_loc_recon, amp_recon = \
        cubical_extract_innovation(a, num_dirac, c1_opt, c2_opt, c3_opt,
                                   samp_loc, func_amp2samp, taus, strategy=strategy)

    return dirac_loc_recon, amp_recon


# TODO: update the implementation with the new formulation (reduce number of unknowns)
def dirac_recon_joint_alg(G, measurement, num_dirac, shape_b, noise_level=0,
                          max_ini=100, stop_cri='mse', max_inner_iter=20,
                          max_num_same_x=1, max_num_same_y=1, max_num_same_z=1):
    """
    ALGORITHM that reconstructs 2D Dirac deltas jointly
        min     |a - Gb|^2
        s.t.    c_1 * b = 0
                c_2 * b = 0
                c_3 * b = 0

    This is the exact form that we have in the paper without any alternations for
    performance considerations, e.g., reusing intermediate results, etc.

    :param G: the linear mapping that links the unknown uniformly sampled
            sinusoids to the given measurements
    :param measurement: the given measurements of the 3D Dirac deltas
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
    compute_mse = (stop_cri == 'mse')
    measurement = measurement.flatten('F')
    num_non_zero = num_dirac + 3
    # choose the shapes of the 3D annihilating filters
    shape_c1_0 = shape_c1_1 = shape_c1_2 = int(np.ceil(num_non_zero ** (1. / 3)))

    if shape_c1_2 > shape_b[2]:
        shape_c1_2 = shape_b[2]
        shape_c1_0 = shape_c1_1 = int(np.ceil(np.sqrt(num_non_zero / shape_c1_2)))
        if shape_c1_1 > shape_b[1]:
            shape_c1_1 = shape_b[1]
            shape_c1_0 = int(np.ceil(num_non_zero / (shape_c1_1 * shape_c1_2)))
        elif shape_c1_0 > shape_b[0]:
            shape_c1_0 = shape_b[0]
            shape_c1_1 = int(np.ceil(num_non_zero / (shape_c1_0 * shape_c1_2)))
    elif shape_c1_1 > shape_b[1]:
        shape_c1_1 = shape_b[1]
        shape_c1_0 = shape_c1_2 = int(np.ceil(np.sqrt(num_non_zero / shape_c1_1)))
        if shape_c1_0 > shape_b[0]:
            shape_c1_0 = shape_b[0]
            shape_c1_2 = int(np.ceil(num_non_zero / (shape_c1_0 * shape_c1_1)))
        elif shape_c1_2 > shape_b[2]:
            shape_c1_2 = shape_b[2]
            shape_c1_0 = int(np.ceil(num_non_zero / (shape_c1_1 * shape_c1_2)))
    elif shape_c1_0 > shape_b[0]:
        shape_c1_0 = shape_b[0]
        shape_c1_1 = shape_c1_2 = int(np.ceil(np.sqrt(num_non_zero / shape_c1_0)))
        if shape_c1_1 > shape_b[1]:
            shape_c1_1 = shape_b[1]
            shape_c1_2 = int(np.ceil(num_non_zero / (shape_c1_0 * shape_c1_1)))
        elif shape_c1_2 > shape_b[2]:
            shape_c1_2 = shape_b[2]
            shape_c1_1 = int(np.ceil(num_non_zero / (shape_c1_0 * shape_c1_2)))

        # sanity check
    assert shape_c1_0 * shape_c1_1 * shape_c1_2 >= num_non_zero
    assert shape_c1_0 <= shape_b[0] and shape_c1_1 <= shape_b[1] and shape_c1_2 <= shape_b[2]

    shape_c3_0, shape_c3_1, shape_c3_2 = \
        shape_c2_0, shape_c2_1, shape_c2_2 = shape_c1_0, shape_c1_1, shape_c1_2

    shape_c1 = (shape_c1_0, shape_c1_1, shape_c1_2)
    shape_c2 = (shape_c2_0, shape_c2_1, shape_c2_2)
    shape_c3 = (shape_c3_0, shape_c3_1, shape_c3_2)

    # check if there will be sufficient number of effective number of equations
    update_shape = True
    dict_keys = ['shape_c1', 'shape_c2', 'shape_c3']
    shapes = {
        'shape_c1': list(shape_c1),
        'shape_c2': list(shape_c2),
        'shape_c3': list(shape_c3)
    }
    shapes_update_eq = [1, 0, 2]
    shapes_update_neq = [2, 1, 0]
    exp_count = 0
    while update_shape:
        if compute_effective_num_eq_3d(shapes['shape_c1'],
                                       shapes['shape_c2'],
                                       shapes['shape_c3']) < num_dirac:
            shape_loop = shapes[dict_keys[exp_count]]
            if shape_loop[0] == shape_loop[1] == shape_loop[2]:
                shapes[dict_keys[exp_count]][shapes_update_eq[exp_count]] += 1
            else:
                shapes[dict_keys[exp_count]][shapes_update_neq[exp_count]] += 1

            exp_count += 1
            exp_count = np.mod(exp_count, 3)
            update_shape = True
        else:
            update_shape = False

    shape_c1 = tuple(shapes['shape_c1'])
    shape_c2 = tuple(shapes['shape_c2'])
    shape_c3 = tuple(shapes['shape_c3'])

    # in case of common roots, the filter has to satisfy a certain minimum dimension
    # TODO: add this constraints

    # total number of coefficients in c1 and c2
    num_coef1 = np.prod(shape_c1)
    num_coef2 = np.prod(shape_c2)
    num_coef3 = np.prod(shape_c3)

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

    # sizes of various matrices / vectors
    sz_coef = num_coef1 + num_coef2 + num_coef3
    sz_S0 = sz_coef - 3 * num_non_zero
    sz_R1 = np.prod(shape_b)

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
    beta = linalg.lstsq(G, measurement)[0]
    Tbeta0 = T_mtx_joint3d(np.reshape(beta, shape_b, order='F'),
                           shape_c1, shape_c2, shape_c3)

    # initializations
    min_error = np.inf
    rhs = np.concatenate((np.zeros(sz_coef + shape_Tb0_effective + sz_R1 + sz_S0, dtype=complex),
                          np.ones(3, dtype=complex)))
    rhs_bl = np.concatenate((Gt_a, np.zeros(shape_Tb0_effective, dtype=Gt_a.dtype)))
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
        c0 = linalg.block_diag(c1.flatten('F')[:, np.newaxis],
                               c2.flatten('F')[:, np.newaxis],
                               c3.flatten('F')[:, np.newaxis])

        # build a selection matrix that chooses a subset of c1 and c2 to ZERO OUT
        S = cubical_sel_coef_subset(shape_c1, shape_c2, shape_c3,
                                    num_non_zero=num_non_zero,
                                    max_num_same_x=max_num_same_x,
                                    max_num_same_y=max_num_same_y,
                                    max_num_same_z=max_num_same_z)
        S_H = S.T  # S is real valued
        mtx_S_row = np.hstack((S, np.zeros((sz_S0, shape_Tb0_effective + sz_R1 + sz_S0 + 3))))

        # last row in mtx_loop
        mtx_last_row = \
            np.hstack((c0.conj().T,
                       np.zeros((3, shape_Tb0_effective + sz_R1 + sz_S0 + 3))))

        R_loop = R_mtx_joint3d(c1, c2, c3, shape_b)
        # use QR decomposition to extract effective lines of equations
        Q_H = linalg.qr(R_loop, mode='economic',
                        pivoting=False)[0][:, :shape_Tb0_effective].conj().T
        R_loop = np.dot(Q_H, R_loop)
        Tbeta_loop = np.dot(Q_H, Tbeta0)

        # inner loop for each random initialization
        for inner in range(max_inner_iter):
            if inner == 0:
                mtx_loop = np.vstack((
                    np.hstack((np.zeros((sz_coef, sz_coef)),
                               Tbeta_loop.conj().T,
                               np.zeros((sz_coef, sz_R1)), S_H, c0)),
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
                coef = linalg.solve(mtx_loop, rhs)[:sz_coef]
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
            Tbeta_loop = np.dot(Q_H, Tbeta0)

            # reconstruct b
            if inner == 0:
                mtx_brecon = np.vstack((
                    np.hstack((GtG, R_loop.conj().T)),
                    np.hstack((R_loop, np.zeros((shape_Tb0_effective, shape_Tb0_effective))))
                ))
            else:
                mtx_brecon[:sz_R1, sz_R1:] = R_loop.conj().T
                mtx_brecon[sz_R1:, :sz_R1] = R_loop

            b_recon = linalg.solve(mtx_brecon, rhs_bl)[:sz_R1]

            # compute fitting error
            error_loop = linalg.norm(measurement - np.dot(G, b_recon))

            if 0 <= error_loop < min_error:
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
                    b_opt = b_recon
                    c1_opt = c1
                    c2_opt = c2
                    c3_opt = c3

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

        if c1_opt is None or c2_opt is None or c3_opt is None:
            max_ini += 1

    print('fitting SNR {:.2f}'.format(20 * np.log10(linalg.norm(measurement) / min_error)))
    return c1_opt, c2_opt, c3_opt, min_error, b_opt, ini


def cubical_extract_innovation(measurement, num_dirac, c1, c2, c3, samp_loc,
                               func_amp2samp, taus, strategy='both',
                               around_zero=False):
    """
    Reconstruct Dirac parameters from the annihilating filter coefficients c1, c2, and c3
    :param measurement: the given measurements
    :param num_dirac: number of Dirac deltas to extract
    :param c1: the first annihilating filter (horizontal direction dominating)
    :param c2: the second annihilating filter (vertical direction dominating)
    :param c3: the third annihilating filter (depth direction dominating)
    :param samp_loc: a 3-COLUMN matrix that contains sampling locations (x, y, z)
    :param func_amp2samp: a function handel that build the linear mapping from the Dirac
            amplitudes to the given samples.
    :param taus: a tuple of size 3 for the periods of the Dirac stream along x, y and z axis
    :param strategy: either 'aggregate', which reconstructs amplitudes for
            all xk_recon and yk_recon and select the ones that have the largest amplitudes;
            Or 'one_by_one', which computes the fitting error by leaving out one of the
            reconstructed xk_recon and yk_recon each time. The ones that contributes least
            are eliminated.
    :param around_zero: whether to center the interval around zero or not.
    :return:
    """
    tau_x, tau_y, tau_z = taus
    # root finding
    z1, z2, z3 = find_roots_3d(c1, c2, c3)
    z1 = z1.squeeze()
    z2 = z2.squeeze()
    z3 = z3.squeeze()
    # the roots should be on a unit circle (for the Dirac cases)
    valid_idx = ~np.bitwise_or(np.isclose(z1, 0), np.isclose(z2, 0), np.isclose(z3, 0))
    if z1.size > 1 and z2.size > 1 and z3.size > 1:
        z1, z2, z3 = z1[valid_idx], z2[valid_idx], z3[valid_idx]
        z1 /= np.abs(z1)
        z2 /= np.abs(z2)
        z3 /= np.abs(z3)

    xk_recon = np.mod(np.angle(z1) * tau_x / (-2 * np.pi), tau_x)
    yk_recon = np.mod(np.angle(z2) * tau_y / (-2 * np.pi), tau_y)
    zk_recon = np.mod(np.angle(z3) * tau_z / (-2 * np.pi), tau_z)
    xk_recon = xk_recon.flatten()
    yk_recon = yk_recon.flatten()
    zk_recon = zk_recon.flatten()

    if around_zero:
        # put Dirac locations within one period center around zero: -tau/2 to tau/2
        xk_recon[xk_recon > 0.5 * tau_x] -= tau_x
        yk_recon[yk_recon > 0.5 * tau_y] -= tau_y
        zk_recon[zk_recon > 0.5 * tau_z] -= tau_z

    print(xk_recon.size, yk_recon.size, zk_recon.size)
    dirac_loc_recon = np.column_stack((xk_recon, yk_recon, zk_recon))

    # select reliable Dirac reconstructions
    total_num_dirac = xk_recon.size

    if total_num_dirac <= num_dirac:
        dirac_loc_reliable = dirac_loc_recon
        if total_num_dirac > 0:
            amp_mtx = func_amp2samp(dirac_loc_reliable, samp_loc)
            amp_reliable = linalg.lstsq(amp_mtx, measurement)[0]
        else:
            amp_reliable = np.array([])
    else:
        dirac_loc_reliable, amp_reliable = cubical_select_reliable_recon(
            measurement, dirac_loc_recon, num_dirac, samp_loc, func_amp2samp, strategy=strategy)

    return dirac_loc_reliable, amp_reliable


def cubical_select_reliable_recon(measurement, dirac_locs, num_dirac_sel,
                                  samp_locs, func_amp2samp, strategy='both'):
    """
    Select reliable reconstructions
    :param measurement: the given measurements
    :param dirac_locs: a num_dirac by 3 matrix with the columns corresponds to
            the Dirac x, y and z locations. num_dirac is the number of Dirac
    :param num_dirac_sel: number of Dirac deltas to be extracted (as reliable reconstruction)
    :param samp_locs: an N by 3 matrix with the columns corresponds to the 3D
            sampling locations along the x, y and z axis. N is the total number of samples.
    :param func_amp2samp: a function handel that build the linear mapping from the Dirac
            amplitudes to the given samples.
    :param taus: a tuple of size 3 for the periods of the Dirac stream along x, y and z axis.
    :param strategy: either 'aggregate', which reconstructs amplitudes for
            all xk_recon and yk_recon and select the ones that have the largest amplitudes;
            Or 'one_by_one', which computes the fitting error by leaving out one of the
            reconstructed xk_recon and yk_recon each time. The ones that contributes least
            are eliminated.
    :return:
    """
    measurement = measurement.flatten('F')
    # build the linear mapping from the Dirac amplitudes to the measurements
    # for all xk_recon, yk_recon and zk_recon
    amp_mtx_all = func_amp2samp(dirac_locs, samp_locs)
    if strategy == 'aggregate':
        # find the amplitudes for all Dirac deltas
        amp_recon_all = compute_fitting_error(measurement, amp_mtx_all)[1]
        amp_sort_idx = np.argsort(np.abs(amp_recon_all))[-num_dirac_sel:]
        # select the reliable reconstruction based on the amplitudes
        dirac_locs_reliable = dirac_locs[amp_sort_idx, :]

        amp_mtx_reliable = func_amp2samp(dirac_locs_reliable, samp_locs)
        amp_reliable = linalg.lstsq(amp_mtx_reliable, measurement)[0]

    elif strategy == 'one_by_one':
        num_dirac_recon = dirac_locs.shape[0]
        if num_dirac_recon > 1:
            # number of Dirac to be removed
            num_removal = num_dirac_recon - num_dirac_sel
            mask_all = (
                np.ones((num_dirac_recon, num_dirac_recon), dtype=int) -
                np.eye(num_dirac_recon, dtype=int)
            ).astype(bool)

            # compute leave-one-out error
            leave_one_out_error = [
                compute_fitting_error(measurement, amp_mtx_all[:, mask_all[removal_ind, :]])[0]
                for removal_ind in range(mask_all.shape[0])]
            idx_opt = np.argsort(np.asarray(leave_one_out_error))[num_removal:]
        else:
            idx_opt = 0

        dirac_locs_reliable = dirac_locs[idx_opt, :]

        amp_mtx_reliable = \
            func_amp2samp(dirac_locs_reliable, samp_locs)
        amp_reliable = linalg.lstsq(amp_mtx_reliable, measurement)[0]

    else:  # use both to use whichever that gives smaller fitting error
        '''aggregate'''
        # find the amplitudes for all Dirac deltas
        amp_recon_all = compute_fitting_error(measurement, amp_mtx_all)[1]
        amp_sort_idx = np.argsort(np.abs(amp_recon_all))[-num_dirac_sel:]
        # select the reliable reconstruction based on the amplitudes
        dirac_locs_reliable_agg = dirac_locs[amp_sort_idx, :]

        amp_mtx_reliable_agg = \
            func_amp2samp(dirac_locs_reliable_agg, samp_locs)
        amp_reliable_agg, fitting_error_agg = \
            linalg.lstsq(amp_mtx_reliable_agg, measurement)[:2]

        '''one by one strategy'''
        num_dirac_recon = dirac_locs.shape[0]
        if num_dirac_recon > 1:
            # number of Dirac to be removed
            num_removal = num_dirac_recon - num_dirac_sel
            mask_all = (
                np.ones((num_dirac_recon, num_dirac_recon), dtype=int) -
                np.eye(num_dirac_recon, dtype=int)
            ).astype(bool)

            # compute leave-one-out error
            leave_one_out_error = [
                compute_fitting_error(measurement, amp_mtx_all[:, mask_all[removal_ind, :]])[0]
                for removal_ind in range(mask_all.shape[0])]
            idx_opt = np.argsort(np.asarray(leave_one_out_error))[num_removal:]
        else:
            idx_opt = 0

        dirac_locs_reliable_1by1 = dirac_locs[idx_opt, :]

        amp_mtx_reliable_1by1 = \
            func_amp2samp(dirac_locs_reliable_1by1, samp_locs)
        amp_reliable_1by1, fitting_error_1by1 = \
            linalg.lstsq(amp_mtx_reliable_1by1, measurement)[:2]

        if fitting_error_agg < fitting_error_1by1:
            dirac_locs_reliable = dirac_locs_reliable_agg
            amp_reliable = amp_reliable_agg
        else:
            dirac_locs_reliable = dirac_locs_reliable_1by1
            amp_reliable = amp_reliable_1by1

    return dirac_locs_reliable, amp_reliable
