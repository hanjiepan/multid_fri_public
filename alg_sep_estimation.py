"""
algorithm used to estimate the Dirac prameters by applying the annihilation constraint
separately along horizontal and vertical direction.
"""
from __future__ import division
import numpy as np
from scipy import linalg
from build_linear_mapping_sep import T_mtx_sep, R_mtx_sep
from alg_joint_estimation_2d import planar_select_reliable_recon
from build_linear_mapping_joint2d import planar_amp_mtx


def dirac_recon_sep_interface(a, num_dirac, samp_loc, bandwidth, taus, noise_level=0,
                              max_ini=100, stop_cri='max_iter', max_inner_iter=20,
                              strategy='both'):
    """
    INTERFACE for the 2D Dirac reconstruction. Here we solve two the annihilation problems
    for the horizontal and vertical directions separately.
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
    :return:
    """
    Bx, By = bandwidth
    taux, tauy = taus
    x_samp_loc, y_samp_loc = samp_loc[:, 0], samp_loc[:, 1]
    shape_b = (int(Bx * taux), int(By * tauy))
    assert shape_b[0] // 2 * 2 + 1 == shape_b[0]
    assert shape_b[1] // 2 * 2 + 1 == shape_b[1]

    freq_limit_x = shape_b[0] // 2
    freq_limit_y = shape_b[1] // 2

    # build the linear mapping that links the uniformly sampled sinusoids
    # to the ideally lowpass filtered samples
    m_grid_samp, n_grid_samp = np.meshgrid(np.arange(-freq_limit_x, freq_limit_x + 1),
                                           np.arange(-freq_limit_y, freq_limit_y + 1))

    # reshape to use broadcasting
    m_grid_samp_col = np.reshape(m_grid_samp, (1, -1), order='F')
    n_grid_samp_col = np.reshape(n_grid_samp, (1, -1), order='F')
    m_grid_samp_row = np.reshape(m_grid_samp, (1, -1), order='C')
    n_grid_samp_row = np.reshape(n_grid_samp, (1, -1), order='C')

    x_samp_loc = np.reshape(x_samp_loc, (-1, 1), order='F')
    y_samp_loc = np.reshape(y_samp_loc, (-1, 1), order='F')

    mtx_fri2samp_col = np.exp(1j * 2 * np.pi / taux * m_grid_samp_col * x_samp_loc +
                              1j * 2 * np.pi / tauy * n_grid_samp_col * y_samp_loc) / (Bx * By)

    mtx_fri2samp_row = np.exp(1j * 2 * np.pi / taux * m_grid_samp_row * x_samp_loc +
                              1j * 2 * np.pi / tauy * n_grid_samp_row * y_samp_loc) / (Bx * By)

    # apply FRI estimation for columns and rows
    c_col = dirac_recon_sep_alg(mtx_fri2samp_col, a, num_dirac,
                                vec_len=shape_b[0], num_vec=shape_b[1],
                                noise_level=noise_level,
                                max_ini=max_ini, stop_cri=stop_cri,
                                max_inner_iter=max_inner_iter)[0]
    c_row = dirac_recon_sep_alg(mtx_fri2samp_row, a, num_dirac,
                                vec_len=shape_b[1], num_vec=shape_b[0],
                                noise_level=noise_level,
                                max_ini=max_ini, stop_cri=stop_cri,
                                max_inner_iter=max_inner_iter)[0]

    # retrieve Dirac parameters
    xk_recon, yk_recon, amp_recon = \
        extract_innovation(a, num_dirac, c_col, c_row, x_samp_loc, y_samp_loc,
                           bandwidth, taus, strategy=strategy)

    return xk_recon, yk_recon, amp_recon


def dirac_recon_sep_alg(G, measurement, num_dirac, vec_len, num_vec,
                        flatten_order='F', noise_level=0,
                        max_ini=100, stop_cri='mse', max_inner_iter=20):
    """
    ALGORITHM that reconstructs the 2D Dirac deltas by enforcing
    annihilation constraint separately (i.e., solving two 1D annihilation problems):
        min     |a - Gb|^2
        s.t.    c_1 * b = 0

    and
        min     |a - Gb|^2
        s.t.    c_2 * b = 0

    :param G: the linear mapping that links the unknown uniformly sampled
            sinusoids to the given measurements
    :param measurement: the given measurements of the 2D Dirac deltas
    :param num_dirac: number of Dirac deltas
    :param vec_len: length of each vector that should satisfy the annihilation constraint
    :param num_vec: total number of such vectors that should be annihilated by the same filter,
            whose coefficients are given by 'c'
    :param flatten_order: flatten order to be used. This is related to how G is build.
            If the dimension 0 of G is 'C' ordered, then flattern_order = 'C';
            otherwise, flattern_order = 'F'.
    :param noise_level: noise level present in the given measurements
    :param max_ini: maximum number of random initializations
    :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
    :param max_inner_iter: maximum number of inner iterations for each random initializations
    :return:
    """
    compute_mse = (stop_cri == 'mse')
    measurement = measurement.flatten(flatten_order)

    # sizes of various matrices / vectors
    sz_coef = num_dirac + 1
    # length of b is at least the filter size
    assert vec_len >= sz_coef
    # total number of elements in the uniformly sampled sinusoids b
    num_b = num_vec * vec_len
    # output size 0 of the annihilation equations
    anni_out_sz0 = num_vec * (vec_len - num_dirac)

    # pre-compute a few things
    GtG = np.dot(G.conj().T, G)
    Gt_a = np.dot(G.conj().T, measurement)
    beta = linalg.lstsq(G, measurement)[0]

    Tbeta = T_mtx_sep(beta, vec_len, num_vec, size_c=num_dirac + 1)

    # initializations
    min_error = np.inf
    rhs = np.zeros(sz_coef + anni_out_sz0 + num_b + 1, dtype=complex)
    rhs[-1] = 1
    rhs_bl = np.concatenate((Gt_a, np.zeros(anni_out_sz0)))

    # iterations over different random initializations of the annihilating filter coefficients
    for ini in range(max_ini):
        coef = np.random.randn(sz_coef) + 1j * np.random.randn(sz_coef)
        coef0 = coef[:, np.newaxis]
        R_loop = R_mtx_sep(coef, vec_len, num_vec)

        # first row in mtx_loop
        mtx_first_row = np.hstack((
            np.zeros((sz_coef, sz_coef)),
            Tbeta.conj().T,
            np.zeros((sz_coef, num_b)),
            coef0
        ))

        # last row in mtx_loop
        mtx_last_row = np.hstack((
            coef0.conj().T,
            np.zeros((1, anni_out_sz0 + num_b + 1))
        ))

        for inner in range(max_inner_iter):
            if inner == 0:
                mtx_loop = np.vstack((
                    mtx_first_row,
                    np.hstack((
                        Tbeta,
                        np.zeros((anni_out_sz0, anni_out_sz0)),
                        -R_loop,
                        np.zeros((anni_out_sz0, 1))
                    )),
                    np.hstack((
                        np.zeros((num_b, sz_coef)),
                        -R_loop.conj().T,
                        GtG,
                        np.zeros((num_b, 1))
                    )),
                    mtx_last_row
                ))
            else:
                mtx_loop[sz_coef:sz_coef + anni_out_sz0,
                sz_coef + anni_out_sz0:sz_coef + anni_out_sz0 + num_b] = -R_loop
                mtx_loop[sz_coef + anni_out_sz0:sz_coef + anni_out_sz0 + num_b,
                sz_coef:sz_coef + anni_out_sz0] = R_loop.conj().T

            # solve annihilating filter coefficients
            coef = linalg.solve(mtx_loop, rhs)[:sz_coef]

            # update the right dual matrix R based on the new coefficients
            R_loop = R_mtx_sep(coef, vec_len, num_vec)

            # reconstruct b
            if inner == 0:
                mtx_brecon = np.vstack((
                    np.hstack((GtG, R_loop.conj().T)),
                    np.hstack((R_loop, np.zeros((anni_out_sz0, anni_out_sz0))))
                ))
            else:
                mtx_brecon[:num_b, num_b:] = R_loop.conj().T
                mtx_brecon[num_b:, :num_b] = R_loop

            b_recon = linalg.solve(mtx_brecon, rhs_bl)[:num_b]

            # compute fitting error
            error_loop = linalg.norm(measurement - np.dot(G, b_recon))
            if error_loop < min_error:
                min_error = error_loop
                b_opt = b_recon
                c_opt = coef

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

    # TODO: remove after debugging
    print('fitting SNR {:.2f}'.format(20 * np.log10(linalg.norm(measurement) / min_error)))
    return c_opt, min_error, b_opt, ini


def extract_innovation(a, num_dirac, coef_col, coef_row, x_samp_loc, y_samp_loc,
                       bandwidth, taus, strategy='both'):
    """
    Reconstruct Dirac parameters from the annihilating filter coefficients
    coef_col and coef_row
    :param a: the given measurements
    :param num_dirac: number of Dirac deltas to extract
    :param coef_col: the annihilating filter that annihilates COLUMNS of b
    :param coef_row: the annihilating filter that annihilates ROWS of b
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
    tau_x, tau_y = taus
    # find roots
    z1 = np.roots(coef_row.squeeze())
    z2 = np.roots(coef_col.squeeze())
    # the roots should be on a unit circle (for the Dirac cases)
    z1 /= np.abs(z1)
    z2 /= np.abs(z2)

    xk_recon_1d = np.mod(np.angle(z1) * tau_x / (-2 * np.pi), tau_x)
    yk_recon_1d = np.mod(np.angle(z2) * tau_y / (-2 * np.pi), tau_y)

    xk_recon, yk_recon = np.meshgrid(xk_recon_1d, yk_recon_1d)
    xk_recon = xk_recon.flatten()
    yk_recon = yk_recon.flatten()

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
