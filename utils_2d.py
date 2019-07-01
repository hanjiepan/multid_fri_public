from __future__ import division

import os

import numexpr as ne
import numpy as np
from scipy import linalg
from skimage.util import view_as_blocks


def distance_sorted(p_ref, p_cmp, interval):
    """
    compute the periodic distance of two SORTED lists of points
    :return:
    """
    dim = np.array(interval).size
    p_ref = np.reshape(p_ref, (-1, dim), order='F')
    p_cmp = np.reshape(p_cmp, (-1, dim), order='F')
    normal_factor = np.pi * 2 / np.array(interval)
    dist_all = np.array([
        linalg.norm(np.arccos(np.cos((p_ref_loop - p_cmp_loop) * normal_factor)) / normal_factor)
        for p_ref_loop, p_cmp_loop in zip(p_ref, p_cmp)
    ])
    return np.mean(dist_all)


def nd_distance(p_ref, p_cmp, interval=None):
    """
    compute the minimum pair-wise distance for a collection of reference points
    and comparing points in n-dimension.
    :param p_ref: reference point locations. The dimension is assumed to be:
            total_num_points x total_num_dimension
    :param p_cmp: point locations to be compared. The dimension is assumed to be:
            total_num_points x total_num_dimension
    :param interval: in case of periodicity, the wrap around distance is used.
            Interval specifies the period along each dimension.
    :return:
    """
    p_ref = p_ref.copy()
    p_cmp = p_cmp.copy()
    # points are in 1D
    if len(p_ref.shape) < 2:
        p_ref = np.reshape(p_ref, (-1, 1))
    if len(p_cmp.shape) < 2:
        p_cmp = np.reshape(p_cmp, (-1, 1))

    N1, dim_vec = p_ref.shape
    assert dim_vec == p_cmp.shape[1]
    N2 = p_cmp.shape[0]
    p_ref = view_as_blocks(p_ref, (1, dim_vec))
    p_cmp = view_as_blocks(p_cmp, (1, dim_vec))
    if interval is None:
        def distance_func(arg1, arg2):
            return np.array([linalg.norm(arg1_loop - arg2_loop)
                             for arg1_loop, arg2_loop in zip(arg1, arg2)])
    else:
        def distance_func(arg1, arg2):
            normal_factor = np.pi * 2 / np.array(interval)
            return np.array([
                linalg.norm(np.arccos(np.cos(((arg1_loop - arg2_loop) * normal_factor))) / normal_factor)
                for arg1_loop, arg2_loop in zip(arg1, arg2)
            ])

    diffmat = np.zeros((N2, N1), dtype=float)
    for n1 in range(N1):
        for n2 in range(N2):
            diffmat[n2, n1] = distance_func(p_ref[n1], p_cmp[n2])

    min_N1_N2 = min(N1, N2)
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in range(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = np.inf
            diffmat[:, index1] = np.inf

        d = np.mean(distance_func(p_ref[index[:, 0]], p_cmp[index[:, 1]]))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([0, index])
        else:
            index = np.array([index, 0])

    return d, index


def planar_distance(x_ref, y_ref, x_recon, y_recon, interval=None):
    """
    Given two arrays of numbers pt_1 and pt_2, pairs the cells that are the
    closest and provides the pairing matrix index: pt_1[index[0, :]] should be as
    close as possible to pt_2[index[2, :]]. The function outputs the average of the
    absolute value of the differences abs(pt_1[index[0, :]]-pt_2[index[1, :]]).
    :param pt_1: vector 1
    :param pt_2: vector 2
    :return: d: minimum distance between d
             index: the permutation matrix
    """
    if interval is None:
        def distance_func(arg1, arg2):
            return np.abs(arg1 - arg2)
    else:
        def distance_func(arg1, arg2):
            return circular_dist_2d(arg1, arg2, interval)
    pt_1 = x_ref + 1j * y_ref
    pt_2 = x_recon + 1j * y_recon
    pt_1 = np.reshape(pt_1, (1, -1), order='F')
    pt_2 = np.reshape(pt_2, (1, -1), order='F')
    N1 = pt_1.size
    N2 = pt_2.size
    diffmat = distance_func(pt_1, np.reshape(pt_2, (-1, 1), order='F'))
    min_N1_N2 = np.min([N1, N2])
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in range(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = float('inf')
            diffmat[:, index1] = float('inf')
        d = np.mean(distance_func(pt_1[:, index[:, 0]], pt_2[:, index[:, 1]]))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([0, index])
        else:
            index = np.array([index, 0])
    return d, index


def circular_dist_2d(pt1, pt2, interval_range):
    """
    compute the circular distance on the interval
    from 0 to interval_range[0] by 0 to interval_range[1].
    e.g., if interval_range = 1, then
    dist(0.1 +  0.1j, 0.9 + 0.9j) = (1 + 0.1) - 0.9 + (1j + 0.1j - 0.9j) = 0.2 + 0.2j
    :param pt1: the first point: real-part -- x axis; imaginary-part -- y axis
    :param pt2: the second point: real-part -- x axis; imaginary-part -- y axis
    :param interval_range: a tuple of size 2 for the range of the interval (x, y)
    :return:
    """
    tau_x, tau_y = interval_range
    return np.minimum.reduce(
        [np.abs(pt1 - pt2),
         np.abs(pt1 - (pt2 + 1j * tau_y)),
         np.abs(pt1 - (pt2 - 1j * tau_y)),
         np.abs(pt1 + tau_x - pt2),
         np.abs(pt1 + tau_x - (pt2 + 1j * tau_y)),
         np.abs(pt1 + tau_x - (pt2 - 1j * tau_y)),
         np.abs(pt1 - tau_x - pt2),
         np.abs(pt1 - tau_x - (pt2 + 1j * tau_y)),
         np.abs(pt1 - tau_x - (pt2 - 1j * tau_y))])


def periodic_sinc(t, M):
    numerator = ne.evaluate('sin(t)')
    denominator = ne.evaluate('M * sin(t / M)')
    idx = ne.evaluate('abs(denominator) < 1e-12')
    t_idx = t[idx]
    numerator[idx] = ne.evaluate('cos(t_idx)')
    denominator[idx] = ne.evaluate('cos(t_idx / M)')
    return ne.evaluate('numerator / denominator')


def dirichlet_kernel(t, B, tau):
    """
    The same function with periodic_sinc but with a different interface, which takes
    the band-width and period as the inputs.
    :param t: input time values
    :param B: band-width of the periodic sinc
    :param tau: period of
    :return:
    """
    B_tau = B * tau
    pi_B = np.pi * B

    pi_B_t = pi_B * t
    pi_t_over_tau = np.pi * t / tau

    numerator = np.sin(pi_B_t)
    denominator = B_tau * np.sin(pi_t_over_tau)

    idx = np.isclose(np.abs(denominator), 0)
    denominator[idx] = np.cos(pi_t_over_tau[idx])
    numerator[idx] = np.cos(pi_B_t[idx])

    return numerator / denominator


def d_periodic_sinc(t, B, tau):
    """
    DERIVATIVE of periodic sinc:
    d (sin(pi * B * t)/(B * tau * sin(pi * t / tau))) / dt
    :param t: input variable
    :param B: band width of th periodic sinc
    :param tau: period of the periodic sinc
    :return:
    """
    B_tau = B * tau
    pi_B = np.pi * B

    pi_B_t = pi_B * t
    pi_t_over_tau = np.pi * t / tau

    numerator = pi_B * B_tau * np.cos(pi_B_t) * np.sin(pi_t_over_tau) - \
                pi_B * np.sin(pi_B_t) * np.cos(pi_t_over_tau)
    denominator = (B_tau * np.sin(pi_t_over_tau)) ** 2

    idx = np.isclose(np.abs(denominator), 0)
    denominator[idx] = 2 * pi_B * pi_B
    numerator[idx] = 0

    return numerator / denominator


def compute_crb(loc_k, amp_k, samp_locs, bandwidths, taus, noise_covariance):
    """
    Compute the Cramer-Rao bound for 2D Dirac
    :param loc_k: a num_dirac x 2 matrix for the Dirac locations (column 0: x, column 1: y)
    :param amp_k: Dirac amplitudes
    :param samp_locs: an N x 2 matrix for the sampling locations of the low-pass filtered 2D Dirac
    :param bandwidths: a tuple of size 2 for the bandwidth along x and y axis, respectively
    :param taus: periods along x and y axis for the periodic 2D Dirac stream
    :param noise_covariance: either a scalar or a matrix for the covariance of noise
    :return:
    """
    # check if it is the covariance matrix that is given or just a scalar that specifies the
    # diagonal of the covariance matrix
    if np.array([noise_covariance]).size > 1:
        is_matrix = True
    else:
        is_matrix = False

    xk, yk = loc_k[:, 0], loc_k[:, 1]
    x_samp_loc, y_samp_loc = samp_locs[:, 0], samp_locs[:, 1]
    Bx, By = bandwidths
    taux, tauy = taus

    num_dirac = xk.size

    # reshape variables to use broadcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    amp_k = np.reshape(amp_k, (1, -1), order='F')
    x_samp_loc = np.reshape(x_samp_loc, (-1, 1), order='F')
    y_samp_loc = np.reshape(y_samp_loc, (-1, 1), order='F')

    # the Phi matrix consists of three parts: the derivatives w.r.t. x, y, and amplitude
    Phi1 = -amp_k * d_periodic_sinc(x_samp_loc - xk, Bx, taux) * \
           dirichlet_kernel(y_samp_loc - yk, By, tauy)
    Phi2 = -amp_k * dirichlet_kernel(x_samp_loc - xk, Bx, taux) * \
           d_periodic_sinc(y_samp_loc - yk, By, tauy)
    Phi3 = dirichlet_kernel(x_samp_loc - xk, Bx, taux) * \
           dirichlet_kernel(y_samp_loc - yk, By, tauy)
    Phi = np.column_stack((Phi1, Phi2, Phi3))

    if is_matrix:
        return linalg.solve(np.dot(Phi.conj().T, linalg.solve(noise_covariance, Phi)),
                            np.eye(3 * num_dirac))
    else:
        return linalg.solve(np.dot(Phi.conj().T, Phi / noise_covariance), np.eye(3 * num_dirac))


def compute_avg_dist_theoretical(loc_k, amp_k, samp_locs, bandwidths, taus, noise_covariance):
    """
    compute the theoretical average reconstruction error on the Dirac locations
    :param loc_k: a num_dirac x 2 matrix for the Dirac locations (column 0: x, column 1: y)
    :param amp_k: Dirac amplitudes
    :param samp_locs: an N x 2 matrix for the sampling locations of the low-pass filtered 2D Dirac
    :param bandwidths: a tuple of size 2 for the bandwidth along x and y axis, respectively
    :param taus: periods along x and y axis for the periodic 2D Dirac stream
    :param noise_covariance: either a scalar or a matrix for the covariance of noise
    :return:
    """
    num_dirac = loc_k.shape[0]
    crb_mtx = compute_crb(loc_k, amp_k, samp_locs, bandwidths, taus, noise_covariance)
    avg_dist_theoretical = \
        np.sum(np.sqrt(np.diag(crb_mtx)[:num_dirac] +
                       np.diag(crb_mtx)[num_dirac:2 * num_dirac])) / num_dirac
    return avg_dist_theoretical


def gen_dirac_param_2d(num_dirac, taus, taus_min=(0, 0), bandwidth=None,
                       save_param=False, file_name=None):
    """
    generate Dirac parameters
    :param num_dirac: number of Dirac deltas
    :param bandwidth: a tuple of size 2 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 2 for the periods of the Dirac stream along x and y axis
    :param save_param: whether to save the Dirac parameters or not
    :return:
    """
    if file_name is None:
        save_param = False

    taux, tauy = taus
    taux0, tauy0 = taus_min
    # amplitudes of the Diracs 0.1 to 1.9 with random sign
    ampk = np.sign(np.random.randn(num_dirac)) * (1 + (np.random.rand(num_dirac) - 0.5) * 1.8)

    if bandwidth is None:
        xk = np.random.rand(num_dirac) * (taux - taux0) + taux0
        yk = np.random.rand(num_dirac) * (tauy - tauy0) + tauy0
    else:
        Bx, By = bandwidth
        # Dirac locations
        a1 = 1. / (Bx * taux)
        a2 = 1. / (By * tauy)

        uk = np.random.exponential(scale=1. / num_dirac, size=(num_dirac - 1, 1))
        vk = np.random.exponential(scale=1. / num_dirac, size=(num_dirac - 1, 1))

        xk = np.cumsum(a1 + (1. - num_dirac * a1) * (1 - 0.1 * np.random.rand()) / uk.sum() * uk)
        xk = np.sort(np.hstack((np.random.rand() * xk[0] / 2., xk)) +
                     (1 - xk[-1]) / 2.) * (taux - taux0) + taux0

        yk = np.cumsum(a2 + (1. - num_dirac * a2) * (1 - 0.1 * np.random.rand()) / vk.sum() * vk)
        yk = np.sort(np.hstack((np.random.rand() * yk[0] / 2., yk)) +
                     (1 - yk[-1]) / 2.) * (tauy - tauy0) + tauy0
        np.random.shuffle(yk)

    loc_k = np.column_stack((xk, yk))

    if save_param:
        dir_name = os.path.dirname(os.path.abspath(file_name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        print('Saving Dirac parameters in {0}'.format(file_name))
        np.savez(file_name, loc_k=loc_k, ampk=ampk)

    return loc_k, ampk


def gen_dirac_samp_2d(loc_k, ampk, num_samp, bandwidth, taus,
                      snr_level=np.inf, uniform_samp=False, **kwargs):
    """
    Generate ideally low-pass filtered samples of Diracs
    :param loc_k: a 2-COLUMN matrix for the Diracs x and y locations, respectively
    :param ampk: the associated Dirac amplitudes
    :param num_samp: number of ideally low-pass filtered samples
    :param bandwidth: a tuple of size 2 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 2 for the periods of the Dirac stream along x and y axis
    :param snr_level: signal to noise ratio in the samples.
    :param uniform_samp: whether to use uniform sampling or not.
    :return:
    """
    complex_val = True
    if np.iscomplexobj(ampk):
        if np.max(np.abs(ampk.imag)) < 1e-12:
            complex_val = False
    else:
        complex_val = False

    Bx, By = bandwidth
    taux, tauy = taus
    xk, yk = loc_k[:, 0], loc_k[:, 1]
    shape_b = (int(By * tauy), int(Bx * taux))
    assert shape_b[0] // 2 * 2 + 1 == shape_b[0]
    assert shape_b[1] // 2 * 2 + 1 == shape_b[1]

    freq_limit_y = shape_b[0] // 2
    freq_limit_x = shape_b[1] // 2

    # generate random sampling locations within the box [-0.5*tau, 0.5*tau]
    if uniform_samp:
        if 'hoizontal_samp_sz' in kwargs and 'vertical_samp_sz' in kwargs:
            x_samp_sz = kwargs['hoizontal_samp_sz']
            y_samp_sz = kwargs['vertical_samp_sz']
        else:
            # assume equal sizes along x and y directions
            x_samp_sz = y_samp_sz = int(np.ceil(np.sqrt(num_samp)))

        x_samp_loc, y_samp_loc = \
            np.meshgrid(np.linspace(0, taux, x_samp_sz, endpoint=False),
                        np.linspace(0, tauy, y_samp_sz, endpoint=False))
    else:
        x_samp_loc = np.random.rand(num_samp) * taux
        y_samp_loc = np.random.rand(num_samp) * tauy

    m_grid_samp, n_grid_samp = np.meshgrid(np.arange(-freq_limit_x, freq_limit_x + 1),
                                           np.arange(-freq_limit_y, freq_limit_y + 1))
    # reshape to use broadcasting
    m_grid_samp = np.reshape(m_grid_samp, (1, -1), order='F')
    n_grid_samp = np.reshape(n_grid_samp, (1, -1), order='F')
    x_samp_loc = np.reshape(x_samp_loc, (-1, 1), order='F')
    y_samp_loc = np.reshape(y_samp_loc, (-1, 1), order='F')
    samp_loc = np.column_stack((x_samp_loc, y_samp_loc))

    mtx_fri2samp = np.exp(1j * 2 * np.pi / taux * m_grid_samp * x_samp_loc +
                          1j * 2 * np.pi / tauy * n_grid_samp * y_samp_loc) / (Bx * By)

    # compute the noiseless Fourier transform of the Dirac deltas
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    ampk = np.reshape(ampk, (-1, 1), order='F')
    fourier_noiseless = \
        np.dot(np.exp(-1j * 2 * np.pi / taux * m_grid_samp.T * xk -
                      1j * 2 * np.pi / tauy * n_grid_samp.T * yk),
               ampk).squeeze() / (taux * tauy)
    samp_noiseless = np.dot(mtx_fri2samp, fourier_noiseless)
    if not complex_val:
        samp_noiseless = np.real(samp_noiseless)

    # add noise based on the noise level
    if complex_val:
        noise = np.random.randn(num_samp) + 1j * np.random.randn(num_samp)
    else:
        noise = np.random.randn(num_samp)

    # normalize based on SNR
    noise /= linalg.norm(noise)
    noise *= linalg.norm(samp_noiseless) / (10 ** (snr_level / 20.))

    samp_noisy = samp_noiseless + noise

    return samp_noisy, samp_loc, samp_noiseless


if __name__ == '__main__':
    '''
    test cases for the utility functions
    '''
    interval = (1., 1.)
    x_ref = np.random.rand(10) * interval[0]
    y_ref = np.random.rand(10) * interval[1]
    x_recon = np.random.rand(10) * interval[0]
    y_recon = np.random.rand(10) * interval[1]

    dist1 = planar_distance(x_ref, y_ref, x_recon, y_recon, interval)[0]
    dist2 = nd_distance(np.column_stack((x_ref, y_ref)),
                        np.column_stack((x_recon, y_recon)),
                        interval)[0]
    print(dist1, dist2, dist1 - dist2)

    # now test 3D distances
    num_pt = 10
    interval = (1, 1, 1)
    pt_ref = np.random.rand(num_pt, 3) * np.reshape(np.array(interval), (1, -1), order='F')
    pt_recon = pt_ref[np.random.permutation(num_pt), :]
    dist3 = nd_distance(pt_ref, pt_recon, interval)[0]
    print(dist3)
