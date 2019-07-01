from __future__ import division
import os
import numpy as np
from scipy import linalg


def gen_dirac_param_3d(num_dirac, taus, bandwidth=None,
                       save_param=False, file_name=None,
                       real_amp=True, unitary_amp=False,
                       around_zero=False, z_offset=0):
    """
    generate Dirac parameters
    :param num_dirac: number of Dirac deltas
    :param bandwidth: a tuple of size 2 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 2 for the periods of the Dirac stream along x and y axis
    :param save_param: whether to save the Dirac parameters or not
    :param file_name: file name used to save Dirac parameters
    :param real_amp: real valued amplitudes or not
    :param unitary_amp: whether the amplitudes are of unitary norm
    :param around_zero: whether to center the Dirac locations around zero or not. In the
            latter case, the Dirac locations will be in [0, tau)
    :return:
    """
    if file_name is None:
        save_param = False

    taux, tauy, tauz = taus
    # amplitudes of the Diracs 0.1 to 1.9 with random sign
    if real_amp:
        if unitary_amp:
            ampk = np.ones(num_dirac)
        else:
            ampk = np.sign(np.random.randn(num_dirac)) * \
                   (1 + (np.random.rand(num_dirac) - 0.5) * 1)
    else:
        ampk = np.exp(1j * np.random.rand(num_dirac) * 2 * np.pi) * \
               (1 + (np.random.rand(num_dirac) - 0.5) * 1)
        if unitary_amp:
            ampk /= np.abs(ampk)

    if bandwidth is None:
        xk = np.random.rand(num_dirac) * taux
        yk = np.random.rand(num_dirac) * tauy
        zk = np.random.rand(num_dirac) * tauz
    else:
        Bx, By, Bz = bandwidth
        # Dirac locations
        a1 = 1. / (Bx * taux)
        a2 = 1. / (By * tauy)
        a3 = 1. / (Bz * tauz)

        uk = np.random.exponential(scale=1. / num_dirac, size=(num_dirac - 1, 1))
        vk = np.random.exponential(scale=1. / num_dirac, size=(num_dirac - 1, 1))
        wk = np.random.exponential(scale=1. / num_dirac, size=(num_dirac - 1, 1))

        xk = np.cumsum(a1 + (1. - num_dirac * a1) * (1 - 0.1 * np.random.rand()) / uk.sum() * uk)
        xk = np.sort(np.hstack((np.random.rand() * xk[0] / 2., xk)) +
                     (1 - xk[-1]) / 2.) * taux

        yk = np.cumsum(a2 + (1. - num_dirac * a2) * (1 - 0.1 * np.random.rand()) / vk.sum() * vk)
        yk = np.sort(np.hstack((np.random.rand() * yk[0] / 2., yk)) +
                     (1 - yk[-1]) / 2.) * tauy
        np.random.shuffle(yk)

        if tauz == 0:
            zk = np.zeros(num_dirac)
        else:
            zk = np.cumsum(a3 + (1. - num_dirac * a3) * (1 - 0.1 * np.random.rand()) / wk.sum() * wk)
            zk = np.sort(np.hstack((np.random.rand() * zk[0] / 2., zk)) +
                         (1 - zk[-1]) / 2.) * tauz
        np.random.shuffle(zk)

    # make sure the Dirac locations are within one period
    xk = np.mod(xk, taux)
    yk = np.mod(yk, tauy)
    zk += z_offset
    if tauz != 0:
        zk = np.mod(zk, tauz)

    if around_zero:
        xk[xk > 0.5 * taux] -= taux
        yk[yk > 0.5 * tauy] -= tauy
        zk[zk > 0.5 * tauz] -= tauz

    loc_k = np.column_stack((xk, yk, zk))

    if save_param:
        dir_name = os.path.dirname(os.path.abspath(file_name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        print('Saving Dirac parameters in {0}'.format(file_name))
        np.savez(file_name, loc_k=loc_k, ampk=ampk)

    return loc_k, ampk


def gen_dirac_samp_ideal_3d(loc_k, ampk, num_samp, bandwidth, taus,
                            snr_level=np.inf, uniform_samp=True, **kwargs):
    """
    Generate ideally low-pass filtered samples of Diracs
    :param loc_k: a 3-COLUMN matrix for the Diracs x and y locations, respectively
    :param ampk: the associated Dirac amplitudes
    :param num_samp: number of ideally low-pass filtered samples
    :param bandwidth: a tuple of size 3 for the bandwidth of the low-pass filtering
    :param taus: a tuple of size 3 for the periods of the Dirac stream along
            x, y, and z axis
    :param snr_level: signal to noise ratio in the samples
    :param uniform_samp: whether to use uniform sampling or not
    :param kwargs: including 'hoizontal_samp_sz', 'vertical_samp_sz', and 'depth_samp_sz'
            for the size of samples for the uniform sampling setup.
    :return:
    """
    complex_val = True
    if np.iscomplexobj(ampk):
        if np.max(np.abs(ampk.imag)) < 1e-12:
            complex_val = False
    else:
        complex_val = False

    Bx, By, Bz = bandwidth
    taux, tauy, tauz = taus
    xk, yk, zk = loc_k[:, 0], loc_k[:, 1], loc_k[:, 2]
    shape_b = (int(By * tauy), int(Bx * taux), int(Bz * tauz))
    assert shape_b[0] // 2 * 2 + 1 == shape_b[0]
    assert shape_b[1] // 2 * 2 + 1 == shape_b[1]
    assert shape_b[2] // 2 * 2 + 1 == shape_b[2]

    freq_limit_x = shape_b[1] // 2
    freq_limit_y = shape_b[0] // 2
    freq_limit_z = shape_b[2] // 2

    # generate random sampling locations within the box [-0.5*tau, 0.5*tau]
    if uniform_samp:
        if 'hoizontal_samp_sz' in kwargs and \
                'vertical_samp_sz' in kwargs and \
                'depth_samp_sz' in kwargs:
            x_samp_sz = kwargs['hoizontal_samp_sz']
            y_samp_sz = kwargs['vertical_samp_sz']
            z_samp_sz = kwargs['depth_samp_sz']
        else:
            # assume equal sizes along x and y directions
            x_samp_sz = y_samp_sz = z_samp_sz = int(np.ceil(num_samp ** (1. / 3)))

        x_samp_loc, y_samp_loc, z_samp_loc = \
            np.meshgrid(np.linspace(0, taux, x_samp_sz, endpoint=False),
                        np.linspace(0, tauy, y_samp_sz, endpoint=False),
                        np.linspace(0, tauz, z_samp_sz, endpoint=False))
    else:
        x_samp_loc = np.random.rand(num_samp) * taux
        y_samp_loc = np.random.rand(num_samp) * tauy
        z_samp_loc = np.random.rand(num_samp) * tauz

    m_grid_samp, n_grid_samp, p_grid_samp = \
        np.meshgrid(np.arange(-freq_limit_x, freq_limit_x + 1),
                    np.arange(-freq_limit_y, freq_limit_y + 1),
                    np.arange(-freq_limit_z, freq_limit_z + 1))

    # reshape to use broadcasting
    m_grid_samp = np.reshape(m_grid_samp, (1, -1), order='F')
    n_grid_samp = np.reshape(n_grid_samp, (1, -1), order='F')
    p_grid_samp = np.reshape(p_grid_samp, (1, -1), order='F')

    x_samp_loc = np.reshape(x_samp_loc, (-1, 1), order='F')
    y_samp_loc = np.reshape(y_samp_loc, (-1, 1), order='F')
    z_samp_loc = np.reshape(z_samp_loc, (-1, 1), order='F')
    samp_loc = np.column_stack((x_samp_loc, y_samp_loc, z_samp_loc))

    mtx_fri2samp = np.exp(1j * 2 * np.pi / taux * x_samp_loc * m_grid_samp +
                          1j * 2 * np.pi / tauy * y_samp_loc * n_grid_samp +
                          1j * 2 * np.pi / tauz * z_samp_loc * p_grid_samp) / (Bx * By * Bz)

    # compute the noiseless Fourier transform of the Dirac deltas
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    zk = np.reshape(zk, (1, -1), order='F')
    ampk = np.reshape(ampk, (-1, 1), order='F')
    fourier_noiseless = \
        np.dot(np.exp(-1j * 2 * np.pi / taux * m_grid_samp.T * xk -
                      1j * 2 * np.pi / tauy * n_grid_samp.T * yk -
                      1j * 2 * np.pi / tauz * p_grid_samp.T * zk),
               ampk).squeeze() / (taux * tauy * tauz)
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
