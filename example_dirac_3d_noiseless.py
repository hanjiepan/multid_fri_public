"""
An example to show that with very few samples, the 3D Dirac can still be reconstructed.
Here for simplicity, we consider the noiseless setup.
"""
from __future__ import division

import os
import subprocess
import warnings
import numpy as np

from alg_joint_estimation_3d import dirac_recon_joint_ideal_interface
from plotter import cubic_plot_diracs, cubic_plot_diracs_plotly, plot_3d_dirac_samples
from utils_2d import nd_distance
from utils_3d import gen_dirac_param_3d, gen_dirac_samp_ideal_3d

try:
    which_latex = subprocess.check_output(['which', 'latex'])
    os.environ['PATH'] = \
        os.environ['PATH'] + ':' + \
        os.path.dirname(which_latex.decode('utf-8').rstrip('\n'))
    use_latex = True
except subprocess.CalledProcessError:
    use_latex = False

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    snr_experiment = np.inf
    fig_format = 'pdf'  # figure format
    dpi = 300  # dpi used to save figure
    save_fig = True
    fig_dir = './result/'
    plt_backend = ['matplotlib', 'plotly']  # plotting tools
    if save_fig and not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    result_dir = './result/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    K = 5  # number of Dirac deltas
    freq_limit_x = freq_limit_y = freq_limit_z = 1
    tau_x = tau_y = tau_z = 1  # period of the Dirac stream in the 3D volume
    taus = (tau_x, tau_y, tau_z)

    Bx = (2 * freq_limit_x + 1) / tau_x
    By = (2 * freq_limit_y + 1) / tau_y
    Bz = (2 * freq_limit_z + 1) / tau_z
    bandwidth = (Bx, By, Bz)

    # number of spatial domain samples
    hoizontal_samp_sz = int(np.ceil(Bx * tau_x))
    vertical_samp_sz = int(np.ceil(By * tau_y))
    depth_samp_sz = int(np.ceil(Bz * tau_z))
    num_samp = hoizontal_samp_sz * vertical_samp_sz * depth_samp_sz

    # print experiment setup
    print('Reconstruct {K} Diracs from {num_samp} of samples'.format(
        K=K, num_samp=num_samp))

    # randomly generate Dirac parameters (locations and amplitudes)
    # the factor 0.95 is not necessary (here it is used for the plotting consideration)
    # dirac_locs, dirac_amp = \
    #     gen_dirac_param_3d(num_dirac=num_drac, bandwidth=bandwidth,
    #                        taus=(tau_x * 0.95, tau_y * 0.95, tau_z * 0.95),
    #                        file_name='./data/example_3d.npz', save_param=True)
    # load saved Dirac parameter
    dirac_param = np.load('./data/example_3d.npz')
    dirac_locs = dirac_param['loc_k']
    dirac_amp = dirac_param['ampk']

    # generate samples of the Dirac deltas
    samp_noisy, samp_loc, samp_noiseless = \
        gen_dirac_samp_ideal_3d(dirac_locs, dirac_amp,
                                num_samp, bandwidth, taus,
                                snr_level=snr_experiment,
                                hoizontal_samp_sz=hoizontal_samp_sz,
                                vertical_samp_sz=vertical_samp_sz,
                                depth_samp_sz=depth_samp_sz)

    # apply FRI reconstructions
    dirac_locs_recon, amp_recon = \
        dirac_recon_joint_ideal_interface(
            samp_noisy, num_dirac=K, samp_loc=samp_loc, bandwidth=bandwidth,
            taus=taus, max_ini=50, max_inner_iter=20, stop_cri='max_iter')

    # compute reconstruction error in Dirac locations
    dist_err, sort_idx = nd_distance(dirac_locs, dirac_locs_recon, interval=taus)
    # sort accordingly
    dirac_locs_sorted = dirac_locs[sort_idx[:, 0], :]
    dirac_locs_recon_sorted = dirac_locs_recon[sort_idx[:, 1], :]

    # print reconstruction results
    print('Reconstruction error: {0:.2e}'.format(dist_err))
    print('Ground truth Dirac parameters (x, y, z, amplitudes) :\n {0}'.format(
        np.column_stack((dirac_locs_sorted, dirac_amp[sort_idx[:, 0]]))))
    print('Reconstructed Dirac parameters (x, y, z, amplitudes):\n {0}'.format(
        np.column_stack((dirac_locs_recon_sorted, amp_recon[sort_idx[:, 1]]))))

    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                        precision=8, suppress=False, threshold=1000, formatter=None)

    # plot reconstruction
    # Dirac locations
    if np.isinf(snr_experiment):
        if use_latex:
            title_str = \
                r'${K}$ Diracs, ${L1}\times{L2}\times{L3}$ noiseless samples'.format(
                    K=K, L1=vertical_samp_sz, L2=hoizontal_samp_sz, L3=depth_samp_sz)
            title_str_plotly = \
                '{K} Diracs, {L1} by {L2} by {L3} noiseless samples'.format(
                    K=K, L1=vertical_samp_sz, L2=hoizontal_samp_sz, L3=depth_samp_sz)
        else:
            title_str = title_str_plotly = \
                '{K} Diracs, {L1} by {L2} by {L3} noiseless samples'.format(
                    K=K, L1=vertical_samp_sz, L2=hoizontal_samp_sz, L3=depth_samp_sz)
    else:
        if use_latex:
            title_str = \
                r'${K}$ Diracs, ' \
                r'${L1}\times{L2}\times{L3}$ samples ' \
                r'(SNR = ${snr}$dB)'.format(
                    K=K, L1=vertical_samp_sz,
                    L2=hoizontal_samp_sz, L3=depth_samp_sz,
                    snr=snr_experiment)
            title_str_plotly = \
                '{K} Diracs, ' \
                '{L1} by {L2} by {L3} samples ' \
                '(SNR = {snr}dB)'.format(
                    K=K, L1=vertical_samp_sz,
                    L2=hoizontal_samp_sz, L3=depth_samp_sz,
                    snr=snr_experiment)
        else:
            title_str = title_str_plotly = \
                '{K} Diracs, ' \
                '{L1} by {L2} by {L3} samples ' \
                '(SNR = {snr}dB)'.format(
                    K=K, L1=vertical_samp_sz,
                    L2=hoizontal_samp_sz, L3=depth_samp_sz,
                    snr=snr_experiment)

    for backend in plt_backend:
        if backend == 'matplotlib':
            cubic_plot_diracs(loc_ref=dirac_locs, amp_ref=dirac_amp,
                              loc_recon=dirac_locs_recon, amp_recon=amp_recon,
                              taus=taus, marker_alpha=0.7,
                              save_fig=save_fig, dpi=dpi, legend_loc=6,
                              file_name=fig_dir + 'recon_3d_dirac_{}dB'.format(snr_experiment),
                              close_fig=True, has_title=True, title_str=title_str,
                              title_fontsize=10)
        elif backend == 'plotly':
            cubic_plot_diracs_plotly(loc_ref=dirac_locs, amp_ref=dirac_amp,
                                     loc_recon=dirac_locs_recon, amp_recon=amp_recon,
                                     taus=taus, auto_open=True, has_title=True,
                                     file_name=fig_dir + 'recon_3d_dirac.html',
                                     title_str=title_str_plotly, marker_alpha=0.7)

    # measurements
    if use_latex:
        title_str = r'${L1}\times{L2}\times{L3}$ samples'.format(
            L1=vertical_samp_sz, L2=hoizontal_samp_sz, L3=depth_samp_sz)
    else:
        title_str = '{L1} x {L2} x {L3} samples'.format(
            L1=vertical_samp_sz, L2=hoizontal_samp_sz, L3=depth_samp_sz)

    # from glue import qglue
    #
    # qglue(samples=np.reshape(samp_noiseless,
    #                          (vertical_samp_sz, hoizontal_samp_sz, -1), order='F').T)

    # use maximum intensity projection
    projection_plane_lst = ['xy', 'yz', 'xz']
    for projection_plane in projection_plane_lst:
        plot_3d_dirac_samples(
            samples=np.reshape(samp_noisy,
                               (vertical_samp_sz, hoizontal_samp_sz, -1),
                               order='F'),
            projection_plane=projection_plane,
            cmap='gray', save_fig=save_fig,
            file_name=fig_dir + 'example_3d_dirac_{}dB'.format(snr_experiment),
            file_format=fig_format, dpi=dpi,
            has_title=True,
            title_str=title_str + ' (${}$-plane projection)'.format(projection_plane)
            if use_latex else '({}-plane projection)'.format(projection_plane),
            close_fig=True,
            nbins=3)
