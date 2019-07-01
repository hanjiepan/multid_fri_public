"""
An example to illustrate the recovery of 5 Dirac deltas with two distinct horizontal and
vertical locations only:
         (x2, y1)
(x1, y2) (x2, y2) (x3, y2)
         (x2, y3)
"""
from __future__ import division
import os
import subprocess
import warnings
import numpy as np
from scipy import linalg
from alg_joint_estimation_2d import dirac_recon_joint_interface
from alg_sep_estimation import dirac_recon_sep_interface
from utils_2d import gen_dirac_samp_2d, planar_distance, compute_avg_dist_theoretical
from plotter import planar_plot_diracs, plot_2d_dirac_samples

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
    snr_experiment = 5  # in dB
    fig_format = 'pdf'  # figure format
    dpi = 600  # dpi used to save figure
    save_fig = True
    fig_dir = './result/'
    if save_fig and not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    result_dir = './result/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    K = 5  # number of Dirac deltas
    freq_limit_x = freq_limit_y = 5
    tau_x = tau_y = 1  # period of the Dirac stream on a 2D plane
    taus = (tau_x, tau_y)

    Bx = (2 * freq_limit_x + 1) / tau_x
    By = (2 * freq_limit_y + 1) / tau_y
    bandwidth = (Bx, By)

    # number of spatial domain samples
    hoizontal_samp_sz = int(np.ceil(Bx * tau_x))
    vertical_samp_sz = int(np.ceil(By * tau_y))
    num_samp = hoizontal_samp_sz * vertical_samp_sz

    # print experiment setup
    print('Reconstruct {K} Dirac delta from '
          '{num_samp} of samples with SNR = {snr}dB'.format(
        K=K, num_samp=num_samp, snr=snr_experiment))

    # randomly generate Dirac parameters (locations and amplitudes)
    # the factor 0.95 is not necessary (here it is used for the plotting consideration)
    xk = np.random.rand() * 0.8 * tau_x + 0.1 * tau_x
    yk = np.random.rand() * 0.7 * tau_y + 0.15 * tau_y
    width1 = 0.15 * tau_x
    width2 = 0.12 * tau_x
    height1 = 0.1 * tau_y
    height2 = 0.18 * tau_y

    # x_gt = np.array([xk,
    #                  np.mod(xk - width1, tau_x), xk, np.mod(xk + width2, tau_x),
    #                  xk])
    # y_gt = np.array([np.mod(yk - height1, tau_y),
    #                  yk, yk, yk,
    #                  np.mod(yk + height2, tau_y)])
    # dirac_locs = np.column_stack((x_gt, y_gt))
    # dirac_amp = np.ones(num_drac)
    #
    # np.savez('./data/dirac_param_shared_xy.npz',
    #          loc_k=dirac_locs, ampk=dirac_amp)

    dirac_param = np.load('./data/dirac_param_shared_xy.npz')
    dirac_locs = dirac_param['loc_k']
    x_gt, y_gt = dirac_locs[:, 0], dirac_locs[:, 1]
    dirac_amp = dirac_param['ampk']

    # generate samples of the Dirac deltas
    samp_noisy, samp_loc, samp_noiseless = \
        gen_dirac_samp_2d(dirac_locs, dirac_amp, num_samp, bandwidth,
                          taus=taus, snr_level=snr_experiment,
                          uniform_samp=True,
                          hoizontal_samp_sz=hoizontal_samp_sz,
                          vertical_samp_sz=vertical_samp_sz)

    # apply FRI reconstructions
    '''joint estimation'''
    xk_recon_joint, yk_recon_joint, amp_recon_joint = \
        dirac_recon_joint_interface(
            samp_noisy, num_dirac=K, samp_loc=samp_loc,
            bandwidth=bandwidth, taus=taus,
            max_ini=50, max_inner_iter=20,
            stop_cri='max_iter', strategy='both',
            max_num_same_x=3, max_num_same_y=3,
            use_new_formulation=True)

    '''separate estimation'''
    xk_recon_sep, yk_recon_sep, amp_recon_sep = \
        dirac_recon_sep_interface(
            samp_noisy, num_dirac=K, samp_loc=samp_loc,
            bandwidth=bandwidth, taus=taus,
            max_ini=50, max_inner_iter=20,
            stop_cri='max_iter', strategy='both')

    # compute noise covariance based on SNR
    noise_covariance = \
        linalg.norm(samp_noiseless.flatten()) ** 2 * \
        10 ** (-0.1 * snr_experiment) / samp_noiseless.size
    metric_crb = \
        compute_avg_dist_theoretical(dirac_locs, dirac_amp, samp_loc,
                                     bandwidth, taus, noise_covariance)

    # compute reconstruction error in Dirac locations
    dist_err_joint, sort_idx_joint = \
        planar_distance(x_gt, y_gt, xk_recon_joint, yk_recon_joint, taus)
    dist_err_sep, sort_idx_sep = \
        planar_distance(x_gt, y_gt, xk_recon_sep, yk_recon_sep, taus)
    # sort accordingly
    arg_sort1 = np.argsort(sort_idx_joint[:, 0])
    x_gt_sorted = x_gt[sort_idx_joint[:, 0][arg_sort1]]
    y_gt_sorted = y_gt[sort_idx_joint[:, 0][arg_sort1]]
    dirac_amp_sorted = dirac_amp[sort_idx_joint[:, 0][arg_sort1]]

    xk_recon_joint_sorted = xk_recon_joint[sort_idx_joint[:, 1][arg_sort1]]
    yk_recon_joint_sorted = yk_recon_joint[sort_idx_joint[:, 1][arg_sort1]]
    amp_recon_joint_sorted = amp_recon_joint[sort_idx_joint[:, 1][arg_sort1]]

    arg_sort2 = np.argsort(sort_idx_sep[:, 0])
    xk_recon_sep_sorted = xk_recon_sep[sort_idx_sep[:, 1][arg_sort2]]
    yk_recon_sep_sorted = yk_recon_sep[sort_idx_sep[:, 1][arg_sort2]]
    amp_recon_sep_sorted = amp_recon_sep[sort_idx_sep[:, 1][arg_sort2]]

    # print reconstruction results
    print('Ground truth Dirac locations (x, y) :\n {0}'.format(
        np.column_stack((x_gt_sorted, y_gt_sorted))))
    print('Ground truth Dirac amplitudes:\n {0}'.format(dirac_amp_sorted))
    print('Cramer-Rao bound: {0:.2e}\n'.format(metric_crb))

    print('Joint estimation result')
    print('---------------------------')
    print('Reconstruction error: {0:.2e}'.format(dist_err_joint))
    print('Reconstructed Dirac locations (x, y):\n {0}'.format(
        np.column_stack((xk_recon_joint_sorted, yk_recon_joint_sorted))))
    print('Reconstructed Dirac amplitudes: {0}\n'.format(amp_recon_joint_sorted))

    print('Separate estimation result')
    print('---------------------------')
    print('Reconstruction error: {0:.2e}'.format(dist_err_sep))
    print('Reconstructed Dirac locations (x, y):\n {0}'.format(
        np.column_stack((xk_recon_sep_sorted, yk_recon_sep_sorted))))
    print('Reconstructed Dirac amplitudes: {0}'.format(amp_recon_sep_sorted))

    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                        precision=8, suppress=False, threshold=1000, formatter=None)

    # plot reconstruction
    # measurements
    if np.isinf(snr_experiment):
        if use_latex:
            title_str = r'${L1}\times{L2}$ noiseless samples'.format(
                L1=vertical_samp_sz, L2=hoizontal_samp_sz)
        else:
            title_str = '{L1} x {L2} noiseless samples'.format(
                L1=vertical_samp_sz, L2=hoizontal_samp_sz)
    else:
        if use_latex:
            title_str = r'${L1}\times{L2}$ samples (SNR = ${snr}$dB)'.format(
                L1=vertical_samp_sz, L2=hoizontal_samp_sz, snr=snr_experiment)
        else:
            title_str = '{L1} x {L2} samples (SNR = {snr}dB)'.format(
                L1=vertical_samp_sz, L2=hoizontal_samp_sz, snr=snr_experiment)
    plot_2d_dirac_samples(
        samples=np.reshape(samp_noisy, (vertical_samp_sz, -1), order='F'),
        save_fig=save_fig,
        file_name=fig_dir +
                  'example_common_cord_measurement_{}dB'.format(snr_experiment),
        file_format=fig_format, dpi=dpi,
        has_title=True, title_str=title_str,
        close_fig=True)
    # Dirac locations
    if np.isinf(snr_experiment):
        if use_latex:
            title_str = r'${K}$ Diracs, ${L1}\times{L2}$ samples'.format(
                K=K, L1=vertical_samp_sz, L2=hoizontal_samp_sz)
        else:
            title_str = 'num_dirac Diracs, {L1} x {L2} samples'.format(
                K=K, L1=vertical_samp_sz, L2=hoizontal_samp_sz)
    else:
        if use_latex:
            title_str = r'${L1}\times{L2}$ samples ' \
                        r'(SNR = ${snr}$dB), ' \
                        r'$\bm{{r}}_{{\text{{error}}}}=' \
                        r'\num{{{recon_error:.2e}}}$'.format(
                L1=vertical_samp_sz, L2=hoizontal_samp_sz,
                snr=snr_experiment,
                recon_error=dist_err_joint)
        else:
            title_str = '{L1} x {L2} samples (SNR = ${snr}$dB), ' \
                        'r_error = {recon_error:.2e}'.format(
                L1=vertical_samp_sz, L2=hoizontal_samp_sz,
                snr=snr_experiment,
                recon_error=dist_err_joint)

    planar_plot_diracs(x_ref=x_gt, y_ref=y_gt, amp_ref=dirac_amp,
                       x_recon=xk_recon_joint,
                       y_recon=yk_recon_joint,
                       amp_recon=amp_recon_joint,
                       xlim=(0, tau_x), ylim=(0, tau_y),
                       save_fig=save_fig,
                       file_name=fig_dir +
                                 'example_common_cord_{}dB'.format(snr_experiment),
                       file_format=fig_format, dpi=dpi,
                       has_title=True,
                       title_str=title_str,
                       title_fontsize=10,
                       close_fig=False)
