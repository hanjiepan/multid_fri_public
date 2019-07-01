"""
An example to show that with very few samples, the 2D Dirac can still be reconstructed.
Here for simplicity, we consider the noiseless setup.
"""
from __future__ import division

import os
import subprocess
import warnings
import numpy as np
from alg_joint_estimation_2d import dirac_recon_joint_interface
from plotter import planar_plot_diracs, plot_2d_dirac_samples
from utils_2d import gen_dirac_param_2d, gen_dirac_samp_2d, planar_distance

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
    dpi = 600  # dpi used to save figure
    save_fig = True
    fig_dir = './result/'
    if save_fig and not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    result_dir = './result/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    K = 7  # number of Dirac deltas
    freq_limit_x = freq_limit_y = 2
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
    print('Reconstruct {K} Diracs from {num_samp} samples'.format(
        K=K, num_samp=num_samp))

    # randomly generate Dirac parameters (locations and amplitudes)
    # the factor 0.95 is not necessary (here it is used for the plotting consideration)
    dirac_locs, dirac_amp = \
        gen_dirac_param_2d(num_dirac=K,
                           taus=(tau_x * 0.95, tau_y * 0.95),
                           taus_min=(tau_x * 0.05, tau_y * 0.05),
                           save_param=False)
    x_gt, y_gt = dirac_locs[:, 0], dirac_locs[:, 1]

    # generate samples of the Dirac deltas
    samp_noisy, samp_loc, samp_noiseless = \
        gen_dirac_samp_2d(dirac_locs, dirac_amp, num_samp, bandwidth,
                          taus=taus, snr_level=snr_experiment,
                          uniform_samp=True,
                          hoizontal_samp_sz=hoizontal_samp_sz,
                          vertical_samp_sz=vertical_samp_sz)

    # apply FRI reconstructions
    xk_recon, yk_recon, amp_recon = \
        dirac_recon_joint_interface(
            samp_noisy, num_dirac=K, samp_loc=samp_loc,
            bandwidth=bandwidth, taus=taus,
            max_ini=50, max_inner_iter=20,
            use_new_formulation=True)

    # compute reconstruction error in Dirac locations
    dist_err_joint, sort_idx_joint = \
        planar_distance(x_gt, y_gt, xk_recon, yk_recon, taus)
    # sort accordingly
    x_gt_sorted = x_gt[sort_idx_joint[:, 0]]
    y_gt_sorted = y_gt[sort_idx_joint[:, 0]]
    xk_recon_sorted = xk_recon[sort_idx_joint[:, 1]]
    yk_recon_sorted = yk_recon[sort_idx_joint[:, 1]]

    # print reconstruction results
    print('Reconstruction error: {0:.2e}'.format(dist_err_joint))
    print('Ground truth Diracs (x, y) :\n {0}'.format(
        np.column_stack((x_gt_sorted, y_gt_sorted))))
    print('Reconstructed Diracs (x, y):\n {0}'.format(
        np.column_stack((xk_recon_sorted, yk_recon_sorted))))

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
            title_str = r'${L1}\times{L2}$ samples, SNR = ${snr}$dB'.format(
                L1=vertical_samp_sz, L2=hoizontal_samp_sz, snr=snr_experiment)
        else:
            title_str = '{L1} x {L2} samples, SNR = {snr}dB'.format(
                L1=vertical_samp_sz, L2=hoizontal_samp_sz, snr=snr_experiment)
    plot_2d_dirac_samples(
        samples=np.reshape(samp_noisy, (vertical_samp_sz, -1), order='F'),
        save_fig=save_fig,
        file_name=fig_dir + 'example_few_samples_measurement',
        file_format=fig_format, dpi=dpi,
        has_title=True, title_str=title_str,
        close_fig=True)
    # Dirac locations
    if use_latex:
        title_str = r'${K}$ Diracs, ${L1}\times{L2}$ samples'.format(
            K=K, L1=vertical_samp_sz, L2=hoizontal_samp_sz)
    else:
        title_str = 'num_dirac Diracs, {L1} x {L2} samples'.format(
            K=K, L1=vertical_samp_sz, L2=hoizontal_samp_sz)
    planar_plot_diracs(x_ref=x_gt, y_ref=y_gt, amp_ref=dirac_amp,
                       x_recon=xk_recon, y_recon=yk_recon, amp_recon=amp_recon,
                       xlim=(0, tau_x), ylim=(0, tau_y),
                       save_fig=save_fig,
                       file_name=fig_dir + 'example_few_samples',
                       file_format=fig_format, dpi=dpi,
                       has_title=True,
                       title_str=title_str,
                       close_fig=False)
