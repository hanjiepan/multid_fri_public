"""
Average performance of the FRI reconstruction algorithms in reconstructing 2D Diracs
with different noise levels.
Two approaches have been considered: joint annihilation and separate annihilation along
horizontal and vertical directions.
"""
import argparse
import datetime
import os
import subprocess
import time
import numpy as np
from scipy import linalg
from alg_joint_estimation_2d import dirac_recon_joint_interface
from alg_sep_estimation import dirac_recon_sep_interface
from plotter import plot_avg_perf_diracs
from utils_2d import gen_dirac_param_2d, gen_dirac_samp_2d, planar_distance, \
    compute_avg_dist_theoretical

try:
    which_latex = subprocess.check_output(['which', 'latex'])
    os.environ['PATH'] = \
        os.environ['PATH'] + ':' + \
        os.path.dirname(which_latex.decode('utf-8').rstrip('\n'))
    use_latex = True
except subprocess.CalledProcessError:
    use_latex = False


def parse_args():
    """
    parse input arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr_bg', type=int, required=False,
                        default=0, help='Starting SNR index for the average performance simulations.')
    parser.add_argument('--snr_end', type=int, required=False,
                        help='End SNR index for the average performance simulations.')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    snr_bg = args['snr_bg']
    snr_end = args['snr_end']

    script_purpose = 'plotting'  # can either be 'testing', 'production' or 'plotting'
    # depends on the purpose, we choose a different set of parameters
    parameter_set = {}
    if script_purpose == 'testing':
        parameter_set = {
            'load_plot_data': False,
            'max_noise_realization': 10,
            'over_samp_factor': 1,
            'stop_cri': 'max_iter',
            'load_dirac_data': False,
            'load_sample_data': False,
            'strategy': 'both',
            'dpi': 300
        }
    elif script_purpose == 'production':
        parameter_set = {
            'load_plot_data': False,
            'max_noise_realization': 1000,
            # over-sampling factor (compared with the critical sampling for the separate recon. cases)
            'over_samp_factor': 1,
            'stop_cri': 'max_iter',
            'load_dirac_data': True,
            'load_sample_data': False,
            'strategy': 'both',
            'dpi': 600
        }
    elif script_purpose == 'plotting':
        parameter_set = {
            'over_samp_factor': 1,
            'load_plot_data': True,
            'load_sample_data': False,
            'dpi': 600
        }
    else:
        RuntimeError('Unknown script purpose: {}'.format(script_purpose))

    print('Parameter profile: {}'.format(script_purpose))
    print('Start simulations at {}'.format(datetime.datetime.now()))

    fig_format = 'pdf'  # figure format
    save_fig = True
    fig_dir = './result/'
    if save_fig and not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    result_dir = './result/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # a list of SNRs
    # -10dB to 40dB
    snr_seq = np.array([-10, -5, 0, 2.5, 5, 7.5,
                        10, 11, 12, 13, 14, 15,
                        20, 25, 30, 35, 40])

    if snr_end is None:
        snr_end = snr_seq.size

    print('SNR list: {0}'.format(snr_seq[snr_bg:snr_end]))

    K = 7  # number of Dirac delats
    tau_x = tau_y = 1  # period of the Dirac stream on a 2D plane
    taus = (tau_x, tau_y)
    # number of spatial domain samples
    hoizontal_samp_sz = int(np.ceil((K + 1) * np.sqrt(parameter_set['over_samp_factor'])))
    vertical_samp_sz = int(np.ceil((K + 1) * np.sqrt(parameter_set['over_samp_factor'])))
    # make sure in the cricitcal case, the samples sizes are not
    # too small for the separate annihilation approach
    hoizontal_samp_sz = max(2 * (hoizontal_samp_sz // 2) + 1, hoizontal_samp_sz)
    vertical_samp_sz = max(2 * (vertical_samp_sz // 2) + 1, vertical_samp_sz)
    num_samp = hoizontal_samp_sz * vertical_samp_sz

    freq_limit_x = hoizontal_samp_sz // 2  # at most sample_sz // 2
    freq_limit_y = vertical_samp_sz // 2
    # if hoizontal_samp_sz % 2 == 0:
    #     freq_limit_x = (hoizontal_samp_sz - 1) // 2
    # else:
    #     freq_limit_x = hoizontal_samp_sz // 2  # at most sample_sz // 2
    #
    # if vertical_samp_sz % 2 == 0:
    #     freq_limit_y = (vertical_samp_sz - 1) // 2
    # else:
    #     freq_limit_y = vertical_samp_sz // 2

    Bx = (2 * freq_limit_x + 1) / tau_x
    By = (2 * freq_limit_y + 1) / tau_y
    bandwidth = (Bx, By)

    if not parameter_set['load_plot_data']:
        sub_folder_name = os.path.abspath('./data/signal_diff_noise/')
        max_noise_realizations = parameter_set['max_noise_realization']
        # initialize
        print_eta = True
        avg_recon_dist_joint = np.zeros(snr_seq.size, dtype=float)
        avg_recon_dist_sep = np.zeros(snr_seq.size, dtype=float)
        metric_crb = np.zeros(snr_seq.size, dtype=float)

        total_noise_realizations_joint = max_noise_realizations
        total_noise_realizations_sep = max_noise_realizations

        # randomly generate Dirac parameters (locations and amplitudes)
        signal_file_name = sub_folder_name + '/dirac_param.npz'
        if parameter_set['load_dirac_data']:
            dirac_data = np.load(signal_file_name)
            dirac_locs = dirac_data['loc_k']
            dirac_amp = dirac_data['ampk']
        else:
            # * 0.9 is not necessary but can avoid displaying Diracs on the boundary of the box
            dirac_locs, dirac_amp = \
                gen_dirac_param_2d(num_dirac=K, taus=(taus[0] * 0.9, taus[1] * 0.9), bandwidth=bandwidth,
                                   save_param=True, file_name=signal_file_name)

        x_gt, y_gt = dirac_locs[:, 0], dirac_locs[:, 1]

        for snr_ind, snr_loop in enumerate(snr_seq):
            # initialize a matrix to store reconstructed Dirac locations
            # the matrix is of shape: 3 x num_dirac x max_noise_realizations
            recon_joint = np.full((3, K, max_noise_realizations),
                                  np.nan, dtype=dirac_amp.dtype)
            recon_sep = np.full((3, K, max_noise_realizations),
                                np.nan, dtype=dirac_amp.dtype)

            if snr_bg <= snr_ind < snr_end:
                for noise_realizations in range(max_noise_realizations):
                    # generate samples of the Dirac deltas
                    dirac_samp_file = \
                        sub_folder_name + \
                        '/samples_snr{0}_realization{1}.npz'.format(
                            snr_ind, noise_realizations)
                    if parameter_set['load_sample_data']:
                        sample_data = np.load(dirac_samp_file)
                        samp_noisy = sample_data['samp_noisy']
                        samp_loc = sample_data['samp_loc']
                        samp_noiseless = sample_data['samp_noiseless']
                    else:
                        samp_noisy, samp_loc, samp_noiseless = \
                            gen_dirac_samp_2d(dirac_locs, dirac_amp, num_samp, bandwidth,
                                              taus=taus,
                                              snr_level=snr_loop,
                                              uniform_samp=True,
                                              hoizontal_samp_sz=hoizontal_samp_sz,
                                              vertical_samp_sz=vertical_samp_sz)

                        if not os.path.exists(sub_folder_name):
                            os.mkdir(sub_folder_name)
                        np.savez(dirac_samp_file,
                                 **{
                                     'samp_noisy': samp_noisy,
                                     'samp_loc': samp_loc,
                                     'samp_noiseless': samp_noiseless
                                 })

                    # compute Cramer-Rao bound
                    if noise_realizations == 0:
                        complex_val = True
                        if np.iscomplexobj(dirac_amp):
                            if np.max(np.abs(dirac_amp.imag)) < 1e-12:
                                complex_val = False
                        else:
                            complex_val = False
                        # compute noise covariance based on SNR
                        noise_covariance = \
                            linalg.norm(samp_noiseless.flatten()) ** 2 * \
                            10 ** (-0.1 * snr_loop) / samp_noiseless.size
                        metric_crb[snr_ind] = \
                            compute_avg_dist_theoretical(dirac_locs, dirac_amp, samp_loc,
                                                         bandwidth, taus, noise_covariance)

                    # apply FRI reconstructions
                    tic = time.time()
                    xk_recon_joint, yk_recon_joint, amp_recon_joint = \
                        dirac_recon_joint_interface(
                            samp_noisy, num_dirac=K, samp_loc=samp_loc,
                            bandwidth=bandwidth, taus=taus,
                            max_ini=50, max_inner_iter=20,
                            stop_cri=parameter_set['stop_cri'],
                            strategy=parameter_set['strategy'])

                    xk_recon_sep, yk_recon_sep, amp_recon_sep = \
                        dirac_recon_sep_interface(
                            samp_noisy, num_dirac=K, samp_loc=samp_loc,
                            bandwidth=bandwidth, taus=taus,
                            max_ini=50, max_inner_iter=20,
                            stop_cri=parameter_set['stop_cri'],
                            strategy=parameter_set['strategy'])
                    if print_eta:
                        duration_one_run = datetime.timedelta(seconds=(time.time() - tic))
                        print('Expected to be finished by {0}'.format(
                            datetime.datetime.now() +
                            duration_one_run * max_noise_realizations *
                            (snr_end - snr_bg)))
                        print_eta = False

                    # store reconstructions in a matrix
                    recon_joint[:, :xk_recon_joint.size, noise_realizations] = \
                        np.row_stack((xk_recon_joint, yk_recon_joint, amp_recon_joint))
                    recon_sep[:, :xk_recon_sep.size, noise_realizations] = \
                        np.row_stack((xk_recon_sep, yk_recon_sep, amp_recon_sep))

                    # compute reconstruction error in Dirac locations
                    if xk_recon_joint.size > 0:
                        dist_err_joint, sort_idx_joint = \
                            planar_distance(x_gt, y_gt, xk_recon_joint, yk_recon_joint, taus)
                        avg_recon_dist_joint[snr_ind] += dist_err_joint
                    else:
                        total_noise_realizations_joint -= 1
                        print('No roots found from joint estimation: '
                              'SNR index {snr_ind}, '
                              'noise realization {noise_real}'.format(
                            snr_ind=snr_ind,
                            noise_real=noise_realizations))

                    if xk_recon_sep.size > 0:
                        dist_err_sep, sort_idx_sep = \
                            planar_distance(x_gt, y_gt, xk_recon_sep, yk_recon_sep, taus)
                        avg_recon_dist_sep[snr_ind] += dist_err_sep
                    else:
                        total_noise_realizations_sep -= 1
                        print('No roots found from separate estimation: '
                              'SNR index {snr_ind}, '
                              'noise realization {noise_real}'.format(
                            snr_ind=snr_ind,
                            noise_real=noise_realizations))

                # save after each noise level
                result_sub_folder_name = \
                    os.path.abspath(result_dir + '/signal_diff_noise/')
                if not os.path.exists(result_sub_folder_name):
                    os.mkdir(result_sub_folder_name)
                np.savez(result_sub_folder_name +
                         '/recon_K{K}_L{L}_snr{snr_loop}dB.npz'.format(
                             K=K, L=num_samp, snr_loop=snr_loop),
                         recon_joint=recon_joint, recon_sep=recon_sep)
                np.savez(result_dir +
                         'avg_perf_K{K}_L{L}_snr{snr_bg}_to_{snr_end}.npz'.format(
                             K=K, L=num_samp, snr_bg=snr_bg, snr_end=snr_end),
                         avg_recon_dist_sep=avg_recon_dist_sep / total_noise_realizations_sep,
                         avg_recon_dist_joint=avg_recon_dist_joint / total_noise_realizations_joint,
                         snr_seq=snr_seq, metric_crb=metric_crb)

        avg_recon_dist_joint /= total_noise_realizations_joint
        avg_recon_dist_sep /= total_noise_realizations_sep

        # save plotting data
        np.savez(result_dir +
                 'avg_perf_K{K}_L{L}_snr{snr_bg}_to_{snr_end}.npz'.format(
                     K=K, L=num_samp, snr_bg=snr_bg, snr_end=snr_end),
                 metric_crb=metric_crb,
                 avg_recon_dist_sep=avg_recon_dist_sep,
                 avg_recon_dist_joint=avg_recon_dist_joint,
                 snr_seq=snr_seq)

    else:
        # load plotting data
        plot_data_file_name = \
            result_dir + \
            'avg_perf_K{K}_L{L}_snr{snr_bg}_to_{snr_end}.npz'.format(
                K=K, L=num_samp, snr_bg=snr_bg, snr_end=snr_end)
        plot_data = np.load(plot_data_file_name)
        avg_recon_dist_joint = plot_data['avg_recon_dist_joint']
        avg_recon_dist_sep = plot_data['avg_recon_dist_sep']
        snr_seq = plot_data['snr_seq']
        metric_crb = plot_data['metric_crb']

    print('Finish simulations by {}'.format(datetime.datetime.now()))

    # plot results
    file_name = fig_dir + 'avg_perf_2d_K{K}_L{L}'.format(K=K, L=num_samp)
    if use_latex:
        fig_title = r'$K = {K}, \mbox{{ sample size: }} {L1}\times{L2}$'.format(
            K=K, L1=vertical_samp_sz, L2=hoizontal_samp_sz)
    else:
        fig_title = 'K = {K}, sample size: {L1} x {L2}'.format(
            K=K, L1=vertical_samp_sz, L2=hoizontal_samp_sz)

    plot_avg_perf_diracs(
        avg_recon_dist_joint,
        avg_recon_dist_sep,
        snr_seq,
        metric_crb=metric_crb,
        xlabel_str='noise level (SNR in [dB])',
        ylabel_str='reconstruction error',
        save_fig=save_fig,
        fig_format=fig_format,
        file_name=file_name,
        label1='joint estimation',
        label2='separate estimation',
        label3=r'Cram\'{e}r-Rao bound' if use_latex else u'Cramér-Rao bound',
        fig_title=fig_title,
        dpi=parameter_set['dpi'],
        close_fig=True)
