from __future__ import division

import os
import subprocess

import matplotlib
import numpy as np

from utils_2d import planar_distance, nd_distance

if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

try:
    which_latex = subprocess.check_output(['which', 'latex'])
    os.environ['PATH'] = \
        os.environ['PATH'] + ':' + \
        os.path.dirname(which_latex.decode('utf-8').rstrip('\n'))
    use_latex = True
except subprocess.CalledProcessError:
    use_latex = False

if use_latex:
    from matplotlib import rcParams

    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True
    rcParams['text.latex.preamble'] = [r"\usepackage{bm,amsmath,siunitx}"]

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as pg


def plot_avg_perf_diracs(metric1, metric2, data_horizontal_seq,
                         metric_crb=None,
                         xlabel_str='noise level (SNR in [dB])',
                         ylabel_str='reconstruction error',
                         save_fig=True, fig_format='pdf',
                         file_name='avg_perf_diracs',
                         label1='joint estimation',
                         label2='separate estimation',
                         label3=r"Cram{\'e}r-Rao bound" if use_latex else u'Cramér-Rao bound',
                         logx=False,
                         fig_title='', dpi=300, close_fig=True):
    """
    Plot the average performance for the joint estimation and separate estimate of 2D Diracs.
    :param metric1: average reconstruction error (approach 1)
    :param metric2: average reconstruction error (approach 2)
    :param data_horizontal_seq: a sequence of different SNRs tested
    :param metric_crb: Cramer-Rao Bound
    :param save_fig: whether to save figure or not
    :param fig_format: file format for the saved figure
    :param file_name: file name
    :param label1: label name for metric 1
    :param label2: label name for metric 2
    :param label3: label name for metric Cramer-Rao bound
    :param logx: whether to use x axis in a log scale or not
    :param fig_title: title of the figure
    :param dpi: dpi used to save the figure
    :param close_fig: close the figure or not
    :return:
    """
    color_red = [0.850, 0.325, 0.098]
    color_blue = [0, 0.447, 0.741]
    color_green = [0.466, 0.674, 0.188]
    fig = plt.figure(figsize=(5, 3), dpi=90)
    ax = plt.axes([0.19, 0.17, 0.72, 0.72])

    if logx:
        plt.loglog(data_horizontal_seq, metric1, label=label1,
                   linestyle='-', linewidth=2, color=color_red,
                   markerfacecolor=color_red, mec=color_red, alpha=0.6)

        plt.loglog(data_horizontal_seq, metric2, label=label2,
                   linestyle='-.', linewidth=1.5, color=color_blue,
                   markerfacecolor=color_blue, mec=color_blue, alpha=1)
    else:
        plt.semilogy(data_horizontal_seq, metric1, label=label1,
                     linestyle='-', linewidth=2, color=color_red,
                     markerfacecolor=color_red, mec=color_red, alpha=0.6)

        plt.semilogy(data_horizontal_seq, metric2, label=label2,
                     linestyle='-.', linewidth=1.5, color=color_blue,
                     markerfacecolor=color_blue, mec=color_blue, alpha=1)

    if metric_crb is not None:
        plt.semilogy(data_horizontal_seq, metric_crb, label=label3,
                     linestyle=':', linewidth=2, color=color_green,
                     markerfacecolor=color_green, mec=color_green, alpha=1)

    plt.xlabel(xlabel_str, fontsize=11)
    plt.ylabel(ylabel_str, fontsize=11)
    plt.title(fig_title)
    plt.grid(b=True, zorder=0, which='major', linestyle='-', color=[0.7, 0.7, 0.7])
    plt.grid(b=True, zorder=0, which='minor', linestyle=':', color=[0.7, 0.7, 0.7])
    plt.legend(scatterpoints=1, fontsize=9,
               ncol=1, markerscale=0.7,
               handletextpad=0.1, columnspacing=0.1,
               labelspacing=0.1, framealpha=0.8, frameon=True)

    if save_fig:
        plt.savefig(file_name + '.' + fig_format, format=fig_format,
                    dpi=dpi, transparent=True)

    if close_fig:
        plt.close()
    else:
        plt.show()


def planar_plot_diracs(x_ref=None, y_ref=None, amp_ref=None,
                       x_recon=None, y_recon=None, amp_recon=None,
                       background_img=None, cmap='magma_r',
                       x_plt_grid=None, y_plt_grid=None,
                       label_ref='ground truth', label_recon='reconstruction',
                       legend_marker_scale=0.7, legend_loc=0,
                       marker_scale=1, marker_alpha=0.6,
                       taus=None, xlim=None, ylim=None,
                       save_fig=False, file_name='recon_2d_dirac',
                       file_format='pdf', dpi=300, close_fig=True,
                       has_title=False, title_str=None, title_fontsize=11):
    """
    Plot the results in the 2D Dirac reconstruction.
    :param x_ref: original Dirac locations (x-axis)
    :param y_ref: original Dirac locations (y-axis)
    :param amp_ref: original Dirac amplitudes
    :param x_recon: reconstructed Dirac locations (x-axis)
    :param y_recon: reconstructed Dirac locations (y-axis)
    :param amp_recon: reconstructed Dirac amplitudes
    :param background_img: background image
    :param label_ref: label for the original Dirac markers
    :param label_recon: label for the reconstructed Dirac markers
    :param legend_marker_scale: scaling factor for the marker size in the legend
    :param legend_loc: legend locations
    :param marker_scale: scale for the markers (in addition to the re-scaling
            based onthe amplitudes
    :param marker_alpha: transparency value of the markers
    :param xlim: x-axis range
    :param ylim: y-axis range
    :param save_fig: whether to save figure or not
    :param file_name: file name used for saving
    :param file_format: figure file format
    :param dpi: dpi for saving the figure
    :param close_fig: close the figure or not
    :param has_title: whether the figure has title or not
    :param title_str: the title string
    :return:
    """
    if y_ref is not None and x_ref is not None and amp_ref is not None:
        ref_pt_available = True
    else:
        ref_pt_available = False

    if y_recon is not None and x_recon is not None and amp_recon is not None:
        recon_pt_available = True
    else:
        recon_pt_available = False

    # plot
    ax = plt.figure(figsize=(4, 3.5), dpi=90).add_subplot(111)
    pos_original = ax.get_position()
    pos_new = [pos_original.x0 + 0.03, pos_original.y0 + 0.02,
               pos_original.width, pos_original.height]
    ax.set_position(pos_new)
    if background_img is not None:
        if x_plt_grid is not None or y_plt_grid is not None:
            plt.pcolormesh(x_plt_grid, y_plt_grid, background_img,
                           shading='flat', cmap=cmap)
        else:
            plt.pcolormesh(background_img, shading='flat', cmap=cmap)

    # normalization for plotting
    if ref_pt_available:
        amp_noramalization = np.max(np.abs(amp_ref))
    elif recon_pt_available:
        amp_noramalization = np.max(np.abs(amp_recon))
    else:
        amp_noramalization = 1

    if ref_pt_available:
        plt.scatter(x_ref, y_ref,
                    s=np.abs(amp_ref) / amp_noramalization * 100 * marker_scale,
                    marker='o', edgecolors='k', linewidths=0.5,
                    alpha=marker_alpha, c='w', label=label_ref)

    if recon_pt_available:
        plt.scatter(x_recon, y_recon,
                    s=np.abs(amp_recon) / amp_noramalization * 100 * marker_scale,
                    marker='*', edgecolors='k', linewidths=0.5,
                    alpha=marker_alpha,
                    c=np.tile([0.996, 0.410, 0.703], (x_recon.size, 1)),
                    label=label_recon)

    if has_title and ref_pt_available and recon_pt_available and \
            title_str is None and taus is not None:
        # compute reconstruction distance
        dist_recon = planar_distance(x_ref, y_ref, x_recon, y_recon, taus)[0]
        plt.title(u'average error = {0}'.format(dist_recon), fontsize=title_fontsize)
    elif has_title and title_str is not None:
        plt.title(title_str, fontsize=title_fontsize)
    else:
        plt.title(u'', fontsize=title_fontsize)

    if ref_pt_available or recon_pt_available:
        plt.legend(scatterpoints=1, loc=legend_loc, fontsize=9,
                   ncol=1, markerscale=legend_marker_scale,
                   handletextpad=0.1, columnspacing=0.1,
                   labelspacing=0.1, framealpha=0.75, frameon=True)

    plt.axis('image')
    if xlim is not None:
        plt.xlim((xlim[0], xlim[1]))
    if ylim is not None:
        plt.ylim((ylim[0], ylim[1]))
    plt.xlabel('horizontal position')
    plt.ylabel('vertical position')

    if save_fig:
        plt.savefig(file_name + '.' + file_format, format=file_format,
                    dpi=dpi, transparent=True)

    if close_fig:
        plt.close()
    else:
        plt.show()


def plot_2d_dirac_samples(samples, cmap='gray', save_fig=False,
                          file_name='samples_2d_dirac',
                          file_format='pdf', dpi=300, close_fig=True,
                          has_title=False, title_str=None):
    """
    Plot 2D Dirac samples
    :param samples: samples of the 2D Dirac deltas
    :param cmap: colormap
    :param save_fig: whether to save figure or not
    :param file_name: file name used for saving
    :param file_format: figure file format
    :param dpi: dpi for saving the figure
    :param close_fig: close the figure or not
    :param has_title: whether the figure has title or not
    :param title_str: the title string
    :return:
    """
    plt.figure(figsize=(3.5, 3.5), dpi=90).add_subplot(111)
    plt.imshow(samples, vmin=samples.min(), vmax=samples.max(),
               origin='lower', cmap=cmap)
    plt.axis('off')

    if has_title and title_str is not None:
        plt.title(title_str, fontsize=11)
    else:
        plt.title(u'', fontsize=11)

    if save_fig:
        plt.savefig(file_name + '.' + file_format, format=file_format,
                    dpi=dpi, transparent=True)

    if close_fig:
        plt.close()
    else:
        plt.show()


def plot_3d_dirac_samples(samples, projection_plane='xy',
                          cmap='gray', save_fig=False,
                          file_name='samples_3d_dirac',
                          file_format='pdf', dpi=300, close_fig=True,
                          has_title=False, title_str=None, nbins=None):
    """
    Plot 3D Dirac samples with 2D projections. We use maximum intensity projection (MIP).
    :param samples: samples of the 2D Dirac deltas. We assume the 3D data cube is arranged
            in such a way that
                axis0 - y axis
                axis1 - x axis
                axis2 - z axis
    :param projection_plane: the projection plane used for plotting. Possible options are
            'xy', 'yz', or 'xz'.
    :param cmap: colormap
    :param save_fig: whether to save figure or not
    :param file_name: file name used for saving
    :param file_format: figure file format
    :param dpi: dpi for saving the figure
    :param close_fig: close the figure or not
    :param has_title: whether the figure has title or not
    :param title_str: the title string
    :return:
    """
    plt.figure(figsize=(3.5, 3.5), dpi=90)
    ax = plt.axes([0.17, 0.15, 0.7, 0.7])
    abs_samples = np.abs(samples)
    sz0, sz1, sz2 = samples.shape
    samples_proj = np.empty((sz0, sz1))
    if projection_plane == 'xy':
        max_abs_idx = np.argmax(abs_samples, axis=2)
        for y_count in range(sz0):
            for x_count in range(sz1):
                samples_proj[y_count, x_count] = samples[y_count, x_count,
                                                         max_abs_idx[y_count, x_count]]
        if use_latex:
            plt.xlabel('$x$', fontsize=12)
            plt.ylabel('$y$', fontsize=12)
        else:
            plt.xlabel('x')
            plt.ylabel('y')

    elif projection_plane == 'yz':
        max_abs_idx = np.argmax(abs_samples, axis=1)
        for y_count in range(sz0):
            for z_count in range(sz2):
                samples_proj[y_count, z_count] = samples[y_count,
                                                         max_abs_idx[y_count, z_count],
                                                         z_count]
        if use_latex:
            plt.xlabel('$z$', fontsize=12)
            plt.ylabel('$y$', fontsize=12)
        else:
            plt.xlabel('z')
            plt.ylabel('y')

    elif projection_plane == 'xz':
        max_abs_idx = np.argmax(abs_samples, axis=0)
        for x_count in range(sz1):
            for z_count in range(sz2):
                samples_proj[x_count, z_count] = samples[max_abs_idx[x_count, z_count],
                                                         x_count, z_count]

        if use_latex:
            plt.xlabel('$z$', fontsize=12)
            plt.ylabel('$x$', fontsize=12)
        else:
            plt.xlabel('z')
            plt.ylabel('x')
    else:
        raise RuntimeError('Unknown projection type!')

    ax.imshow(samples_proj, origin='lower', cmap=cmap,
              vmin=samples.min(), vmax=samples.max())
    if nbins is not None:
        plt.locator_params(nbins=nbins)
    # plt.axis('off')

    if has_title and title_str is not None:
        plt.title(title_str, fontsize=12)
    else:
        plt.title(u'', fontsize=12)

    if save_fig:
        plt.savefig(file_name + '_{}.'.format(projection_plane) + file_format,
                    format=file_format, dpi=dpi, transparent=True)

    if close_fig:
        plt.close()
    else:
        plt.show()


def cubic_plot_diracs(loc_ref=None, amp_ref=None,
                      loc_recon=None, amp_recon=None,
                      label_ref='ground truth', label_recon='reconstruction',
                      legend_marker_scale=0.7, legend_loc=0,
                      marker_scale=1, marker_alpha=1,
                      taus=None, around_zero=False, axis_type='image',
                      xlim=None, ylim=None, zlim=None, use_scientific_axis=False,
                      save_fig=False, file_name='recon_3d_dirac',
                      file_format='pdf', dpi=300, close_fig=True,
                      has_title=False, title_str=None, title_fontsize=11):
    """
    Plot the results in the 3D Dirac reconstruction.
    :param loc_ref: an N by 3 matrix for the original Dirac locations (x, y, and z)
    :param amp_ref: original Dirac amplitudes
    :param loc_recon: an N by 3 matrix for the reconstructed Dirac locations (x, y, and z)
    :param amp_recon: reconstructed Dirac amplitudes
    :param label_ref: label for the original Dirac markers
    :param label_recon: label for the reconstructed Dirac markers
    :param legend_marker_scale: scaling factor for the marker size in the legend
    :param legend_loc: legend locations
    :param marker_scale: scale for the markers (in addition to the re-scaling
            based onthe amplitudes
    :param marker_alpha: transparency value of the markers
    :param xlim: x-axis range
    :param ylim: y-axis range
    :param save_fig: whether to save figure or not
    :param file_name: file name used for saving
    :param file_format: figure file format
    :param dpi: dpi for saving the figure
    :param close_fig: close the figure or not
    :param has_title: whether the figure has title or not
    :param title_str: the title string
    :return:
    """
    if loc_ref is not None and amp_ref is not None:
        ref_pt_available = True
        if len(loc_ref.shape) < 2:
            loc_ref = loc_ref[np.newaxis, :]
    else:
        ref_pt_available = False

    if loc_recon is not None and amp_recon is not None:
        recon_pt_available = True
        if len(loc_recon.shape) < 2:
            loc_recon = loc_recon[np.newaxis, :]
    else:
        recon_pt_available = False

    # plot
    ax = plt.figure().add_subplot(111, projection='3d')

    # normalization for plotting
    if ref_pt_available:
        amp_noramalization = np.max(np.abs(amp_ref))
    elif recon_pt_available:
        amp_noramalization = np.max(np.abs(amp_recon))
    else:
        amp_noramalization = 1

    if ref_pt_available:
        x_ref, y_ref, z_ref = loc_ref[:, 0], loc_ref[:, 1], loc_ref[:, 2]
        ax.scatter(x_ref, y_ref, z_ref,
                   s=np.abs(amp_ref) / amp_noramalization * 100 * marker_scale,
                   marker='o', edgecolors='k', linewidths=0.5,
                   alpha=marker_alpha, zorder=-10,
                   c='w', label=label_ref)

    if recon_pt_available:
        x_recon, y_recon, z_recon = loc_recon[:, 0], loc_recon[:, 1], loc_recon[:, 2]
        ax.scatter(x_recon, y_recon, z_recon,
                   s=np.abs(amp_recon) / amp_noramalization * 100 * marker_scale,
                   marker='*', edgecolors='k', linewidths=0.5,
                   alpha=marker_alpha, zorder=20,
                   c=np.tile([0.996, 0.410, 0.703], (x_recon.size, 1)),
                   label=label_recon)

    if has_title and ref_pt_available and recon_pt_available and \
            title_str is None and taus is not None:
        # compute reconstruction distance
        dist_recon = nd_distance(loc_ref, loc_recon, taus)[0]
        plt.title(u'average error = {0:.2e}'.format(dist_recon), fontsize=title_fontsize)
    elif has_title and title_str is not None:
        plt.title(title_str, fontsize=title_fontsize)
    else:
        plt.title(u'', fontsize=title_fontsize)

    if ref_pt_available or recon_pt_available:
        plt.legend(scatterpoints=1, loc=legend_loc, fontsize=10,
                   ncol=1, markerscale=legend_marker_scale,
                   handletextpad=0.1, columnspacing=0.1,
                   labelspacing=0.1, framealpha=0.75, frameon=True)

    if axis_type == 'equal' and taus is not None:
        ax.set_aspect('equal')
        max_range = np.max(np.array(taus)) * 0.5
        if around_zero:
            mid_x = mid_y = mid_z = 0
        else:
            mid_x = taus[0] * 0.5
            mid_y = taus[1] * 0.5
            mid_z = taus[2] * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        plt.axis(axis_type)

    if xlim is not None:
        ax.set_xlim((xlim[0], xlim[1]))
    elif taus is not None:
        if around_zero:
            ax.set_xlim((-0.5 * taus[0], 0.5 * taus[0]))
        else:
            ax.set_xlim((0, taus[0]))
    if ylim is not None:
        ax.set_ylim((ylim[0], ylim[1]))
    elif taus is not None:
        if around_zero:
            ax.set_ylim((-0.5 * taus[1], 0.5 * taus[1]))
        else:
            ax.set_ylim((0, taus[1]))
    if zlim is not None:
        ax.set_zlim((zlim[0], zlim[1]))
    elif taus is not None:
        if around_zero:
            ax.set_zlim((-0.5 * taus[2], 0.5 * taus[2]))
        else:
            ax.set_zlim((0, taus[2]))

    if use_scientific_axis:
        ax.ticklabel_format(scilimits=(0, 0), style='scientific',
                            useMathText=True, useOffset=True)

    if use_latex:
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$y$', fontsize=12)
        ax.set_zlabel('$z$', fontsize=12)
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    if save_fig:
        plt.savefig(file_name + '.' + file_format, format=file_format,
                    dpi=dpi, transparent=True)

    if close_fig:
        plt.close()
    else:
        plt.show()


def cubic_plot_diracs_plotly(loc_ref=None, amp_ref=None,
                             loc_recon=None, amp_recon=None,
                             taus=None, around_zero=False,
                             label_ref='ground truth', label_recon='reconstruction',
                             marker_scale=1, marker_alpha=1, fixed_marker_sz=False,
                             marker_color_varying=False,
                             auto_open=False, file_name='recon_3d_dirac.html',
                             has_title=False, title_str=None):
    """
    Plot the results in the 3D Dirac reconstruction with Plotly.
    :param loc_ref: an N by 3 matrix for the original Dirac locations (x, y, and z)
    :param amp_ref: original Dirac amplitudes
    :param loc_recon: an N by 3 matrix for the reconstructed Dirac locations (x, y, and z)
    :param amp_recon: reconstructed Dirac amplitudes
    :param label_ref: label for the original Dirac markers
    :param label_recon: label for the reconstructed Dirac markers
    :param legend_marker_scale: scaling factor for the marker size in the legend
    :param legend_loc: legend locations
    :param marker_scale: scale for the markers (in addition to the re-scaling
            based onthe amplitudes
    :param marker_alpha: transparency value of the markers
    :param has_title: whether the figure has title or not
    :param title_str: the title string
    :return:
    """
    if loc_ref is not None and amp_ref is not None:
        ref_pt_available = True
        if len(loc_ref.shape) < 2:
            loc_ref = loc_ref[np.newaxis, :]
    else:
        ref_pt_available = False

    if loc_recon is not None and amp_recon is not None:
        recon_pt_available = True
        if len(loc_recon.shape) < 2:
            loc_recon = loc_recon[np.newaxis, :]
    else:
        recon_pt_available = False

    # normalization for plotting
    if ref_pt_available:
        amp_noramalization = np.max(np.abs(amp_ref))
    elif recon_pt_available:
        amp_noramalization = np.max(np.abs(amp_recon))
    else:
        amp_noramalization = 1

    if has_title and ref_pt_available and recon_pt_available and \
            title_str is None and taus is not None:
        # compute reconstruction distance
        dist_recon = nd_distance(loc_ref, loc_recon, taus)[0]
        title_str = u'average error = {0:.1f} [nm]'.format(dist_recon * 1e9)

    if taus is not None and around_zero:
        scene_dict = {
            'xaxis': dict(nticks=4, range=[-0.5 * taus[0], 0.5 * taus[0]]),
            'yaxis': dict(nticks=4, range=[-0.5 * taus[1], 0.5 * taus[1]]),
            'zaxis': dict(nticks=4, range=[-0.5 * taus[2], 0.5 * taus[2]]),
            'aspectratio': dict(x=taus[0] / taus[2] * 0.3, y=taus[1] / taus[2] * 0.3, z=0.3),
            # 'camera': dict(eye=dict(x=0, y=0, z=1))
        }
    elif taus is not None and not around_zero:
        scene_dict = {
            'xaxis': dict(nticks=4, range=[0, taus[0]]),
            'yaxis': dict(nticks=4, range=[0, taus[1]]),
            'zaxis': dict(nticks=4, range=[0, taus[2]]),
            'aspectratio': dict(x=taus[0] / taus[2] * 0.3, y=taus[1] / taus[2] * 0.3, z=0.3),
            # 'camera': dict(eye=dict(x=0, y=0, z=1))
        }
    else:
        scene_dict = None

    layout = pg.Layout(
        margin=dict(l=25, r=0, b=0, t=25, pad=4),
        title=title_str,
        scene=scene_dict)

    if ref_pt_available:
        x_ref, y_ref, z_ref = loc_ref[:, 0], loc_ref[:, 1], loc_ref[:, 2]
        trace1 = pg.Scatter3d(
            x=x_ref, y=y_ref, z=z_ref, name=label_ref,
            mode='markers',
            marker=dict(
                size=marker_scale if fixed_marker_sz
                else np.abs(amp_ref) / amp_noramalization * 30 * marker_scale,
                symbol='circle-dot',
                color=z_ref if marker_color_varying else 'w',
                colorscale='magma',
                opacity=marker_alpha,
            ),
            text=['amplitude: {0:.2f}'.format(amp_ref_loop) for amp_ref_loop in amp_ref]
        )

    if recon_pt_available:
        x_recon, y_recon, z_recon = loc_recon[:, 0], loc_recon[:, 1], loc_recon[:, 2]
        trace2 = pg.Scatter3d(
            x=x_recon, y=y_recon, z=z_recon, name=label_recon,
            mode='markers',
            marker=dict(
                size=marker_scale if fixed_marker_sz
                else np.abs(amp_recon) / amp_noramalization * 21 * marker_scale,
                symbol='diamond',
                color=z_recon if marker_color_varying else 'rgb(254, 104, 179)',
                colorscale='magma',
                opacity=marker_alpha,
            ),
            text=['amplitude: {0:.2f}'.format(amp_recon_loop) for amp_recon_loop in amp_recon]
        )

    data = []
    if ref_pt_available:
        data.append(trace1)
    if recon_pt_available:
        data.append(trace2)

    fig = pg.Figure(data=data, layout=layout)
    offline.plot(fig, show_link=False, auto_open=auto_open, filename=file_name)
