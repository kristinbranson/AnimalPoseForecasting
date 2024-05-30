import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import torch

from config import (
    SENSORY_PARAMS, ARENA_RADIUS_MM,
    posenames, keypointnames, scalenames,
    featglobal, featrelative, kpvision_other,
    nglobal, nrelative, nkptouch
)
from data import get_batch_idx, load_and_filter_data, split_data_by_id, select_bin_edges
from features import get_sensory_feature_idx, zscore, unzscore
from models import criterion_wrapper
from utils import npindex


def select_featidx_plot(train_dataset, ntspred_plot, ntsplot_global=None, ntsplot_relative=None):
    if ntsplot_global is None:
        ntsplot_global = np.minimum(train_dataset.ntspred_global, ntspred_plot)
    if ntsplot_relative is None:
        ntsplot_relative = np.minimum(train_dataset.ntspred_relative, ntspred_plot)

    if ntsplot_global == 0:
        tidxplot_global = None
    elif ntsplot_global == 1:
        tidxplot_global = np.zeros((nglobal, 1), dtype=int)
    elif ntsplot_global == train_dataset.ntspred_global:
        tidxplot_global = np.tile(np.arange(ntsplot_global, dtype=int)[None, :], (nglobal, 1))
    else:
        # choose 0 + a variety of different timepoints for each global feature so that a variety of timepoints are selected
        tidxplot_global = np.concatenate((np.zeros((nglobal, 1), dtype=int),
                                          np.round(np.linspace(1, train_dataset.ntspred_global - 1,
                                                               (ntsplot_global - 1) * nglobal)).astype(int).reshape(-1,
                                                                                                                    nglobal).T),
                                         axis=-1)
    if ntsplot_relative == 0:
        tsplot_relative = None
    elif ntsplot_relative == 1:
        tsplot_relative = np.ones((train_dataset.nrelrep, 1), dtype=int)
    elif ntsplot_relative == train_dataset.ntspred_relative:
        tsplot_relative = np.tile(np.arange(ntsplot_relative, dtype=int)[None, :] + 1, (train_dataset.nrelrep, 1))
    else:
        # choose 0 + a variety of different timepoints for each feature so that a variety of timepoints are selected
        tsplot_relative = np.concatenate((np.zeros((train_dataset.nrelrep, 1), dtype=int),
                                          np.round(np.linspace(1, train_dataset.ntspred_relative - 1,
                                                               (ntsplot_relative - 1) * nrelative)).astype(int).reshape(
                                              -1, train_dataset.nrelrep).T), axis=-1)
    ftidx = []
    for fi, f in enumerate(featglobal):
        for ti in range(ntsplot_global):
            ftidx.append((f, train_dataset.tspred_global[tidxplot_global[fi, ti]]))
    for fi, f in enumerate(np.nonzero(train_dataset.featrelative)[0]):
        for ti in range(ntsplot_relative):
            ftidx.append((f, tsplot_relative[fi, ti]))
    featidxplot = train_dataset.ravel_label_index(ftidx)
    return featidxplot


def get_n_colors_from_colormap(colormapname, n):
    # Get the colormap from matplotlib
    colormap = plt.cm.get_cmap(colormapname)

    # Generate a range of values from 0 to 1
    values = np.linspace(0, 1, n)

    # Get 'n' colors from the colormap
    colors = colormap(values)

    return colors


def set_fig_ax(fig=None, ax=None):
    """
    fig,ax,isnewaxis = set_fig_ax(fig=None,ax=None)
    Create new figure and/or axes if those are not input.
    Returns the handles to those figures and axes.
    isnewaxis is whether a new set of axes was created.
    """
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        isnewaxis = True
    else:
        isnewaxis = False
    return fig, ax, isnewaxis


def get_Dark3_cmap():
    """
    dark3cm = get_Dark3_cmap()
    Returns a new matplotlib colormap based on the Dark2 colormap.
    I didn't have quite enough unique colors in Dark2, so I made Dark3 which
    is Dark2 followed by all Dark2 colors with the same hue and saturation but
    half the brightness.
    """


    dark2 = list(cm.get_cmap('Dark2').colors)
    dark3 = dark2.copy()
    for c in dark2:
        chsv = colors.rgb_to_hsv(c)
        chsv[2] = chsv[2] / 2.
        crgb = colors.hsv_to_rgb(chsv)
        dark3.append(crgb)
    dark3cm = colors.ListedColormap(tuple(dark3))
    return dark3cm


def plot_fly(self, x=None, idx=None, f=None, **kwargs):
    """
    hkpt,hedge,htxt,fig,ax = data.plot_fly(pose=None, idx=None, f=None,
                                          fig=None, ax=None, kptcolors=None, color=None, name=None,
                                          plotskel=True, plotkpts=True, hedge=None, hkpt=None)
    Visualize the single fly position specified by pose
    Inputs:
    pose: Optional. nfeatures x 2 ndarray. Default is None.
    idx: Data sample index to plot. Only used if pose is not input. Default: None.
    f: Frame of data sample idx to plot. Only used if pose is not input. Default: dataset.ctrf.
    fig: Optional. Handle to figure to plot in. Only used if ax is not specified. Default = None.
    ax: Optional. Handle to axes to plot in. Default = None.
    kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
    colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to 'hsv'.
    Default: None
    color: Optional. Color for edges plotted. If None, it is set to [.6,.6,.6]. efault: None.
    name: Optional. String defining an identifying label for these plots. Default None.
    plotskel: Optional. Whether to plot skeleton edges. Default: True.
    plotkpts: Optional. Whether to plot key points. Default: True.
    hedge: Optional. Handle of edges to update instead of plot new edges. Default: None.
    hkpt: Optional. Handle of keypoints to update instead of plot new key points. Default: None.
    """
    if x is not None:
        return plot_fly(pose=self.unnormalize(x), kptidx=self.keypointidx, skelidx=self.skeleton_edges, **kwargs)
    else:
        assert (idx is not None)
        if f is None:
            f = self.ctrf
        x = self.getitem(idx)
        return plot_fly(pose=self.unnormalize(x['x'][:, :, f, 0]), kptidx=self.keypointidx, skelidx=self.skeleton_edges,
                        **kwargs)


def plot_flies(poses=None, fig=None, ax=None, colors=None, kptcolors=None, hedges=None, hkpts=None, htxt=None,
               textlabels=None, skeledgecolors=None, **kwargs):
    """
    hkpt,hedge,fig,ax = plot_flies(poses=None, kptidx=None, skelidx=None,
                                   colors=None,kptcolors=None,hedges=None,hkpts=None,
                                   **kwargs)
    Visualize all flies for a single frame specified by poses.
    Inputs:
    poses: Required. nfeatures x 2 x nflies ndarray.
    kptidx: Required. 1-dimensional array specifying which keypoints to plot
    skelidx: Required. nedges x 2 ndarray specifying which keypoints to connect with edges
    colors: Optional. Color scheme for edges plotted for each fly. Can be a string defining a matplotlib
    colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to the Dark3
    colormap I defined in get_Dark3_cmap(). Default: None.
    kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
    colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to [0,0,0].
    Default: None
    hedges: Optional. List of handles of edges, one per fly, to update instead of plot new edges. Default: None.
    hkpts: Optional. List of handles of keypoints, one per fly,  to update instead of plot new key points.
    Default: None.
    Extra arguments: All other arguments will be passed directly to plot_fly.
    """
    fig, ax, isnewaxis = set_fig_ax(fig=fig, ax=ax)
    if colors is None and skeledgecolors is None:
        colors = get_Dark3_cmap()
    if kptcolors is None:
        kptcolors = [0, 0, 0]
    nflies = poses.shape[-1]
    if colors is not None and (not (type(colors) == list or type(colors) == np.ndarray)):
        if type(colors) == str:
            cmap = cm.get_cmap(colors)
        else:
            cmap = colors
        colors = cmap(np.linspace(0., 1., nflies))

    if hedges is None:
        hedges = [None, ] * nflies
    if hkpts is None:
        hkpts = [None, ] * nflies
    if htxt is None:
        htxt = [None, ] * nflies

    textlabels1 = textlabels

    for fly in range(nflies):
        if not (textlabels is None or (textlabels == 'keypoints')):
            textlabels1 = '%d' % fly
        if skeledgecolors is not None:
            colorcurr = skeledgecolors
        else:
            colorcurr = colors[fly, ...]
        hkpts[fly], hedges[fly], htxt[fly], fig, ax = plot_fly(poses[..., fly], fig=fig, ax=ax, color=colorcurr,
                                                               kptcolors=kptcolors, hedge=hedges[fly], hkpt=hkpts[fly],
                                                               htxt=htxt, textlabels=textlabels1, **kwargs)

    if isnewaxis:
        ax.axis('equal')

    return hkpts, hedges, htxt, fig, ax


def plot_arena(ax=None):
    if ax is None:
        ax = plt.gca()
    theta = np.linspace(0, 2 * np.pi, 360)
    h = ax.plot(ARENA_RADIUS_MM * np.cos(theta), ARENA_RADIUS_MM * np.sin(theta), 'k-', zorder=-10)
    return h


def plot_scale_stuff(data, scale_perfly):
    eps_sex = .05
    nbins = 20
    axlim_prctile = .5
    catidx = data['categories'].index('female')

    maxid = np.max(data['ids'])
    maxnflies = data['X'].shape[3]
    fracfemale = np.zeros(maxid + 1)
    nframes = np.zeros(maxid + 1)
    minnframes = 40000
    prctiles_compute = np.array([50, 75, 90, 95, 99, 99.5, 99.9])
    midleglength = np.zeros((maxid + 1, len(prctiles_compute)))

    for flynum in range(maxnflies):

        idscurr = np.unique(data['ids'][data['ids'][:, flynum] >= 0, flynum])
        for id in idscurr:
            idx = data['ids'][:, flynum] == id
            fracfemale[id] = np.count_nonzero(data['y'][catidx, idx, flynum] == 1) / np.count_nonzero(idx)
            nframes[id] = np.count_nonzero(idx)
            xcurr = data['X'][:, :, idx, flynum]
            midtip = xcurr[keypointnames.index('left_middle_leg_tip'), :]
            midbase = xcurr[keypointnames.index('left_middle_femur_base'), :]
            lmidl = np.sqrt(np.sum((midtip - midbase) ** 2, axis=0))
            midtip = xcurr[keypointnames.index('right_middle_leg_tip'), :]
            midbase = xcurr[keypointnames.index('right_middle_femur_base'), :]
            rmidl = np.sqrt(np.sum((midtip - midbase) ** 2, axis=0))
            midleglength[id, :] = np.percentile(np.hstack((lmidl, rmidl)), prctiles_compute)

    plotnames = ['thorax_width', 'thorax_length', 'abdomen_length', 'head_width', 'head_height']
    plotidx = np.array([v in plotnames for v in scalenames])
    plotidx = np.nonzero(plotidx)[0]
    plotfly = nframes >= minnframes
    fig, ax = plt.subplots(len(plotidx), len(plotidx))
    fig.set_figheight(20)
    fig.set_figwidth(20)

    idxfemale = plotfly & (fracfemale >= 1 - eps_sex)
    idxmale = plotfly & (fracfemale <= eps_sex)

    lims = np.percentile(scale_perfly[:, plotfly], [axlim_prctile, 100 - axlim_prctile], axis=1)

    for ii in range(len(plotidx)):
        i = plotidx[ii]
        for jj in range(len(plotidx)):
            j = plotidx[jj]
            if i == j:
                binedges = np.linspace(lims[0, i], lims[1, i], nbins + 1)
                ax[ii, ii].hist([scale_perfly[i, idxfemale], scale_perfly[i, idxmale]],
                                bins=nbins, range=(lims[0, i], lims[1, i]),
                                label=['female', 'male'])
                ax[ii, ii].set_ylabel('N. flies')
            else:
                ax[jj, ii].plot(scale_perfly[i, idxfemale],
                                scale_perfly[j, idxfemale], '.', label='female')
                ax[jj, ii].plot(scale_perfly[i, idxmale],
                                scale_perfly[j, idxmale], '.', label='male')
                ax[jj, ii].set_ylabel(scalenames[j])
                ax[jj, ii].set_xlabel(scalenames[i])
                ax[jj, ii].set_ylim(lims[:, j])
            ax[jj, ii].set_xlim(lims[:, i])
            ax[jj, ii].set_xlabel(scalenames[i])
    ax[0, 0].legend()
    ax[0, 1].legend()
    fig.tight_layout()

    scalefeat = 'thorax_length'
    scalei = scalenames.index(scalefeat)
    fig, ax = plt.subplots(2, len(prctiles_compute), sharex='row', sharey='row')
    fig.set_figwidth(20)
    fig.set_figheight(8)
    lims = np.percentile(midleglength[plotfly, :].flatten(), [axlim_prctile, 100 - axlim_prctile])
    for i in range(len(prctiles_compute)):
        ax[0, i].plot(scale_perfly[scalei, idxfemale], midleglength[idxfemale, i], '.', label='female')
        ax[0, i].plot(scale_perfly[scalei, idxmale], midleglength[idxmale, i], '.', label='male')
        ax[0, i].set_xlabel(scalefeat)
        ax[0, i].set_ylabel(f'{prctiles_compute[i]}th %ile middle leg length')
        ax[1, i].hist([midleglength[idxfemale, i], midleglength[idxmale, i]],
                      bins=nbins, range=(lims[0], lims[1]), label=['female', 'male'],
                      density=True)
        ax[1, i].set_xlabel(f'{prctiles_compute[i]}th %ile middle leg length')
        ax[1, i].set_ylabel('Density')
    ax[0, 0].legend()
    ax[1, 0].legend()
    fig.tight_layout()

    return


def debug_plot_otherflies_vision(t, xother, yother, xeye_main, yeye_main, theta_main,
                                 angle0, angle, dist, b_all, otherflies_vision, params):
    npts = xother.shape[0]
    nflies = xother.shape[1]

    rplot = 2 * params['outer_arena_radius']
    plt.figure()
    ax = plt.subplot(1, 3, 1)
    hother = ax.plot(xother[:, :, t], yother[:, :, t], '.-')
    ax.set_aspect('equal')
    # ax.plot(X[:,0,0,flynum],X[:,1,0,flynum],'k.')
    ax.plot(xeye_main[0, 0, t], yeye_main[0, 0, t], 'r.')
    ax.plot([xeye_main[0, 0, t], xeye_main[0, 0, t] + rplot * np.cos(theta_main[0, 0, t])],
            [yeye_main[0, 0, t], yeye_main[0, 0, t] + rplot * np.sin(theta_main[0, 0, t])], 'r--')
    for tmpfly in range(nflies):
        ax.plot(xeye_main[0, 0, t] + np.c_[np.zeros((npts, 1)), np.cos(angle0[:, tmpfly, t]) * rplot].T,
                yeye_main[0, 0, t] + np.c_[np.zeros((npts, 1)), np.sin(angle0[:, tmpfly, t]) * rplot].T,
                color=hother[tmpfly].get_color(), alpha=.5)

    ax = plt.subplot(1, 3, 2)
    for tmpfly in range(nflies):
        ax.plot(np.c_[np.zeros((npts, 1)), np.cos(angle[:, tmpfly, t])].T,
                np.c_[np.zeros((npts, 1)), np.sin(angle[:, tmpfly, t])].T,
                color=hother[tmpfly].get_color(), alpha=.5)
    ax.plot(0, 0, 'r.')
    ax.plot([0, 1], [0, 0], 'r--')
    ax.set_aspect('equal')

    ax = plt.subplot(1, 3, 3)
    for tmpfly in range(nflies):
        ax.plot(b_all[:, tmpfly, t], dist[:, tmpfly, t], 'o', color=hother[tmpfly].get_color())
    ax.set_xlim([-.5, params['n_oma'] - .5])
    ax.set_xlabel('bin')
    ax.set_ylabel('dist')

    tmpvision = np.minimum(50, otherflies_vision[:, t])
    ax.plot(tmpvision, 'k-')


def debug_plot_wall_touch(t, xwall, ywall, distleg, wall_touch):
    plt.figure()
    plt.clf()
    ax = plt.subplot(1, 2, 1)
    ax.plot(xwall.flatten(), ywall.flatten(), 'k.')
    theta_arena = np.linspace(-np.pi, np.pi, 100)
    ax.plot(np.cos(theta_arena) * SENSORY_PARAMS['inner_arena_radius'],
            np.sin(theta_arena) * SENSORY_PARAMS['inner_arena_radius'], '-')
    ax.plot(np.cos(theta_arena) * SENSORY_PARAMS['outer_arena_radius'],
            np.sin(theta_arena) * SENSORY_PARAMS['outer_arena_radius'], '-')
    hpts = []
    for pti in range(nkptouch):
        hpts.append(ax.plot(xwall[pti, t], ywall[pti, t], 'o')[0])
    ax.set_aspect('equal')
    ax = plt.subplot(1, 2, 2)
    ax.plot(distleg.flatten(), wall_touch.flatten(), 'k.')
    ax.plot([0, SENSORY_PARAMS['inner_arena_radius'], SENSORY_PARAMS['outer_arena_radius']],
            [SENSORY_PARAMS['arena_height'], SENSORY_PARAMS['arena_height'], 0], '-')
    for pti in range(nkptouch):
        ax.plot(distleg[pti, t], wall_touch[pti, t], 'o', color=hpts[pti].get_color())
    ax.set_aspect('equal')


def debug_plot_compute_features(X, porigin, theta, Xother, Xnother):
    t = 0
    rplot = 5.
    plt.clf()
    ax = plt.subplot(1, 2, 1)
    plot_flies(X[:, :, t, :], ax=ax, textlabels='fly', colors=np.zeros((X.shape[-1], 3)))
    ax.plot(porigin[0, t], porigin[1, t], 'rx', linewidth=2)
    ax.plot([porigin[0, t, 0], porigin[0, t, 0] + np.cos(theta[t, 0]) * rplot],
            [porigin[1, t, 0], porigin[1, t, 0] + np.sin(theta[t, 0]) * rplot], 'r-')
    ax.plot(Xother[kpvision_other, 0, t, :], Xother[kpvision_other, 1, t, :], 'o')
    ax.set_aspect('equal')

    ax = plt.subplot(1, 2, 2)
    ax.plot(0, 0, 'rx')
    ax.plot([0, np.cos(0.) * rplot], [0, np.sin(0.) * rplot], 'r-')
    ax.plot(Xnother[:, 0, t, :], Xnother[:, 1, t, :], 'o')
    ax.set_aspect('equal')


def debug_plot_batch_state(stateprob, nsamplesplot=3,
                           h=None, ax=None, fig=None):
    batch_size = stateprob.shape[0]

    samplesplot = np.round(np.linspace(0, batch_size - 1, nsamplesplot)).astype(int)

    if ax is None:
        fig, ax = plt.subplots(nsamplesplot, 1)
    if h is None:
        h = [None, ] * nsamplesplot

    for i in range(nsamplesplot):
        iplot = samplesplot[i]
        if h[i] is None:
            h[i] = ax[i].imshow(stateprob[iplot, :, :].T, vmin=0., vmax=1.)
        else:
            h[i].set_data(stateprob[iplot, :, :].T)
        ax[i].axis('auto')

    fig.tight_layout(h_pad=0)
    return h, ax, fig


def debug_plot_batch_traj(example, train_dataset, criterion=None, config=None,
                          pred=None, data=None, nsamplesplot=3,
                          true_discrete_mode='to_discretize',
                          pred_discrete_mode='sample',
                          h=None, ax=None, fig=None, label_true='True', label_pred='Pred',
                          ntspred_plot=2):
    batch_size = example['input'].shape[0]
    contextl = train_dataset.ntimepoints

    true_color = [0, 0, 0]
    true_color_y = [.5, .5, .5]
    pred_cmap = lambda x: plt.get_cmap("tab10")(x % 10)

    if train_dataset.ismasked():
        mask = example['mask']
    else:
        mask = None

    if ax is None:
        fig, ax = plt.subplots(1, nsamplesplot, squeeze=False)
        ax = ax[0, :]

    if true_discrete_mode == 'to_discretize':
        true_args = {'use_todiscretize': True}
    elif true_discrete_mode == 'sample':
        true_args = {'sample': True}
    else:
        true_args = {}

    if pred_discrete_mode == 'sample':
        pred_args = {'sample': True}
    else:
        pred_args = {}

    featidxplot = select_featidx_plot(train_dataset, ntspred_plot)
    samplesplot = np.round(np.linspace(0, batch_size - 1, nsamplesplot)).astype(int)
    for i in range(nsamplesplot):
        iplot = samplesplot[i]
        examplecurr = get_batch_idx(example, iplot)
        zmovement_continuous_true, zmovement_discrete_true = train_dataset.get_continuous_discrete_labels(examplecurr)
        ntimepoints = train_dataset.ntimepoints

        # zmovement_true = train_dataset.get_full_labels(examplecurr,**true_args)
        # zmovement_true = example['labels'][iplot,...].numpy()
        # err_movement = None
        err_total = None
        if mask is not None:
            maskcurr = np.zeros(mask.shape[-1] + 1, dtype=bool)
            maskcurr[:-1] = mask[iplot, ...].numpy()
            maskidx = np.nonzero(maskcurr)[0]
        if pred is not None:
            predcurr = get_batch_idx(pred, iplot)
            zmovement_continuous_pred, zmovement_discrete_pred = train_dataset.get_continuous_discrete_labels(predcurr)
            # zmovement_pred = train_dataset.get_full_labels(predcurr,**pred_args)
            # if mask is not None:
            #   nmask = np.count_nonzero(maskcurr)
            # else:
            #   nmask = np.prod(zmovement_continuous_pred.shape[:-1])
            if criterion is not None:
                err_total, err_discrete, err_continuous = criterion_wrapper(examplecurr, predcurr, criterion,
                                                                            train_dataset, config)
            zmovement_discrete_pred = torch.softmax(zmovement_discrete_pred, dim=-1)
            # err_movement = torch.abs(zmovement_true[maskidx,:]-zmovement_pred[maskidx,:])/nmask
            # err_total = torch.sum(err_movement).item()/d

        elif data is not None:
            # t0 = example['metadata']['t0'][iplot].item()
            # flynum = example['metadata']['flynum'][iplot].item()
            zmovement_pred = None
        else:
            # Xcenter_pred = None
            # Xtheta_pred = None
            zmovement_pred = None

        mult = 6.
        d = len(featidxplot)
        outnames = train_dataset.get_outnames()
        contextl = train_dataset.ntimepoints

        ax[i].cla()

        for featii, feati in enumerate(featidxplot):
            featidx = np.nonzero(train_dataset.discrete_idx == feati)[0]
            if len(featidx) == 0:
                continue
            featidx = featidx[0]
            im = np.ones((train_dataset.discretize_nbins, contextl, 3))
            ztrue = zmovement_discrete_true[:, featidx, :].cpu().T
            ztrue = ztrue - torch.min(ztrue)
            ztrue = ztrue / torch.max(ztrue)
            im[:, :, 0] = 1. - ztrue
            if pred is not None:
                zpred = zmovement_discrete_pred[:, featidx, :].detach().cpu().T
                zpred = zpred - torch.min(zpred)
                zpred = zpred / torch.max(zpred)
                im[:, :, 1] = 1. - zpred
            ax[i].imshow(im, extent=(0, contextl, (featii - .5) * mult, (featii + .5) * mult), origin='lower',
                         aspect='auto')

        for featii, feati in enumerate(featidxplot):
            featidx = np.nonzero(train_dataset.continuous_idx == feati)[0]
            if len(featidx) == 0:
                continue
            featidx = featidx[0]
            ax[i].plot([0, contextl], [mult * featii, ] * 2, ':', color=[.5, .5, .5])
            ax[i].plot(mult * featii + zmovement_continuous_true[:, featidx], '-', color=true_color,
                       label=f'{outnames[feati]}, true')
            if mask is not None:
                ax[i].plot(maskidx, mult * featii + zmovement_continuous_true[maskcurr[:-1], featidx], 'o',
                           color=true_color, label=f'{outnames[feati]}, true')

            labelcurr = outnames[feati]
            if pred is not None:
                h = ax[i].plot(mult * featii + zmovement_continuous_pred[:, featidx], '--',
                               label=f'{outnames[feati]}, pred', color=pred_cmap(featii))
                if mask is not None:
                    ax[i].plot(maskidx, mult * featii + zmovement_continuous_pred[maskcurr[:-1], featidx], 'o',
                               color=pred_cmap(featii), label=f'{outnames[feati]}, pred')

        for featii, feati in enumerate(featidxplot):
            labelcurr = outnames[feati]
            ax[i].text(0, mult * (featii + .5), labelcurr, horizontalalignment='left', verticalalignment='top')

        if (err_total is not None):
            if train_dataset.discretize:
                ax[i].set_title(
                    f'Err: {err_total.item(): .2f}, disc: {err_discrete.item(): .2f}, cont: {err_continuous.item(): .2f}')
            else:
                ax[i].set_title(f'Err: {err_total.item(): .2f}')
        ax[i].set_xlabel('Frame')
        ax[i].set_ylabel('Movement')
        ax[i].set_ylim([-mult, mult * d])

    # fig.tight_layout()

    return ax, fig


def debug_plot_pose_prob(example, train_dataset, predcpu, tplot, fig=None, ax=None, h=None, minalpha=.25):
    batch_size = predcpu['stateprob'].shape[0]
    contextl = predcpu['stateprob'].shape[1]
    nstates = predcpu['stateprob'].shape[2]
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    Xkp_true = train_dataset.get_Xkp(example['input'][0, ...].numpy(),
                                     example['init'].numpy(),
                                     example['labels'][:tplot + 1, ...].numpy(),
                                     example['scale'].numpy())
    Xkp_true = Xkp_true[..., 0]

    order = torch.argsort(predcpu['stateprob'][0, tplot, :])
    rank = torch.argsort(order)
    labels = example['labels'][:tplot, :]
    state_cmap = lambda x: plt.get_cmap("tab10")(rank[x] % 10)

    if h is None:
        h = {'kpt_true': None, 'kpt_state': [None, ] * nstates,
             'edge_true': None, 'edge_state': [None, ] * nstates}
    h['kpt_true'], h['edge_true'], _, _, _ = plot_fly(Xkp_true[:, :, -1],
                                                      skel_lw=2, color=[0, 0, 0],
                                                      ax=ax, hkpt=h['kpt_true'], hedge=h['edge_true'])
    for i in range(nstates):
        labelspred = torch.cat((labels, predcpu['perstate'][0, [tplot, ], :, i]), dim=0)

        Xkp_pred = train_dataset.get_Xkp(example['input'][0, ...].numpy(),
                                         example['init'].numpy(),
                                         labelspred,
                                         example['scale'].numpy())
        Xkp_pred = Xkp_pred[..., 0]
        p = predcpu['stateprob'][0, tplot, i].item()
        alpha = minalpha + p * (1 - minalpha)
        color = state_cmap(i)
        h['kpt_state'][i], h['edge_state'][i], _, _, _ = plot_fly(Xkp_pred[:, :, -1],
                                                                  skel_lw=2, color=color,
                                                                  ax=ax, hkpt=h['kpt_state'][i],
                                                                  hedge=h['edge_state'][i])
        h['edge_state'][i].set_alpha(alpha)
        h['kpt_state'][i].set_alpha(alpha)

    return h, ax, fig


def debug_plot_batch_pose(example, train_dataset, pred=None, data=None,
                          true_discrete_mode='to_discretize',
                          pred_discrete_mode='sample',
                          ntsplot=5, nsamplesplot=3, h=None, ax=None, fig=None,
                          tsplot=None):
    batch_size = example['input'].shape[0]
    contextl = train_dataset.ntimepoints
    nsamplesplot = np.minimum(nsamplesplot, batch_size)

    if tsplot is None:
        tsplot = np.round(np.linspace(0, contextl, ntsplot)).astype(int)
    else:
        ntsplot = len(tsplot)
    samplesplot = np.round(np.linspace(0, batch_size - 1, nsamplesplot)).astype(int)

    if ax is None:
        fig, ax = plt.subplots(nsamplesplot, ntsplot, squeeze=False)

    if h is None:
        h = {'kpt0': [], 'kpt1': [], 'edge0': [], 'edge1': []}

    if true_discrete_mode == 'to_discretize':
        true_args = {'use_todiscretize': True}
    elif true_discrete_mode == 'sample':
        true_args = {'sample': True}
    else:
        true_args = {}

    if pred_discrete_mode == 'sample':
        pred_args = {'sample': True}
    else:
        pred_args = {}

    for i in range(nsamplesplot):
        iplot = samplesplot[i]
        examplecurr = get_batch_idx(example, iplot)
        Xkp_true = train_dataset.get_Xkp(examplecurr, **true_args)
        # ['input'][iplot,0,...].numpy(),
        #                            example['init'][iplot,...].numpy(),
        #                            example['labels'][iplot,...].numpy(),
        #                            example['scale'][iplot,...].numpy())
        Xkp_true = Xkp_true[..., 0]
        t0 = example['metadata']['t0'][iplot].item()
        flynum = example['metadata']['flynum'][iplot].item()
        if pred is not None:
            predcurr = get_batch_idx(pred, iplot)
            Xkp_pred = train_dataset.get_Xkp(examplecurr, pred=predcurr, **pred_args)
            Xkp_pred = Xkp_pred[..., 0]
        elif data is not None:
            Xkp_pred = data['X'][:, :, t0:t0 + contextl + 1, flynum]
        else:
            Xkp_pred = None
        for key in h.keys():
            if len(h[key]) <= i:
                h[key].append([None, ] * ntsplot)

        minxy = np.nanmin(np.nanmin(Xkp_true[:, :, tsplot], axis=0), axis=-1)
        maxxy = np.nanmax(np.nanmax(Xkp_true[:, :, tsplot], axis=0), axis=-1)
        if Xkp_pred is not None:
            minxy_pred = np.nanmin(np.nanmin(Xkp_pred[:, :, tsplot], axis=0), axis=-1)
            maxxy_pred = np.nanmax(np.nanmax(Xkp_pred[:, :, tsplot], axis=0), axis=-1)
            minxy = np.minimum(minxy, minxy_pred)
            maxxy = np.maximum(maxxy, maxxy_pred)
        for j in range(ntsplot):
            tplot = tsplot[j]
            if j == 0:
                ax[i, j].set_title(f'fly: {flynum} t0: {t0}')
            else:
                ax[i, j].set_title(f't = {tplot}')

            h['kpt0'][i][j], h['edge0'][i][j], _, _, _ = plot_fly(Xkp_true[:, :, tplot],
                                                                  skel_lw=2, color=[0, 0, 0],
                                                                  ax=ax[i, j], hkpt=h['kpt0'][i][j],
                                                                  hedge=h['edge0'][i][j])
            if Xkp_pred is not None:
                h['kpt1'][i][j], h['edge1'][i][j], _, _, _ = plot_fly(Xkp_pred[:, :, tplot],
                                                                      skel_lw=1, color=[0, 1, 1],
                                                                      ax=ax[i, j], hkpt=h['kpt1'][i][j],
                                                                      hedge=h['edge1'][i][j])
            ax[i, j].set_aspect('equal')
            # minxy = np.nanmin(Xkp_true[:,:,tplot],axis=0)
            # maxxy = np.nanmax(Xkp_true[:,:,tplot],axis=0)
            # if Xkp_pred is not None:
            #   minxy_pred = np.nanmin(Xkp_pred[:,:,tplot],axis=0)
            #   maxxy_pred = np.nanmax(Xkp_pred[:,:,tplot],axis=0)
            #   minxy = np.minimum(minxy,minxy_pred)
            #   maxxy = np.maximum(maxxy,maxxy_pred)
            ax[i, j].set_xlim([minxy[0], maxxy[0]])
            ax[i, j].set_ylim([minxy[1], maxxy[1]])

    return h, ax, fig


def debug_plot_sample_inputs(dataset, example, nplot=3):
    nplot = np.minimum(nplot, example['input'].shape[0])

    fig, ax = plt.subplots(nplot, 2, squeeze=False)
    idx = get_sensory_feature_idx(dataset.simplify_in)
    labels = dataset.get_full_labels(example=example, use_todiscretize=True)
    nlabels = labels.shape[-1]
    inputs = dataset.get_full_inputs(example=example, use_stacked=False)

    inputidxstart = [x[0] - .5 for x in idx.values()]
    inputidxtype = list(idx.keys())

    for iplot in range(nplot):
        ax[iplot, 0].cla()
        ax[iplot, 1].cla()
        ax[iplot, 0].imshow(inputs[iplot, ...], vmin=-3, vmax=3, cmap='coolwarm', aspect='auto')
        ax[iplot, 0].set_title(f'Input {iplot}')
        # ax[iplot,0].set_xticks(inputidxstart)
        for j in range(len(inputidxtype)):
            ax[iplot, 0].plot([inputidxstart[j], ] * 2, [-.5, inputs.shape[1] - .5], 'k-')
            ax[iplot, 0].text(inputidxstart[j], inputs.shape[1] - 1, inputidxtype[j], horizontalalignment='left')
        lastidx = list(idx.values())[-1][1]
        ax[iplot, 0].plot([lastidx - .5, ] * 2, [-.5, inputs.shape[1] - .5], 'k-')

        # ax[iplot,0].set_xticklabels(inputidxtype)
        ax[iplot, 1].imshow(labels[iplot, ...], vmin=-3, vmax=3, cmap='coolwarm', aspect='auto')
        ax[iplot, 1].set_title(f'Labels {iplot}')
    return fig, ax


def debug_plot_predictions_vs_labels(predv, labelsv, pred_discretev=None, labels_discretev=None, outnames=None,
                                     maskidx=None, ax=None, prctile_lim=.1, naxc=1, featidxplot=None, dataset=None):
    d_output = predv.shape[-1]

    if featidxplot is not None:
        predv = predv[:, featidxplot]
        labelsv = labelsv[:, featidxplot]
        outnames = [outnames[i] for i in featidxplot]
        if labels_discretev is not None and len(labels_discretev) > 0:
            tmp = torch.zeros((labels_discretev.shape[0], d_output, labels_discretev.shape[2]),
                              dtype=labels_discretev.dtype)
            tmp[:] = np.nan
            tmp[:, dataset.discrete_idx, :] = labels_discretev
            labels_discretev = tmp[:, featidxplot, :]
            tmp = torch.zeros((labels_discretev.shape[0], d_output, labels_discretev.shape[2]),
                              dtype=labels_discretev.dtype)
            tmp[:] = np.nan
            tmp[:, dataset.discrete_idx, :] = pred_discretev
            pred_discretev = tmp[:, featidxplot, :]
        else:
            labels_discretev = None
            pred_discretev = None
        d_output = len(featidxplot)

    ismasked = maskidx is not None and len(maskidx) > 0
    naxr = int(np.ceil(d_output / naxc))
    if ax is None:
        fig, ax = plt.subplots(naxr, naxc, sharex='all', figsize=(20, 20))
        ax = ax.flatten()
        plt.tight_layout(h_pad=0)

    pred_cmap = lambda x: plt.get_cmap("tab10")(x % 10)
    for i in range(d_output):
        ax[i].cla()
        if outnames is not None:
            ti = ax[i].set_title(outnames[i], y=1.0, pad=-14, color=pred_cmap(i), loc='left')

        if pred_discretev is not None and len(pred_discretev) > 0 and (
                not torch.all(torch.isnan(pred_discretev[:, i, :]))):
            predcurr = pred_discretev[:, i, :].T
            labelscurr = labels_discretev[:, i, :].T
            zlabels = torch.max(labelscurr[torch.isnan(labelscurr) == False])
            zpred = torch.max(predcurr[torch.isnan(predcurr) == False])
            im = torch.stack((1 - labelscurr / zlabels, 1 - predcurr / zpred, torch.ones(predcurr.shape)), dim=-1)
            im[torch.isnan(im)] = 1.
            ax[i].imshow(im.numpy(), aspect='auto')
        else:
            lims = torch.quantile(labelsv[:, i][torch.isnan(labelsv[:, i]) == False],
                                  torch.tensor([prctile_lim / 100., 1 - prctile_lim / 100.]))
            ax[i].plot(labelsv[:, i], 'k-', label='True')
            if ismasked:
                ax[i].plot(maskidx, predv[maskidx, i], '.', color=pred_cmap(i), label='Pred')
            else:
                ax[i].plot(predv[:, i], '-', color=pred_cmap(i), label='Pred')
            # ax[i].set_ylim([-ylim_nstd,ylim_nstd])
            ax[i].set_ylim(lims)
            if outnames is not None:
                plt.setp(ti, color=pred_cmap(i))
    ax[0].set_xlim([0, labelsv.shape[0]])

    return fig, ax


def debug_plot_histograms(dataset, alpha=1):
    r = np.random.rand(dataset.discretize_bin_samples.shape[0]) - .5
    ftidx = dataset.unravel_label_index(dataset.discrete_idx)
    # ftidx[featrelative[ftidx[:,0]],1]+=1
    fs = np.unique(ftidx[:, 0])
    ts = np.unique(ftidx[:, 1])
    nfs = len(fs)
    fig, ax = plt.subplots(nfs, 1, sharey=True)
    fig.set_figheight(17)
    fig.set_figwidth(30)
    colors = get_n_colors_from_colormap('hsv', dataset.discretize_nbins)
    colors[:, :-1] *= .7
    colors = colors[np.random.permutation(dataset.discretize_nbins), :]
    colors[:, -1] = alpha
    edges = np.zeros((len(fs), 2))
    edges[:, 0] = np.inf
    edges[:, 1] = -np.inf
    bin_edges = dataset.discretize_bin_edges
    bin_samples = dataset.discretize_bin_samples
    if dataset.sig_labels is not None:
        bin_edges = unzscore(bin_edges, dataset.mu_labels[dataset.discrete_idx, None],
                             dataset.sig_labels[dataset.discrete_idx, None])
        bin_samples = unzscore(bin_samples, dataset.mu_labels[None, dataset.discrete_idx, None],
                               dataset.sig_labels[None, dataset.discrete_idx, None])
    for i, idx in enumerate(dataset.discrete_idx):
        f = ftidx[i, 0]
        t = ftidx[i, 1]
        fi = np.nonzero(fs == f)[0][0]
        ti = np.nonzero(ts == t)[0][0]
        edges[fi, 0] = np.minimum(edges[fi, 0], bin_edges[i, 0])
        edges[fi, 1] = np.maximum(edges[fi, 1], bin_edges[i, -1])
        for j in range(dataset.discretize_nbins):
            ax[fi].plot(bin_samples[:, i, j], ti + r, '.', ms=.01, color=colors[j])
            ax[fi].plot([bin_edges[i, j], ] * 2, ti + np.array([-.5, .5]), 'k-')
        ax[fi].plot([bin_edges[i, -1], ] * 2, ti + np.array([-.5, .5]), 'k-')
        ax[fi].plot(bin_edges[i, [0, -1]], [ti + .5, ] * 2, 'k-')
        ax[fi].plot(bin_edges[i, [0, -1]], [ti - .5, ] * 2, 'k-')
    fnames = dataset.get_movement_names()
    for i, f in enumerate(fs):
        ax[i].set_title(fnames[f])
        ax[i].set_xlim(edges[i, 0], edges[i, 1])
        ax[i].set_yticks(np.arange(len(ts)))
        ax[i].set_yticklabels([str(t) for t in ts])
        ax[i].set_ylim(-.5, len(ts) - .5)
        ax[i].set_xscale('symlog')
    ax[-1].set_ylabel('Delta t')
    fig.tight_layout()
    return


def debug_plot_global_histograms(predv, labelsv, train_dataset, nbins=50, subsample=1, compare='time'):
    outnames_global = train_dataset.get_movement_names_global()

    # global labels, continuous representation, unzscored
    # ntimepoints x tspred x nglobal
    unz_labelsv = train_dataset.unzscore_labels(labelsv.numpy())
    unz_glabelsv = train_dataset.get_global_movement(unz_labelsv)

    # global predictions, continuous representation, unzscored
    # ntimepoints x tspred x nglobal
    unz_predv = train_dataset.unzscore_labels(predv.numpy())
    unz_gpredv = train_dataset.get_global_movement(unz_predv)

    if train_dataset.discretize:

        bin_edges = train_dataset.get_bin_edges(zscored=False)
        ftidx = train_dataset.unravel_label_index(train_dataset.discrete_idx)
        bins = []
        for f in featglobal:
            j = np.nonzero(np.all(ftidx == np.array([f, 1])[None, ...], axis=1))[0][0]
            bins.append(bin_edges[j])
        nbins = train_dataset.discretize_nbins
    else:
        lims = [[np.percentile(unz_glabelsv[::100, :, axi].flatten(), i).item() for i in [.1, 99.9]] for axi in
                range(nglobal)]
        bins = [np.arange(l[0], l[1], nbins + 1) for l in lims]

    ntspred = len(train_dataset.tspred_global)
    off0 = .1

    if compare == 'time':
        colors = get_n_colors_from_colormap('jet', len(train_dataset.tspred_global))
        colors[:, :-1] *= .8

        fig, ax = plt.subplots(2, nglobal, figsize=(30, 10), sharex='col')
        w = (1 - 2 * off0) / ntspred
        for axj, (datacurr, datatype) in enumerate(zip([unz_glabelsv, unz_gpredv], ['label ', 'pred '])):
            for axi in range(nglobal):
                ax[axj, axi].cla()
                off = off0
                for i in range(unz_glabelsv.shape[1]):
                    density, _ = np.histogram(datacurr[::subsample, i, axi], bins=bins[axi], density=True)
                    ax[axj, axi].bar(np.arange(nbins) + off, density, width=w, color=colors[i], log=True,
                                     align='edge', label=str(train_dataset.tspred_global[i]))
                    off += w
                ax[axj, axi].set_xticks(np.arange(nbins + 1))
                ax[axj, axi].set_xticklabels(['%.2f' % x for x in bins[axi]], rotation=90)
                ax[axj, axi].set_title(datatype + outnames_global[axi])
    elif compare == 'pred':
        colors = [[0, .5, .8], [.8, .5, .8]]

        fig, ax = plt.subplots(ntspred, nglobal, figsize=(20, 30), sharex='col', sharey='all')
        w = (1 - 2 * off0) / 2
        for ti in range(ntspred):
            for fi in range(nglobal):
                axcurr = ax[ti, fi]
                axcurr.cla()
                off = off0
                for i, (datacurr, datatype) in enumerate(zip([unz_glabelsv, unz_gpredv], ['label', 'pred'])):
                    density, _ = np.histogram(datacurr[::subsample, ti, fi], bins=bins[fi], density=True)
                    axcurr.bar(np.arange(nbins) + off, density, width=w, color=colors[i], log=False,
                               align='edge', label=datatype)
                    off += w
                axcurr.set_xticks(np.arange(nbins + 1))
                axcurr.set_xticklabels(['%.2f' % x for x in bins[fi]], rotation=90)
                axcurr.set_title(f'{outnames_global[fi]} t = {train_dataset.tspred_global[ti]}')

    ax[0, 0].legend()
    fig.tight_layout()

    return fig, ax


def debug_plot_global_error(predv, labelsv, pred_discretev, labels_discretev, train_dataset):
    """
  debug_plot_global_error(predv,labelsv,pred_discretev,labels_discretev,train_dataset)
  inputs:
  predv: torch.Tensor, shape (ntimepoints,d_output), all prediction features in continuous representation
  labelsv: torch.Tensor, shape (ntimepoints,d_output) all labels in continuous representation
  pred_discretev: torch.Tensor, shape (ntimepoints,len(train_dataset.discrete_idx),train_dataset.discretize_nbins), discrete predictions
  labels_discretev: torch.Tensor, shape (ntimepoints,len(train_dataset.discrete_idx),train_dataset.discretize_nbins), discrete labels
  train_dataset: FlyMLMDataset, the training dataset
  """
    outnames_global = train_dataset.get_movement_names_global()
    ntimepoints = predv.shape[0]

    # global predictions, continuous representation
    # ntimepoints x tspred x nglobal
    gpredv = train_dataset.get_global_movement(predv)

    # global labels, continuous representation
    # ntimepoints x tspred x nglobal
    glabelsv = train_dataset.get_global_movement(labelsv)

    # compute L1 error from continuous representations, all global features
    # network predictions
    errcont = np.nanmean(torch.nn.L1Loss(reduction='none')(gpredv, glabelsv), axis=0)
    # just predicting zero velocity all the time
    pred0 = np.zeros((ntimepoints, train_dataset.d_output))
    if train_dataset.mu_labels is not None:
        pred0 = train_dataset.zscore_labels(pred0)
    gpred0 = torch.as_tensor(train_dataset.get_global_movement(pred0))
    err0cont = np.nanmean(torch.nn.L1Loss(reduction='none')(gpred0, glabelsv), axis=0)

    # constant velocity predictions: use real labels from dt frames previous.
    # note we we won't have predictions for the first dt frames
    gpredprev = torch.zeros(glabelsv.shape)
    gpredprev[:] = np.nan
    for i, dt in enumerate(train_dataset.tspred_global):
        gpredprev[dt:, i, :] = glabelsv[:-dt, i, :]
    errprevcont = np.nanmean(torch.nn.L1Loss(reduction='none')(gpredprev, glabelsv), axis=0)

    if train_dataset.discretize:
        # ntimepoints x tspred x nglobal x nbins: discretized global predictions
        gpreddiscretev = torch.as_tensor(train_dataset.get_global_movement_discrete(pred_discretev))
        # ntimepoints x tspred x nglobal x nbins: discretized global labels
        glabelsdiscretev = torch.as_tensor(train_dataset.get_global_movement_discrete(labels_discretev))
        # cross entropy error
        errdiscrete = np.nanmean(torch.nn.CrossEntropyLoss(reduction='none')(gpreddiscretev.moveaxis(-1, 1),
                                                                             glabelsdiscretev.moveaxis(-1, 1)), axis=0)

        zerodiscretev = train_dataset.discretize_fun(pred0)
        gzerodiscretev = torch.as_tensor(train_dataset.get_global_movement_discrete(zerodiscretev))
        err0discrete = np.nanmean(torch.nn.CrossEntropyLoss(reduction='none')(gzerodiscretev.moveaxis(-1, 1),
                                                                              glabelsdiscretev.moveaxis(-1, 1)), axis=0)

        gpredprevdiscrete = torch.zeros(gpreddiscretev.shape, dtype=gpreddiscretev.dtype)
        gpredprevdiscrete[:] = np.nan
        for i, dt in enumerate(train_dataset.tspred_global):
            gpredprevdiscrete[dt:, i, :, :] = glabelsdiscretev[:-dt, i, :, :]
        errprevdiscrete = np.nanmean(torch.nn.CrossEntropyLoss(reduction='none')(gpredprevdiscrete.moveaxis(-1, 1),
                                                                                 glabelsdiscretev.moveaxis(-1, 1)),
                                     axis=0)

    if train_dataset.discretize:
        nc = 2
    else:
        nc = 1
    nr = nglobal
    fig, ax = plt.subplots(nr, nc, sharex=True, squeeze=False)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    # colors = mabe.get_n_colors_from_colormap('viridis',train_dataset.dct_tau)
    for i in range(nglobal):
        ax[i, 0].plot(errcont[:, i], 'o-', label=f'Pred')
        ax[i, 0].plot(err0cont[:, i], 's-', label=f'Zero')
        ax[i, 0].plot(errprevcont[:, i], 's-', label=f'Prev')
        if train_dataset.discretize:
            ax[i, 1].plot(errdiscrete[:, i], 'o-', label=f'Pred')
            ax[i, 1].plot(err0discrete[:, i], 's-', label=f'Zero')
            ax[i, 1].plot(errprevdiscrete[:, i], 's-', label=f'Prev')
            ax[i, 0].set_title(f'{outnames_global[i]} L1 error')
            ax[i, 1].set_title(f'{outnames_global[i]} CE error')
        else:
            ax[i, 0].set_title(outnames_global[i])
    ax[-1, -1].set_xticks(np.arange(train_dataset.ntspred_global))
    ax[-1, -1].set_xticklabels([str(t) for t in train_dataset.tspred_global])
    ax[-1, -1].set_xlabel('Delta t')
    ax[0, 0].legend()
    plt.tight_layout()

    return


def debug_plot_dct_relative_error(predv, labelsv, train_dataset):
    dt = train_dataset.dct_tau
    dcterr = np.sqrt(
        np.nanmean((predv[:, train_dataset.idxdct_relative] - labelsv[:, train_dataset.idxdct_relative]) ** 2., axis=0))
    dcterr0 = np.sqrt(np.nanmean((labelsv[:, train_dataset.idxdct_relative]) ** 2., axis=0))
    dcterrprev = np.sqrt(
        np.nanmean((labelsv[:-dt, train_dataset.idxdct_relative] - labelsv[dt:, train_dataset.idxdct_relative]) ** 2.,
                   axis=0))

    nc = int(np.ceil(np.sqrt(nrelative)))
    nr = int(np.ceil(nrelative / nc))
    fig, ax = plt.subplots(nr, nc, sharex=True, sharey=True)
    fig.set_figheight(14)
    fig.set_figwidth(23)
    ax = ax.flatten()
    for i in range(nrelative, nc * nr):
        ax[i].remove()
    ax = ax[:nrelative]
    for i in range(nrelative):
        ax[i].plot(dcterr[:, i], 'o-', label=f'pred')
        ax[i].plot(dcterr0[:, i], 's-', label=f'zero')
        ax[i].plot(dcterrprev[:, i], 's-', label=f'prev')
        ax[i].set_title(posenames[np.nonzero(featrelative)[0][i]])
    ax[-1].set_xticks(np.arange(train_dataset.dct_tau))
    ax[(nc - 1) * nr - 1].set_xlabel('DCT feature')
    ax[0].legend()
    plt.tight_layout()

    predrelative_dct = train_dataset.get_relative_movement_dct(predv.numpy())
    labelsrelative_dct = train_dataset.get_relative_movement_dct(labelsv.numpy())
    zpredrelative_dct = np.zeros(predrelative_dct.shape)
    zlabelsrelative_dct = np.zeros(labelsrelative_dct.shape)
    for i in range(predrelative_dct.shape[1]):
        zpredrelative_dct[:, i, :] = zscore(predrelative_dct[:, i, :],
                                            train_dataset.mu_labels[train_dataset.nextframeidx_relative],
                                            train_dataset.sig_labels[train_dataset.nextframeidx_relative])
        zlabelsrelative_dct[:, i, :] = zscore(labelsrelative_dct[:, i, :],
                                              train_dataset.mu_labels[train_dataset.nextframeidx_relative],
                                              train_dataset.sig_labels[train_dataset.nextframeidx_relative])
    idcterr = np.sqrt(np.nanmean((zpredrelative_dct - zlabelsrelative_dct) ** 2., axis=0))
    nexterr = np.sqrt(np.nanmean(
        (train_dataset.get_next_relative_movement(predv) - train_dataset.get_next_relative_movement(labelsv)) ** 2,
        axis=0))
    err0 = np.sqrt(np.nanmean((zlabelsrelative_dct) ** 2, axis=0))
    errprev = np.sqrt(np.nanmean((zlabelsrelative_dct[:-dt, :, :] - zlabelsrelative_dct[dt:, :, :]) ** 2, axis=0))

    plt.figure()
    plt.clf()
    plt.plot(idcterr[0, :], 's-', label='dct pred')
    plt.plot(nexterr, 'o-', label='next pred')
    plt.plot(err0[0, :], 's-', label='zero')
    plt.plot(errprev[0, :], 's-', label='prev')
    plt.legend()
    plt.xticks(np.arange(nrelative))
    plt.gca().set_xticklabels([posenames[i] for i in np.nonzero(featrelative)[0]])
    plt.xticks(rotation=90)
    plt.title('Next frame prediction')
    plt.tight_layout()

    fig, ax = plt.subplots(nr, nc, sharex=True, sharey=True)
    fig.set_figheight(14)
    fig.set_figwidth(23)
    ax = ax.flatten()
    for i in range(nrelative, nc * nr):
        ax[i].remove()
    ax = ax[:nrelative]
    for i in range(nrelative):
        ax[i].plot(idcterr[:, i], 'o-', label=f'pred')
        ax[i].plot(err0[:, i], 's-', label=f'zero')
        ax[i].plot(errprev[:, i], 's-', label=f'prev')
        ax[i].set_title(posenames[np.nonzero(featrelative)[0][i]])
    ax[-1].set_xticks(np.arange(train_dataset.dct_tau))
    ax[(nc - 1) * nr - 1].set_xlabel('Delta t')
    ax[0].legend()
    plt.tight_layout()

    return


def debug_plot_histogram_edges(train_dataset):
    bin_edges = train_dataset.get_bin_edges(zscored=False)
    ftidx = train_dataset.unravel_label_index(train_dataset.discrete_idx)
    fs = np.unique(ftidx[:, 0])
    ts = np.unique(ftidx[:, 1])
    fig, ax = plt.subplots(1, len(fs), sharey=True)
    movement_names = train_dataset.get_movement_names()
    for i, f in enumerate(fs):
        ax[i].cla()
        idx = ftidx[:, 0] == f
        tscurr = ftidx[idx, 1]
        tidx = npindex(ts, tscurr)
        ax[i].plot(bin_edges[idx, :], tidx, '.-')
        ax[i].set_title(movement_names[f])
        ax[i].set_xscale('symlog')
    ax[0].set_yticks(np.arange(len(ts)))
    ax[0].set_yticklabels([str(t) for t in ts])
    return fig, ax


def initialize_debug_plots(dataset, dataloader, data, name='', tsplot=None, traj_nsamplesplot=3):
    example = next(iter(dataloader))

    # plot to visualize input features
    fig, ax = debug_plot_sample_inputs(dataset, example)

    # plot to check that we can get poses from examples
    hpose, ax, fig = debug_plot_batch_pose(example, dataset, data=data, tsplot=tsplot)
    ax[-1, 0].set_xlabel('Train')

    # plot to visualize motion outputs
    axtraj, figtraj = debug_plot_batch_traj(example, dataset, data=data,
                                            label_true='Label',
                                            label_pred='Raw',
                                            nsamplesplot=traj_nsamplesplot)
    figtraj.set_figheight(18)
    figtraj.set_figwidth(12)
    axtraj[0].set_title(name)
    figtraj.tight_layout()

    hdebug = {
        'figpose': fig,
        'axpose': ax,
        'hpose': hpose,
        'figtraj': figtraj,
        'axtraj': axtraj,
        'hstate': None,
        'axstate': None,
        'figstate': None,
        'example': example
    }

    plt.show()
    plt.pause(.001)

    return hdebug


def update_debug_plots(hdebug, config, model, dataset, example, pred, criterion=None, name='', tsplot=None,
                       traj_nsamplesplot=3):
    if config['modelstatetype'] == 'prob':
        pred1 = model.maxpred({k: v.detach() for k, v in pred.items()})
    elif config['modelstatetype'] == 'best':
        pred1 = model.randpred(pred.detach())
    else:
        if isinstance(pred, dict):
            pred1 = {k: v.detach().cpu() for k, v in pred.items()}
        else:
            pred1 = pred.detach().cpu()
    debug_plot_batch_pose(example, dataset, pred=pred1, h=hdebug['hpose'], ax=hdebug['axpose'], fig=hdebug['figpose'],
                          tsplot=tsplot)
    debug_plot_batch_traj(example, dataset, criterion=criterion, config=config, pred=pred1, ax=hdebug['axtraj'],
                          fig=hdebug['figtraj'], nsamplesplot=traj_nsamplesplot)
    if config['modelstatetype'] == 'prob':
        hstate, axstate, figstate = debug_plot_batch_state(pred['stateprob'].detach().cpu(), nsamplesplot=3,
                                                           h=hdebug['hstate'], ax=hdebug['axstate'],
                                                           fig=hdebug['figstate'])
        hdebug['axstate'][0].set_title(name)

    hdebug['axtraj'][0].set_title(name)


def initialize_loss_plots(loss_epoch):
    nax = len(loss_epoch) // 2
    assert (nax >= 1) and (nax <= 3)
    hloss = {}

    hloss['fig'], hloss['ax'] = plt.subplots(nax, 1)
    if nax == 1:
        hloss['ax'] = [hloss['ax'], ]

    hloss['train'], = hloss['ax'][0].plot(loss_epoch['train'].cpu(), '.-', label='Train')
    hloss['val'], = hloss['ax'][0].plot(loss_epoch['val'].cpu(), '.-', label='Val')

    if 'train_continuous' in loss_epoch:
        hloss['train_continuous'], = hloss['ax'][1].plot(loss_epoch['train_continuous'].cpu(), '.-',
                                                         label='Train continuous')
    if 'train_discrete' in loss_epoch:
        hloss['train_discrete'], = hloss['ax'][2].plot(loss_epoch['train_discrete'].cpu(), '.-', label='Train discrete')
    if 'val_continuous' in loss_epoch:
        hloss['val_continuous'], = hloss['ax'][1].plot(loss_epoch['val_continuous'].cpu(), '.-', label='Val continuous')
    if 'val_discrete' in loss_epoch:
        hloss['val_discrete'], = hloss['ax'][2].plot(loss_epoch['val_discrete'].cpu(), '.-', label='Val discrete')

    hloss['ax'][-1].set_xlabel('Epoch')
    hloss['ax'][-1].set_ylabel('Loss')
    for l in hloss['ax']:
        l.legend()
    return hloss


def update_loss_plots(hloss, loss_epoch):
    hloss['train'].set_ydata(loss_epoch['train'].cpu())
    hloss['val'].set_ydata(loss_epoch['val'].cpu())
    if 'train_continuous' in loss_epoch:
        hloss['train_continuous'].set_ydata(loss_epoch['train_continuous'].cpu())
    if 'train_discrete' in loss_epoch:
        hloss['train_discrete'].set_ydata(loss_epoch['train_discrete'].cpu())
    if 'val_continuous' in loss_epoch:
        hloss['val_continuous'].set_ydata(loss_epoch['val_continuous'].cpu())
    if 'val_discrete' in loss_epoch:
        hloss['val_discrete'].set_ydata(loss_epoch['val_discrete'].cpu())
    for l in hloss['ax']:
        l.relim()
        l.autoscale()


def explore_representation(configfile):
    config = read_config(configfile)

    np.random.seed(config['numpy_seed'])
    torch.manual_seed(config['torch_seed'])
    device = torch.device(config['device'])

    plt.ion()

    data, scale_perfly = load_and_filter_data(config['intrainfile'], config)
    splitdata = split_data_by_id(data)

    for i in range(len(splitdata)):
        scurr = splitdata[i]
        fcurr = compute_features(data['X'][..., scurr['i0']:scurr['i1'], :],
                                 scurr['id'], scurr['flynum'], scale_perfly, smush=False, simplify_in='no_sensory')
        movecurr = fcurr['labels']
        if i == 0:
            move = movecurr
        else:
            move = np.r_[move, movecurr]

    outnames_global = ['forward', 'sideways', 'orientation']
    outnames = outnames_global + [posenames[x] for x in np.nonzero(featrelative)[0]]

    mu = np.nanmean(move, axis=0)
    sig = np.nanstd(move, axis=0)
    zmove = (move - mu) / sig

    bin_edges = np.zeros((nfeatures, config['discretize_nbins'] + 1))
    for feati in range(nfeatures):
        bin_edges[feati, :] = select_bin_edges(move[:, feati], config['discretize_nbins'],
                                               config['all_discretize_epsilon'][feati], feati=feati)

    featpairs = [
        ['left_front_leg_tip_angle', 'left_front_leg_tip_dist'],
        ['left_middle_femur_base_angle', 'left_middle_femur_tibia_joint_angle'],
        ['left_middle_femur_tibia_joint_angle', 'left_middle_leg_tip_angle'],
    ]
    nax = len(featpairs)
    nc = int(np.ceil(np.sqrt(nax)))
    nr = int(np.ceil(nax / nc))
    fig, ax = plt.subplots(nr, nc, squeeze=False)
    ax = ax.flatten()

    for i in range(len(featpairs)):
        feati = [outnames.index(x) for x in featpairs[i]]
        density, _, _ = np.histogram2d(zmove[:, feati[0]], zmove[:, feati[1]],
                                       bins=[bin_edges[feati[0], :], bin_edges[feati[1], :]], density=True)
        ax[i].cla()
        X, Y = np.meshgrid(bin_edges[feati[0], 1:-1], bin_edges[feati[1], 1:-1])
        density = density[1:-1, 1:-1]
        him = ax[i].pcolormesh(X, Y, density,
                               norm=colors.LogNorm(vmin=np.min(density[density > 0]), vmax=np.max(density)),
                               edgecolors='k')
        ax[i].set_xlabel(outnames[feati[0]])
        ax[i].set_ylabel(outnames[feati[1]])
    fig.tight_layout()

    valdata, val_scale_perfly = load_and_filter_data(config['invalfile'], config)

