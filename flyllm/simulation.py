import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from apf.data import get_real_agents
from flyllm.features import compute_pose_features, split_features
from flyllm.config import ARENA_RADIUS_MM
from flyllm.plotting import plot_flies, plot_arena
from flyllm.pose import FlyPoseLabels, FlyExample
from flyllm.prediction import predict_iterative


def get_pose_future(data, scales, tspred_global, ts=None, fliespred=None):
    maxT = data['X'].shape[2]
    if ts is None:
        ts = np.arange(maxT)
    if fliespred is None:
        fliespred = np.arange(data['X'].shape[3])

    Xkpfuture = np.zeros((data['X'].shape[0], data['X'].shape[1], len(ts), len(tspred_global), len(fliespred)))
    Xkpfuture[:] = np.nan
    for ti, toff in enumerate(tspred_global):
        idxcurr = ts < maxT - toff
        tscurr = ts[idxcurr]
        Xkpfuture[:, :, idxcurr, ti] = data['X'][:, :, tscurr + toff][..., fliespred]
        isbad = data['videoidx'][tscurr, 0] != data['videoidx'][tscurr + toff, 0]
        Xkpfuture[:, :, isbad] = np.nan

    relposefuture, globalposfuture = compute_pose_features(Xkpfuture, scales)
    if globalposfuture.ndim == 3:  # when there is one fly, it gets collapsed
        globalposfuture = globalposfuture[..., None]
        relposefuture = relposefuture[..., None]
    globalposfuture = globalposfuture.transpose(1, 2, 0, 3)
    relposefuture = relposefuture.transpose(1, 2, 0, 3)
    return globalposfuture, relposefuture


def compute_attention_weight_rollout(w0):
    # w0 is nlayers x T x T x ...
    w = np.zeros(w0.shape, dtype=w0.dtype)
    wcurr = np.ones(list(w0.shape)[1:], dtype=w0.dtype)
    # I = np.eye(w0.shape[1],dtype=w0.dtype)
    # sz = np.array(w0.shape[1:])
    # sz[2:] = 1
    # I = I.reshape(sz)

    for i in range(w0.shape[0]):
        wcurr = wcurr * (w0[i, ...])
        z = np.maximum(np.sum(wcurr, axis=0, keepdims=True), np.finfo(w0.dtype).eps)
        wcurr = wcurr / z
        w[i, ...] = wcurr
    return w


def compute_all_attention_weight_rollouts(attn_weights0):
    attn_weights_rollout = None
    firstframe = None
    attn_context = None
    tpred = attn_weights0[0].size  # to do check
    for t, w0 in enumerate(attn_weights0):
        if w0 is None:
            continue
        w = compute_attention_weight_rollout(w0)
        w = w[-1, -1, ...]
        if attn_weights_rollout is None:
            attn_weights_rollout = np.zeros((tpred,) + w.shape)
            attn_weights_rollout[:] = np.nan
            firstframe = t
            attn_context = w.shape[0]
        if attn_context < w.shape[0]:
            pad = np.zeros([tpred, w.shape[0] - attn_context, ] + list(w.shape)[1:])
            pad[:firstframe, ...] = np.nan
            attn_context = w.shape[0]
            attn_weights_rollout = np.concatenate((attn_weights_rollout, pad), axis=1)
        attn_weights_rollout[t, :] = 0.
        attn_weights_rollout[t, :w.shape[0]] = w
    return attn_weights_rollout




def animate_pose(Xkps, focusflies=[], ax=None, fig=None, t0=0,
                 figsizebase=11, ms=6, lw=1, focus_ms=12, focus_lw=3,
                 titletexts={}, savevidfile=None, fps=30, trel0=0,
                 inputs=None, nstd_input=3, contextl=10, axinput=None,
                 attn_weights=None, skeledgecolors=None,
                 globalpos_future=None, tspred_future=None,
                 futurecolor=[0, 0, 0, .25], futurelw=1, futurems=6,
                 futurealpha=.25):
    plotinput = inputs is not None and len(inputs) > 0

    # attn_weights[key] should be T x >=contextl x nfocusflies
    plotattn = attn_weights is not None

    plotfuture = globalpos_future is not None

    ninputs = 0
    if plotinput:
        inputnames = []
        for v in inputs.values():
            if v is not None:
                inputnames = list(v.keys())
                break
        ninputs = len(inputnames)
        if ninputs == 0:
            plotinput = False

    if plotinput or plotattn:
        naxc = len(Xkps)
        naxr = 1
        nax = naxc * naxr
    else:
        nax = len(Xkps)
        naxc = int(np.ceil(np.sqrt(nax)))
        naxr = int(np.ceil(nax / naxc))

    if plotattn:
        nsubax = ninputs + 1
    else:
        nsubax = ninputs

    # get rid of blank flies
    Xkp = list(Xkps.values())[0]
    T = Xkp.shape[-2]
    isreal = get_real_agents(Xkp)
    nflies = Xkp.shape[-1]
    isfocusfly = np.zeros(nflies, dtype=bool)
    isfocusfly[focusflies] = True
    for Xkp in Xkps.values():
        assert (nflies == Xkp.shape[-1])
        isreal = isreal | get_real_agents(Xkp)

    for k, v in Xkps.items():
        Xkps[k] = v[..., isreal]
    focusflies = np.nonzero(isfocusfly[isreal])[0]

    nflies = np.count_nonzero(isreal)

    minv = -ARENA_RADIUS_MM * 1.01
    maxv = ARENA_RADIUS_MM * 1.01

    h = {}

    trel = trel0
    t = t0 + trel
    createdax = False
    if ax is None:
        if fig is None:
            fig = plt.figure()
            if plotinput or plotattn:
                fig.set_figheight(figsizebase * 1.5)
            else:
                fig.set_figheight(figsizebase * naxr)
            fig.set_figwidth(figsizebase * naxc)

        if plotinput or plotattn:
            gs = matplotlib.gridspec.GridSpec(3, len(Xkps) * nsubax, figure=fig)
            ax = np.array([fig.add_subplot(gs[:2, nsubax * i:nsubax * (i + 1)]) for i in range(len(Xkps))])
        else:
            ax = fig.subplots(naxr, naxc)

        for axcurr in ax:
            axcurr.set_xticks([])
            axcurr.set_yticks([])
        createdax = True
    else:
        assert (ax.size >= nax)
    ax = ax.flatten()
    if (plotinput or plotattn) and (axinput is None):
        gs = matplotlib.gridspec.GridSpec(3, len(Xkps) * nsubax, figure=fig)
        axinput = {}
        for i, k in enumerate(Xkps):
            if k in inputs:
                axinput[k] = np.array([fig.add_subplot(gs[-1, i * nsubax + j]) for j in range(nsubax)])
                for axcurr in axinput[k][1:]:
                    axcurr.set_yticks([])

        createdax = True

    if createdax:
        fig.tight_layout()

    h['kpt'] = []
    h['edge'] = []
    h['ti'] = []
    if plotfuture:
        h['future'] = []
        nsamples = {k: globalpos_future[k].shape[0] for k in globalpos_future.keys()}

    titletext_ts = np.array(list(titletexts.keys()))

    if 0 in titletexts:
        titletext_str = titletexts[0]
    else:
        titletext_str = ''

    for i, k in enumerate(Xkps):

        if plotfuture and k in globalpos_future:
            hfuture = []
            ntsfuture = globalpos_future[k].shape[2]
            for j in range(len(focusflies)):
                futurecolors = plt.get_cmap('jet')(np.linspace(0, 1, ntsfuture))
                futurecolors[:, -1] = futurealpha
                hfuturefly = [None, ] * ntsfuture
                for tfuturei in range(ntsfuture - 1, -1, -1):
                    hfuturecurr = ax[i].plot(globalpos_future[k][:, trel, tfuturei, 0, j],
                                             globalpos_future[k][:, trel, tfuturei, 1, j], '.',
                                             color=futurecolors[tfuturei], ms=futurems, lw=futurelw)[0]
                    hfuturefly[tfuturei] = hfuturecurr
                # for samplei in range(nsamples[k]):
                #   hfuturecurr = ax[i].plot(globalpos_future[k][samplei,trel,:,0,j],globalpos_future[k][samplei,trel,:,1,j],'.-',color=futurecolor,ms=futurems,lw=futurelw)[0]
                #   hfuturefly.append(hfuturecurr)
                hfuture.append(hfuturefly)
            h['future'].append(hfuture)

        hkpt, hedge, _, _, _ = plot_flies(Xkps[k][..., trel, :], ax=ax[i], kpt_ms=ms, skel_lw=lw,
                                               skeledgecolors='tab20')

        for j in focusflies:
            hkpt[j].set_markersize(focus_ms)
            hedge[j].set_linewidth(focus_lw)
        h['kpt'].append(hkpt)
        h['edge'].append(hedge)

        ax[i].set_aspect('equal')
        plot_arena(ax=ax[i])
        if i == 0:
            hti = ax[i].set_title(f'{titletext_str} {k}, t = {t}')
        else:
            hti = ax[i].set_title(k)
        h['ti'].append(hti)

        ax[i].set_xlim(minv, maxv)
        ax[i].set_ylim(minv, maxv)

    if plotinput or plotattn:
        h['input'] = {}
        t0input = np.maximum(0, trel - contextl)
        contextlcurr = trel0 - t0input + 1

        if plotinput:
            for k in inputs.keys():
                h['input'][k] = []
                for i, inputname in enumerate(inputnames):
                    inputcurr = inputs[k][inputname][trel + 1:t0input:-1, :]
                    if contextlcurr < contextl:
                        pad = np.zeros([contextl - contextlcurr, ] + list(inputcurr.shape)[1:])
                        pad[:] = np.nan
                        inputcurr = np.r_[inputcurr, pad]
                    hin = axinput[k][i].imshow(inputcurr, vmin=-nstd_input, vmax=nstd_input, cmap='coolwarm')
                    axinput[k][i].set_title(inputname)
                    axinput[k][i].axis('auto')
                    h['input'][k].append(hin)
        if plotattn:
            for k in attn_weights.keys():
                if k not in h['input']:
                    h['input'][k] = []
                # currently only support one focus fly
                hattn = axinput[k][-1].plot(attn_weights[k][trel, -contextl:, 0], np.arange(contextl, 0, -1))[0]
                # axinput[k][-1].set_xscale('log')
                axinput[k][-1].set_ylim([-.5, contextl - .5])
                axinput[k][-1].set_xlim([0, 1])
                axinput[k][-1].invert_yaxis()
                axinput[k][-1].set_title('attention')
                h['input'][k].append(hattn)

    hlist = []
    for hcurr in h.values():
        if type(hcurr) == list:
            hlist += hcurr
        else:
            hlist += [hcurr, ]

    def update(trel):

        t = t0 + trel
        if np.any(titletext_ts <= trel):
            titletext_t = np.max(titletext_ts[titletext_ts <= trel])
            titletext_str = titletexts[titletext_t]
        else:
            titletext_str = ''

        for i, k in enumerate(Xkps):
            plot_flies(Xkps[k][..., trel, :], ax=ax[i], hkpts=h['kpt'][i], hedges=h['edge'][i])
            if plotfuture and k in globalpos_future:
                ntsfuture = globalpos_future[k].shape[2]
                for j in range(len(focusflies)):

                    for tfuturei in range(ntsfuture - 1, -1, -1):
                        h['future'][i][j][tfuturei].set_xdata(globalpos_future[k][:, trel, tfuturei, 0, j])
                        h['future'][i][j][tfuturei].set_ydata(globalpos_future[k][:, trel, tfuturei, 1, j])

                    # for samplei in range(nsamples[k]):
                    #   h['future'][i][j][samplei].set_xdata(globalpos_future[k][samplei,trel,:,0,j])
                    #   h['future'][i][j][samplei].set_ydata(globalpos_future[k][samplei,trel,:,1,j])
            if i == 0:
                h['ti'][i].set_text(f'{titletext_str} {k}, t = {t}')
            else:
                h['ti'][i].set_text(k)

        if plotinput or plotattn:
            t0input = np.maximum(0, trel - contextl)
            contextlcurr = trel - t0input + 1

        if plotinput:
            for k in inputs.keys():
                for i, inputname in enumerate(inputnames):

                    inputcurr = inputs[k][inputname][trel + 1:t0input:-1, :]
                    if contextlcurr < contextl:
                        pad = np.zeros([contextl - contextlcurr, ] + list(inputcurr.shape)[1:])
                        pad[:] = np.nan
                        inputcurr = np.r_[inputcurr, pad]
                    h['input'][k][i].set_data(inputcurr)

        if plotattn:
            for k in attn_weights.keys():
                attn_curr = attn_weights[k][trel, -contextl:, 0]
                h['input'][k][-1].set_xdata(attn_curr)
                # if any(np.isnan(attn_curr)==False):
                #   axinput[k][-1].set_xlim([0,np.nanmax(attn_curr)])
        return hlist

    ani = animation.FuncAnimation(fig, update, frames=range(trel0, T))

    if savevidfile is not None:
        print('Saving animation to file %s...' % savevidfile)
        writer = animation.PillowWriter(fps=30)
        ani.save(savevidfile, writer=writer)
        print('Finished writing.')

    return ani


def animate_predict_open_loop(model, dataset, Xkp_init, fliespred, scales, tpred, burnin=None,
                              debug=False, plotattnweights=False, plotfuture=False, nsamplesfuture=1, metadata=None):
    # ani = animate_predict_open_loop(model,val_dataset,valdata,val_scale_perfly,config,fliespred,t0,tpred,debug=False,
    #                            plotattnweights=False,plotfuture=train_dataset.ntspred_global>1,nsamplesfuture=nsamplesfuture)

    if burnin is None:
        burnin = dataset.contextl

    # Xkp_init is nkp x 2 x Tinit x nflies
    Tinit = Xkp_init.shape[-2]
    nflies = Xkp_init.shape[-1]
    szrest = Xkp_init.shape[:-2]

    # Xkp* should be nkp x 2 x T x nflies 
    # where t is npred + ntspred_max
    ntspred_max = dataset.ntspred_max
    T = tpred + ntspred_max

    # true keypoints for comparing to predictions
    Xkp_true = np.zeros(szrest+(T,nflies))+np.nan
    Xkp_true[...,:min(Tinit,T),:] = Xkp_init[...,:min(Tinit,T),:]

    # initialize Xkp with the true data, use real data for burn in
    Xkp = Xkp_true.copy()
    # overwrite data we will predict with nans to make sure we aren't cheating
    Xkp[...,burnin+1:,:][...,fliespred] = np.nan

    # fliespred = np.nonzero(mabe.get_real_agents(Xkp))[0]
    labels_true = []
    examples_pred = []
    for i, agentnum in enumerate(fliespred):
        scale = scales[i]
        label_true = FlyPoseLabels(Xkp=Xkp_true[..., agentnum], scale=scale, dataset=dataset)
        labels_true.append(label_true)
        # note that labels for > 1 frame into the future may be nan here if t+tspred > burnin
        # labels and pred should match up to t = burnin
        example_pred = FlyExample(Xkp=Xkp, agentnum=agentnum, scale=scale, dataset=dataset)
        examples_pred.append(example_pred)

    # TODOKB: reimplement this
    if plotfuture:
        plotfuture = False
        # warn that plot future is not implemented currently
        print('TODO Plot future is not implemented currently')

        # subtract one from tspred_global, as the tspred_global for predicted data come from the previous frame
        #globalposfuture_true, relposefuture_true = get_pose_future(data, scales, [t + 1 for t in dataset.tspred_global],
        #                                                           ts=np.arange(t0, t0 + tpred), fliespred=fliespred)
        globalposfuture_true = None
        relposefuture_true = None

    model.eval()

    if dataset.ntspred_max == 1:
        endoff = None
    else:
        endoff = -dataset.ntspred_max+1
    Xkp_fill = Xkp[...,:endoff,:].copy()        

    # capture all outputs of predict_open_loop in a tuple
    res = predict_iterative(examples_pred, fliespred, scales, Xkp_fill, burnin, model, dataset, maxcontextl=dataset.contextl+1,
                                    debug=debug, need_weights=plotattnweights, nsamples=nsamplesfuture, 
                                    labels_true=labels_true)
    if plotattnweights:
        examples_pred,attn_weights0 = res
    else:
        examples_pred = res

    Xkps = {'Pred': Xkp_fill.copy(), 'True': Xkp_true.copy()}
    if len(fliespred) == 1:
        starttoff = dataset.get_start_toff()
        split_inputs = examples_pred[0].inputs.get_split_inputs(zscored=False,makecopy=True)
        if starttoff > 0:
            for k, v in split_inputs.items():
                split_inputs[k] = np.r_[np.nan+np.zeros((starttoff,v.shape[-1])),v]
        inputs = {'Pred': split_inputs}
    else:
        inputs = None

    if plotattnweights:
        attn_weights = {'Pred': compute_all_attention_weight_rollouts(attn_weights0)}
    else:
        attn_weights = None

    focusflies = fliespred
    titletexts = {0: 'Initialize', burnin: ''}

    if plotfuture:
        future_args = {'globalpos_future': {'Pred': globalposfuture_pred, 'True': globalposfuture_true[None, ...]},
                       'tspred_future': dataset.tspred_global}
    else:
        future_args = {}

    if (metadata is not None) and ('t0' in metadata):
        t0 = metadata['t0']
    else:
        t0 = 0
    ani = animate_pose(Xkps, focusflies=focusflies, t0=t0, titletexts=titletexts,
                       trel0=np.maximum(0, dataset.contextl - 63),
                       inputs=inputs, contextl=dataset.contextl, attn_weights=attn_weights,
                       **future_args)

    return ani
