import numpy as np
import tqdm
import re
import torch
import logging

from apf.config import SENSORY_PARAMS, PXPERMM, keypointnames, featglobal
from apf.features import (
    compute_features,
    compute_scale_perfly,
    compute_pose_features,
    compute_movement,
    compute_otherflies_touch_mult
)
# TODO: would be nice if data did not depend on features
from apf.utils import get_interval_ends


def interval_all(x, l):
    """
        y = interval_all(x,l)

        Computes logical all over intervals of length l in the first dimension
        y[i,j] is whether all entries in the l-length interval x[i:i+l,j] are true.
        x: input matrix of any shape. all will be computed over x[i:i+l,j,k]
        outputs a matrix y of size (x.shape[0]-l,)+x.shape[1:]).
    """
    csx = np.concatenate((np.zeros((1,) + x.shape[1:], dtype=int), np.cumsum(x, axis=0)), axis=0)
    y = csx[l:-1, ...] - csx[:-l - 1, ...] == l
    return y


def chunk_data(data, contextl, reparamfun, npad=1):
    contextlpad = contextl + npad

    # all frames for the main fly must have real data
    allisdata = interval_all(data['isdata'], contextlpad)
    isnotsplit = interval_all(data['isstart'] == False, contextlpad - 1)[1:, ...]
    canstart = np.logical_and(allisdata, isnotsplit)

    # X is nkeypts x 2 x T x nflies
    nkeypoints = data['X'].shape[0]
    T = data['X'].shape[2]
    maxnflies = data['X'].shape[3]
    assert T > 2 * contextlpad, 'Assumption that data has more frames than 2*(contextl+1) is incorrect, code will fail'

    # last possible start frame = T - contextl
    maxt0 = canstart.shape[0] - 1
    # X is a dict with chunked data
    X = []
    # loop through ids
    nframestotal = 0
    for flynum in tqdm.trange(maxnflies, desc='Fly'):
        # choose a first frame near the beginning, but offset a bit
        # first possible start
        canstartidx = np.nonzero(canstart[:, flynum])[0]
        if canstartidx.size == 0:
            continue

        mint0curr = canstartidx[0]
        # offset a bit
        t0 = mint0curr + np.random.randint(0, contextl, None)
        # find the next allowed frame
        if canstart[t0, flynum] == False:
            if not np.any(canstart[t0:, flynum]):
                continue
            t0 = np.nonzero(canstart[t0:, flynum])[0][0] + t0

        maxt0curr = canstartidx[-1]
        # maxt1curr = maxt0curr+contextlpad-1
        ndata = np.count_nonzero(data['isdata'][:, flynum])
        maxintervals = ndata // contextl + 1
        for i in tqdm.trange(maxintervals, desc='Interval'):
            if t0 > maxt0:
                break
            # this is guaranteed to be < T
            t1 = t0 + contextlpad - 1
            id = data['ids'][t0, flynum]
            xcurr = reparamfun(data['X'][..., t0:t1 + 1, :], id, flynum, npad=npad)
            xcurr['metadata'] = {'flynum': flynum, 'id': id, 't0': t0, 'videoidx': data['videoidx'][t0, 0],
                                 'frame0': data['frames'][t0, 0]}
            xcurr['categories'] = data['y'][:, t0:t1 + 1, flynum].astype(np.float32)
            X.append(xcurr)
            if t0 + contextl >= maxt0curr:
                break
            elif canstart[t0 + contextl, flynum]:
                t0 = t0 + contextl
            else:
                t0 = np.nonzero(canstart[t1 + 1:, flynum])[0]
                if t0 is None or t0.size == 0:
                    break
                t0 = t0[0] + t1 + 1
            nframestotal += contextl

    logging.info(f'In total {nframestotal} frames of data after chunking')

    return X


def select_bin_edges(movement, nbins, bin_epsilon, outlierprct=0, feati=None):
    n = movement.shape[0]
    lims = np.percentile(movement, [outlierprct, 100 - outlierprct])
    max_bin_epsilon = (lims[1] - lims[0]) / (nbins + 1)
    if bin_epsilon >= max_bin_epsilon:
        logging.info(
            f'{feati}: bin_epsilon {bin_epsilon} bigger than max bin epsilon {max_bin_epsilon}, '
            f'setting all bins to be the same size'
        )
        bin_edges = np.linspace(lims[0], lims[1], nbins + 1)
        return bin_edges

    bin_edges = np.arange(lims[0], lims[1], bin_epsilon)
    bin_edges[-1] = lims[1]

    counts, _ = np.histogram(movement, bin_edges)
    mergecounts = counts[1:] + counts[:-1]
    for iter in range(len(bin_edges) - nbins - 1):
        mincount = np.min(mergecounts)
        bini = np.random.choice(np.nonzero(mergecounts == mincount)[0], 1)[0]
        if bini > 0:
            mergecounts[bini - 1] += counts[bini]
        if bini < len(mergecounts) - 1:
            mergecounts[bini + 1] += counts[bini]
        mergecounts = np.delete(mergecounts, bini)
        counts[bini] = mincount
        counts = np.delete(counts, bini + 1)
        bin_edges = np.delete(bin_edges, bini + 1)

    return bin_edges


def weighted_sample(w, nsamples=0):
    SMALLNUM = 1e-6
    assert (torch.all(w >= 0.))
    nbins = w.shape[-1]
    szrest = w.shape[:-1]
    n = int(np.prod(szrest))
    p = torch.cumsum(w.reshape((n, nbins)), dim=-1)
    assert (torch.all(torch.abs(p[:, -1] - 1) <= SMALLNUM))
    p[p > 1.] = 1.
    p[:, -1] = 1.
    if nsamples == 0:
        nsamples1 = 1
    else:
        nsamples1 = nsamples
    r = torch.rand((nsamples1, n), device=w.device)
    s = torch.zeros((nsamples1,) + p.shape, dtype=w.dtype, device=w.device)
    s[:] = r[..., None] <= p
    idx = torch.argmax(s, dim=-1)
    if nsamples > 0:
        szrest = (nsamples,) + szrest
    return idx.reshape(szrest)


def fit_discretize_labels(data, featidx, nbins=50, bin_epsilon=None, outlierprct=.001, fracsample=None, nsamples=None):
    # compute percentiles
    nfeat = len(featidx)
    prctiles_compute = np.linspace(0, 100, nbins + 1)
    prctiles_compute[0] = outlierprct
    prctiles_compute[-1] = 100 - outlierprct
    movement = np.concatenate([example['labels'][:, featidx] for example in data], axis=0)
    dtype = movement.dtype

    # bin_edges is nfeat x nbins+1
    if bin_epsilon is not None:
        bin_edges = np.zeros((nfeat, nbins + 1), dtype=dtype)
        for feati in range(nfeat):
            bin_edges[feati, :] = select_bin_edges(movement[:, feati], nbins, bin_epsilon[feati],
                                                   outlierprct=outlierprct, feati=feati)
    else:
        bin_edges = np.percentile(movement, prctiles_compute, axis=0)
        bin_edges = bin_edges.astype(dtype).T

    binnum = np.zeros(movement.shape, dtype=int)
    for i in range(nfeat):
        binnum[:, i] = np.digitize(movement[:, i], bin_edges[i, :])
    binnum = np.minimum(np.maximum(0, binnum - 1), nbins - 1)

    if nsamples is None:
        if fracsample is None:
            fracsample = 1 / nbins / 5
        nsamples = int(np.round(fracsample * movement.shape[0]))

    # for each bin, approximate the distribution
    samples = np.zeros((nsamples, nfeat, nbins), movement.dtype)
    bin_means = np.zeros((nfeat, nbins), movement.dtype)
    bin_medians = np.zeros((nfeat, nbins), movement.dtype)
    for i in range(nfeat):
        for j in range(nbins):
            movementcurr = torch.tensor(movement[binnum[:, i] == j, i])
            if movementcurr.shape[0] == 0:
                bin_means[i, j] = (bin_edges[i, j] + bin_edges[i, j + 1]) / 2.
                bin_medians[i, j] = bin_means[i, j]
                samples[:, i, j] = bin_means[i, j]
            else:
                samples[:, i, j] = np.random.choice(movementcurr, size=nsamples, replace=True)
                bin_means[i, j] = np.nanmean(movementcurr)
                bin_medians[i, j] = np.nanmedian(movementcurr)

            # kde[j,i] = KernelDensity(kernel='tophat',bandwidth=kde_bandwidth).fit(movementcurr[:,None])

    return bin_edges, samples, bin_means, bin_medians


def discretize_labels(movement, bin_edges, soften_to_ends=False):
    n = movement.shape[0]
    nfeat = bin_edges.shape[0]
    nbins = bin_edges.shape[1] - 1

    bin_centers = (bin_edges[:, 1:] + bin_edges[:, :-1]) / 2.
    bin_width = (bin_edges[:, 1:] - bin_edges[:, :-1])

    # d = np.zeros((n,nbins+1))
    labels = np.zeros((n, nfeat, nbins), dtype=movement.dtype)  # ,dtype=bool)
    if soften_to_ends:
        lastbin = 0
    else:
        lastbin = 1

    for i in range(nfeat):
        binnum = np.digitize(movement[:, i], bin_edges[i, :])
        binnum = np.minimum(nbins - 1, np.maximum(0, binnum - 1))
        # soft binning
        # don't soften into end bins
        idxsmall = (movement[:, i] < bin_centers[i, binnum]) & (binnum > lastbin)
        idxlarge = (movement[:, i] > bin_centers[i, binnum]) & (binnum < (nbins - 1 - lastbin))
        idxedge = (idxsmall == False) & (idxlarge == False)
        # distance from bin center, max should be .5
        d = (np.abs(movement[:, i] - bin_centers[i, binnum]) / bin_width[i, binnum])
        d[idxedge] = 0.
        labels[np.arange(n), i, binnum] = 1. - d
        labels[idxsmall, i, binnum[idxsmall] - 1] = d[idxsmall]
        labels[idxlarge, i, binnum[idxlarge] + 1] = d[idxlarge]

        # d[:,-1] = True
        # d[:,1:-1] = movement[:,i,None] <= bin_edges[None,1:-1,i]
        # labels[:,:,i] = (d[:,:-1] == False) & (d[:,1:] == True)

    return labels


def labels_discrete_to_continuous(labels_discrete, bin_edges):
    sz = labels_discrete.shape

    nbins = sz[-1]
    nfeat = sz[-2]
    szrest = sz[:-2]
    n = np.prod(np.array(szrest))
    labels_discrete = torch.reshape(labels_discrete, (n, nfeat, nbins))
    # nfeat x nbins
    bin_centers = (bin_edges[:, 1:] + bin_edges[:, :-1]) / 2.
    s = torch.sum(labels_discrete, dim=-1)
    assert torch.max(torch.abs(1 - s)) < .01, 'discrete labels do not sum to 1'
    movement = torch.sum(bin_centers[None, ...] * labels_discrete, dim=-1) / s
    movement = torch.reshape(movement, szrest + (nfeat,))

    return movement


def get_batch_idx(example, idx):
    if isinstance(example, np.ndarray) or torch.is_tensor(example):
        return example[idx, ...]

    example1 = {}
    for kw, v in example.items():
        if isinstance(v, np.ndarray) or torch.is_tensor(v):
            example1[kw] = v[idx, ...]
        elif isinstance(v, dict):
            example1[kw] = get_batch_idx(v, idx)

    return example1


def get_flip_idx():
    isright = np.array([re.search('right', kpn) is not None for kpn in keypointnames])
    flipidx = np.arange(len(keypointnames), dtype=int)
    idxright = np.nonzero(isright)[0]
    for ir in idxright:
        kpnr = keypointnames[ir]
        kpnl = kpnr.replace('right', 'left')
        il = keypointnames.index(kpnl)
        flipidx[ir] = il
        flipidx[il] = ir

    return flipidx


def flip_flies(X, arena_center=[0, 0], flipdim=0):
    flipX = X.copy()
    flipidx = get_flip_idx()
    for i in range(len(flipidx)):
        flipX[i, flipdim, ...] = arena_center[flipdim] - X[flipidx[i], flipdim, ...]
        flipX[i, 1 - flipdim, ...] = X[flipidx[i], 1 - flipdim, ...]

    return flipX


def split_data_by_id(data):
    splitdata = []
    nflies = data['X'].shape[-1]
    for flynum in range(nflies):
        isdata = data['isdata'][:, flynum] & (data['isstart'][:, flynum] == False)
        idxstart, idxend = get_interval_ends(isdata)
        for i in range(len(idxstart)):
            i0 = idxstart[i]
            i1 = idxend[i]
            id = data['ids'][i0, flynum]
            if data['isdata'][i0 - 1, flynum] and data['ids'][i0 - 1, flynum] == id:
                i0 -= 1
            splitdata.append({
                'flynum': flynum,
                'id': id,
                'i0': i0,
                'i1': i1,
            })
    return splitdata


def sanity_check_tspred(data, compute_feature_params, npad, scale_perfly, contextl=512, t0=510, flynum=0):
    # sanity check on computing features when predicting many frames into the future
    # compute inputs and outputs for frames t0:t0+contextl+npad+1 with tspred_global set by config
    # and inputs ant outputs for frames t0:t0+contextl+1 with just next frame prediction.
    # the inputs should match each other
    # the outputs for each of the compute_feature_params['tspred_global'] should match the next frame
    # predictions for the corresponding frame

    epsilon = 1e-6
    id = data['ids'][t0, flynum]

    # compute inputs and outputs with tspred_global = compute_feature_params['tspred_global']
    contextlpad = contextl + npad
    t1 = t0 + contextlpad - 1
    x = data['X'][..., t0:t1 + 1, :]
    xcurr1, idxinfo1 = compute_features(x, id, flynum, scale_perfly, outtype=np.float32, returnidx=True, npad=npad,
                                        **compute_feature_params)

    # compute inputs and outputs with tspred_global = [1,]
    contextlpad = contextl + 1
    t1 = t0 + contextlpad - 1
    x = data['X'][..., t0:t1 + 1, :]
    xcurr0, idxinfo0 = compute_features(x, id, flynum, scale_perfly, outtype=np.float32, tspred_global=[1, ],
                                        returnidx=True,
                                        **{k: v for k, v in compute_feature_params.items() if k != 'tspred_global'})

    assert np.all(np.abs(xcurr0['input'] - xcurr1['input']) < epsilon)
    for f in featglobal:
        # find row of np.array idxinfo1['labels']['global_feat_tau'] that equals (f,1)
        i1 = np.nonzero(
            (idxinfo1['labels']['global_feat_tau'][:, 0] == f) & (idxinfo1['labels']['global_feat_tau'][:, 1] == 1))[0][
            0]
        i0 = np.nonzero(
            (idxinfo0['labels']['global_feat_tau'][:, 0] == f) & (idxinfo0['labels']['global_feat_tau'][:, 1] == 1))[0][
            0]
        assert np.all(np.abs(xcurr1['labels'][:, i1] - xcurr0['labels'][:, i0]) < epsilon)

    return


def compare_dicts(old_ex, new_ex):
    for k, v in old_ex.items():
        if not k in new_ex:
            print(f'Missing key {k}')
        elif type(v) is torch.Tensor:
            v = v.cpu().numpy()
            newv = new_ex[k]
            if type(newv) is torch.Tensor:
                newv = newv.cpu().numpy()
            err = np.nanmax(np.abs(v - newv))
            print(f'max diff {k}: {err:e}')
        elif type(v) is np.ndarray:
            err = np.nanmax(np.abs(v - new_ex[k]))
            print(f'max diff {k}: {err:e}')
        elif type(v) is dict:
            print(f'Comparing dict {k}')
            compare_dicts(v, new_ex[k])
        else:
            try:
                err = np.nanmax(np.abs(v - new_ex[k]))
                print(f'max diff {k}: {err:e}')
            except:
                print(f'not comparing {k}')
    return


def data_to_kp_from_metadata(data, metadata, ntimepoints):
    t0 = metadata['t0']
    flynum = metadata['flynum']
    id = metadata['id']
    datakp = data['X'][:, :, t0:t0 + ntimepoints + 1, flynum].transpose(2, 0, 1)
    return datakp, id


def debug_less_data(data, T=10000):
    data['videoidx'] = data['videoidx'][:T, :]
    data['ids'] = data['ids'][:T, :]
    data['frames'] = data['frames'][:T, :]
    data['X'] = data['X'][:, :, :T, :]
    data['y'] = data['y'][:, :T, :]
    data['isdata'] = data['isdata'][:T, :]
    data['isstart'] = data['isstart'][:T, :]
    return


def compute_scale_allflies(data):
    maxid = np.max(data['ids'])
    maxnflies = data['X'].shape[3]
    scale_perfly = None

    for flynum in range(maxnflies):

        idscurr = np.unique(data['ids'][data['ids'][:, flynum] >= 0, flynum])
        for id in idscurr:
            idx = data['ids'][:, flynum] == id
            s = compute_scale_perfly(data['X'][..., idx, flynum])
            if scale_perfly is None:
                scale_perfly = np.zeros((s.size, maxid + 1))
                scale_perfly[:] = np.nan
            else:
                assert (np.all(np.isnan(scale_perfly[:, id])))
            scale_perfly[:, id] = s.flatten()

    return scale_perfly


def compute_noise_params(data, scale_perfly, sig_tracking=.25 / PXPERMM,
                         simplify_out=None, compute_pose_vel=True):
    # contextlpad = 2

    # # all frames for the main fly must have real data
    # allisdata = interval_all(data['isdata'],contextlpad)
    # isnotsplit = interval_all(data['isstart']==False,contextlpad-1)[1:,...]
    # canstart = np.logical_and(allisdata,isnotsplit)

    # X is nkeypts x 2 x T x nflies
    maxnflies = data['X'].shape[3]

    alld = 0.
    n = 0
    # loop through ids
    logging.info('Computing noise parameters...')
    for flynum in tqdm.trange(maxnflies):
        idx0 = data['isdata'][:, flynum] & (data['isstart'][:, flynum] == False)
        # bout starts and ends
        t0s = np.nonzero(np.r_[idx0[0], (idx0[:-1] == False) & (idx0[1:] == True)])[0]
        t1s = np.nonzero(np.r_[(idx0[:-1] == True) & (idx0[1:] == False), idx0[-1]])[0]

        for i in range(len(t0s)):
            t0 = t0s[i]
            t1 = t1s[i]
            id = data['ids'][t0, flynum]
            scale = scale_perfly[:, id]
            xkp = data['X'][:, :, t0:t1 + 1, flynum]
            relpose, globalpos = compute_pose_features(xkp, scale)
            movement = compute_movement(relpose=relpose, globalpos=globalpos, simplify=simplify_out,
                                        compute_pose_vel=compute_pose_vel)
            nu = np.random.normal(scale=sig_tracking, size=xkp.shape)
            relpose_pert, globalpos_pert = compute_pose_features(xkp + nu, scale)
            movement_pert = compute_movement(relpose=relpose_pert, globalpos=globalpos_pert, simplify=simplify_out,
                                             compute_pose_vel=compute_pose_vel)
            alld += np.nansum((movement_pert - movement) ** 2., axis=1)
            ncurr = np.sum((np.isnan(movement) == False), axis=1)
            n += ncurr

    epsilon = np.sqrt(alld / n)

    return epsilon.flatten()


""" Load and filter data
"""


def load_raw_npz_data(infile: str) -> dict:
    """ Loads fly data that has been pre-curated via ____.
    Args
        infile: Datafile with .npz extension. Data is expected to have the following fields:
            'X': nkpts x 2 x T x maxnflies array of floats containing pose data for all flies and frames
            'videoidx': T x 1 array of ints containing index of video pose is computed from
            'ids': T x maxnflies array of ints containing fly id
            'frames': T x 1 array of ints containing video frame number
            'y': ncategories x T x maxnflies binary matrix indicating supervised behavior categories
            'categories': ncategories list of category names
            'kpnames': nkpts list of keypoint names

    Returns
        A dictionary with the fields contained in the infile with these additional fields:
            'isdata': T x maxnflies indicating whether data is valid
            'isstart': T x maxnflies indicating whether frame is a start frame
    """
    data = {}
    with np.load(infile) as data1:
        for key in data1:
            logging.info(f'loading {key}')
            data[key] = data1[key]
    logging.info('data loaded')

    maxnflies = data['ids'].shape[1]
    # ids start at 1, make them start at 0
    data['ids'][data['ids'] >= 0] -= 1
    # starts of sequences, either because video changes or identity tracking issues
    # or because of filtering of training data
    isstart = (data['ids'][1:, :] != data['ids'][:-1, :]) | \
              (data['frames'][1:, :] != (data['frames'][:-1, :] + 1))
    isstart = np.concatenate((np.ones((1, maxnflies), dtype=bool), isstart), axis=0)

    data['isdata'] = data['ids'] >= 0
    data['isstart'] = isstart

    data['categories'] = list(data['categories'])

    return data


def filter_data_by_categories(data, categories):
    iscategory = np.ones(data['y'].shape[1:], dtype=bool)
    for category in categories:
        if category == 'male':
            category = 'female'
            val = 0
        else:
            val = 1
        catidx = data['categories'].index(category)
        iscategory = iscategory & (data['y'][catidx, ...] == val)
    data['isdata'] = data['isdata'] & iscategory


def load_and_filter_data(infile, config):
    # load data
    logging.info(f"loading raw data from {infile}...")
    data = load_raw_npz_data(infile)

    # compute noise parameters
    if (len(config['discreteidx']) > 0) and config['discretize_epsilon'] is None:
        if (config['all_discretize_epsilon'] is None):
            scale_perfly = compute_scale_allflies(data)
            config['all_discretize_epsilon'] = compute_noise_params(data, scale_perfly,
                                                                    simplify_out=config['simplify_out'])
        config['discretize_epsilon'] = config['all_discretize_epsilon'][config['discreteidx']]

    # filter out data
    logging.info('filtering data...')
    if config['categories'] is not None and len(config['categories']) > 0:
        filter_data_by_categories(data, config['categories'])

    # augment by flipping
    if 'augment_flip' in config and config['augment_flip']:
        flipvideoidx = np.max(data['videoidx']) + 1 + data['videoidx']
        data['videoidx'] = np.concatenate((data['videoidx'], flipvideoidx), axis=0)
        firstid = np.max(data['ids']) + 1
        flipids = data['ids'].copy()
        flipids[flipids >= 0] += firstid
        data['ids'] = np.concatenate((data['ids'], flipids), axis=0)
        data['frames'] = np.tile(data['frames'], (2, 1))
        flipX = flip_flies(data['X'])
        data['X'] = np.concatenate((data['X'], flipX), axis=2)
        data['y'] = np.tile(data['y'], (1, 2, 1))
        data['isdata'] = np.tile(data['isdata'], (2, 1))
        data['isstart'] = np.tile(data['isstart'], (2, 1))

    # compute scale parameters
    logging.info('computing scale parameters...')
    scale_perfly = compute_scale_allflies(data)

    if np.isnan(SENSORY_PARAMS['otherflies_touch_mult']):
        logging.info('computing touch parameters...')
        SENSORY_PARAMS['otherflies_touch_mult'] = compute_otherflies_touch_mult(data)

    # throw out data that is missing scale information - not so many frames
    idsremove = np.nonzero(np.any(np.isnan(scale_perfly), axis=0))[0]
    data['isdata'][np.isin(data['ids'], idsremove)] = False

    return data, scale_perfly


def get_real_flies(x, tgtdim=-1):
    # x is ... x ntgts
    dims = list(range(x.ndim))
    if tgtdim < 0:
        tgtdim = x.ndim + tgtdim
    dims.remove(tgtdim)

    isreal = np.all(np.isnan(x), axis=tuple(dims)) == False
    return isreal
