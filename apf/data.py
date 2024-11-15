import numpy as np
import tqdm
import re
import torch
import logging

from apf.utils import get_interval_ends

LOG = logging.getLogger(__name__)

#TODO: Change flynum to agent_num safely


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

    # all frames for the main agent must have real data
    allisdata = interval_all(data['isdata'], contextlpad)
    isnotsplit = interval_all(data['isstart'] == False, contextlpad - 1)[1:, ...]
    canstart = np.logical_and(allisdata, isnotsplit)

    # X is nkeypts x 2 x T x n_agents
    nkeypoints = data['X'].shape[0]
    T = data['X'].shape[-2]
    max_n_agents = data['X'].shape[-1]
    assert T > 2 * contextlpad, f'Assumption that data has more frames than 2*(contextl+1) is incorrect, code will fail, T = {T}, contextl={contextl}'

    # last possible start frame = T - contextl
    maxt0 = canstart.shape[0] - 1
    # X is a dict with chunked data
    X = []
    # loop through ids
    nframestotal = 0
    for agent_num in tqdm.trange(max_n_agents, desc='Agent'):
        # choose a first frame near the beginning, but offset a bit
        # first possible start
        canstartidx = np.nonzero(canstart[:, agent_num])[0]
        if canstartidx.size == 0:
            continue

        mint0curr = canstartidx[0]
        # offset a bit
        t0 = mint0curr + np.random.randint(0, contextl, None)
        # find the next allowed frame
        if canstart[t0, agent_num] == False:
            if not np.any(canstart[t0:, agent_num]):
                continue
            t0 = np.nonzero(canstart[t0:, agent_num])[0][0] + t0

        maxt0curr = canstartidx[-1]
        # maxt1curr = maxt0curr+contextlpad-1
        ndata = np.count_nonzero(data['isdata'][:, agent_num])
        maxintervals = ndata // contextl + 1
        for i in tqdm.trange(maxintervals, desc='Interval'):
            if t0 > maxt0:
                break
            # this is guaranteed to be < T
            t1 = t0 + contextlpad - 1
            id = data['ids'][t0, agent_num]
            xcurr = reparamfun(data['X'][..., t0:t1 + 1, :], id, agent_num, npad=npad)
            metadata = {'flynum': agent_num, 'id': id, 't0': t0, 'frame0': data['frames'][t0, 0]}
            if 'videoidx' in data:
                metadata['videoidx'] = data['videoidx'][t0, 0]
            else:
                metadata['sessionidx'] = data['sessionids'][id]
            xcurr['metadata'] = metadata
            xcurr['categories'] = data['y'][:, t0:t1 + 1, agent_num].astype(np.float32)
            X.append(xcurr)
            if t0 + contextl >= maxt0curr:
                break
            elif canstart[t0 + contextl, agent_num]:
                t0 = t0 + contextl
            else:
                t0 = np.nonzero(canstart[t1 + 1:, agent_num])[0]
                if t0 is None or t0.size == 0:
                    break
                t0 = t0[0] + t1 + 1
            nframestotal += contextl

    LOG.info(f'In total {nframestotal} frames of data after chunking')

    return X


def select_bin_edges(movement, nbins, bin_epsilon, outlierprct=0, feati=None):
    n = movement.shape[0]
    lims = np.percentile(movement, [outlierprct, 100 - outlierprct])
    max_bin_epsilon = (lims[1] - lims[0]) / (nbins + 1)
    if bin_epsilon >= max_bin_epsilon:
        LOG.info(
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
    return fit_discretize_data(
        data=np.concatenate([example['labels'][:, featidx] for example in data], axis=0),
        nbins=nbins,
        bin_epsilon=bin_epsilon,
        outlierprct=outlierprct,
        fracsample=fracsample,
        nsamples=nsamples,
    )


def fit_discretize_data(data, nbins=50, bin_epsilon=None, outlierprct=.001, fracsample=None, nsamples=None):
    """
    Args:
        data: n_frames x n_feat, float
        ...
    """
    # compute percentiles
    nfeat = data.shape[1]
    prctiles_compute = np.linspace(0, 100, nbins + 1)
    prctiles_compute[0] = outlierprct
    prctiles_compute[-1] = 100 - outlierprct
    dtype = data.dtype

    # bin_edges is nfeat x nbins+1
    if bin_epsilon is not None:
        bin_edges = np.zeros((nfeat, nbins + 1), dtype=dtype)
        for feati in range(nfeat):
            bin_edges[feati, :] = select_bin_edges(data[:, feati], nbins, bin_epsilon[feati],
                                                   outlierprct=outlierprct, feati=feati)
    else:
        bin_edges = np.percentile(data, prctiles_compute, axis=0)
        bin_edges = bin_edges.astype(dtype).T

    binnum = np.zeros(data.shape, dtype=int)
    for i in range(nfeat):
        binnum[:, i] = np.digitize(data[:, i], bin_edges[i, :])
    binnum = np.minimum(np.maximum(0, binnum - 1), nbins - 1)

    if nsamples is None:
        if fracsample is None:
            fracsample = 1 / nbins / 5
        nsamples = int(np.round(fracsample * data.shape[0]))

    # for each bin, approximate the distribution
    samples = np.zeros((nsamples, nfeat, nbins), data.dtype)
    bin_means = np.zeros((nfeat, nbins), data.dtype)
    bin_medians = np.zeros((nfeat, nbins), data.dtype)
    for i in range(nfeat):
        for j in range(nbins):
            movementcurr = torch.tensor(data[binnum[:, i] == j, i])
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
        # nans will get put in the last bin...
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


def get_flip_idx(keypointnames):
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


def flip_agents(X, keypointnames, arena_center=[0, 0], flipdim=0):
    flipX = X.copy()
    flipidx = get_flip_idx(keypointnames)
    for i in range(len(flipidx)):
        flipX[i, flipdim, ...] = arena_center[flipdim] - X[flipidx[i], flipdim, ...]
        flipX[i, 1 - flipdim, ...] = X[flipidx[i], 1 - flipdim, ...]

    return flipX


def split_data_by_id(data):
    splitdata = []
    n_agents = data['X'].shape[-1]
    for agent_num in range(n_agents):
        isdata = data['isdata'][:, agent_num] & (data['isstart'][:, agent_num] == False)
        idxstart, idxend = get_interval_ends(isdata)
        for i in range(len(idxstart)):
            i0 = idxstart[i]
            i1 = idxend[i]
            id = data['ids'][i0, agent_num]
            if data['isdata'][i0 - 1, agent_num] and data['ids'][i0 - 1, agent_num] == id:
                i0 -= 1
            splitdata.append({
                'flynum': agent_num,
                'id': id,
                'i0': i0,
                'i1': i1,
            })
    return splitdata


def compare_dicts(old_ex, new_ex, maxerr=None):
    for k, v in old_ex.items():
        err = 0.
        if not k in new_ex:
            LOG.info(f'Missing key {k}')
        elif type(v) is torch.Tensor:
            v = v.cpu().numpy()
            newv = new_ex[k]
            if type(newv) is torch.Tensor:
                newv = newv.cpu().numpy()
            err = np.nanmax(np.abs(v - newv))
            LOG.info(f'max diff {k}: {err:e}')
        elif type(v) is np.ndarray:
            err = np.nanmax(np.abs(v - new_ex[k]))
            LOG.info(f'max diff {k}: {err:e}')
        elif type(v) is dict:
            LOG.info(f'Comparing dict {k}')
            compare_dicts(v, new_ex[k])
        else:
            try:
                err = np.nanmax(np.abs(v - new_ex[k]))
                LOG.info(f'max diff {k}: {err:e}')
            except:
                LOG.info(f'not comparing {k}')
        if maxerr is not None:
            assert err < maxerr

    return


def data_to_kp_from_metadata(data, metadata, ntimepoints):
    t0 = metadata['t0']
    agent_num = metadata['flynum']
    id = metadata['id']
    datakp = data['X'][:, :, t0:t0 + ntimepoints + 1, agent_num].transpose(2, 0, 1)
    return datakp, id


def debug_less_data(data, n_frames_per_video=10000, max_n_videos=1):
    frame_ids = [np.where(data['videoidx'] == idx)[0][:n_frames_per_video] for idx in np.unique(data['videoidx'])]
    frame_ids = np.concatenate(frame_ids[:max_n_videos])

    data['videoidx'] = data['videoidx'][frame_ids, :]
    data['ids'] = data['ids'][frame_ids, :]
    data['frames'] = data['frames'][frame_ids, :]
    data['X'] = data['X'][:, :, frame_ids, :]
    data['y'] = data['y'][:, frame_ids, :]
    data['isdata'] = data['isdata'][frame_ids, :]
    data['isstart'] = data['isstart'][frame_ids, :]
    return


""" Load and filter data
"""


def load_raw_npz_data(infile: str) -> dict:
    """ Loads tracking data.
    Args
        infile: Datafile with .npz extension. Data is expected to have the following fields:
            'X': nkpts x 2 x T x max_n_agents array of floats containing pose data for all agents and frames
            'videoidx': T x 1 array of ints containing index of video pose is computed from
            'ids': T x max_n_agents array of ints containing agent id
            'frames': T x 1 array of ints containing video frame number
            'y': ncategories x T x max_n_agents binary matrix indicating supervised behavior categories
            'categories': ncategories list of category names
            'kpnames': nkpts list of keypoint names

    Returns
        A dictionary with the fields contained in the infile with these additional fields:
            'isdata': T x max_n_agents indicating whether data is valid
            'isstart': T x max_n_agents indicating whether frame is a start frame
    """
    data = {}
    with np.load(infile) as data1:
        for key in data1:
            LOG.info(f'loading {key}')
            data[key] = data1[key]
    LOG.info('data loaded')

    # if ids start at 1, make them start at 0
    min_valid_id = data['ids'][data['ids'] > -1].min()
    if min_valid_id > 0:
        data['ids'][data['ids'] >= 0] -= 1

    # starts of sequences, either because video changes or identity tracking issues
    # or because of filtering of training data
    max_n_agents = data['ids'].shape[1]
    isstart = (data['ids'][1:, :] != data['ids'][:-1, :]) | \
              (data['frames'][1:, :] != (data['frames'][:-1, :] + 1))
    isstart = np.concatenate((np.ones((1, max_n_agents), dtype=bool), isstart), axis=0)
    data['isstart'] = isstart

    data['isdata'] = data['ids'] >= 0
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


def get_real_agents(x, tgtdim=-1):
    """
        isreal = get_real_flies(x)

        Input:
        x: ndarray of arbitrary dimensions, as long as the tgtdim-dimension corresponds to targets.
        tgtdim: dimension corresponding to targets. default: -1 (last)

        Returns which flies in the input ndarray x correspond to real data (are not nan).
    """
    # x is ... x ntgts
    dims = list(range(x.ndim))
    if tgtdim < 0:
        tgtdim = x.ndim + tgtdim
    dims.remove(tgtdim)

    isreal = np.all(np.isnan(x), axis=tuple(dims)) == False
    return isreal
