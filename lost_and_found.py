import numpy as np
import torch
import tqdm
import sklearn.cluster
import sklearn.decomposition
import matplotlib.pyplot as plt
import gzip
import pickle

from plotting import debug_plot_batch_pose
from features import compute_features


def to_size(sz):
    if sz is None:
        sz = (1,)
    elif isinstance(sz, int):
        sz = (sz,)
    elif isinstance(sz, list):
        sz = tuple(sz)
    elif isinstance(sz, tuple):
        pass
    else:
        raise ValueError('Input sz must be an int, list, or tuple')
    return sz


def init_train_bpe(zlabels, transform=True, max_val=10,
                   n_clusters=int(1e3)):
    def apply_transform(z):
        x = np.zeros(z.shape)
        x = np.sqrt(np.minimum(max_val, np.abs(z))) * np.sign(z)
        return x

    def apply_inverse_transform(x):
        z = np.zeros(x.shape)
        z = x ** 2 * np.sign(z)
        return z

    # for |x| <= transform_thresh, use sqrt. above, use log
    if transform:
        x = apply_transform(zlabels)
    else:
        x = zlabels.copy()

    # k-means clustering of inter-frame motion
    alg = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters)
    token = alg.fit_predict(x)
    centers = alg.cluster_centers_
    err = np.abs(x - centers[token, :])

    cmap = plt.get_cmap("rainbow")
    colors = cmap(np.arange(n_clusters) / n_clusters)
    colors = colors[np.random.permutation(n_clusters), :] * .7

    nplot = 1000
    nstd = 1
    fig, ax = plt.subplots(2, 1, sharex='all')
    ax[0].cla()
    ax[1].cla()
    for i in range(x.shape[1]):
        xrecon = centers[token[:nplot], i]
        ax[0].plot([0, nplot], [i * nstd, ] * 2, ':', color=[.5, .5, .5])
        ax[0].plot(np.arange(nplot), i * nstd * 2 + x[:nplot, i], 'k.-')
        tmpy = np.c_[xrecon, x[:nplot, i], np.zeros(nplot)]
        tmpy[:, 2] = np.nan
        tmpx = np.tile(np.arange(nplot)[:, None], (1, 3))
        ax[0].plot(tmpx.flatten(), i * nstd * 2 + tmpy.flatten(), 'k-')
        ax[0].scatter(np.arange(nplot), i * nstd * 2 + xrecon, c=colors[token[:nplot], :], marker='o')
        # ax[0].text(0,2*nstd*(i+.5),outnames[i],horizontalalignment='left',verticalalignment='top')

    ax[1].plot(np.arange(nplot), token[:nplot], 'k-')
    ax[1].scatter(np.arange(nplot), token[:nplot], c=colors[token[:nplot], :], marker='o')
    ax[1].set_ylabel('Token ID')
    return


def train_bpe(data, scale_perfly, simplify_out=None):
    # collect motion data
    nflies = data['X'].shape[3]
    T = data['X'].shape[2]

    isdata = np.any(np.isnan(data['X']), axis=(0, 1)) == False
    isstart = (data['ids'][1:, :] != data['ids'][:-1, :]) | \
              (data['frames'][1:, :] != (data['frames'][:-1, :] + 1))
    isstart = np.r_[np.ones((1, nflies), dtype=bool), isstart]
    labels = None

    print('Computing movements over all data')
    for i in tqdm.trange(nflies, desc='animal'):
        isstart = isdata[1:, i] & \
                  ((isdata[:-1, i] == False) | \
                   (data['ids'][1:, i] != data['ids'][:-1, i]) | \
                   (data['frames'][1:, 0] != (data['frames'][:-1, 0] + 1)))
        isstart = np.r_[isdata[0, i], isstart]
        isend = isdata[:-1, i] & \
                ((isdata[1:, i] == False) | \
                 (data['ids'][1:, i] != data['ids'][:-1, i]) | \
                 (data['frames'][1:, 0] != (data['frames'][:-1, 0] + 1)))
        isend = np.r_[isend, isdata[-1, i]]
        t0s = np.nonzero(isstart)[0]
        t1s = np.nonzero(isend)[0] + 1
        for j in tqdm.trange(len(t0s), desc='frames'):
            t0 = t0s[j]
            t1 = t1s[j]
            id = data['ids'][t0, i]
            xcurr = compute_features(data['X'][:, :, t0:t1, :], id, i, scale_perfly, None,
                                     simplify_in='no_sensory')
            # simplify_out=simplify_out)
            labelscurr = xcurr['labels']
            if labels is None:
                labels = labelscurr
            else:
                labels = np.r_[labels, labelscurr]

        # zscore
        mu = np.mean(labels, axis=0)
        sig = np.std(labels, axis=0)
        zlabels = (labels - mu) / sig

    return


def stackhelper(all_pred, all_labels, all_mask, all_pred_discrete, all_labels_discrete, nplot):
    predv = torch.stack(all_pred[:nplot], dim=0)
    if len(all_mask) > 0:
        maskv = torch.stack(all_mask[:nplot], dim=0)
    else:
        maskv = None
    labelsv = torch.stack(all_labels[:nplot], dim=0)
    s = list(predv.shape)
    s[2] = 1
    nan = torch.zeros(s, dtype=predv.dtype)
    nan[:] = torch.nan
    predv = torch.cat((predv, nan), dim=2)
    predv = predv.flatten(0, 2)
    labelsv = torch.cat((labelsv, nan), dim=2)
    labelsv = labelsv.flatten(0, 2)
    if maskv is not None:
        maskv = torch.cat((maskv, torch.zeros(s[:-1], dtype=bool)), dim=2)
        maskv = maskv.flatten()
    if len(all_pred_discrete) > 0:
        pred_discretev = torch.stack(all_pred_discrete[:nplot], dim=0)
        s = list(pred_discretev.shape)
        s[2] = 1
        nan = torch.zeros(s, dtype=pred_discretev.dtype)
        nan[:] = torch.nan
        pred_discretev = torch.cat((pred_discretev, nan), dim=2)
        pred_discretev = pred_discretev.flatten(0, 2)
    else:
        pred_discretev = None
    if len(all_labels_discrete) > 0:
        pred_discretev = torch.stack(all_labels_discrete[:nplot], dim=0)
        s = list(pred_discretev.shape)
        s[2] = 1
        nan = torch.zeros(s, dtype=pred_discretev.dtype)
        nan[:] = torch.nan
        pred_discretev = torch.cat((pred_discretev, nan), dim=2)
        pred_discretev = pred_discretev.flatten(0, 2)
    else:
        pred_discretev = None

    return predv, labelsv, maskv, pred_discretev


def debug_add_noise(train_dataset, data, idxsample=0, tsplot=None):
    # debugging adding noise
    train_dataset.set_eval_mode()
    extrue = train_dataset[idxsample]
    train_dataset.set_train_mode()
    exnoise = train_dataset[idxsample]
    exboth = {}
    for k in exnoise.keys():
        if type(exnoise[k]) == torch.Tensor:
            exboth[k] = torch.stack((extrue[k], exnoise[k]), dim=0)
        elif type(exnoise[k]) == dict:
            exboth[k] = {}
            for k1 in exnoise[k].keys():
                exboth[k][k1] = torch.stack((torch.tensor(extrue[k][k1]), torch.tensor(exnoise[k][k1])))
        else:
            raise ValueError('huh')
    if tsplot is None:
        tsplot = np.round(np.linspace(0, 64, 4)).astype(int)
    hpose, ax, fig = debug_plot_batch_pose(exboth, train_dataset, data=data, tsplot=tsplot)
    Xfeat_true = train_dataset.get_Xfeat(example=extrue, use_todiscretize=True)
    Xfeat_noise = train_dataset.get_Xfeat(example=exnoise, use_todiscretize=True)


def gzip_pickle_dump(filename, data):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)


def gzip_pickle_load(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def save_chunked_data(savefile, d):
    gzip_pickle_dump(savefile, d)
    return


def load_chunked_data(savefile):
    return gzip_pickle_load(savefile)
