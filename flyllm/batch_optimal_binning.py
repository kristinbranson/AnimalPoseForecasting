"""
Find optimal bin edges for the discretized outputs of a config, as notebooks/flyllm_optimal_binning.py does, but as
a script that runs headless.  Run it as python -m flyllm.batch_optimal_binning.  Works for any animal
(flyllm/animals.py).

Unlike the notebook, uses the config's own categories (the notebook overrides them to ['courtship']); pass
--overrideconfig '{"categories": ["courtship"]}' to do as the notebook does.  Everything goes in <out dir>, with
file names starting <animal>llm_binning_<model type>_<timestamp>, e.g. flyllm_binning_...:
    _percentiles.{png,pdf}: percentiles of each discretized feature's velocity
    _<feature index>_<feature name>_K<bin count>.{png,pdf}: each feature's bin widths, edges and transition matrix
    _bininfo.npz: each feature's bin edges, transition matrix and log likelihood
    _bin_edges.txt: each feature's bin edges, as printed, to copy into config files

Doesn't use the GPU, so it can run on CPU-only nodes.
"""

import argparse
import json
import logging
import os
import time

from apf.lsf import limit_threads_to_lsf_slots

limit_threads_to_lsf_slots()

import matplotlib

# a non-interactive backend, so this runs headless
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import SymmetricalLogLocator

from apf.io import get_modeltype_str
from flyllm import animals, optimal_binning
from flyllm.animals import Animal
from flyllm.prepare import init_config

LOG = logging.getLogger('flyllm.batch_optimal_binning')


def save_figure(fig, file_path_without_extension: str):
    """ Saves fig as a PNG and a PDF file, and closes it. """
    for extension in ['png', 'pdf']:
        fig.savefig(f'{file_path_without_extension}.{extension}')
    plt.close(fig)


def plot_percentiles(animal: Animal, config, velocity, isdata, useoutputmask):
    """ Returns a figure of the percentiles of the velocity of each of config['discreteidx']'s features, positive
    and negative values separately, on log-log axes. """
    featangle, posenames = animal.featangle, animal.posenames
    nfeat = len(config['discreteidx'])
    prcts = np.logspace(-5, 2, 100)
    prcts = np.vstack((100 - prcts, prcts)).T
    POS = 1
    NEG = 0

    prctvals = np.zeros((nfeat, 2, len(prcts))) + np.nan
    for idx, featidx in enumerate(config['discreteidx']):
        feature_data = velocity.array[..., featidx].copy()
        feature_data = feature_data[(isdata & useoutputmask).T & ~np.isnan(feature_data)]
        if featangle[featidx]:
            feature_data *= 180 / np.pi
        prctvals[idx, POS] = np.percentile(feature_data[feature_data >= 0], prcts[:, 0])
        prctvals[idx, NEG] = -np.percentile(feature_data[feature_data <= 0], prcts[:, 1])

    minprctval = 1e-8
    fig, ax = plt.subplots(nfeat, 2, figsize=(6 * 2, 6 * nfeat), sharex=False, squeeze=False)
    for idx, featidx in enumerate(config['discreteidx']):
        ax[idx, POS].plot(np.maximum(100 - prcts[:, 0], minprctval), prctvals[idx, 0], '.')
        ax[idx, POS].set_ylabel(f'{idx}: +{posenames[featidx]}')
        ax[idx, POS].invert_xaxis()
        ax[idx, NEG].plot(np.maximum(prcts[:, 1], minprctval), prctvals[idx, 1], '.')
        ax[idx, NEG].set_ylabel(f'{idx}: -{posenames[featidx]}')
        for i in range(2):
            ax[idx, i].set_yscale('log')
            ax[idx, i].set_xscale('log')
            ax[idx, i].yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=1, subs=range(1, 10)))
            ax[idx, i].minorticks_on()
            ax[idx, i].grid(True, which='both', ls='--')
    ax[-1, POS].set_xlabel('100-Percentile')
    ax[-1, NEG].set_xlabel('Percentile')
    fig.tight_layout()
    return fig


def optimize_bins(animal: Animal, config, velocity, useoutputmask) -> list[dict]:
    """ Returns, for each of config['discreteidx']'s features, a dict of the optimal bin edges for its velocity,
    the transition matrix between the bins, and the log likelihood. """
    posenames = animal.posenames
    K = config['discretize_nbins']
    allbininfo = []
    for idx, featidx in enumerate(config['discreteidx']):
        LOG.info(f'Optimizing bins for feature {featidx}: {posenames[featidx]} '
                 f'({idx + 1} of {len(config["discreteidx"])})')

        # data for current feature
        feature_data = velocity.array[..., featidx].copy()
        feature_data[useoutputmask.T == 0] = np.nan  # mask out invalid data

        # split and subsample
        sequences = optimal_binning.split_sequences(feature_data)
        optimize_bins_params = config.get('optimize_bins', {}).copy()
        nsamples = optimize_bins_params.pop('nsamples', None)
        sequences = optimal_binning.subsample_sequences(sequences, nsamples=nsamples)

        all_outlier_thresh = optimize_bins_params.pop('all_outlier_thresh', None)
        prctile_outlier_thresh = optimize_bins_params.pop('prctile_outlier_thresh', None)
        outlier_thresh = None
        if all_outlier_thresh is not None:
            outlier_thresh = all_outlier_thresh[featidx]
        elif prctile_outlier_thresh is not None:
            outlier_thresh = np.percentile(feature_data[~np.isnan(feature_data)],
                                           [prctile_outlier_thresh, 100 - prctile_outlier_thresh])

        all_discretize_epsilon = config.get('all_discretize_epsilon', None)
        min_width = all_discretize_epsilon[featidx] if all_discretize_epsilon is not None else 0.0
        LOG.info(f'outlier_thresh: {outlier_thresh}, min_width: {min_width}')

        edges_fixed, boundary_edges = optimal_binning.set_boundary_edges(sequences, K, outlier_thresh)
        edges, prob_matrix, log_lik = optimal_binning.optimize_bin_edges(
            sequences,
            K=K,
            boundary_edges=boundary_edges,
            min_width=min_width,
            edges_fixed=edges_fixed,
            **optimize_bins_params
        )
        allbininfo.append({'edges': edges, 'prob_matrix': prob_matrix, 'log_lik': log_lik,
                           'featidx': featidx, 'posename': posenames[featidx]})
    return allbininfo


def plot_bininfo(animal: Animal, bininfo: dict):
    """ Returns a figure of one feature's bin widths vs. index and center, edges vs. index, and transition
    matrix, from bininfo, one of optimize_bins's dicts. """
    featangle = animal.featangle
    featidx = bininfo['featidx']
    edges = bininfo['edges']
    widths = np.diff(edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    if featangle[featidx]:
        edges = edges * 180 / np.pi
        widths = widths * 180 / np.pi
        bin_centers = bin_centers * 180 / np.pi
        units = 'deg/fr'
    else:
        units = f'{animal.length_unit}/fr'

    fig, ax = plt.subplots(4, 1, figsize=(10, 20))
    ax[0].plot(widths, '.')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Bin index')
    ax[0].set_ylabel(f'Bin width ({units})')

    ax[1].plot(bin_centers, widths, '.')
    ax[1].set_yscale('log')
    ax[1].set_xscale('symlog')
    ax[1].set_xlabel(f'Bin center ({units})')
    ax[1].set_ylabel(f'Bin width ({units})')

    ax[2].plot(edges, '.')
    ax[2].set_xlabel('Bin index')
    ax[2].set_ylabel(f'Bin edge ({units})')
    ax[2].set_yscale('symlog')

    ax[3].imshow(bininfo['prob_matrix'])
    ax[3].set_xlabel('To bin index')
    ax[3].set_ylabel('From bin index')
    ax[0].set_title(f'Feature {featidx}: {bininfo["posename"]} velocity')

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='config file name')
    parser.add_argument('--outdir', default='.', help='directory for all output, created if needed (default: .)')
    parser.add_argument('--overrideconfig', type=json.loads, default={},
                        help='JSON dict of config entries to override, e.g. \'{"categories": ["courtship"]}\'')
    parser.add_argument('--debug-uselessdata', action='store_true',
                        help='use a small subset of the data, for quick debugging runs')
    args = parser.parse_args()
    if not isinstance(args.overrideconfig, dict):
        parser.error('--overrideconfig must be a JSON dict')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    config_file_path = os.path.abspath(args.configfile)
    out_dir_path = os.path.abspath(args.outdir)
    os.makedirs(out_dir_path, exist_ok=True)
    LOG.info(f'Config overrides: {args.overrideconfig}')
    LOG.info(f'Writing all output to {out_dir_path}')
    overrideconfig = {'device': 'cpu', **args.overrideconfig}

    config = init_config(configfile=config_file_path, overrideconfig=overrideconfig)['config']
    animal = animals.config_animal(config)
    _, _, _, _, velocity, _, _, isdata, _, useoutputmask, _, _ = \
        animal.make_dataset(config, 'intrainfile', return_all=True, debug=args.debug_uselessdata)
    # a fresh config, as the notebook does, in case making the dataset changed it
    config = init_config(configfile=config_file_path, overrideconfig=overrideconfig)['config']

    timestamp = time.strftime('%Y%m%dT%H%M%S')
    out_file_path_prefix = os.path.join(out_dir_path,
                                        f'{animal.name}llm_binning_{get_modeltype_str(config)}_{timestamp}')
    LOG.info(f'Output file names start with {out_file_path_prefix}')

    save_figure(plot_percentiles(animal, config, velocity, isdata, useoutputmask),
                f'{out_file_path_prefix}_percentiles')

    allbininfo = optimize_bins(animal, config, velocity, useoutputmask)
    K = config['discretize_nbins']
    for bininfo in allbininfo:
        save_figure(plot_bininfo(animal, bininfo),
                    f'{out_file_path_prefix}_{bininfo["featidx"]}_{bininfo["posename"]}_K{K}')

    bininfo_file_path = f'{out_file_path_prefix}_bininfo.npz'
    LOG.info(f'Saving binning info to {bininfo_file_path}')
    np.savez_compressed(bininfo_file_path, bininfo=allbininfo)

    # print bin edges so that they can be copy-pasted into config files, and save them too
    edges_file_path = f'{out_file_path_prefix}_bin_edges.txt'
    LOG.info(f'Saving bin edges to {edges_file_path}')
    with open(edges_file_path, 'w') as f:
        for bininfo in allbininfo:
            line = f"{bininfo['featidx']}: {np.array2string(bininfo['edges'], separator=', ')}"
            print(line)
            f.write(line + '\n')


if __name__ == '__main__':
    main()
