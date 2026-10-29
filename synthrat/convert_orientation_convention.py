"""Convert a synthrat model trained with the old orientation convention to the fly convention.

Before synthrat adopted the fly convention (orientation = heading - pi/2; see synthrat.sensory),
synthrat orientation was the heading itself, and the velocity features computed by
apf.dataset.GlobalVelocity came out as (left, forward, turn) instead of (forward, right, turn). In
the current convention
    forward = old feature 1,   right = -(old feature 0),   turn = old feature 2.
Nothing else the model sees changes: the sensory firing rates depend on the physical heading, which
is the same. So an old model can be converted exactly, without retraining:
  - inputs: the z-scored velocity enters the model through one linear layer
    (encoder.encoder_dict.velocity); its columns are rearranged the same way, the lateral one
    negated. Z-scoring commutes with this, since the new means are the old ones rearranged and
    negated, and the standard deviations the old ones rearranged.
  - outputs: one linear layer (decoder) gives n_bins logits per velocity feature, in feature-major
    order. The forward and lateral blocks are swapped, and the lateral block's bins reversed:
    negating a value mirrors the bins, so new bin j is old bin n_bins - 1 - j.
  - saved dataset parameters (z-score means and standard deviations, bin edges, centers and
    samples) are rearranged to match, so the checkpoint stays self-consistent.
The converted checkpoint records orientation_convention in its config, so
experiments.synthrat.check_orientation_convention accepts it. Models trained in the AnimalPoseForecasting
clone record their Sensory operation under that clone's module, synthrat.apf_ratinabox; the
converted checkpoint names its current module, experiments.synthrat, so its dataset can be rebuilt
from the saved parameters (config['dataset_params']) in this repository. The optimizer and scheduler state
are dropped (Adam's moment estimates would need the same rearrangement); resuming training from a
converted file starts a fresh optimizer.

Usage:
    python -m synthrat.convert_orientation_convention <old.pth> <new.pth>
Keep the new name ending in epoch<N>, e.g. <modeltype>_<savetime>_flyorientation_bestepoch100.pth,
so that apf.io.parse_modelfile still recovers the model type and save time from it.
"""
import argparse
import copy
import os

import numpy as np
import torch

from synthrat.sensory import ORIENTATION_CONVENTION

# New velocity feature i is OLD_TO_NEW_SIGN[i] * old feature OLD_TO_NEW_ORDER[i].
OLD_TO_NEW_ORDER = [1, 0, 2]
OLD_TO_NEW_SIGN = np.array([1., -1., 1.])
N_VELOCITY_FEATURES = len(OLD_TO_NEW_ORDER)

# Layers of the synthrat model that touch the velocity features.
VELOCITY_ENCODER_WEIGHT = 'encoder.encoder_dict.velocity.weight'   # (d_model, n_velocity_features)
DECODER_WEIGHT = 'decoder.weight'                                   # (n_velocity_features * n_bins, d_model)
DECODER_BIAS = 'decoder.bias'                                       # (n_velocity_features * n_bins,)

# Checkpoint entries left out of the converted file.
DROPPED_ENTRIES = ['lr_optimizer', 'scheduler']

# Where saved operations' classes now live, by the module recorded in older checkpoints.
MODULE_RENAMES = {'synthrat.apf_ratinabox': 'experiments.synthrat'}


def rearrange_values(values: np.ndarray, negate: bool) -> np.ndarray:
    """Rearranges per-feature values along their last axis from the old to the new feature order.

    Args:
        values: (..., N_VELOCITY_FEATURES) array of per-feature values, old order.
        negate: whether to apply OLD_TO_NEW_SIGN (for quantities that change sign with the feature,
            such as means), or only reorder (for quantities that do not, such as standard deviations).

    Returns:
        same shape, new order.
    """
    values = np.asarray(values)[..., OLD_TO_NEW_ORDER]
    return values * OLD_TO_NEW_SIGN if negate else values


def rearrange_bins(binned: np.ndarray, feature_axis: int) -> np.ndarray:
    """Rearranges per-feature, per-bin values (bin edges, centers or samples) to the new order.

    A negated feature's values are negated and its bins reversed, so they stay increasing.

    Args:
        binned: array with N_VELOCITY_FEATURES along feature_axis and bins along the last axis,
            old order, in z-scored units.
        feature_axis: the axis indexing features.

    Returns:
        same shape, new order.
    """
    binned = np.moveaxis(np.asarray(binned), feature_axis, 0)               # (n_features, ..., n_bins)
    converted = np.stack([sign * binned[old][..., ::-1] if sign < 0 else binned[old]
                          for old, sign in zip(OLD_TO_NEW_ORDER, OLD_TO_NEW_SIGN)])
    return np.moveaxis(converted, 0, feature_axis)


def convert_dataset_params(dataset_params: dict) -> dict:
    """Converts saved dataset operation parameters (Dataset.get_params()) to the new convention.

    Args:
        dataset_params: {'inputs': {key: [operation dicts]}, 'labels': {...}}, as saved with a model.

    Returns:
        a converted deep copy: the velocity input's and label's Zscore and Discretize parameters are
        rearranged, and operation modules are renamed per MODULE_RENAMES.
    """
    converted = copy.deepcopy(dataset_params)
    for group in ['inputs', 'labels']:
        for ops in converted[group].values():
            for op in ops:
                op['module'] = MODULE_RENAMES.get(op['module'], op['module'])
    for group in ['inputs', 'labels']:
        for op in converted[group]['velocity']:
            attributes = op['attributes']
            if op['class'] == 'Zscore':
                attributes['mean'] = rearrange_values(attributes['mean'], negate=True)
                attributes['std'] = rearrange_values(attributes['std'], negate=False)
            elif op['class'] == 'Discretize':
                attributes['bin_edges'] = rearrange_bins(attributes['bin_edges'], feature_axis=0)
                attributes['bin_centers'] = rearrange_bins(attributes['bin_centers'], feature_axis=0)
                attributes['bin_samples'] = rearrange_bins(attributes['bin_samples'], feature_axis=1)
                epsilon = attributes['fit_discretize_data_args'].get('bin_epsilon')
                if epsilon is not None:
                    attributes['fit_discretize_data_args']['bin_epsilon'] = rearrange_values(epsilon, negate=False)
    return converted


def convert_weights(weights: dict, n_bins: int) -> dict:
    """Converts the model weights that touch the velocity features.

    Args:
        weights: model state dict (name -> tensor), old convention.
        n_bins: number of bins per velocity label feature.

    Returns:
        a new state dict with the velocity encoder's columns and the decoder's rows rearranged; all
        other tensors are shared with the input.
    """
    converted = dict(weights)
    encoder = weights[VELOCITY_ENCODER_WEIGHT]
    assert encoder.shape[1] == N_VELOCITY_FEATURES, f'{VELOCITY_ENCODER_WEIGHT} has shape {tuple(encoder.shape)}'
    sign = torch.as_tensor(OLD_TO_NEW_SIGN, dtype=encoder.dtype)
    converted[VELOCITY_ENCODER_WEIGHT] = encoder[:, OLD_TO_NEW_ORDER] * sign

    for name in [DECODER_WEIGHT, DECODER_BIAS]:
        tensor = weights[name]
        assert tensor.shape[0] == N_VELOCITY_FEATURES * n_bins, f'{name} has shape {tuple(tensor.shape)}'
        blocks = tensor.reshape((N_VELOCITY_FEATURES, n_bins) + tuple(tensor.shape[1:]))   # (feature, bin, ...)
        new_blocks = [blocks[old].flip(0) if s < 0 else blocks[old] for old, s in zip(OLD_TO_NEW_ORDER, OLD_TO_NEW_SIGN)]
        converted[name] = torch.stack(new_blocks).reshape(tensor.shape)
    return converted


def convert_checkpoint(checkpoint: dict) -> dict:
    """Converts a whole old-convention synthrat checkpoint.

    Args:
        checkpoint: dict loaded from a model file written by apf.io.save_model.

    Returns:
        the converted checkpoint: model weights, dataset parameters and config converted, loss kept,
        optimizer and scheduler state dropped.

    Raises:
        ValueError: if the checkpoint already records the current convention.
    """
    config = checkpoint.get('config') or {}
    if config.get('orientation_convention') == ORIENTATION_CONVENTION:
        raise ValueError('this checkpoint already uses the current orientation convention')
    discretize = next(op for op in checkpoint['dataset_params']['labels']['velocity'] if op['class'] == 'Discretize')
    n_bins = np.shape(discretize['attributes']['bin_centers'])[-1]

    converted = {key: value for key, value in checkpoint.items() if key not in DROPPED_ENTRIES}
    converted['model'] = convert_weights(checkpoint['model'], n_bins)
    converted['dataset_params'] = convert_dataset_params(checkpoint['dataset_params'])
    converted['config'] = dict(config, orientation_convention=ORIENTATION_CONVENTION)
    return converted


def main() -> None:
    """Command-line entry point: converts <old.pth> and writes <new.pth>, which must not exist."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('old_checkpoint', help='model file trained with the old convention')
    parser.add_argument('new_checkpoint', help='where to write the converted model; must not exist')
    args = parser.parse_args()
    if os.path.exists(args.new_checkpoint):
        raise FileExistsError(f'{args.new_checkpoint} exists; not overwriting')
    checkpoint = torch.load(args.old_checkpoint, map_location='cpu', weights_only=False)
    torch.save(convert_checkpoint(checkpoint), args.new_checkpoint)
    print(f'wrote {args.new_checkpoint}')


if __name__ == '__main__':
    main()
