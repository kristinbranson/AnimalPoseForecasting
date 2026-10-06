"""APF experiment glue for the RatInABox synthetic rat.

Builds an APF `Dataset` from saved RatInABox trajectories, provides the `Sensory`
operation that turns pose into sensory-neuron firing rates, and rolls out the trained
transformer.

Entry points
------------
- `make_dataset(config, filename, ...)` -- build an `apf.dataset.Dataset` from a saved
  trajectory pkl. Wraps pose into `GlobalVelocity`, computes sensory features via the
  `Sensory` operation, and applies `Zscore` / `Discretize` per the config. Pass
  `cached_sensory_array=` to skip the slow sensory pass when you already have it.
- `make_dataset_cache_wrapper(input_file, config)` -- same, but caches the expensive
  sensory firing-rate array to disk next to the input file and reuses it on later runs.
- `Sensory` -- an `apf.dataset.Operation` that bakes sensory firing-rate computation
  into a Dataset's preprocessing chain.
- `simulate(...)` -- open-loop rollout of the trained transformer, recomputing sensory
  input from the predicted pose at each step.
- `debug_plot_sample` / `initialize_debug_plots` / `update_debug_plots` --
  training-time visualization of inputs and predictions.

Expected input pkl
------------------
    {'track':  [{'pos': (T,2), 'head_direction': (T,2), 'vel': (T,2)}, ...],
     'hidden': [{firingrate_key: (T, n_cells)}, ...],
     'env_info': {...}, 'agent_info': {...}, 'sensory_info': {name: {...}}}

Requires a patched RatInABox
----------------------------
`synthrat.sensory` calls `get_state` with `is_array=True` and the `chunk_size` /
`n_workers` / `parallel_threshold` arguments, which exist only in our fork:

    git clone -b speedup git@github.com:kristinbranson/RatInABox.git
    pip install -e RatInABox

Stock upstream RatInABox fails **silently** in two ways, so check this first if results
look wrong: `chunk_size` / `n_workers` / `parallel_threshold` are swallowed by `**kwargs`
(giving an out-of-memory error or a ~8x slowdown), and `SpeedCell` returns shape `(1,)`
instead of `(1, T)`. The other paths fail loudly.
"""

import os
import pickle
import copy
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
import tqdm.auto as tqdm


import apf.dataset
import apf.models
import apf.utils

from synthrat.sensory import (
    ORIENTATION_CONVENTION, compute_sensory, get_all_feature_names, head_direction_from_orientation,
    orientation_from_head_direction, rehydrate_agent, rehydrate_env, rehydrate_sensory,
)
from synthrat.plotting import visualize_sensory

LOG = logging.getLogger(__name__)

@dataclass
class Sensory(apf.dataset.Operation):
    """ Computes sensory neuron firing rates

    NOTE: this operation is not invertible.
    Attributes: 
        idxinfo: Keeps track of which dimensions of the sensory output correspond to what cell type
    """
    
    localattrs = ['idxinfo','feature_names','sensory_info']
    idxinfo: dict | None = None
    feature_names: list | None = None
    # dict containing 'env_info', 'agent_info', and 'sensory_info' needed to rehydrate the Sensory neurons
    info: dict | None = None 
    
    def apply(self, X: np.ndarray, info: dict | None = None, isdata: np.ndarray | None = None) -> np.ndarray:
        """ Computes sensory features from keypoints.

        Args:
            X: (x, y, orientation, vel_x, vel_y) position of the agents, (n_agents,  n_frames, 5) float array.
                Orientation follows the fly convention, heading - pi/2 (see synthrat.sensory.ORIENTATION_OFFSET).
            isdata: indicates whether there is data for a given frame or agent, only used to speed up computation, 
            (n_frames, n_agents) bool array

        Returns:
            sensory_data: (n_agents,  n_frames, n_sensory_features) float array
        """
        
        szrest = X.shape[:-2]
        ndrest = len(szrest)
        T, d = X.shape[-2:]
        if ndrest != 1:
            X = X.reshape([-1, T, d])
            if isdata is not None:
                isdata = isdata.reshape([-1, T])
        
        # Allow re-applying the operation (e.g. during predict_iterative
        # rollout) without explicitly re-passing info — fall back to whatever
        # was set on the instance from the original call. Otherwise overwrite.
        if info is None:
            assert self.info is not None, "Sensory.apply needs info, either as a kwarg or set on self.info"
            info = self.info
        else:
            self.info = info
        self.idxinfo = {}
        sensory_data = None
        for ratid in range(X.shape[0]):
            if isdata is not None:
                isdatacurr = isdata[ratid]
                head_direction = head_direction_from_orientation(X[ratid,isdatacurr,2])
                track = {
                    'pos': X[ratid,isdatacurr,0:2],
                    'head_direction': head_direction
                }
            else:
                isdatacurr = None
                head_direction = head_direction_from_orientation(X[ratid,:,2])
                track = {
                    'pos': X[ratid,:,0:2],
                    'head_direction': head_direction
                }
            featscurr = compute_sensory(track,info=self.info)
            if ratid == 0:
                # store idxinfo
                idxoff = 0
                idxinfo = {}
                for k,v in featscurr.items():
                    ncurr = v.shape[1]
                    idxinfo[k] = (idxoff, idxoff + ncurr)
                    idxoff += ncurr
                sensory_data = np.full((X.shape[0],X.shape[1],idxoff),np.nan)
            sensory_data_curr = np.concatenate(list(featscurr.values()), axis=1)
            if isdatacurr is not None:
                sensory_data[ratid,isdatacurr,:] = sensory_data_curr
            else:
                sensory_data[ratid,:,:] = sensory_data_curr
                
        if ndrest != 1:
            sensory_data = sensory_data.reshape(list(szrest) + list(sensory_data.shape[-2:]))
            if isdata is not None:
                isdata = isdata.reshape(list(szrest) + [isdata.shape[-1]])
        self.idxinfo = idxinfo
        self.feature_names = get_all_feature_names(self.info)
        return sensory_data

    def invert(self, sensory: np.ndarray) -> None:
        LOG.error(f"Operation {self} is not invertible")
        return None
    
    def __str__(self):
        s = f"Operation {self.name} of class Sensory with idxinfo keys:\n"
        if self.idxinfo is not None:
            for key in self.idxinfo:
                s += f"  {key}: {self.idxinfo[key]}\n"
        else:
            s += "  idxinfo is None\n"
        return s[:-1]
    
    def update_feature_names(self, input_feature_names):
        return self.feature_names
    
    def invert_feature_names(self, input_feature_names):
        LOG.error(f"Operation {self.name} is not invertible")
        return None

        
def make_dataset(
        config: dict,
        filename: str,
        ref_dataset: apf.dataset.Dataset | None = None,
        return_all: bool = False,
        debug: bool = True,
        data: dict | None = None,
        cached_sensory_array: np.ndarray | None = None,
) -> apf.dataset.Dataset | tuple[apf.dataset.Dataset, np.ndarray, apf.dataset.Data, apf.dataset.Data, apf.dataset.Data, apf.dataset.Data]:
    """ Creates a dataset from config, for a given file name and optionally using a reference dataset.

    Args:
        config: Config for loading the data
        filename: Name of file to read the data from (e.g. 'intrainfile', 'invalfile')
        ref_dataset: Dataset to copy postprocessing operations from.
            When loading validation/test set, provide training set.
        return_all: Whether to return intermediate variables in addition to Dataset (see returns)
        debug: Whether to use less data for debugging
        data: Optionally provide pre-loaded data dict to avoid re-loading from file.
        cached_sensory_array: Optional precomputed sensory firing-rate array of
            shape (n_agents, n_total_frames, n_sensory_features). If provided,
            Sensory.apply is skipped and this array is wrapped in a Data with
            the matching Sensory operation. The shape must match what
            Sensory.apply would have produced for the given inputs (otherwise
            downstream Zscore/Discretize/etc. shapes won't line up).
        data dict: {
            'track': [{
                'pos': (T,2) array of x,y positions
                'head_direction': (T,2) array of unit vectors
            }, ...] list of episodes
            'hidden': [{
                firingrate_key: (T, n_cells) array of firing rates for each hidden neuron type
            }, ...] list of episodes
            'env_info': dict of environment setup
            'agent_info': dict of agent setup
            'sensory_info': {
                sensory_key: dict of sensory cell setup 
            }
        }
    

    Returns:
        dataset: Dataset for flyllm experiment.
        [
          track: x,y,orientation
          velocity: Egocentric velocity data
          sensory: Sensory data
        ]
    """
    
    # Load data
    if data is None:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

    dt = data['agent_info']['dt']

    if debug:
        n_episodes = 5
        data['track'] = data['track'][:n_episodes]
        data['hidden'] = data['hidden'][:n_episodes]

    info = {
        'env_info': data['env_info'],
        'agent_info': data['agent_info'],
        'sensory_info': data['sensory_info']
    }

    # create track from X which is (n_agents, n_frames, 5):
    # x, y, orientation, vel_x, vel_y

    # concatenate all episodes together and create isstart arrays to keep track of 
    # episode boundaries
    episode_lengths = [len(ep['pos']) for ep in data['track']]
    ntotal_frames = np.sum(episode_lengths)
    isstart = np.zeros(ntotal_frames, dtype=bool)
    isstart[0] = True
    isstart[np.cumsum(episode_lengths)[:-1]] = True
    pos = np.concatenate([ep['pos'] for ep in data['track']], axis=0)
    head_direction = np.concatenate([ep['head_direction'] for ep in data['track']], axis=0)
    # fly convention (heading - pi/2), so that GlobalVelocity's first feature is forward movement
    orientation = orientation_from_head_direction(head_direction)
    # vel can be computed from np.diff(pos,axis=0)/dt, except for first time point
    vel = np.concatenate([ep['vel'] for ep in data['track']], axis=0)
    X = np.concatenate([pos,orientation[:,None]],axis=-1)
    
    pose = apf.dataset.Data('pos',X[None])

    # position_velocity = apf.dataset.Data('pos_vel', X[None,...], [], feature_names=['x', 'y', 'orientation', 'vel_x', 'vel_y'])

    if cached_sensory_array is not None:
        # Skip the expensive Sensory.apply call by wrapping the cached firing-
        # rate array in a Data with the same Sensory operation metadata.
        sensory_op = Sensory()
        sensory_op.info = info
        # Replicate what Sensory.apply normally writes onto the operation +
        # propagates onto the resulting Data, so downstream ops (Zscore, etc.)
        # see proper feature_names and idxinfo instead of None.
        feature_names = get_all_feature_names(info)
        # idxinfo (per-population slice into the concatenated firing-rate axis)
        # is needed if downstream code (e.g., debug_plot_sample) wants to split
        # the array back per population.
        rehydrated_sensory = rehydrate_sensory(
            rehydrate_agent(rehydrate_env(info['env_info']), info['agent_info']),
            info['sensory_info'],
        )
        idxinfo, off = {}, 0
        for pop_name, neuron in rehydrated_sensory.items():
            idxinfo[pop_name] = (off, off + neuron.n)
            off += neuron.n
        sensory_op.idxinfo = idxinfo
        sensory_op.feature_names = feature_names
        sensory = apf.dataset.Data('sensory', cached_sensory_array,
                                   operations=[sensory_op],
                                   feature_names=feature_names)
    else:
        sensory = Sensory()(pose, info=info)

    # (n_frames,) -> (n_frames, n_agents), the orientation both Dataset.compute_sessions
    # and set_invalid_ends (via GlobalVelocity) expect. Note this is the opposite of the
    # agent-first convention used for the data arrays themselves.
    isstart = isstart[:,None]

    velocity = apf.dataset.GlobalVelocity(tspred=[1,])(pose,isstart=isstart)

    args = {
        'context_length': config['contextl'],
        'isstart': isstart,
        # Carry the true pose (x, y, orientation) per chunk. The velocity labels alone
        # invert to a trajectory starting at the origin with zero heading
        # (GlobalVelocity.invert defaults x0 to zeros), so debug plots need the real
        # pose to place the agent and align the sensory overlays.
        'metadata': {'pose': X[None]},
    }

    # Assemble the dataset
    if ref_dataset is not None:
        dataset = apf.dataset.Dataset(
            inputs=apf.dataset.apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apf.dataset.apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            **args
        )
    elif 'dataset_params' in config and config['dataset_params'] is not None and \
        ('inputs' in config['dataset_params']) and ('labels' in config['dataset_params']):
        dataset = apf.dataset.Dataset(
            inputs=apf.dataset.apply_opers_from_data_params(config['dataset_params']['inputs'], {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels=apf.dataset.apply_opers_from_data_params(config['dataset_params']['labels'], {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            **args
        )
    else:
        # velocity = OddRoot(5)(velocity)

        # discretize everything
        discreteidx = config['discreteidx']    

        # Need to zscore before binning, otherwise bin_epsilon values need to be divided by zscore stds
        zscored_velocity = apf.dataset.Zscore()(velocity)

        zsig = zscored_velocity.operations[-1].std
        bin_config = {'nbins': config['discretize_nbins'],
                      'bin_epsilon': config['discretize_epsilon'] / zsig[discreteidx]}

        if 'bin_edges_absolute' in config and config['bin_edges_absolute'] is not None and len(config['bin_edges_absolute']) > 0:
            # check that all discreteidx are present
            assert all(idx in config['bin_edges_absolute'] for idx in discreteidx), "Not all discreteidx are present in bin_edges_absolute"
            bin_config['bin_edges'] = np.vstack([config['bin_edges_absolute'][featidx]/zsig[featidx] for featidx in discreteidx]) # nfeat x nbins + 1

        dataset = apf.dataset.Dataset(
            inputs={
                'velocity': apf.dataset.Zscore()(apf.dataset.Roll(dt=1)(velocity)),
                'sensory': apf.dataset.Zscore()(sensory),
            },
            labels={
                'velocity': apf.dataset.Discretize(**bin_config)(zscored_velocity),
            },
            **args
        )
    dataset_params = dataset.get_params()
    if return_all:
        return dataset, info, pose, velocity, sensory, dataset_params, isstart
    else:
        return dataset, info

def debug_plot_sample(example_in, ratinabox_info, nplot=3, pred=None, fig=None, ax=None):
    """Visualize nplot random batch elements via visualize_sensory.

    Layout: nplot rows × (1 trajectory + n_sensory) columns. The trajectory
    column shows the inverted pose plus vector-cell ellipses; subsequent
    columns show one panel per non-vector sensory population.

    Parameters
    ----------
    example_in : dict from dataset.item_to_data — must have 'labels'/'velocity'
        and 'inputs'/'sensory' Data objects.
    ratinabox_info : dict with 'Env' and 'Sensory' (live populations).
    dataset : (unused, accepted for caller compatibility)
    nplot : number of batch elements to draw.
    pred : (unused for now; reserved for overlaying model predictions).
    fig, ax : if both None, a new figure is created and returned.
        If provided, ax must be the list-of-dicts structure that this function
        previously returned (one dict per row, mapping 'traj' / population
        names to Axes); the panels are cleared and redrawn so subsequent
        update calls reuse the same figure.

    Returns
    -------
    fig, ax : matplotlib Figure and the per-row list of Axes-dicts.
    """
    # Prefer the true pose carried in metadata. Inverting the velocity labels alone
    # yields a trajectory integrated from the origin with zero initial heading
    # (GlobalVelocity.invert x0=0), which would misplace the agent in the corner and
    # rotate the heading, so the sensory overlays would not match the firing rates.
    if example_in.get('metadata', {}).get('pose') is not None:
        pose_true = np.asarray(example_in['metadata']['pose'])
    else:
        pose_true = apf.dataset.apply_inverse_operations(example_in['labels']['velocity'])
    sensory_op = apf.dataset.get_operation(example_in['inputs']['sensory'].operations, 'sensory')
    fr_all = example_in['inputs']['sensory'].array      # (B, T, total_cells)

    Sensory = ratinabox_info['Sensory']
    pop_names = list(Sensory.keys())
    n_cols = 1 + len(pop_names)
    t = pose_true.shape[1] - 1
    batch_size = pose_true.shape[0]
    samples_plot = np.random.choice(batch_size, size=min(nplot, batch_size), replace=False)
    nrows = len(samples_plot)

    # --- Figure / axes setup ----------------------------------------------
    if fig is None and ax is None:
        fig, ax_grid = plt.subplots(nrows, n_cols, figsize=(5 * n_cols, 5 * nrows),
                                    squeeze=False)
        ax = []
        for i in range(nrows):
            ax_row = {'traj': ax_grid[i, 0]}
            for j, name in enumerate(pop_names):
                ax_row[name] = ax_grid[i, 1 + j]
            ax.append(ax_row)
    else:
        # Reuse prior axes; clear them so the redraw doesn't pile up.
        if fig is None:
            fig = ax[0]['traj'].figure
        for ax_row in ax:
            for axcurr in ax_row.values():
                axcurr.clear()

    # --- Draw each sample --------------------------------------------------
    for ax_row, sample_idx in zip(ax, samples_plot):
        sensory_curr = {name: fr_all[sample_idx, ..., start:end]
                        for name, (start, end) in sensory_op.idxinfo.items()}
        visualize_sensory(pose_true[sample_idx], sensory_curr, t,
                          ratinabox_info['Env'], Sensory,
                          fig=fig, ax=ax_row)

        # Overlay the predicted trajectory + final state on the trajectory
        # panel, in a different color, so it's directly comparable to the
        # ground truth drawn by visualize_sensory.
        if pred is not None:
            pose_p = pred[sample_idx]                           # (T, 3)
            ax_row['traj'].scatter(pose_p[:t + 1, 0], pose_p[:t + 1, 1],
                                   s=12, alpha=0.5, linewidth=0,
                                   c='C3', zorder=3)
            ax_row['traj'].scatter(pose_p[t, 0], pose_p[t, 1],
                                   s=80, c='C3', linewidth=0, zorder=6)

    return fig, ax


def initialize_debug_plots(dataset, dataloader, ratinabox_info, name='', nplot=3):

    example_batch = next(iter(dataloader))
    example = dataset.item_to_data(apf.utils.convert_torch_to_numpy(example_batch))

    # plot to visualize input features
    figsample, axsample = debug_plot_sample(example, ratinabox_info, nplot=nplot)

    axsample[0]['traj'].set_title(name)
    figsample.tight_layout()

    hdebug = {
        'figsample': figsample,
        'axsample': axsample,
        'example': example
    }

    return hdebug


def update_debug_plots(hdebug, config, model, dataset, ratinabox_info, example, pred,
                       criterion=None, name='', nplot=3):
    if config['modelstatetype'] == 'prob':
        pred1 = model.maxpred({k: v.detach() for k, v in pred.items()})
    elif config['modelstatetype'] == 'best':
        pred1 = model.randpred(pred.detach())
    else:
        if isinstance(pred, dict):
            pred1 = {k: v.detach().cpu() for k, v in pred.items()}
        else:
            pred1 = pred.detach().cpu()
    # `example` from the training loop is a raw batched torch dict
    # ({'input', 'labels', 'labels_discrete', 'metadata'}). Convert to the
    # Data-dict form ({'inputs', 'labels'} of Data objects) that
    # debug_plot_sample expects.
    example_data = dataset.item_to_data(apf.utils.convert_torch_to_numpy(example))

    # Build a "prediction example" that uses pred1 in place of the discrete
    # labels (with the example's metadata so invertdata is intact), then run
    # the same inversion to recover a (B, T, 3) predicted pose array. We use
    # do_sampling=False so the inverted bins become the deterministic weighted
    # average of bin centers.
    pred_item = {'metadata': example['metadata']}
    if isinstance(pred1, dict):
        if 'labels_discrete' in pred1:
            pred_item['labels_discrete'] = pred1['labels_discrete']
        elif 'discrete' in pred1:
            pred_item['labels_discrete'] = pred1['discrete']
        if 'labels' in pred1:
            pred_item['labels'] = pred1['labels']
        elif 'continuous' in pred1:
            pred_item['labels'] = pred1['continuous']
    else:
        pred_item['labels_discrete'] = pred1
    pred_data = dataset.item_to_data(apf.utils.convert_torch_to_numpy(pred_item))
    # Integrate the predicted per-step velocity from the true starting pose so the
    # predicted trajectory overlay lines up with the ground truth (rather than
    # starting from the origin). x0 is (B, 3) = (x, y, orientation) at the first frame.
    extraargs = {'discretize': {'do_sampling': False}}
    if example_data.get('metadata', {}).get('pose') is not None:
        extraargs['globalvelocity'] = {
            'x0': np.asarray(example_data['metadata']['pose'])[:, 0, :]
        }
    pose_pred = apf.dataset.apply_inverse_operations(
        pred_data['labels']['velocity'], extraargs=extraargs)

    debug_plot_sample(example_data, ratinabox_info, nplot=nplot,
                      fig=hdebug['figsample'], ax=hdebug['axsample'], pred=pose_pred)
    hdebug['axsample'][0]['traj'].set_title(name)
    hdebug['figsample'].tight_layout()


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

# Bump only when the cached sensory-firing-rate computation itself changes. The
# isstart transpose and the Sensory/Data API changes in the port do NOT affect the
# cached array: make_dataset_from_cache reuses only cached['sensory_array'] (which is
# per-frame firing rates, independent of isstart) and recomputes isstart fresh, and
# the firing-rate math is unchanged from the code that built the existing caches.
CACHE_VERSION = 1

def input_file_to_cache_path(input_file):
    """
    cache_path = input_file_to_cache_path(input_file)
    Computing sensory information is slow. To speed up repeated runs during development, we cache the 
    preprocessed sensory arrays to disk. Given an input file path, return the corresponding cache file 
    path for storing the preprocessed dataset, constructed as 
    {input_file_directory}/`apf_cache_v{CACHE_VERSION}_`{input_file_basename}.pkl. 
    The cache file name is derived from the input file name and includes a version number to manage cache 
    invalidation when the preprocessing logic changes.
    Arguments:
    input_file: str, path to the raw input data file for which to compute the cache path.
    Returns:
    str, path to the cache file corresponding to the input file.
    """
    
    base = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(os.path.dirname(input_file),
                        f'apf_cache_v{CACHE_VERSION}_{base}.pkl')

def save_cache(path, data):
    """
    save_cache(path, data_dict)
    Save the preprocessed dataset arrays to a cache file. Only plain numpy arrays are
    stored -- no operations, dataset params, or ratinabox_info, which can carry
    references to code (e.g. the rect_polar_grid cell_arrangement callable) and are all
    re-derived from the raw input file on load anyway. This keeps the cache a pure data
    artifact that loads without importing any project module.

    The only array actually consumed on load is `sensory_array` (the expensive one);
    the others are kept for provenance/inspection.
    Arguments:
    path: str, file path where the cache should be saved.
    data: dict with 'pose', 'velocity', 'sensory' Data objects and 'isstart' array.
    """

    cached = {
        'sensory_array':  data['sensory'].array,           # the expensive one
        'pose_array':     data['pose'].array,
        'velocity_array': data['velocity'].array,
        'isstart':        data['isstart'],
    }
    print(f'Saving cache to: {path}')
    with open(path, 'wb') as f:
        pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

def make_dataset_from_cache(input_file, cache_path, config, debug_uselessdata=False):
    """
    make_dataset_from_cache(input_file, cache_path, config, debug_uselessdata=False)
    Load the preprocessed dataset components from a cache file and reconstruct the dataset and ratinabox_info. This 
    function reads the cached raw data arrays and ratinabox_info from the specified cache file and then calls 
    make_dataset with the cached sensory array to reconstruct the dataset. The cache file is expected to contain:
    - ratinabox_info: dict-of-dicts containing environment and sensory setup info needed for rehydration.
    - pose_array: raw pose data array from the dataset (before any operations).
    - velocity_array: raw velocity data array from the dataset (before any operations).
    - sensory_array: raw sensory firing-rate array from the dataset (before any operations)
    - isstart: array indicating episode start points.
    Arguments:
    input_file: str, path to the raw input data file (used for consistency but not directly read since we're loading from cache).
    cache_path: str, file path from which to load the cached data.
    config: dict, configuration parameters for dataset creation (passed to make_dataset).
    debug_uselessdata: bool, if True, enables debug mode for handling useless data (passed to make_dataset).
    Returns:
    dataset: the reconstructed dataset object using the cached sensory array.
    data_dict: dict containing raw data arrays for pose, velocity, sensory, and isstart loaded from the cache.
    ratinabox_info: dict-of-dicts containing environment and sensory setup info needed for rehydration, loaded from the cache.
    """
    
    print(f'Loading cache from: {cache_path}')
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    data = {}
    dataset, ratinabox_info, data['pose'], data['velocity'], \
        data['sensory'], _, data['isstart'] = \
            make_dataset(config=config, filename=input_file,
                        debug=debug_uselessdata, return_all=True,
                        cached_sensory_array=cached['sensory_array'])
    return dataset, data, ratinabox_info

def make_dataset_cache_wrapper(input_file, config, debug_uselessdata=False):
    """
    dataset,data,ratinabox_info = make_dataset_cache_wrapper(input_file, config, debug_uselessdata=False)
    Wrapper function for make_dataset that handles caching of the preprocessed sensory arrays. This function first checks if a 
    cache file exists for the given input file and loads the dataset from cache if available. 
    If no cache is found, it calls make_dataset to build the dataset from scratch, which includes the expensive computation of sensory
    firing rates. After building the dataset, it saves the relevant components to cache for future runs. The cache file path is derived 
    from the input file path and includes a version number for cache management (see input_file_to_cache_path). 
    Arguments:
    input_file: str, path to the raw input data file for which to create the dataset (and corresponding cache).
    config: dict, configuration parameters for dataset creation (passed to make_dataset).
    debug_uselessdata: bool, if True, enables debug mode for handling useless data (passed to make_dataset and make_dataset_from_cache).
    Returns:
    dataset: the dataset object created either from cache or from scratch.
    data_dict: dict containing raw data arrays for pose, velocity, sensory, and isstart, either loaded from cache or computed from scratch.
    ratinabox_info: dict-of-dicts containing environment and sensory setup info needed for rehydration, either loaded from cache or computed from scratch
    """
    
    cache_path = input_file_to_cache_path(input_file)
    if (not debug_uselessdata) and os.path.exists(cache_path):
        print(f'Loading cache: {cache_path}')
        return make_dataset_from_cache(input_file, cache_path, config, debug_uselessdata)
    print(f'Building (slow), will cache to: {cache_path}')
    data = {}
    dataset, ratinabox_info, data['pose'], data['velocity'], \
        data['sensory'], _, data['isstart'] = \
            make_dataset(config=config, filename=input_file,
                        debug=debug_uselessdata, return_all=True)
    if not debug_uselessdata:
        save_cache(cache_path, data)
    return dataset, data, ratinabox_info


def check_orientation_convention(checkpoint: dict) -> None:
    """Refuses a saved synthrat model trained with a different orientation convention.

    Synthrat orientation follows the fly convention (heading - pi/2; synthrat.sensory), so the
    velocity features are (forward, sideways, turn). Models trained before that convention was
    adopted used orientation = heading, which puts lateral movement in feature 0 and forward
    movement in feature 1; with the current code such a model would receive inputs and predict
    labels in a different order from the one it learned, without any error. Pass this function
    as apf.io.load_model(..., check_state=check_orientation_convention).

    Args:
        checkpoint: dict loaded from a model file written by apf.io.save_model; its 'config'
            entry is the config the model was trained with.

    Raises:
        ValueError: if the saved config's 'orientation_convention' is missing or differs from
            synthrat.sensory.ORIENTATION_CONVENTION.
    """
    saved = (checkpoint.get('config') or {}).get('orientation_convention')
    if saved != ORIENTATION_CONVENTION:
        raise ValueError(
            f"This synthrat model was trained with orientation convention {saved!r}, but the current "
            f"code uses {ORIENTATION_CONVENTION!r} (orientation = heading - pi/2). Models saved before "
            "the convention was adopted have none recorded; their velocity features 0 and 1 are "
            "swapped relative to the current data, so the model must be retrained.")


def simulate(
    dataset: apf.dataset.Dataset,
    model: apf.models.TransformerModel,
    pose: apf.dataset.Data,  # TODO: embed this in dataset?
    track_len: int = 1000,
    burn_in: int | None = None,
    max_contextl: int | None = -1,
    agent_ids: list[int] | None = None,
    start_frame: int | None = None,
    progress_bar: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """ Simulates an agent given model and some initialization.

    Args:
        dataset: Dataset that defines which operations need to be applied to the data, also used for burn-in.
        model: Transformer model used for predicting future motion.
        track: Track array corresponding to dataset (used for ground truth comparison). TODO: Can we make this optional?
        pose: Pose array corresponding to dataset (used for ground truth comparison). TODO: Can we make this optional?
        identities: Maps frame and agent id to a unique individual id. Corresponds to dataset.
        track_len: How long the total track should be after simulation (including burnin).
        burn_in: How many frames to use for the initialization.
        max_contextl: Max number of frames to feed as input to the model. If None, uses the full history.
        agent_ids: Which agents to simulate. If None, simulates all the agents.
        start_frame: Which start frame to use for the initialization.

    Returns:
        gt_track: ground truth 2d track at each frame. Used for first burn_in frames.
        pred_track: 2d track at each frame based on open loop simulation.
    """
    
    if burn_in is None:
        burn_in = dataset.context_length
    if max_contextl == -1:
        max_contextl = dataset.context_length
    if start_frame is None:
        start_frame = burn_in
    
    if agent_ids is None:
        n_agents = pose.array.shape[0]
        agent_ids = np.arange(n_agents)
    n_sim_agents = len(agent_ids)

    # Extract ground truth
    gt_input = []
    for agent_idx in agent_ids:
        # Get data input chunk for this agent
        gt_chunk = dataset.get_chunk(start_frame=start_frame, duration=track_len, agent_id=agent_idx)
        gt_input.append(gt_chunk['input'])
    gt_input = np.array(gt_input)
    gt_pose = pose.array[:, start_frame:start_frame + track_len]

    # Initialize model input
    device = next(model.parameters()).device
    model_input = torch.full((n_sim_agents, track_len, gt_input.shape[-1]),torch.nan)
    curr_frame = burn_in
    model_input[:, :curr_frame, :] = torch.from_numpy(gt_input[:, :curr_frame])
    model_input = model_input.to(device)

    # Initialize tracks (set the future to nan for all agents to be simulated)
    pred_pose = copy.deepcopy(gt_pose)
    pred_pose[agent_ids, curr_frame + 1:] = np.nan

    if progress_bar:
        pbar = tqdm.trange(curr_frame, track_len, desc='Simulating')
    else:        
        pbar = range(curr_frame, track_len)

    masksizeprev = 0
    model.eval()
    for curr_frame in pbar:
        # Make a motion prediction
        frame0 = 0
        if max_contextl is not None:
            frame0 = max(0, curr_frame - max_contextl)

        masksize = curr_frame - frame0
        if masksize != masksizeprev:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(masksize, device=device)
            masksizeprev = masksize

        # Apply model to previous frames
        with torch.no_grad():
            pred = model.output(model_input[:, frame0:curr_frame, :], mask=mask, is_causal=True)

        # Extract velocity from the prediction
        proc_velocity = dataset.split_output_by_names(pred)['velocity'][:, -1:, :]

        # Invert preprocessing operations to obtain raw velocity
        velocity_operations = dataset.labels['velocity'].operations
        preproc_opers =  apf.dataset.get_post_operations(velocity_operations, 'globalvelocity')
        velocity =  apf.dataset.apply_inverse_operations(proc_velocity, preproc_opers)

        # Apply velocity to current pose
        curr_pose = pred_pose[agent_ids, curr_frame - 1]
        velocity_op =  apf.dataset.get_operation(velocity_operations, 'globalvelocity')
        velocity = np.concatenate([velocity, np.zeros_like(velocity)], axis=1)
        new_pose = velocity_op.invert(velocity, x0=curr_pose)[:, -1, :]
        pred_pose[agent_ids, curr_frame] = new_pose

        if np.isnan(new_pose).sum() > 0:
            print(f"Predicted pose contains a NaN, aborting at frame {curr_frame}.")
            break

        # Compute sensory information
        sensory_op =  apf.dataset.get_operation(dataset.inputs['sensory'].operations, 'sensory')
        sensory = sensory_op.apply(pred_pose[:, curr_frame:curr_frame+1])

        # Assemble inputs and apply the same preproc operations to the data as the training data
        inputs = {'velocity': velocity[:, :1, :],
                  'sensory': sensory[agent_ids]}
        inputs_proc = {}
        preproc_operations = apf.dataset.get_post_operations(dataset.inputs['velocity'].operations, 'roll')
        inputs_proc['velocity'] = apf.dataset.apply_operations(inputs['velocity'], preproc_operations)
        preproc_operations = apf.dataset.get_post_operations(dataset.inputs['sensory'].operations, 'sensory')
        inputs_proc['sensory'] = apf.dataset.apply_operations(inputs['sensory'], preproc_operations)
        # Operations applied to a raw ndarray return a bare ndarray (only Data input
        # yields Data), so unwrap defensively rather than assuming either form.
        curr_in = np.concatenate(
            [getattr(x, 'array', x) for x in inputs_proc.values()], axis=-1)
        model_input[:, curr_frame, :] = torch.from_numpy(curr_in[:, 0, :].astype(np.float32)).to(device)

    return gt_pose, pred_pose
