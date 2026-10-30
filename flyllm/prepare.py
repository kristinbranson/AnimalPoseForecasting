import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import datetime
import os
from matplotlib import animation
import pickle
import copy
import hashlib
import json

from apf.utils import get_dct_matrix, compute_npad, save_animation
from apf.data import process_test_data, interval_all, debug_less_data, chunk_data
from apf.io import read_config, get_modeltype_str, load_and_filter_data, save_model, load_model, parse_modelfile, load_config_from_model_file
from flyllm.config import read_config, keypointnames
from flyllm.features import compute_features, sanity_check_tspred, get_sensory_feature_idx, compute_scale_perfly, compute_pose_distribution_stats
from flyllm.pose import FlyExample, FlyPoseLabels, FlyObservationInputs
from flyllm.plotting import (
    initialize_debug_plots, 
    initialize_loss_plots, 
    update_debug_plots,
    update_loss_plots,
    debug_plot_global_histograms, 
    debug_plot_dct_relative_error, 
    debug_plot_global_error, 
    debug_plot_predictions_vs_labels,
    select_featidx_plot,
)
from apf.models import (
    initialize_model, 
    initialize_loss,
    sanity_check_temporal_dep,
    update_loss_nepochs,
    generate_square_full_mask,
    criterion_wrapper
)
# from apf.models_unused import compute_loss, stack_batch_list
from apf.dataset import Dataset, DataLoader
from apf.training import init_optimizer
from experiments.flyllm import make_dataset
from flyllm.simulation import animate_predict_open_loop
from flyllm.prediction import predict_all
from IPython.display import HTML

import logging
LOG = logging.getLogger(__name__)

mpl_backend = plt.get_backend()
if mpl_backend == 'inline':
    from IPython import display

def init_config(configfile=None,config=None,mode='train',loadmodelfile=None,overrideconfig={},res={}, weights_only=True):
    """
    res = init_config(configfile=None,config=None,mode='train',loadmodelfile=None,res={})
    Read configuration from configfile and optionally loadmodelfile
    Inputs:
    configfile: str, path to configuration file
    config: dict, configuration dictionary read friom configfile
    mode: str, one of ['train','test'], whether training or testing
    loadmodelfile: str, path to model file to load
    res: dict, dictionary to store results
    Outputs:
    res: dict, updated dictionary with the following keys:
        'config': dict, configuration dictionary
    """
    
    if config is None:
        assert configfile is not None, "No configuration file provided"        
        config = read_config(configfile) # flyllm.config.read_config
        
    if overrideconfig is not None:
        for k in overrideconfig:
            config[k] = overrideconfig[k]
    
    if mode in ['test']:
        # set loadmodelfile from config if not specified
        if (loadmodelfile is None) and ('loadmodelfile' in config):
            loadmodelfile = config['loadmodelfile']
        assert loadmodelfile is not None, "No model file provided"
        load_config_from_model_file(loadmodelfile=loadmodelfile,config=config, weights_only=weights_only)
        assert 'dataset_params' in config, 'dataset_params not in config'
        
    res['config'] = config
    return res

def init_state(config=None,seedrandom=True,res={}):
    """
    res = init_state(config,seedrandom=True,res={})
    Initialize run state variables: random seed, and torch device
    Inputs:
    config: dict, configuration dictionary
    seedrandom: bool, whether to seed the random number generators, default = True
    res: dict, dictionary to store results with the following keys:
        'device': torch.device, device to run torch computations on
    """

    LOG.info('CUDA available: ' + str(torch.cuda.is_available()))
    LOG.info('matplotlib backend: ' + mpl_backend)

    assert config is not None, "No configuration provided"    
    if seedrandom:
        # seed the random number generators
        np.random.seed(config['numpy_seed'])
        torch.manual_seed(config['torch_seed'])
        
    # set device (cuda/cpu)
    device = torch.device(config['device'])
    if device.type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA is not available'

    # float32 matmul precision: 'highest' is full fp32 (the PyTorch default), 'high' allows TF32 on
    # Ampere and newer GPUs, which is much faster and has a negligible effect on training
    matmul_precision = config.get('float32_matmul_precision', 'highest')
    torch.set_float32_matmul_precision(matmul_precision)
    LOG.info(f'float32 matmul precision: {matmul_precision}')
        
    res['device'] = device
    
    
    return res

def init_raw_data(config=None,quickdebugdatafile=None,needtraindata=True,needvaldata=True,res={}):
    """
    res = init_raw_data(config=None,quickdebugdatafile=None,needtraindata=True,needvaldata=True,res={})
    Load raw data from files specified in config, and filter according to configuration parameters.
    Inputs:
    config: dict, configuration dictionary    
    quickdebugdatafile: str, path to small data file with data already filtered, used for speeding up loading
    and debugging. If None, load data from config['intrainfile'] and config['invalfile']. Default = None
    needtraindata: bool, whether to load training data. Default = True
    needvaldata: bool, whether to load validation data. Default = True
    res: dict, dictionary to store results with the following keys:
        'data': dict, training data dictionary
        'scale_perfly': ndarray of shape nscale x nflies with scale data for each fly
        'valdata': dict, validation data dictionary
        'val_scale_perfly': ndarray of shape nscale x nflies with scale data for each fly
    """

    assert config is not None, "No configuration provided"
    
    res['data'] = None
    res['scale_perfly'] = None
    res['valdata'] = None
    res['val_scale_perfly'] = None
    
    ## load raw data
    if quickdebugdatafile is None:
        if needtraindata:
            data, scale_perfly = load_and_filter_data(config['intrainfile'], config, compute_scale_perfly,
                                                    keypointnames=keypointnames)
            LOG.info(f"training data X shape: {data['X'].shape}")
            res['data'] = data
            res['scale_perfly'] = scale_perfly
        if needvaldata:
            valdata, val_scale_perfly = load_and_filter_data(config['invalfile'], config, compute_scale_perfly,
                                                            keypointnames=keypointnames)
            LOG.info(f"val data X shape: {valdata['X'].shape}")
            res['valdata'] = valdata
            res['val_scale_perfly'] = val_scale_perfly
    else:
        LOG.info("Loading data from quick debug data file ", quickdebugdatafile)
        with open(quickdebugdatafile,'rb') as f:
            tmp = pickle.load(f)
            data = tmp['data']
            scale_perfly = tmp['scale_perfly']
            valdata = tmp['valdata']
            val_scale_perfly = tmp['val_scale_perfly']
        res['data'] = data
        res['scale_perfly'] = scale_perfly
        res['valdata'] = valdata
        res['val_scale_perfly'] = val_scale_perfly
        
    return res

def init_process_data(config=None,data=None,scale_perfly=None,
                      valdata=None,val_scale_perfly=None,
                      traindataprocess='chunk',valdataprocess='test',res={}):
    """
    res = init_process_data(config=None,data=None,scale_perfly=None,
                            valdata=None,val_scale_perfly=None,
                            traindataprocess='chunk',valdataprocess='test',res={})
    Process raw data into features for training and validation.
    Inputs:
    config: dict, configuration dictionary
    data: dict, training data dictionary
    scale_perfly: ndarray of shape nscale x nflies with scale data for each fly
    valdata: dict, validation data dictionary
    val_scale_perfly: ndarray of shape nscale x nflies with scale data for each fly
    traindataprocess: str, one of ['chunk','test'], how to process training data
    valdataprocess: str, one of ['chunk','test'], how to process validation data
    res: dict, dictionary to store results with the following keys:
        'npad': int, number of frames to pad the output by
        'compute_feature_params': dict, parameters for computing features
        'reparamfun': function, function for computing features from training data
        'val_reparamfun': function, function for computing features from validation data
        'X': list of dicts, processed training data
        'valX': list of dicts, processed validation data
        'train_chunk_data_params': dict, parameters for chunking training data
        'val_chunk_data_params': dict, parameters for chunking validation data
        'dct_m': ndarray or None, DCT matrix
        'idct_m': ndarray or None, inverse DCT matrix
    """
    
    assert config is not None
    
    # if using discrete cosine transform, create dct matrix
    # this didn't seem to work well, so probably won't use in the future
    if config['dct_tau'] is not None and config['dct_tau'] > 0:
        dct_m, idct_m = get_dct_matrix(config['dct_tau'])
        # this gives the maximum of 
        #   a) max number of frames to lookahead or 
        #   b) dct_tau (number of timepoints for cosine transform)
    else:
        dct_m = None
        idct_m = None

    res['dct_m'] = dct_m
    res['idct_m'] = idct_m
    
    # how much to pad outputs by -- depends on how many frames into the future we will predict
    npad = compute_npad(config['tspred_global'], dct_m)

    compute_feature_params = {
        "simplify_out": config['simplify_out'],
        "simplify_in": config['simplify_in'],
        "dct_m": dct_m,
        "tspred_global": config['tspred_global'],
        "compute_pose_vel": config['compute_pose_vel'],
        "discreteidx": config['discreteidx'],
    }

    # function for computing features
    reparamfun = lambda x, id, flynum, **kwargs: compute_features(
        x, id, flynum, scale_perfly, outtype=np.float32, **compute_feature_params, **kwargs)

    val_reparamfun = lambda x, id, flynum, **kwargs: compute_features(
        x, id, flynum, val_scale_perfly, outtype=np.float32, **compute_feature_params, **kwargs)

    res['npad'] = npad
    res['compute_feature_params'] = compute_feature_params
    res['reparamfun'] = reparamfun
    res['val_reparamfun'] = val_reparamfun
    res['X'] = None
    res['valX'] = None
    res['train_chunk_data_params'] = None
    res['val_chunk_data_params'] = None

    # process the data

    if data is not None:
        LOG.info('Processing training data...')
        if traindataprocess == 'chunk':
            train_chunk_data_params = {'npad': npad}
            X = chunk_data(data, config['contextl'], reparamfun, **train_chunk_data_params)
        elif traindataprocess == 'test':
            train_chunk_data_params = {'npad': npad, 'minnframes': config['contextl']+1}
            X = process_test_data(data, reparamfun, **train_chunk_data_params)
        else:
            raise ValueError(f'traindataprocess {traindataprocess} not recognized')
        res['X'] = X
        res['train_chunk_data_params'] = train_chunk_data_params
        LOG.info(f"{len(X)} training ids, total of {sum([x['input'].shape[0] for x in X])} time points")
        
    if valdata is not None:
        LOG.info('Processing val data...')
        if valdataprocess == 'chunk':
            val_chunk_data_params = {'npad': npad}
            valX = chunk_data(valdata, config['contextl'], val_reparamfun, **val_chunk_data_params)
        elif valdataprocess == 'test':
            val_chunk_data_params = {'npad': npad, 'minnframes': config['contextl']+1}
            valX = process_test_data(valdata, val_reparamfun, **val_chunk_data_params)
        else:
            raise ValueError(f'valdataprocess {valdataprocess} not recognized')
        res['valX'] = valX
        res['val_chunk_data_params'] = val_chunk_data_params
        LOG.info(f" {len(valX)} val ids, total of {sum([x['input'].shape[0] for x in valX])} time points")
    
    return res

def _dataloader_worker_args(config):
    """
    Keyword arguments for DataLoader controlling background worker processes, from config['num_workers'].
    Workers are forked, so they share the in-memory data arrays with the main process copy-on-write.
    Workers are deliberately not persistent: the training dataset re-tiles its chunks at the end of each
    epoch (recompute_chunk_indices), and fresh workers forked at the start of each epoch see the new tiling.
    """
    num_workers = config.get('num_workers', 0)
    if num_workers <= 0:
        return {}
    return {'num_workers': num_workers, 'persistent_workers': False, 'multiprocessing_context': 'fork'}

# Bump this when something other than DATASET_SOURCE_FILES and DATASET_CONFIG_KEYS alters the datasets
# init_datasets builds (e.g. a numpy upgrade that changes results), so that existing dataset cache files
# are no longer used.
DATASET_CACHE_VERSION = 2

# Source files, relative to the repository root, whose code builds the datasets init_datasets builds: the
# files whose functions run while building them, plus flyllm/config.py, whose constants they use. Their
# contents are part of the dataset cache key, so editing any of them gives a new cache file. If
# dataset-building code starts using another file, add it here.
DATASET_SOURCE_FILES = [
    'apf/data.py', 'apf/dataset.py', 'apf/features.py', 'apf/io.py', 'apf/utils.py',
    'experiments/flyllm.py', 'flyllm/config.py', 'flyllm/features.py', 'flyllm/prepare.py',
]

# Config entries that affect the datasets init_datasets builds (via make_dataset and load_and_filter_data).
# If dataset-building code starts reading another config entry, add it here.
DATASET_CONFIG_KEYS = [
    'intrainfile', 'invalfile', 'contextl', 'tspred_global', 'dataset_params',
    'discreteidx', 'discretize_nbins', 'discretize_epsilon', 'all_discretize_epsilon', 'simplify_out',
    'bin_edges_absolute', 'categories', 'output_categories', 'augment_flip',
]

def _json_default(x):
    """ json.dumps default for dataset cache keys: numpy values as lists/scalars, anything else by repr. """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return repr(x)

def dataset_source_hash():
    """ Returns a SHA-256 hex digest of the names and contents of DATASET_SOURCE_FILES. """
    repodir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256()
    for relpath in DATASET_SOURCE_FILES:
        with open(os.path.join(repodir, relpath), 'rb') as f:
            contents = f.read()
        h.update(relpath.encode() + b'\0' + str(len(contents)).encode() + b'\0' + contents)
    return h.hexdigest()

def dataset_cache_key(config, needtraindata, needvaldata, debug_uselessdata):
    """
    Returns a dict of everything the datasets built by init_datasets depend on: the DATASET_CONFIG_KEYS
    entries of config, the input files' sizes and modification times, the arguments, a hash of the contents
    of DATASET_SOURCE_FILES, and DATASET_CACHE_VERSION.
    """
    key = {k: config.get(k) for k in DATASET_CONFIG_KEYS}
    key['input_file_stats'] = {}
    for k in ['intrainfile', 'invalfile']:
        if config.get(k) is not None and os.path.exists(config[k]):
            st = os.stat(config[k])
            key['input_file_stats'][k] = [st.st_size, st.st_mtime_ns]
    key['needtraindata'] = needtraindata
    key['needvaldata'] = needvaldata
    key['debug_uselessdata'] = debug_uselessdata
    key['source_hash'] = dataset_source_hash()
    key['version'] = DATASET_CACHE_VERSION
    return key

def dataset_cache_file_path(config, key):
    """
    Returns the absolute path of the dataset cache file for the datasets with this dataset_cache_key, or
    None if caching is off (config['dataset_cache_dir'] not set). The file name contains a hash of key, so
    any change to what the datasets depend on gives a different file. Relative cache dirs are relative to
    the working directory.
    """
    cachedir = config.get('dataset_cache_dir')
    if not cachedir:
        return None
    keystr = json.dumps(key, sort_keys=True, default=_json_default)
    return os.path.join(os.path.abspath(cachedir), f'datasets_{hashlib.sha256(keystr.encode()).hexdigest()[:16]}.pkl')

def _same_value(a, b):
    try:
        return pickle.dumps(a) == pickle.dumps(b)
    except Exception:
        return False

def save_dataset_cache(cache_file_path, key, config_before, config, res):
    """
    Saves the datasets init_datasets built to the file at cache_file_path, along with the config entries
    building them changed (e.g. dataset_params, discretize_epsilon) and the numpy and torch random number
    generator states afterwards (building them draws random numbers), so that loading the cache leaves
    everything as building them would.
    Python's random module is not seeded by init_state, so its state is left alone.
    Also writes the cache key to a .json file next to it, for reference.
    """
    keys = ['dataset_params']
    if res.get('train_dataset') is not None:
        keys += ['train_dataset', 'train_data']
    if res.get('val_dataset') is not None:
        keys += ['val_dataset', 'val_data']
    cache = {
        'res': {k: res[k] for k in keys},
        'config_changes': {k: v for k, v in config.items()
                           if k not in config_before or not _same_value(v, config_before[k])},
        'rng_state': {'numpy': np.random.get_state(), 'torch': torch.get_rng_state()},
    }
    LOG.info(f'Saving datasets to cache file {cache_file_path}...')
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
    # write to a temporary file and rename, so an interrupted save never leaves a partial cache file
    tmp_file_path = f'{cache_file_path}.{os.getpid()}.tmp'
    with open(tmp_file_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_file_path, cache_file_path)
    with open(os.path.splitext(cache_file_path)[0] + '.json', 'w') as f:
        json.dump(key, f, indent=2, sort_keys=True, default=_json_default)
    LOG.info('Saved datasets to cache file')

def load_dataset_cache_bang_bang(config, res, cache_file_path):
    """
    Loads datasets saved by save_dataset_cache, from the file at cache_file_path, into res, and applies
    its config changes to config. Also sets the numpy and torch RNG states to the saved ones.  The
    `_bang_bang` in the name is to indicate that it modifies `config` and `res` in-place.
    """
    LOG.info(f'Loading datasets from cache file {cache_file_path}...')
    with open(cache_file_path, 'rb') as f:
        cache = pickle.load(f)
    res.update(cache['res'])
    config.update(cache['config_changes'])
    res['config']['dataset_params'] = res['dataset_params']
    np.random.set_state(cache['rng_state']['numpy'])
    torch.set_rng_state(cache['rng_state']['torch'])
    LOG.info('Loaded datasets from cache file')

def build_datasets_bang_bang(config, res, needtraindata, needvaldata, debug_uselessdata):
    """
    Builds the training and/or validation datasets with make_dataset, and stores them and their intermediate
    data in res. Building them also changes config (e.g. dataset_params, discretize_epsilon).
    """
    if needtraindata:
        # create dataset
        res['train_data'] = {}
        res['train_dataset'], res['train_data']['flyids'], res['train_data']['track'], res['train_data']['pose'], \
            res['train_data']['velocity'], res['train_data']['sensory'], res['config']['dataset_params'], \
            res['train_data']['isdata'], res['train_data']['isstart'], res['train_data']['useoutputmask'], \
            res['train_data']['labels'], res['train_data']['categories'] = \
            make_dataset(config,'intrainfile',return_all=True,debug=debug_uselessdata)
        res['dataset_params'] = res['config']['dataset_params']

    if needvaldata:
        # create dataset
        res['val_data'] = {}
        res['val_dataset'], res['val_data']['flyids'], res['val_data']['track'], res['val_data']['pose'], \
            res['val_data']['velocity'], res['val_data']['sensory'], res['config']['dataset_params'], \
            res['val_data']['isdata'], res['val_data']['isstart'], res['val_data']['useoutputmask'], \
            res['val_data']['labels'], res['val_data']['categories'] = \
            make_dataset(config,'invalfile',return_all=True,debug=debug_uselessdata)
        res['dataset_params'] = res['config']['dataset_params']

def generate_and_save_dataset_cache_bang_bang(config, res, cache_file_path, cachekey, needtraindata, needvaldata,
                                              debug_uselessdata):
    """
    Builds the datasets into res and config, as build_datasets_bang_bang does, and saves them to the dataset
    cache file at cache_file_path, with key cachekey.
    """
    config_before = copy.deepcopy(config)
    build_datasets_bang_bang(config, res, needtraindata, needvaldata, debug_uselessdata)
    save_dataset_cache(cache_file_path, cachekey, config_before, config, res)

def init_datasets(config=None,needtraindata=True,needvaldata=True,dct_m=None,idct_m=None,debug_uselessdata=False,res={}):
    """
    res = init_datasets(config=None,X=None,valX=None,traindataprocess='chunk',
                        valdataprocess='test',dct_m=None,idct_m=None,res={})
    Create training and validation datasets from processed data.
    Inputs:
    config: dict, configuration dictionary
    X: list of dicts, processed training data
    valX: list of dicts, processed validation data
    traindataprocess: str, one of ['chunk','test'], how to process training data and represent it as a dataset.
    If 'chunk', then FlyMLMDataset is returned. If 'test', then FlyTestDataset is returned.
    valdataprocess: str, one of ['chunk','test'], how to process validation data and represent it as a dataset.
    If 'chunk', then FlyMLMDataset is returned. If 'test', then FlyTestDataset is returned.
    dct_m: ndarray or None, DCT matrix. 
    idct_m: ndarray or None, inverse DCT matrix.
    res: dict, dictionary to store results with the following keys:
    Output:
    res: dict, updated dictionary with the following keys:
        'dataset_params': dict, parameters for creating the datasets
        'train_dataset': FlyMLMDataset or FlyTestDataset, training dataset
        'train_dataloader': torch DataLoader, training data loader
        'val_dataset': FlyMLMDataset or FlyTestDataset, validation dataset
        'val_dataloader': torch DataLoader, validation data loader
        'ntrain_batches': int, number of training batches
        'nval_batches': int, number of validation batches
    """
    
    assert config is not None, "No configuration provided"

    res['dataset_params'] = config.get('dataset_params',{})
    res['train_dataset'] = None
    res['train_dataloader'] = None
    res['val_dataset'] = None
    res['val_dataloader'] = None
    res['ntrain_batches'] = None
    res['nval_batches'] = None

    cachekey = dataset_cache_key(config, needtraindata, needvaldata, debug_uselessdata)
    cache_file_path = dataset_cache_file_path(config, cachekey)
    if cache_file_path is None:
        build_datasets_bang_bang(config, res, needtraindata, needvaldata, debug_uselessdata)
    elif os.path.exists(cache_file_path):
        load_dataset_cache_bang_bang(config, res, cache_file_path)
    else:
        generate_and_save_dataset_cache_bang_bang(config, res, cache_file_path, cachekey, needtraindata, needvaldata,
                                                  debug_uselessdata)

    if needtraindata:
        # create dataloader
        res['train_dataloader'] = DataLoader(res['train_dataset'], batch_size=config['batch_size'], shuffle=True,
                                             pin_memory=True, **_dataloader_worker_args(config))
        res['ntrain_batches'] = len(res['train_dataloader'])

    if needvaldata:
        # create dataloader
        # validation needs no gradients, so it can use the (typically larger) test batch size
        val_batch_size = config.get('test_batch_size', config['batch_size'])
        res['val_dataloader'] = DataLoader(res['val_dataset'], batch_size=val_batch_size, shuffle=False,
                                           pin_memory=True, **_dataloader_worker_args(config))
        res['nval_batches'] = len(res['val_dataloader'])
    return res


def init_model(config=None,somedataset=None,somedataloader=None,device=None,loadmodelfile=None,
               restartmodelfile=None,mode='train',res={}, optimizer=None, lr_scheduler=None,
               optimizer_args={},weights_only=True):
    """
    res = init_model(config=None,somedataset=None,somedataloader=None,device=None,loadmodelfile=None,
                     restartmodelfile=None,mode='train',ntrain_batches=None,res={})
    Initialize the model, loss function, optimizer, and scheduler.
    Inputs:
    config: dict, configuration dictionary
    somedataset: FlyMLMDataset or FlyTestDataset, dataset to use for training or testing
    somedataloader: torch DataLoader, dataloader for somedataset
    device: torch device, device to run computations on
    loadmodelfile: str, path to model file to load
    restartmodelfile: str, path to model file to restart training from
    mode: str, one of ['train','test'], whether training or testing
    res: dict, dictionary to store results with the following keys:
    Output:
    res: dict, updated dictionary with the following keys:
        'model': torch Module, model
        'criterion': torch Module, loss function
        'optimizer': if mode in ['train',] and restartmodelfile is not None, torch Optimizer, optimizer, otherwise None
        'lr_scheduler': if mode in ['train',] and restartmodelfile is not None, torch Scheduler, learning rate scheduler, otherwise None
        'opt_model': if mode in ['test',], torch Module, compiled model for testing, otherwise None
        'modeltype_str': str, model type string, used in creating unique names for various files
        'model_savetime': str, time the model was saved
        'loss_epoch': dict, loss history
        'epoch': int, current epoch number if mode in ['train',], otherwise None. only non-zeros if restarting from restartmodelfile
    """
    
    assert config is not None, "No configuration provided"
    assert somedataset is not None, "No dataset provided"
    assert somedataloader is not None, "No dataloader provided"
    assert device is not None, "No device provided"
    
    # create the model
    LOG.info('Initializing model object from config parameters...')
    model, criterion = initialize_model(config, somedataset, device)
    res['criterion'] = criterion

    # load the model
    if loadmodelfile is not None:
        LOG.info(f'Loading model from file {loadmodelfile}...')
        modeltype_str, savetime = parse_modelfile(loadmodelfile)
        LOG.info(f'Parsed model type string: {modeltype_str}, savetime: {savetime}')
        loss_epoch = load_model(loadmodelfile, model, device, weights_only=weights_only)
        if 'train' in loss_epoch and loss_epoch['train'] is not None:
            if np.any(np.isnan(loss_epoch['train'].cpu().numpy())):
                epoch = np.nonzero(np.isnan(loss_epoch['train'].cpu().numpy()))[0][0]
            else:
                epoch = loss_epoch['train'].shape[0]
            LOG.info(f'Loaded model has been trained for {epoch} epochs, based on train loss history')
        else:
            epoch = config['num_train_epochs']
            LOG.info(f'Loaded model has no train loss history, setting epoch to {epoch} from config')
    else:
        modeltype_str = get_modeltype_str(config)
        if ('model_nickname' in config) and (config['model_nickname'] is not None):
            modeltype_str = config['model_nickname']
        if mode in ['train',]:
            if restartmodelfile is not None:
                LOG.info(f'Restarting model from file {restartmodelfile}...')
                num_training_steps = config['num_train_epochs'] * len(somedataloader)
                if optimizer is None or lr_scheduler is None:
                    optimizer, lr_scheduler = init_optimizer(num_training_steps, model, optimizer_args)
                loss_epoch = load_model(restartmodelfile, model, device,
                                        lr_optimizer=optimizer, scheduler=lr_scheduler, weights_only=weights_only)
                update_loss_nepochs(loss_epoch, config['num_train_epochs'])
                if np.any(np.isnan(loss_epoch['train'].cpu().numpy())):
                    epoch = np.nonzero(np.isnan(loss_epoch['train'].cpu().numpy()))[0][0]
                else:
                    epoch = loss_epoch['train'].shape[0]
                LOG.info(f'Restarted model has been trained for {epoch} epochs, based on train loss history')
            else:
                epoch = 0
                # initialize structure to keep track of loss
                loss_epoch = initialize_loss(somedataset, config)
            savetime = datetime.datetime.now()
            savetime = savetime.strftime('%Y%m%dT%H%M%S')
        else:
            loss_epoch = None
            epoch = None
            savetime = None

    res['model'] = model
    res['criterion'] = criterion
    res['optimizer'] = optimizer
    res['lr_scheduler'] = lr_scheduler
    res['modeltype_str'] = modeltype_str
    res['model_savetime'] = savetime
    res['loss_epoch'] = loss_epoch
    res['epoch'] = epoch
        
    if mode in ['test',]:
        res['opt_model'] = torch.compile(model)
    else:
        res['opt_model'] = None

    return res

def init_flyllm(configfile=None,config=None,
                mode='test',loadmodelfile=None,seedrandom=True,
                needtraindata=None,needvaldata=None,debug_uselessdata=False,
                restartmodelfile=None,
                doinitconfig=True,doinitstate=True,
                doinitdatasets=True,doinitmodel=True,overrideconfig={},
                res={},
                weights_only=True):
    """
    Standard use:
    res = init_flyllm(configfile,mode)
    Full parameters with defaults:
    res = init_flyllm(configfile=None,config=None,mode='test',loadmodelfile=None,seedrandom=True,
                      needtraindata=None,needvaldata=None,debug_uselessdata=False,
                      doinitconfig=True,doinitstate=True,
                      doinitdatasets=True,doinitmodel=True,
                      overrideconfig={},res={})
    Perform all initialization steps for flyllm data. 
    Calls init_config to load the configuration.
    Calls init_state to set the random seed and device.
    Calls init_datasets to load the data, compute features, and create the training and validation datasets and dataloaders.
    Calls init_model to initialize the model and loss function

    Inputs:
    configfile: str, path to configuration file from which to load configuration, used if config is None
    mode: str, one of ['train','test'], whether training or testing, default = 'test'

    More inputs:
    debug_uselessdata: bool, whether to use only a small subset of the data for debugging. Default = False.
    restartmodelfile: str, path to model file to restart training from. Default = None. Only used if mode == 'train'.
    config: dict, configuration dictionary, default = None. If not None, use this instead of loading from configfile
    loadmodelfile: str, path to model file to load. Only used if mode == 'test'. Default = None. If None and mode == 'test', will be set from config['loadmodelfile']
    seedrandom: bool, whether to seed the random number generators. Default = True.
    needtraindata: bool, whether to load training data. If None, will be set to True if mode == 'train'. Default = None.
    needvaldata: bool, whether to load validation data. If None, will be set to True if mode in ['train','test']. Default = None.
    traindataprocess: str, one of ['chunk','test'], how to process training data. If None, will be set to 'chunk' if 
    mode in ['train','test']. Default = None.
    valdataprocess: str, one of ['chunk','test'], how to process validation data. If None, will be set to 'chunk' if
    mode == 'train', and 'test' if mode == 'test'. Default = None.
    overrideconfig: dict, dictionary of configuration parameters to override those in configfile or loadmodelfile. Default = {}
    res: dict, dictionary to store results. Default = {}. Provide this if you are skipping steps and want to provide 
    results from previous steps.
    
    Can skip any of these steps. If you skip them, their results should already be in res. 
    doinitconfig: bool, whether to call init_config. Default = True. 
    doinitstate: bool, whether to call init_state. Default = True.
    doinitrawdata: bool, whether to call init_raw_data. Default = True.
    doinitprocessdata: bool, whether to call init_process_data. Default = True.
    doinitdatasets: bool, whether to call init_datasets. Default = True.
    doinitmodel: bool, whether to call init_model. Default = True.
    Output:
    res: dict, updated dictionary with the following keys:

        'config': dict, configuration dictionary (init_config)

        'device': torch device, device to run computations on (init_state)

        'data': dict, training data dictionary (init_raw_data)
        'scale_perfly': ndarray of shape nscale x nflies with scale data for each fly (init_raw_data)
        'valdata': dict, validation data dictionary (init_raw_data)
        'val_scale_perfly': ndarray of shape nscale x nflies with scale data for each fly (init_raw_data)

        'X': list of dicts, processed training data (init_process_data)
        'valX': list of dicts, processed validation data (init_process_data)
        'npad': int, number of frames to pad the output by (init_process_data)
        'compute_feature_params': dict, parameters for computing features (init_process_data)
        'reparamfun': function, function for computing features from training data (init_process_data)
        'val_reparamfun': function, function for computing features from validation data (init_process_data)
        'train_chunk_data_params': dict, parameters for chunking training data (init_process_data)
        'val_chunk_data_params': dict, parameters for chunking validation data (init_process_data)
        'dct_m': ndarray or None, DCT matrix (init_process_data)
        'idct_m': ndarray or None, inverse DCT matrix (init_process_data)

        'train_dataset': FlyMLMDataset or FlyTestDataset, training dataset (init_datasets)
        'train_dataloader': torch DataLoader, training data loader (init_datasets)
        'val_dataset': FlyMLMDataset or FlyTestDataset, validation dataset (init_datasets)
        'val_dataloader': torch DataLoader, validation data loader (init_datasets)
        'dataset_params': dict, parameters for creating the datasets (init_datasets)
        'ntrain_batches': int, number of training batches (init_datasets)
        'nval_batches': int, number of validation batches (init_datasets)
        'train_sz': tuple, shape of training data (init_datasets)
        'val_sz': tuple, shape of validation data (init_datasets)

        'model': torch Module, model (init_model)
        'criterion': torch Module, loss function (init_model)
        'num_training_steps': if mode in ['train',], int, number of training steps, otherwise None (init_model)
        'optimizer': if mode in ['train',], torch Optimizer, optimizer, otherwise None (init_model)
        'lr_scheduler': if mode in ['train',], torch Scheduler, learning rate scheduler, otherwise None (init_model)
        'opt_model': if mode in ['test',], torch Module, compiled model for testing, otherwise None (init_model)
        'modeltype_str': str, model type string, used in creating unique names for various files (init_model)
        'model_savetime': str, time the model was saved (init_model)
        'loss_epoch': dict, loss history (init_model)
        'epoch': int, epoch number (init_model)
        'train_src_mask': torch Tensor, attention mask (init_model)
        'is_causal': bool, whether the model is causal (init_model)
    """

    ## set defaults in some kind of reasonable way    
    if (needtraindata is None):
        needtraindata = mode in ['train',]
    if (needvaldata is None):
        needvaldata = mode in ['train','test']
        
    res['success'] = False
        
    ## load config from configfile and/or loadmodelfile
    # sets res['config']
    # if loading from loadmodelfile, also sets res['config']['dataset_params']
    # note that dataset_params are NOT loaded from restartmodelfile
    if doinitconfig:
        LOG.info('Initializing configuration...')
        try:
            init_config(configfile=configfile,config=config,mode=mode,loadmodelfile=loadmodelfile,overrideconfig=overrideconfig,res=res,weights_only=weights_only)
        except Exception as e:
            LOG.exception(f'Error in init_config: {e}\nAborting init_flyllm')
            return res

    ## setup device and random
    if doinitstate:
        LOG.info('Initializing state...')
        try:
            # adds 'device' to res
            res = init_state(config=res['config'],seedrandom=seedrandom,res=res)
        except Exception as e:
            LOG.exception(f'Error in init_state: {e}\nAborting init_flyllm')
            return res
        
    ## create training data sets
    if doinitdatasets:
        LOG.info('Initializing datasets...')
        try:
            init_datasets(config=res['config'],needtraindata=needtraindata,needvaldata=needvaldata,debug_uselessdata=debug_uselessdata,res=res)
        except Exception as e:
            LOG.exception(f'Error in init_datasets: {e}\nAborting init_flyllm')
            return res
        
    if doinitmodel:
        LOG.info('Initializing model...')
        try:
            # use training or test dataset to initialize, depending on mode
            args = {}
            if mode in ['train',]:
                args['somedataset'] = res['train_dataset']
                args['somedataloader'] = res['train_dataloader']
                loadmodelfile = None
                #args['ntrain_batches'] = res['ntrain_batches']
            elif mode in ['test',]:
                args['somedataset'] = res['val_dataset']
                args['somedataloader'] = res['val_dataloader']
            else:
                if res['traindataset'] is not None:
                    args['somedataset'] = res['train_dataset']
                    args['somedataloader'] = res['train_dataloader']
                    #args['ntrain_batches'] = res['ntrain_batches']
                elif res['val_dataset'] is not None:
                    args['somedataset'] = res['val_dataset']
                    #args['somedataloader'] = res['val_dataloader']
                else:
                    raise ValueError('No dataset computed')

            # adds 'model', 'criterion', 'num_training_steps', 'optimizer', 'lr_scheduler', 'opt_model', 'modeltype_str',
            # 'model_savetime', 'loss_epoch', 'epoch', 'train_src_mask', and 'is_causal' to res
            res = init_model(config=res['config'],device=res['device'],loadmodelfile=loadmodelfile,
                            restartmodelfile=restartmodelfile,mode=mode,res=res,weights_only=weights_only,**args)
        except Exception as e:
            LOG.exception(f'Error in init_model: {e}\nAborting init_flyllm')
            return res
        
    res['success'] = True
    return res