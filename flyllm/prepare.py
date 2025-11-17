import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import datetime
import os
from matplotlib import animation
import pickle

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
    compute_loss,
    generate_square_full_mask, 
    sanity_check_temporal_dep,
    criterion_wrapper,
    update_loss_nepochs,
    stack_batch_list,
)
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

def init_config(configfile=None,config=None,mode='train',loadmodelfile=None,overrideconfig={},res={}):
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
        load_config_from_model_file(loadmodelfile=loadmodelfile,config=config)
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
        # LOG.info(f'{len(X)} training ids, total of {sum([x['input'].shape[0] for x in X])} time points')
        
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
        # LOG.info(f' {len(valX)} val ids, total of {sum([x['input'].shape[0] for x in valX])} time points')
    
    return res

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

    if needtraindata:
        
        # create dataset
        res['train_data'] = {}
        res['train_dataset'], res['train_data']['flyids'], res['train_data']['track'], res['train_data']['pose'], \
            res['train_data']['velocity'], res['train_data']['sensory'], res['config']['dataset_params'], \
            res['train_data']['isdata'], res['train_data']['isstart'], res['train_data']['useoutputmask'] = \
            make_dataset(config,'intrainfile',return_all=True,debug=debug_uselessdata)
        res['dataset_params'] = res['config']['dataset_params']
        
        # create dataloader
        res['train_dataloader'] = DataLoader(res['train_dataset'], batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        
        res['ntrain_batches'] = len(res['train_dataloader'])
        
    if needvaldata:
        
        # create dataset
        res['val_data'] = {}
        res['val_dataset'], res['val_data']['flyids'], res['val_data']['track'], res['val_data']['pose'], \
            res['val_data']['velocity'], res['val_data']['sensory'], res['config']['dataset_params'], \
            res['val_data']['isdata'], res['val_data']['isstart'], res['val_data']['useoutputmask'] = \
            make_dataset(config,'invalfile',return_all=True,debug=debug_uselessdata)
        res['dataset_params'] = res['config']['dataset_params']

        # create dataloader
        res['val_dataloader'] = DataLoader(res['val_dataset'], batch_size=config['batch_size'], shuffle=False, pin_memory=True)
        res['nval_batches'] = len(res['val_dataloader'])
    return res

def init_model(config=None,somedataset=None,somedataloader=None,device=None,loadmodelfile=None,
               restartmodelfile=None,mode='train',res={}, optimizer=None, lr_scheduler=None,
               optimizer_args={}):
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
        loss_epoch = load_model(loadmodelfile, model, device)
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
        modeltype_str = get_modeltype_str(config, somedataset)
        if ('model_nickname' in config) and (config['model_nickname'] is not None):
            modeltype_str = config['model_nickname']
        if mode in ['train',]:
            if restartmodelfile is not None:
                LOG.info(f'Restarting model from file {restartmodelfile}...')
                num_training_steps = config['num_train_epochs'] * len(somedataloader)
                if optimizer is None or lr_scheduler is None:
                    optimizer, lr_scheduler = init_optimizer(num_training_steps, model, optimizer_args)
                loss_epoch = load_model(restartmodelfile, model, device, lr_optimizer=optimizer, scheduler=lr_scheduler)
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
                res={}):
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
            init_config(configfile=configfile,config=config,mode=mode,loadmodelfile=loadmodelfile,overrideconfig=overrideconfig,res=res)
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
                            restartmodelfile=restartmodelfile,mode=mode,res=res,**args)
        except Exception as e:
            LOG.exception(f'Error in init_model: {e}\nAborting init_flyllm')
            return res
        
    res['success'] = True
    return res