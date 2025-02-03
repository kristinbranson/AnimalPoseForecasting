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
from flyllm.config import scalenames, nfeatures, DEFAULTCONFIGFILE, featglobal, posenames, keypointnames
from flyllm.features import compute_features, sanity_check_tspred, get_sensory_feature_idx, compute_scale_perfly, compute_pose_distribution_stats
from flyllm.dataset import FlyTestDataset, FlyMLMDataset
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
from flyllm.simulation import animate_predict_open_loop
from flyllm.prediction import predict_all
from IPython.display import HTML

import logging
LOG = logging.getLogger(__name__)

mpl_backend = plt.get_backend()
if mpl_backend == 'inline':
    from IPython import display

def init_config(configfile=None,config=None,mode='train',loadmodelfile=None,res={}):
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
        
        config = read_config(configfile,
                            default_configfile=DEFAULTCONFIGFILE,
                            get_sensory_feature_idx=get_sensory_feature_idx,
                            featglobal=featglobal,
                            posenames=posenames)
    
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
        LOG.info(f'{len(X)} training ids, total of {sum([x['input'].shape[0] for x in X])} time points')
        
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
        LOG.info(f' {len(valX)} val ids, total of {sum([x['input'].shape[0] for x in valX])} time points')
    
    return res

def init_datasets(config=None,X=None,valX=None,traindataprocess='chunk',
                  valdataprocess='test',dct_m=None,idct_m=None,res={}):
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
        'train_sz': tuple, shape of training data
        'val_sz': tuple, shape of validation data
    """
    
    assert config is not None, "No configuration provided"
    
    dataset_params = {
        'max_mask_length': config['max_mask_length'],
        'pmask': config['pmask'],
        'masktype': config['masktype'],
        'simplify_out': config['simplify_out'],
        'pdropout_past': config['pdropout_past'],
        'input_labels': config['input_labels'],
        'dozscore': True,
        'discreteidx': config['discreteidx'],
        'discretize_nbins': config['discretize_nbins'],
        'discretize_epsilon': config['discretize_epsilon'],
        'flatten_labels': config['flatten_labels'],
        'flatten_obs_idx': config['flatten_obs_idx'],
        'flatten_do_separate_inputs': config['flatten_do_separate_inputs'],
        'p_add_input_noise': config['p_add_input_noise'],
        'dct_ms': (dct_m,idct_m),
        'tspred_global': config['tspred_global'],
        'discrete_tspred': config['discrete_tspred'],
        'compute_pose_vel': config['compute_pose_vel'],
        
    }

    if 'dataset_params' in config:
        # zscore and discretize parameters
        for k in config['dataset_params']:
            dataset_params[k] = config['dataset_params'][k]

    res['dataset_params'] = dataset_params
    res['train_dataset'] = None
    res['train_dataloader'] = None
    res['val_dataset'] = None
    res['val_dataloader'] = None
    res['ntrain_batches'] = None
    res['nval_batches'] = None
    res['train_sz'] = None
    res['val_sz'] = None

    if X is not None:
        LOG.info('Creating training data set...')
        if traindataprocess == 'chunk':
            train_dataset = FlyMLMDataset(X,**dataset_params)
        elif traindataprocess == 'test':
            train_dataset = FlyTestDataset(X,config['contextl'],**dataset_params)
        else:
            raise ValueError(f'traindataprocess {traindataprocess} not recognized')
        LOG.info(f'Train dataset size: {len(train_dataset)}')

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=config['batch_size'],
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        )
        ntrain_batches = len(train_dataloader)
        LOG.info(f'Number of training batches: {ntrain_batches}')
        res['train_dataset'] = train_dataset
        example = next(iter(train_dataloader))
        LOG.info(f'val batch keys: {example.keys()}')
        sz = example['input'].shape
        LOG.info(f'val batch input shape = {sz}')
        
        if ('zscore_params' not in dataset_params) or (dataset_params['zscore_params'] is None):
            dataset_params['zscore_params'] = train_dataset.get_zscore_params()
        if ('discretize_params' not in dataset_params) or (dataset_params['discretize_params'] is None):
            dataset_params['discretize_params'] = train_dataset.get_discretize_params()
            
        
        res['train_dataset'] = train_dataset
        res['train_dataloader'] = train_dataloader
        res['ntrain_batches'] = ntrain_batches
        res['train_sz'] = sz

    if valX is not None:
        LOG.info('Creating validation data set...')
        if valdataprocess == 'chunk':
            val_dataset = FlyMLMDataset(valX,**dataset_params)
        elif valdataprocess == 'test':
            val_dataset = FlyTestDataset(valX,config['contextl'],**dataset_params)
        else:
            raise ValueError(f'valdataprocess {valdataprocess} not recognized')
        LOG.info(f'Validation dataset size: {len(val_dataset)}')
        
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=config['test_batch_size'],
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    )
        nval_batches = len(val_dataloader)
        LOG.info(f'Number of validation batches: {nval_batches}')
        example = next(iter(val_dataloader))
        LOG.info(f'val batch keys: {example.keys()}')
        sz = example['input'].shape
        LOG.info(f'val batch input shape = {sz}')
        res['val_dataset'] = val_dataset
        res['val_dataloader'] = val_dataloader
        res['nval_batches'] = nval_batches
        res['val_sz'] = sz
        
    return res

def init_model(config=None,somedataset=None,somedataloader=None,device=None,loadmodelfile=None,
               restartmodelfile=None,mode='train',ntrain_batches=None,res={}):
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
    ntrain_batches: int, number of training batches
    res: dict, dictionary to store results with the following keys:
    Output:
    res: dict, updated dictionary with the following keys:
        'model': torch Module, model
        'criterion': torch Module, loss function
        'num_training_steps': if mode in ['train',], int, number of training steps, otherwise None
        'optimizer': if mode in ['train',], torch Optimizer, optimizer, otherwise None
        'lr_scheduler': if mode in ['train',], torch Scheduler, learning rate scheduler, otherwise None
        'opt_model': if mode in ['test',], torch Module, compiled model for testing, otherwise None
        'modeltype_str': str, model type string, used in creating unique names for various files
        'model_savetime': str, time the model was saved
        'loss_epoch': dict, loss history
        'epoch': int, epoch number
        'train_src_mask': torch Tensor, attention mask
        'is_causal': bool, whether the model is causal
    """
    
    assert config is not None, "No configuration provided"
    assert somedataset is not None, "No dataset provided"
    assert somedataloader is not None, "No dataloader provided"
    assert device is not None, "No device provided"
    
    # create the model
    model, criterion = initialize_model(config, somedataset, device)
    res['criterion'] = criterion

    if mode in ['train',]:
        assert ntrain_batches is not None, "No number of training batches provided"
        # optimizer
        num_training_steps = config['num_train_epochs'] * ntrain_batches
        optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer_args']) 
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0., total_iters=num_training_steps)
        res['num_training_steps'] = num_training_steps
        res['optimizer'] = optimizer
        res['lr_scheduler'] = lr_scheduler
    else:
        res['num_training_steps'] = None
        res['optimizer'] = None
        res['lr_scheduler'] = None

    # load the model
    if loadmodelfile is not None:
        modeltype_str, savetime = parse_modelfile(loadmodelfile)
        loss_epoch = load_model(loadmodelfile, model, device)
        if np.any(np.isnan(loss_epoch['train'].cpu().numpy())):
            epoch = np.nonzero(np.isnan(loss_epoch['train'].cpu().numpy()))[0][0]
        else:
            epoch = loss_epoch['train'].shape[0]
    else:
        modeltype_str = get_modeltype_str(config, somedataset)
        if ('model_nickname' in config) and (config['model_nickname'] is not None):
            modeltype_str = config['model_nickname']
        if mode in ['train',]:
            if restartmodelfile is not None:
                loss_epoch = load_model(restartmodelfile, model, device, lr_optimizer=optimizer, scheduler=lr_scheduler)
                update_loss_nepochs(loss_epoch, config['num_train_epochs'])
                if np.any(np.isnan(loss_epoch['train'].cpu().numpy())):
                    epoch = np.nonzero(np.isnan(loss_epoch['train'].cpu().numpy()))[0][0]
                else:
                    epoch = loss_epoch['train'].shape[0]
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
    res['modeltype_str'] = modeltype_str
    res['model_savetime'] = savetime
    res['loss_epoch'] = loss_epoch
    res['epoch'] = epoch
        
    # create attention mask
    example = next(iter(somedataloader))
    contextl = example['input'].shape[1]
    if config['modeltype'] == 'mlm':
        train_src_mask = generate_square_full_mask(contextl).to(device)
        is_causal = False
    elif config['modeltype'] == 'clm':
        train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(contextl, device=device)
        is_causal = True
        #train_src_mask = generate_square_subsequent_mask(contextl).to(device)
    else:
        raise ValueError(f'Unknown modeltype: {config["modeltype"]}')
    res['train_src_mask'] = train_src_mask
    res['is_causal'] = is_causal

    if mode in ['test',]:
        res['opt_model'] = torch.compile(model)
    else:
        res['opt_model'] = None

    return res

def init_flyllm(configfile=None,config=None,mode='test',loadmodelfile=None,seedrandom=True,
                quickdebugdatafile=None,needtraindata=None,needvaldata=None,traindataprocess=None,
                valdataprocess=None,restartmodelfile=None,res={},
                doinitconfig=True,doinitstate=True,doinitrawdata=True,doinitprocessdata=True,
                doinitdatasets=True,doinitmodel=True):
    """
    res = init_flyllm(configfile=None,config=None,mode='test',loadmodelfile=None,seedrandom=True,
                      quickdebugdatafile=None,needtraindata=None,needvaldata=None,traindataprocess=None,
                      valdataprocess=None,restartmodelfile=None,res={},
                      doinitconfig=True,doinitstate=True,doinitrawdata=True,doinitprocessdata=True,
                      doinitdatasets=True,doinitmodel=True)
    Perform all initialization steps for flyllm data. 
    Calls init_config to load the configuration.
    Calls init_state to set the random seed and device.
    Calls init_raw_data to load the raw data.
    Calls init_process_data to compute features from the raw data.
    Calls init_datasets to create the training and validation datasets and dataloaders.
    Calls init_model to initialize the model, loss function, optimizer, and scheduler.
    Inputs:
    configfile: str, path to configuration file from which to load configuration if config is None
    config: dict, configuration dictionary, default = None
    mode: str, one of ['train','test'], whether training or testing
    loadmodelfile: str, path to model file to load. Only used if mode == 'test'. Default = None.
    seedrandom: bool, whether to seed the random number generators. Default = True.
    quickdebugdatafile: str, path to small data file with data already filtered, used for speeding up loading
    and debugging. If None, load data from config['intrainfile'] and config['invalfile']. Default = None
    needtraindata: bool, whether to load training data. If None, will be set to True if mode == 'train'. Default = None.
    needvaldata: bool, whether to load validation data. If None, will be set to True if mode in ['train','test']. Default = None.
    traindataprocess: str, one of ['chunk','test'], how to process training data. If None, will be set to 'chunk' if 
    mode in ['train','test']. Default = None.
    valdataprocess: str, one of ['chunk','test'], how to process validation data. If None, will be set to 'chunk' if
    mode == 'train', and 'test' if mode == 'test'. Default = None.
    restartmodelfile: str, path to model file to restart training from. Default = None. Only used if mode == 'train'.
    res: dict, dictionary to store results. Default = {}.
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
            
    if needtraindata and (traindataprocess is None):
        if mode in ['train','test']:
            traindataprocess = 'chunk'
        else:
            raise ValueError(f'traindataprocess not specified')
        
    if needvaldata and (valdataprocess is None):
        if mode in ['train',]:
            valdataprocess = 'chunk'
        elif mode in ['test',]:
            valdataprocess = 'test'
        else:
            raise ValueError(f'valdataprocess not specified')
            
    ## load config
    if doinitconfig:
        # adds 'config' to res
        res = init_config(configfile=configfile,config=config,mode=mode,loadmodelfile=loadmodelfile,res=res)

    ## setup device and random
    if doinitstate:
        # adds 'device' to res
        res = init_state(config=res['config'],seedrandom=seedrandom,res=res)
                
    ## load raw data
    if doinitrawdata:
        # adds 'data', 'scale_perfly', 'valdata', 'val_scale_perfly' to res
        try:
            res = init_raw_data(config=res['config'],quickdebugdatafile=quickdebugdatafile,needtraindata=needtraindata,
                                needvaldata=needvaldata,res=res)
        except Exception as e:
            LOG.error(f'Error in init_raw_data: {e}\nAborting init_flyllm')
            return res
            

    ## compute features
    if doinitprocessdata:
        # adds 'npad', 'compute_feature_params', 'reparamfun', 'val_reparamfun', 'X', 'valX', 'train_chunk_data_params', 
        # 'val_chunk_data_params', 'dct_m', and 'idct_m' to res
        try:
            res = init_process_data(config=res['config'],data=res['data'],scale_perfly=res['scale_perfly'],
                                    valdata=res['valdata'],val_scale_perfly=res['val_scale_perfly'],
                                    traindataprocess=traindataprocess,valdataprocess=valdataprocess,res=res)
        except Exception as e:
            LOG.error(f'Error in init_process_data: {e}\nAborting init_flyllm')
            return res

    ## create datasets
    if doinitdatasets:
        # adds 'dataset_params', 'train_dataset', 'train_dataloader', 'val_dataset', 'val_dataloader', 'ntrain_batches',
        # 'nval_batches', 'train_sz', and 'val_sz' to res
        try:
            res = init_datasets(config=res['config'],X=res['X'],valX=res['valX'],traindataprocess=traindataprocess,
                                valdataprocess=valdataprocess,dct_m=res['dct_m'],idct_m=res['idct_m'],res=res)
        except Exception as e:
            LOG.error(f'Error in init_datasets: {e}\nAborting init_flyllm')
            return res
        
    ## print data/dataset stats
    
    try:
        if needvaldata:
            ntimepoints_valdata = np.count_nonzero(res['valdata']['isdata'])
            LOG.info(f'ntimepoints_valdata = {ntimepoints_valdata}')
            ntimepoints_valX = np.sum([x['input'].shape[0] for x in res['valX']])
            LOG.info(f'ntimepoints_valX = {ntimepoints_valX}')
            nexamples_val = len(res['val_dataset'])
            LOG.info(f'nexamples_val = {nexamples_val}')
            LOG.info(f'nbatches_val = {res["nval_batches"]}, batch size = {res["config"]["test_batch_size"]}, total examples = {res["nval_batches"]*res["config"]["test_batch_size"]}')

        if needtraindata:
            ntimepoints_data = np.count_nonzero(res['data']['isdata'])
            LOG.info(f'ntimepoints_data = {ntimepoints_data}')
            ntimepoints_X = np.sum([x['input'].shape[0] for x in res['X']])
            LOG.info(f'ntimepoints_X = {ntimepoints_X}')
            nexamples_train = len(res['train_dataset'])
            LOG.info(f'nexamples_train = {nexamples_train}')
            LOG.info(f'nbatches_train = {res["ntrain_batches"]}, batch size = {res["config"]["batch_size"]}, total examples = {res["ntrain_batches"]*res["config"]["batch_size"]}')
    except Exception as e:
        LOG.error(f'Error in printing data/dataset stats: {e}\n aborting init_flyllm')
        return res

    if doinitmodel:
        try:
            
            # use training or test dataset to initialize, depending on mode
            args = {}
            if mode in ['train',]:
                args['somedataset'] = res['train_dataset']
                args['somedataloader'] = res['train_dataloader']
                args['ntrain_batches'] = res['ntrain_batches']
            elif mode in ['test',]:
                args['somedataset'] = res['val_dataset']
                args['somedataloader'] = res['val_dataloader']
            else:
                if res['traindataset'] is not None:
                    args['somedataset'] = res['train_dataset']
                    args['somedataloader'] = res['train_dataloader']
                    args['ntrain_batches'] = res['ntrain_batches']
                elif res['val_dataset'] is not None:
                    args['somedataset'] = res['val_dataset']
                    args['somedataloader'] = res['val_dataloader']
                else:
                    raise ValueError('No dataset computed')

            # adds 'model', 'criterion', 'num_training_steps', 'optimizer', 'lr_scheduler', 'opt_model', 'modeltype_str',
            # 'model_savetime', 'loss_epoch', 'epoch', 'train_src_mask', and 'is_causal' to res
            res = init_model(config=res['config'],device=res['device'],loadmodelfile=loadmodelfile,
                            restartmodelfile=restartmodelfile,mode=mode,res=res,**args)
        except Exception as e:
            LOG.error(f'Error in init_model: {e}\nAborting init_flyllm')
            return res
    return res