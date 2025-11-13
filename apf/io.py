import os
import re
import torch
import json
import numpy as np
import pathlib
import logging
import tqdm

from apf.data import flip_agents, filter_data_by_categories, load_raw_npz_data

LOG = logging.getLogger(__name__)


def save_model(savefile, model, lr_optimizer=None, scheduler=None, loss=None, config=None, sensory_params=None):
    tosave = {'model': model.state_dict()}
    if lr_optimizer is not None:
        tosave['lr_optimizer'] = lr_optimizer.state_dict()
    if scheduler is not None:
        tosave['scheduler'] = scheduler.state_dict()
    if loss is not None:
        tosave['loss'] = loss
    if config is not None:
        tosave['config'] = config
    elif hasattr(model, 'config') and model.config is not None:
        tosave['config'] = model.config
    if sensory_params is not None:
        tosave['SENSORY_PARAMS'] = sensory_params
    if hasattr(model, 'dataset_params'):
        tosave['dataset_params'] = model.dataset_params
        
    # make sure parent directory exists
    savedir = os.path.dirname(savefile)
    if len(savedir) > 0 and (not os.path.exists(savedir)):
        os.makedirs(savedir)
        
    torch.save(tosave, savefile)
    return

def modernize_model_file(loadfile,dataset,config,device):
    LOG.info(f'Loading model from file {loadfile}...')
    state = torch.load(loadfile, map_location=device, weights_only=False)
    if 'model' not in state:
        state = {'model': state}
    if 'dataset_params' not in state['model']:
        state['dataset_params'] = dataset.get_params()
    if 'config' not in state:
        state['config'] = config
    return state    
    

def load_model(loadfile, model, device, lr_optimizer=None, scheduler=None, config=None):
    LOG.info(f'Loading model from file {loadfile}...')
    state = torch.load(loadfile, map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(state['model'])
    if lr_optimizer is not None and ('lr_optimizer' in state):
        lr_optimizer.load_state_dict(state['lr_optimizer'])
    if scheduler is not None and ('scheduler' in state):
        scheduler.load_state_dict(state['scheduler'])
    if config is not None:
        load_config_from_model_file(config=config, state=state)

    loss = {'train': None, 'val': None}
    if 'loss' in state:
        if isinstance(loss, dict):
            loss = state['loss']
        else:
            # backwards compatible
            loss['train'] = loss
            if 'val_loss' in state:
                loss['val'] = state['val_loss']
    return loss


def load_config_from_model_file(loadmodelfile=None, config=None, state=None, no_overwrite=(), sensory_params=None):
    if state is None:
        assert loadmodelfile is not None
        LOG.info(f'Loading config from file {loadmodelfile}...')
        state = torch.load(loadmodelfile)
    if config is not None and 'config' in state:
        overwrite_config(config, state['config'], no_overwrite=no_overwrite)
    else:
        LOG.info(f'config not stored in model file {loadmodelfile}')
    if 'SENSORY_PARAMS' in state:
        assert sensory_params is not None, "SENSORY_PARAMS stored with model but no sensory_params provided for update"
        sensory_params.update(state['SENSORY_PARAMS'])
    else:
        LOG.info(f'SENSORY_PARAMS not stored in model file {loadmodelfile}')
    if (config is not None) and ('dataset_params' in state):
        config['dataset_params'] = state['dataset_params']      

    return


def json_load_helper(jsonfile):
    with open(jsonfile, 'r') as f:
        config = json.load(f)
    config = {k: v for k, v in config.items() if re.search('^_comment', k) is None}
    return config

def json_save_helper(jsonfile, config, indent=4):
    with open(jsonfile, 'w') as f:
        json.dump(config, f, indent=indent)
    return

def read_config(jsonfile, default_configfile=None, get_sensory_feature_idx=None, featglobal=None, posenames=None):
    if default_configfile is None:
        config = json_load_helper(jsonfile)
    else:
        config = json_load_helper(default_configfile)
        config1 = json_load_helper(jsonfile)
        # destructive to config
        overwrite_config(config, config1)

    config['intrainfile'] = os.path.join(config['datadir'], config['intrainfilestr'])
    config['invalfile'] = os.path.join(config['datadir'], config['invalfilestr'])

    if type(config['flatten_obs_idx']) == str:
        if config['flatten_obs_idx'] == 'sensory':
            assert get_sensory_feature_idx is not None, "Need 'get_sensory_feature_idx' to set 'flatten_obs_idx'"
            config['flatten_obs_idx'] = get_sensory_feature_idx()
        else:
            raise ValueError(f"Unknown type {config['flatten_obs_idx']} for flatten_obs_idx")

    # discreteidx will reference apf.config.posenames
    if type(config['discreteidx']) == str:
        if config['discreteidx'] == 'global':
            assert featglobal is not None, "Need 'feat_global' to set 'discreteidx'"
            config['discreteidx'] = featglobal.copy()
        else:
            raise ValueError(f"Unknown type {config['discreteidx']} for discreteidx")
    if type(config['discreteidx']) == list:
        for i, v in enumerate(config['discreteidx']):
            if type(v) == str:
                assert posenames is not None, "Need 'posenames' to set 'discreteidx'"
                config['discreteidx'][i] = posenames.index(v)
        config['discreteidx'] = np.array(config['discreteidx'])

    if config['modelstatetype'] == 'prob' and config['minstateprob'] is None:
        config['minstateprob'] = 1 / config['nstates']

    if 'all_discretize_epsilon' in config:
        config['all_discretize_epsilon'] = np.array(config['all_discretize_epsilon'])
        if 'discreteidx' in config and config['discreteidx'] is not None:
            config['discretize_epsilon'] = config['all_discretize_epsilon'][config['discreteidx']]

    if 'input_noise_sigma' in config and config['input_noise_sigma'] is not None:
        config['input_noise_sigma'] = np.array(config['input_noise_sigma'])
    # elif 'input_noise_sigma_mult' in config and 'all_discretize_epsilon' in config:
    #  config['input_noise_sigma'] = np.zeros(config['all_discretize_epsilon'].shape)
    #  l = len(config['input_noise_sigma_mult'])
    #  config['input_noise_sigma'][:l] = config['all_discretize_epsilon'][:l]*np.array(config['input_noise_sigma_mult'])

    assert config['modeltype'] in ['mlm', 'clm']
    assert config['modelstatetype'] in ['prob', 'best', None]
    assert config['masktype'] in ['ind', 'block', None]

    if ('obs_embedding_types' in config) and (type(config['obs_embedding_types']) == dict):
        for k, v in config['obs_embedding_types'].items():
            if v == 'conv1d':
                # modernize
                config['obs_embedding_types'][k] = 'conv1d_feat'
        if 'obs_embedding_params' not in config:
            config['obs_embedding_params'] = {}
        else:
            if type(config['obs_embedding_params']) != dict:
                assert config['obs_embedding_params'] is None
                config['obs_embedding_params'] = {}

        for k, et in config['obs_embedding_types'].items():
            if k not in config['obs_embedding_params']:
                config['obs_embedding_params'][k] = {}
            params = config['obs_embedding_params'][k]
            if et == 'conv1d_time':
                if 'stride' not in params:
                    params['stride'] = 1
                if 'dilation' not in params:
                    params['dilation'] = 1
                if 'kernel_size' not in params:
                    params['kernel_size'] = 2
                if 'padding' not in params:
                    w = (params['stride'] - 1) + (params['kernel_size'] * params['dilation']) - 1
                    params['padding'] = (w, 0)
                if 'channels' not in params:
                    params['channels'] = [64, 256, 512]
            elif et == 'conv2d':
                if 'stride' not in params:
                    params['stride'] = (1, 1)
                elif type(params['stride']) == int:
                    params['stride'] = (params['stride'], params['stride'])
                if 'dilation' not in params:
                    params['dilation'] = (1, 1)
                elif type(params['dilation']) == int:
                    params['dilation'] = (params['dilation'], params['dilation'])
                if 'kernel_size' not in params:
                    params['kernel_size'] = (2, 3)
                elif type(params['kernel_size']) == int:
                    params['kernel_size'] = (params['kernel_size'], params['kernel_size'])
                if 'padding' not in params:
                    w1 = (params['stride'][0] - 1) + (params['kernel_size'][0] * params['dilation'][0]) - 1
                    w2 = (params['stride'][1] - 1) + (params['kernel_size'][1] * params['dilation'][1]) - 1
                    w2a = int(np.ceil(w2 / 2))
                    params['padding'] = ((w1, 0), (w2a, w2 - w2a))
                    # params['padding'] = 'same'
                if 'channels' not in params:
                    params['channels'] = [16, 64, 128]
            elif et == 'conv1d_feat':
                if 'stride' not in params:
                    params['stride'] = 1
                if 'dilation' not in params:
                    params['dilation'] = 1
                if 'kernel_size' not in params:
                    params['kernel_size'] = 3
                if 'padding' not in params:
                    params['padding'] = 'same'
                if 'channels' not in params:
                    params['channels'] = [16, 64, 128]
            elif et == 'fc':
                pass
            else:
                raise ValueError(f'Unknown embedding type {et}')
            # end switch over embedding types
        # end if obs_embedding_types in config
        
        if ('test_batch_size' not in config) and ('batch_size' in config):
            config['test_batch_size'] = config['batch_size']

    return config


def overwrite_config(config0, config1, no_overwrite=()):
    # maybe fix: no_overwrite is just a list of parameter names. this may fail in recursive calls
    for k, v in config1.items():
        if k in no_overwrite:
            continue
        if (k in config0) and (config0[k] is not None) and (type(v) == dict):
            overwrite_config(config0[k], config1[k], no_overwrite=no_overwrite)
        else:
            config0[k] = v
    return


def get_modeltype_str(config, dataset):
    if config['modelstatetype'] is not None:
        modeltype_str = f"{config['modelstatetype']}_{config['modeltype']}"
    else:
        modeltype_str = config['modeltype']
    if config['categories'] is None or len(config['categories']) == 0:
        category_str = 'all'
    else:
        category_str = '_'.join(config['categories'])
    modeltype_str += f'_{category_str}'

    return modeltype_str

modelname_patterns = [r'^(.*)_(\d{8}T\d{6})_.*epoch(\d+)$', # new pattern 20251028, <modeltype_str>_<savetime>_epoch<epoch>
                      r'^(.*)_epoch(\d+)_(\d{8}T\d{6})$', # old pattern <modeltype_str>_epoch<epoch>_<savetime>
                      r'^(.*)_(\d{8}T\d{6})$'] # old pattern <modeltype_str>_<savetime>


def parse_modelfile(modelfile, modelname_pattern=None):
    _, filestr = os.path.split(modelfile)
    filestr, _ = os.path.splitext(filestr)
    
    if modelname_pattern is None:
        for modelname_pattern in modelname_patterns:
            m = re.match(modelname_pattern, filestr)
            if m is not None:
                break
    else:
        m = re.match(modelname_pattern, filestr)
    if m is None:
        modeltype_str = ''
        savetime = ''
    else:
        modeltype_str = m.groups(1)[0]
        savetime = m.groups(1)[1]
    return modeltype_str, savetime


def clean_intermediate_results(savedir):
    modelfiles = list(pathlib.Path(savedir).glob('*.pth'))
    modelfilenames = [p.name for p in modelfiles]
    
    pold = re.compile(r'^(?P<prefix>.+)_epoch(?P<epoch>\d+)_(?P<suffix>.*).pth$')
    pnew = re.compile(r'^(?P<prefix>.+)_(?P<suffix>.*)_epoch(?P<epoch>\d+).pth$')
    m = []
    for modelfilename in modelfilenames:
        mcurr = pnew.match(modelfilename)
        if mcurr is None:
            mcurr = pold.match(modelfilename)
        m.append(mcurr)
    ids = np.array([x.group('prefix') + '___' + x.group('suffix') for x in m])
    epochs = np.array([int(x.group('epoch')) for x in m])
    uniqueids, idx = np.unique(ids, return_inverse=True)
    removed = []
    nremoved = 0
    for i, id in enumerate(uniqueids):
        idxcurr = np.nonzero(ids == id)[0]
        if len(idxcurr) <= 1:
            continue
        j = idxcurr[np.argmax(epochs[idxcurr])]
        idxremove = idxcurr[epochs[idxcurr] < epochs[j]]
        while True:
            print(f'Keep {modelfilenames[j]} and remove the following files:')
            for k in idxremove:
                print(f'Remove {modelfilenames[k]}')
            r = input('(y/n) ?  ')
            if r == 'y':
                for k in idxremove:
                    LOG.info(f'Removing {modelfiles[k]}')
                    os.remove(modelfiles[k])
                    removed.append(modelfiles[k])
                    nremoved += 1
                break
            elif r == 'n':
                break
            else:
                LOG.warning('Bad input, response must be y or n')
    LOG.info(f'Removed {nremoved} files')
    return removed


def compute_scale_all_agents(data, compute_scale_per_agent):
    maxid = np.max(data['ids'])
    max_n_agents = data['X'].shape[-1]
    scale_per_agent = None

    for agent_num in tqdm.trange(max_n_agents, desc='Computing scale per agent'):
        idscurr = np.unique(data['ids'][(data['ids'][:, agent_num] >= 0) & data['isdata'][:,agent_num], agent_num])
        for id in idscurr:
            idx = data['ids'][:, agent_num] == id
            s = compute_scale_per_agent(data['X'][..., idx, agent_num])
            if scale_per_agent is None:
                scale_per_agent = np.zeros((s.size, maxid + 1))
                scale_per_agent[:] = np.nan
            else:
                assert (np.all(np.isnan(scale_per_agent[:, id]))), \
                    f"Scale for agent id {id} already computed, cannot recompute"
            scale_per_agent[:, id] = s.flatten()

    return scale_per_agent


def load_and_filter_data(infile, config, compute_scale_per_agent=None, compute_noise_params=None, keypointnames=None,
                         debug=False, n_frames_per_video=None, max_n_videos=None):
    # load data
    LOG.info(f"loading raw data from {infile}...")
    data = load_raw_npz_data(infile,debug=debug, n_frames_per_video=n_frames_per_video, max_n_videos=max_n_videos)
    LOG.info(f"loaded data with X.shape {data['X'].shape}")

    scale_per_agent = None

    # compute noise parameters
    if type(config['discreteidx']) == list and (len(config['discreteidx']) > 0) and config['discretize_epsilon'] is None:
        LOG.info('computing noise parameters...')
        if (config['all_discretize_epsilon'] is None):
            assert compute_noise_params is not None, \
                "Need 'compute_noise_params' to compute 'all_discrete_epsilon'"
            scale_per_agent = compute_scale_all_agents(data, compute_scale_per_agent)
            config['all_discretize_epsilon'] = compute_noise_params(data, scale_per_agent,
                                                                    simplify_out=config['simplify_out'])
        config['discretize_epsilon'] = config['all_discretize_epsilon'][config['discreteidx']]

    # filter out data
    LOG.info('filtering data...')
    if config['categories'] is not None and len(config['categories']) > 0:
        fn = 'isdata'
        LOG.info(f"filtering {fn} by categories {config['categories']}")
        nframespre = np.count_nonzero(data[fn])
        nidspre = len(np.unique(data['ids'][data[fn]]))
        filter_data_by_categories(data, config['categories'],fn)
        nframespost = np.count_nonzero(data[fn])
        nidspost = len(np.unique(data['ids'][data[fn]]))
        LOG.info(f"After filtering {fn} nids {nidspre} -> {nidspost}, n agent-frames {nframespre} -> {nframespost}")
        assert nidspost > 0, "No valid agents remain after filtering by categories"
        assert nframespost > 0, "No valid data remains after filtering by categories"
        
    # set useoutputmask
    if ('output_categories' in config) and (config['output_categories'] is not None) and (len(config['output_categories']) > 0):
        fn = 'useoutputmask'
        LOG.info(f"filtering {fn} by categories {config['output_categories']}")
        nframespre = np.count_nonzero(data['isdata']&data[fn])
        nidspre = len(np.unique(data['ids'][data['isdata']&data[fn]]))
        filter_data_by_categories(data, config['output_categories'],fn)
        nframespost = np.count_nonzero(data['isdata']&data[fn])
        nidspost = len(np.unique(data['ids'][data['isdata']&data[fn]]))
        LOG.info(f"After filtering {fn} nids {nidspre} -> {nidspost}, n agent-frames {nframespre} -> {nframespost}")
        assert nidspost > 0, "No valid agents remain after filtering by categories"
        assert nframespost > 0, "No valid data remains after filtering by categories"

    # augment by flipping
    if 'augment_flip' in config and config['augment_flip']:
        assert keypointnames is not None, "Need keypointnames to perform flip augmentation"
        LOG.info('augmenting data by flipping...')
        nframespre = np.count_nonzero(data['isdata']&data['useoutputmask'])
        nidspre = len(np.unique(data['ids'][data['isdata']&data['useoutputmask']]))
        flipvideoidx = np.max(data['videoidx']) + 1 + data['videoidx']
        data['videoidx'] = np.concatenate((data['videoidx'], flipvideoidx), axis=0)
        firstid = np.max(data['ids']) + 1
        flipids = data['ids'].copy()
        flipids[flipids >= 0] += firstid
        data['ids'] = np.concatenate((data['ids'], flipids), axis=0)
        data['frames'] = np.tile(data['frames'], (2, 1))
        flipX = flip_agents(data['X'], keypointnames)
        data['X'] = np.concatenate((data['X'], flipX), axis=2)
        data['y'] = np.tile(data['y'], (1, 2, 1))
        data['isdata'] = np.tile(data['isdata'], (2, 1))
        data['isstart'] = np.tile(data['isstart'], (2, 1))
        data['useoutputmask'] = np.tile(data['useoutputmask'], (2, 1))
        nframespost = np.count_nonzero(data['isdata']&data['useoutputmask'])
        nidspost = len(np.unique(data['ids'][data['isdata']&data['useoutputmask']]))
        LOG.info(f"After flip augmentation nids {nidspre} -> {nidspost}, nframes {nframespre} -> {nframespost}")

    # compute scale parameters
    if compute_scale_per_agent is None:
        return data, None

    LOG.info('computing scale parameters...')
    if scale_per_agent is None:
        scale_per_agent = compute_scale_all_agents(data, compute_scale_per_agent)

    # throw out data that is missing scale information - not so many frames
    idsremove = np.nonzero(np.any(np.isnan(scale_per_agent), axis=0))[0]
    data['isdata'][np.isin(data['ids'], idsremove)] = False
    
    # condense, only keep videos that have data, makes feature computation faster later...
    hasdata = np.any(data['isdata'], axis=1)
    videoidx = np.unique(data['videoidx'][hasdata])
    keepidx = np.isin(data['videoidx'], videoidx)[:,0]
    LOG.info(f'Condensing data: frames reduced from {len(keepidx)} to {np.count_nonzero(keepidx)}')
    if not np.all(keepidx):
        assert ~np.any(data['isdata'][~keepidx,:])
        data['X'] = data['X'][:,:,keepidx,:]
        data['y'] = data['y'][:,keepidx,:]
        data['videoidx'] = data['videoidx'][keepidx]
        data['ids'] = data['ids'][keepidx,:]
        data['frames'] = data['frames'][keepidx]
        data['isstart'] = data['isstart'][keepidx,:]
        data['isdata'] = data['isdata'][keepidx,:]    
        data['useoutputmask'] = data['useoutputmask'][keepidx,:]

    return data, scale_per_agent
