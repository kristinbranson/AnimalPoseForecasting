import os
import re
import torch
import json
import numpy as np
import pathlib
import logging

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
    if sensory_params is not None:
        tosave['SENSORY_PARAMS'] = sensory_params
    torch.save(tosave, savefile)
    return


def load_model(loadfile, model, device, lr_optimizer=None, scheduler=None, config=None):
    LOG.info(f'Loading model from file {loadfile}...')
    state = torch.load(loadfile, map_location=device)
    if model is not None:
        model.load_state_dict(state['model'])
    if lr_optimizer is not None and ('lr_optimizer' in state):
        lr_optimizer.load_state_dict(state['lr_optimizer'])
    if scheduler is not None and ('scheduler' in state):
        scheduler.load_state_dict(state['scheduler'])
    if config is not None:
        load_config_from_model_file(config=config, state=state)

    loss = {}
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
    return


def json_load_helper(jsonfile):
    with open(jsonfile, 'r') as f:
        config = json.load(f)
    config = {k: v for k, v in config.items() if re.search('^_comment', k) is None}
    return config


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
    if dataset.flatten:
        modeltype_str += '_flattened'
    if dataset.continuous and dataset.discretize:
        reptype = 'mixed'
    elif dataset.continuous:
        reptype = 'continuous'
    elif dataset.discretize:
        reptype = 'discrete'
    modeltype_str += f'_{reptype}'
    if config['categories'] is None or len(config['categories']) == 0:
        category_str = 'all'
    else:
        category_str = '_'.join(config['categories'])
    modeltype_str += f'_{category_str}'

    return modeltype_str


def parse_modelfile(modelfile, modelname_pattern=r'fly(.*)_epoch\d+_(\d{8}T\d{6})'):
    _, filestr = os.path.split(modelfile)
    filestr, _ = os.path.splitext(filestr)
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
    p = re.compile('^(?P<prefix>.+)_epoch(?P<epoch>\d+)_(?P<suffix>.*).pth$')
    m = [p.match(n) for n in modelfilenames]
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
    max_n_agents = data['X'].shape[3]
    scale_per_agent = None

    for agent_num in range(max_n_agents):

        idscurr = np.unique(data['ids'][data['ids'][:, agent_num] >= 0, agent_num])
        for id in idscurr:
            idx = data['ids'][:, agent_num] == id
            s = compute_scale_per_agent(data['X'][..., idx, agent_num])
            if scale_per_agent is None:
                scale_per_agent = np.zeros((s.size, maxid + 1))
                scale_per_agent[:] = np.nan
            else:
                assert (np.all(np.isnan(scale_per_agent[:, id])))
            scale_per_agent[:, id] = s.flatten()

    return scale_per_agent


def load_and_filter_data(infile, config, compute_scale_per_agent, compute_noise_params=None, keypointnames=None):
    # load data
    LOG.info(f"loading raw data from {infile}...")
    data = load_raw_npz_data(infile)

    # compute noise parameters
    if (len(config['discreteidx']) > 0) and config['discretize_epsilon'] is None:
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
        filter_data_by_categories(data, config['categories'])

    # augment by flipping
    if 'augment_flip' in config and config['augment_flip']:
        assert keypointnames is not None, "Need keypointnames to perform flip augmentation"
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

    # compute scale parameters
    LOG.info('computing scale parameters...')
    scale_per_agent = compute_scale_all_agents(data, compute_scale_per_agent)

    # throw out data that is missing scale information - not so many frames
    idsremove = np.nonzero(np.any(np.isnan(scale_per_agent), axis=0))[0]
    data['isdata'][np.isin(data['ids'], idsremove)] = False

    return data, scale_per_agent
