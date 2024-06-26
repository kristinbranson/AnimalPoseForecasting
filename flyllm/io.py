import os
import re
import torch
import json
import numpy as np
import pathlib

from flyllm.config import SENSORY_PARAMS, featglobal, posenames
from flyllm.features import get_sensory_feature_idx

codedir = pathlib.Path(__file__).parent.resolve()
DEFAULTCONFIGFILE = os.path.join(codedir, 'config_fly_llm_default.json')
assert os.path.exists(DEFAULTCONFIGFILE), f"{DEFAULTCONFIGFILE} does not exist."


def save_model(savefile, model, lr_optimizer=None, scheduler=None, loss=None, config=None):
    tosave = {'model': model.state_dict()}
    if lr_optimizer is not None:
        tosave['lr_optimizer'] = lr_optimizer.state_dict()
    if scheduler is not None:
        tosave['scheduler'] = scheduler.state_dict()
    if loss is not None:
        tosave['loss'] = loss
    if config is not None:
        tosave['config'] = config
    tosave['SENSORY_PARAMS'] = SENSORY_PARAMS
    torch.save(tosave, savefile)
    return


def load_model(loadfile, model, device, lr_optimizer=None, scheduler=None, config=None):
    print(f'Loading model from file {loadfile}...')
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


def load_config_from_model_file(loadmodelfile=None, config=None, state=None, no_overwrite=[]):
    if state is None:
        assert loadmodelfile is not None
        print(f'Loading config from file {loadmodelfile}...')
        state = torch.load(loadmodelfile)
    if config is not None and 'config' in state:
        overwrite_config(config, state['config'], no_overwrite=no_overwrite)
    else:
        print(f'config not stored in model file {loadmodelfile}')
    if 'SENSORY_PARAMS' in state:
        for k, v in state['SENSORY_PARAMS'].items():
            SENSORY_PARAMS[k] = v
    else:
        print(f'SENSORY_PARAMS not stored in model file {loadmodelfile}')
    return


def json_load_helper(jsonfile):
    with open(jsonfile, 'r') as f:
        config = json.load(f)
    config = {k: v for k, v in config.items() if re.search('^_comment', k) is None}
    return config


def read_config(jsonfile):
    config = json_load_helper(DEFAULTCONFIGFILE)
    config1 = json_load_helper(jsonfile)

    # destructive to config
    overwrite_config(config, config1)

    config['intrainfile'] = os.path.join(config['datadir'], config['intrainfilestr'])
    config['invalfile'] = os.path.join(config['datadir'], config['invalfilestr'])

    if type(config['flatten_obs_idx']) == str:
        if config['flatten_obs_idx'] == 'sensory':
            config['flatten_obs_idx'] = get_sensory_feature_idx()
        else:
            raise ValueError(f"Unknown type {config['flatten_obs_idx']} for flatten_obs_idx")

    # discreteidx will reference apf.config.posenames
    if type(config['discreteidx']) == str:
        if config['discreteidx'] == 'global':
            config['discreteidx'] = featglobal.copy()
        else:
            raise ValueError(f"Unknown type {config['discreteidx']} for discreteidx")
    if type(config['discreteidx']) == list:
        for i, v in enumerate(config['discreteidx']):
            if type(v) == str:
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


def overwrite_config(config0, config1, no_overwrite=[]):
    # maybe fix: no_overwrite is just a list of parameter names. this may fail in recursive calls
    for k, v in config1.items():
        if k in no_overwrite:
            continue
        if (k in config0) and (config0[k] is not None) and (type(v) == dict):
            overwrite_config(config0[k], config1[k], no_overwrite=no_overwrite)
        else:
            config0[k] = v
    return


# MLM - no sensory
# loadmodelfile = os.path.join(savedir,'flymlm_71G01_male_epoch100_202301215712.pth')
# MLM with sensory
# loadmodelfile = os.path.join(savedir,'flymlm_71G01_male_epoch100_202301003317.pth')
# CLM with sensory
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_202301211242.pth')
# CLM with sensory but only global motion output
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch15_202301014322.pth')
# loadmodelfile = None
# CLM, predicting forward, sideways vel
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_202302060458.pth')
# CLM, trained with dropout = 0.8 on movement
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230228T193725.pth')
# CLM, trained with dropout = 0.8 on movement, more wall touch keypoints
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230302T221828.pth')
# CLM, trained with dropout = 0.8 on movement, other fly touch features
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230303T230750.pth')
# CLM, trained with dropout = 1.0 on movement, other fly touch features, 10 layers, 512 context
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230305T135655.pth')
# CLM with mixed continuous and discrete state
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230419T175759.pth')
# CLM with mixed continuous and discrete state, movement input
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230421T223920.pth')
# flattened CLM, forward, sideways, orientation are binned outputs
# loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230512T202000.pth')
# flattened CLM, forward, sideways, orientation are binned outputs, do_separate_inputs = True
# loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flyclm_flattened_mixed_71G01_male_epoch54_20230517T153613.pth'

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


def parse_modelfile(modelfile):
    _, filestr = os.path.split(modelfile)
    filestr, _ = os.path.splitext(filestr)
    m = re.match(r'fly(.*)_epoch\d+_(\d{8}T\d{6})', filestr)
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
                    print(f'Removing {modelfiles[k]}')
                    os.remove(modelfiles[k])
                    removed.append(modelfiles[k])
                    nremoved += 1
                break
            elif r == 'n':
                break
            else:
                print('Bad input, response must be y or n')
    print(f'Removed {nremoved} files')
    return removed

