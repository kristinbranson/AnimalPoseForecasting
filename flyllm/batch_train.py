"""
Train a forecasting model of any animal (flyllm/animals.py) from a config file, as notebooks/train_fly_llm.py
does for flies, but as a script.  Model checkpoints go to <out dir>/models.  The debug and loss figures are
saved as PNG files in <out dir>/figs, overwritten each time they're updated.

Trains on all the GPUs CUDA can see, or on the first --gpus of them, with DistributedDataParallel, one process
per GPU, even for one GPU.  The datasets are loaded first, and then the processes are forked, so they share the
datasets' memory.  config['batch_size'] is each GPU's batch size, so the effective batch size is that times the
GPU count, and the learning rate, config['optimizer_args']['lr'], is multiplied by a function of the GPU count
set by config['lr_gpu_scaling']: 'sqrt' (the default), the square root of the GPU count, the scaling rule for
Adam-like optimizers (Malladi et al. 2022), or 'linear', the GPU count itself (Goyal et al. 2017), which
usually needs warmup (config['num_warmup_epochs']).  The GPU count is saved in the models' configs, as
config['train_gpu_count'], and when restarting from a model file trained on a different number of GPUs or
with a different rule, the restored learning rate is rescaled to what this run's rule and GPU count give.
Only rank 0 makes the figures and saves models.

Run it as python -m flyllm.batch_train.
"""

import argparse
import json
import logging
import math
import os

from apf.lsf import limit_threads_to_lsf_slots

limit_threads_to_lsf_slots()

# no display on cluster nodes, and the figures are only saved to files
os.environ.setdefault('MPLBACKEND', 'Agg')
# so that torch.cuda.is_available() doesn't initialize CUDA before the training processes are forked, which would
# keep them from using it
os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')

import numpy as np
import torch
from torch.utils.data import DataLoader

import apf.distributed as distributed
import apf.utils as utils
from apf.training import scale_learning_rate_bang_bang, train
from flyllm import animals
from flyllm.plotting import initialize_debug_plots, initialize_loss_plots, update_debug_plots, update_loss_plots
from flyllm.prepare import _dataloader_worker_args, init_flyllm, init_model

LOG = logging.getLogger('flyllm.batch_train')

# The values config['lr_gpu_scaling'] can have, and the learning rate factor each gives for a GPU count
LR_GPU_SCALING_FUNCTIONS = {'sqrt': math.sqrt, 'linear': float}
LR_GPU_SCALING_DESCRIPTIONS = {'sqrt': 'the square root of the GPU count', 'linear': 'the GPU count'}


def lr_gpu_scale(gpu_count: int, lr_gpu_scaling: str) -> float:
    """ The factor the learning rate is multiplied by when training on gpu_count GPUs, for config['lr_gpu_scaling']
    lr_gpu_scaling. """
    if lr_gpu_scaling not in LR_GPU_SCALING_FUNCTIONS:
        raise ValueError(f"lr_gpu_scaling is {lr_gpu_scaling!r}, but must be one of {list(LR_GPU_SCALING_FUNCTIONS)}")
    return LR_GPU_SCALING_FUNCTIONS[lr_gpu_scaling](gpu_count)


def save_figures(h: dict, fig_file_path_prefix: str) -> None:
    """ Saves each figure in the plot handle dict h, the entries whose keys start with 'fig', to a PNG file. """
    for k, fig in h.items():
        if k.startswith('fig') and fig is not None:
            fig.savefig(f'{fig_file_path_prefix}_{k}.png')


def train_on_gpu(device: torch.device, res: dict, restartmodelfile_maybe: str | None, out_model_dir_path: str,
                 out_fig_dir_path: str) -> None:
    """ Trains in one of the processes forked by distributed.run_on_gpus, on device, with the datasets in res. """
    config = res['config']
    res['device'] = device
    rank = distributed.rank()
    # each process gets its own dropout masks.  The model is initialized from rank 0's GPU RNG, since DDP copies
    # rank 0's weights to the others.
    torch.cuda.manual_seed(config['torch_seed'] + rank)

    # each process loads its share of the batches
    train_dataset = res['train_dataset']
    val_dataset = res['val_dataset']
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  sampler=distributed.ShardedSampler(train_dataset, shuffle=True,
                                                                     seed=config['torch_seed']),
                                  pin_memory=True, **_dataloader_worker_args(config))
    val_dataloader = DataLoader(val_dataset, batch_size=config.get('test_batch_size', config['batch_size']),
                                sampler=distributed.ShardedSampler(val_dataset, shuffle=False),
                                pin_memory=True, **_dataloader_worker_args(config))
    res['train_dataloader'] = train_dataloader
    res['val_dataloader'] = val_dataloader

    gpu_count = distributed.world_size()
    LOG.info('Initializing model...')
    init_model(config=config, somedataset=train_dataset, somedataloader=train_dataloader, device=device,
               restartmodelfile=restartmodelfile_maybe, mode='train', res=res,
               # model files hold their configs, which hold numpy arrays, which weights_only loading refuses
               weights_only=False)
    if restartmodelfile_maybe is not None:
        # the restored learning rate was scaled for the GPU count the model was trained on, which is 1 for model
        # files saved before the count was
        restartmodelfile: str = restartmodelfile_maybe
        saved_config = torch.load(restartmodelfile, map_location='cpu', mmap=True, weights_only=False).get('config')
        saved_gpu_count = (saved_config or {}).get('train_gpu_count', 1)
        saved_lr_gpu_scaling = (saved_config or {}).get('lr_gpu_scaling', 'sqrt')
        lr_scale = (lr_gpu_scale(gpu_count, config.get('lr_gpu_scaling', 'sqrt'))
                    / lr_gpu_scale(saved_gpu_count, saved_lr_gpu_scaling))
        if lr_scale != 1:
            LOG.info(f'{restartmodelfile} was trained on {saved_gpu_count} GPU(s), with {saved_lr_gpu_scaling} '
                     f'learning rate scaling, so scaling its learning rate by {lr_scale:.3g}')
            scale_learning_rate_bang_bang(res['optimizer'], res['lr_scheduler'], lr_scale)
    # saved with the models, via model.config
    config['train_gpu_count'] = gpu_count
    model = res['model']
    criterion = res['criterion']
    loss_epoch = res['loss_epoch']

    animal = animals.config_animal(config)
    run_name = f"{animal.name}{res['modeltype_str']}_{res['model_savetime']}"

    # the figures, on rank 0 only
    if rank == 0:
        fig_file_path_prefix = os.path.join(out_fig_dir_path, run_name)

        # debug and loss plots, as in the notebook
        debug_params = dict(animal.debug_plot_kwargs)
        # if contextl is long, still just look at samples from the first 64 frames
        if config['contextl'] > 64:
            debug_params['tsplot'] = np.round(np.linspace(0, 64, 5)).astype(int)
            debug_params['traj_nsamplesplot'] = 1
        hdebug = {
            'train': initialize_debug_plots(train_dataset, train_dataloader, res['train_data'], name='Train',
                                            **debug_params),
            'val': initialize_debug_plots(val_dataset, val_dataloader, res['val_data'], name='Val', **debug_params),
        }
        hloss = initialize_loss_plots(loss_epoch)
        valexample = next(iter(val_dataloader))

    # train() calls the hooks on rank 0 only
    def end_iter_hook(model=None, step=None, example=None, predfn=None, **kwargs):
        if step % config['niterplot'] != 0:
            return
        LOG.info(f'Updating debug plots at step {step}')
        with torch.no_grad():
            trainpred = predfn(example['input'].to(device=device))
            valpred = predfn(valexample['input'].to(device=device))
        update_debug_plots(hdebug['train'], config, model, train_dataset, example, trainpred, name='Train',
                           criterion=criterion, **debug_params)
        update_debug_plots(hdebug['val'], config, model, val_dataset, valexample, valpred, name='Val',
                           criterion=criterion, **debug_params)
        save_figures(hdebug['train'], f'{fig_file_path_prefix}_train')
        save_figures(hdebug['val'], f'{fig_file_path_prefix}_val')

    def end_epoch_hook(loss_epoch=None, epoch=None, **kwargs):
        LOG.info(f'Updating loss plots at end of epoch {epoch}')
        update_loss_plots(hloss, loss_epoch)
        save_figures(hloss, f'{fig_file_path_prefix}_loss')

    train_args = utils.function_args_from_config(config, train)
    if res['optimizer'] is None:
        # train() makes the optimizer, so scale its learning rate for the effective batch size.  The config, which
        # is saved with the models, keeps the unscaled one.
        lr_gpu_scaling = config.get('lr_gpu_scaling', 'sqrt')
        lr_scale = lr_gpu_scale(gpu_count, lr_gpu_scaling)
        train_args['optimizer_args'] = {**config['optimizer_args'], 'lr': config['optimizer_args']['lr'] * lr_scale}
        LOG.info(f"Learning rate {config['optimizer_args']['lr']} times {lr_scale:.3g}, "
                 f"{LR_GPU_SCALING_DESCRIPTIONS[lr_gpu_scaling]}: {train_args['optimizer_args']['lr']:.3g}")
    train_args['train_dataloader'] = train_dataloader
    train_args['val_dataloader'] = val_dataloader
    train_args['model'] = model
    train_args['loss_epoch'] = loss_epoch
    train_args['end_epoch_hook'] = end_epoch_hook
    train_args['end_iter_hook'] = end_iter_hook
    train_args['optimizer'] = res['optimizer']
    train_args['lr_scheduler'] = res['lr_scheduler']
    train_args['start_epoch'] = res['epoch']
    train_args['savefilestr'] = os.path.join(out_model_dir_path, run_name)
    train(**train_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='config file name')
    parser.add_argument('--overrideconfig', type=json.loads, default={},
                        help='JSON dict of config entries to override, e.g. \'{"num_train_epochs": 25}\'')
    parser.add_argument('--restartmodelfile', default=None, help='model file to restart training from')
    parser.add_argument('--debug-uselessdata', action='store_true',
                        help='train on a small subset of the data, for quick debugging runs')
    parser.add_argument('--outdir', default='.', help='directory for the models and figs dirs (default: .)')
    parser.add_argument('--gpus', type=int, default=None,
                        help='number of GPUs to train on (default: all the GPUs CUDA can see)')
    args = parser.parse_args()
    if not isinstance(args.overrideconfig, dict):
        parser.error('--overrideconfig must be a JSON dict')

    logging.basicConfig(level=logging.INFO)
    out_dir_path = os.path.abspath(args.outdir)
    out_model_dir_path = os.path.join(out_dir_path, 'models')
    out_fig_dir_path = os.path.join(out_dir_path, 'figs')
    os.makedirs(out_model_dir_path, exist_ok=True)
    os.makedirs(out_fig_dir_path, exist_ok=True)
    LOG.info(f'Writing model checkpoints to {out_model_dir_path} and figures to {out_fig_dir_path}')

    LOG.info(f'Config overrides: {args.overrideconfig}')
    # everything but the model, which is made on each process's GPU after forking
    res = init_flyllm(configfile=os.path.abspath(args.configfile), mode='train',
                      debug_uselessdata=args.debug_uselessdata, overrideconfig=args.overrideconfig, doinitmodel=False)
    if not res['success']:
        raise RuntimeError('init_flyllm failed')
    gpu_count = args.gpus if args.gpus is not None else torch.cuda.device_count()
    LOG.info(f'Training on {gpu_count} GPU(s), with an effective batch size of '
             f'{gpu_count * res["config"]["batch_size"]}')
    distributed.run_on_gpus(train_on_gpu, gpu_count, args=(res, args.restartmodelfile, out_model_dir_path,
                                                            out_fig_dir_path))
    LOG.info('Done training')


if __name__ == '__main__':
    main()
