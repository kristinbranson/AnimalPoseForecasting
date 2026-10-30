"""
Train a fly forecasting model from a config file, as notebooks/train_fly_llm_female_for_reals.py does, but as a
script.  Model checkpoints go to <out dir>/models.  The debug and loss figures are saved as PNG files in
<out dir>/figs, overwritten each time they're updated.

Run it as train-flyllm.
"""

import argparse
import json
import logging
import os

from apf.lsf import limit_threads_to_lsf_slots

limit_threads_to_lsf_slots()

# no display on cluster nodes, and the figures are only saved to files
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import torch

import apf.utils as utils
from apf.training import train
from flyllm.plotting import initialize_debug_plots, initialize_loss_plots, update_debug_plots, update_loss_plots
from flyllm.prepare import init_flyllm

LOG = logging.getLogger('flyllm.train')


def save_figures(h: dict, fig_file_path_prefix: str) -> None:
    """ Saves each figure in the plot handle dict h, the entries whose keys start with 'fig', to a PNG file. """
    for k, fig in h.items():
        if k.startswith('fig') and fig is not None:
            fig.savefig(f'{fig_file_path_prefix}_{k}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='config file name')
    parser.add_argument('--overrideconfig', type=json.loads, default={},
                        help='JSON dict of config entries to override, e.g. \'{"num_train_epochs": 25}\'')
    parser.add_argument('--restartmodelfile', default=None, help='model file to restart training from')
    parser.add_argument('--outdir', default='.', help='directory for the models and figs dirs (default: .)')
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
    res = init_flyllm(configfile=os.path.abspath(args.configfile), mode='train',
                      restartmodelfile=args.restartmodelfile, overrideconfig=args.overrideconfig)
    if not res['success']:
        raise RuntimeError('init_flyllm failed')
    config = res['config']
    device = res['device']
    model = res['model']
    train_dataset = res['train_dataset']
    train_dataloader = res['train_dataloader']
    val_dataset = res['val_dataset']
    val_dataloader = res['val_dataloader']
    criterion = res['criterion']
    loss_epoch = res['loss_epoch']

    run_name = f"fly{res['modeltype_str']}_{res['model_savetime']}"
    fig_file_path_prefix = os.path.join(out_fig_dir_path, run_name)

    # debug and loss plots, as in the notebook
    debug_params = {}
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
    LOG.info('Done training')


if __name__ == '__main__':
    main()
