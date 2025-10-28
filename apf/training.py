import sys
import torch
from torch.utils.data import DataLoader
import tqdm
import copy
import logging
from typing import Callable
import os
import time

from apf.models import (
    sanity_check_temporal_dep,
    compute_loss_mixed,
    mixed_causal_criterion,
    TransformerModel,
)
from apf.io import save_model

LOG = logging.getLogger(__name__)


def init_optimizer(num_training_steps: int, model: TransformerModel, optimizer_args: dict = {}) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """ Initializes AdamW optimizer and linear learning rate scheduler.

    Args:
        num_training_steps: Total number of training steps that will be taken.
        optimizer_args: Named arguments used for the AdamW optimizer.
    Returns:
        optimizer: AdamW optimizer
        lr_scheduler: Linear learning rate scheduler
    """
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0.,
                                                    total_iters=num_training_steps)
    return optimizer, lr_scheduler

def train(
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        model: TransformerModel | None = None,
        num_train_epochs: int | None = None,
        max_grad_norm: float | None = None,
        optimizer_args: dict | None = None,
        loss_epoch: dict | None = None,
        end_epoch_hook: Callable | None = None,
        end_iter_hook: Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
        start_epoch: int = 0,
        savefilestr: str | None = None,
        save_epoch: int | None = None,
) -> tuple[TransformerModel, TransformerModel, dict]:
    """ Trains a model on train_dataloader, using val_dataloader to select the best_model.

    Args:
        train_dataloader: Dataloader used for training, provides examples with 'input' of shape
           (batch_size, contextl, n_in_features) and 'labels' of shape (batch_size, contextl, n_out_features)
        val_dataloader: Dataloader usef for validation.
        model: Transformer model to be trained
        num_train_epochs: How many times to loop through all examples in the dataset during training.
        max_grad_norm: Threshold for clipping gradients during training.
        optimizer_args: Named arguments used for the AdamW optimizer.
        loss_epoch: Keeps track of training and validation losses. If None, initializes a new structure.
        Useful for resuming training. Defaults to None.
        end_epoch_hook: Function called at the end of each epoch. Signature:
            def end_epoch_hook(model: TransformerModel, epoch: int, loss_epoch: dict) -> None
        end_iter_hook: Function called at the end of each training iteration. Signature:
            def end_iter_hook(model: TransformerModel, step: int, example: dict, predfn: Callable) -> None
            where predfn is a function that predicts with the model on the input example. 
        optimizer: If provided, uses this optimizer instead of initializing a new optimizer. Useful for 
        resuming training. Defaults to None.
        lr_scheduler: If provided, uses this learning rate scheduler instead of initializing a new one. Useful 
        for resuming training. Defaults to None.
        start_epoch: If resuming training, the epoch to start from. Defaults to 0.
        savefilestr: Saves the model to this file every save_epoch epochs. Defaults to None.
        save_epoch: Number of epochs between saving the model. If None, does not save. Defaults to None.

    Returns:
        model: Model after training on all epochs
        best_model: Model with the lowest validation loss
        losses: Stores losses per epoch for continuous, discrete, and combined objectives for training and validation.
    """
    device = next(model.parameters()).device

    # Optimizer
    num_training_steps = num_train_epochs * len(train_dataloader)
    if optimizer is None or lr_scheduler is None:
        optimizer, lr_scheduler = init_optimizer(num_training_steps, model, optimizer_args)

    # Criterion
    criterion = mixed_causal_criterion
        
    # set savefilestr
    if savefilestr is None:
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        savefilestr = f'apf_model_{timestamp}'
        print(f'Model(s) will be saved to {savefilestr}_epoch<epoch>.pth')

    # Initialize structure to keep track of loss
    if loss_epoch is None:
        loss_epoch = {}
    for key in ['train', 'val', 'train_continuous', 'train_discrete', 'val_continuous', 'val_discrete']:
        if key not in loss_epoch:
            loss_epoch[key] = torch.zeros(num_train_epochs)
            loss_epoch[key][:] = torch.nan
        else:
            # overwrite future epochs with nan in case they are not nan
            loss_epoch[key][epoch:] = torch.nan
            
    last_val_loss = None

    # Create attention mask
    example = next(iter(train_dataloader))
    contextl = example['input'].shape[1]
    train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(contextl, device=device)
    is_causal = True

    # Sanity check on temporal dependencies
    # TODO: Does tmess=300 make sense when contextl is < 300?
    sanity_check_temporal_dep(train_dataloader, device, train_src_mask, is_causal, model, tmess=300)

    # Train
    d_discrete = train_dataloader.dataset.d_output_discrete
    d_continuous = train_dataloader.dataset.d_output_continuous
    weight_discrete = d_discrete / (d_discrete + d_continuous)
    LOG.info(f"weight_discrete = {weight_discrete}")
    progress_bar = tqdm.tqdm(range(num_training_steps), file=sys.stdout)
    best_model = model
    best_epoch = None
    best_val_loss = 10000
        
    for epoch in range(start_epoch, num_train_epochs):
        model.train()
        tr_loss = torch.tensor(0.0).to(device)
        tr_loss_discrete = torch.tensor(0.0).to(device)
        tr_loss_continuous = torch.tensor(0.0).to(device)

        nmask_train = 0
        for step, example in enumerate(train_dataloader):

            pred = model(example['input'].to(device=device), mask=train_src_mask, is_causal=is_causal)
            loss, loss_discrete, loss_continuous = criterion(
                example, pred, weight_discrete=weight_discrete, extraout=True
            )
            loss.backward()

            # how many timepoints are in this batch for normalization
            nmask_train += example['input'].shape[0] * contextl

            tr_loss_step = loss.detach()
            tr_loss += tr_loss_step
            tr_loss_discrete += loss_discrete.detach()
            tr_loss_continuous += loss_continuous.detach()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            
            if end_iter_hook is not None:
                end_iter_hook(model=model, step=step, example=example, 
                            predfn=lambda input: model.output(input, mask=train_src_mask, is_causal=is_causal))

        # update progress bar
        stat = {'train loss': tr_loss.item() / nmask_train, 'last val loss': last_val_loss, 'epoch': epoch}
        stat['train loss discrete'] = tr_loss_discrete.item() / nmask_train
        stat['train loss continuous'] = tr_loss_continuous.item() / nmask_train
        progress_bar.set_postfix(stat)
            # progress_bar.update(1)
        progress_bar.update(len(train_dataloader))

        # training epoch complete
        loss_epoch['train'][epoch] = tr_loss.item() / nmask_train
        loss_epoch['train_discrete'][epoch] = tr_loss_discrete.item() / nmask_train
        loss_epoch['train_continuous'][epoch] = tr_loss_continuous.item() / nmask_train

        # compute validation loss after this epoch
        loss_epoch['val'][epoch], loss_epoch['val_discrete'][epoch], loss_epoch['val_continuous'][epoch] = \
            compute_loss_mixed(model, val_dataloader, device, train_src_mask, weight_discrete=weight_discrete)
        last_val_loss = loss_epoch['val'][epoch].item()
        
        if end_epoch_hook is not None:
            end_epoch_hook(model=model, epoch=epoch, loss_epoch=loss_epoch)

        if last_val_loss < best_val_loss:
            best_model = copy.deepcopy(model)
            best_epoch = epoch            
            
        if ((epoch + 1) % save_epoch == 0) or (epoch == num_train_epochs - 1):
            savefile = f'{savefilestr}_epoch{epoch + 1}.pth'
            print(f'Saving to file {savefile}')
            save_model(savefile, model,
                        lr_optimizer=optimizer,
                        scheduler=lr_scheduler,
                        loss=loss_epoch)

        # if np.mod(epoch + 1, 5) == 0:
        train_dataloader.dataset.recompute_chunk_indices()

    # save best model
    if best_epoch is not None:
        savefile = f'{savefilestr}_bestmodel_epoch{best_epoch + 1}.pth'
        print(f'Saving best model to file {savefile}')
        save_model(savefile, best_model,
                    lr_optimizer=optimizer,
                    scheduler=lr_scheduler,
                    loss=loss_epoch)

    LOG.info('Done training')

    return model, best_model, loss_epoch
