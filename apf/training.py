import sys
import torch
from torch.utils.data import DataLoader
import tqdm
import copy
import logging
import os
import time

from apf.models import (
    sanity_check_temporal_dep,
    compute_loss_mixed,
    mixed_causal_criterion,
    TransformerModel,
)

LOG = logging.getLogger(__name__)


def train(
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: TransformerModel,
        num_train_epochs: int,
        max_grad_norm: float,
        optimizer_args: dict,
        loss_epoch: dict = None,
        savedir: str = None,
        save_epoch: int = 10,
        model_nickname: str = "model",
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
         loss_epoch: Keeps track of training and validation losses. If provided, morphs this input so that
            if training is aborted the losses are still available.
         savedir: If not None, specifies the directory of where to save the model.
         save_epoch: Save model every this many epochs.
         model_nickname: What to name the model.

    Returns:
        model: Model after training on all epochs
        best_model: Model with the lowest validation loss
        losses: Stores losses per epoch for continuous, discrete, and combined objectives for training and validation.
    """
    save_path = None
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        save_path = os.path.join(savedir, f'{model_nickname}_{timestamp}.pth')
        print(save_path)

    device = next(model.parameters()).device

    # Optimizer
    num_training_steps = num_train_epochs * len(train_dataloader)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0.,
                                                     total_iters=num_training_steps)

    # Initialize structure to keep track of loss
    if loss_epoch is None:
        loss_epoch = {}
    for key in ['train', 'val', 'train_continuous', 'train_discrete', 'val_continuous', 'val_discrete']:
        loss_epoch[key] = torch.zeros(num_train_epochs)
        loss_epoch[key][:] = torch.nan
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
    best_val_loss = 10000
    for epoch in range(0, num_train_epochs):
        model.train()
        tr_loss = torch.tensor(0.0).to(device)
        tr_loss_discrete = torch.tensor(0.0).to(device)
        tr_loss_continuous = torch.tensor(0.0).to(device)

        nmask_train = 0
        for step, example in enumerate(train_dataloader):

            pred = model(example['input'].to(device=device), mask=train_src_mask, is_causal=is_causal)
            loss, loss_discrete, loss_continuous = mixed_causal_criterion(
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

        # update progress bar
        stat = {'trainloss': tr_loss.item() / nmask_train, 'lastvalloss': last_val_loss, 'epoch': epoch}
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

        if last_val_loss < best_val_loss:
            best_model = copy.deepcopy(model)

        if savedir is not None and epoch > 0 and epoch % save_epoch == 0:
            # TODO: Switch apf.io.save_model once that's refactored
            #   (this will save config and dataset parameters along with the model)
            torch.save(model.state_dict(), save_path.replace('.pth', f'_epoch{epoch}.pth'))
            torch.save(best_model.state_dict(), save_path.replace('.pth', f'_epoch{epoch}_best.pth'))

        # if np.mod(epoch + 1, 5) == 0:
        train_dataloader.dataset.recompute_chunk_indices()

    if savedir is not None:
        torch.save(model.state_dict(), save_path)
        torch.save(model.state_dict(), save_path.replace('.pth', f'_best.pth'))

    LOG.info('Done training')

    return model, best_model, loss_epoch, save_path
