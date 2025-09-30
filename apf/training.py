import sys
import torch
from torch.utils.data import DataLoader
import tqdm
import copy
import logging
from diffusers import DDPMScheduler

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
        loss_epoch: dict | None = None,
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

    Returns:
        model: Model after training on all epochs
        best_model: Model with the lowest validation loss
        losses: Stores losses per epoch for continuous, discrete, and combined objectives for training and validation.
    """
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
    if not model.variational:
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
            if model.variational:
                kld = -0.5 * torch.sum(1 + pred['logvar'] - pred['mu'].pow(2) - pred['logvar'].exp())
                beta = 0.001  # trying to keep this low to start with
                loss = loss + beta * kld
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
        if model.variational:
            stat['train KLD'] = stat['trainloss'] - stat['train loss continuous']
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

        # if np.mod(epoch + 1, 5) == 0:
        train_dataloader.dataset.recompute_chunk_indices()

    LOG.info('Done training')

    return model, best_model, loss_epoch


def train_diffusion(
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: TransformerModel,
        num_train_epochs: int,
        max_grad_norm: float,
        optimizer_args: dict,
        loss_epoch: dict | None = None,
        max_noise_T = 1000,
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

    Returns:
        model: Model after training on all epochs
        best_model: Model with the lowest validation loss
        losses: Stores losses per epoch for continuous, discrete, and combined objectives for training and validation.
    """
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
    weight_discrete = 0
    progress_bar = tqdm.notebook.tqdm(range(num_training_steps), file=sys.stdout)
    best_model = model
    best_val_loss = 10000
    noise_scheduler = DDPMScheduler(num_train_timesteps=max_noise_T, prediction_type='sample', beta_schedule="linear",
                                    beta_start=0.0001, beta_end=0.02 * (1000 / max_noise_T), clip_sample=False)
    for epoch in range(0, num_train_epochs):
        model.train()
        tr_loss = torch.tensor(0.0).to(device)
        tr_loss_discrete = torch.tensor(0.0).to(device)
        tr_loss_continuous = torch.tensor(0.0).to(device)

        nmask_train = 0
        for step, example in enumerate(train_dataloader):

            # Add noise to the input
            label = example['labels']
            noise = torch.randn_like(label)
            t = torch.randint(0, max_noise_T-1, (label.shape[0],)).long().to(device)
            noisy_label = noise_scheduler.add_noise(label, noise, t)
            example['input'][:, :, -label.shape[-1]:] = noisy_label

            pred = model(example['input'].to(device=device), t_noise=t, mask=train_src_mask, is_causal=is_causal)
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
        progress_bar.set_postfix(stat)
            # progress_bar.update(1)
        progress_bar.update(len(train_dataloader))

        # training epoch complete
        loss_epoch['train'][epoch] = tr_loss.item() / nmask_train

        # compute validation loss after this epoch
        loss_epoch['val'][epoch], _, _ = \
            compute_loss_mixed(model, val_dataloader, device, train_src_mask, weight_discrete=weight_discrete)
        ## ---
        model.eval()
        with torch.no_grad():
            all_loss = torch.zeros(len(val_dataloader), device=device)
            loss = torch.tensor(0.0).to(device)
            loss_discrete = torch.tensor(0.0).to(device)
            loss_continuous = torch.tensor(0.0).to(device)
            nmask = 0
            for i, example in enumerate(val_dataloader):
                label = example['labels']
                noise = torch.randn_like(label)
                t = torch.randint(0, max_noise_T-1, (label.shape[0],)).long().to(device)  # Should we add per batch and frame?
                noisy_label = noise_scheduler.add_noise(label, noise, t)
                example['input'][:, :, -label.shape[-1]:] = noisy_label
                pred = model(example['input'].to(device=device), mask=train_src_mask, is_causal=True)
                loss_curr, loss_discrete_curr, loss_continuous_curr = mixed_causal_criterion(
                    example, pred, weight_discrete=weight_discrete, extraout=True
                )
                nmask += example['input'].shape[0] * example['input'].shape[1]
                all_loss[i] = loss_curr
                loss += loss_curr
                loss_discrete += loss_discrete_curr
                loss_continuous += loss_continuous_curr

            loss_epoch['val'][epoch] = loss.item() / nmask
        ## ---
        last_val_loss = loss_epoch['val'][epoch].item()

        if last_val_loss < best_val_loss:
            best_model = copy.deepcopy(model)

        # if np.mod(epoch + 1, 5) == 0:
        train_dataloader.dataset.recompute_chunk_indices()

    LOG.info('Done training')

    return model, best_model, loss_epoch
