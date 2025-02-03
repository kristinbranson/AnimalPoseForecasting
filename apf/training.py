import sys
import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import copy

from apf.dataset import Dataset
from apf.models import (
    sanity_check_temporal_dep,
    compute_loss_mixed,
    mixed_causal_criterion,
    TransformerModel,
)


def to_dataloader(dataset: Dataset, device: torch.device, batch_size: int, shuffle: bool) -> DataLoader:
    """ Convert all data in dataset to tensors and wrap dataset in a torch data loader.

    Args:
        dataset: Dataset to be wrapped into data loader
        device: Device to put the tensors on
        batch_size: Batch size that the data loader will use for training
        shuffle: Whether to shuffle items in dataset.
    """
    # Map data to torch
    for data in dataset.all_data():
        if not isinstance(data.processed, torch.Tensor):
            data.processed = torch.from_numpy(data.processed.astype(np.float32)).to(device)

    # Wrap dataset in data loader
    # TODO: See why pin_memory=True was failing
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False
    )


def init_model(train_dataset: Dataset, model_args: dict) -> TransformerModel:
    """ Initialize a transformer model based on training dataset dimensions and config parameters.

    Args:
        train_dataset: Dataset containing d_input, d_output_continuous, d_output_discrete and n_bins attributes
        model_args: Dictionary containing arguments for TransformerModel.

    Returns:
        model: Transformer model iwt random weights.
    """
    # Initialize a transformer model
    model = TransformerModel(
        d_input=train_dataset.d_input,
        d_output=train_dataset.d_output_continuous,
        d_output_discrete=train_dataset.d_output_discrete,
        nbins=train_dataset.n_bins,
        **model_args,
    )
    return model


def train(
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: TransformerModel,
        num_train_epochs: int,
        max_grad_norm: float,
        optimizer_args,
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
    # weight_discrete = 0.5
    print(f"weight_discrete = {weight_discrete}")
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

        # if np.mod(epoch + 1, 5) == 0:
        train_dataloader.dataset.recompute_chunk_indices()

    print('Done training')

    return model, best_model, loss_epoch
