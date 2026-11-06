import sys
import contextlib
import torch
from torch.utils.data import DataLoader
import tqdm
import copy
import logging
from typing import Callable
import time
import numpy as np
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel

import apf.distributed as distributed

from apf.models import (
    sanity_check_temporal_dep,
    compute_loss_mixed,
    mixed_causal_criterion,
    pred_to_float32,
    TransformerModel,
)
from apf.io import save_model

LOG = logging.getLogger(__name__)


class WarmupLinearDecay:
    """ Learning rate factor for step: rises linearly over the first num_warmup_steps steps, from
    1/num_warmup_steps to 1, then falls linearly to 0 at num_training_steps.  With no warmup steps, it's the
    same as LinearLR(start_factor=1, end_factor=0).  It's a class rather than a function so that LambdaLR's
    state_dict saves its step counts, and a restart gets the schedule it was trained with.
    """
    def __init__(self, num_warmup_steps: int, num_training_steps: int):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def __call__(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return (step + 1) / self.num_warmup_steps
        return max(0., (self.num_training_steps - step) / max(1, self.num_training_steps - self.num_warmup_steps))


class WarmupLinearLR(torch.optim.lr_scheduler.LambdaLR):
    """ LambdaLR with a WarmupLinearDecay schedule, which also loads the state of the LinearLR(start_factor=1,
    end_factor=0) that init_optimizer made before there was warmup, so older checkpoints can still be restarted.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int):
        super().__init__(optimizer, WarmupLinearDecay(num_warmup_steps, num_training_steps))

    def load_state_dict(self, state_dict: dict) -> None:
        if 'lr_lambdas' not in state_dict:
            # a LinearLR's state.  Its schedule is a WarmupLinearDecay with no warmup over its total_iters.
            schedule = {'num_warmup_steps': 0, 'num_training_steps': state_dict['total_iters']}
            state_dict = {k: v for k, v in state_dict.items() if k not in ('start_factor', 'end_factor', 'total_iters')}
            state_dict['lr_lambdas'] = [dict(schedule) for _ in self.lr_lambdas]
        super().load_state_dict(state_dict)


def init_optimizer(num_training_steps: int, model: TransformerModel, optimizer_args: dict = {},
                   num_warmup_steps: int = 0) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """ Initializes AdamW optimizer and linear learning rate scheduler.

    Args:
        num_training_steps: Total number of training steps that will be taken.
        optimizer_args: Named arguments used for the AdamW optimizer.
        num_warmup_steps: Number of steps over which the learning rate rises linearly to its full value, at the
            start of the num_training_steps.  Defaults to 0, no warmup.
    Returns:
        optimizer: AdamW optimizer
        lr_scheduler: Linear learning rate scheduler: warmup, then a linear decay to 0 at num_training_steps
    """
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    lr_scheduler = WarmupLinearLR(optimizer, num_warmup_steps, num_training_steps)
    return optimizer, lr_scheduler

def scale_learning_rate_bang_bang(optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                                  factor: float) -> None:
    """ Multiplies the learning rate of every optimizer param group by factor, now and for the rest of the
    schedule, e.g. after restoring them from a checkpoint trained with a different GPU count. """
    for group in optimizer.param_groups:
        group['lr'] *= factor
        if 'initial_lr' in group:
            group['initial_lr'] *= factor
    lr_scheduler.base_lrs = [lr * factor for lr in lr_scheduler.base_lrs]
    lr_scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]


def train(
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        model: TransformerModel | None = None,
        num_train_epochs: int | None = None,
        num_warmup_epochs: float = 0,
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
        max_nonfinite_steps: int = 100,
        use_autocast: bool = False,
        autocast_dtype: str | torch.dtype = 'bfloat16',
        val_epoch: int = 1,
        use_compile: bool = False,
        use_sdpa_math_kernel: bool = True,
) -> tuple[TransformerModel, TransformerModel, dict]:
    """ Trains a model on train_dataloader, using val_dataloader to select the best_model.

    Args:
        train_dataloader: Dataloader used for training, provides examples with 'input' of shape
           (batch_size, contextl, n_in_features) and 'labels' of shape (batch_size, contextl, n_out_features)
        val_dataloader: Dataloader usef for validation.
        model: Transformer model to be trained
        num_train_epochs: How many times to loop through all examples in the dataset during training.
        num_warmup_epochs: How many epochs, possibly fractional, the learning rate rises linearly over at the start
            of training, before it falls linearly to 0 at the end.  Only used if train makes the optimizer and
            scheduler; a restart keeps the schedule it was trained with.  Defaults to 0, no warmup.
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
        max_nonfinite_steps: Training steps whose prediction, loss, or gradient norm is non-finite are skipped
            (no optimizer step) and logged. Raises RuntimeError once more than this many steps have been
            skipped in total. Defaults to 100.
        use_autocast: If True, runs model forward passes (training and validation) under torch.autocast with
            dtype autocast_dtype. This is much faster on recent GPUs. Weights, gradients, optimizer state, and
            losses stay float32. Defaults to False.
        autocast_dtype: dtype used for autocast if use_autocast is True, e.g. 'bfloat16'. Ignored if
            use_autocast is False. Defaults to 'bfloat16'.
        val_epoch: Computes the validation loss every val_epoch epochs, and always after the last epoch.
            Validation losses for other epochs are left as nan, and the best model is selected among the
            epochs that were validated. Defaults to 1.
        use_compile: If True, runs model forward passes (training and validation) through torch.compile(model).
            This is faster per step, but the first step (and the first validation pass) takes about a minute
            to compile. Saved models, best_model, and the hooks use the uncompiled model, which shares the
            compiled model's weights. Defaults to False.
        use_sdpa_math_kernel: If True, training and validation run under sdpa_kernel(SDPBackend.MATH), which
            restricts attention to PyTorch's unfused implementation. This was needed to avoid nans in
            training, but it is slower than the fused kernels PyTorch otherwise chooses, and stores the full
            attention matrix. Defaults to True.

    In a process started by apf.distributed.run_on_gpus, trains with DistributedDataParallel, together with the
    other processes: model must be on this process's GPU, and the dataloaders should sample with
    apf.distributed.ShardedSampler, so each process gets its share of each epoch.  Each training step's
    gradients are averaged over the processes, so the effective batch size is the processes' batch sizes
    summed.  Losses are summed over all the processes' batches, and a step is skipped in every process if
    it's non-finite in any.  Only rank 0 calls the hooks, saves models, and returns the best model; the other
    ranks return model as best_model.  Outside of run_on_gpus, trains in this process alone, as before.

    Returns:
        model: Model after training on all epochs
        best_model: Model with the lowest validation loss
        losses: Stores losses per epoch for continuous, discrete, and combined objectives for training and validation.
    """
    device = next(model.parameters()).device
    is_rank0 = distributed.rank() == 0

    if use_autocast:
        if autocast_dtype is None:
            raise ValueError('use_autocast is True, but autocast_dtype is None')
        if isinstance(autocast_dtype, str):
            autocast_dtype = getattr(torch, autocast_dtype)
        LOG.info(f'Using autocast with dtype {autocast_dtype}')
    else:
        # compute_loss_mixed uses autocast iff autocast_dtype is not None
        autocast_dtype = None
    if val_epoch is None or val_epoch < 1:
        val_epoch = 1

    # compiled wrapper sharing model's parameters, used only for forward passes
    if use_compile:
        LOG.info('Using torch.compile')
        fwd_model = torch.compile(model)
    else:
        fwd_model = model
    # training forward passes go through DDP, which averages the gradients over the processes in backward().
    # Validation uses fwd_model, since DDP syncs the batch norm buffers in some forward passes, and the
    # processes can have different numbers of validation batches.
    if dist.is_initialized():
        train_model = DistributedDataParallel(fwd_model, device_ids=[device.index])
    else:
        train_model = fwd_model

    # Optimizer
    num_training_steps = num_train_epochs * len(train_dataloader)
    if optimizer is None or lr_scheduler is None:
        num_warmup_steps = round(num_warmup_epochs * len(train_dataloader))
        if num_warmup_steps > 0:
            LOG.info(f'Warming up the learning rate over {num_warmup_steps} steps ({num_warmup_epochs} epochs)')
        optimizer, lr_scheduler = init_optimizer(num_training_steps, model, optimizer_args, num_warmup_steps)

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
            loss_epoch[key][start_epoch:] = torch.nan
            
    last_val_loss = None

    # Create attention mask
    example = next(iter(train_dataloader))
    contextl = example['input'].shape[1]
    train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(contextl, device=device)
    is_causal = True

    # Sanity check on temporal dependencies
    # TODO: Does tmess=300 make sense when contextl is < 300?
    if not model.input_gt_for_training:
        sanity_check_temporal_dep(train_dataloader, device, train_src_mask, is_causal, model, tmess=300)

    # Train
    d_discrete = train_dataloader.dataset.d_output_discrete
    d_continuous = train_dataloader.dataset.d_output_continuous
    weight_discrete = d_discrete / (d_discrete + d_continuous)
    LOG.info(f"weight_discrete = {weight_discrete}")
    progress_bar = tqdm.tqdm(range(num_training_steps), file=sys.stdout, disable=not is_rank0)
    best_model = model
    best_epoch = None
    best_val_loss = 10000
    n_nonfinite_steps = 0

    if use_sdpa_math_kernel:
        LOG.info('Using the math SDPA kernel for attention')
    with sdpa_kernel(SDPBackend.MATH) if use_sdpa_math_kernel else contextlib.nullcontext():
        for epoch in range(start_epoch, num_train_epochs):
            # sets fwd_model and model too, since they're submodules of train_model
            train_model.train()
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            tr_loss = torch.tensor(0.0).to(device)
            tr_loss_discrete = torch.tensor(0.0).to(device)
            tr_loss_continuous = torch.tensor(0.0).to(device)

            nmask_train = 0
            for step, example in enumerate(train_dataloader):

                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                    if model.input_gt_for_training:
                        pred = train_model(example['input'].to(device=device), mask=train_src_mask,
                                           is_causal=is_causal, gt=example['labels_discrete'].to(device=device))
                    else:
                        pred = train_model(example['input'].to(device=device), mask=train_src_mask, is_causal=is_causal)
                # compute the loss in float32
                pred = pred_to_float32(pred)
                # check the prediction first, since the criterion asserts on nans.  The prediction and loss
                # checks agree across processes, since a step's backward() must run in all of them or none.
                nonfinite_reason = None
                pred_finite = torch.stack([torch.isfinite(v).all() for v in pred.values() if torch.is_tensor(v)]).all()
                if not distributed.all_true(pred_finite):
                    nonfinite_reason = 'prediction'
                else:
                    loss, loss_discrete, loss_continuous = criterion(
                        example, pred, weight_discrete=weight_discrete, extraout=True
                    )
                    if not distributed.all_true(torch.isfinite(loss)):
                        nonfinite_reason = 'loss'
                    else:
                        loss.backward()
                        # gradient clipping; a non-finite norm would turn all gradients (and then weights) into nans.
                        # The gradients are averaged over the processes by now, so the norm is the same in all.
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        if not torch.isfinite(grad_norm).item():
                            nonfinite_reason = f'gradient norm ({grad_norm.item()})'

                if nonfinite_reason is not None:
                    n_nonfinite_steps += 1
                    metadata = {k: [int(x) for x in v]
                                for k, v in example.get('metadata', {}).items() if k in ('start_frame', 'agent_id')}
                    LOG.warning(f'Skipping step {step} of epoch {epoch}: non-finite {nonfinite_reason} '
                                f'({n_nonfinite_steps} skipped so far). Batch chunks: {metadata}')
                    # free the autograd graph now, rather than holding it through the next forward pass
                    del pred
                    if nonfinite_reason == 'loss':
                        del loss, loss_discrete, loss_continuous
                    model.zero_grad()
                    # keep the learning rate schedule aligned with the step count
                    lr_scheduler.step()
                    if n_nonfinite_steps > max_nonfinite_steps:
                        raise RuntimeError(f'More than {max_nonfinite_steps} training steps had non-finite '
                                           f'{nonfinite_reason}; aborting')
                    continue

                # how many timepoints are in this batch for normalization
                nmask_train += example['useoutputmask'].sum().detach()

                tr_loss_step = loss.detach()
                tr_loss += tr_loss_step
                tr_loss_discrete += loss_discrete.detach()
                tr_loss_continuous += loss_continuous.detach()

                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

                if end_iter_hook is not None and is_rank0:
                    end_iter_hook(model=model, step=step, example=example,
                                predfn=lambda input: model.output(input, mask=train_src_mask, is_causal=is_causal))

            # sum the losses over the processes
            tr_loss = distributed.all_reduce_sum(tr_loss)
            tr_loss_discrete = distributed.all_reduce_sum(tr_loss_discrete)
            tr_loss_continuous = distributed.all_reduce_sum(tr_loss_continuous)
            nmask_train = distributed.all_reduce_sum(torch.as_tensor(nmask_train, dtype=torch.float64,
                                                                      device=device)).item()

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

            # compute validation loss after this epoch, every val_epoch epochs
            do_val = ((epoch + 1) % val_epoch == 0) or (epoch == num_train_epochs - 1)
            if do_val:
                loss_epoch['val'][epoch], loss_epoch['val_discrete'][epoch], loss_epoch['val_continuous'][epoch] = \
                    compute_loss_mixed(fwd_model, val_dataloader, device, train_src_mask, weight_discrete=weight_discrete,
                                       autocast_dtype=autocast_dtype, input_gt=model.input_gt_for_training)
                last_val_loss = loss_epoch['val'][epoch].item()

            if end_epoch_hook is not None and is_rank0:
                end_epoch_hook(model=model, epoch=epoch, loss_epoch=loss_epoch)

            if do_val and last_val_loss < best_val_loss and is_rank0:
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                best_val_loss = last_val_loss

            if (((epoch + 1) % save_epoch == 0) or (epoch == num_train_epochs - 1)) and is_rank0:
                savefile = f'{savefilestr}_epoch{epoch + 1}.pth'
                print(f'Saving to file {savefile}')
                save_model(savefile, model,
                            lr_optimizer=optimizer,
                            scheduler=lr_scheduler,
                            loss=loss_epoch)

            # if np.mod(epoch + 1, 5) == 0:
            # rank 0 picks the random offset, so that all the processes' datasets have the same chunks
            start_offset = distributed.broadcast_int(np.random.randint(train_dataloader.dataset.context_length),
                                                     device)
            train_dataloader.dataset.recompute_chunk_indices(start_offset=start_offset)

    # save best model
    if best_epoch is not None and is_rank0:
        savefile = f'{savefilestr}_bestepoch{best_epoch + 1}.pth'
        print(f'Saving best model to file {savefile}')
        save_model(savefile, best_model,
                    lr_optimizer=optimizer,
                    scheduler=lr_scheduler,
                    loss=loss_epoch)

    LOG.info('Done training')

    return model, best_model, loss_epoch
