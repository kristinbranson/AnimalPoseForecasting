import numpy as np
import math
import typing
import torch
from apf.models import TransformerModel, PositionalEncoding, lossfcn_discrete, lossfcn_continuous


def CrossEntropyLossEps(eps=1e-8,**kwargs):
    ce = torch.nn.CrossEntropyLoss(**kwargs)
    return lambda pred,target: ce((pred+eps)/(1+eps*pred.shape[1]),target)


def dct_consistency(pred):
    return


def prob_causal_criterion(tgt, pred, useoutputmask=None):
    stateprob = pred['stateprob']
    perstate = pred['perstate']
    if useoutputmask is not None:
        stateprob = stateprob[useoutputmask.to(device=stateprob.device)]
        perstate = perstate[useoutputmask.to(device=perstate.device)]
        tgt = tgt[useoutputmask.to(device=tgt.device)]
    d = tgt.shape[-1]
    err = torch.sum(
        stateprob * torch.sum(torch.abs(tgt[..., None] - perstate) / d, keepdim=False, axis=-2))
    return err


def min_causal_criterion(tgt, pred, useoutputmask=None):
    d = tgt.shape[-1]
    if useoutputmask is not None:
        pred = pred[useoutputmask.to(device=pred.device)]
        tgt = tgt[useoutputmask.to(device=tgt.device)]
    errperstate = torch.sum(torch.abs(tgt[..., None] - pred) / d, keepdim=False, dim=tuple(range(pred.ndim - 1)))
    err = torch.min(errperstate, dim=-1)
    return err


def masked_criterion(tgt, pred, mask, useoutputmask=None):
    if useoutputmask is not None:
        mask = mask & useoutputmask.to(device=mask.device)
    d = tgt.shape[-1]
    err = torch.sum(torch.abs(tgt[mask, :] - pred[mask, :])) / d
    return err


def mixed_masked_criterion(tgt, pred, mask, device, weight_discrete=.5, extraout=False):
    n = torch.count_nonzero(mask)
    if 'labels' in tgt:
        err_continuous = lossfcn_continuous(pred['continuous'][mask, :], tgt['labels'].to(device=device)[mask, :]) * n
    else:
        err_continuous = 0.
    if 'labels_discrete' in tgt:
        err_discrete = lossfcn_discrete(pred['discrete'][mask, ...],
                                        tgt['labels_discrete'].to(device=device)[mask, ...]) * n
    else:
        err_discrete = 0.
    err = (1 - weight_discrete) * err_continuous + weight_discrete * err_discrete
    if extraout:
        return err, err_discrete, err_continuous
    else:
        return err




def mixed_criterion_wrapper(example, pred, criterion, dataset, config):
    if 'continuous' in example:
        tgt_continuous = example['continuous']
    elif 'labels' in example:
        tgt_continuous = example['labels']
    else:
        tgt_continuous = None
    if 'discrete' in example:
        tgt_discrete = example['discrete']
    elif 'labels_discrete' in example:
        tgt_discrete = example['labels_discrete']
    else:
        tgt_discrete = None
    tgt = {'labels': tgt_continuous, 'labels_discrete': tgt_discrete, 'useoutputmask': example.get('useoutputmask', None)}

    if 'continuous' in pred:
        pred_continuous = pred['continuous']
    elif 'labels' in pred:
        pred_continuous = pred['labels']
    else:
        pred_continuous = None

    if 'labels_discrete' in pred:
        pred_discrete = pred['labels_discrete']
    else:
        pred_discrete = None

    pred1 = {'continuous': pred_continuous, 'discrete': pred_discrete}

    if dataset.discretize:
        loss, loss_discrete, loss_continuous = criterion(tgt, pred1, weight_discrete=config['weight_discrete'],
                                                         extraout=True)
    else:
        loss = criterion(tgt_continuous.to(device=pred_continuous.device), pred_continuous, useoutputmask=tgt['useoutputmask'])
        loss_continuous = loss
        loss_discrete = 0.
    return loss, loss_discrete, loss_continuous


class TransformerBestStateModel(torch.nn.Module):

    def __init__(self, d_input: int, d_output: int,
                 d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
                 nlayers: int = 12, dropout: float = 0.1, nstates: int = 8,
                 dataset_params=None,config=None):
        super().__init__()
        self.model_type = 'TransformerBestState'

        # frequency-based representation of word position with dropout
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # create self-attention + feedforward network module
        # d_model: number of input features
        # nhead: number of heads in the multiheadattention models
        # dhid: dimension of the feedforward network model
        # dropout: dropout value
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)

        # stack of nlayers self-attention + feedforward layers
        # nlayers: number of sub-encoder layers in the encoder
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)

        # encoder and decoder are currently not tied together, but maybe they should be?
        # fully-connected layer from input size to d_model
        self.encoder = torch.nn.Linear(d_input, d_model)

        # for each hidden state, fully connected layer from model to output size
        # concatenated together, so output is size d_output * nstates
        self.decode = torch.nn.Linear(d_model, nstates * d_output)

        # store hyperparameters
        self.d_model = d_model
        self.nstates = nstates
        self.d_output = d_output

        # dataset parameters
        self.dataset_params = dataset_params

        # config, primarily for book-keeping
        self.config = config

        self.init_weights()

    def init_weights(self) -> None:
        pass

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
        """
    Args:
      src: Tensor, shape [batch_size,seq_len,dinput]
      src_mask: Tensor, shape [seq_len,seq_len]
    Returns:
      Tensor of shape [batch_size, seq_len, d_output, nstates]
    """

        # project input into d_model space, multiple by sqrt(d_model) for reasons?
        src = self.encoder(src) * math.sqrt(self.d_model)

        # add in the positional encoding of where in the sentence the words occur
        # it is weird to me that these are added, but I guess it would be almost
        # the same to have these be combined in a single linear layer
        src = self.pos_encoder(src)

        # main transformer layers
        transformer_output = self.transformer_encoder(src, mask=src_mask, is_causal=is_causal)

        # output given each hidden state
        # batch_size x seq_len x d_output x nstates
        output = self.decode(transformer_output).reshape(src.shape[:-1] + (self.d_output, self.nstates))

        return output

    def randpred(self, pred):
        contextl = pred.shape[-3]
        draw = torch.randint(0, pred.shape[-1], contextl)
        return pred[..., np.arange(contextl, dtype=int), :, draw]


class TransformerStateModel(torch.nn.Module):

    def __init__(self, d_input: int, d_output: int,
                 d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
                 nlayers: int = 12, dropout: float = 0.1, nstates: int = 64,
                 minstateprob: float = None, dataset_params=None,
                 config=None):
        super().__init__()
        self.model_type = 'TransformerState'

        # frequency-based representation of word position with dropout
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # create self-attention + feedforward network module
        # d_model: number of input features
        # nhead: number of heads in the multiheadattention models
        # dhid: dimension of the feedforward network model
        # dropout: dropout value
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)

        # stack of nlayers self-attention + feedforward layers
        # nlayers: number of sub-encoder layers in the encoder
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)

        # encoder and decoder are currently not tied together, but maybe they should be?
        # fully-connected layer from input size to d_model
        self.encoder = torch.nn.Linear(d_input, d_model)

        # from output of transformer layers to hidden state probabilities
        self.state = torch.nn.Sequential(
            torch.nn.Linear(d_model, nstates),
            torch.nn.Dropout(dropout),
            torch.nn.Softmax(dim=-1)
        )
        if minstateprob is None:
            minstateprob = .01 / nstates
        # for each hidden state, fully connected layer from model to output size
        # concatenated together, so output is size d_output * nstates
        self.decode = torch.nn.Linear(d_model, nstates * d_output)

        # store hyperparameters
        self.d_model = d_model
        self.nstates = nstates
        self.d_output = d_output
        self.minstateprob = minstateprob

        # dataset parameters
        self.dataset_params = dataset_params

        # config, primarily for book-keeping
        self.config = config

        self.init_weights()

    def init_weights(self) -> None:
        pass

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
        """
    Args:
      src: Tensor, shape [batch_size,seq_len,dinput]
      src_mask: Tensor, shape [seq_len,seq_len]
    Returns:
      output dict with the following fields:
      stateprob: Tensor of shape [batch_size, seq_len, nstates] indicating the
      probability of each state
      perstate: Tensor of shape [batch_size, seq_len, d_output, nstates] where
      perstate[t,i,:,j] is the output for time t, example i, and state j.
    """

        # project input into d_model space, multiple by sqrt(d_model) for reasons?
        src = self.encoder(src) * math.sqrt(self.d_model)

        # add in the positional encoding of where in the sentence the words occur
        # it is weird to me that these are added, but I guess it would be almost
        # the same to have these be combined in a single linear layer
        src = self.pos_encoder(src)

        # main transformer layers
        transformer_output = self.transformer_encoder(src, mask=src_mask, is_causal=is_causal)

        output = {}
        # probability of each of the hidden states
        # batch_size x seq_len x nstates
        output['stateprob'] = self.state(transformer_output)

        # make sure that every state has some probability
        if self.training:
            output['stateprob'] = (output['stateprob'] + self.minstateprob) / (1 + self.nstates * self.minstateprob)

        # output given each hidden state
        # batch_size x seq_len x d_output x nstates
        output['perstate'] = self.decode(transformer_output).reshape(src.shape[:-1] + (self.d_output, self.nstates))

        return output

    def maxpred(self, pred):
        state = torch.argmax(pred['stateprob'], axis=-1)
        perstate = pred['perstate'].flatten(end_dim=1)
        out = perstate[torch.arange(perstate.shape[0], dtype=int), :, state.flatten()].reshape(
            pred['perstate'].shape[:-1])
        return out

    def randpred(self, pred):
        state = torch.multinomial(pred['stateprob'].flatten(end_dim=-2), 1)
        perstate = pred['perstate'].flatten(end_dim=1)
        out = perstate[torch.arange(perstate.shape[0], dtype=int), :, state.flatten()].reshape(
            pred['perstate'].shape[:-1])
        return out



class DictSum(torch.nn.Module):
    def __init__(self, moduledict):
        super().__init__()
        self.moduledict = moduledict

    def forward(self, x):
        out = 0.
        for k, v in x.items():
            out += self.moduledict[k](v)
        return out


# deprecated, here for backward compatibility
class TransformerMixedModel(TransformerModel):

    def __init__(self, d_input: int, d_output_continuous: int = 0,
                 d_output_discrete: int = 0, nbins: int = 0,
                 **kwargs):
        self.d_output_continuous = d_output_continuous
        self.d_output_discrete = d_output_discrete
        self.nbins = nbins
        d_output = d_output_continuous + d_output_discrete * nbins
        assert d_output > 0
        super().__init__(d_input, d_output, **kwargs)

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None, is_causal: bool = False) -> dict:
        output_all = super().forward(src, mask=mask, is_causal=is_causal)
        output_continuous = output_all[..., :self.d_output_continuous]
        output_discrete = output_all[..., self.d_output_continuous:].reshape(
            output_all.shape[:-1] + (self.d_output_discrete, self.nbins))
        return {'continuous': output_continuous, 'discrete': output_discrete}

    def output(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        output['discrete'] = torch.softmax(output['discrete'], dim=-1)
        return output







def compute_loss(model, dataloader, dataset, device, mask, criterion, config):
    is_causal = dataset.ismasked() == False
    if is_causal:
        mask = None

    model.eval()
    with torch.no_grad():
        all_loss = torch.zeros(len(dataloader), device=device)
        loss = torch.tensor(0.0).to(device)
        if dataset.discretize:
            loss_discrete = torch.tensor(0.0).to(device)
            loss_continuous = torch.tensor(0.0).to(device)
        nmask = 0
        for i, example in enumerate(dataloader):
            pred = model(example['input'].to(device=device), mask=mask, is_causal=is_causal)
            loss_curr, loss_discrete_curr, loss_continuous_curr = criterion_wrapper(example, pred, criterion, dataset,
                                                                                    config)

            if config['modeltype'] == 'mlm':
                nmask += torch.count_nonzero(example['mask'])
            else:
                nmask += example['input'].shape[0] * dataset.ntimepoints
            all_loss[i] = loss_curr
            loss += loss_curr
            if dataset.discretize:
                loss_discrete += loss_discrete_curr
                loss_continuous += loss_continuous_curr

        loss = loss.item() / nmask

        if dataset.discretize:
            loss_discrete = loss_discrete.item() / nmask
            loss_continuous = loss_continuous.item() / nmask
            return loss, loss_discrete, loss_continuous
        else:
            return loss







def stack_batch_list(allx, n=None):
    if len(allx) == 0:
        return []
    xv = torch.cat(allx[:n], dim=0)
    nan = torch.zeros((xv.shape[0], 1) + xv.shape[2:], dtype=xv.dtype)
    nan[:] = torch.nan
    xv = torch.cat((xv, nan), dim=1)
    xv = xv.flatten(0, 1)
    return xv





