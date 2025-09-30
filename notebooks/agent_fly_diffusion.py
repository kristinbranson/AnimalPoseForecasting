# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle

from apf.io import read_config
from apf.training import train, train_diffusion
from apf.utils import function_args_from_config
from apf.simulation import simulate
from apf.models import initialize_model

from flyllm.config import DEFAULTCONFIGFILE, posenames
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose

from experiments.flyllm import make_dataset, make_diffusion_dataset

# -

configfile = "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241125.json"
config = read_config(
    configfile,
    default_configfile=DEFAULTCONFIGFILE,
    posenames=posenames,
    featglobal=featglobal,
    get_sensory_feature_idx=get_sensory_feature_idx,
)

# +
# Try: oddroot, bin_epsilon very large (to get evenly sized bins)

# config['discretize_epsilon'] = np.ones(5) * 1000
# -
config['discreteidx'] = np.array([])

train_dataset, flyids, track, pose, velocity, sensory = make_diffusion_dataset(config, 'intrainfile', return_all=True, debug=False)

val_dataset = make_diffusion_dataset(config, 'invalfile', train_dataset, debug=False)



# +
# for i in range(5):
#     plt.figure(figsize=(10, 3))
#     # centers = train_dataset.labels['velocity'].operations[-1].operations[0].bin_centers[i]
#     edges = train_dataset.labels['velocity'].operations[-1].operations[0].bin_edges[i]
#     centers = (edges[:-1] + edges[1:])/2
#     counts = train_dataset.labels['velocity'].array[:, :, i*25:(i+1)*25].reshape([-1, 25]).sum(0)
#     counts_val = val_dataset.labels['velocity'].array[:, :, i*25:(i+1)*25].reshape([-1, 25]).sum(0)
#     plt.plot(centers[:-1], counts[:-1], '.')
#     plt.plot(centers[:-1], counts_val[:-1], '.')
#     plt.show()
# -

# Wrap into dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# +
# Initialize the model
device = torch.device(config['device'])

max_noise_T = 100
config['max_t'] = max_noise_T
config['nwave'] = 32
# -

model, criterion = initialize_model(config, train_dataset, device)

# +
# train_args['optimizer_args']['lr'] /= 2
# -

# Train the model
train_args = function_args_from_config(config, train)
train_args['num_train_epochs'] = 200
# train_args['optimizer_args']['lr'] *= 0.1
init_loss_epoch = {}
model, best_model, loss_epoch = train_diffusion(
    train_dataloader, val_dataloader, model, 
    loss_epoch=init_loss_epoch, max_noise_T=max_noise_T,
    **train_args
)

loss_epoch = init_loss_epoch

# +
# Plot the losses
idx = np.argmin(loss_epoch['val'])
print((idx, loss_epoch['val'][idx]))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(loss_epoch['train'])
plt.plot(loss_epoch['val'])
plt.plot(idx, loss_epoch['val'][idx], '.g')
plt.title('total loss')

plt.subplot(1, 3, 2)
plt.plot(loss_epoch['train_continuous'])
plt.plot(loss_epoch['val_continuous'])
plt.title('continuous loss')

plt.subplot(1, 3, 3)
# plt.plot(loss_epoch['train_discrete'])
# plt.plot(loss_epoch['val_discrete'])
# plt.title('discrete loss')
# plt.show()

# plt.figure()
plt.plot(loss_epoch['train'] - loss_epoch['train_continuous'])
plt.plot(loss_epoch['val'] - loss_epoch['val_continuous'])
plt.title('KLD')
plt.show()

# +
model_file = "/groups/branson/home/eyjolfsdottire/data/flyllm/model_diffusion_250528.pkl"
# pickle.dump(model, open(model_file, "wb"))
model = pickle.load(open(model_file, "rb"))

# torch.save(model, model_file)
# -

# from diffusers import DDPMScheduler
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
noise_scheduler = DDPMScheduler(num_train_timesteps=max_noise_T, prediction_type='sample',
                                beta_schedule="linear",
                                beta_start=0.0001, beta_end=0.02 * (1000 / max_noise_T), clip_sample=False)


# +
from apf.simulation import simulate_diffusion
import time

t0 = time.time()

agent_idx = 4
gt_track, pred_track = simulate_diffusion(
    dataset=train_dataset,
    model=model,
    track=track,
    pose=pose,
    identities=flyids,
    track_len=4000 + config['contextl'] + 1,
    burn_in=800,  # config['contextl'] * 2,
    max_contextl=config['contextl'],
    agent_idx=agent_idx,
    start_frame=512,
    max_noise_T=max_noise_T,
)

print(time.time() - t0)


# +
def plot_arena():
    ARENA_RADIUS_MM = 26.689
    n_pts = 1000
    theta = np.arange(0, np.pi * 2, np.pi * 2 / n_pts)
    x = np.cos(theta) * ARENA_RADIUS_MM
    y = np.sin(theta) * ARENA_RADIUS_MM
    plt.plot(x, y, '-', color=[.8, .8, .8])


plt.figure()
plot_arena()

# agent_idx = 4

last_frame = None
x, y = gt_track[agent_idx, :last_frame, :, 0].T
plt.plot(x, y, '.', markersize=3)
x, y = pred_track[agent_idx, :last_frame, :, 0].T
plt.plot(x, y, '.', markersize=1)
x, y = pred_track[agent_idx, 800:last_frame, :, 0].T
plt.plot(x, y, '.', markersize=1)
plt.axis('equal')
plt.show()
# -


agent_id = agent_idx
savevidfile = "/groups/branson/home/eyjolfsdottire/data/flyllm/animation_250528_diffusion_agent4_t100_t0_790.gif"
ani = animate_pose({'Pred': pred_track.T[:, :, 790:].copy(), 'True': gt_track.T[:, :, 790:].copy()}, focusflies=[agent_id],
                   savevidfile=savevidfile)#, contextl=config['contextl'])

model.encoder.encoder_dict
'velocity': Linear
'pose': ResNet1d
'wall_touch': ResNet1d
'other_flies_vision': ResNet2d
'other_flies_touch': ResNet1d


class TransformerModel(torch.nn.Module):

    def __init__(self, d_input: int, d_output: int,
                 d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
                 nlayers: int = 12, dropout: float = 0.1,
                 ntokens_per_timepoint: int = 1,
                 input_idx=None, input_szs=None, embedding_types=None, embedding_params=None,
                 d_output_discrete=None, nbins=None, variational=False, n_variational=None,
                 ):
        super().__init__()
        self.model_type = 'Transformer'

        self.is_mixed = nbins is not None
        if self.is_mixed:
            self.d_output_continuous = d_output
            self.d_output_discrete = d_output_discrete
            self.nbins = nbins
            d_output = self.d_output_continuous + self.d_output_discrete * self.nbins

        # frequency-based representation of word position with dropout
        self.pos_encoder = PositionalEncoding(d_model, dropout, ntokens_per_timepoint=ntokens_per_timepoint)

        # create self-attention + feedforward network module
        # d_model: number of input features
        # nhead: number of heads in the multiheadattention models
        # dhid: dimension of the feedforward network model
        # dropout: dropout value
        encoder_layers = myTransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)

        # stack of nlayers self-attention + feedforward layers
        # nlayers: number of sub-encoder layers in the encoder
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)

        # encoder and decoder are currently not tied together, but maybe they should be?
        # fully-connected layer from input size to d_model

        if input_idx is not None:
            self.encoder = ObsEmbedding(d_model=d_model, input_idx=input_idx, input_szs=input_szs,
                                        embedding_types=embedding_types, embedding_params=embedding_params)
        else:
            self.encoder = torch.nn.Linear(d_input, d_model)

        # fully-connected layer from d_model to input size
        if variational:
            if n_variational is None:
                # This way, the hidden state ends up being approximately 50% regular and 50% variational
                self.n_sub = d_model // 3
            else:
                self.n_sub = n_variational

            # self.decoder1 = torch.nn.Linear(d_model - self.n_sub, d_model // 2)
            # self.decoder2 = torch.nn.Linear(d_model // 2, d_output)
            self.decoder = torch.nn.Linear(d_model - self.n_sub, d_output)
        else:
            self.decoder = torch.nn.Linear(d_model, d_output)

        # store hyperparameters
        self.d_model = d_model
        self.variational = variational

        self.init_weights()

    def init_weights(self) -> None:
        pass

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
        """
  Args:
    src: Tensor, shape [seq_len,batch_size,dinput]
    src_mask: Tensor, shape [seq_len,seq_len]
  Returns:
    output Tensor of shape [seq_len, batch_size, ntoken]
  """

        # project input into d_model space, multiple by sqrt(d_model) for reasons?
        src = self.encoder(src) * math.sqrt(self.d_model)

        # add in the positional encoding of where in the sentence the words occur
        # it is weird to me that these are added, but I guess it would be almost
        # the same to have these be combined in a single linear layer
        src = self.pos_encoder(src)

        # main transformer layers
        hidden = self.transformer_encoder(src, mask=mask, is_causal=is_causal)

        output = {}
        if self.variational:
            # do the variational re-parameterization trick

            # mu = hidden[..., : self.d_model // 2]
            # logvar = hidden[..., self.d_model // 2:]

            mu = hidden[..., self.d_model - self.n_sub * 2: self.d_model - self.n_sub]
            logvar = hidden[..., self.d_model - self.n_sub:]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            # new_hidden = mu + eps * std
            new_hidden = torch.concatenate((hidden[..., :self.d_model - self.n_sub * 2], mu + eps * std), dim=-1)

            output['mu'] = mu
            output['logvar'] = logvar

            prediction = self.decoder(new_hidden)
            # prediction = self.decoder2(self.decoder1(new_hidden))
        else:
            # project back to d_input space
            prediction = self.decoder(hidden)

        if self.is_mixed:
            output['continuous'] = prediction[..., :self.d_output_continuous]
            output['discrete'] = prediction[..., self.d_output_continuous:].reshape(
                prediction.shape[:-1] + (self.d_output_discrete, self.nbins))
        else:
            output['continuous'] = prediction

        return output

    def set_need_weights(self, need_weights):
        for layer in self.transformer_encoder.layers:
            layer.set_need_weights(need_weights)

    def output(self, *args, **kwargs):

        output = self.forward(*args, **kwargs)
        if self.is_mixed:
            output['discrete'] = torch.softmax(output['discrete'], dim=-1)

        return output
