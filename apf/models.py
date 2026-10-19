import numpy as np
import math
import typing
import torch

lossfcn_discrete = torch.nn.CrossEntropyLoss()
lossfcn_continuous = torch.nn.L1Loss()


def unpack_input(input, featidx, sz, dim=-1):
    res = {}
    idx = [slice(None), ] * input.ndim
    sz0 = input.shape
    if dim < 0:
        dim = input.ndim + dim
    for k, v in featidx.items():
        idx[dim] = slice(v[0], v[1])
        newsz = sz0[:dim] + sz[k] + sz0[dim + 1:]
        res[k] = input[idx].reshape(newsz)

    return res


class PositionalEncoding(torch.nn.Module):
    """
        This module injects some information about the
        relative or absolute position of the tokens in the sequence. The
        positional encodings have the same dimension as the embeddings so that
        the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
        different frequencies.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000,
                 ntokens_per_timepoint: int = 1):
        super().__init__()

        # during training, randomly zero some of the inputs with probability p=dropout
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)

        # if many tokens per time point, then have a one-hot encoding of token type
        if ntokens_per_timepoint > 1:
            nwave = d_model - ntokens_per_timepoint
            for i in range(ntokens_per_timepoint):
                pe[0, :, nwave + i] = 2 * ((position[:, 0] % ntokens_per_timepoint) == i).to(float) - 1
        else:
            nwave = d_model

        # compute sine and cosine waves at different frequencies
        # pe[0,:,i] will have a different value for each word (or whatever)
        # will be sines for even i, cosines for odd i,
        # exponentially decreasing frequencies with i
        div_term = torch.exp(torch.arange(0, nwave, 2) * (-math.log(10000.0) / nwave))
        nsinwave = int(np.ceil(nwave / 2))
        ncoswave = nwave - nsinwave
        pe[0, :, 0:nwave:2] = torch.sin(position * div_term[:nsinwave])
        pe[0, :, 1:nwave:2] = torch.cos(position * div_term[:ncoswave])

        # buffers will be saved with model parameters, but are not model parameters
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
    Args:
      x: Tensor, shape [batch_size, seq_len, embedding_dim]
    """

        # add positional encoding
        x = x + self.pe[:, :x.size(1), :]

        # zero out a randomly selected subset of entries
        return self.dropout(x)


class myTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):

    def __init__(self, *args, need_weights=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_weights = need_weights

    def _sa_block(self, x: torch.Tensor,
                  attn_mask: typing.Optional[torch.Tensor],
                  key_padding_mask: typing.Optional[torch.Tensor],
                  is_causal: bool = False) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=self.need_weights,
                           is_causal=is_causal)[0]
        return self.dropout1(x)

    def set_need_weights(self, need_weights):
        self.need_weights = need_weights


class ResidualBlock1d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='same',
                 padding_mode='zeros'):
        super().__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv1 = torch.nn.Sequential(
            Conv1d_asym(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                        padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            Conv1d_asym(out_channels, out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding,
                        padding_mode=padding_mode, dilation=self.dilation),
            torch.nn.BatchNorm1d(out_channels)
        )
        if (in_channels != out_channels) or (self.stride > 1):
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                torch.nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    def compute_output_shape(self, din):
        if type(self.padding) == str:
            if self.padding == 'same':
                return (self.out_channels, din)
            elif self.padding == 'valid':
                padding = 0
            else:
                raise ValueError(f'Unknown padding type {self.padding}')
        if len(self.padding) == 1:
            padding = 2 * self.padding
        else:
            padding = np.sum(self.padding)
        dout1 = np.floor((din + padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        dout = (dout1 + padding - self.dilation * (self.kernel_size - 1) - 1) + 1
        sz = (self.out_channels, int(dout))
        return sz


class Conv1d_asym(torch.nn.Conv1d):
    def __init__(self, *args, padding='same', **kwargs):
        self.padding_off = [0, 0]
        padding_sym = padding
        if (type(padding) == tuple) or (type(padding) == list):
            padding_sym = int(np.max(padding))
            for j in range(2):
                self.padding_off[j] = padding_sym - padding[j]
        super().__init__(*args, padding=padding_sym, **kwargs)

    def asymmetric_crop(self, out):
        out = out[..., self.padding_off[0]:out.shape[-1] - self.padding_off[1]]
        return out

    def forward(self, x, *args, **kwargs):
        out = super().forward(x, *args, **kwargs)
        out = self.asymmetric_crop(out)
        return out


class Conv2d_asym(torch.nn.Conv2d):
    def __init__(self, *args, padding='same', **kwargs):
        self.padding_off = [[0, 0], [0, 0]]
        padding_sym = padding
        if (type(padding) == tuple) or (type(padding) == list):
            padding_sym = list(padding_sym)
            for i in range(2):
                if type(padding[i]) != int:
                    padding_sym[i] = int(np.max(padding[i]))
                    for j in range(2):
                        self.padding_off[i][j] = padding_sym[i] - padding[i][j]
            padding_sym = tuple(padding_sym)
        super().__init__(*args, padding=padding_sym, **kwargs)

    def asymmetric_crop(self, out):
        out = out[..., self.padding_off[0][0]:out.shape[-2] - self.padding_off[0][1],
              self.padding_off[1][0]:out.shape[-1] - self.padding_off[1][1]]
        return out

    def forward(self, x, *args, **kwargs):
        out = super().forward(x, *args, **kwargs)
        out = self.asymmetric_crop(out)
        return out


class ResidualBlock2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same',
                 padding_mode='zeros'):
        super().__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv1 = torch.nn.Sequential(
            Conv2d_asym(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                        padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            Conv2d_asym(out_channels, out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding,
                        padding_mode=padding_mode, dilation=self.dilation),
            torch.nn.BatchNorm2d(out_channels)
        )
        if (in_channels != out_channels) or (np.any(np.array(self.stride) > 1)):
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    def compute_output_shape(self, din):
        if type(self.padding) == str:
            if self.padding == 'same':
                return (self.out_channels,) + din
            elif self.padding == 'valid':
                padding = (0, 0)
            else:
                raise ValueError(f'Unknown padding type {self.padding}')

        if len(self.padding) == 1:
            padding = [self.padding, self.padding]
        else:
            padding = self.padding
        padding = np.array(padding)
        paddingsum = np.zeros(2, dtype=int)
        for i in range(2):
            if len(padding[i]) == 1:
                paddingsum[i] = 2 * padding[i]
            else:
                paddingsum[i] = int(np.sum(padding[i]))
        dout1 = np.floor(
            (np.array(din) + paddingsum - np.array(self.dilation) * (np.array(self.kernel_size) - 1) - 1) / np.array(
                self.stride) + 1).astype(int)
        dout = ((dout1 + paddingsum - np.array(self.dilation) * (np.array(self.kernel_size) - 1) - 1) + 1).astype(int)
        sz = (self.out_channels,) + tuple(dout)
        return sz


class ResNet1d(torch.nn.Module):
    def __init__(self, channels, d_output, d_input=None, no_input_channels=False, single_output=False, transpose=False,
                 **kwargs):
        super().__init__()
        self.channels = channels
        self.d_output = d_output
        self.d_input = d_input
        self.no_input_channels = no_input_channels
        self.transpose = transpose
        self.single_output = single_output

        if no_input_channels:
            assert channels[0] == 1

        nblocks = len(channels) - 1
        self.layers = torch.nn.ModuleList()
        sz = (channels[0], d_input)
        for i in range(nblocks):
            self.layers.append(ResidualBlock1d(channels[i], channels[i + 1], **kwargs))
            if d_input is not None:
                sz = self.layers[-1].compute_output_shape(sz[-1])
        if single_output:
            if d_input is None:
                self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
                self.fc = torch.nn.Linear(channels[-1], d_output)
            else:
                self.avg_pool = None
                self.fc = torch.nn.Linear(int(np.prod(sz)), d_output)
        else:
            self.avg_pool = None
            self.fc = torch.nn.Conv1d(channels[-1], d_output, 1)

    def forward(self, x):

        if self.transpose and not self.no_input_channels:
            x = x.transpose(-1, -2)

        if self.no_input_channels:
            dim = -1
        else:
            dim = -2

        sz0 = x.shape
        d_input = sz0[-1]

        if self.single_output and (self.d_input is not None):
            assert d_input == self.d_input

        sz = (int(np.prod(sz0[:dim])), self.channels[0], d_input)
        x = x.reshape(sz)

        for layer in self.layers:
            x = layer(x)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        if self.single_output:
            x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.single_output:
            dimout = -1
        else:
            dimout = -2
        x = x.reshape(sz0[:dim] + x.shape[dimout:])

        if self.transpose and not self.single_output:
            x = x.transpose(-1, -2)

        return x


class ResNet2d(torch.nn.Module):
    def __init__(self, channels, d_output, d_input=None, no_input_channels=False, single_output=False, transpose=False,
                 **kwargs):
        super().__init__()
        self.channels = channels
        self.d_output = d_output
        self.d_input = d_input
        self.no_input_channels = no_input_channels
        self.transpose = transpose

        if no_input_channels:
            assert channels[0] == 1

        nblocks = len(channels) - 1
        self.layers = torch.nn.ModuleList()
        is_d_input = [False, False]
        if d_input is not None:
            if type(d_input) == int:
                d_input(0, d_input)
            elif len(d_input) < 2:
                d_input = (0,) * (2 - len(d_input)) + d_input
            is_d_input = [d != 0 for d in d_input]
            sz = (channels[0],) + d_input
        for i in range(nblocks):
            self.layers.append(ResidualBlock2d(channels[i], channels[i + 1], **kwargs))
            if d_input is not None:
                sz = self.layers[-1].compute_output_shape(sz[1:])
        self.collapse_dim = []
        if single_output:
            if d_input is None:
                self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
                self.fc = torch.nn.Linear(channels[-1], d_output)
                self.collapse_dim = [-2, -1]
            else:
                self.avg_pool = None
                k = [1, 1]
                for i in range(2):
                    if is_d_input[i]:
                        k[i] = sz[i + 1]
                        self.collapse_dim.append(-2 + i)
                self.fc = torch.nn.Conv2d(channels[-1], d_output, k, padding='valid')
        else:
            self.avg_pool = None
            self.fc = torch.nn.Conv2d(channels[-1], d_output, 1)

    def forward(self, x):

        if self.transpose and not self.no_input_channels:
            x = torch.movedim(x, -1, -3)

        if self.no_input_channels:
            dim = -2
        else:
            dim = -3

        sz0 = x.shape
        d_input = sz0[-2:]

        sz = (int(np.prod(sz0[:dim])), self.channels[0]) + d_input
        x = x.reshape(sz)

        for layer in self.layers:
            x = layer(x)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            dimout = -1
        else:
            dimout = -3
        x = self.fc(x)
        x = x.reshape(sz0[:dim] + x.shape[dimout:])
        dim_channel = len(sz0[:dim])
        x = torch.squeeze(x, self.collapse_dim)

        if self.transpose:
            x = torch.movedim(x, dim_channel, -1)

        return x


class ObsEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, input_idx, input_szs, embedding_types, embedding_params):

        super().__init__()

        assert input_idx is not None
        assert input_szs is not None
        assert embedding_types is not None
        assert embedding_params is not None

        self.input_idx = input_idx
        self.input_szs = input_szs

        self.encoder_dict = torch.nn.ModuleDict()
        for k in input_idx.keys():
            emb = embedding_types.get(k, 'fc')
            params = embedding_params.get(k, {})
            szcurr = input_szs[k]
            if emb == 'conv1d_feat':
                if len(szcurr) < 2:
                    input_channels = 1
                else:
                    input_channels = szcurr[1]
                channels = [input_channels, ] + params['channels']
                params = {k1: v for k1, v in params.items() if k1 != 'channels'}
                encodercurr = ResNet1d(channels, d_model, d_input=szcurr[0], no_input_channels=True, single_output=True,
                                       transpose=False, **params)
            elif emb == 'fc':
                encodercurr = torch.nn.Linear(szcurr[0], d_model)
            elif emb == 'conv1d_time':
                assert (len(szcurr) == 1)
                input_channels = szcurr[0]
                channels = [input_channels, ] + params['channels']
                params = {k1: v for k1, v in params.items() if k1 != 'channels'}
                encodercurr = ResNet1d(channels, d_model, no_input_channels=False, single_output=False, transpose=True,
                                       **params)
            elif emb == 'conv2d':
                assert (len(szcurr) <= 2)
                if len(szcurr) > 1:
                    input_channels = szcurr[1]
                    no_input_channels = False
                else:
                    input_channels = 1
                    no_input_channels = True
                channels = [input_channels, ] + params['channels']
                params = {k1: v for k1, v in params.items() if k1 != 'channels'}
                encodercurr = ResNet2d(channels, d_model, no_input_channels=no_input_channels, d_input=szcurr,
                                       single_output=True, transpose=True, **params)
            else:
                # consider adding graph networks
                raise ValueError(f'Unknown embedding type {emb}')
            self.encoder_dict[k] = encodercurr

    def forward(self, src):
        src = unpack_input(src, self.input_idx, self.input_szs)
        out = 0.
        for k, v in src.items():
            out += self.encoder_dict[k](v)
        return out


class VAEDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, hidden):
        pass


from diffusers import DDPMScheduler
class SinusoidalEmbeddings(torch.nn.Module):
    def __init__(self, n_timesteps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(n_timesteps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(n_timesteps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, t):
        return self.embeddings[t]


class TimeEmbeddedMLP(torch.nn.Module):
    def __init__(self, d_input, d_output, channels, n_timesteps, embed_dim=256):
        super(TimeEmbeddedMLP, self).__init__()

        # self.embed_weight = embed_weight
        self.t_embeddings = SinusoidalEmbeddings(n_timesteps=n_timesteps, embed_dim=embed_dim)
        embed_layers = [torch.nn.Linear(embed_dim, d_input)] + [torch.nn.Linear(embed_dim, channel)
                                                                for channel in channels]
        self.embed_layers = torch.nn.ModuleList(embed_layers)

        if len(channels) == 0:
            layers = [torch.nn.Linear(d_input, d_output)]
        else:
            layers = [torch.nn.Linear(d_input, channels[0])]
            for i in range(1, len(channels)):
                layers.append(torch.nn.Linear(channels[i - 1], channels[i]))
            layers.append(torch.nn.Linear(channels[-1], d_output))
        self.layers = torch.nn.ModuleList(layers)

        print((len(layers), len(embed_layers)))

    def forward(self, x, t):
        # Embed t into the same dimension as each layer vector, so that it can be added to each layer
        # x can be of shape (n_seq_len, batch_size, d_input)
        # t should be a scalar or a vector

        embed = self.t_embeddings(t)

        for i, layer in enumerate(self.layers[:-1]):
            x = x + self.embed_layers[i](embed)
            # TODO: Try swapping out for relu
            x = torch.nn.functional.tanh(layer(x))

            # embed = self.t_embeddings(t)[..., :x.shape[-1]]
            # x = x + embed * self.embed_weight
        x = x + self.embed_layers[-1](embed)
        output = self.layers[-1](x)
        return output


class DiffusionDecoder(torch.nn.Module):
    def __init__(self, d_model, d_output, n_timesteps, decoder_channels, time_embed_dim):
        super().__init__()
        # diffusion noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=n_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            thresholding=False,
            prediction_type='sample',
        )
        # MLP from transformers hidden state and noisy output to output
        self.decoder = TimeEmbeddedMLP(
            d_input=d_model+d_output, # TODO: modify this, first embed the noisy gt into d_model, then add
            d_output=d_output,
            channels=decoder_channels,
            n_timesteps=n_timesteps,
            embed_dim=time_embed_dim,
        )

    def forward(self, hidden, gt=None):
        # generate random noise
        seq_len, batch_size = hidden.shape[:2]
        noise = torch.randn((seq_len, batch_size, self.d_output)).to(hidden.device)
        if gt is not None:
            # Training phase
            # generate random t and add noise to the gt output
            t = torch.randint(0, self.n_timesteps, (1, ))
            # TODO: Embed ground truth into d_model, add the noise to gt_embed
            noisy_gt = self.noise_scheduler.add_noise(gt, noise, t)
            input = torch.concatenate([hidden, noisy_gt], dim=-1)
            output = self.decoder(input, t)
        else:
            # Inference phase
            # run the full diffusion denoising process
            x = noise
            input = torch.concatenate([hidden, x], dim=-1)
            for i, t in enumerate(self.noise_scheduler.timesteps):
                residual = self.decoder(input, torch.as_tensor([t]).to(hidden.device))
                out = self.noise_scheduler.step(residual, t, x)
                x = out.prev_sample
                input[:, :, -self.d_output:] = x
            output = out.pred_original_sample

        return output


from apf.data import weighted_sample

class FlattenedDecoderSimple(torch.nn.Module):
    # This assumes all outputs are discretized
    # TODO: Remove this assumption. If it is a mix, first predict the binned features, then all the rest.
    #
    # During training, we need the ground truth bins
    #  - At first, we condition on the gt bin of features when predicting the next feature
    #  - Gradually we increase the likelihood of sampling from the bin or taking the argmax
    #    - if the sampling is different from ground truth, we are still trying to force the network
    #      to output the next feature to correspond to the sampled feature,
    #      that doesn't seem like a good strategy...
    #    - I think I'll just skip this to start with, I think it will work without it.
    # flattened_params:
    # - feature order: in which order to predict the output
    #
    # How it works
    # - from the hidden state (d_model), predict the first feature (n_bins).
    # - embed the true bin on the first feature into a d_model and add it to the hidden state
    #   (this is a learned n_bins to d_model mapping)
    # - combine h and this embedding: either sum or do it in a multiplicative manner
    # - this means we have to learn n_feat x (n_bins x d_model) * 2 models (encode and decode)

    # TODO: consider non-categorical bin encoding to preserve correlations between bins
    #       e.g. use the scalar value, bin center, or use the enumeration of the chosen bin
    #       (some work says this will be implicitly learned given enough data)

    # TODO: Instead of adding the encoded bin vector I could provide n_feat x n_bin_embed with a mask
    #  as extra input to the predictor. With that approach we could randomize the order of feature sampling.
    #  (I'm not sure randomizing the order would be better, but maybe better to keep the state.
    #  We would end up learning n_feat times more weights... that's maybe not too bad

    def __init__(self, d_model, n_feat, n_bins, feature_order=None):
        super().__init__()
        self.d_model = d_model
        self.n_feat = n_feat
        self.n_bins = n_bins

        self.encoders = torch.nn.ModuleList([torch.nn.Linear(n_bins, d_model) for _ in range(n_feat-1)])
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(d_model, n_bins) for _ in range(n_feat)])
        if feature_order is None:
            feature_order = np.arange(n_feat)
        self.feature_order = feature_order

    def forward(self, hidden, gt=None):
        """
        Args:
            hidden: shape (batch_size, seq_len, n_model)
            gt: shape (batch_size, seq_len, n_feat * n_bins)
        """
        hidden_ = hidden
        batch_size, seq_len = hidden.shape[:2]
        out = torch.zeros((batch_size, seq_len, self.n_feat, self.n_bins)).to(hidden.device)
        if gt is not None:
            gt_ = gt.reshape(out.shape)
        for k, i in enumerate(self.feature_order):
            out[:, :, i] = self.decoders[k](hidden_)
            if gt is not None:
                out_i = gt_[:, :, i]
            else:
                # shape of out[i] is (batch_size, seq_len, n_bins)
                # shape of sampels is (batch_size, seq_len), with entries corresonding to the selected bin
                samples = weighted_sample(torch.softmax(out[:, :, i], dim=-1), 1)[0]
                cols, rows = np.indices(samples.shape)
                out_i = torch.zeros_like(out[:, :, i])
                out_i[cols, rows, samples] = 1
                out[:, :, i] = out_i

            if k < self.n_feat-1:
                hidden_ = hidden_ + self.encoders[k](out_i)
        return out.reshape((batch_size, seq_len, -1))


class FlattenedDecoder(torch.nn.Module):
    # This decoder uses a transfomer encoder to decode the hidden state, producing outputs in a sequence
    #
    # For now, assume that features are inferred in their original order (this can later be changed
    # by modifying the mask (or re-ordering the features)
    #
    # This assumes all outputs are discretized
    # TODO: Remove this assumption. If it is a mix, first predict the binned features, then all the rest.
    #
    # During training, we need the ground truth bins
    #  - At first, we condition on the gt bin of features when predicting the next feature
    #  - Gradually we increase the likelihood of sampling from the bin or taking the argmax
    #    - if the sampling is different from ground truth, we are still trying to force the network
    #      to output the next feature to correspond to the sampled feature,
    #      that doesn't seem like a good strategy...
    #    - I think I'll just skip this to start with, I think it will work without it.
    # flattened_params:
    # - feature order: in which order to predict the output
    #
    # How it works
    # - from the hidden state (d_model), predict the first feature (n_bins).
    # - embed the true bin on the first feature into a d_model and add it to the hidden state
    #   (this is a learned n_bins to d_model mapping)
    # - combine h and this embedding: either sum or do it in a multiplicative manner
    # - this means we have to learn n_feat x (n_bins x d_model) * 2 models (encode and decode)

    # TODO: consider non-categorical bin encoding to preserve correlations between bins
    #       e.g. use the scalar value, bin center, or use the enumeration of the chosen bin
    #       (some work says this will be implicitly learned given enough data)

    # TODO: Instead of adding the encoded bin vector I could provide n_feat x n_bin_embed with a mask
    #  as extra input to the predictor. With that approach we could randomize the order of feature sampling.
    #  (I'm not sure randomizing the order would be better, but maybe better to keep the state.
    #  We would end up learning n_feat times more weights... that's maybe not too bad

    def __init__(self, d_model, n_feat, n_bins, nhead, nlayers, d_hid, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_feat = n_feat
        self.n_bins = n_bins

        self.value_proj = torch.nn.Linear(n_bins, d_model)          # embeds previous feature's value
        self.feat_pos = torch.nn.Embedding(n_feat, d_model)     # which feature we're predicting

        # Note, Claude suggested n_layers=2 and d_hid=4 * d_model
        encoder_layers = myTransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.out = torch.nn.Linear(d_model, n_bins)

        mask = torch.nn.Transformer.generate_square_subsequent_mask(self.n_feat)
        self.register_buffer("mask", mask, persistent=False)


    def sample(self, hidden):
        """Given x (B, T, F*n_bins), sample the next step's features one at a time."""
        h = hidden[:, -1:]                              # (B, 1, d) last time step
        y = torch.zeros(hidden.shape[0], 1, self.n_feat, self.n_bins, device=hidden.device)
        for j in range(self.n_feat):
            logits = self.decode(h, y)[:, 0, j]             # (B, n_bins)
            idx = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            y[:, 0, j] = torch.nn.functional.one_hot(idx, self.n_bins).float()
        return y.reshape(h.shape[0], 1, -1)

    def decode(self, hidden, gt):
        B, T, d = hidden.shape
        y = gt.reshape(B, T, self.n_feat, self.n_bins)
        # shift right along features: feature j receives the value of feature j-1 (zeros for j=0)
        y_prev = torch.cat([torch.zeros_like(y[:, :, :1]), y[:, :, :-1]], dim=2)
        z = hidden.unsqueeze(2) + self.value_proj(y_prev) + self.feat_pos.weight  # (B, T, F, d)
        z = z.reshape(B * T, self.n_feat, d)
        z = self.transformer_encoder(z, mask=self.mask, is_causal=True)
        return self.out(z).reshape(B, T, self.n_feat, self.n_bins)

    def forward(self, hidden, gt=None):
        """
        Args:
            hidden: shape (batch_size, seq_len, n_model)
            gt: shape (batch_size, seq_len, n_feat * n_bins)
        """
        if gt is None:
            return self.sample(hidden)
        else:
            out = self.decode(hidden, gt)
            B, T = out.shape[:2]
            return out.reshape(B, T, -1)


class HybridFeatureDecoderPlus(torch.nn.Module):
    """Autoregressive decoder over discretized features within one time step.

    logits_k = direct_k(hidden) + residual_k(transformer(features < k, hidden))

    - direct path: per-feature linear head straight from `hidden`, as in your model,
      so temporal context always contributes to every feature with full weight.
    - small transformer (width d_dec, not d_model) over the feature axis with a
      causal mask, so feature k can condition nonlinearly on features < k.
    - separate value embedding and separate output head for each feature.
    - residual heads are zero-initialized: training starts as a model that predicts
      every feature from `hidden` alone, and the transformer learns corrections.

    All per-feature parameters are indexed by *position in feature_order*.
    """

    def __init__(self, d_model, n_feat, n_bins, d_dec=256, nhead=4, nlayers=2,
                 d_hid=512, dropout=0.1, feature_order=None):
        super().__init__()
        self.n_feat, self.n_bins, self.d_dec = n_feat, n_bins, d_dec

        if feature_order is None:
            feature_order = list(range(n_feat))
        order = torch.as_tensor(list(feature_order), dtype=torch.long)
        assert sorted(order.tolist()) == list(range(n_feat)), "feature_order must be a permutation"
        self.register_buffer("order", order, persistent=False)
        self.register_buffer("inv_order", torch.argsort(order), persistent=False)

        # Direct path from the temporal context to every feature's logits
        self.direct_head = torch.nn.Linear(d_model, n_feat * n_bins)

        # Context projected down once per (batch, time) step, then added to every feature token
        self.ctx_proj = torch.nn.Linear(d_model, d_dec)

        # Separate encoder per feature: value_emb[k] embeds the chosen bin of feature k
        # (same as a bias-free Linear(n_bins, d_dec) per feature; works for soft labels too).
        # The last feature is never fed forward, so it needs no embedding.
        self.value_emb = torch.nn.Parameter(torch.randn(n_feat - 1, n_bins, d_dec) * 0.02)
        self.bos = torch.nn.Parameter(torch.zeros(d_dec))                      # "nothing chosen yet"
        self.feat_pos = torch.nn.Parameter(torch.randn(n_feat, d_dec) * 0.02)  # which feature is predicted

        layer = torch.nn.TransformerEncoderLayer(d_dec, nhead, d_hid, dropout,
                                           batch_first=True, norm_first=True)
        self.transformer = torch.nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        self.final_norm = torch.nn.LayerNorm(d_dec)

        # Separate output head per feature, zero-initialized (residual on top of direct path)
        self.head_w = torch.nn.Parameter(torch.zeros(n_feat, d_dec, n_bins))
        self.head_b = torch.nn.Parameter(torch.zeros(n_feat, n_bins))

        mask = torch.nn.Transformer.generate_square_subsequent_mask(n_feat)
        self.register_buffer("mask", mask, persistent=False)

    def _context(self, hidden):
        """Parts that depend only on `hidden`; computed once, reused in every sampling step."""
        B, T, _ = hidden.shape
        direct = self.direct_head(hidden).view(B, T, self.n_feat, self.n_bins)[:, :, self.order]
        ctx = self.ctx_proj(hidden).unsqueeze(2)                         # (B, T, 1, d_dec)
        return direct, ctx

    def _logits(self, direct, ctx, y):
        """y: (B, T, F, n_bins) one-hot (or soft) values in feature_order. Returns logits in order space."""
        B, T = y.shape[:2]
        emb = torch.einsum("btfk,fkd->btfd", y[:, :, :-1], self.value_emb)      # (B, T, F-1, d)
        prev = torch.cat([self.bos.expand(B, T, 1, self.d_dec), emb], dim=2)    # shift right
        z = prev + self.feat_pos + ctx
        z = self.transformer(z.reshape(B * T, self.n_feat, self.d_dec),
                             mask=self.mask, is_causal=True)
        z = self.final_norm(z).view(B, T, self.n_feat, self.d_dec)
        resid = torch.einsum("btfd,fdk->btfk", z, self.head_w) + self.head_b
        return direct + resid

    def forward(self, hidden, gt=None, temperature=1.0, return_samples=False):
        """
        Args:
            hidden: (B, T, d_model)
            gt: (B, T, n_feat * n_bins) one-hot targets for teacher forcing, or None to sample.
            temperature: sampling temperature; 0 means argmax.
            return_samples: when sampling, also return the sampled one-hot values.
        Returns:
            logits (B, T, n_feat * n_bins) in the original feature order
            (and samples of the same shape if return_samples=True).
        """
        B, T, _ = hidden.shape
        if gt is not None:
            y = gt.reshape(B, T, self.n_feat, self.n_bins)[:, :, self.order]
            logits = self._logits(*self._context(hidden), y)
            return logits[:, :, self.inv_order].reshape(B, T, -1)

        logits, y = self._sample(hidden, temperature)
        logits = logits[:, :, self.inv_order].reshape(B, T, -1)
        if return_samples:
            return logits, y[:, :, self.inv_order].reshape(B, T, -1)
        return logits

    @torch.no_grad()
    def _sample(self, hidden, temperature):
        """Sample features one at a time. For rollouts, pass hidden[:, -1:] to sample only the next step."""
        B, T, _ = hidden.shape
        direct, ctx = self._context(hidden)
        y = hidden.new_zeros(B, T, self.n_feat, self.n_bins)
        logits = hidden.new_zeros(B, T, self.n_feat, self.n_bins)
        for k in range(self.n_feat):
            lk = self._logits(direct, ctx, y)[:, :, k]                   # (B, T, n_bins)
            logits[:, :, k] = lk
            if temperature == 0:
                idx = lk.argmax(-1)
            else:
                probs = (lk.float() / temperature).softmax(-1).reshape(-1, self.n_bins)
                idx = torch.multinomial(probs, 1).view(B, T)
            y[:, :, k] = torch.nn.functional.one_hot(idx, self.n_bins).to(y.dtype)
        return logits, y


class HybridFeatureDecoder(torch.nn.Module):
    """Autoregressive decoder over the discretized features of one time step.

    For each (batch, time) step, a small transformer runs over a sequence of n_feat tokens.
    Token k is the sum of:
      - the embedded value of feature k-1 (each feature has its own encoder; token 0 gets zeros)
      - feat_pos[k]: which feature this token predicts
      - ctx: the encoder's hidden state, projected from d_model to d_dec
    A causal mask lets token k see tokens 0..k only, so the prediction for feature k
    depends on hidden and on features 0..k-1. Each feature has its own output head.

    forward(hidden, gt)  -> logits (B, T, n_feat * n_bins)    (teacher forcing, for training)
    forward(hidden)      -> samples (B, T, n_feat * n_bins)   (one-hot, features sampled in order)
    """

    def __init__(self, d_model, n_feat, n_bins, d_dec=256, nhead=4, nlayers=2, d_hid=512, dropout=0.1):
        super().__init__()
        self.n_feat, self.n_bins, self.d_dec = n_feat, n_bins, d_dec

        self.ctx_proj = torch.nn.Linear(d_model, d_dec)
        # One encoder per feature whose value is fed forward (the last feature never is).
        # # No bias: a constant per position is already provided by feat_pos.
        # self.value_encoders = torch.nn.ModuleList([torch.nn.Linear(n_bins, d_dec, bias=False) for _ in range(n_feat - 1)])
        self.value_encoders = torch.nn.ModuleList([torch.nn.Linear(n_bins, d_dec) for _ in range(n_feat - 1)])

        # # Feature positional encoding (do we really need this, couldn't this be implicitly learned by the value_encoders?)
        # self.feat_pos = torch.nn.Embedding(n_feat, d_dec)

        layer = torch.nn.TransformerEncoderLayer(d_dec, nhead, d_hid, dropout, batch_first=True, norm_first=True)
        self.transformer = torch.nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        self.final_norm = torch.nn.LayerNorm(d_dec)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(d_dec, n_bins) for _ in range(n_feat)])

        mask = torch.nn.Transformer.generate_square_subsequent_mask(n_feat)
        self.register_buffer("mask", mask, persistent=False)

    def decode(self, ctx, y):
        """ctx: (B, T, d_dec), y: (B, T, n_feat, n_bins) one-hot. Returns logits (B, T, n_feat, n_bins)."""
        B, T = y.shape[:2]
        # Shift right: token k carries the value of feature k-1; token 0 carries nothing.
        prev = [torch.zeros(B, T, self.d_dec, device=y.device, dtype=ctx.dtype)]
        prev += [enc(y[:, :, k]) for k, enc in enumerate(self.value_encoders)]
        # z = torch.stack(prev, dim=2) + self.feat_pos.weight + ctx.unsqueeze(2)    # (B, T, n_feat, d_dec)
        z = torch.stack(prev, dim=2) + ctx.unsqueeze(2)

        z = self.transformer(z.reshape(B * T, self.n_feat, self.d_dec), mask=self.mask, is_causal=True)
        z = self.final_norm(z).view(B, T, self.n_feat, self.d_dec)
        return torch.stack([head(z[:, :, k]) for k, head in enumerate(self.heads)], dim=2)

    def forward(self, hidden, gt=None, temperature=1.0):
        """
        Args:
            hidden: (B, T, d_model) from the temporal encoder.
            gt: (B, T, n_feat * n_bins) one-hot targets, or None to sample.
            temperature: sampling temperature; 0 means argmax.
        """
        B, T, _ = hidden.shape
        ctx = self.ctx_proj(hidden)
        if gt is not None:
            y = gt.reshape(B, T, self.n_feat, self.n_bins)
            return self.decode(ctx, y).reshape(B, T, -1)
        return self.sample(ctx, temperature).reshape(B, T, -1)

    @torch.no_grad()
    def sample(self, ctx, temperature):
        """Sample features one at a time. For rollouts, pass hidden[:, -1:] to sample only the next step."""
        B, T, _ = ctx.shape
        y = ctx.new_zeros(B, T, self.n_feat, self.n_bins)
        for k in range(self.n_feat):
            logits = self.decode(ctx, y)[:, :, k]                 # (B, T, n_bins), uses features < k only
            if temperature == 0:
                idx = logits.argmax(-1)
            else:
                probs = (logits.float() / temperature).softmax(-1)
                idx = torch.multinomial(probs.reshape(-1, self.n_bins), 1).view(B, T)
            y[:, :, k] = torch.nn.functional.one_hot(idx, self.n_bins).to(y.dtype)
        return y


class TransformerModel(torch.nn.Module):

    def __init__(self, d_input: int, d_output: int,
                 d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
                 nlayers: int = 12, dropout: float = 0.1,
                 ntokens_per_timepoint: int = 1,
                 input_idx=None, input_szs=None, embedding_types=None, embedding_params=None,
                 d_output_discrete=None, nbins=None,
                 dataset_params=None, config=None,
                 decoder_type='linear', # 'diffusion', 'flattened'
                 nlayers_decoder=2,
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

        if decoder_type == 'linear':
            # fully-connected layer from d_model to input size
            self.decoder = torch.nn.Linear(d_model, d_output)
        elif decoder_type == 'diffusion':
            # This assumes all outputs are continuous
            # .forward requires ground truth output at train time
            n_timesteps=100,
            decoder_channels=(512, 256, 128, 64),
            time_embed_dim=256
            self.decoder = DiffusionDecoder(d_model, d_output, n_timesteps, decoder_channels, time_embed_dim)
        elif decoder_type == 'vae':
            pass
            # self.decoder = VAEDecoder(d_model, d_output)
        elif decoder_type == 'flattened':
            # This assumes all outputs are discretized
            # .forward requires ground truth output at train time
            # self.decoder = FlattenedDecoder(d_model, d_output_discrete, nbins)
            # self.decoder = FlattenedDecoder(d_model, d_output_discrete, nbins, nhead, nlayers_decoder, d_hid, dropout)
            self.decoder = HybridFeatureDecoder(d_model, d_output_discrete, nbins,
                                                d_dec=d_model, #d_dec=256,
                                                nhead=nhead, nlayers=nlayers_decoder, d_hid=d_hid, dropout=0.1)
        else:
            assert False, f"Decoder type '{decoder_type}' is not supported"

        # store hyperparameters
        self.d_model = d_model

        # dataset parameters
        self.dataset_params = dataset_params

        # config, primarily for book-keeping
        self.config = config

        self.init_weights()

    def init_weights(self) -> None:
        pass

    def forward(
            self, src: torch.Tensor, mask: torch.Tensor = None, is_causal: bool = False,
            return_hidden: bool = False, gt: torch.Tensor = None,
    ) -> torch.Tensor:
        """
    Args:
      src: Tensor, shape [batch_size, seq_len, dinput]
      src_mask: Tensor, shape [seq_len, seq_len]
      return_hidden: Whether to include hidden states in the output. Hidden states are stored as a list of float arrays
        of shape (batch_size, seq_len, d_model).
    Returns:
      output Tensor of shape [batch_size, seq_len, ntoken]
    """

        # project input into d_model space, multiple by sqrt(d_model) for reasons?
        src = self.encoder(src) * math.sqrt(self.d_model)

        # add in the positional encoding of where in the sentence the words occur
        # it is weird to me that these are added, but I guess it would be almost
        # the same to have these be combined in a single linear layer
        src = self.pos_encoder(src)

        # main transformer layers
        if return_hidden:
            hidden_states = []
            output = src
            for layer in self.transformer_encoder.layers:
                output = layer(
                    output,
                    src_mask=mask,
                    is_causal=is_causal,
                )
                hidden_states.append(output.detach().cpu().numpy())
        else:
            output = self.transformer_encoder(src, mask=mask, is_causal=is_causal)

        # project back to d_output space
        if gt is not None:
            output = self.decoder(output, gt=gt)
        else:
            output = self.decoder(output)

        if self.is_mixed:
            output_continuous = output[..., :self.d_output_continuous]
            output_discrete = output[..., self.d_output_continuous:].reshape(
                output.shape[:-1] + (self.d_output_discrete, self.nbins))
            output = {'continuous': output_continuous, 'discrete': output_discrete}
        else:
            output = {'continuous': output}

        if return_hidden:
            output['hidden'] = hidden_states

        return output

    def set_need_weights(self, need_weights):
        for layer in self.transformer_encoder.layers:
            layer.set_need_weights(need_weights)

    def output(self, *args, **kwargs):

        output = self.forward(*args, **kwargs)
        if self.is_mixed and not (isinstance(self.decoder, FlattenedDecoder) or isinstance(self.decoder, HybridFeatureDecoder)):
            output['discrete'] = torch.softmax(output['discrete'], dim=-1)

        return output


class MLP(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size=1):
        super(MLP, self).__init__()

        if len(layer_sizes) == 0:
            layers = [torch.nn.Linear(input_size, output_size)]
        else:
            layers = [torch.nn.Linear(input_size, layer_sizes[0])]
            for i in range(len(layer_sizes) - 1):
                layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.Linear(layer_sizes[-1], output_size))

        self.layers = torch.nn.ModuleList(layers)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        return self.sigmoid(self.layers[-1](x))


## Support functions
def initialize_model(config, train_dataset, device):
    # architecture arguments
    MODEL_ARGS = {
        'd_model': config['d_model'],
        'nhead': config['nhead'],
        'd_hid': config['d_hid'],
        'nlayers': config['nlayers'],
        'dropout': config['dropout'],
        'config': config
    }
    if 'decoder_type' in config:
        MODEL_ARGS['decoder_type'] = config['decoder_type']
    if 'dataset_params' in config:
        MODEL_ARGS['dataset_params'] = config['dataset_params']
    elif hasattr(train_dataset, 'get_params'):
        MODEL_ARGS['dataset_params'] = train_dataset.get_params()

    if config['modelstatetype'] is not None:
        MODEL_ARGS['nstates'] = config['nstates']
        assert config['obs_embedding'] == False, 'Not implemented'
        assert train_dataset.flatten == False, 'Not implemented'
    if config['modelstatetype'] == 'prob':
        MODEL_ARGS['minstateprob'] = config['minstateprob']

    if config['obs_embedding']:
        MODEL_ARGS['input_idx'], MODEL_ARGS['input_szs'] = train_dataset.get_input_shapes()
        MODEL_ARGS['embedding_types'] = config['obs_embedding_types']
        MODEL_ARGS['embedding_params'] = config['obs_embedding_params']
    d_input = train_dataset.d_input
    if train_dataset.flatten:
        MODEL_ARGS['ntokens_per_timepoint'] = train_dataset.ntokens_per_timepoint
        d_input = train_dataset.flatten_dinput
        d_output = train_dataset.flatten_max_doutput
    elif train_dataset.discretize:
        MODEL_ARGS['d_output_discrete'] = train_dataset.d_output_discrete
        MODEL_ARGS['nbins'] = train_dataset.discretize_nbins
        d_output = train_dataset.d_output_continuous
    else:
        d_output = train_dataset.d_output

    modelstatetype = config.get('modelstatetype', None)
    assert modelstatetype not in ['prob', 'min'], f"modelstatetype '{modelstatetype}' not supported"
    modeltype = config.get('modeltype', None)
    assert modeltype != 'mlm', "modeltype 'mlm' not supported"

    # if config['modelstatetype'] == 'prob':
    #     model = TransformerStateModel(d_input, d_output, **MODEL_ARGS).to(device)
    #     criterion = prob_causal_criterion
    # elif config['modelstatetype'] == 'min':
    #     model = TransformerBestStateModel(d_input, d_output, **MODEL_ARGS).to(device)
    #     criterion = min_causal_criterion
    # else:
    model = TransformerModel(d_input, d_output, **MODEL_ARGS).to(device)

    if train_dataset.discretize:
        # Before refactor this was: config['weight_discrete'] = len(config['discreteidx']) / nfeatures
        config['weight_discrete'] = train_dataset.d_output_discrete / train_dataset.d_output
        # if config['modeltype'] == 'mlm':
        #     criterion = mixed_masked_criterion
        # else:
        criterion = mixed_causal_criterion
    else:
        # if config['modeltype'] == 'mlm':
        #     criterion = masked_criterion
        # else:
        criterion = causal_criterion

    # if train_dataset.dct_m is not None and config['weight_dct_consistency'] > 0:
    #   criterion = lambda tgt,pred,**kwargs: criterion(tgt,pred,**kwargs) + train_dataset.compare_dct_to_next_relative(pred)

    return model, criterion


def sanity_check_temporal_dep(train_dataloader, device, train_src_mask, is_causal, model, tmess=300):
    # sanity check on temporal dependences
    # create xin2 that is like xin, except xin2 from frame tmess onwards is set to garbage value 100.
    # make sure that model(xin) and model(xin2) matches before frame tmess
    x = next(iter(train_dataloader))
    xin = x['input'].clone()
    xin2 = xin.clone()
    tmess = 300
    contextl = xin.shape[1]
    if tmess >= contextl:
        tmess = contextl // 2
    xin2[:, tmess:, :] = 100.
    model.eval()
    with torch.no_grad():
        pred = model(xin.to(device), mask=train_src_mask, is_causal=is_causal)
        pred2 = model(xin2.to(device), mask=train_src_mask, is_causal=is_causal)
    if type(pred) == dict:
        for k in pred.keys():
            matches = torch.allclose(pred2[k][:, :tmess], pred[k][:, :tmess], atol=1e-2)
            assert matches, f'Mismatch in {k}'
    else:
        matches = torch.allclose(pred2[:, :tmess],pred[:, :tmess], atol=1e-2)
        assert matches


def causal_criterion(tgt, pred, useoutputmask=None):
    if useoutputmask is not None:
        pred = pred[useoutputmask.to(device=pred.device)]
        tgt = tgt[useoutputmask.to(device=tgt.device)]
    assert torch.isnan(tgt).any() == False, 'tgt contains nans'
    assert torch.isnan(pred).any() == False, 'pred contains nans'
    d = tgt.shape[-1]
    assert d > 0, 'tgt has no dimensions'
    err = torch.sum(torch.abs(tgt - pred)) / d
    return err


def mixed_causal_criterion(tgt, pred, weight_discrete=.5, extraout=False):
    tgt_continuous = tgt.get('labels', None)
    tgt_discrete = tgt.get('labels_discrete', None)
    pred_continuous = pred.get('continuous', None)
    pred_discrete = pred.get('discrete', None)
    useoutputmask = tgt.get('useoutputmask', None)

    iscontinuous = tgt_continuous is not None and tgt_continuous.shape[-1] > 0
    isdiscrete = tgt_discrete is not None

    if useoutputmask is not None:
        assert torch.any(useoutputmask).item(), 'useoutputmask is all false'
        if iscontinuous:
            pred_continuous = pred_continuous[useoutputmask.to(device=pred_continuous.device)]
            tgt_continuous = tgt_continuous[useoutputmask]
        if isdiscrete:
            pred_discrete = pred_discrete[useoutputmask.to(device=pred_discrete.device)]
            tgt_discrete = tgt_discrete[useoutputmask]

    if iscontinuous:
        n = np.prod(tgt_continuous.shape[:-1])
    else:
        n = np.prod(tgt_discrete.shape[:-2])
    assert n > 0, 'no examples in batch'
    if iscontinuous:
        err_continuous = lossfcn_continuous(pred_continuous, tgt_continuous.to(device=pred_continuous.device)) * n
    else:
        err_continuous = torch.tensor(0., dtype=tgt_discrete.dtype, device=tgt_discrete.device)
    if isdiscrete:
        pd = pred_discrete
        newsz = (np.prod(pd.shape[:-1]), pd.shape[-1])
        pd = pd.reshape(newsz)
        td = tgt_discrete.to(device=pd.device).reshape(newsz)
        err_discrete = lossfcn_discrete(pd, td) * n
        assert torch.isnan(err_discrete).any() == False, 'discrete loss contains nans'
    else:
        err_discrete = torch.tensor(0., dtype=tgt_continuous.dtype, device=tgt_continuous.device)
    err = (1 - weight_discrete) * err_continuous + weight_discrete * err_discrete
    if extraout:
        return err, err_discrete, err_continuous
    else:
        return err


def compute_loss_mixed(model, dataloader, device, mask, weight_discrete, input_gt=False):
    model.eval()
    with torch.no_grad():
        all_loss = torch.zeros(len(dataloader), device=device)
        loss = torch.tensor(0.0).to(device)
        loss_discrete = torch.tensor(0.0).to(device)
        loss_continuous = torch.tensor(0.0).to(device)
        nmask = 0
        for i, example in enumerate(dataloader):
            if input_gt:
                pred = model(example['input'].to(device=device, non_blocking=True), mask=mask, is_causal=True,
                             gt=example['labels_discrete'].to(device=device))
            else:
                pred = model(example['input'].to(device=device, non_blocking=True), mask=mask, is_causal=True)
            loss_curr, loss_discrete_curr, loss_continuous_curr = mixed_causal_criterion(
                example, pred, weight_discrete=weight_discrete, extraout=True
            )
            nmask += example['useoutputmask'].sum().detach()
            all_loss[i] = loss_curr
            loss += loss_curr
            loss_discrete += loss_discrete_curr
            loss_continuous += loss_continuous_curr

        loss = loss.item() / nmask
        loss_discrete = loss_discrete.item() / nmask
        loss_continuous = loss_continuous.item() / nmask
        return loss, loss_discrete, loss_continuous


def initialize_loss(train_dataset, config):
    loss_epoch = {}
    keys = ['train', 'val']
    if train_dataset.discretize:
        keys = keys + ['train_continuous', 'train_discrete', 'val_continuous', 'val_discrete']
    for key in keys:
        loss_epoch[key] = torch.zeros(config['num_train_epochs'])
        loss_epoch[key][:] = torch.nan
    return loss_epoch


def update_loss_nepochs(loss_epoch, nepochs):
    for k, v in loss_epoch.items():
        if v.numel() < nepochs:
            n = torch.zeros(nepochs - v.numel(), dtype=v.dtype, device=v.device) + torch.nan
            loss_epoch[k] = torch.cat((v, n))
    return


def criterion_wrapper(example, pred, criterion, dataset, config):
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

    if 'discrete' in pred:
        pred_discrete = pred['discrete']
    elif 'labels_discrete' in pred:
        pred_discrete = pred['labels_discrete']
    else:
        pred_discrete = None
    pred1 = {'continuous': pred_continuous, 'discrete': pred_discrete}
    if tgt['labels'] is not None:
        assert torch.isnan(tgt['labels']).any() == False, 'tgt labels contains nans'
    if pred1['continuous'] is not None:
        assert torch.isnan(pred1['continuous']).any() == False, 'pred continuous contains nans'
    if tgt['labels_discrete'] is not None:
        assert torch.isnan(tgt['labels_discrete']).any() == False, 'tgt labels discrete contains nans'
    if pred1['discrete'] is not None:
        assert torch.isnan(pred1['discrete']).any() == False, 'pred discrete contains nans'

    if config['modeltype'] == 'mlm':
        if dataset.discretize:
            loss, loss_discrete, loss_continuous = criterion(tgt, pred1, mask=example['mask'].to(pred.device),
                                                             weight_discrete=config['weight_discrete'], extraout=True)
        else:
            loss = criterion(tgt_continuous.to(device=pred.device), pred_continuous,
                             example['mask'].to(pred.device), useoutputmask=tgt['useoutputmask'])
            loss_continuous = loss
            loss_discrete = 0.
    else:
        if dataset.discretize:
            loss, loss_discrete, loss_continuous = criterion(tgt, pred1, weight_discrete=config['weight_discrete'],
                                                             extraout=True)
        else:
            loss = criterion(tgt_continuous.to(device=pred_continuous.device), pred_continuous, useoutputmask=tgt['useoutputmask'])
            loss_continuous = loss
            loss_discrete = 0.
    return loss, loss_discrete, loss_continuous


def generate_square_full_mask(sz: int) -> torch.Tensor:
    """
    Generates a zero matrix. All words allowed.
    """
    return torch.zeros(sz, sz)


def apply_mask(x, mask, nin=0, maskflagged=False):
    # mask with zeros
    if maskflagged:
        if mask is not None:
            x[mask, :-nin - 1] = 0.
            x[mask, -1] = 1.
    else:
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=x.dtype)
        else:
            x[mask, :-nin] = 0.
        x = torch.cat((x, mask[..., None].type(x.dtype)), dim=-1)
    return x


def get_output_and_attention_weights(model, inputs, mask=None, is_causal=False):
    # set need_weights to True for this function call
    model.set_need_weights(True)

    # where attention weights will be stored, one list element per layer
    activation = [None, ] * model.transformer_encoder.num_layers

    def get_activation(layer_num):
        # the hook signature
        def hook(model, inputs, output):
            # attention weights are the second output
            activation[layer_num] = output[1]

        return hook

    # register the hooks
    hooks = [None, ] * model.transformer_encoder.num_layers
    for i, layer in enumerate(model.transformer_encoder.layers):
        hooks[i] = layer.self_attn.register_forward_hook(get_activation(i))

    # call the model
    with torch.no_grad():
        output = model.output(inputs, mask=mask, is_causal=is_causal)

    # remove the hooks
    for hook in hooks:
        hook.remove()

    # return need_weights to False
    model.set_need_weights(False)

    return output, activation


def pred_apply_fun(pred, fun):
    if isinstance(pred, dict):
        return {k: fun(v) for k, v in pred.items()}
    else:
        return fun(pred)

