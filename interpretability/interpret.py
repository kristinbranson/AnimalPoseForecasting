#!/usr/bin/env python
"""
Interpretability tools for the FlyLLM pose-forecasting transformer.

Asks whether the model internally represents a human-interpretable behavioral concept
(see apf/concepts.py -- e.g. "this fly is stopped and about to start walking"), and if so,
where. Three entry points:

    probe-residual   linear / MLP probe on the residual stream at a given layer
    probe-attention  CNN probe on a layer's attention maps
    simulate         roll the model out around concept events and animate it

All three share the concept-anchoring convention below, one ModelInterpreter, and one
sampler -- previously these were duplicated, with drift, across three scripts.

Trajectory anchoring
--------------------
A concept frame `f` for agent `a` becomes the model input window starting at
`f - contextl`, so the concept event lands at the FINAL position of the context window.
The model therefore sees only the run-up to the event, never the event itself.

Binary probe convention: positives are `start_walking` (+1) and negatives are
`stays_stopped` (-1). `neutral` (0) is sampled but unused by the probes.

Examples
--------
    python -m interpretability.interpret probe-residual --layer 9 --pooling mean
    python -m interpretability.interpret probe-attention --layer 9 --max-samples 1000
    python -m interpretability.interpret simulate --n-samples 5 --output-dir animations/

    # control baseline: probe an UNTRAINED model, accuracy should be near chance
    python -m interpretability.interpret probe-residual --layer 9 --random-init
"""

import argparse
import logging
import os
import pickle

import matplotlib
matplotlib.use('Agg')  # scripted use; no display on the cluster
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from apf.simulation import simulate
from flyllm.prepare import init_flyllm
from flyllm.simulation import animate_pose

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Defaults matching apf.concepts.HumanConcept; overridable from the CLI.
DEFAULT_CONCEPT_PARAMS = {
    'concept_type': 'start_walking',
    'sigma': 2,
    'thresh_stopped': 5.0,
    'thresh_walking': 15.0,
    'tstopped': 0.5,
    'tfuture': 1.0,
}

POSITIVE_CONCEPT = 'start_walking'   # +1
NEGATIVE_CONCEPT = 'stays_stopped'   # -1


# ============================================================================
# SETUP
# ============================================================================

def load_flyllm(config_file, model_file, mode='test', random_init=False, concept_params=None):
    """
    Load config, dataset (with concepts computed) and model.

    The checkpoint is loaded whenever model_file is passed through; `mode` does NOT control
    weight loading (init_flyllm loads whenever loadmodelfile is set -- mode='test'
    additionally pulls the config from the checkpoint so the architecture matches).
    Probing a randomly initialized model measures nothing, so it must be asked for
    explicitly via random_init.

    Returns:
        dict from init_flyllm, with 'config', 'train_dataset', 'model', 'device'.
    """
    override = {
        'compute_concepts': True,
        'concept_params': concept_params or dict(DEFAULT_CONCEPT_PARAMS),
    }

    if random_init:
        # mode='test' asserts a checkpoint is present, so the untrained control runs
        # under 'train', which initializes the weights randomly.
        LOG.warning("--random-init: probing an UNTRAINED model. Expect near-chance accuracy; "
                    "this is a control, not a result.")
        mode, load_file = 'train', None
    else:
        load_file = model_file
        LOG.info(f"Probing trained model: {load_file}")

    res = init_flyllm(
        configfile=config_file,
        mode=mode,
        restartmodelfile=None,
        loadmodelfile=load_file,
        debug_uselessdata=False,
        needtraindata=True,
        overrideconfig=override,
    )

    if res['train_dataset'].concepts is None:
        raise RuntimeError(
            "Concepts were not computed. compute_concepts=True was passed to init_flyllm but "
            "dataset.concepts is None -- check that experiments/flyllm.py:make_dataset ran."
        )
    return res


# ============================================================================
# MODEL INTERPRETER
# ============================================================================

class ModelInterpreter:
    """
    Captures a model's internals during a forward pass.

    Two things can be captured, independently:
      - the residual stream, via forward hooks on pos_encoder (layer 0) and each
        transformer block (layers 1..n)
      - attention maps, via forward hooks on each MultiheadAttention

    Attention needs more than a hook: PyTorch's fused attention path does not return
    per-head weights unless asked, so enable_attention_output() monkey-patches each
    MultiheadAttention.forward to force need_weights=True, average_attn_weights=False.
    That patch mutates the model, so it MUST be undone -- use this as a context manager,
    or call close() yourself.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.activations = {}        # residual stream, by layer name
        self.attention_weights = {}  # attention maps, by module name
        self.hooks = []
        self._patched_modules = []   # modules whose forward we replaced

    # -------------------------------------------------------------- lifecycle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Remove hooks and undo the attention monkey-patch. Safe to call twice."""
        self.remove_hooks()
        self.restore_attention_forward()
        self.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        self.activations = {}
        self.attention_weights = {}

    # ------------------------------------------------------- residual stream

    def register_residual_hooks(self):
        """Hook the residual stream. Layer 0 is the input to the transformer (post
        positional encoding); layer i+1 is the output of transformer block i."""
        self.remove_hooks()

        def capture(name):
            def hook(module, inp, output):
                if isinstance(output, dict):
                    return
                tensor = output[0] if isinstance(output, tuple) else output
                self.activations[name] = tensor.detach().cpu()
            return hook

        for name, module in self.model.named_modules():
            if name == 'pos_encoder':
                self.hooks.append(module.register_forward_hook(capture(residual_layer_name(0))))
            elif name.startswith('transformer_encoder.layers.'):
                parts = name.split('.')
                if len(parts) == 3 and parts[-1].isdigit():
                    layer = int(parts[-1])
                    self.hooks.append(
                        module.register_forward_hook(capture(residual_layer_name(layer + 1)))
                    )

        LOG.info(f"Registered {len(self.hooks)} residual stream hooks")

    # ------------------------------------------------------------- attention

    def enable_attention_output(self):
        """Force every MultiheadAttention to return per-head weights.

        PyTorch's fused path returns None for the weights unless need_weights is set, and
        averages over heads unless average_attn_weights=False. Neither can be requested from
        outside, so we wrap forward. Undone by restore_attention_forward().
        """
        import inspect

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.MultiheadAttention):
                continue

            original_forward = module.__class__.forward.__get__(module, module.__class__)
            supports_is_causal = 'is_causal' in inspect.signature(original_forward).parameters

            def make_wrapper(orig, supports_causal):
                def wrapper(query, key, value, key_padding_mask=None, need_weights=True,
                            attn_mask=None, average_attn_weights=False, is_causal=False, **kwargs):
                    call_kwargs = {
                        'key_padding_mask': key_padding_mask,
                        'need_weights': True,
                        'attn_mask': attn_mask,
                        'average_attn_weights': False,
                    }
                    if supports_causal:
                        call_kwargs['is_causal'] = is_causal
                    call_kwargs.update(kwargs)
                    return orig(query, key, value, **call_kwargs)
                return wrapper

            module.forward = make_wrapper(original_forward, supports_is_causal)
            self._patched_modules.append(module)

    def restore_attention_forward(self):
        """Undo enable_attention_output(). Without this the model stays patched for the
        rest of the process, and every later forward pass pays for weight materialization."""
        for module in self._patched_modules:
            # Drop the instance attribute so the class's forward takes over again.
            module.__dict__.pop('forward', None)
        self._patched_modules = []

    def register_attention_hooks(self):
        """Hook every MultiheadAttention to capture its per-head attention maps."""
        self.remove_hooks()
        self.enable_attention_output()

        def capture(name):
            def hook(module, inp, output):
                if not (isinstance(output, tuple) and len(output) > 1):
                    return
                attn = output[1]
                if attn is None:
                    return
                if attn.ndim == 4:
                    # (batch, heads, L, S)
                    self.attention_weights[name] = attn.detach().cpu()
                elif attn.ndim == 3:
                    if attn.shape[1] == attn.shape[2]:
                        self.attention_weights[name] = attn.detach().cpu()
                    else:
                        # (L, batch*heads, S) -> (batch, heads, L, S)
                        L, N, S = attn.shape
                        heads = module.num_heads
                        attn = attn.view(L, N // heads, heads, S).permute(1, 2, 0, 3)
                        self.attention_weights[name] = attn.detach().cpu()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                self.hooks.append(module.register_forward_hook(capture(name)))

        LOG.info(f"Registered {len(self.hooks)} attention hooks")

    # --------------------------------------------------------------- forward

    def forward(self, input_array):
        """Run one chunk through the model. is_causal=True; apf.models.TransformerModel
        generates the causal mask itself when none is supplied, which is what keeps the
        captured internals causal outside the training loop."""
        tensor = torch.FloatTensor(input_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model(tensor, is_causal=True)


def residual_layer_name(layer):
    return f'residual_stream_layer_{layer}'


def attention_layer_name(layer):
    return f'transformer_encoder.layers.{layer}.self_attn'


# ============================================================================
# SAMPLING
# ============================================================================

def sample_trajectories_by_concept(concept_labels, n_samples=100, method='random', rng=None):
    """
    Sample trajectory anchors (agent_id, frame) for each concept class.

    Args:
        concept_labels: (n_agents, n_frames, 1) array, i.e. dataset.concepts.array
        n_samples: max anchors per class
        method:
            'random'   -- uniformly from all frames carrying the label
            'boundary' -- the last frame of each +1 run and the first frame of each -1/0 run,
                          i.e. anchors sitting exactly at the behavioral transition
        rng: np.random.Generator, for reproducibility

    Returns:
        dict: 'start_walking' | 'stays_stopped' | 'neutral' -> list of (agent_id, frame)
    """
    if rng is None:
        rng = np.random.default_rng()

    class_names = {1: POSITIVE_CONCEPT, -1: NEGATIVE_CONCEPT, 0: 'neutral'}
    samples = {name: [] for name in class_names.values()}

    if method == 'random':
        for value, name in class_names.items():
            positions = np.argwhere(concept_labels == value)
            if len(positions) == 0:
                LOG.warning(f"No frames found for {name} ({value:+d})")
                continue
            idx = rng.choice(len(positions), size=min(n_samples, len(positions)), replace=False)
            samples[name] = [(int(positions[i][0]), int(positions[i][1])) for i in idx]

    elif method == 'boundary':
        for agent_id in range(concept_labels.shape[0]):
            agent_labels = concept_labels[agent_id, :, 0]
            for value, name in class_names.items():
                runs = _find_runs(agent_labels == value)
                # The +1 transition is at the END of the run (the fly is about to move);
                # for the others the informative anchor is the START.
                anchors = [end for _, end in runs] if value == 1 else [start for start, _ in runs]
                samples[name].extend(
                    (int(agent_id), int(f)) for f in anchors if not np.isnan(agent_labels[f])
                )

        for name in samples:
            if len(samples[name]) > n_samples:
                idx = rng.choice(len(samples[name]), size=n_samples, replace=False)
                samples[name] = [samples[name][i] for i in idx]

    else:
        raise ValueError(f"Unknown sampling method {method!r}. Choose 'random' or 'boundary'.")

    for name, anchors in samples.items():
        LOG.info(f"{name}: sampled {len(anchors)} trajectories ({method})")
    return samples


def _find_runs(mask):
    """Contiguous True runs in a 1D bool array, as (first, last) inclusive."""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0] - 1))


# ============================================================================
# EXTRACTION
# ============================================================================

def extract_dataset(interpreter, samples, dataset, contextl, capture, layer_name,
                    max_per_class=None, skip_nan=False):
    """
    Build a binary classification dataset from the model's internals.

    For each sampled anchor, the input window is [frame - contextl, frame), so the concept
    event sits just past the end of the context -- the model sees the run-up, not the event.

    Args:
        interpreter: ModelInterpreter, with the relevant hooks already registered
        samples: dict class_name -> [(agent_id, frame)], from sample_trajectories_by_concept
        dataset: the Dataset to pull chunks from
        contextl: config['contextl']
        capture: dict on the interpreter holding the captured tensors
                 (interpreter.activations or interpreter.attention_weights)
        layer_name: key into `capture`
        max_per_class: cap per class
        skip_nan: drop samples containing NaN (attention maps can be NaN where data is missing)

    Returns:
        (features, labels): features (n, ...) float array; labels (n,) with 1=positive, 0=negative
    """
    features, labels = [], []

    for concept, label in ((POSITIVE_CONCEPT, 1), (NEGATIVE_CONCEPT, 0)):
        anchors = samples[concept]
        if max_per_class is not None:
            anchors = anchors[:max_per_class]

        n_skipped_edge = n_skipped_nan = n_failed = 0

        for agent_id, frame in tqdm(anchors, desc=f"extracting {concept}"):
            start = frame - contextl
            if start < 0:
                n_skipped_edge += 1
                continue

            try:
                chunk = dataset.get_chunk(start, dataset.context_length, agent_id)
                interpreter.forward(chunk['input'])

                if layer_name not in capture:
                    n_failed += 1
                    continue

                array = capture[layer_name][0].numpy()
                if skip_nan and np.isnan(array).any():
                    n_skipped_nan += 1
                    continue

                features.append(array)
                labels.append(label)
            except Exception as e:
                LOG.warning(f"agent={agent_id} frame={frame}: {e}")
                n_failed += 1
            finally:
                interpreter.clear()

        # Report what was dropped -- silent truncation reads as full coverage when it isn't.
        LOG.info(f"{concept}: kept {sum(1 for l in labels if l == label)}/{len(anchors)} "
                 f"(dropped {n_skipped_edge} too close to start, {n_skipped_nan} NaN, {n_failed} errors)")

    if not features:
        raise ValueError("No valid samples extracted -- every candidate was dropped.")

    features = np.array(features)
    labels = np.array(labels)
    LOG.info(f"Dataset: {features.shape}, {np.sum(labels == 1)} positive / {np.sum(labels == 0)} negative")
    return features, labels


def apply_pca(x_train, x_val, n_components, seed=None):
    """
    Reduce residual stream states to their leading principal components.

    PCA is fit on the TRAINING split only and applied to validation -- fitting on both would
    leak validation structure into the probe's input and inflate its accuracy.

    Each timestep is treated as a PCA sample, so the fit sees (n_samples * seq_len, hidden_dim)
    and the sequence dimension is preserved in the output.

    Args:
        x_train: (n_train, seq_len, hidden_dim), or (n_train, 1, seq_len, hidden_dim)
        x_val: same, with n_val
        n_components: components to keep
        seed: random_state for the SVD solver. At this size sklearn picks the *randomized*
            solver, which is nondeterministic when unseeded -- two runs on identical data
            would produce different components and an unreproducible probe.

    Returns:
        (x_train_pca, x_val_pca, pca) with the arrays shaped (n, seq_len, n_components)
    """
    if x_train.ndim == 4:
        x_train, x_val = x_train[:, 0], x_val[:, 0]

    n_train, seq_len, hidden_dim = x_train.shape
    n_val = x_val.shape[0]

    train_flat = np.nan_to_num(x_train.reshape(-1, hidden_dim), nan=0.0)
    val_flat = np.nan_to_num(x_val.reshape(-1, hidden_dim), nan=0.0)

    LOG.info(f"PCA: fitting on {train_flat.shape[0]:,} timesteps x {hidden_dim} dims "
             f"-> {n_components} components")
    pca = PCA(n_components=n_components, random_state=seed)
    train_pca = pca.fit_transform(train_flat).reshape(n_train, seq_len, n_components)
    val_pca = pca.transform(val_flat).reshape(n_val, seq_len, n_components)

    explained = pca.explained_variance_ratio_
    LOG.info(f"PCA: {explained.sum():.4f} of variance explained by {n_components} components")
    LOG.info(f"PCA: per-component variance {np.round(explained, 4)}")
    return train_pca, val_pca, pca


class ArrayDataset(torch.utils.data.Dataset):
    """Features + binary labels. NaN->0 happens per-sample in __getitem__ rather than up
    front: the residual stream for a full run is billions of elements, and scanning it
    eagerly costs more than the training does."""

    def __init__(self, features, labels):
        if features.ndim == 4 and features.shape[1] == 1:
            features = features[:, 0]  # drop the batch dim left by the forward pass
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(np.nan_to_num(self.features[idx], nan=0.0))
        y = torch.FloatTensor([self.labels[idx]])
        return x, y


# ============================================================================
# PROBES
# ============================================================================

class ResidualProbe(nn.Module):
    """Probe on residual stream states, (batch, seq_len, hidden_dim) -> (batch, 1) logits."""

    def __init__(self, hidden_dim=2048, pooling='mean', probe_type='linear',
                 mlp_hidden_dim1=256, mlp_hidden_dim2=128):
        super().__init__()
        self.pooling = pooling

        if pooling == 'cnn':
            self.encoder = nn.Sequential(
                nn.Conv1d(hidden_dim, 512, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(512, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            in_dim = 128
        else:
            self.encoder = None
            in_dim = hidden_dim

        if probe_type == 'linear':
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, 1))
        elif probe_type == '2layer':
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, mlp_hidden_dim1), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(mlp_hidden_dim1, 1),
            )
        elif probe_type == 'deep':
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, mlp_hidden_dim1), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(mlp_hidden_dim1, mlp_hidden_dim2), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(mlp_hidden_dim2, 1),
            )
        else:
            raise ValueError(f"Unknown probe_type {probe_type!r}. Choose linear, 2layer or deep.")

    def forward(self, x):
        if self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        elif self.pooling == 'last':
            x = x[:, -1, :]
        elif self.pooling == 'cnn':
            x = self.encoder(x.transpose(1, 2))
        else:
            raise ValueError(f"Unknown pooling {self.pooling!r}")
        return self.classifier(x)


class AttentionProbe(nn.Module):
    """CNN probe on attention maps, (batch, heads, seq_len, seq_len) -> (batch, 1) logits."""

    def __init__(self, num_heads=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_heads, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


# ============================================================================
# TRAINING
# ============================================================================

def train_probe(model, train_loader, val_loader, device, epochs=50, lr=1e-3, patience=10,
                clip_grad=None, lr_scheduler=None, lr_patience=5, lr_factor=0.5):
    """Train a probe with early stopping on validation loss. Returns the history dict; the
    model is left holding the best (not the last) weights."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=lr_patience, factor=lr_factor)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss, best_state, waited = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            loss_sum += loss.item()
            correct += ((torch.sigmoid(out) > 0.5).float() == y).sum().item()
            total += y.size(0)
        train_loss, train_acc = loss_sum / len(train_loader), correct / total

        model.eval()
        loss_sum = correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss_sum += criterion(out, y).item()
                correct += ((torch.sigmoid(out) > 0.5).float() == y).sum().item()
                total += y.size(0)
        val_loss, val_acc = loss_sum / len(val_loader), correct / total

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        if scheduler is not None:
            scheduler.step(val_loss) if lr_scheduler == 'plateau' else scheduler.step()

        LOG.info(f"epoch {epoch + 1}/{epochs}: train {train_loss:.4f}/{train_acc:.4f}  "
                 f"val {val_loss:.4f}/{val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, waited = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                LOG.info(f"early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate(model, loader, device):
    """Returns (predictions, true_labels) over the loader."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.extend((torch.sigmoid(out) > 0.5).cpu().numpy().flatten())
            trues.extend(y.numpy().flatten())
    return np.array(preds), np.array(trues)


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set(xlabel='epoch', ylabel='loss', title='Loss')
    axes[0].legend()
    axes[1].plot(history['train_acc'], label='train')
    axes[1].plot(history['val_acc'], label='val')
    axes[1].axhline(0.5, ls='--', c='gray', lw=1, label='chance')
    axes[1].set(xlabel='epoch', ylabel='accuracy', title='Accuracy')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOG.info(f"wrote {path}")


def plot_confusion(trues, preds, path):
    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[NEGATIVE_CONCEPT, POSITIVE_CONCEPT],
                yticklabels=[NEGATIVE_CONCEPT, POSITIVE_CONCEPT])
    ax.set(xlabel='predicted', ylabel='true', title='Confusion matrix')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOG.info(f"wrote {path}")


# ============================================================================
# COMMANDS
# ============================================================================

def _probe_common(args, kind):
    """Shared body of probe-residual and probe-attention: load, sample, extract, train, save."""
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    res = load_flyllm(args.config, args.model, mode=args.mode, random_init=args.random_init)
    config, dataset, model, device = res['config'], res['train_dataset'], res['model'], res['device']

    samples = sample_trajectories_by_concept(
        dataset.concepts.array, n_samples=args.n_samples, method=args.sampling_method, rng=rng)

    with ModelInterpreter(model, device=device) as interpreter:
        if kind == 'residual':
            interpreter.register_residual_hooks()
            capture, layer_name = interpreter.activations, residual_layer_name(args.layer)
            skip_nan = False
        else:
            interpreter.register_attention_hooks()
            capture, layer_name = interpreter.attention_weights, attention_layer_name(args.layer)
            skip_nan = True  # attention is NaN wherever the input chunk has missing data

        features, labels = extract_dataset(
            interpreter, samples, dataset, config['contextl'], capture, layer_name,
            max_per_class=args.max_samples, skip_nan=skip_nan)

    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=args.seed, stratify=labels)

    pca = None
    if kind == 'residual' and args.pca_components:
        # Fit on the train split only -- see apply_pca.
        x_train, x_val, pca = apply_pca(x_train, x_val, args.pca_components, seed=args.seed)

    train_loader = DataLoader(ArrayDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ArrayDataset(x_val, y_val), batch_size=args.batch_size)

    if kind == 'residual':
        probe = ResidualProbe(
            hidden_dim=x_train.shape[-1], pooling=args.pooling, probe_type=args.probe_type,
            mlp_hidden_dim1=args.hidden_dim1, mlp_hidden_dim2=args.hidden_dim2)
        stem = f"residual_layer{args.layer}_{args.pooling}_{args.probe_type}_{args.sampling_method}"
        if pca is not None:
            stem += f"_pca{args.pca_components}"
    else:
        probe = AttentionProbe(num_heads=features.shape[1])
        stem = f"attention_layer{args.layer}_{args.sampling_method}"

    if args.random_init:
        stem += "_randominit"  # keep the untrained control from overwriting a real result

    LOG.info(f"probe: {sum(p.numel() for p in probe.parameters()):,} parameters")

    history = train_probe(
        probe, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        clip_grad=args.clip_grad, lr_scheduler=args.lr_scheduler)

    preds, trues = evaluate(probe, val_loader, device)
    LOG.info("\n" + classification_report(trues, preds, target_names=[NEGATIVE_CONCEPT, POSITIVE_CONCEPT]))
    LOG.info(f"best val accuracy: {max(history['val_acc']):.4f}  (chance = 0.5)")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        'model_state_dict': probe.state_dict(),
        'kind': kind,
        'layer': args.layer,
        'layer_name': layer_name,
        'feature_shape': features.shape[1:],
        'probe_input_shape': x_train.shape[1:],
        # The fitted PCA is needed to reproduce the probe's input at inference time.
        'pca': pca,
        'explained_variance': None if pca is None else pca.explained_variance_ratio_,
        'history': history,
        'args': vars(args),
    }, os.path.join(args.output_dir, f"{stem}.pth"))
    plot_history(history, os.path.join(args.output_dir, f"{stem}_history.png"))
    plot_confusion(trues, preds, os.path.join(args.output_dir, f"{stem}_confusion.png"))


def cmd_probe_residual(args):
    _probe_common(args, 'residual')


def cmd_probe_attention(args):
    _probe_common(args, 'attention')


def cmd_simulate(args):
    """Roll the model out around concept events and animate predicted vs ground-truth pose."""
    rng = np.random.default_rng(args.seed)

    res = load_flyllm(args.config, args.model, mode=args.mode, random_init=args.random_init)
    config, dataset, model = res['config'], res['train_dataset'], res['model']
    # track/pose/flyids are nested under train_data, not at the top level of res.
    train_data = res['train_data']
    track, pose, flyids = train_data['track'], train_data['pose'], train_data['flyids']

    samples = sample_trajectories_by_concept(
        dataset.concepts.array, n_samples=args.n_samples, method=args.sampling_method, rng=rng)

    contextl = config['contextl']
    track_len = args.rollout_frames + contextl + 1
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model).replace('.pth', '')

    results = {}
    for concept in (POSITIVE_CONCEPT, NEGATIVE_CONCEPT):
        results[concept] = []
        for i, (agent_id, frame) in enumerate(samples[concept]):
            start = frame - contextl
            if start < 0:
                LOG.info(f"skipping agent={agent_id} frame={frame}: not enough context")
                continue

            # An agent_id is a positional slot, not a stable identity -- if the slot changes
            # fly mid-window the rollout is meaningless.
            identities = np.unique(flyids[start:start + track_len, agent_id])
            if len(identities) != 1:
                LOG.info(f"skipping agent={agent_id} frame={frame}: slot switches identity {identities}")
                continue

            try:
                gt_track, pred_track = simulate(
                    dataset=dataset, model=model, track=track, pose=pose, identities=flyids,
                    track_len=track_len, burn_in=contextl, max_contextl=contextl,
                    agent_ids=[agent_id], start_frame=start)
            except Exception as e:
                LOG.warning(f"agent={agent_id} frame={frame}: rollout failed: {e}")
                continue

            if np.isnan(pred_track).all():
                # Rollouts blow up to NaN when the burn-in window has missing data.
                LOG.warning(f"agent={agent_id} frame={frame}: rollout is all NaN, skipping")
                continue

            out = os.path.join(
                args.output_dir, f"{concept}_sample{i}_agent{agent_id}_frame{frame}_{model_name}.gif")
            animate_pose({'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()},
                         focusflies=[agent_id], savevidfile=out, contextl=contextl)
            LOG.info(f"wrote {out}")
            results[concept].append({'agent_id': agent_id, 'frame': frame, 'gif': out})

    with open(os.path.join(args.output_dir, 'simulation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    for concept, items in results.items():
        LOG.info(f"{concept}: {len(items)} rollouts animated")


# ============================================================================
# CLI
# ============================================================================

DEFAULT_CONFIG = ("/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/"
                  "flyllm/configs/config_fly_llm_predvel_optimalbinning_20251113.json")
DEFAULT_MODEL = ("/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/"
                 "notebooks/flyllm_models/flypredvel_20251007_20251114T194024_bestepoch200.pth")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def add_common(p):
        p.add_argument('--config', default=DEFAULT_CONFIG, help='Path to the flyllm config JSON')
        p.add_argument('--model', default=DEFAULT_MODEL, help='Checkpoint to interpret')
        p.add_argument('--mode', default='test', choices=['train', 'test'],
                       help='init_flyllm mode. "test" also loads the config from the checkpoint')
        p.add_argument('--random-init', action='store_true',
                       help='Interpret a randomly initialized model instead of --model. Control '
                            'baseline only: a probe should score near chance on an untrained model')
        p.add_argument('--n-samples', type=int, default=100, help='Trajectories sampled per concept')
        p.add_argument('--sampling-method', default='random', choices=['random', 'boundary'],
                       help='random: any frame with the label. boundary: frames at the transition')
        p.add_argument('--output-dir', default='.', help='Where to write results')
        p.add_argument('--seed', type=int, default=11111)

    def add_probe(p):
        add_common(p)
        p.add_argument('--layer', type=int, default=9, help='Layer to probe')
        p.add_argument('--max-samples', type=int, default=None, help='Cap samples per class')
        p.add_argument('--batch-size', type=int, default=32)
        p.add_argument('--epochs', type=int, default=50)
        p.add_argument('--lr', type=float, default=1e-3)
        p.add_argument('--patience', type=int, default=10, help='Early-stopping patience')
        p.add_argument('--clip-grad', type=float, default=None, help='Gradient-norm clip')
        p.add_argument('--lr-scheduler', default=None, choices=[None, 'plateau', 'cosine'])

    p_res = sub.add_parser('probe-residual', help='Probe the residual stream')
    add_probe(p_res)
    p_res.add_argument('--pooling', default='mean', choices=['mean', 'max', 'last', 'cnn'])
    p_res.add_argument('--probe_type', '--probe-type', dest='probe_type', default='linear',
                       choices=['linear', '2layer', 'deep'])
    p_res.add_argument('--pca-components', type=int, default=None,
                       help='Reduce the residual stream to N principal components before probing. '
                            'PCA is fit on the training split only')
    p_res.add_argument('--hidden-dim1', type=int, default=256)
    p_res.add_argument('--hidden-dim2', type=int, default=128)
    p_res.set_defaults(func=cmd_probe_residual)

    p_attn = sub.add_parser('probe-attention', help='Probe attention maps')
    add_probe(p_attn)
    p_attn.set_defaults(lr=1e-4, batch_size=16, clip_grad=1.0, lr_scheduler='plateau',
                        pca_components=None)
    p_attn.set_defaults(func=cmd_probe_attention)

    p_sim = sub.add_parser('simulate', help='Roll out and animate around concept events')
    add_common(p_sim)
    p_sim.add_argument('--rollout-frames', type=int, default=150,
                       help='Frames to predict past the end of the context window')
    p_sim.set_defaults(n_samples=5, func=cmd_simulate)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
