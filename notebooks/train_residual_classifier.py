#!/usr/bin/env python
"""
Train a binary classifier on residual stream hidden states.

Usage:
    python train_residual_classifier.py --layer 9 --pooling mean --probe_type linear --sampling_method random --n_samples 100
    python train_residual_classifier.py --layer 9 --pooling mean --probe_type 2layer --sampling_method boundary --n_samples 100
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import argparse
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from flyllm.prepare import init_flyllm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class ResidualStreamClassifier(torch.nn.Module):
    """Classifier for residual stream hidden states.

    Takes hidden states from a single layer (seq_len, hidden_dim)
    and predicts a binary label.
    """
    def __init__(self, seq_len=512, hidden_dim=2048, pooling='mean', probe_type='linear',
                 mlp_hidden_dim1=256, mlp_hidden_dim2=128):
        """
        Args:
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            pooling: How to aggregate sequence ('mean', 'max', 'last', 'cnn')
            probe_type: Type of probe ('linear', '2layer', 'deep')
            mlp_hidden_dim1: Hidden dimension for first MLP layer (default 256)
            mlp_hidden_dim2: Hidden dimension for second MLP layer in deep probe (default 128)
        """
        super().__init__()
        self.pooling = pooling
        self.probe_type = probe_type
        self.mlp_hidden_dim1 = mlp_hidden_dim1
        self.mlp_hidden_dim2 = mlp_hidden_dim2

        if pooling == 'cnn':
            # Use 1D CNN to process sequence
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv1d(hidden_dim, 512, kernel_size=5, stride=2, padding=2),  # 512→256
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),  # 256→128

                torch.nn.Conv1d(512, 256, kernel_size=3, stride=2, padding=1),  # 128→64
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),  # 64→32

                torch.nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),  # 32→16
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),  # 16→1
            )
            classifier_input_dim = 128
        else:
            # For pooling methods, we don't need an encoder
            self.encoder = None
            classifier_input_dim = hidden_dim

        # Build probe classifier based on type
        if probe_type == 'linear':
            # Linear probe: single linear layer
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(classifier_input_dim, 1)
            )
        elif probe_type == '2layer':
            # 2-layer MLP with ReLU and dropout
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(classifier_input_dim, mlp_hidden_dim1),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(mlp_hidden_dim1, 1)
            )
        elif probe_type == 'deep':
            # Original 3-layer MLP
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(classifier_input_dim, mlp_hidden_dim1),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(mlp_hidden_dim1, mlp_hidden_dim2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(mlp_hidden_dim2, 1)  # Binary classification
            )
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}. Choose from 'linear', '2layer', 'deep'.")

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len, hidden_dim) residual stream states

        Returns:
            logits: (batch, 1) binary classification logits
        """
        if self.pooling == 'mean':
            x = torch.mean(x, dim=1)  # (batch, hidden_dim)
        elif self.pooling == 'max':
            x = torch.max(x, dim=1)[0]  # (batch, hidden_dim)
        elif self.pooling == 'last':
            x = x[:, -1, :]  # (batch, hidden_dim)
        elif self.pooling == 'cnn':
            x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
            x = self.encoder(x)  # (batch, 128, 1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        x = self.classifier(x)
        return x


class ResidualStreamDataset(torch.utils.data.Dataset):
    """Dataset for residual stream states and binary labels."""

    def __init__(self, hidden_states, labels):
        """
        Args:
            hidden_states: Array of shape (n_samples, seq_len, hidden_dim) or (n_samples, 1, seq_len, hidden_dim)
            labels: Array of shape (n_samples,) with binary labels (0 or 1)
        """
        # Remove batch dimension if present
        if hidden_states.ndim == 4:
            hidden_states = hidden_states[:, 0, :, :]  # (n_samples, seq_len, hidden_dim)

        # Keep as numpy arrays - convert to tensors on-demand in __getitem__
        # NaN replacement is also done per-sample in __getitem__ to avoid
        # scanning the entire dataset upfront (which can be 18+ billion elements)
        self.hidden_states = hidden_states
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert to tensors on-demand (only for the requested batch)
        # Handle NaNs per-sample instead of on the whole dataset
        state = self.hidden_states[idx]
        state = np.nan_to_num(state, nan=0.0)  # Only ~1M elements per sample
        state = torch.FloatTensor(state)
        label = torch.FloatTensor([self.labels[idx]])
        return state, label


# ============================================================================
# MODEL INTERPRETER
# ============================================================================

class ModelInterpreter:
    """Simplified interpreter for extracting residual stream states."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.activations = {}
        self.hooks = []

    def register_residual_stream_hooks(self):
        """Register hooks to capture residual stream hidden states."""
        def get_residual_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                elif isinstance(output, dict):
                    return
                else:
                    self.activations[name] = output.detach().cpu()
            return hook

        # Remove existing hooks
        self.remove_hooks()

        # Hook pos_encoder output (residual stream layer 0 input)
        for name, module in self.model.named_modules():
            if name == 'pos_encoder':
                hook = module.register_forward_hook(get_residual_activation('residual_stream_layer_0'))
                self.hooks.append(hook)
                LOG.info("Registered residual stream hook at pos_encoder (layer 0 input)")
                break

        # Hook each transformer layer output (residual stream after each layer)
        for name, module in self.model.named_modules():
            if name.startswith('transformer_encoder.layers.'):
                parts = name.split('.')
                if len(parts) == 3 and parts[-1].isdigit():
                    layer_num = int(parts[-1])
                    hook_name = f'residual_stream_layer_{layer_num + 1}'
                    hook = module.register_forward_hook(get_residual_activation(hook_name))
                    self.hooks.append(hook)
                    LOG.info(f"Registered residual stream hook after transformer layer {layer_num}")

        LOG.info(f"Registered {len(self.hooks)} residual stream hooks total")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}


# ============================================================================
# SAMPLING
# ============================================================================

def sample_trajectories_by_concept(concept_labels, n_samples=100, sampling_method='random'):
    """
    Sample start positions (agent_id, start_frame) for each concept category.

    Args:
        concept_labels: Array of shape (n_agents, n_timepoints, 1)
        n_samples: Number of samples per concept category
        sampling_method: 'random' or 'boundary'
            - 'random': Randomly sample positions where concept occurs
            - 'boundary': For start_walking, take last frame of +1 sequences;
                         for stays_stopped, take first frame of -1 sequences

    Returns:
        dict with keys 'start_walking' (+1), 'stays_stopped' (-1), 'neutral' (0)
        Each value is a list of (agent_id, start_frame) tuples
    """
    samples = {
        'start_walking': [],  # +1
        'stays_stopped': [],  # -1
        'neutral': []  # 0
    }

    concept_map = {
        1: 'start_walking',
        -1: 'stays_stopped',
        0: 'neutral'
    }

    if sampling_method == 'random':
        # Original random sampling
        for concept_value, concept_name in concept_map.items():
            # Find all positions where this concept occurs
            positions = np.argwhere(concept_labels == concept_value)

            if len(positions) == 0:
                LOG.warning(f"No positions found for {concept_name} ({concept_value})")
                continue

            # Randomly sample n_samples positions
            n_to_sample = min(n_samples, len(positions))
            sampled_indices = np.random.choice(len(positions), size=n_to_sample, replace=False)

            # Convert to (agent_id, start_frame) tuples
            for idx in sampled_indices:
                agent_id, frame, _ = positions[idx]
                samples[concept_name].append((int(agent_id), int(frame)))

            LOG.info(f"{concept_name} ({concept_value}): Sampled {len(samples[concept_name])} trajectories")

    elif sampling_method == 'boundary':
        # Boundary-based sampling: last frame for +1, first frame for -1
        n_agents = concept_labels.shape[0]

        for agent_id in range(n_agents):
            agent_labels = concept_labels[agent_id, :, 0]  # (n_timepoints,)

            # Find sequences of +1 (start_walking)
            start_walking_mask = (agent_labels == 1)
            if np.any(start_walking_mask):
                # Find boundaries: where it changes from +1 to not +1
                padded = np.concatenate([[False], start_walking_mask, [False]])
                diff = np.diff(padded.astype(int))
                starts = np.where(diff == 1)[0]  # Start of +1 sequences
                ends = np.where(diff == -1)[0] - 1  # End of +1 sequences (last frame of +1)

                # Take the last frame of each +1 sequence
                for end_frame in ends:
                    if not np.isnan(agent_labels[end_frame]):
                        samples['start_walking'].append((int(agent_id), int(end_frame)))

            # Find sequences of -1 (stays_stopped)
            stays_stopped_mask = (agent_labels == -1)
            if np.any(stays_stopped_mask):
                # Find boundaries: where it changes to -1
                padded = np.concatenate([[False], stays_stopped_mask, [False]])
                diff = np.diff(padded.astype(int))
                starts = np.where(diff == 1)[0]  # First frame of -1 sequences

                # Take the first frame of each -1 sequence
                for start_frame in starts:
                    if not np.isnan(agent_labels[start_frame]):
                        samples['stays_stopped'].append((int(agent_id), int(start_frame)))

            # Neutral (0) - take first frame of each neutral sequence
            neutral_mask = (agent_labels == 0)
            if np.any(neutral_mask):
                padded = np.concatenate([[False], neutral_mask, [False]])
                diff = np.diff(padded.astype(int))
                starts = np.where(diff == 1)[0]

                for start_frame in starts:
                    if not np.isnan(agent_labels[start_frame]):
                        samples['neutral'].append((int(agent_id), int(start_frame)))

        # Subsample if we have more than n_samples
        for concept_name in samples.keys():
            if len(samples[concept_name]) > n_samples:
                sampled_indices = np.random.choice(len(samples[concept_name]), size=n_samples, replace=False)
                samples[concept_name] = [samples[concept_name][i] for i in sampled_indices]

            LOG.info(f"{concept_name}: Found {len(samples[concept_name])} boundary samples")

    else:
        raise ValueError(f"Unknown sampling_method: {sampling_method}. Choose 'random' or 'boundary'.")

    return samples


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_residual_stream_dataset(interpreter, sampled_trajectories, config,
                                     train_dataset, device, layer_name,
                                     positive_concept='start_walking',
                                     negative_concept='stays_stopped',
                                     max_samples_per_class=None):
    """Extract residual stream states and create binary classification dataset."""
    # Register residual stream hooks
    interpreter.remove_hooks()
    interpreter.register_residual_stream_hooks()

    hidden_states = []
    labels = []

    # Process positive examples
    LOG.info(f"Extracting positive examples ({positive_concept})...")
    positive_trajectories = sampled_trajectories[positive_concept]
    if max_samples_per_class is not None:
        positive_trajectories = positive_trajectories[:max_samples_per_class]

    for agent_id, start_frame in positive_trajectories:
        sim_start_frame = start_frame - config['contextl']
        if sim_start_frame < 0:
            continue

        try:
            example_data = train_dataset.get_chunk(sim_start_frame, train_dataset.context_length, agent_id)
            input_data = example_data['input']
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)

            # Forward pass
            with torch.no_grad():
                _ = interpreter.model(input_tensor)

            # Extract residual stream for specified layer
            if layer_name in interpreter.activations:
                states = interpreter.activations[layer_name]
                hidden_states.append(states.numpy())
                labels.append(1)

            interpreter.clear_activations()
        except Exception as e:
            LOG.warning(f"Failed to process positive example: {e}")
            continue

    LOG.info(f"Extracted {len([l for l in labels if l == 1])} positive examples")

    # Process negative examples
    LOG.info(f"Extracting negative examples ({negative_concept})...")
    negative_trajectories = sampled_trajectories[negative_concept]
    if max_samples_per_class is not None:
        negative_trajectories = negative_trajectories[:max_samples_per_class]

    for agent_id, start_frame in negative_trajectories:
        sim_start_frame = start_frame - config['contextl']
        if sim_start_frame < 0:
            continue

        try:
            example_data = train_dataset.get_chunk(sim_start_frame, train_dataset.context_length, agent_id)
            input_data = example_data['input']
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)

            # Forward pass
            with torch.no_grad():
                _ = interpreter.model(input_tensor)

            # Extract residual stream for specified layer
            if layer_name in interpreter.activations:
                states = interpreter.activations[layer_name]
                hidden_states.append(states.numpy())
                labels.append(0)

            interpreter.clear_activations()
        except Exception as e:
            LOG.warning(f"Failed to process negative example: {e}")
            continue

    LOG.info(f"Extracted {len([l for l in labels if l == 0])} negative examples")

    hidden_states = np.array(hidden_states)
    labels = np.array(labels)

    LOG.info(f"Final dataset: {hidden_states.shape}, labels: {labels.shape}")
    LOG.info(f"Label distribution: {np.sum(labels == 1)} positive, {np.sum(labels == 0)} negative")

    return hidden_states, labels


# ============================================================================
# TRAINING
# ============================================================================

def train_classifier(model, train_loader, val_loader, device,
                     num_epochs=50, lr=1e-3, patience=10):
    """Train the classifier."""
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_states, batch_labels in train_loader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_states, batch_labels in val_loader:
                batch_states = batch_states.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_states)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        LOG.info(f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOG.info(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    return history


def evaluate_model(model, val_loader, device):
    """Evaluate model and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_states, batch_labels in val_loader:
            batch_states = batch_states.to(device)
            outputs = model(batch_states)
            predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(predictions.flatten())
            all_labels.extend(batch_labels.numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train residual stream classifier')
    parser.add_argument('--config', type=str,
                        default="/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json",
                        help='Path to config file')
    parser.add_argument('--model', type=str,
                        default='/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/llmnets/predvel_20251007_20251002T000000_epoch200.pth',
                        help='Path to pretrained model')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode for data loading')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of trajectories to sample per concept')
    parser.add_argument('--layer', type=int, default=9,
                        help='Residual stream layer to use (0-10)')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'last', 'cnn'],
                        help='Pooling method for classifier')
    parser.add_argument('--probe_type', type=str, default='linear',
                        choices=['linear', '2layer', 'deep'],
                        help='Type of probe: linear (single layer), 2layer (2-layer MLP), deep (3-layer MLP)')
    parser.add_argument('--sampling_method', type=str, default='random',
                        choices=['random', 'boundary'],
                        help='Sampling method: random (sample any frame with concept) or boundary (last frame of +1, first frame of -1)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per class (None = all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden_dim1', type=int, default=256,
                        help='Hidden dimension for first layer of MLP probes')
    parser.add_argument('--hidden_dim2', type=int, default=128,
                        help='Hidden dimension for second layer of deep probe')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for saving results')
    parser.add_argument('--seed', type=int, default=11111,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    LOG.info(f"Set random seed to {args.seed}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize FlyLLM
    LOG.info("Initializing FlyLLM...")
    overrideconfig = {
        'compute_concepts': True,
        'concept_params': {
            'concept_type': 'start_walking',
            'sigma': 2,
            'thresh_stopped': 5.0,
            'thresh_walking': 15.0,
            'tstopped': 0.5,
            'tfuture': 1.0,
        }
    }

    loadmodelfile = args.model if args.mode == 'test' else None

    res = init_flyllm(
        configfile=args.config,
        mode=args.mode,
        restartmodelfile=None,
        loadmodelfile=loadmodelfile,
        debug_uselessdata=False,
        needtraindata=True,
        overrideconfig=overrideconfig
    )

    config = res['config']
    train_dataset = res['train_dataset']
    model = res['model']
    device = res['device']

    # Get concept labels and sample trajectories
    LOG.info("Sampling trajectories based on concept labels...")
    if train_dataset.concepts is None:
        LOG.error("Concepts not computed! Make sure compute_concepts=True in config.")
        return

    concept_labels = train_dataset.concepts.array
    LOG.info(f"Concepts shape: {concept_labels.shape}")

    # Count labels
    labels_flat = concept_labels.flatten()
    valid_mask = ~np.isnan(labels_flat)
    labels_valid = labels_flat[valid_mask]

    LOG.info("\nLabel counts:")
    LOG.info(f"  Start walking (+1): {np.sum(labels_valid == 1)}")
    LOG.info(f"  Neutral (0): {np.sum(labels_valid == 0)}")
    LOG.info(f"  Stays stopped (-1): {np.sum(labels_valid == -1)}")
    LOG.info(f"  Invalid (NaN): {np.sum(~valid_mask)}")

    # Sample trajectories
    LOG.info(f"Using sampling method: {args.sampling_method}")
    sampled_trajectories = sample_trajectories_by_concept(
        concept_labels=concept_labels,
        n_samples=args.n_samples,
        sampling_method=args.sampling_method
    )

    # Initialize interpreter
    LOG.info("Initializing ModelInterpreter...")
    interpreter = ModelInterpreter(model=model, device=device)

    # Extract residual stream dataset
    layer_name = f'residual_stream_layer_{args.layer}'
    LOG.info(f"Extracting residual stream states from {layer_name}...")

    hidden_states, labels = extract_residual_stream_dataset(
        interpreter=interpreter,
        sampled_trajectories=sampled_trajectories,
        config=config,
        train_dataset=train_dataset,
        device=device,
        layer_name=layer_name,
        positive_concept='start_walking',
        negative_concept='stays_stopped',
        max_samples_per_class=args.max_samples
    )

    # Split into train and validation sets
    LOG.info("Splitting data into train/val sets...")
    train_states, val_states, train_labels, val_labels = train_test_split(
        hidden_states, labels, test_size=0.1, random_state=11111, stratify=labels
    )

    LOG.info(f"Training set: {train_states.shape}, "
            f"{np.sum(train_labels == 1)} positive, {np.sum(train_labels == 0)} negative")
    LOG.info(f"Validation set: {val_states.shape}, "
            f"{np.sum(val_labels == 1)} positive, {np.sum(val_labels == 0)} negative")

    # Create datasets and dataloaders
    LOG.info("Creating datasets...")
    try:
        train_dataset_clf = ResidualStreamDataset(train_states, train_labels)
        val_dataset_clf = ResidualStreamDataset(val_states, val_labels)
        LOG.info(f"✓ Datasets created successfully")
    except Exception as e:
        LOG.error(f"Failed to create datasets: {e}")
        raise

    LOG.info(f"Creating dataloaders (batch_size={args.batch_size})...")
    try:
        train_loader = DataLoader(train_dataset_clf, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_clf, batch_size=args.batch_size, shuffle=False)
        LOG.info(f"✓ Dataloaders created successfully")
    except Exception as e:
        LOG.error(f"Failed to create dataloaders: {e}")
        raise

    # Initialize classifier
    seq_len = hidden_states.shape[2] if hidden_states.ndim == 4 else hidden_states.shape[1]
    hidden_dim = hidden_states.shape[3] if hidden_states.ndim == 4 else hidden_states.shape[2]

    LOG.info(f"Initializing classifier (seq_len={seq_len}, hidden_dim={hidden_dim}, pooling={args.pooling}, probe_type={args.probe_type}, hidden_dims={args.hidden_dim1}/{args.hidden_dim2})...")
    clf_model = ResidualStreamClassifier(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        pooling=args.pooling,
        probe_type=args.probe_type,
        mlp_hidden_dim1=args.hidden_dim1,
        mlp_hidden_dim2=args.hidden_dim2
    )

    num_params = sum(p.numel() for p in clf_model.parameters())
    LOG.info(f"Total parameters: {num_params:,}")

    # Train
    LOG.info("Starting training...")
    history = train_classifier(
        model=clf_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )

    # Evaluate
    LOG.info("Evaluating on validation set...")
    predictions, true_labels = evaluate_model(clf_model, val_loader, device)

    # Print results
    cm = confusion_matrix(true_labels, predictions)
    LOG.info("\nConfusion Matrix:")
    LOG.info(str(cm))

    LOG.info("\nClassification Report:")
    LOG.info(classification_report(true_labels, predictions,
                                   target_names=['stays_stopped', 'start_walking']))

    best_val_acc = max(history['val_acc'])
    final_val_acc = history['val_acc'][-1]
    LOG.info(f"\nBest validation accuracy: {best_val_acc:.4f}")
    LOG.info(f"Final validation accuracy: {final_val_acc:.4f}")

    # Save model and results
    output_prefix = f"residual_classifier_layer{args.layer}_{args.pooling}_{args.probe_type}_{args.sampling_method}"
    # Add lr and hidden dims to filename if non-default
    if args.lr != 1e-3:
        output_prefix += f"_lr{args.lr}"
    if args.hidden_dim1 != 256 or args.hidden_dim2 != 128:
        output_prefix += f"_h{args.hidden_dim1}_{args.hidden_dim2}"

    # Save model
    model_path = os.path.join(args.output_dir, f"{output_prefix}.pth")
    torch.save({
        'model_state_dict': clf_model.state_dict(),
        'layer_name': layer_name,
        'seq_len': seq_len,
        'hidden_dim': hidden_dim,
        'pooling': args.pooling,
        'probe_type': args.probe_type,
        'sampling_method': args.sampling_method,
        'history': history,
        'args': vars(args)
    }, model_path)
    LOG.info(f"Model saved to {model_path}")

    # Save training history plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training and Validation Loss\n({layer_name}, pool={args.pooling}, probe={args.probe_type}, sample={args.sampling_method})')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Training and Validation Accuracy\n({layer_name}, pool={args.pooling}, probe={args.probe_type}, sample={args.sampling_method})')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    history_plot_path = os.path.join(args.output_dir, f"{output_prefix}_history.png")
    plt.savefig(history_plot_path, dpi=150)
    LOG.info(f"Training history plot saved to {history_plot_path}")
    plt.close()

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['stays_stopped', 'start_walking'],
                yticklabels=['stays_stopped', 'start_walking'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {layer_name}\nPool: {args.pooling}, Probe: {args.probe_type}, Sample: {args.sampling_method}')
    plt.tight_layout()
    cm_plot_path = os.path.join(args.output_dir, f"{output_prefix}_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    LOG.info(f"Confusion matrix plot saved to {cm_plot_path}")
    plt.close()

    LOG.info("\nTraining complete!")


if __name__ == '__main__':
    main()
