#!/usr/bin/env python
"""
Train a binary classifier on attention maps from a transformer model.

This script extracts attention patterns from a specified layer and trains
a CNN to classify between two behavioral concepts (e.g., start_walking vs stays_stopped).
"""

import os
import sys
import argparse
import logging
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apf.io import read_config
from apf.models import initialize_model
from flyllm.prepare import init_flyllm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class AttentionCNN(nn.Module):
    """CNN classifier for attention patterns.

    Takes attention maps from a single layer (num_heads, seq_len, seq_len)
    and predicts a binary label.
    """
    def __init__(self, num_heads=8, seq_len=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_heads, 16, kernel_size=5, stride=2, padding=2),  # 512�256
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256�128

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128�64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64�32

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32�16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 16�4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Binary classification
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, num_heads, seq_len, seq_len) attention maps

        Returns:
            logits: (batch, 1) binary classification logits
        """
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class AttentionDataset(Dataset):
    """Dataset for attention maps and binary labels."""

    def __init__(self, attention_maps, labels):
        """
        Args:
            attention_maps: Array of shape (n_samples, num_heads, seq_len, seq_len)
            labels: Array of shape (n_samples,) with binary labels (0 or 1)
        """
        self.attention_maps = torch.FloatTensor(attention_maps)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)  # (n_samples, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.attention_maps[idx], self.labels[idx]


# ============================================================================
# MODEL INTERPRETER (Simplified for attention extraction)
# ============================================================================

class ModelInterpreter:
    """Extract attention weights from a trained model."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.attention_weights = {}
        self.hooks = []
        LOG.info(f"Initialized ModelInterpreter on device: {device}")

    def enable_attention_output(self):
        """Enable attention weight output for all MultiheadAttention modules."""
        import inspect

        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                original_forward = module.__class__.forward.__get__(module, module.__class__)
                module._original_forward_unhooked = original_forward

                sig = inspect.signature(original_forward)
                supports_is_causal = 'is_causal' in sig.parameters

                def make_wrapper(orig_forward, supports_causal):
                    def wrapper(query, key, value, key_padding_mask=None,
                               need_weights=True, attn_mask=None, average_attn_weights=False,
                               is_causal=False, **kwargs):
                        call_kwargs = {
                            'key_padding_mask': key_padding_mask,
                            'need_weights': True,
                            'attn_mask': attn_mask,
                            'average_attn_weights': False,
                        }
                        if supports_causal:
                            call_kwargs['is_causal'] = is_causal
                        call_kwargs.update(kwargs)
                        return orig_forward(query, key, value, **call_kwargs)
                    return wrapper

                module.forward = make_wrapper(original_forward, supports_is_causal)

    def register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        self.enable_attention_output()

        def get_attention(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        if attn_weights.ndim == 4:
                            self.attention_weights[name] = attn_weights.detach().cpu()
                        elif attn_weights.ndim == 3:
                            if attn_weights.shape[1] == attn_weights.shape[2]:
                                self.attention_weights[name] = attn_weights.detach().cpu()
                            else:
                                L, N, S = attn_weights.shape
                                num_heads = module.num_heads
                                batch_size = N // num_heads
                                attn_reshaped = attn_weights.view(L, batch_size, num_heads, S)
                                attn_reshaped = attn_reshaped.permute(1, 2, 0, 3)
                                self.attention_weights[name] = attn_reshaped.detach().cpu()
            return hook

        self.remove_hooks()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(get_attention(name))
                self.hooks.append(hook)
                LOG.info(f"Registered hook for: {name}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_activations(self):
        """Clear stored attention weights."""
        self.attention_weights = {}


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def load_sampled_trajectories(trajectories_file):
    """Load pre-sampled trajectories from pickle file."""
    LOG.info(f"Loading sampled trajectories from {trajectories_file}")
    with open(trajectories_file, 'rb') as f:
        sampled_trajectories = pickle.load(f)
    for concept, trajs in sampled_trajectories.items():
        LOG.info(f"  {concept}: {len(trajs)} trajectories")
    return sampled_trajectories


def sample_trajectories_by_concept(concept_labels, n_samples=1000):
    """Sample trajectories for each concept category."""
    samples = {
        'start_walking': [],
        'stays_stopped': [],
    }

    concept_map = {
        1: 'start_walking',
        -1: 'stays_stopped',
    }

    for concept_value, concept_name in concept_map.items():
        positions = np.argwhere(concept_labels == concept_value)
        if len(positions) == 0:
            LOG.warning(f"No positions found for {concept_name}")
            continue

        n_to_sample = min(n_samples, len(positions))
        sampled_indices = np.random.choice(len(positions), size=n_to_sample, replace=False)

        for idx in sampled_indices:
            agent_id, frame, _ = positions[idx]
            samples[concept_name].append((int(agent_id), int(frame)))

        LOG.info(f"{concept_name}: Sampled {len(samples[concept_name])} trajectories")

    return samples


def extract_attention_dataset(interpreter, sampled_trajectories, config,
                              train_dataset, device, layer_name,
                              positive_concept='start_walking',
                              negative_concept='stays_stopped',
                              max_samples_per_class=None):
    """Extract attention maps and create binary classification dataset."""
    interpreter.remove_hooks()
    interpreter.register_attention_hooks()

    attention_maps = []
    labels = []

    # Process positive examples
    LOG.info(f"Extracting positive examples ({positive_concept})...")
    positive_trajectories = sampled_trajectories[positive_concept]
    if max_samples_per_class is not None:
        positive_trajectories = positive_trajectories[:max_samples_per_class]

    for agent_id, start_frame in tqdm(positive_trajectories, desc=f"Extracting {positive_concept}"):
        sim_start_frame = start_frame - config['contextl']
        if sim_start_frame < 0:
            continue

        try:
            example_data = train_dataset.get_chunk(sim_start_frame, train_dataset.context_length, agent_id)
            input_data = example_data['input']
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)

            with torch.no_grad():
                _ = interpreter.model(input_tensor)

            if layer_name in interpreter.attention_weights:
                attn = interpreter.attention_weights[layer_name]
                attn_numpy = attn[0].numpy()

                # Check for NaN in this sample
                if np.isnan(attn_numpy).any():
                    LOG.warning(f"Skipping positive sample (agent={agent_id}, frame={start_frame}): contains NaN")
                    interpreter.clear_activations()
                    continue

                attention_maps.append(attn_numpy)
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

    for agent_id, start_frame in tqdm(negative_trajectories, desc=f"Extracting {negative_concept}"):
        sim_start_frame = start_frame - config['contextl']
        if sim_start_frame < 0:
            continue

        try:
            example_data = train_dataset.get_chunk(sim_start_frame, train_dataset.context_length, agent_id)
            input_data = example_data['input']
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)

            with torch.no_grad():
                _ = interpreter.model(input_tensor)

            if layer_name in interpreter.attention_weights:
                attn = interpreter.attention_weights[layer_name]
                attn_numpy = attn[0].numpy()

                # Check for NaN in this sample
                if np.isnan(attn_numpy).any():
                    LOG.warning(f"Skipping negative sample (agent={agent_id}, frame={start_frame}): contains NaN")
                    interpreter.clear_activations()
                    continue

                attention_maps.append(attn_numpy)
                labels.append(0)

            interpreter.clear_activations()
        except Exception as e:
            LOG.warning(f"Failed to process negative example: {e}")
            continue

    LOG.info(f"Extracted {len([l for l in labels if l == 0])} negative examples")

    attention_maps = np.array(attention_maps)
    labels = np.array(labels)

    LOG.info(f"Final dataset: {attention_maps.shape}, labels: {labels.shape}")
    LOG.info(f"Label distribution: {np.sum(labels == 1)} positive, {np.sum(labels == 0)} negative")

    # Check if we got any valid samples
    if len(attention_maps) == 0:
        LOG.error("ERROR: No valid attention maps extracted! All samples contained NaN.")
        raise ValueError("No valid attention maps extracted")

    # Validate attention data
    LOG.info("Validating attention data...")
    LOG.info(f"  Min: {np.min(attention_maps):.6f}, Max: {np.max(attention_maps):.6f}")
    LOG.info(f"  Mean: {np.mean(attention_maps):.6f}, Std: {np.std(attention_maps):.6f}")
    LOG.info(f"  Contains NaN: {np.any(np.isnan(attention_maps))}")
    LOG.info(f"  Contains Inf: {np.any(np.isinf(attention_maps))}")

    # Final check - should not happen since we filter above
    if np.any(np.isnan(attention_maps)):
        LOG.warning("WARNING: Some NaN values slipped through filtering")
    if np.any(np.isinf(attention_maps)):
        LOG.error("ERROR: Attention maps contain Inf values!")
        raise ValueError("Attention maps contain Inf values")

    return attention_maps, labels


# ============================================================================
# TRAINING
# ============================================================================

def train_attention_classifier(model, train_loader, val_loader, device,
                               num_epochs=50, lr=1e-3, patience=10, use_wandb=False,
                               clip_grad_norm=1.0, lr_scheduler_type='plateau',
                               lr_patience=5, lr_factor=0.5):
    """Train the attention classifier with gradient clipping and LR scheduling."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Setup learning rate scheduler
    scheduler = None
    if lr_scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True
        )
        LOG.info(f"Using ReduceLROnPlateau scheduler (patience={lr_patience}, factor={lr_factor})")
    elif lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        LOG.info(f"Using CosineAnnealingLR scheduler (T_max={num_epochs})")
    elif lr_scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=lr_factor
        )
        LOG.info(f"Using StepLR scheduler (step_size=10, gamma={lr_factor})")
    else:
        LOG.info("No LR scheduler")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    LOG.info(f"Training with lr={lr}, gradient clipping={clip_grad_norm}")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (batch_attn, batch_labels) in enumerate(train_loader):
            batch_attn = batch_attn.to(device)
            batch_labels = batch_labels.to(device)

            # Check for NaN in input
            if torch.isnan(batch_attn).any():
                LOG.error(f"NaN detected in input batch {batch_idx}")
                continue

            optimizer.zero_grad()
            outputs = model(batch_attn)

            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                LOG.error(f"NaN detected in model outputs at batch {batch_idx}")
                continue

            loss = criterion(outputs, batch_labels)

            # Check for NaN in loss
            if torch.isnan(loss):
                LOG.error(f"NaN loss at batch {batch_idx}")
                continue

            loss.backward()

            # Gradient clipping
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        if train_total > 0:
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
        else:
            LOG.error("No valid training batches in this epoch!")
            train_loss = float('nan')
            train_acc = 0.0

        # Check for NaN loss
        if np.isnan(train_loss):
            LOG.error(f"Training loss is NaN at epoch {epoch+1}! Stopping training.")
            break

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_attn, batch_labels in val_loader:
                batch_attn = batch_attn.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_attn)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        LOG.info(f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6f}")

        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'learning_rate': current_lr,
            })

        # Step the learning rate scheduler
        if scheduler is not None:
            if lr_scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOG.info(f"Early stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    return history


def evaluate_model(model, val_loader, device):
    """Evaluate model and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_attn, batch_labels in val_loader:
            batch_attn = batch_attn.to(device)
            outputs = model(batch_attn)
            predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(predictions.flatten())
            all_labels.extend(batch_labels.numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


def plot_training_history(history, save_path=None):
    """Plot training curves including learning rate."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Learning Rate
    if 'lr' in history and len(history['lr']) > 0:
        axes[2].plot(history['lr'], label='Learning Rate', marker='o', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True)
    else:
        axes[2].text(0.5, 0.5, 'No LR data', ha='center', va='center')
        axes[2].set_title('Learning Rate Schedule')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        LOG.info(f"Saved training plot to {save_path}")

    plt.close()


def plot_confusion_matrix(y_true, y_pred, layer_idx, save_path=None):
    """Plot confusion matrix."""
    try:
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['stays_stopped', 'start_walking'],
                    yticklabels=['stays_stopped', 'start_walking'],
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - Layer {layer_idx}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            LOG.info(f"Saved confusion matrix to {save_path}")

        plt.close()
    except ImportError:
        LOG.warning("seaborn not available, skipping confusion matrix plot")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train attention-based binary classifier')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pretrained model file')
    parser.add_argument('--trajectories', type=str, default=None,
                        help='Path to sampled trajectories pickle file (optional)')
    parser.add_argument('--attention-cache', type=str, default=None,
                        help='Path to save/load extracted attention data (optional)')
    parser.add_argument('--layer', type=int, default=9,
                        help='Transformer layer index to use (default: 9)')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Maximum samples per class (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4, reduced from 1e-3)')
    parser.add_argument('--lr-scheduler', type=str, default='plateau',
                        choices=['none', 'plateau', 'cosine', 'step'],
                        help='Learning rate scheduler (default: plateau)')
    parser.add_argument('--lr-patience', type=int, default=5,
                        help='LR scheduler patience for plateau (default: 5)')
    parser.add_argument('--lr-factor', type=float, default=0.5,
                        help='LR reduction factor for plateau/step (default: 0.5)')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0, None to disable)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results (default: current dir)')
    parser.add_argument('--seed', type=int, default=11111,
                        help='Random seed (default: 11111)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='attention-classifier',
                        help='W&B project name (default: attention-classifier)')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='W&B run name (default: auto-generated)')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize wandb if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        LOG.warning("wandb requested but not available. Install with: pip install wandb")
        use_wandb = False

    if use_wandb:
        wandb_config = {
            'layer': args.layer,
            'max_samples': args.max_samples,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'patience': args.patience,
            'seed': args.seed,
        }
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=wandb_config
        )
        LOG.info(f"Initialized wandb: {wandb.run.name}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model and data
    LOG.info("Initializing model and dataset...")
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

    res = init_flyllm(
        configfile=args.config,
        mode='test',
        loadmodelfile=args.model,
        needtraindata=True,
        overrideconfig=overrideconfig
    )

    config = res['config']
    train_dataset = res['train_dataset']
    model = res['model']
    device = res['device']

    LOG.info(f"Using device: {device}")

    # Load or sample trajectories
    if args.trajectories and os.path.exists(args.trajectories):
        sampled_trajectories = load_sampled_trajectories(args.trajectories)
    else:
        LOG.info("Sampling trajectories from dataset...")
        concept_labels = train_dataset.concepts.array
        sampled_trajectories = sample_trajectories_by_concept(
            concept_labels, n_samples=args.max_samples * 10  # Sample more, filter later
        )
        # Save for future use
        traj_path = os.path.join(args.output_dir, 'sampled_trajectories.pkl')
        with open(traj_path, 'wb') as f:
            pickle.dump(sampled_trajectories, f)
        LOG.info(f"Saved sampled trajectories to {traj_path}")

    # Initialize interpreter
    LOG.info("Setting up model interpreter...")
    interpreter = ModelInterpreter(model=model, device=device)

    # Extract or load attention data
    layer_name = f'transformer_encoder.layers.{args.layer}.self_attn'

    # Try to load from cache
    if args.attention_cache and os.path.exists(args.attention_cache):
        LOG.info(f"Loading attention data from cache: {args.attention_cache}")
        with open(args.attention_cache, 'rb') as f:
            cache_data = pickle.load(f)
        attention_maps = cache_data['attention_maps']
        labels = cache_data['labels']
        cached_layer = cache_data.get('layer_name', 'unknown')
        LOG.info(f"Loaded {len(labels)} samples from cache (layer: {cached_layer})")

        # Warn if layer mismatch
        if cached_layer != layer_name:
            LOG.warning(f"Cache layer ({cached_layer}) != requested layer ({layer_name})")
    else:
        # Extract attention data
        LOG.info(f"Extracting attention from {layer_name}...")
        attention_maps, labels = extract_attention_dataset(
            interpreter=interpreter,
            sampled_trajectories=sampled_trajectories,
            config=config,
            train_dataset=train_dataset,
            device=device,
            layer_name=layer_name,
            max_samples_per_class=args.max_samples
        )

        # Save to cache if requested
        if args.attention_cache:
            LOG.info(f"Saving attention data to cache: {args.attention_cache}")
            cache_data = {
                'attention_maps': attention_maps,
                'labels': labels,
                'layer_name': layer_name,
                'max_samples': args.max_samples,
            }
            with open(args.attention_cache, 'wb') as f:
                pickle.dump(cache_data, f)
            LOG.info(f"Saved {len(labels)} samples to cache")

    # Split data
    LOG.info("Splitting data into train/val sets...")
    train_attn, val_attn, train_labels, val_labels = train_test_split(
        attention_maps, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )

    LOG.info(f"Training set: {train_attn.shape}, "
            f"{np.sum(train_labels == 1)} positive, {np.sum(train_labels == 0)} negative")
    LOG.info(f"Validation set: {val_attn.shape}, "
            f"{np.sum(val_labels == 1)} positive, {np.sum(val_labels == 0)} negative")

    # Create datasets and loaders
    train_dataset_clf = AttentionDataset(train_attn, train_labels)
    val_dataset_clf = AttentionDataset(val_attn, val_labels)

    train_loader = DataLoader(train_dataset_clf, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_clf, batch_size=args.batch_size, shuffle=False)

    # Inspect first batch for debugging
    LOG.info("Inspecting first training batch...")
    first_batch_attn, first_batch_labels = next(iter(train_loader))
    LOG.info(f"  Batch shape: {first_batch_attn.shape}")
    LOG.info(f"  Batch min: {first_batch_attn.min():.6f}, max: {first_batch_attn.max():.6f}")
    LOG.info(f"  Batch mean: {first_batch_attn.mean():.6f}, std: {first_batch_attn.std():.6f}")
    LOG.info(f"  Labels: {first_batch_labels.flatten()[:10].tolist()}...")

    # Initialize classifier
    num_heads = attention_maps.shape[1]
    seq_len = attention_maps.shape[2]
    LOG.info(f"Attention shape: num_heads={num_heads}, seq_len={seq_len}")

    clf_model = AttentionCNN(num_heads=num_heads, seq_len=seq_len)
    total_params = sum(p.numel() for p in clf_model.parameters())
    LOG.info(f"Total parameters: {total_params:,}")

    # Train
    LOG.info("Starting training...")
    clip_grad_norm = args.clip_grad if args.clip_grad > 0 else None
    history = train_attention_classifier(
        model=clf_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        use_wandb=use_wandb,
        clip_grad_norm=clip_grad_norm,
        lr_scheduler_type=args.lr_scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor
    )

    # Evaluate
    LOG.info("Evaluating model...")
    y_pred, y_true = evaluate_model(clf_model, val_loader, device)

    # Print results
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred,
                                       target_names=['stays_stopped', 'start_walking'],
                                       output_dict=True)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['stays_stopped', 'start_walking']))

    # Save results
    LOG.info("Saving results...")

    # Save model
    model_path = os.path.join(args.output_dir, f'attention_classifier_layer{args.layer}.pth')
    torch.save({
        'model_state_dict': clf_model.state_dict(),
        'layer_name': layer_name,
        'num_heads': num_heads,
        'seq_len': seq_len,
        'history': history,
        'args': vars(args)
    }, model_path)
    LOG.info(f"Saved model to {model_path}")

    # Save plots
    plot_path = os.path.join(args.output_dir, f'training_history_layer{args.layer}.png')
    plot_training_history(history, save_path=plot_path)

    cm_path = os.path.join(args.output_dir, f'confusion_matrix_layer{args.layer}.png')
    plot_confusion_matrix(y_true, y_pred, args.layer, save_path=cm_path)

    # Save history
    history_path = os.path.join(args.output_dir, f'history_layer{args.layer}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    LOG.info(f"Saved training history to {history_path}")

    # Log to wandb
    if use_wandb:
        # Log final metrics
        wandb.log({
            'final/accuracy': report_dict['accuracy'],
            'final/stays_stopped_precision': report_dict['stays_stopped']['precision'],
            'final/stays_stopped_recall': report_dict['stays_stopped']['recall'],
            'final/stays_stopped_f1': report_dict['stays_stopped']['f1-score'],
            'final/start_walking_precision': report_dict['start_walking']['precision'],
            'final/start_walking_recall': report_dict['start_walking']['recall'],
            'final/start_walking_f1': report_dict['start_walking']['f1-score'],
        })

        # Log confusion matrix
        wandb.log({
            'confusion_matrix': wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true.astype(int),
                preds=y_pred.astype(int),
                class_names=['stays_stopped', 'start_walking']
            )
        })

        # Log plots as images
        if os.path.exists(plot_path):
            wandb.log({'training_history': wandb.Image(plot_path)})
        if os.path.exists(cm_path):
            wandb.log({'confusion_matrix_plot': wandb.Image(cm_path)})

        # Save model as artifact
        artifact = wandb.Artifact(
            name=f'attention_classifier_layer{args.layer}',
            type='model',
            description=f'Attention classifier trained on layer {args.layer}'
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

        wandb.finish()
        LOG.info("Logged results to wandb")

    print("\n" + "="*60)
    print(f"Training complete! Results saved to {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
