# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: flyllm
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
    
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle
from collections import defaultdict

from apf.io import read_config
from apf.training import train
from apf.utils import function_args_from_config
from apf.simulation import simulate
from apf.models import initialize_model

from flyllm.config import DEFAULTCONFIGFILE, posenames
from flyllm.features import featglobal, get_sensory_feature_idx
from flyllm.simulation import animate_pose
from flyllm.plotting import plot_arena
from flyllm.prepare import init_flyllm
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
configfile = "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_optimalbinning_20251113.json"
mode = 'test' # can toggle to 'train'/'test'
pretrained_modelfile = os.path.join('/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/notebooks/flyllm_models',
                                    'flypredvel_20251007_20251114T194024_bestepoch200.pth')
restartmodelfile = None
debug_uselessdata=False

# Enable concept computation
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
# %%
if mode == 'train':
    loadmodelfile = None
else:
    loadmodelfile = pretrained_modelfile
    
res = init_flyllm(configfile=configfile,mode=mode,restartmodelfile=restartmodelfile,
                loadmodelfile=loadmodelfile,debug_uselessdata=debug_uselessdata,
                needtraindata=True,overrideconfig=overrideconfig)

# %%
# unpack the results
config = res['config']
if 'train_dataset' in res:
    train_dataset = res['train_dataset']
if 'train_dataloader' in res:
    train_dataloader = res['train_dataloader']
if 'train_data' in res:
    flyids = res['train_data']['flyids']
    track = res['train_data']['track']
    pose = res['train_data']['pose']
    velocity = res['train_data']['velocity']
    sensory = res['train_data']['sensory']
if 'val_dataset' in res:
    val_dataset = res['val_dataset']
if 'val_dataloader' in res:
    val_dataloader = res['val_dataloader']
criterion = res['criterion']
model = res['model']
optimizer = res['optimizer']
lr_scheduler = res['lr_scheduler']
loss_epoch = res['loss_epoch']
start_epoch = res['epoch']
modeltype_str = res['modeltype_str']
device = res['device']
savetime = res['model_savetime']

# %%
print(track.array.shape,pose.array.shape,velocity.array.shape,sensory.array.shape)

# %%
# Access concepts from dataset.concepts (not dataset.labels)
if train_dataset.concepts is not None:
    concept_data = train_dataset.concepts
    concept_labels = concept_data.array
    print(f"Concepts shape: {concept_labels.shape}")
    print(f"Concepts dtype: {concept_labels.dtype}")

    # Count labels
    labels_flat = concept_labels.flatten()
    valid_mask = ~np.isnan(labels_flat)
    labels_valid = labels_flat[valid_mask]

    print(f"\nLabel counts:")
    print(f"  Start walking (+1): {np.sum(labels_valid == 1)}")
    print(f"  Neutral (0): {np.sum(labels_valid == 0)}")
    print(f"  Stays stopped (-1): {np.sum(labels_valid == -1)}")
    print(f"  Invalid (NaN): {np.sum(~valid_mask)}")
else:
    print("Concepts not computed")
# Sample trajectories based on concept labels
def sample_trajectories_by_concept(concept_labels, n_samples=5):
    """
    Randomly sample start positions (agent_id, start_frame) for each concept category.

    Args:
        concept_labels: Array of shape (n_agents, n_timepoints, 1)
        n_samples: Number of samples per concept category

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
        0: 'neutral'  # Uncomment if neutral sampling is desired
    }

    for concept_value, concept_name in concept_map.items():
        # Find all positions where this concept occurs
        positions = np.argwhere(concept_labels == concept_value)

        if len(positions) == 0:
            print(f"Warning: No positions found for {concept_name} ({concept_value})")
            continue

        # Randomly sample n_samples positions
        n_to_sample = min(n_samples, len(positions))
        sampled_indices = np.random.choice(len(positions), size=n_to_sample, replace=False)

        # Convert to (agent_id, start_frame) tuples
        for idx in sampled_indices:
            agent_id, frame, _ = positions[idx]
            samples[concept_name].append((int(agent_id), int(frame)))

        print(f"{concept_name} ({concept_value}): Sampled {len(samples[concept_name])} trajectories")

    return samples

# Sample trajectories for each concept
np.random.seed(11111)  # For reproducibility
sampled_trajectories = sample_trajectories_by_concept(
    concept_labels=concept_labels,
    n_samples=100  # Sample 100 trajectories per concept
)

# Display sampled trajectories
print("\n=== Sampled Trajectories ===")
for concept_name, trajectories in sampled_trajectories.items():
    print(f"\n{concept_name}:")
    for i, (agent_id, start_frame) in enumerate(trajectories):
        print(f"  Sample {i+1}: agent_id={agent_id}, start_frame={start_frame}")
# simulate single trajectory
track_len = 150 + config['contextl'] + 1  # Shorter for faster simulation
burn_in = config['contextl']

j = 4
concept_name = 'start_walking'

agent_id, start_frame = sampled_trajectories[concept_name][j]
sim_start_frame = start_frame - config['contextl']
example_data = train_dataset.get_chunk(sim_start_frame, train_dataset.context_length, agent_id)
input_data = example_data['input']
agent_identity = np.unique(flyids[sim_start_frame:sim_start_frame + track_len, agent_id])

# %%
try:
    gt_track, pred_track = simulate(
        dataset=train_dataset,
        model=model,
        track=track,
        pose=pose,
        identities=flyids,
        track_len=track_len,
        burn_in=burn_in,
        max_contextl=config['contextl'],
        agent_ids=[agent_id],
        start_frame=sim_start_frame,  # Start earlier to have context
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f}s")
    
    # Compute velocity from track positions using center keypoint
    gt_track_agent = gt_track[agent_id, :, :, :]  # (track_len, 2, n_keypoints)
    center_pos = gt_track_agent[:,:,7]  # (track_len, 2) - keypoint 7 is center of body
    gt_velocity = np.sqrt(np.sum(np.diff(center_pos, axis=0)**2, axis=1)) * 150.0  # (track_len-1,) in pixels/sec

    # Extract concept labels for this trajectory from the dataset
    # Use train_dataset.concepts.array directly to avoid variable reassignment issues
    dataset_concept_labels = train_dataset.concepts.array
    end_frame = min(sim_start_frame + track_len, dataset_concept_labels.shape[1])
    labels_slice = dataset_concept_labels[agent_id, sim_start_frame:end_frame]
    if labels_slice.ndim > 1:
        labels_slice = labels_slice[:, 0]  # Remove last dimension if present
    
    savedir = "concept_animations"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    modelname = os.path.split(pretrained_modelfile)[-1].replace('.pth', '')
    # Create animation filename
    savevidfile = os.path.join(
        savedir,
        f"{concept_name}_sample{j+1}_agent{agent_id}_frame{start_frame}_{modelname}.gif"
    )

    ani = animate_pose(
                    {'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()},
                    focusflies=[agent_id],
                    savevidfile=savevidfile,
                    contextl=config['contextl']
                )
except Exception as e:
    print(f"  Error: {e}")
# Simulate for each concept category
simulation_results = {}

track_len = 150 + config['contextl'] + 1  # Shorter for faster simulation
burn_in = config['contextl']

for concept_name, trajectories in sampled_trajectories.items():
    print(f"\n=== Simulating {concept_name} ===")
    simulation_results[concept_name] = []

    for i, (agent_id, start_frame) in enumerate(trajectories):
        # start_frame is where the concept label occurs
        # We need to start simulation earlier to have context before the concept event
        concept_frame = start_frame
        sim_start_frame = concept_frame - config['contextl']

        print(f"Simulating sample {i+1}: agent_id={agent_id}, concept_frame={concept_frame}, sim_start={sim_start_frame}")

        # Make sure we have enough context before the concept frame
        if sim_start_frame < 0:
            print(f"  Skipping: sim_start_frame {sim_start_frame} < 0")
            continue

        # Check for consistent identity (skip if agent switches identities)
        agent_identity = np.unique(flyids[sim_start_frame:sim_start_frame + track_len, agent_id])
        if len(agent_identity) != 1:
            print(f"  Skipping: Agent switches identities {agent_identity}")
            continue

        t0 = time.time()
        try:
            gt_track, pred_track = simulate(
                dataset=train_dataset,
                model=model,
                track=track,
                pose=pose,
                identities=flyids,
                track_len=track_len,
                burn_in=burn_in,
                max_contextl=config['contextl'],
                agent_ids=[agent_id],
                start_frame=sim_start_frame,  # Start earlier to have context
            )
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.2f}s")
            
            # Compute velocity from track positions using center keypoint
            gt_track_agent = gt_track[agent_id, :, :, :]  # (track_len, 2, n_keypoints)
            center_pos = gt_track_agent[:,:,7]  # (track_len, 2) - keypoint 7 is center of body
            gt_velocity = np.sqrt(np.sum(np.diff(center_pos, axis=0)**2, axis=1)) * 150.0  # (track_len-1,) in pixels/sec

            # Extract concept labels for this trajectory from the dataset
            # Use train_dataset.concepts.array directly to avoid variable reassignment issues
            dataset_concept_labels = train_dataset.concepts.array
            end_frame = min(sim_start_frame + track_len, dataset_concept_labels.shape[1])
            labels_slice = dataset_concept_labels[agent_id, sim_start_frame:end_frame]
            if labels_slice.ndim > 1:
                labels_slice = labels_slice[:, 0]  # Remove last dimension if present

            simulation_results[concept_name].append({
                'agent_id': agent_id,
                'concept_frame': concept_frame,  # Original sampled frame with concept label
                'sim_start_frame': sim_start_frame,  # Actual simulation start (concept_frame - contextl)
                'gt_track': gt_track,
                'pred_track': pred_track,
                'gt_velocity': gt_velocity,
                'concept_labels': labels_slice,
            })
        except Exception as e:
            print(f"  Error: {e}")
            continue

# %%
# Visualize simulation results for each concept
fig, axes = plt.subplots(len(sampled_trajectories), 2, figsize=(15, 5 * len(sampled_trajectories)))
if len(sampled_trajectories) == 1:
    axes = axes.reshape(1, -1)

for row_idx, (concept_name, results) in enumerate(simulation_results.items()):
    ax_gt = axes[row_idx, 0]
    ax_pred = axes[row_idx, 1]

    ax_gt.set_title(f'{concept_name} - Ground Truth')
    ax_pred.set_title(f'{concept_name} - Predicted')

    plot_arena(ax=ax_gt)
    plot_arena(ax=ax_pred)

    first_frame = config['contextl']

    for result in results:
        gt_track = result['gt_track']
        pred_track = result['pred_track']
        agent_id = result['agent_id']

        # Plot ground truth
        x, y = gt_track[agent_id, first_frame:, :, 7].T
        ax_gt.plot(x, y, '.', markersize=2, alpha=0.6)
        ax_gt.axis('equal')

        # Plot prediction
        x, y = pred_track[agent_id, first_frame:, :, 7].T
        ax_pred.plot(x, y, '.', markersize=2, alpha=0.6)
        ax_pred.axis('equal')

    ax_gt.legend()
    ax_pred.legend()

plt.tight_layout()
plt.show()

# %%
# Save animations for each concept trajectory
savedir = "concept_animations"
if not os.path.exists(savedir):
    os.makedirs(savedir)

modelname = os.path.split(pretrained_modelfile)[-1].replace('.pth', '')
    
concept_name = 'stays_stopped'  # Change as needed
results = simulation_results[concept_name]

i = 38  # Change index as needed
result = results[i]
gt_track = result['gt_track']
pred_track = result['pred_track']
agent_id = result['agent_id']
concept_frame = result['concept_frame']
    
# Create animation filename
savevidfile = os.path.join(
    savedir,
    f"{concept_name}_sample{i+1}_agent{agent_id}_frame{concept_frame}_{modelname}.gif"
)

ani = animate_pose(
                {'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()},
                focusflies=[agent_id],
                savevidfile=savevidfile,
                contextl=config['contextl']
            )

print(f"Saved animation to {savevidfile}")


# %%
class ModelInterpreter:
    """Interpret concepts in a trained model."""

    def __init__(self, model, device='cpu', sampled_trajectories=None, dataset=None):
        """
        Args:
            model: Trained model to interpret
            device: Device to run on
            sampled_trajectories: Dict of concept_name -> list of (agent_id, start_frame) tuples
            dataset: Dataset object to extract examples from
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Store sampled trajectories
        self.sampled_trajectories = sampled_trajectories
        self.dataset = dataset

        # Storage for activations and attention weights
        self.activations = {}
        self.attention_weights = {}
        self.hooks = []

        LOG.info(f"Initialized ModelInterpreter on device: {device}")
        LOG.info(f"Model type: {type(model)}")
        if sampled_trajectories is not None:
            LOG.info(f"Loaded sampled trajectories for concepts: {list(sampled_trajectories.keys())}")

    def register_activation_hooks(self, layer_names=None):
        """Register hooks to capture activations from specific layers.

        Args:
            layer_names: List of layer names to hook. If None, hooks all layers.
        """
        def get_activation(name):
            def hook(model, input, output):
                # Store activation (detach to avoid memory issues)
                if isinstance(output, dict):
                    # For dict outputs, store the dict (don't process further)
                    # These are usually from the final layer
                    return
                elif isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook

        # Remove existing hooks
        self.remove_hooks()

        # Register new hooks
        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

        LOG.info(f"Registered {len(self.hooks)} activation hooks")

    def restore_attention_forward(self):
        """Restore original forward methods for all MultiheadAttention modules.

        Call this to undo the modifications made by enable_attention_output().
        """
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                if hasattr(module, '_original_forward_unhooked'):
                    module.forward = module._original_forward_unhooked
                    delattr(module, '_original_forward_unhooked')
                    LOG.info(f"Restored original forward for: {name}")

    def enable_attention_output(self):
        """Enable attention weight output for all MultiheadAttention modules.

        This modifies the model to return attention weights during forward passes.
        Call this before registering hooks.
        """
        import inspect

        # First, restore any existing wrappers
        self.restore_attention_forward()

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                # Get the true original forward (should be clean now)
                # Use the bound method from the class to get a truly clean version
                original_forward = module.__class__.forward.__get__(module, module.__class__)

                # Save it
                module._original_forward_unhooked = original_forward

                # Check what parameters it accepts
                sig = inspect.signature(original_forward)
                supports_is_causal = 'is_causal' in sig.parameters

                def make_wrapper(orig_forward, supports_causal):
                    def wrapper(query, key, value, key_padding_mask=None,
                               need_weights=True, attn_mask=None, average_attn_weights=False,
                               is_causal=True, **kwargs):
                        # Build kwargs for original forward
                        call_kwargs = {
                            'key_padding_mask': key_padding_mask,
                            'need_weights': True,  # Force to True
                            'attn_mask': attn_mask,
                            'average_attn_weights': False,  # Force to False for per-head attention
                        }

                        # Only pass is_causal if supported
                        if supports_causal:
                            call_kwargs['is_causal'] = is_causal

                        # Add any other kwargs
                        call_kwargs.update(kwargs)

                        return orig_forward(query, key, value, **call_kwargs)
                    return wrapper

                module.forward = make_wrapper(original_forward, supports_is_causal)
                LOG.info(f"Enabled attention output for: {name} (is_causal support: {supports_is_causal})")

    def register_residual_stream_hooks(self):
        """Register hooks to capture residual stream hidden states at specific points.

        Hooks are registered at:
        1. pos_encoder output → residual stream at layer 0 (input to first transformer layer)
        2. transformer_encoder.layers.i output → residual stream after layer i

        This captures the hidden states flowing through the residual connections.
        """
        def get_residual_activation(name):
            def hook(model, input, output):
                # Store the output (this is the residual stream state)
                if isinstance(output, tuple):
                    # Some layers return (output, extra_info)
                    self.activations[name] = output[0].detach().cpu()
                elif isinstance(output, dict):
                    # Skip dict outputs
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
            # Match exactly 'transformer_encoder.layers.{digit}' without submodules
            if name.startswith('transformer_encoder.layers.'):
                parts = name.split('.')
                # Should be exactly ['transformer_encoder', 'layers', '{layer_num}']
                if len(parts) == 3 and parts[-1].isdigit():
                    layer_num = int(parts[-1])
                    # This output is the residual stream after layer_num (i.e., input to layer_num+1)
                    hook_name = f'residual_stream_layer_{layer_num + 1}'
                    hook = module.register_forward_hook(get_residual_activation(hook_name))
                    self.hooks.append(hook)
                    LOG.info(f"Registered residual stream hook after transformer layer {layer_num}")

        LOG.info(f"Registered {len(self.hooks)} residual stream hooks total")

    def register_attention_hooks(self, enable_output=True):
        """Register hooks to capture attention weights from transformer layers.

        Args:
            enable_output: If True, automatically calls enable_attention_output() first

        For PyTorch's nn.MultiheadAttention, we need to enable need_weights and
        average_attn_weights=False to get per-head attention.
        """
        # First, enable attention output if requested
        if enable_output:
            self.enable_attention_output()

        def get_attention(name):
            def hook(module, input, output):
                # For MultiheadAttention: output is (attn_output, attn_weights)
                # With average_attn_weights=False, attn_weights shape is already:
                #   (batch_size, num_heads, seq_len, seq_len)
                # With average_attn_weights=True, it's:
                #   (batch_size, seq_len, seq_len)
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        # Check the shape to see what format we got
                        if attn_weights.ndim == 4:
                            # Already in (batch, num_heads, seq_len, seq_len) format
                            self.attention_weights[name] = attn_weights.detach().cpu()
                        elif attn_weights.ndim == 3:
                            # Could be (batch, seq_len, seq_len) averaged
                            # or (seq_len, batch*num_heads, seq_len) in older PyTorch
                            # Try to determine which one
                            if attn_weights.shape[1] == attn_weights.shape[2]:
                                # Likely (batch, seq_len, seq_len)
                                self.attention_weights[name] = attn_weights.detach().cpu()
                            else:
                                # Older format: (L, N, S) where N = batch * num_heads
                                L, N, S = attn_weights.shape
                                num_heads = module.num_heads
                                batch_size = N // num_heads
                                # Reshape to (batch, num_heads, L, S)
                                attn_reshaped = attn_weights.view(L, batch_size, num_heads, S)
                                attn_reshaped = attn_reshaped.permute(1, 2, 0, 3)
                                self.attention_weights[name] = attn_reshaped.detach().cpu()
                        else:
                            LOG.warning(f"Unexpected attention shape for {name}: {attn_weights.shape}")
            return hook

        # Remove existing hooks first
        self.remove_hooks()

        # Find MultiheadAttention layers
        attention_layers_found = 0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                hook = module.register_forward_hook(get_attention(name))
                self.hooks.append(hook)
                attention_layers_found += 1
                LOG.info(f"Registered hook for: {name}")

        if attention_layers_found == 0:
            LOG.warning("No MultiheadAttention layers found! Trying broader search...")
            # Fallback: search for any attention-related module
            for name, module in self.model.named_modules():
                if any(x in name.lower() for x in ['attention', 'attn', 'self_attn']):
                    hook = module.register_forward_hook(get_attention(name))
                    self.hooks.append(hook)
                    LOG.info(f"Registered hook for: {name}")

        LOG.info(f"Registered {len(self.hooks)} attention hooks total")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def remove_all_hooks_from_model(self):
        """Remove ALL hooks from the model, including ones not registered by this interpreter.

        This is a nuclear option to clean up any hooks that might have been left behind.
        """
        for name, module in self.model.named_modules():
            # Clear forward hooks
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            # Clear backward hooks (just in case)
            module._backward_hooks.clear()
            module._backward_pre_hooks.clear()

        LOG.info("Removed all hooks from model")
        self.hooks = []

    def clear_activations(self):
        """Clear stored activations and attention weights."""
        self.activations = {}
        self.attention_weights = {}

    def forward_with_hooks(self, input_data, src_mask=None):
        """Run forward pass and collect activations.

        Args:
            input_data: Input tensor or dict
            src_mask: Optional source mask

        Returns:
            model_output: Model output
        """
        self.clear_activations()

        with torch.no_grad():
            if src_mask is not None:
                output = self.model(input_data, mask=src_mask, is_causal=True)
            else:
                output = self.model(input_data, is_causal=True)

        return output

    def extract_concept_activations(self, examples, dataset, src_mask=None, batch_size=32):
        """Extract activations for a set of concept examples.

        Args:
            examples: List of example indices, (agent_id, start_frame) tuples, or data dicts
            dataset: Dataset to get examples from
            src_mask: Optional source mask
            batch_size: Number of examples to process in parallel (default: 32)

        Returns:
            concept_activations: Dict mapping layer names to lists of activations
        """
        concept_activations = defaultdict(list)

        LOG.info(f"Extracting activations for {len(examples)} examples with batch_size={batch_size}...")

        # Process examples in batches
        for batch_start in range(0, len(examples), batch_size):
            batch_end = min(batch_start + batch_size, len(examples))
            batch_examples = examples[batch_start:batch_end]

            LOG.info(f"Processing batch {batch_start}-{batch_end}/{len(examples)}")

            # Collect batch data
            batch_inputs = []
            for example_idx in batch_examples:
                # Get example from dataset
                if isinstance(example_idx, int):
                    # Integer index - use dataset's __getitem__
                    example_data = dataset[example_idx]
                elif isinstance(example_idx, tuple) and len(example_idx) == 2:
                    # (agent_id, start_frame) tuple - use get_chunk directly
                    agent_id, start_frame = example_idx
                    sim_start_frame = start_frame - config['contextl']

                    # Assertions to validate assumptions
                    assert sim_start_frame >= 0, f"Not enough context: sim_start_frame={sim_start_frame} < 0"

                    example_data = dataset.get_chunk(sim_start_frame, dataset.context_length, agent_id)

                    # Validate that we got data back
                    assert example_data is not None, f"get_chunk returned None for agent_id={agent_id}, frame={sim_start_frame}"
                else:
                    # Assume it's already a data dict
                    example_data = example_idx

                # Extract input data
                if isinstance(example_data, dict):
                    # Extract the 'input' key from the data dictionary
                    input_data = example_data['input']
                else:
                    input_data = example_data

                batch_inputs.append(input_data)

            # Convert batch to tensor
            batch_tensor = torch.FloatTensor(np.array(batch_inputs)).to(self.device)

            # Forward pass on batch
            _ = self.forward_with_hooks(batch_tensor, src_mask)

            # Collect activations from this batch
            for layer_name, activation in self.activations.items():
                if activation.shape[-1] == 2048 or activation.shape[-1] == 149 or activation.shape[-1] == 512:
                    # activation shape: (batch_size, seq_len, hidden_dim)
                    # Split batch into individual examples
                    for i in range(activation.shape[0]):
                        concept_activations[layer_name].append(activation[i:i+1].numpy())

        # Convert lists to arrays
        for layer_name in concept_activations:
            concept_activations[layer_name] = np.array(concept_activations[layer_name])

        LOG.info(f"Extracted activations from {len(concept_activations)} layers")
        return dict(concept_activations)

    def compute_activation_statistics(self, activations_dict):
        """Compute statistics for each layer's activations.

        Args:
            activations_dict: Dict mapping layer names to activation arrays

        Returns:
            stats: Dict of statistics per layer
        """
        stats = {}

        for layer_name, activations in activations_dict.items():
            # activations shape: (n_examples, batch, seq_len, hidden_dim) or similar
            stats[layer_name] = {
                'mean': np.mean(activations, axis=0),
                'std': np.std(activations, axis=0),
                'max': np.max(activations, axis=0),
                'min': np.min(activations, axis=0),
            }

        return stats

    def compare_concept_vs_baseline(self, concept_activations, baseline_activations):
        """Compare concept activations to baseline to find distinctive features.

        Args:
            concept_activations: Activations for concept examples
            baseline_activations: Activations for baseline/random examples

        Returns:
            comparison: Dict with comparison metrics per layer
        """
        comparison = {}

        for layer_name in concept_activations.keys():
            if layer_name not in baseline_activations:
                continue

            concept_acts = concept_activations[layer_name]
            baseline_acts = baseline_activations[layer_name]

            # Compute means
            concept_mean = np.mean(concept_acts, axis=0)
            baseline_mean = np.mean(baseline_acts, axis=0)

            # Compute difference
            diff = concept_mean - baseline_mean

            # Compute effect size (Cohen's d)
            concept_std = np.std(concept_acts, axis=0)
            baseline_std = np.std(baseline_acts, axis=0)
            pooled_std = np.sqrt((concept_std**2 + baseline_std**2) / 2)
            effect_size = diff / (pooled_std + 1e-8)

            comparison[layer_name] = {
                'concept_mean': concept_mean,
                'baseline_mean': baseline_mean,
                'difference': diff,
                'effect_size': effect_size,
                'concept_std': concept_std,
                'baseline_std': baseline_std,
            }

        return comparison

    def find_concept_neurons(self, comparison, threshold=1.0):
        """Find neurons that are most selective for the concept.

        Args:
            comparison: Output from compare_concept_vs_baseline
            threshold: Effect size threshold for selectivity

        Returns:
            concept_neurons: Dict mapping layer names to selective neuron indices
        """
        concept_neurons = {}

        for layer_name, comp in comparison.items():
            effect_size = comp['effect_size']

            # Flatten to find selective neurons across all positions
            flat_effect = effect_size.reshape(-1, effect_size.shape[-1])

            # Find neurons with high effect size
            max_effect_per_neuron = np.max(np.abs(flat_effect), axis=0)
            selective_neurons = np.where(max_effect_per_neuron > threshold)[0]

            concept_neurons[layer_name] = {
                'neuron_indices': selective_neurons,
                'max_effect_sizes': max_effect_per_neuron[selective_neurons],
            }

            LOG.info(f"{layer_name}: Found {len(selective_neurons)} concept-selective neurons")

        return concept_neurons

    def analyze_temporal_dynamics(self, activations_dict, onset_frame):
        """Analyze how activations change around the concept onset.

        Args:
            activations_dict: Activations for concept examples
            onset_frame: Frame index where concept occurs

        Returns:
            temporal_analysis: Time-resolved activation patterns
        """
        temporal_analysis = {}

        for layer_name, activations in activations_dict.items():
            # Average over examples and hidden dimensions
            # Shape: (n_examples, batch, seq_len, hidden_dim)
            if activations.ndim >= 3:
                # Average over batch and hidden dims, keep time
                time_course = np.mean(activations, axis=(0, -1))  # Shape: (seq_len,)

                temporal_analysis[layer_name] = {
                    'time_course': time_course,
                    'onset_frame': onset_frame,
                }

        return temporal_analysis

    def trajectory_to_dataset_index(self, agent_id, start_frame):
        """Convert (agent_id, start_frame) to a format usable by extract_concept_activations.

        Args:
            agent_id: Agent ID
            start_frame: Starting frame number

        Returns:
            dataset_idx: Tuple (agent_id, start_frame) that will be handled by
                        extract_concept_activations using dataset.get_chunk()
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided to convert trajectories to indices")

        # Return tuple that extract_concept_activations will handle via get_chunk()
        return (agent_id, start_frame)

    def extract_activations_from_trajectories(self, concept_name, src_mask=None, max_examples=None, batch_size=32):
        """Extract activations for sampled trajectories of a specific concept.

        Args:
            concept_name: Name of concept ('start_walking', 'stays_stopped', etc.)
            src_mask: Optional source mask
            max_examples: Maximum number of examples to process (None = all)
            batch_size: Number of examples to process in parallel (default: 32)

        Returns:
            concept_activations: Dict mapping layer names to activation arrays
        """
        if self.sampled_trajectories is None:
            raise ValueError("Sampled trajectories must be provided during initialization")

        if concept_name not in self.sampled_trajectories:
            raise ValueError(f"Concept '{concept_name}' not found in sampled trajectories. "
                           f"Available: {list(self.sampled_trajectories.keys())}")

        trajectories = self.sampled_trajectories[concept_name]
        if max_examples is not None:
            trajectories = trajectories[:max_examples]

        # Convert trajectories to dataset indices
        example_indices = [self.trajectory_to_dataset_index(agent_id, start_frame)
                          for agent_id, start_frame in trajectories]

        # Extract activations using existing method
        return self.extract_concept_activations(example_indices, self.dataset, src_mask, batch_size)

    def extract_all_concept_activations(self, src_mask=None, max_examples_per_concept=None, batch_size=32):
        """Extract activations for all sampled concept trajectories.

        Args:
            src_mask: Optional source mask
            max_examples_per_concept: Maximum examples per concept (None = all)
            batch_size: Number of examples to process in parallel (default: 32)

        Returns:
            all_activations: Dict mapping concept names to activation dicts
        """
        if self.sampled_trajectories is None:
            raise ValueError("Sampled trajectories must be provided during initialization")

        all_activations = {}

        for concept_name in self.sampled_trajectories.keys():
            LOG.info(f"Extracting activations for concept: {concept_name}")
            all_activations[concept_name] = self.extract_activations_from_trajectories(
                concept_name, src_mask, max_examples_per_concept, batch_size
            )

        return all_activations

    def visualize_attention_head(self, layer_name, head_idx, batch_idx=0,
                                 tokens=None, figsize=(8, 8), cmap='viridis',
                                 vmin=None, vmax=None, norm='linear',
                                 percentile_range=(0, 100)):
        """Visualize a single attention head as a heatmap.

        Args:
            layer_name: Name of the layer to visualize
            head_idx: Index of the attention head
            batch_idx: Batch index to visualize (default: 0)
            tokens: Optional list of token labels for axes
            figsize: Figure size
            cmap: Colormap for heatmap
            vmin, vmax: Min/max values for color scale (overrides percentile_range if provided)
            norm: Normalization method ('linear', 'log', 'sqrt', 'power')
            percentile_range: Tuple of (min, max) percentiles for auto-scaling (default: (0, 100))

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        from matplotlib.colors import LogNorm, PowerNorm, Normalize
        import matplotlib.pyplot as plt

        if layer_name not in self.attention_weights:
            raise ValueError(f"Layer '{layer_name}' not found in attention weights. "
                           f"Available: {list(self.attention_weights.keys())}")

        attn = self.attention_weights[layer_name]  # Shape: (batch, num_heads, T, T) or (batch, T, T)

        # Handle different attention tensor shapes
        if attn.ndim == 4:
            # (batch, num_heads, T, T)
            if head_idx >= attn.shape[1]:
                raise ValueError(f"head_idx {head_idx} out of range (max: {attn.shape[1]-1})")
            attn_head = attn[batch_idx, head_idx].numpy()
        elif attn.ndim == 3:
            # (batch, T, T) - averaged attention
            if head_idx != 0:
                LOG.warning(f"Attention is averaged, ignoring head_idx={head_idx}")
            attn_head = attn[batch_idx].numpy()
        else:
            raise ValueError(f"Unexpected attention shape: {attn.shape}")

        # Auto-scale using percentiles if vmin/vmax not provided
        if vmin is None:
            vmin = np.percentile(attn_head, percentile_range[0])
        if vmax is None:
            vmax = np.percentile(attn_head, percentile_range[1])

        # Create normalization object
        if norm == 'log':
            # For log, ensure vmin > 0
            vmin = max(vmin, 1e-10)
            norm_obj = LogNorm(vmin=vmin, vmax=vmax)
        elif norm == 'sqrt':
            norm_obj = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        elif norm == 'power':
            norm_obj = PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
        else:  # linear
            norm_obj = Normalize(vmin=vmin, vmax=vmax)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(attn_head, cmap=cmap, aspect='auto', norm=norm_obj)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')

        # Set labels
        ax.set_xlabel('Key (attended to)', fontsize=12)
        ax.set_ylabel('Query (attending)', fontsize=12)
        ax.set_title(f'{layer_name} - Head {head_idx}', fontsize=14)

        # Add token labels if provided
        if tokens is not None:
            T = attn_head.shape[0]
            if len(tokens) != T:
                LOG.warning(f"Token count mismatch: {len(tokens)} tokens vs {T} positions")
            else:
                # Show subset of tokens to avoid overcrowding
                stride = max(1, T // 20)  # Show ~20 labels
                positions = list(range(0, T, stride))
                ax.set_xticks(positions)
                ax.set_yticks(positions)
                ax.set_xticklabels([tokens[i] if i < len(tokens) else '' for i in positions],
                                  rotation=90, fontsize=8)
                ax.set_yticklabels([tokens[i] if i < len(tokens) else '' for i in positions],
                                  fontsize=8)

        plt.tight_layout()
        return fig, ax

    def visualize_multihead_attention(self, layer_name, batch_idx=0, tokens=None,
                                      figsize=None, cmap='viridis', vmin=None, vmax=None,
                                      norm='linear', percentile_range=(0, 100)):
        """Visualize all attention heads in a layer as a grid of heatmaps.

        Args:
            layer_name: Name of the layer to visualize
            batch_idx: Batch index to visualize (default: 0)
            tokens: Optional list of token labels
            figsize: Figure size (auto-computed if None)
            cmap: Colormap for heatmaps
            vmin, vmax: Min/max values for color scale (overrides percentile_range if provided)
            norm: Normalization method ('linear', 'log', 'sqrt', 'power')
            percentile_range: Tuple of (min, max) percentiles for auto-scaling (default: (0, 100))

        Returns:
            fig, axes: Matplotlib figure and axes array
        """
        from matplotlib.colors import LogNorm, PowerNorm, Normalize

        if layer_name not in self.attention_weights:
            raise ValueError(f"Layer '{layer_name}' not found in attention weights. "
                           f"Available: {list(self.attention_weights.keys())}")

        attn = self.attention_weights[layer_name]

        # Check if multihead
        if attn.ndim != 4:
            LOG.warning(f"Attention shape {attn.shape} is not multihead (expected 4D). "
                       "Using single head visualization.")
            return self.visualize_attention_head(layer_name, 0, batch_idx, tokens,
                                                figsize or (8, 8), cmap, vmin, vmax,
                                                norm, percentile_range)

        # Get dimensions
        num_heads = attn.shape[1]
        attn_batch = attn[batch_idx].numpy()  # (num_heads, T, T)

        # Auto-scale using percentiles across ALL heads if vmin/vmax not provided
        if vmin is None:
            vmin = np.percentile(attn_batch, percentile_range[0])
        if vmax is None:
            vmax = np.percentile(attn_batch, percentile_range[1])

        # Create normalization object
        if norm == 'log':
            vmin = max(vmin, 1e-10)
            norm_obj = LogNorm(vmin=vmin, vmax=vmax)
        elif norm == 'sqrt':
            norm_obj = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        elif norm == 'power':
            norm_obj = PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
        else:  # linear
            norm_obj = Normalize(vmin=vmin, vmax=vmax)

        # Compute grid layout
        cols = int(np.ceil(np.sqrt(num_heads)))
        rows = int(np.ceil(num_heads / cols))

        # Set figure size
        if figsize is None:
            figsize = (4 * cols, 4 * rows)

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)

        # Plot each head
        for h in range(num_heads):
            row = h // cols
            col = h % cols
            ax = axes[row, col]

            # Plot heatmap
            im = ax.imshow(attn_batch[h], cmap=cmap, aspect='auto', norm=norm_obj)
            ax.set_title(f'Head {h}', fontsize=10)
            ax.set_xlabel('Key', fontsize=8)
            ax.set_ylabel('Query', fontsize=8)

            # Optionally add token labels (only for small sequences)
            if tokens is not None and len(tokens) <= 20:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90, fontsize=6)
                ax.set_yticklabels(tokens, fontsize=6)
            else:
                ax.tick_params(labelsize=6)

        # Hide unused subplots
        for h in range(num_heads, rows * cols):
            row = h // cols
            col = h % cols
            axes[row, col].axis('off')

        # Add colorbar on the right side
        # Use the last axis to determine position
        fig.subplots_adjust(right=0.90)  # Make room for colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, label='Attention Weight')

        plt.suptitle(f'{layer_name} - All Heads', fontsize=14, y=0.98)
        return fig, axes

    def visualize_aggregated_attention(self, layer_name, batch_idx=0,
                                       aggregation='mean', tokens=None,
                                       figsize=(8, 8), cmap='viridis',
                                       vmin=None, vmax=None, norm='linear',
                                       percentile_range=(0, 100)):
        """Visualize attention aggregated across heads.

        Args:
            layer_name: Name of the layer to visualize
            batch_idx: Batch index to visualize (default: 0)
            aggregation: How to aggregate ('mean', 'max', 'min')
            tokens: Optional list of token labels
            figsize: Figure size
            cmap: Colormap for heatmap
            vmin, vmax: Min/max values for color scale (overrides percentile_range if provided)
            norm: Normalization method ('linear', 'log', 'sqrt', 'power')
            percentile_range: Tuple of (min, max) percentiles for auto-scaling (default: (0, 100))

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        from matplotlib.colors import LogNorm, PowerNorm, Normalize

        if layer_name not in self.attention_weights:
            raise ValueError(f"Layer '{layer_name}' not found in attention weights. "
                           f"Available: {list(self.attention_weights.keys())}")

        attn = self.attention_weights[layer_name]

        # Get batch
        if attn.ndim == 4:
            attn_batch = attn[batch_idx].numpy()  # (num_heads, T, T)

            # Aggregate across heads
            if aggregation == 'mean':
                attn_agg = np.mean(attn_batch, axis=0)
            elif aggregation == 'max':
                attn_agg = np.max(attn_batch, axis=0)
            elif aggregation == 'min':
                attn_agg = np.min(attn_batch, axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        elif attn.ndim == 3:
            # Already aggregated
            attn_agg = attn[batch_idx].numpy()
        else:
            raise ValueError(f"Unexpected attention shape: {attn.shape}")

        # Auto-scale using percentiles if vmin/vmax not provided
        if vmin is None:
            vmin = np.percentile(attn_agg, percentile_range[0])
        if vmax is None:
            vmax = np.percentile(attn_agg, percentile_range[1])

        # Create normalization object
        if norm == 'log':
            vmin = max(vmin, 1e-10)
            norm_obj = LogNorm(vmin=vmin, vmax=vmax)
        elif norm == 'sqrt':
            norm_obj = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        elif norm == 'power':
            norm_obj = PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
        else:  # linear
            norm_obj = Normalize(vmin=vmin, vmax=vmax)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(attn_agg, cmap=cmap, aspect='auto', norm=norm_obj)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')

        # Set labels
        ax.set_xlabel('Key (attended to)', fontsize=12)
        ax.set_ylabel('Query (attending)', fontsize=12)
        ax.set_title(f'{layer_name} - {aggregation.capitalize()} Attention', fontsize=14)

        # Add token labels if provided
        if tokens is not None:
            T = attn_agg.shape[0]
            if len(tokens) != T:
                LOG.warning(f"Token count mismatch: {len(tokens)} tokens vs {T} positions")
            else:
                # Show subset of tokens to avoid overcrowding
                stride = max(1, T // 20)
                positions = list(range(0, T, stride))
                ax.set_xticks(positions)
                ax.set_yticks(positions)
                ax.set_xticklabels([tokens[i] if i < len(tokens) else '' for i in positions],
                                  rotation=90, fontsize=8)
                ax.set_yticklabels([tokens[i] if i < len(tokens) else '' for i in positions],
                                  fontsize=8)

        plt.tight_layout()
        return fig, ax

    def visualize_attention_summary(self, layer_name, batch_idx=0, tokens=None,
                                   figsize=(16, 12), cmap='viridis'):
        """Create a comprehensive attention visualization with multiple views.

        Shows:
        - Grid of all heads
        - Mean aggregated attention
        - Max aggregated attention

        Args:
            layer_name: Name of the layer to visualize
            batch_idx: Batch index to visualize
            tokens: Optional list of token labels
            figsize: Figure size
            cmap: Colormap

        Returns:
            fig: Matplotlib figure
        """
        if layer_name not in self.attention_weights:
            raise ValueError(f"Layer '{layer_name}' not found in attention weights. "
                           f"Available: {list(self.attention_weights.keys())}")

        attn = self.attention_weights[layer_name]

        if attn.ndim != 4:
            LOG.warning("Not multihead attention, showing single view")
            return self.visualize_attention_head(layer_name, 0, batch_idx, tokens,
                                                figsize, cmap)

        # Get dimensions
        num_heads = attn.shape[1]
        attn_batch = attn[batch_idx].numpy()  # (num_heads, T, T)

        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, num_heads, hspace=0.3, wspace=0.3)

        # Top rows: Individual heads
        for h in range(num_heads):
            ax = fig.add_subplot(gs[0:2, h])
            im = ax.imshow(attn_batch[h], cmap=cmap, aspect='auto')
            ax.set_title(f'Head {h}', fontsize=9)
            ax.tick_params(labelsize=6)
            if h == 0:
                ax.set_ylabel('Query', fontsize=8)
            ax.set_xlabel('Key', fontsize=7)

        # Bottom row: Aggregated views
        # Mean
        ax_mean = fig.add_subplot(gs[2, :num_heads//2])
        attn_mean = np.mean(attn_batch, axis=0)
        im_mean = ax_mean.imshow(attn_mean, cmap=cmap, aspect='auto')
        ax_mean.set_title('Mean Across Heads', fontsize=10)
        ax_mean.set_xlabel('Key', fontsize=8)
        ax_mean.set_ylabel('Query', fontsize=8)
        plt.colorbar(im_mean, ax=ax_mean, fraction=0.046)

        # Max
        ax_max = fig.add_subplot(gs[2, num_heads//2:])
        attn_max = np.max(attn_batch, axis=0)
        im_max = ax_max.imshow(attn_max, cmap=cmap, aspect='auto')
        ax_max.set_title('Max Across Heads', fontsize=10)
        ax_max.set_xlabel('Key', fontsize=8)
        ax_max.set_ylabel('Query', fontsize=8)
        plt.colorbar(im_max, ax=ax_max, fraction=0.046)

        fig.suptitle(f'{layer_name} - Attention Summary', fontsize=14)
        return fig
# Initialize interpreter with sampled trajectories
interpreter = ModelInterpreter(
    model=model,
    device=device,
    sampled_trajectories=sampled_trajectories,
    dataset=train_dataset
)

# %%
interpreter.remove_hooks()
interpreter.register_activation_hooks()

start_walking_acts = interpreter.extract_activations_from_trajectories('start_walking', max_examples=1)
print(f"Start walking activations layers: {list(start_walking_acts.keys())}")

# %%
# ============================================================================
# EXTRACT RESIDUAL STREAM HIDDEN STATES
# ============================================================================

# Register residual stream hooks (instead of all activation hooks)
interpreter.remove_hooks()
interpreter.register_residual_stream_hooks()

# Extract residual stream states for both concepts
print("\n=== Extracting Residual Stream States ===")
start_walking_residual = interpreter.extract_activations_from_trajectories('start_walking', max_examples=100)
stays_stopped_residual = interpreter.extract_activations_from_trajectories('stays_stopped', max_examples=100)

# %%
# Display residual stream information
print("\n=== Residual Stream Hidden States ===")
print(f"Layers captured: {list(start_walking_residual.keys())}")

print("\nShapes of residual stream states (n_examples, batch, seq_len, hidden_dim):")
for layer_name in sorted(start_walking_residual.keys()):
    shape = start_walking_residual[layer_name].shape
    print(f"  {layer_name}: {shape}")
    
# Debug: Check for NaN values in residual stream
print("\n=== Debugging NaN Values ===")
for layer_name in sorted(start_walking_residual.keys()):
    acts = start_walking_residual[layer_name]
    print(f"\n{layer_name}:")
    print(f"  Shape: {acts.shape}")
    print(f"  Contains NaN: {np.isnan(acts).any()}")
    print(f"  Num NaN: {np.isnan(acts).sum()}/{acts.size}")
    print(f"  Min: {np.nanmin(acts) if not np.isnan(acts).all() else 'all nan'}")
    print(f"  Max: {np.nanmax(acts) if not np.isnan(acts).all() else 'all nan'}")
    print(f"  Mean (ignoring NaN): {np.nanmean(acts) if not np.isnan(acts).all() else 'all nan'}")


# %%
# Example: Analyze residual stream at different layers
# Get mean activation at each layer
print("\n=== Mean Activation Statistics per Layer ===")
for layer_name in sorted(start_walking_residual.keys()):
    # Use nanmean to ignore NaN values
    start_walk_mean = np.nanmean(start_walking_residual[layer_name])
    start_walk_std = np.nanstd(start_walking_residual[layer_name])
    stays_stop_mean = np.nanmean(stays_stopped_residual[layer_name])
    stays_stop_std = np.nanstd(stays_stopped_residual[layer_name])

    print(f"\n{layer_name}:")
    print(f"  Start walking: mean={start_walk_mean:.4f}, std={start_walk_std:.4f}")
    print(f"  Stays stopped: mean={stays_stop_mean:.4f}, std={stays_stop_std:.4f}")
    print(f"  Difference:    {start_walk_mean - stays_stop_mean:.4f}")


# %%
# ============================================================================
# ATTENTION VISUALIZATION EXAMPLES
# ============================================================================

# Example 1: Register attention hooks and run a forward pass
# First, create a new interpreter instance or clear hooks from the existing one
interpreter.remove_hooks()
interpreter.register_attention_hooks()

ex_idx = 4  # Example index to test

# Get an example from the dataset
agent_id, start_frame = sampled_trajectories['start_walking'][ex_idx]
# sim_start_frame = start_frame - config['contextl']
sim_start_frame = start_frame - config['contextl'] + 150
example_data = train_dataset.get_chunk(sim_start_frame, train_dataset.context_length, agent_id)
input_data = example_data['input']
input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)

# Run forward pass to capture attention
with torch.no_grad():
    output = model(input_tensor, is_causal=True)

# Check what attention weights were captured
print("Captured attention weights from layers:")
for layer_name in interpreter.attention_weights.keys():
    attn = interpreter.attention_weights[layer_name]
    print(f"  {layer_name}: shape {attn.shape}")

# %%
# Example 2: Visualize a single attention head
# If you have attention weights captured, visualize a specific head
if len(interpreter.attention_weights) > 0:
    # Get first layer with attention
    layer_name = list(interpreter.attention_weights.keys())[3]

    # Visualize head 0
    fig, ax = interpreter.visualize_attention_head(
        layer_name=layer_name,
        head_idx=0,
        batch_idx=0,
        figsize=(10, 10),
        cmap='plasma',
        norm='power'
    )
    plt.show()

# %%
# Example 3: Visualize all heads in a layer as a grid
if len(interpreter.attention_weights) > 0:
    if not os.path.exists(f'attn_examples/ex{ex_idx}'):
        os.makedirs(f'attn_examples/ex{ex_idx}')
    for i in range(10):
        layer_name = list(interpreter.attention_weights.keys())[i]

        fig, axes = interpreter.visualize_multihead_attention(
            layer_name=layer_name,
            batch_idx=0,
            cmap='viridis'
        )
        fig.savefig(f'attn_examples/ex{ex_idx}/sim_attention_{layer_name.replace(".", "_")}.png')
