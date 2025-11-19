# ---
# jupyter:
#   jupytext:
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

# %%
configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json"
mode = 'test' # can toggle to 'train'/'test'
pretrained_modelfile = os.path.join('/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/llmnets',
                                    'predvel_20251007_20251002T000000_epoch200.pth')
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


# %%
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

# %%
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