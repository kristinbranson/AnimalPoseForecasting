"""
Behavior Extractor for Animal Pose Forecasting

This module provides a class to extract specific behaviors from the dataset,
such as start-of-movement events (when animals transition from rest to motion).
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import logging
import pickle

LOG = logging.getLogger(__name__)


class StartOfMovementExtractor:
    """Extracts trajectories where animals start moving (velocity changes from 0 to non-zero).

    This class identifies moments when an animal transitions from rest (low velocity)
    to movement (higher velocity), and extracts trajectory windows around those events.

    Args:
        dataset: The dataset containing pose and velocity information
        context_length: Length of trajectory windows to extract (default: 512)
        velocity_threshold: Velocity magnitude below which is considered "at rest" (default: 0.1)
        min_rest_duration: Minimum number of frames animal must be at rest before movement (default: 10)
        min_movement_duration: Minimum number of frames of movement after start (default: 5)
        position_in_window: Position of the movement onset within the extracted window,
                          from 0 (at start) to 1 (at end). Default 0.3 means onset is at 30% into window.
    """

    def __init__(
        self,
        dataset,
        context_length: int = 512,
        velocity_threshold: float = 0.1,
        min_rest_duration: int = 10,
        min_movement_duration: int = 5,
        position_in_window: float = 0.3,
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.velocity_threshold = velocity_threshold
        self.min_rest_duration = min_rest_duration
        self.min_movement_duration = min_movement_duration
        self.position_in_window = position_in_window

        # Calculate how many frames before the onset event
        self.frames_before_onset = int(context_length * position_in_window)
        self.frames_after_onset = context_length - self.frames_before_onset

        LOG.info(f"StartOfMovementExtractor initialized:")
        LOG.info(f"  Context length: {context_length}")
        LOG.info(f"  Velocity threshold: {velocity_threshold}")
        LOG.info(f"  Min rest duration: {min_rest_duration} frames")
        LOG.info(f"  Min movement duration: {min_movement_duration} frames")
        LOG.info(f"  Frames before onset: {self.frames_before_onset}")
        LOG.info(f"  Frames after onset: {self.frames_after_onset}")

    def compute_velocity_magnitude(self, velocity: np.ndarray) -> np.ndarray:
        """Compute velocity magnitude from velocity features.

        Args:
            velocity: (n_agents, n_frames, n_features) array

        Returns:
            velocity_mag: (n_agents, n_frames) array of velocity magnitudes
        """
        # Get velocity from dataset - use global velocity (thorax position changes)
        # The first 3 features are typically: forward velocity, sideways velocity, angular velocity
        # We'll compute magnitude from the first 2 (forward and sideways movement)
        velocity_xy = velocity[..., :2]  # forward and sideways velocity
        velocity_mag = np.sqrt(np.sum(velocity_xy**2, axis=-1))

        return velocity_mag

    def detect_movement_onsets(
        self,
        velocity_mag: np.ndarray,
        agent_id: int,
    ) -> List[int]:
        """Detect frames where movement starts.

        Args:
            velocity_mag: (n_frames,) array of velocity magnitudes for one agent
            agent_id: ID of the agent

        Returns:
            onset_frames: List of frame indices where movement starts
        """
        onset_frames = []
        n_frames = len(velocity_mag)

        # Find periods of rest and movement
        is_rest = velocity_mag < self.velocity_threshold
        is_moving = ~is_rest

        i = self.min_rest_duration  # Start after minimum rest duration
        while i < n_frames - self.min_movement_duration:
            # Check if we're transitioning from rest to movement
            rest_period = is_rest[i - self.min_rest_duration:i]
            movement_period = is_moving[i:i + self.min_movement_duration]

            # If all frames in rest period are rest AND all frames in movement period are moving
            if np.all(rest_period) and np.all(movement_period):
                # Also check the frame isn't NaN
                if not np.isnan(velocity_mag[i]):
                    onset_frames.append(i)
                    # Skip ahead to avoid detecting the same onset multiple times
                    i += self.min_rest_duration + self.min_movement_duration
                else:
                    i += 1
            else:
                i += 1

        LOG.info(f"  Agent {agent_id}: Found {len(onset_frames)} movement onsets")
        return onset_frames

    def extract_trajectories(self) -> Dict[str, np.ndarray]:
        """Extract all trajectories containing start-of-movement events.

        Returns:
            trajectories: Dictionary containing:
                'inputs': (n_trajectories, context_length, n_input_features)
                'labels': (n_trajectories, context_length, n_output_features)
                'labels_discrete': (n_trajectories, context_length, n_discrete_features) if present
                'onset_frames': (n_trajectories,) frame index of onset within each trajectory
                'agent_ids': (n_trajectories,) agent ID for each trajectory
                'global_frames': (n_trajectories,) global frame index of onset in original data
        """
        LOG.info("Extracting start-of-movement trajectories...")

        # Get velocity data - need to extract the CONTINUOUS part only, not the discrete bins
        # The dataset.labels['velocity'].array contains BOTH discrete bins and continuous features
        # We need to split them to get the actual velocity values for detection
        from apf.dataset import get_data_chunk, split_discr_cont

        velocity_full = self.dataset.labels['velocity'].array  # (n_agents, n_frames, n_features_with_bins)
        n_agents, n_frames, _ = velocity_full.shape

        # Split to get only continuous velocity (not the discrete probability bins)
        velocity_discrete, velocity_continuous = split_discr_cont(
            velocity_full,
            self.dataset.label_bin_indices
        )

        LOG.info(f"Dataset shape: {n_agents} agents, {n_frames} frames")
        LOG.info(f"Velocity features: {velocity_full.shape[-1]} total, {velocity_continuous.shape[-1]} continuous, {velocity_discrete.shape[-1]} discrete")
        LOG.info("Note: Using CONTINUOUS z-scored velocity for detection (excluding discrete bins).")

        # Use continuous velocity for detection
        velocity_array = velocity_continuous

        all_trajectories = []
        all_onset_frames = []
        all_agent_ids = []
        all_global_frames = []

        # For each agent, detect movement onsets and extract trajectories
        for agent_id in range(n_agents):
            velocity_mag = self.compute_velocity_magnitude(velocity_array[agent_id])
            onset_frames = self.detect_movement_onsets(velocity_mag, agent_id)

            # Extract trajectory windows around each onset
            for onset_frame in onset_frames:
                # Calculate window boundaries
                start_frame = onset_frame - self.frames_before_onset
                end_frame = onset_frame + self.frames_after_onset

                # Check if window is valid (within bounds and no NaN breaks)
                if start_frame >= 0 and end_frame <= n_frames:
                    # Check for valid data in the entire window
                    chunk = self.dataset.get_chunk(start_frame, self.context_length, agent_id)

                    # Check if chunk has NaNs
                    has_nans = np.any(np.isnan(chunk['input']))
                    if 'labels' in chunk:
                        has_nans = has_nans or np.any(np.isnan(chunk['labels']))

                    if not has_nans:
                        all_trajectories.append(chunk)
                        all_onset_frames.append(self.frames_before_onset)  # Position within window
                        all_agent_ids.append(agent_id)
                        all_global_frames.append(onset_frame)

        LOG.info(f"Extracted {len(all_trajectories)} valid trajectories")

        if len(all_trajectories) == 0:
            LOG.warning("No valid trajectories found!")
            return None

        # Stack all trajectories into arrays
        result = {
            'inputs': np.stack([traj['input'] for traj in all_trajectories], axis=0),
            'onset_frames': np.array(all_onset_frames),
            'agent_ids': np.array(all_agent_ids),
            'global_frames': np.array(all_global_frames),
        }

        # Add labels if present
        if 'labels' in all_trajectories[0]:
            result['labels'] = np.stack([traj['labels'] for traj in all_trajectories], axis=0)

        if 'labels_discrete' in all_trajectories[0]:
            result['labels_discrete'] = np.stack([traj['labels_discrete'] for traj in all_trajectories], axis=0)

        LOG.info(f"Result shapes:")
        for key, val in result.items():
            LOG.info(f"  {key}: {val.shape}")

        return result

    def save_trajectories(self, trajectories: Dict[str, np.ndarray], filename: str):
        """Save extracted trajectories to a file.

        Args:
            trajectories: Dictionary of trajectory data from extract_trajectories()
            filename: Output filename (will add .npz extension if not present)
        """
        if not filename.endswith('.npz'):
            filename += '.npz'

        # Save as compressed numpy archive
        np.savez_compressed(filename, **trajectories)
        LOG.info(f"Saved trajectories to {filename}")

        # Also save metadata in a separate pickle file
        metadata = {
            'context_length': self.context_length,
            'velocity_threshold': self.velocity_threshold,
            'min_rest_duration': self.min_rest_duration,
            'min_movement_duration': self.min_movement_duration,
            'position_in_window': self.position_in_window,
            'frames_before_onset': self.frames_before_onset,
            'frames_after_onset': self.frames_after_onset,
            'n_trajectories': len(trajectories['agent_ids']),
        }

        metadata_filename = filename.replace('.npz', '_metadata.pkl')
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)
        LOG.info(f"Saved metadata to {metadata_filename}")

    @staticmethod
    def load_trajectories(filename: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Load trajectories from a saved file.

        Args:
            filename: Input filename

        Returns:
            trajectories: Dictionary of trajectory data
            metadata: Dictionary of extraction parameters
        """
        if not filename.endswith('.npz'):
            filename += '.npz'

        # Load trajectories
        data = np.load(filename)
        trajectories = {key: data[key] for key in data.keys()}

        # Load metadata
        metadata_filename = filename.replace('.npz', '_metadata.pkl')
        try:
            with open(metadata_filename, 'rb') as f:
                metadata = pickle.load(f)
        except FileNotFoundError:
            LOG.warning(f"Metadata file {metadata_filename} not found")
            metadata = {}

        LOG.info(f"Loaded {len(trajectories['agent_ids'])} trajectories from {filename}")
        return trajectories, metadata
