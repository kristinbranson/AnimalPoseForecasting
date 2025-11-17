"""
Human-interpretable behavioral concept computation as Dataset Operations.

This module provides Operation classes for computing behavioral concepts from
raw velocity data, before any z-scoring or discretization is applied.

The HumanConcept operation computes velocity magnitude from the first 2 components
of the Velocity operation output (dside, dfwd from GlobalVelocity), which corresponds
to the velocity of the fly's center position (base_thorax). The magnitude
sqrt(dside² + dfwd²) equals the velocity magnitude of the center point.

Example usage in experiments/flyllm.py:
    from apf.concepts import HumanConcept

    # After computing raw velocity (GlobalVelocity + LocalVelocity fusion)
    velocity = Velocity(featrelative, featangle)(pose, isstart=isstart)

    # Compute concepts from raw velocity (before z-scoring/discretization)
    concepts = HumanConcept(
        concept_type="start_walking",
        fps=150.0,
        sigma=2,
        concept_params={
            'thresh_stopped': 5.0,
            'thresh_walking': 15.0,
            'tstopped': 0.5,
            'tfuture': 1.0
        }
    )(velocity, isstart=isstart)

    # Add to dataset labels (process velocity separately for model training)
    dataset = Dataset(
        inputs=...,
        labels={
            'velocity': apply_operations(velocity, [Zscore(), Discretize(), ...]),
            'concepts': concepts  # Keep concepts as raw labels
        },
        ...
    )
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter1d, convolve

from apf.dataset import Operation

LOG = logging.getLogger(__name__)


def same2valid(x, filsize, debug=False):
    """
    Unpad an array x that was convolved with a filter of size filsize so that all entries are valid.
    This only handles 1D convolution along axis=0, but x can have any number of trailing dimensions.

    Args:
        x: array of shape (nframes, ...), the result of scipy.ndimage.convolve of an array
           with a filter of size (filsize,...).
        filsize: int, size of the filter used in the convolution
        debug: bool, if True, assumes that the padding regions are NaN, and checks that
               padleft and padright regions are all NaN, and that the valid region has no NaNs.

    Returns:
        array of shape (nframes - filsize + 1, ...), the unpadded valid region of x
    """
    padright = filsize // 2
    padleft = filsize - padright - 1
    if debug:
        print(f'padleft = {padleft}, padright = {padright}')
        assert padleft == 0 or np.all(np.isnan(x[:padleft]))
        assert padright == 0 or np.all(np.isnan(x[-padright:]))
        assert not np.any(np.isnan(x[padleft:-padright]))
    return x[padleft:-padright] if padright > 0 else x[padleft:]


@dataclass
class HumanConcept(Operation):
    """
    Computes human-interpretable behavioral concepts from velocity data.

    This operation should be applied to raw velocity before z-scoring or discretization.
    The output is concept labels encoded as floats:
        +1: Positive concept (e.g., "start_walking")
         0: Neutral/no concept detected
        -1: Negative concept (e.g., "stays_stopped")
       NaN: Invalid data (where velocity is NaN)

    Attributes:
        concept_type: str, which concept to compute (default: "start_walking")
        fps: float, frames per second (default: 150.0)
        sigma: float, Gaussian smoothing sigma in frames (default: 2.0)
        concept_params: dict, concept-specific parameters
            For "start_walking":
                thresh_stopped: float (default 5.0) - velocity threshold for "stopped" in mm/s
                thresh_walking: float (default 15.0) - velocity threshold for "walking" in mm/s
                tstopped: float (default 0.5) - time in seconds fly must have been stopped
                tfuture: float (default 1.0) - time in seconds to look into future

    Example:
        concepts = HumanConcept(
            concept_type="start_walking",
            fps=150.0,
            sigma=2,
            concept_params={'thresh_stopped': 5.0, 'thresh_walking': 15.0}
        )(velocity, isstart=isstart)
    """
    concept_type: str = "start_walking"
    fps: float = 150.0
    sigma: float = 2.0
    concept_params: dict = field(default_factory=dict)

    def apply(self, velocity: np.ndarray, isstart: np.ndarray | None = None) -> np.ndarray:
        """
        Compute concept labels from velocity data.

        Args:
            velocity: (n_agents, n_frames, n_velocity_features) float array
                Raw velocity in physical units (mm/frame). Should contain at least 2 features
                representing x,y velocity components (dfwd, dside from GlobalVelocity).
                Note: GlobalVelocity already computes velocity of the center position (base_thorax),
                so velocity magnitude sqrt(dfwd² + dside²) equals the center point's velocity magnitude.
            isstart: (n_frames, n_agents) bool array
                Whether a new track starts at each frame. Used to set NaNs at boundaries.

        Returns:
            labels: (n_agents, n_frames, 1) float array with values {-1, 0, 1, NaN}
                NaN indicates invalid data regions (where velocity is NaN)
        """
        # Extract velocity components (dside, dfwd from GlobalVelocity via Fusion)
        if velocity.shape[-1] < 2:
            raise ValueError("Velocity data must have at least 2 dimensions for magnitude computation")

        vel_xy = velocity[..., :2]  # (n_agents, n_frames, 2) - [dside, dfwd]
        velmag = np.linalg.norm(vel_xy, axis=-1)  # (n_agents, n_frames)

        # Transpose to (n_frames, n_agents) for compatibility with concept computation
        velmag = velmag.T
        if isstart is None:
            isstart = np.zeros_like(velmag, dtype=bool)

        # Smooth velocity
        if self.sigma > 0:
            velmag_smooth = gaussian_filter1d(velmag, sigma=self.sigma, axis=0)
        else:
            velmag_smooth = velmag

        # Convert to mm/s (assuming input is mm/frame)
        velmag_mmps = velmag_smooth * self.fps

        # Set NaNs at track starts
        velmag_mmps[isstart] = np.nan

        # Compute concept-specific labels
        if self.concept_type == "start_walking":
            labels = self._compute_start_walking(velmag_mmps, isstart)
        else:
            raise ValueError(f"Unsupported concept: {self.concept_type}. Currently supported: ['start_walking']")

        # Return as (n_agents, n_frames, 1) to match Data conventions
        # Keep as float32 to preserve NaN for invalid regions
        return labels.T[..., None].astype(np.float32)

    def _compute_start_walking(self, velmag_mmps: np.ndarray, isstart: np.ndarray) -> np.ndarray:
        """
        Compute start_walking concept labels.

        Detects frames where:
            +1: Fly was stopped and will start walking
            -1: Fly was stopped and will stay stopped
             0: Neither condition applies
           NaN: Invalid data (velocity is NaN)

        Args:
            velmag_mmps: (n_frames, n_agents) float array, velocity magnitude in mm/s
            isstart: (n_frames, n_agents) bool array

        Returns:
            labels: (n_frames, n_agents) float array with values {-1, 0, 1, NaN}
        """
        # Get parameters with defaults
        thresh_stopped = self.concept_params.get('thresh_stopped', 5.0)
        thresh_walking = self.concept_params.get('thresh_walking', 15.0)
        tstopped = self.concept_params.get('tstopped', 0.5)
        tfuture = self.concept_params.get('tfuture', 1.0)

        # Convert time to frames
        tstopped_frames = int(tstopped * self.fps)
        tfuture_frames = int(tfuture * self.fps)

        # Create convolution filters
        filstopped = np.ones((tstopped_frames, 1), dtype=int)
        filfuture = np.ones((tfuture_frames, 1), dtype=int)
        nflies = velmag_mmps.shape[1]

        # Track invalid data (NaN in velocity magnitude)
        invalid_mask = np.isnan(velmag_mmps)

        # Compute stopped and walking states
        # Use np.nan_to_num to avoid NaN comparison issues
        velmag_for_thresh = np.nan_to_num(velmag_mmps, nan=np.inf)  # NaN becomes inf, never matches thresholds
        isstop = velmag_for_thresh <= thresh_stopped
        iswalk = velmag_for_thresh >= thresh_walking

        # Was stopped: stopped for at least tstopped_frames
        wasstopped = convolve(isstop.astype(int), filstopped, mode='constant', cval=0) >= tstopped_frames
        wasstopped = same2valid(wasstopped, tstopped_frames)
        # Align with original frames (first tstopped_frames-1 are False)
        wasstopped = np.r_[np.zeros((tstopped_frames - 1, nflies), dtype=bool), wasstopped]

        # Will walk: will be walking at some point in next tfuture_frames
        willwalk = convolve(iswalk.astype(int), filfuture, mode='constant', cval=0) >= 1
        willwalk = same2valid(willwalk, tfuture_frames)
        # Shift to align with original frames
        willwalk = np.r_[willwalk[1:], np.zeros((tfuture_frames, nflies), dtype=bool)]

        # Will stop: will remain stopped for next tfuture_frames
        willstop = convolve(isstop.astype(int), filfuture, mode='constant', cval=0) >= tfuture_frames
        willstop = same2valid(willstop, tfuture_frames)
        # Shift to align with original frames
        willstop = np.r_[willstop[1:], np.zeros((tfuture_frames, nflies), dtype=bool)]

        # Compute start_walking concept
        start_walking = wasstopped & willwalk
        stays_stopped = wasstopped & willstop

        # Create labels array as float to support NaN, then convert
        labels = np.zeros_like(velmag_mmps, dtype=float)
        labels[start_walking] = 1
        labels[stays_stopped] = -1

        # Mark invalid regions as NaN
        labels[invalid_mask] = np.nan

        return labels

    def invert(self, labels: np.ndarray) -> None:
        """
        Concept labels are not invertible.

        Args:
            labels: (n_agents, n_frames, 1) int array
        """
        LOG.error(f"HumanConcept operation is not invertible")
        return None
