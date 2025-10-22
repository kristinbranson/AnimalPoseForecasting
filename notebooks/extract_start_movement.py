"""
Example script to extract start-of-movement behaviors from the fly dataset.

This script loads the dataset and extracts trajectories where animals transition
from rest to movement, saving them to a data file for later analysis.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)

from apf.io import read_config
from flyllm.config import DEFAULTCONFIGFILE, posenames
from flyllm.features import featglobal, get_sensory_feature_idx
from experiments.flyllm import make_dataset
from behavior_extractor import StartOfMovementExtractor


def main():
    """Extract start-of-movement trajectories from the fly dataset."""

    # Configuration
    configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_20251007.json"

    # Use absolute path for output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "extracted_behaviors")
    os.makedirs(output_dir, exist_ok=True)

    LOG.info(f"Output directory: {output_dir}")

    LOG.info("=" * 80)
    LOG.info("EXTRACTING START-OF-MOVEMENT BEHAVIORS")
    LOG.info("=" * 80)

    # Load configuration
    LOG.info(f"Loading config from: {configfile}")
    config = read_config(
        configfile,
        default_configfile=DEFAULTCONFIGFILE,
        posenames=posenames,
        featglobal=featglobal,
        get_sensory_feature_idx=get_sensory_feature_idx,
    )

    # Load dataset
    LOG.info("Loading training dataset...")
    LOG.info("Note: This may take several minutes for large datasets...")

    # Set debug=True to load less data for faster testing
    # Set debug=False to load full dataset
    use_debug = False  # Change to True for quick testing

    train_dataset, flyids, track, pose, velocity, sensory = make_dataset(
        config,
        'intrainfile',
        return_all=True,
        debug=use_debug
    )

    LOG.info(f"Dataset loaded successfully!")
    LOG.info(f"Dataset size: {len(train_dataset)} chunks")

    # Initialize the behavior extractor
    LOG.info("\nInitializing StartOfMovementExtractor...")
    # Note: velocity_threshold is in z-scored units (standard deviations from mean)
    # Since data is z-scored, a threshold of ~0.5 means within 0.5 std of mean velocity
    extractor = StartOfMovementExtractor(
        dataset=train_dataset,
        context_length=1024,  # 512 frames before + 512 frames after onset
        velocity_threshold=0.5,  # Threshold for "at rest" in z-scored units (tune this parameter)
        min_rest_duration=50,  # Minimum frames at rest before movement
        min_movement_duration=10,  # Minimum frames of movement after start
        position_in_window=0.5,  # Put onset at center (50% into the window = 512 frames before, 512 after)
    )

    # Extract trajectories
    LOG.info("\nExtracting trajectories with start-of-movement events...")
    trajectories = extractor.extract_trajectories()

    if trajectories is None:
        LOG.error("No trajectories extracted! Try adjusting the parameters.")
        return

    # Save to file
    output_file = os.path.join(output_dir, "start_of_movement_trajectories_ctx1024_vth0.5_rest50_move10_pos0.5.npz")
    LOG.info(f"\nSaving trajectories to: {output_file}")
    extractor.save_trajectories(trajectories, output_file)

    # Print summary statistics
    LOG.info("\n" + "=" * 80)
    LOG.info("EXTRACTION SUMMARY")
    LOG.info("=" * 80)
    LOG.info(f"Total trajectories extracted: {len(trajectories['agent_ids'])}")
    LOG.info(f"Unique agents: {len(set(trajectories['agent_ids']))}")
    LOG.info(f"Context length: {config['contextl']} frames")
    LOG.info(f"Onset position: {extractor.frames_before_onset} frames into window")
    LOG.info(f"\nData shapes:")
    for key, val in trajectories.items():
        LOG.info(f"  {key}: {val.shape}")

    LOG.info("\n" + "=" * 80)
    LOG.info("DONE!")
    LOG.info("=" * 80)

    # Example: How to load the saved data later
    LOG.info("\nTo load this data later, use:")
    LOG.info(f"  from behavior_extractor import StartOfMovementExtractor")
    LOG.info(f"  trajectories, metadata = StartOfMovementExtractor.load_trajectories('{output_file}')")


if __name__ == "__main__":
    main()
