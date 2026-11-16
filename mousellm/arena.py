"""Estimates where the arena walls are in each MABe22 mouse sequence.

The MABe22 mouse sequences were sampled from about 12 four-day experiments (MABe22 appendix C.2), but the
data don't say which sequence came from which: the sequence names are random hashes.  Where the arena is in
the frame, and how big it looks, differs between experiments by a few cm, since the camera was placed a
little differently for each.  So the arena of each sequence is estimated:

1. The arena floor is bright in the videos, so its bounding box is measured in the video of each sequence
   that has one (the two files with videos, a third of the sequences).
2. Those boxes fall into tight clusters, one per camera placement, so presumably per experiment (11 of
   them, of about 200 to 280 sequences each, for MABe22).  Each placement's arena is the median of its
   sequences' boxes.
3. Each sequence, with a video or not, is assigned to a placement by a logistic regression on the histograms
   of its keypoints' x and y, trained on the sequences with videos.  Mice spend much of their time along
   the walls, so the histograms show where the walls are.  (A sequence with a video in one of the
   placements is assigned to it directly.)

In cross-validation on the sequences with videos, step 3 assigns 93% of them to the right placement, and
leaves only 1% with a wall more than 2 cm off.  The sequences without videos fit the placements as well as
those with videos do, so they seem to come from the same experiments.

The keypoints overshoot the floor box a little, about 1 cm on three sides and 3 cm at the north wall,
which has the food hopper, since mice rear and climb on the walls, and so the arena is the floor box, not a
robust min and max of the keypoints.

Boxes are (x0, x1, y0, y1), in cm in the keypoint frame (mousellm.config).
"""
import logging
import os

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from mousellm.config import ARENA_CENTER_PX, CM_PER_PX

LOG = logging.getLogger(__name__)

# The video dir of each MABe22 keypoint file that has videos, in the MABe22 data dir
VIDEO_DIR_NAMES = {'mouse_user_train.npy': 'mouse_userTrain_videos_resized_224',
                   'mouse_submission_keypoints.npy': 'mouse_submission_videos_resized_224'}
# The MABe22 videos are resized to this many pixels across, from the keypoints' 512
VIDEO_FRAME_SIZE_PX = 224
KEYPOINT_FRAME_SIZE_PX = 512
# Number of frames of each video whose max is taken, to remove the dark mice from the bright floor
N_FLOOR_FRAMES = 9
# Video floor boxes within this distance of each other, in cm, in each wall (complete linkage), are the
# same placement
PLACEMENT_LINKAGE_CM = 1.6
# Smaller clusters of video floor boxes are outliers (a misdetected floor), not placements, and their
# sequences are assigned by their keypoints, like those without videos
MIN_PLACEMENT_N_SEQS = 30
# Keypoint histogram bin edges, in cm
HIST_EDGES_CM = np.arange(-45., 45.0001, .5)
# Inverse regularization strength of the logistic regression
ASSIGN_C = 10.


def video_floor_bounds(video_file_path: str) -> np.ndarray:
    """ The bounding box of the bright arena floor in a MABe22 video, (x0, x1, y0, y1) in cm in the keypoint
    frame: the rows and columns that are more than half floor, in the max over N_FLOOR_FRAMES frames, which
    removes the dark mice. """
    video = cv2.VideoCapture(video_file_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for t in np.linspace(0, n_frames - 2, N_FLOOR_FRAMES).astype(int):
        video.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = video.read()
        if ok:
            frames.append(frame[..., 0].astype(float))
    video.release()
    im = np.max(frames, axis=0)
    floor = im > .5 * np.percentile(im, 95)
    rows = np.nonzero(floor.mean(1) > .5)[0]
    cols = np.nonzero(floor.mean(0) > .5)[0]
    bounds_px = np.array([cols[0], cols[-1] + 1, rows[0], rows[-1] + 1]) * KEYPOINT_FRAME_SIZE_PX / VIDEO_FRAME_SIZE_PX
    return (bounds_px - ARENA_CENTER_PX[[0, 0, 1, 1]]) * CM_PER_PX


def video_file_path_maybe(in_dir_path: str, seq_name: str, file_name: str) -> str | None:
    """ The path of a MABe22 sequence's video, or None if its file has no videos or the video is missing. """
    if file_name not in VIDEO_DIR_NAMES:
        return None
    path = os.path.join(in_dir_path, VIDEO_DIR_NAMES[file_name], f'{seq_name}.avi')
    return path if os.path.exists(path) else None


def keypoint_histograms(X: np.ndarray) -> np.ndarray:
    """ Histograms of a sequence's keypoints' x and y, over all its mice and keypoints, in HIST_EDGES_CM bins.

    Args:
        X: (n_keypoints, 2, n_frames, n_mice) keypoint positions in cm, NaN where untracked.

    Returns:
        hists: (2, n_bins) int array
    """
    hists = []
    for d in range(2):
        v = X[:, d].ravel()
        hists.append(np.histogram(v[~np.isnan(v)], HIST_EDGES_CM)[0])
    return np.array(hists)


def find_placements(video_bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Clusters video floor boxes into camera placements.

    Args:
        video_bounds: (n_videos, 4) floor boxes.

    Returns:
        placement: (n_videos,) int array, each video's placement, or -1 for an outlier box.
        placement_bounds: (n_placements, 4) each placement's arena, the median of its boxes.
    """
    cluster = fcluster(linkage(video_bounds, 'complete'), PLACEMENT_LINKAGE_CM, 'distance')
    ks, counts = np.unique(cluster, return_counts=True)
    big = ks[counts >= MIN_PLACEMENT_N_SEQS]
    placement = np.full(len(video_bounds), -1)
    for i, k in enumerate(big):
        placement[cluster == k] = i
    placement_bounds = np.array([np.median(video_bounds[placement == i], axis=0) for i in range(len(big))])
    return placement, placement_bounds


def estimate_arena_bounds(hists: np.ndarray, video_bounds: np.ndarray) -> dict:
    """ Estimates the arena of each sequence, as the module docstring describes.

    Args:
        hists: (n_seqs, 2, n_bins) keypoint_histograms of each sequence.
        video_bounds: (n_seqs, 4) video_floor_bounds of each sequence, NaN for sequences without a video.

    Returns:
        dict with
            arena_bounds: (n_seqs, 4) each sequence's arena, its placement's.
            placement: (n_seqs,) int array, each sequence's placement.
            confidence: (n_seqs,) the logistic regression's probability of the placement, or 1 for sequences
                placed by their video.
            placement_bounds: (n_placements, 4) each placement's arena.
            cv_accuracy: the fraction of the video-placed sequences that cross-validation assigns right.
            cv_frac_wall_off_2cm: the fraction for which it gets a wall more than 2 cm off.
    """
    hists = hists.astype(float)
    # square roots of the normalized histograms, so that a few heavily used bins don't dominate
    features = np.sqrt(hists / np.maximum(hists.sum(-1, keepdims=True), 1)).reshape(len(hists), -1)
    has_video = ~np.isnan(video_bounds).any(axis=1)
    video_idx = np.nonzero(has_video)[0]
    video_placement, placement_bounds = find_placements(video_bounds[has_video])
    placed = video_idx[video_placement >= 0]
    y = video_placement[video_placement >= 0]
    LOG.info(f'{len(placement_bounds)} camera placements, from the floor boxes of {len(video_idx)} videos '
             f'({np.sum(video_placement < 0)} outliers)')

    clf = LogisticRegression(C=ASSIGN_C, max_iter=3000)
    cv_pred = cross_val_predict(clf, features[placed], y, cv=10)
    cv_accuracy = float(np.mean(cv_pred == y))
    wall_error = np.abs(placement_bounds[cv_pred] - placement_bounds[y]).max(axis=1)
    cv_frac_wall_off_2cm = float(np.mean(wall_error > 2.))
    LOG.info(f'Placement assignment cross-validation: {cv_accuracy:.1%} right, {cv_frac_wall_off_2cm:.1%} '
             f'with a wall more than 2 cm off')

    clf.fit(features[placed], y)
    proba = clf.predict_proba(features)
    placement = clf.classes_[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)
    placement[placed] = y
    confidence[placed] = 1.
    return {'arena_bounds': placement_bounds[placement], 'placement': placement, 'confidence': confidence,
            'placement_bounds': placement_bounds, 'cv_accuracy': cv_accuracy,
            'cv_frac_wall_off_2cm': cv_frac_wall_off_2cm}
