"""Converts the MABe22 mouse triplet keypoints to the .npz files APF trains on.

The MABe22 data (https://data.caltech.edu/records/rdsa8-rde65) has four keypoint files, from two
challenges: mouse_user_train.npy and mouse_submission_keypoints.npy, which have videos, and
mouse_triplet_train.npy and mouse_triplet_test.npy, which are pose only.  Each holds one-minute sequences
of three mice, 1800 frames at 30 Hz, as (frame, mouse, keypoint, (y, x)) int pixels.  The pixels are
in a 512 x 512 frame in the files with videos, and in the camera's 800 x 800 frame in the pose-only
files (IN_FILE_FRAME_SIZES_PX).  Missing keypoints are (0, 0).

This pools the sequences of all four files, drops all-zero and duplicate sequences, and splits the rest
at random into train, val and test sets, by sequence.  For each sequence and mouse, it:
- marks keypoints that are (0, 0) or outside the frame as missing,
- scales the pose-only files' pixels to the 512 x 512 frame,
- fills gaps of up to MAX_GAP_N_FRAMES frames in each keypoint by linear interpolation, if the keypoint
  moves no more than MAX_STEP_PX across the gap,
- counts a frame as tracked if all the mouse's keypoints are then present,
- starts a new track where the center keypoint jumps more than MAX_STEP_PX in one frame, which is faster
  than a mouse runs, so is an identity swap or a tracking error, or where the heading, from the tail base
  to the neck, turns more than MAX_TURN_RAD in one frame, which is a pose flip, and
- converts pixels to cm about the typical arena center (mousellm.config).

The output files have the fields apf.data.load_raw_npz_data reads.  'ids' numbers each track: a run of
tracked frames of one mouse with no jumps.  It is -1 for untracked frames, whose keypoints are NaN.
'videoidx' is the sequence's index in the file, and 'frames' the frame within the sequence.  There are no
categories.  The files also have 'seqnames' and 'seqsourcefiles', the MABe22 name and file of each
sequence, by videoidx.  A JSON file lists the sequences in each split.

Run it as python -m mousellm.convert_mabe22.
"""
import argparse
import hashlib
import json
import logging
import os

import numpy as np

from mousellm.config import ARENA_CENTER_PX, CM_PER_PX, keypointnames, kpheading_back, kpheading_front, kporigin

LOG = logging.getLogger('mousellm.convert_mabe22')

DEFAULT_IN_DIR_PATH = '/nrs/branson/AnimalPoseForecasting/mabe22-mouse'
DEFAULT_OUT_DIR_PATH = '/nrs/branson/AnimalPoseForecasting/mabe22-mouse-apf-ready'
# The size of the frame each file's keypoint pixels are in
IN_FILE_FRAME_SIZES_PX = {'mouse_user_train.npy': 512, 'mouse_submission_keypoints.npy': 512,
                          'mouse_triplet_train.npy': 800, 'mouse_triplet_test.npy': 800}
IN_FILE_NAMES = list(IN_FILE_FRAME_SIZES_PX)
# The frame size mousellm.config's pixel units are in
FRAME_SIZE_PX = 512
SPLIT_NAMES = ['train', 'val', 'test']
SPLIT_FRACTIONS = [.8, .1, .1]

# Longest keypoint gap that is filled by interpolation, in frames
MAX_GAP_N_FRAMES = 5
# Largest believable movement of a keypoint in one frame, in 512-frame pixels: about 4 cm, or 1.2 m/s
MAX_STEP_PX = 30.
# Largest believable turn of the heading in one frame, in radians: 90 degrees, or 2700 degrees/s
MAX_TURN_RAD = np.pi / 2


def nan_runs(isnan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ The [start, end) frames of each run of True in the 1-d bool array isnan. """
    d = np.diff(np.r_[0, isnan.astype(np.int8), 0])
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def fill_short_gaps_bang(x: np.ndarray) -> int:
    """ Fills the gaps in one keypoint's track, x, (n_frames, 2) with NaNs where missing, that are at most
    MAX_GAP_N_FRAMES frames long, are between tracked frames, and across which the keypoint moves at most
    MAX_STEP_PX, by linear interpolation, in place.  Returns the number of frames filled. """
    starts, ends = nan_runs(np.isnan(x[:, 0]))
    n_filled = 0
    for s, e in zip(starts, ends):
        if s == 0 or e == len(x) or e - s > MAX_GAP_N_FRAMES:
            continue
        if np.linalg.norm(x[e] - x[s - 1]) > MAX_STEP_PX:
            continue
        w = (np.arange(s, e) - (s - 1))[:, None] / (e - (s - 1))
        x[s:e] = (1 - w) * x[s - 1] + w * x[e]
        n_filled += int(e - s)
    return n_filled


def clean_sequence(kp: np.ndarray, frame_size_px: int, first_id: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """ Cleans one MABe22 sequence's keypoints and splits each mouse's frames into tracks.

    Args:
        kp: (n_frames, n_mice, n_keypoints, 2) int keypoints, (y, x) pixels, (0, 0) where missing.
        frame_size_px: the size of the frame kp's pixels are in.
        first_id: the id to give the sequence's first track.

    Returns:
        X: (n_keypoints, 2, n_frames, n_mice) float32 (x, y) positions in cm, NaN where untracked.
        ids: (n_frames, n_mice) int track ids, from first_id, -1 where untracked.
        stats: counts of missing, filled and untracked keypoints and frames, and of jumps and flips.
    """
    xy = kp[..., ::-1].astype(np.float64)
    missing = (kp == 0).all(-1) | (xy < 0).any(-1) | (xy >= frame_size_px).any(-1)
    xy[missing] = np.nan
    xy *= FRAME_SIZE_PX / frame_size_px
    n_frames, n_mice, n_kpts, _ = xy.shape
    stats = {'n_keypoints_missing': int(missing.sum()), 'n_keypoints_filled': 0, 'n_jumps': 0}
    for m in range(n_mice):
        for k in range(n_kpts):
            stats['n_keypoints_filled'] += fill_short_gaps_bang(xy[:, m, k])
    tracked = ~np.isnan(xy).any(axis=(-1, -2))                          # (n_frames, n_mice)
    ids = np.full((n_frames, n_mice), -1, dtype=np.int64)
    next_id = first_id
    for m in range(n_mice):
        center = xy[:, m, kporigin]
        step = np.linalg.norm(np.diff(center, axis=0), axis=-1)
        heading = xy[:, m, kpheading_front] - xy[:, m, kpheading_back]
        theta = np.arctan2(heading[:, 1], heading[:, 0])
        turn = np.abs(np.angle(np.exp(1j * np.diff(theta))))
        # a track starts at each tracked frame after an untracked one, or after a jump or a flip
        isjump = np.r_[False, tracked[1:, m] & tracked[:-1, m] & ((step > MAX_STEP_PX) | (turn > MAX_TURN_RAD))]
        isstart = tracked[:, m] & (np.r_[True, ~tracked[:-1, m]] | isjump)
        stats['n_jumps'] += int(isjump.sum())
        trackidx = np.cumsum(isstart) - 1
        ids[tracked[:, m], m] = next_id + trackidx[tracked[:, m]]
        next_id += int(isstart.sum())
    stats['n_mouse_frames_untracked'] = int((~tracked).sum())
    X = (xy - ARENA_CENTER_PX) * CM_PER_PX
    X[~tracked] = np.nan
    return np.transpose(X, (2, 3, 0, 1)).astype(np.float32), ids, stats


def load_sequences(in_dir_path: str) -> list[tuple[str, str, np.ndarray]]:
    """ Loads the sequences of the four MABe22 mouse keypoint files, as (name, file name, keypoints),
    dropping all-zero sequences and repeats of a sequence's keypoints (keeping the first). """
    seqs = []
    seen = set()
    n_zero = n_repeat = 0
    for file_name in IN_FILE_NAMES:
        sequences = np.load(os.path.join(in_dir_path, file_name), allow_pickle=True).item()['sequences']
        for name in sorted(sequences):
            kp = sequences[name]['keypoints']
            if not np.any(kp):
                n_zero += 1
                continue
            h = hashlib.md5(np.ascontiguousarray(kp).tobytes()).hexdigest()
            if h in seen:
                n_repeat += 1
                continue
            seen.add(h)
            seqs.append((name, file_name, kp))
        LOG.info(f'Loaded {file_name}: {len(sequences)} sequences')
    LOG.info(f'Dropped {n_zero} all-zero sequences and {n_repeat} repeated sequences, leaving {len(seqs)}')
    return seqs


def split_sequences(n_seqs: int, seed: int) -> list[np.ndarray]:
    """ Splits sequence indices 0..n_seqs-1 at random into SPLIT_FRACTIONS, sorted within each split. """
    perm = np.random.default_rng(seed).permutation(n_seqs)
    ends = np.round(np.cumsum(SPLIT_FRACTIONS) * n_seqs).astype(int)
    return [np.sort(perm[s:e]) for s, e in zip(np.r_[0, ends[:-1]], ends)]


def convert_split(seqs: list[tuple[str, str, np.ndarray]]) -> tuple[dict, dict]:
    """ Converts a list of (name, file name, keypoints) sequences to the fields of one output file. """
    Xs, idss = [], []
    stats = {}
    next_id = 0
    for name, file_name, kp in seqs:
        X, ids, s = clean_sequence(kp, IN_FILE_FRAME_SIZES_PX[file_name], next_id)
        next_id = max(next_id, ids.max() + 1)
        Xs.append(X)
        idss.append(ids)
        for k, v in s.items():
            stats[k] = stats.get(k, 0) + v
    n_frames_per_seq = [X.shape[2] for X in Xs]
    data = {
        'X': np.concatenate(Xs, axis=2),
        'ids': np.concatenate(idss, axis=0),
        'frames': np.concatenate([np.arange(n) for n in n_frames_per_seq])[:, None],
        'videoidx': np.repeat(np.arange(len(seqs)), n_frames_per_seq)[:, None],
        'y': np.zeros((0, sum(n_frames_per_seq), Xs[0].shape[-1]), dtype=np.float32),
        'categories': np.array([], dtype=str),
        'kpnames': np.array(keypointnames),
        'seqnames': np.array([name for name, _, _ in seqs]),
        'seqsourcefiles': np.array([file_name for _, file_name, _ in seqs]),
    }
    stats['n_seqs'] = len(seqs)
    stats['n_frames'] = sum(n_frames_per_seq)
    stats['n_tracks'] = int(next_id)
    return data, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--indir', default=DEFAULT_IN_DIR_PATH, help=f'MABe22 mouse data dir (default: {DEFAULT_IN_DIR_PATH})')
    parser.add_argument('--outdir', default=DEFAULT_OUT_DIR_PATH, help=f'output dir (default: {DEFAULT_OUT_DIR_PATH})')
    parser.add_argument('--seed', type=int, default=0, help='random seed for the split (default: 0)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    seqs = load_sequences(args.indir)
    os.makedirs(args.outdir, exist_ok=True)
    manifest = {'seed': args.seed, 'max_gap_n_frames': MAX_GAP_N_FRAMES, 'max_step_px': MAX_STEP_PX,
                'max_turn_rad': MAX_TURN_RAD,
                'cm_per_px': CM_PER_PX, 'arena_center_px': ARENA_CENTER_PX.tolist(), 'splits': {}}
    for split_name, idx in zip(SPLIT_NAMES, split_sequences(len(seqs), args.seed)):
        split_seqs = [seqs[i] for i in idx]
        data, stats = convert_split(split_seqs)
        out_file_path = os.path.join(args.outdir, f'{split_name}.npz')
        # write to a temporary file and rename, so an interrupted run never leaves a partial file
        tmp_file_path = f'{out_file_path}.{os.getpid()}.tmp.npz'
        np.savez(tmp_file_path, **data)
        os.replace(tmp_file_path, out_file_path)
        n_mouse_frames = stats['n_frames'] * data['X'].shape[-1]
        LOG.info(f"Wrote {out_file_path}: {stats['n_seqs']} sequences, {stats['n_frames']} frames, "
                 f"{stats['n_tracks']} tracks, {stats['n_mouse_frames_untracked'] / n_mouse_frames:.1%} of "
                 f"mouse-frames untracked, {stats['n_keypoints_filled']} of {stats['n_keypoints_missing']} "
                 f"missing keypoints filled, {stats['n_jumps']} jumps or flips")
        manifest['splits'][split_name] = {'stats': stats,
                                          'sequences': [[name, file_name] for name, file_name, _ in split_seqs]}
    manifest_file_path = os.path.join(args.outdir, 'splits.json')
    with open(manifest_file_path, 'w') as f:
        json.dump(manifest, f, indent=1)
    LOG.info(f'Wrote {manifest_file_path}')


if __name__ == '__main__':
    main()
