"""Score APF-simulated fly tracks with trained JAABA behavior classifiers.

The forecasting models are evaluated by open-loop simulation: a model is prompted with
`contextl` frames of real tracking and then predicts the following `sim_len` frames on
its own, for a subset of the flies in the arena. Those predictions are cached on disk as
one `.npy` per window, holding one tracklet per simulated agent. For each window this
module takes the real segment (prompt plus simulated stretch) and the same segment with
the simulated agents' tracklets substituted, runs JAABA classifiers (chase / wing
extension / courtship) over both, and keeps the simulated stretch, so simulated behavior
can be compared against real behavior on matched agent-frames.

A tracklet here is a contiguous portion of one agent's trajectory.

Unlike the hidden-state probes in `experiments/full_evaluation.py`, a JAABA classifier is
independent of the forecasting model being evaluated: the same classifier is applied to
every model's output, and to the real data.

Running this requires two source trees on PYTHONPATH, in this order:

    PYTHONPATH=<eyrun_tree>:<this_repo>

`apf`, `flyllm` and `experiments` must resolve to the tree that produced the cached
simulations, because the cached windows index into a ground-truth track built by that
code; `jaaba_detect` resolves to this repo. Set PYTHONDONTWRITEBYTECODE=1 when the other
tree is read-only, otherwise every import silently fails to cache bytecode.

This file lives in notebooks/ rather than at the repo root on purpose. Python puts a
script's own directory first on sys.path, so a copy at the root would make this repo's
`apf`/`flyllm` shadow the ones on PYTHONPATH and `apf.evaluation` would not be found.
notebooks/ holds no package of those names, so the PYTHONPATH order is preserved.
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing
import os
import re
import time

import numpy as np

# Resolved from the tree that generated the cached simulations (see module docstring).
from apf.evaluation import load_sim_track
from flyllm.prepare import read_config, load_config_from_model_file, init_datasets

# Resolved from this repo.
from jaaba_detect import jab_io, jaaba_detect_from_track, flyllm_keypoint_names


CLASSIFIER_DIR = "/groups/branson/home/bransonk/behavioranalysis/code/MABe2022"

# Where `score` writes jaaba_scores_<nickname>_<classifier set>.npz, one file per model.
# A README there describes the arrays.
RESULTS_PARENT_DIR = "/nrs/branson/AnimalPoseForecasting/jaaba_scores"

# behavior -> the stem every classifier file for it starts with. The 'jaaba_' prefix keeps
# these behaviors distinct from the 'perframe_*' hidden-state probes, which score the
# same three behaviors by a completely different route.
CLASSIFIER_STEM = {
    'jaaba_chase': 'chase_apt',
    'jaaba_wingext': 'wingextension_apt',
    'jaaba_courtship': 'courtship_v2pt3_apt',
}

# Classifier sets, and the files each names in CLASSIFIER_DIR:
#   'original'            <stem>.classifier.mat                      radius 25
#   'originalsplit'       <stem>_split.classifier.mat                radius 25, held out
#   'r<N>'                <stem>_r<N>.classifier.mat                 radius N
#   'r<N>split'           <stem>_r<N>_split.classifier.mat           radius N, held out
#   'r<N>nowingtip'       <stem>_r<N>_nowingtip.classifier.mat       radius N, no wing tips
#   'r<N>nowingtipsplit'  <stem>_r<N>_nowingtip_split.classifier.mat as above, held out
# "Held out" sets were trained with test1/test2 held out, so only their accuracy on those
# splits is a held-out measure; the others were trained on every labeled experiment and
# are the stronger classifiers to score with. "No wing tips" sets were trained without
# the per-frame features that reference the outer wing tips (APT landmarks 19, 21),
# which simulated tracks lack; they are pruned with jaaba_detect/prune_apt_landmarks.m.
#
# Only N = 5 exists, trained in the JAABA UI; r5nowingtip is the set to score
# simulations with. Radius 5 is used because a classifier's window features reach 2*r+3
# frames either side of the frame they score (3 being the widest change-window radius):
# 53 frames at radius 25, 13 at radius 5. That reach is the span over which a simulated
# frame's classification also sees the real prompt preceding it, so radius 5 cuts the
# affected part of a 512-frame simulation from 10.4% to 2.5%, at about the same held-out
# accuracy as radius 25.
#
# Held-out balanced accuracy (chase / wingext / courtship):
#   r5split            0.859 / 0.938 / 0.981 in the JAABA UI
#   r5nowingtipsplit   0.863 / 0.915 / 0.962 in the JAABA UI; 0.823 / 0.837 / 0.831 on
#                      19-keypoint tracks, whose body ellipse has to be rebuilt from
#                      keypoints
# The nowingtip sets' social_* features still take the closest landmark on another fly
# among all 21, wing tips included, so they differ slightly on tracks without wing tips
# (accuracy moves by at most ~0.02).
_RADIUS_SET_NAME = re.compile(r"r(?P<radius>\d+)(?P<nowingtip>nowingtip)?(?P<split>split)?")

DEFAULT_CLASSIFIER_SET = 'r5nowingtip'


def classifier_set(name: str) -> dict:
    """Paths to one set of exported JAABA classifiers, all in CLASSIFIER_DIR.

    Args:
        name: 'original', 'originalsplit', or r<N>[nowingtip][split] as listed above,
            e.g. 'r5nowingtip' to score simulations, or 'r5nowingtipsplit' to measure
            that set's held-out accuracy.

    Returns:
        {behavior name: path to .classifier.mat}. The files are not checked for
        existence.

    Raises:
        ValueError: if the name is not one of those forms.
    """
    if name == 'original':
        suffix = ''
    elif name == 'originalsplit':
        suffix = '_split'
    else:
        match = _RADIUS_SET_NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"unknown classifier set {name!r}; expected 'original', "
                             f"'originalsplit' or r<N>[nowingtip][split]")
        suffix = (f"_r{match['radius']}" + ('_nowingtip' if match['nowingtip'] else '')
                  + ('_split' if match['split'] else ''))
    return {behavior: f"{CLASSIFIER_DIR}/{stem}{suffix}.classifier.mat"
            for behavior, stem in CLASSIFIER_STEM.items()}


def classifier_set_name(name: str) -> str:
    """argparse type for a classifier set name: returns it if classifier_set accepts it."""
    classifier_set(name)
    return name


# The raw MABe tracks carry two landmarks after the 19 flyllm keypoints: the outer wing
# tips (APT landmarks 19 and 21), which the forecasting models do not predict. flyllm's
# config lists them, commented out, in this order.
OUTER_WING_KEYPOINT_NAMES = ['right_outer_wing', 'left_outer_wing']

# Acquisition rate of the FlyBubble rigs, per the dataset datasheet. Not recorded in the
# APF configs, and needed for every velocity-derived per-frame feature.
FPS = 150.0

# Pixels per mm. The classifiers were trained on pixel tracking, and the raw APT social
# distance features are still in pixels, so the mm track is scaled back up by this.
# Tight across the FlyBubble dataset (18.86-19.09); jaaba_detect's validated default.
# Note flyllm.config.PXPERMM is 19.02, computed as a median arena radius instead.
PXPERMM = 18.9


# Root holding one subdirectory of cached simulation windows per model, each named
# "<configname>_<modelname>" where the names are the config/checkpoint basenames with
# their extensions stripped.
SIM_PARENT_DIR = "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/train_data/synthetic_test"

EYRUN_CODE_DIR = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting"
EYRUN_MODEL_DIR = "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/flyllm_models"
BRANSONK_CODE_DIR = "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting"

# The 8 models with cached simulations, as (config file, checkpoint file). Nicknames and
# the first seven entries follow load_sim_data.ipynb; 'alldata' has cached simulations but
# does not appear there. 'rawkp' cannot currently be loaded: it was trained and simulated
# with at most 10 flies per arena, via a fly-elimination step in experiments/flyllm.py
# that is now disabled, so its inputs come out wider (380) than its checkpoint's
# normalization (342). 'short' uses a 64-frame prompt; the others use 512.
EXPERIMENTS = {
    'ref': (
        f"{BRANSONK_CODE_DIR}/flyllm/configs/config_fly_llm_predvel_optimalbinning_20251113.json",
        f"{BRANSONK_CODE_DIR}/notebooks/flyllm_models/flypredvel_20251007_20251114T194024_bestepoch200.pth",
    ),
    'short': (
        f"{EYRUN_CODE_DIR}/config_fly_llm_predvel_optimalbinning_short_20260430.json",
        f"{EYRUN_MODEL_DIR}/flypredvel_optimal_binning_64_20260502T050431_bestepoch500.pth",
    ),
    'nobin': (
        f"{EYRUN_CODE_DIR}/config_fly_llm_predvel_nobinning_20260430.json",
        f"{EYRUN_MODEL_DIR}/flypredvel_nobinning_20260501T050753_bestepoch500.pth",
    ),
    'binall': (
        f"{EYRUN_CODE_DIR}/config_fly_llm_predvel_optimalbinning_20251113_binall.json",
        f"{EYRUN_MODEL_DIR}/flypredvel_optimal_binning_binall_20260605T113925_epoch205.pth",
    ),
    'predpose': (
        f"{EYRUN_CODE_DIR}/config_fly_llm_predvel_optimalbinning_globalvel_20260430.json",
        f"{EYRUN_MODEL_DIR}/flypredvelandpose_20260605T110442_epoch320.pth",
    ),
    'bodycentric': (
        f"{EYRUN_CODE_DIR}/config_fly_llm_predvel_optimalbinning_bodycentrickp_20260430.json",
        f"{EYRUN_MODEL_DIR}/flypredvel_body_centric_kp_20260501T060729_bestepoch500.pth",
    ),
    'rawkp': (
        f"{EYRUN_CODE_DIR}/config_fly_llm_predvel_raw_kp.json",
        f"{EYRUN_MODEL_DIR}/flypredvel_nobinning_20260619T070821_bestepoch500.pth",
    ),
    'alldata': (
        f"{EYRUN_CODE_DIR}/config_fly_llm_predvel_optimalbinning_20251113_alldata.json",
        f"{EYRUN_MODEL_DIR}/flypredvel_optbin_alldata_20260907T081727_epoch155.pth",
    ),
}

# The simulations were cached against the test2 split, whereas the configs name the
# testtrain split as their validation set.
VALIDATION_SPLIT = 'testtrain'
EVALUATION_SPLIT = 'test2'

# Cached window filenames, e.g.
#   session_10_startsimframe_136051_agentid_3_5_6_7_9_simlen_512.npy
_WINDOW_FILENAME = re.compile(
    r"session_(?P<session>\d+)"
    r"_startsimframe_(?P<start_frame>\d+)"
    r"_agentid_(?P<agents>[\d_]+)"
    r"_simlen_(?P<sim_len>\d+)\.npy$"
)


def sim_dir(configfile: str, modelfile: str) -> str:
    """Directory of cached simulation windows for one (config, checkpoint) pair.

    Args:
        configfile: path to the model's .json config.
        modelfile: path to the model's .pth checkpoint.

    Returns:
        Absolute path to the "<configname>_<modelname>" directory under SIM_PARENT_DIR.
        Not guaranteed to exist; callers should check.
    """
    config_name = os.path.basename(configfile).removesuffix('.json')
    model_name = os.path.basename(modelfile).removesuffix('.pth')
    return os.path.join(SIM_PARENT_DIR, f"{config_name}_{model_name}")


def parse_windows(sim_savedir: str) -> list[dict]:
    """Read the metadata encoded in every cached window filename.

    Args:
        sim_savedir: directory of cached "session_*.npy" windows.

    Returns:
        One dict per window, sorted by (session, start_frame), with keys:
            path (str), session (int), start_frame (int, first simulated frame),
            agents (list[int], simulated agent indices), sim_len (int, simulated frames).
        The file itself holds (len(agents), contextl + sim_len, 2, n_keypoints) float32
        mm keypoints: one tracklet per simulated agent, whose leading contextl frames are
        the real prompt.
    """
    windows = []
    for path in sorted(glob.glob(os.path.join(sim_savedir, '*.npy'))):
        match = _WINDOW_FILENAME.search(os.path.basename(path))
        if match is None:
            raise ValueError(f"unrecognised cached window filename: {path}")
        windows.append({
            'path': path,
            'session': int(match['session']),
            'start_frame': int(match['start_frame']),
            'agents': [int(a) for a in match['agents'].split('_')],
            'sim_len': int(match['sim_len']),
        })
    windows.sort(key=lambda w: (w['session'], w['start_frame']))
    return windows


def load_ground_truth(configfile: str, modelfile: str,
                      split: str = EVALUATION_SPLIT,
                      augment_flip: bool | None = None,
                      categories: list | None = None,
                      all_keypoints: bool = False) -> dict:
    """Load the ground-truth evaluation dataset the cached simulations were built from.

    Reproduces the loading recipe in load_sim_data.ipynb: read the config, overlay the
    settings stored in the checkpoint, then switch the validation split to the one the
    simulations used. Loading the checkpoint is required because it carries config fields
    (e.g. discretisation bins) that affect dataset construction.

    Args:
        configfile: path to the model's .json config.
        modelfile: path to the model's .pth checkpoint.
        split: which MABe split to load in place of the config's validation split,
            e.g. 'test2' or 'test1'. Both are held out of the *_split classifier sets;
            the all-data sets were trained on them.
        categories: override the config's category filter. The configs restrict the
            data to ['courtship', 'male'], which marks every other agent-frame invalid,
            and invalid frames drop out of the sessions the scoring blocks come from.
            That is the right population for the simulations, since those are the flies
            that were simulated, but it discards most of the behavior labels when
            measuring classifier accuracy. Pass [] to score every fly.
        augment_flip: override the config's flip augmentation. The configs enable it,
            which appends a mirror-image copy of the whole dataset, doubling the frame
            count. That belongs in training, not in evaluation, where it would count
            every labeled frame twice. Pass False when measuring accuracy. Leave None
            to keep the config's setting, which is required when working with the
            cached simulations, since their frame indices address the augmented track.
        all_keypoints: keep the two outer wing tips after the 19 flyllm keypoints. By
            default the track is trimmed to the 19 the simulations have, so the real
            track is scored the way simulated ones are. Keeping them scores it the way
            JAABA saw it in training. Requires flip augmentation off, since the loader
            trims to 19 inside its flip branch.

    Returns:
        dict with keys:
            kpt_names (list of str, the track's keypoint names in order),
            config (dict), dataset (apf.dataset.Dataset), data (dict of apf.dataset.Data
            and mask arrays), track (Data; .array is (n_agents, n_frames, 2, n_keypoints)
            float32 mm keypoints), contextl (int, prompt length in frames).

    Side effects:
        Reads the split .npz from the config's datadir and the checkpoint. The loaded
        dataset holds ~90-115 GB with flip augmentation on (~175 GB for 'alldata').
    """
    config = read_config(configfile)
    load_config_from_model_file(loadmodelfile=modelfile, config=config, weights_only=False)

    # The cached simulations were generated against the evaluation split, not the split
    # named in the config.
    for key in ('invalfilestr', 'invalfile'):
        config[key] = config[key].replace(VALIDATION_SPLIT, split)
    if augment_flip is not None:
        config['augment_flip'] = augment_flip
    if categories is not None:
        config['categories'] = list(categories)

    result = init_datasets(config, needtraindata=False, needvaldata=True,
                           res={'config': config}, debug_uselessdata=False)
    data = result['val_data']

    # The loader only trims the raw 21 APT keypoints down to the 19 flyllm ones inside
    # its flip-augmentation branch, so with augmentation off the track keeps all 21.
    # Trim here so the keypoint set does not depend on an augmentation setting, unless
    # the outer wings are wanted. The first 19 are the flyllm ones; the last two are the
    # outer wings, which the forecasting models do not predict.
    track = data['track']
    kpt_names = flyllm_keypoint_names()
    if all_keypoints:
        kpt_names = kpt_names + OUTER_WING_KEYPOINT_NAMES
        if track.array.shape[3] != len(kpt_names):
            raise ValueError(f"expected {len(kpt_names)} keypoints including the outer "
                             f"wings, got {track.array.shape[3]}; flip augmentation "
                             f"must be off to keep them")
    elif track.array.shape[3] > len(kpt_names):
        data['track'] = track._replace(array=track.array[:, :, :, :len(kpt_names)])
    return {
        'config': config,
        'dataset': result['val_dataset'],
        'data': data,
        'track': data['track'],
        'kpt_names': kpt_names,
        'contextl': int(config['contextl']),
    }


def load_simulated(track, sim_savedir: str, contextl: int) -> dict:
    """Insert cached simulated windows into a copy of the ground-truth track.

    A cached window holds tracklets only for the agents that were simulated in it, so it is
    not a usable track on its own -- JAABA's social features need every fly in the arena.
    The predictions are therefore written into a copy of the real track, leaving the
    unsimulated flies at their real positions. Used by `describe` for statistics only:
    scoring does not splice windows into one track, since neighbouring windows were
    simulated independently and the spliced track jumps at every boundary (see
    window_segment).

    Args:
        track: ground-truth Data whose .array is (n_agents, n_frames, 2, n_keypoints) mm.
        sim_savedir: directory of cached "session_*.npy" windows.
        contextl: prompt length in frames; the leading contextl frames of each cached
            window are the real prompt and are not inserted.

    Returns:
        dict with keys:
            sim_track (Data, same shape as `track`, real everywhere except simulated
                agent-frames), sim_frame ((n_agents, n_frames) float; 0 where the
                agent-frame is real, k for the k-th simulated frame after its prompt),
            n_windows (int, windows found), n_skipped (int, windows referring to agents
                outside the loaded track and therefore ignored).

    Raises:
        AssertionError: if a window's prompt frames do not match the ground-truth track,
            meaning the track differs from the one the simulations were generated against.
    """
    n_agents = track.array.shape[0]
    windows = parse_windows(sim_savedir)
    n_skipped = sum(1 for w in windows if max(w['agents']) >= n_agents)

    sim_track, sim_frame = load_sim_track(track, sim_savedir, contextl)
    return {
        'sim_track': sim_track,
        'sim_frame': sim_frame,
        'n_windows': len(windows),
        'n_skipped': n_skipped,
    }


def scoring_blocks(sessions, n_agents: int, n_frames: int,
                   min_frames: int = 1) -> list[dict]:
    """Split the timeline into intervals over which the set of tracked agents is fixed.

    The loaded track concatenates many videos end to end, so it must not be treated as
    one continuous trajectory: velocities would be meaningless across a join, and flies
    from different videos would appear to interact. `sessions` already encodes the
    breaks -- they are derived from `isstart`, which flags a new sequence wherever the
    tracked identity in an agent slot changes or the source video's frame numbering
    jumps -- so cutting wherever any agent's session changes gives intervals in which
    every present agent is continuously tracked.

    Cutting on any agent's change is deliberately conservative: it sometimes also cuts an
    agent that was continuous, which costs a little window context at the seam, but it
    can never let a window span two videos or two identities.

    Args:
        sessions: iterable of apf.dataset.Session (agent_id, start_frame, duration).
        n_agents: number of agent slots in the track.
        n_frames: number of frames in the track.
        min_frames: omit intervals shorter than this many frames.

    Returns:
        List of dicts with keys start (int), stop (int, exclusive) and agents
        (np.ndarray of agent indices tracked throughout the interval), ordered by start.
        Intervals with no tracked agent, or shorter than min_frames, are omitted, so
        their frames end up unscored.
    """
    # session_id[agent, frame] is 1-based so that 0 can mean "no data here"
    session_id = np.zeros((n_agents, n_frames), dtype=np.int32)
    for i, session in enumerate(sessions):
        stop = session.start_frame + session.duration
        session_id[session.agent_id, session.start_frame:stop] = i + 1

    changed = np.zeros(n_frames, dtype=bool)
    changed[1:] = (np.diff(session_id, axis=1) != 0).any(axis=0)
    edges = np.concatenate(([0], np.flatnonzero(changed), [n_frames]))

    blocks = []
    for start, stop in zip(edges[:-1], edges[1:]):
        agents = np.flatnonzero(session_id[:, start] > 0)
        if agents.size and stop - start >= min_frames:
            blocks.append({'start': int(start), 'stop': int(stop), 'agents': agents})
    return blocks


# Set before a parallel scoring run and inherited by forked workers, so the track (tens
# of GB) is shared copy-on-write rather than pickled to every worker.
_SCORING_TRACK = None
_SCORING_BLOCKS = None
_SCORING_CLASSIFIER = None
_SCORING_PXPERMM = None
_SCORING_FPS = None
_SCORING_KPT_NAMES = None


def _score_one_block(index: int) -> tuple:
    """Score a single block in a worker process.

    Args:
        index: position in the module-level block list set up by score_blocks().

    Returns:
        (block index, detector result dict) -- only the per-tracklet lists needed to
        place the scores back into the dense arrays, which is far smaller than the block.
    """
    block = _SCORING_BLOCKS[index]
    segment = _SCORING_TRACK[:, block['start']:block['stop']]
    result = jaaba_detect_from_track(segment, _SCORING_CLASSIFIER,
                                     kpt_names=_SCORING_KPT_NAMES,
                                     pxpermm=_SCORING_PXPERMM, fps=_SCORING_FPS,
                                     first_frame=1, verbose=False)
    return index, {'agents': result['agents'], 'tStart': result['tStart'],
                   'scores': result['scores'], 'postprocessed': result['postprocessed']}


def score_blocks(track_array: np.ndarray, blocks: list[dict], classifier,
                 *, pxpermm: float = PXPERMM, fps: float = FPS,
                 kpt_names: list | None = None,
                 verbose: bool = False, n_workers: int = 1) -> dict:
    """Run one JAABA classifier over a track, one scoring block at a time.

    Every agent in a block is passed to the classifier together: JAABA's social features,
    which chase and courtship rely on heavily, are defined relative to the other flies
    in the arena, so scoring a subset in isolation would change the answer. The detector
    splits each agent into tracklets at gaps and returns one result per tracklet.

    Used to score the real track for `validate`; simulated windows go through
    score_windows.

    Args:
        track_array: (n_agents, n_frames, 2, n_keypoints) float mm keypoints.
        blocks: output of scoring_blocks().
        classifier: path to an exported .classifier.mat, or a loaded jab_io.Classifier.
            Passing a loaded classifier avoids re-reading it for every block.
        pxpermm: pixels per mm used to rebuild the pixel frame the APT social distance
            features are expressed in.
        fps: acquisition frame rate, used for the per-frame time step.
        kpt_names: names of the track's keypoints, in order; None means the 19 flyllm
            keypoints. Landmarks are matched by name, so a 21-keypoint track with the
            outer wings named supplies real wing tips instead of mid-wing copies.
        verbose: forwarded to the detector when scoring serially.
        n_workers: processes to score blocks with. Blocks are independent, so this
            scales nearly linearly. Workers are forked so they share the track rather
            than copying it.

    Returns:
        dict with keys:
            scores ((n_agents, n_frames) float32, NaN where not scored),
            behavior ((n_agents, n_frames) float32, 1.0 where the behavior is on and 0.0
                elsewhere -- the dense layout apf.evaluation.detect_walk produces),
            scored ((n_agents, n_frames) bool, True where a score was produced).
    """
    global _SCORING_TRACK, _SCORING_BLOCKS, _SCORING_CLASSIFIER
    global _SCORING_PXPERMM, _SCORING_FPS, _SCORING_KPT_NAMES

    n_agents, n_frames = track_array.shape[:2]
    scores = np.full((n_agents, n_frames), np.nan, dtype=np.float32)
    behavior = np.zeros((n_agents, n_frames), dtype=np.float32)
    scored = np.zeros((n_agents, n_frames), dtype=bool)

    def place(index: int, result: dict) -> None:
        """Write one block's per-tracklet results into the dense arrays."""
        start = blocks[index]['start']
        for tracklet, agent in enumerate(result['agents']):
            tracklet_scores = np.asarray(result['scores'][tracklet], dtype=np.float32)
            # tStart is 1-based within the block (first_frame=1 when scoring)
            frame0 = start + int(result['tStart'][tracklet]) - 1
            frame1 = frame0 + tracklet_scores.size
            scores[agent, frame0:frame1] = tracklet_scores
            behavior[agent, frame0:frame1] = (
                np.asarray(result['postprocessed'][tracklet], dtype=np.float32) > 0)
            scored[agent, frame0:frame1] = True

    if n_workers <= 1:
        for index, block in enumerate(blocks):
            segment = track_array[:, block['start']:block['stop']]
            place(index, jaaba_detect_from_track(
                segment, classifier, kpt_names=kpt_names, pxpermm=pxpermm, fps=fps,
                first_frame=1, verbose=verbose))
    else:
        _SCORING_TRACK = track_array
        _SCORING_BLOCKS = blocks
        _SCORING_CLASSIFIER = classifier
        _SCORING_PXPERMM = pxpermm
        _SCORING_FPS = fps
        _SCORING_KPT_NAMES = kpt_names
        # Longest blocks first, so one straggler does not hold up the whole pool.
        order = sorted(range(len(blocks)),
                       key=lambda i: blocks[i]['stop'] - blocks[i]['start'], reverse=True)
        context = multiprocessing.get_context('fork')
        with context.Pool(min(n_workers, len(blocks))) as pool:
            for index, result in pool.imap_unordered(_score_one_block, order):
                place(index, result)
        _SCORING_TRACK = _SCORING_BLOCKS = _SCORING_CLASSIFIER = None
        _SCORING_KPT_NAMES = None

    return {'scores': scores, 'behavior': behavior, 'scored': scored}


def _score(nickname: str, max_windows: int | None, out_dir: str,
           n_workers: int, set_name: str) -> None:
    """Score one experiment's simulated windows and their ground-truth counterparts.

    Writes one .npz holding, per behavior, dense (n_agents, n_frames) arrays for the
    ground-truth and simulated sides plus sim_frame, so downstream analysis can select
    frames by how far the simulation had run unaided.
    """
    configfile, modelfile = EXPERIMENTS[nickname]
    savedir = sim_dir(configfile, modelfile)
    classifiers = classifier_set(set_name)
    print(f"=== {nickname}, classifier set {set_name!r} ===", flush=True)

    # Flip augmentation stays on: the cached windows' frame indices address the
    # augmented track, and half of them fall in its mirrored half.
    ground_truth = load_ground_truth(configfile, modelfile)
    track = ground_truth['track']
    contextl = ground_truth['contextl']
    n_agents, n_frames = track.array.shape[:2]
    print(f"  track {track.array.shape}, contextl {contextl}", flush=True)

    windows = parse_windows(savedir)
    windows = usable_windows(windows, n_agents, n_frames, contextl)
    if max_windows is not None:
        windows = windows[:max_windows]
    print(f"  {len(windows)} usable windows of {len(parse_windows(savedir))}", flush=True)

    started = time.time()
    print(f"  verified {verify_window_prompts(track.array, windows, contextl)} prompts "
          f"against the ground truth in {time.time() - started:.1f}s", flush=True)

    sim_frame = np.zeros((n_agents, n_frames), dtype=np.float32)
    for window in windows:
        stop = window['start_frame'] + window['sim_len']
        sim_frame[window['agents'], window['start_frame']:stop] = (
            np.arange(window['sim_len'], dtype=np.float32) + 1)[None, :]
    print(f"  {int((sim_frame > 0).sum())} simulated agent-frames", flush=True)

    arrays = {'sim_frame': sim_frame}
    for behavior, path in classifiers.items():
        classifier = jab_io.load_classifier(path)
        for label, simulated in (('gt', False), ('sim', True)):
            started = time.time()
            result = score_windows(track.array, windows, contextl, classifier,
                                   simulated=simulated, n_workers=n_workers)
            positive = int((result['behavior'] > 0).sum())
            total = int(result['scored'].sum())
            print(f"    {behavior:<16} {label:<4} {time.time() - started:6.1f}s  "
                  f"{positive}/{total} frames positive "
                  f"({100.0 * positive / max(total, 1):.3f}%)", flush=True)
            arrays[f"{label}_behavior_{behavior}"] = result['behavior']
            arrays[f"{label}_scores_{behavior}"] = result['scores']
            arrays[f"{label}_scored_{behavior}"] = result['scored']

    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"jaaba_scores_{nickname}_{set_name}.npz")
    np.savez_compressed(outpath, **arrays)
    print(f"  wrote {outpath}", flush=True)


# The sparse per-frame behavior annotations shipped in the dataset's `y` array, keyed by
# the JAABA classifier expected to reproduce them. Values are 1 (positive), 0 (negative)
# or NaN (unlabeled); only a small fraction of frames carry a label.
LABEL_FOR_BEHAVIOR = {
    'jaaba_chase': 'perframe_chase',
    'jaaba_wingext': 'perframe_wingext',
    'jaaba_courtship': 'perframe_courtship',
}


def _agent_major(array: np.ndarray, n_agents: int, n_frames: int) -> np.ndarray:
    """Return a 2-D per-agent-per-frame array as (n_agents, n_frames)."""
    if array.shape == (n_agents, n_frames):
        return array
    if array.shape == (n_frames, n_agents):
        return array.T
    raise ValueError(f"expected ({n_agents}, {n_frames}) or ({n_frames}, {n_agents}); "
                     f"got {array.shape}")


def _validate(nickname: str, max_blocks: int | None, set_names: list,
              splits: list, n_workers: int, all_keypoints: bool = False) -> None:
    """Score the held-out splits and compare against the dataset's own annotations.

    The dataset ships sparse per-frame labels for these three behaviors, so scoring the
    real track and comparing is both an end-to-end check of the pipeline -- keypoint
    naming, mm units, pixels per mm, frame rate, ellipse reconstruction -- and the way
    the retrained classifiers are compared against each other, since every classifier
    set is scored over identical blocks.

    Confusion counts are pooled across splits before any rate is computed, so pooling
    test1 and test2 gives the same quantity as evaluating them together rather than an
    average of two rates. Only blocks holding at least one labeled frame and at least
    2 * window_reach + 1 frames long, for the classifier being scored, are scored; the
    progress line reports how many labeled frames that leaves out.

    Args:
        nickname: which experiment's config supplies the data paths. The real track
            depends only on the config's split and category filter, not on the model.
        max_blocks: score at most this many labeled blocks per behavior, for quick runs.
        set_names: classifier sets to compare, as accepted by classifier_set().
        splits: MABe splits whose labeled frames are pooled, normally ['test1', 'test2'].
            They are held out of the *_split classifier sets only; for an all-data set
            the result is not a held-out accuracy.
        n_workers: processes to score blocks with.
        all_keypoints: score the real track with its outer wing tips, as JAABA saw it in
            training, instead of trimmed to the 19 keypoints simulations have.
    """
    configfile, modelfile = EXPERIMENTS[nickname]
    # A set is scored for whichever behaviors are exported, so classifiers retrained one
    # at a time can be checked as each is finished.
    available = [name for name in set_names
                 if any(os.path.exists(path) for path in classifier_set(name).values())]
    for name in set_names:
        missing = [os.path.basename(path) for path in classifier_set(name).values()
                   if not os.path.exists(path)]
        if missing:
            what = 'skipping those behaviors' if name in available else 'skipping the set'
            print(f"  classifier set {name!r}: not exported yet {missing}, {what}",
                  flush=True)
    if not available:
        raise SystemExit("no classifier sets available to score")
    keypoint_note = ('21 keypoints (real outer wings)' if all_keypoints
                     else '19 keypoints (outer wings replaced by mid-wing copies)')
    print(f"  scoring with {keypoint_note}", flush=True)

    # counts[(set, behavior)] = {'tp':..,'fp':..,'fn':..,'tn':..}; pooled over splits
    counts: dict = {}
    reaches: dict = {}

    for split in splits:
        print(f"\n---- loading {split} ----", flush=True)
        # No category filter and no flip augmentation: accuracy is measured over every
        # labeled fly, on real data only.
        ground_truth = load_ground_truth(configfile, modelfile, split=split,
                                         augment_flip=False, categories=[],
                                         all_keypoints=all_keypoints)
        track = ground_truth['track']
        n_agents, n_frames = track.array.shape[:2]
        data = ground_truth['data']
        categories = [str(c) for c in data['categories']]
        all_blocks = scoring_blocks(ground_truth['dataset'].sessions, n_agents, n_frames)
        print(f"  track {track.array.shape}, {len(all_blocks)} blocks", flush=True)

        for set_name in available:
            classifiers = classifier_set(set_name)
            print(f"\n  ---- classifier set {set_name!r} ----", flush=True)
            for behavior, label_name in LABEL_FOR_BEHAVIOR.items():
                if not os.path.exists(classifiers[behavior]):
                    continue
                if label_name not in categories:
                    print(f"    {behavior}: no {label_name!r} category, skipping",
                          flush=True)
                    continue
                label = _agent_major(
                    np.asarray(data['labels'][categories.index(label_name)], dtype=float),
                    n_agents, n_frames)
                labeled = np.isfinite(label)               # (n_agents, n_frames) bool

                classifier = jab_io.load_classifier(classifiers[behavior])
                reach = window_reach(classifier)
                reaches[set_name] = max(reaches.get(set_name, 0), reach)
                # A block shorter than 2*reach+1 frames has no frame whose windows all fit
                # inside it. Its frames would be scored from truncated windows, and window
                # features that cannot reach any frame come out NaN, which sends their
                # stumps to the -1 branch -- biasing such blocks toward negative. They are
                # left unscored and out of the denominator instead.
                min_frames = 2 * reach + 1
                blocks = [b for b in all_blocks
                          if b['stop'] - b['start'] >= min_frames
                          and labeled[:, b['start']:b['stop']].any()]
                if max_blocks is not None:
                    blocks = blocks[:max_blocks]

                started = time.time()
                result = score_blocks(track.array, blocks, classifier,
                                      kpt_names=ground_truth['kpt_names'],
                                      n_workers=n_workers)
                elapsed = time.time() - started

                predicted = result['behavior'] > 0
                comparable = labeled & result['scored']
                split_counts = {
                    'tp': int((predicted & (label == 1) & comparable).sum()),
                    'fp': int((predicted & (label == 0) & comparable).sum()),
                    'fn': int((~predicted & (label == 1) & comparable).sum()),
                    'tn': int((~predicted & (label == 0) & comparable).sum()),
                }
                running = counts.setdefault((set_name, behavior),
                                            {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
                for key, value in split_counts.items():
                    running[key] += value
                # progress only: accuracy is reported once, on the pooled counts
                print(f"    {behavior:<16} {elapsed:6.1f}s  "
                      f"{int(comparable.sum())} labeled frames scored, "
                      f"{int(labeled.sum() - comparable.sum())} not (outside blocks of "
                      f">= {min_frames} frames)", flush=True)
        del ground_truth

    _print_validation_table(counts, reaches, splits)


def _rates(count: dict) -> dict:
    """Recall, specificity, precision and balanced accuracy from confusion counts."""
    n_positive = count['tp'] + count['fn']
    n_negative = count['tn'] + count['fp']
    n_predicted = count['tp'] + count['fp']
    recall = count['tp'] / n_positive if n_positive else float('nan')
    specificity = count['tn'] / n_negative if n_negative else float('nan')
    precision = count['tp'] / n_predicted if n_predicted else float('nan')
    return {'recall': recall, 'specificity': specificity, 'precision': precision,
            'balanced_accuracy': 0.5 * (recall + specificity),
            'n_labeled': n_positive + n_negative}


def _print_validation_table(counts: dict, reaches: dict, splits: list) -> None:
    """Print the comparison across classifier sets.

    Args:
        counts: {(classifier set, behavior): {'tp','fp','fn','tn'}}, pooled over splits.
        reaches: {classifier set: window reach in frames}.
        splits: the splits the counts were pooled over, for the heading.
    """
    if not counts:
        print("\nnothing was scored")
        return
    sets = list(dict.fromkeys(key[0] for key in counts))
    behaviors = list(dict.fromkeys(key[1] for key in counts))

    print(f"\n=== held-out accuracy on {'+'.join(splits)} ===")
    print(f"{'classifier':<12} {'reach':>6} {'behavior':<18} {'labeled':>9} "
          f"{'recall':>8} {'specif':>8} {'precis':>8} {'bal acc':>8}")
    print('-' * 80)
    for set_name in sets:
        for behavior in behaviors:
            count = counts.get((set_name, behavior))
            if count is None:
                continue
            rate = _rates(count)
            print(f"{set_name:<12} {reaches.get(set_name, 0):>5}f "
                  f"{behavior:<18} {rate['n_labeled']:>9d} "
                  f"{rate['recall']:>8.4f} {rate['specificity']:>8.4f} "
                  f"{rate['precision']:>8.4f} {rate['balanced_accuracy']:>8.4f}")
        print()

    print("balanced accuracy, classifier set x behavior:")
    header = f"{'':<14}" + ''.join(f"{b.replace('jaaba_',''):>14}" for b in behaviors)
    print(header)
    for set_name in sets:
        line = f"{set_name:<14}"
        for behavior in behaviors:
            count = counts.get((set_name, behavior))
            line += ('-'.rjust(14) if count is None
                     else f"{_rates(count)['balanced_accuracy']:>14.4f}")
        print(line)


def window_reach(classifier) -> int:
    """Frames a classifier's window features reach either side of the frame they score.

    Args:
        classifier: a loaded jab_io.Classifier.

    Returns:
        Maximum of |offset| + radius over the classifier's window features, widened by
        change_window_radius where the 'change' statistic uses it. A frame closer than
        this to the edge of the scored trajectory has a truncated window.
    """
    reach = 0
    for desc in classifier.unique_descs:
        pad = int(dict(desc.extra).get('change_window_radius', 0))
        reach = max(reach, abs(desc.offset) + desc.radius + pad)
    return reach


def window_segment(gt_array: np.ndarray, window: dict, contextl: int,
                   tracklets: np.ndarray | None) -> np.ndarray:
    """Assemble the keypoint track for one cached simulation window.

    The segment runs from contextl frames before the simulation starts to the end of the
    simulated stretch. Every agent is taken from the ground truth; when tracklets are
    given, the simulated agents are replaced by them over the whole segment. Each cached
    tracklet holds the real prompt in its leading contextl frames -- checked against the
    ground truth by verify_window_prompts -- so the segment is continuous across the
    prompt-to-simulation join.

    Scoring this segment on its own, rather than scoring a track with every window
    spliced in, is what avoids the discontinuity between neighbouring windows: the
    windows tile at stride sim_len, and each was simulated independently from its own
    real starting state, so a spliced track jumps at every window boundary.

    Args:
        gt_array: (n_agents, n_frames, 2, n_keypoints) float mm ground-truth keypoints.
        window: one entry from parse_windows().
        contextl: prompt length in frames.
        tracklets: (len(window['agents']), contextl + sim_len, 2, n_keypoints), one
            tracklet per simulated agent (real prompt, then predictions), or None to build
            the ground-truth counterpart of the same segment.

    Returns:
        (n_agents, contextl + sim_len, 2, n_keypoints) float mm keypoints.
    """
    start = window['start_frame'] - contextl
    stop = window['start_frame'] + window['sim_len']
    segment = gt_array[:, start:stop].astype(float, copy=True)
    if tracklets is not None:
        segment[window['agents']] = tracklets
    return segment


# Set before a parallel window run and inherited by forked workers, as for the blocks.
_WINDOW_TRACK = None
_WINDOW_LIST = None
_WINDOW_CONTEXTL = None
_WINDOW_SIMULATED = None


def _score_one_window(index: int) -> tuple:
    """Score one cached window in a worker process; returns per-tracklet results."""
    window = _WINDOW_LIST[index]
    tracklets = np.load(window['path']) if _WINDOW_SIMULATED else None
    segment = window_segment(_WINDOW_TRACK, window, _WINDOW_CONTEXTL, tracklets)
    result = jaaba_detect_from_track(segment, _SCORING_CLASSIFIER,
                                     pxpermm=_SCORING_PXPERMM, fps=_SCORING_FPS,
                                     first_frame=1, verbose=False)
    return index, {'agents': result['agents'], 'tStart': result['tStart'],
                   'scores': result['scores'], 'postprocessed': result['postprocessed']}


def score_windows(gt_array: np.ndarray, windows: list[dict], contextl: int,
                  classifier, *, simulated: bool, pxpermm: float = PXPERMM,
                  fps: float = FPS, n_workers: int = 1) -> dict:
    """Score one classifier over every cached window, ground truth or simulated.

    Each window is scored as its own trajectory: the real prompt concatenated with the
    simulated stretch, contextl + sim_len frames (512 + 512 for most models, 64 + 512
    for 'short'). That is what keeps the trajectory
    continuous -- the windows tile at stride sim_len and each was simulated
    independently from its own real starting state, so a track with all of them spliced
    in jumps at every window boundary.

    Only the simulated stretch is recorded; the prompt frames are context for the window
    features, not results. Ground truth is scored over identical segments so the two
    sides see the same sequence structure.

    Note that JAABA's 'relative' transform takes its percentile bins from the whole
    scored trajectory, so for the simulated side those bins come from a sequence that is
    half real. Ground truth's come from an all-real sequence. This is a deliberate
    choice to give the early simulated frames genuine context.

    Args:
        gt_array: (n_agents, n_frames, 2, n_keypoints) float mm ground-truth keypoints.
        windows: entries from parse_windows() that fit inside the track.
        contextl: prompt length in frames.
        classifier: a loaded jab_io.Classifier.
        simulated: True to substitute the cached predictions, False for ground truth.
        pxpermm: pixels per mm for the APT social distance features.
        fps: acquisition frame rate.
        n_workers: processes to score windows with; windows are independent.

    Returns:
        dict with (n_agents, n_frames) arrays 'scores' (NaN where not scored),
        'behavior' (1.0 where the behavior is on, else 0.0) and 'scored' (bool).
    """
    global _WINDOW_TRACK, _WINDOW_LIST, _WINDOW_CONTEXTL, _WINDOW_SIMULATED
    global _SCORING_CLASSIFIER, _SCORING_PXPERMM, _SCORING_FPS

    n_agents, n_frames = gt_array.shape[:2]
    scores = np.full((n_agents, n_frames), np.nan, dtype=np.float32)
    behavior = np.zeros((n_agents, n_frames), dtype=np.float32)
    scored = np.zeros((n_agents, n_frames), dtype=bool)

    def place(index: int, result: dict) -> None:
        """Write one window's per-tracklet, simulated-stretch results into the arrays."""
        window = windows[index]
        for tracklet, agent in enumerate(result['agents']):
            tracklet_scores = np.asarray(result['scores'][tracklet], dtype=np.float32)
            tracklet_behavior = (
                np.asarray(result['postprocessed'][tracklet], dtype=np.float32) > 0)
            # tStart is 1-based within the segment; shifting by contextl puts it in the
            # simulated stretch, where out-of-range means prompt or past the end.
            offset = int(result['tStart'][tracklet]) - 1
            simulated_index = np.arange(tracklet_scores.size) + offset - contextl
            keep = (simulated_index >= 0) & (simulated_index < window['sim_len'])
            frames = window['start_frame'] + simulated_index[keep]
            scores[agent, frames] = tracklet_scores[keep]
            behavior[agent, frames] = tracklet_behavior[keep]
            scored[agent, frames] = True

    _SCORING_CLASSIFIER = classifier
    _SCORING_PXPERMM = pxpermm
    _SCORING_FPS = fps
    if n_workers <= 1:
        for index, window in enumerate(windows):
            tracklets = np.load(window['path']) if simulated else None
            segment = window_segment(gt_array, window, contextl, tracklets)
            place(index, jaaba_detect_from_track(segment, classifier, pxpermm=pxpermm,
                                                 fps=fps, first_frame=1, verbose=False))
    else:
        _WINDOW_TRACK = gt_array
        _WINDOW_LIST = windows
        _WINDOW_CONTEXTL = contextl
        _WINDOW_SIMULATED = simulated
        context = multiprocessing.get_context('fork')
        with context.Pool(min(n_workers, len(windows))) as pool:
            for index, result in pool.imap_unordered(_score_one_window,
                                                     range(len(windows)), chunksize=4):
                place(index, result)
        _WINDOW_TRACK = _WINDOW_LIST = None

    return {'scores': scores, 'behavior': behavior, 'scored': scored}


def usable_windows(windows: list[dict], n_agents: int, n_frames: int,
                   contextl: int) -> list[dict]:
    """Keep the windows that address agents and frames the loaded track actually has."""
    return [w for w in windows
            if max(w['agents']) < n_agents
            and w['start_frame'] - contextl >= 0
            and w['start_frame'] + w['sim_len'] <= n_frames]


def verify_window_prompts(gt_array: np.ndarray, windows: list[dict],
                          contextl: int) -> int:
    """Check every cached window's prompt against the ground-truth track.

    Each cached window's tracklets store the real frames they were prompted with. If the
    dataset code
    no longer reproduces the track the simulations were generated against, those frames
    will not match and every downstream frame index would be wrong, so this is a hard
    gate rather than a diagnostic.

    Args:
        gt_array: (n_agents, n_frames, 2, n_keypoints) float mm ground-truth keypoints.
        windows: entries from parse_windows(), already filtered by usable_windows().
        contextl: prompt length in frames.

    Returns:
        The number of windows checked.

    Raises:
        AssertionError: naming the first window whose prompt does not match.
    """
    for window in windows:
        tracklets = np.load(window['path'], mmap_mode='r')
        start = window['start_frame'] - contextl
        expected = gt_array[window['agents'], start:window['start_frame']]
        assert np.allclose(expected, tracklets[:, :contextl], equal_nan=True), (
            f"prompt frames do not match the ground truth for {window['path']}; "
            f"the loaded track is not the one the simulations were generated against")
    return len(windows)


def _describe(nickname: str) -> None:
    """Load one experiment's ground truth and simulations and print a summary."""
    configfile, modelfile = EXPERIMENTS[nickname]
    savedir = sim_dir(configfile, modelfile)
    print(f"=== {nickname} ===")
    print(f"  config     {configfile}")
    print(f"  checkpoint {modelfile}")
    print(f"  cached sim {savedir}")
    if not os.path.isdir(savedir):
        raise SystemExit(f"no cached simulations at {savedir}")

    windows = parse_windows(savedir)
    sim_lens = sorted({w['sim_len'] for w in windows})
    agent_counts = sorted({len(w['agents']) for w in windows})
    print(f"  {len(windows)} windows, {len({w['session'] for w in windows})} sessions, "
          f"sim_len {sim_lens}, agents/window {agent_counts}")

    ground_truth = load_ground_truth(configfile, modelfile)
    track = ground_truth['track']
    print(f"  track      {track.array.shape} {track.array.dtype}  "
          f"(n_agents, n_frames, 2, n_keypoints)")
    print(f"  contextl   {ground_truth['contextl']}")

    simulated = load_simulated(track, savedir, ground_truth['contextl'])
    sim_frame = simulated['sim_frame']
    n_simulated = int((sim_frame > 0).sum())
    print(f"  windows used {simulated['n_windows'] - simulated['n_skipped']}"
          f" / {simulated['n_windows']}  (skipped {simulated['n_skipped']}:"
          f" agent index >= {track.array.shape[0]})")
    print(f"  simulated agent-frames {n_simulated} "
          f"({100.0 * n_simulated / sim_frame.size:.2f}% of all agent-frames)")
    print(f"  distance-to-prompt range {sim_frame[sim_frame > 0].min():.0f}"
          f"..{sim_frame.max():.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest='command', required=True)

    describe = subparsers.add_parser(
        'describe', help="load one experiment and report track and window statistics")
    describe.add_argument('nickname', choices=sorted(EXPERIMENTS))

    score = subparsers.add_parser(
        'score', help="score one experiment's ground-truth and simulated tracks")
    score.add_argument('nickname', choices=sorted(EXPERIMENTS))
    score.add_argument('--max-windows', type=int, default=None,
                       help="only score this many windows, for timing runs")
    score.add_argument('--workers', type=int, default=1,
                       help="processes to score windows with; windows are independent")
    score.add_argument('--classifiers', default=DEFAULT_CLASSIFIER_SET,
                       type=classifier_set_name,
                       help="classifier set to score with: 'original', 'originalsplit' or "
                            "r<N>[nowingtip][split] (default r5nowingtip)")
    score.add_argument('--out-dir', default=RESULTS_PARENT_DIR,
                       help=f"output directory (default {RESULTS_PARENT_DIR})")

    validate = subparsers.add_parser(
        'validate',
        help="score the real track and compare against the dataset's own annotations")
    validate.add_argument('nickname', choices=sorted(EXPERIMENTS), default='ref',
                          nargs='?')
    validate.add_argument('--max-blocks', type=int, default=None,
                          help="only score this many labeled blocks per behavior")
    validate.add_argument('--classifiers', nargs='+', default=['original'],
                          type=classifier_set_name, metavar='SET',
                          help="classifier sets to compare, e.g. originalsplit r5split "
                               "r5nowingtipsplit")
    validate.add_argument('--splits', nargs='+', default=['test1', 'test2'],
                          help="MABe splits whose labeled frames are pooled (default "
                               "test1 test2, held out of the *_split sets only)")
    validate.add_argument('--workers', type=int, default=1,
                          help="processes to score blocks with; blocks are independent")
    validate.add_argument('--all-keypoints', action='store_true',
                          help="score with the real outer wing tips (21 keypoints) "
                               "rather than the 19 simulated tracks have")

    args = parser.parse_args()
    if args.command == 'describe':
        _describe(args.nickname)
    elif args.command == 'validate':
        _validate(args.nickname, args.max_blocks, args.classifiers, args.splits,
                  args.workers, all_keypoints=args.all_keypoints)
    else:
        _score(args.nickname, args.max_windows, args.out_dir, args.workers,
               args.classifiers)


if __name__ == '__main__':
    main()
