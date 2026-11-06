"""Metrics and figures, one function per task. Everything compares simulated frames to the same frames of the test set.

Each function writes its results to the run's results directory, and is repeated in full if it did not finish.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from apf.eval_pipeline.context import RunContext, job_items
from apf.eval_pipeline.params import atomic_path, write_json
from apf.eval_pipeline.probes import NO_LABEL, binary_metrics, probe_path
from apf.eval_pipeline.summary import summarize_run
from apf.ml import classify_real_fake


def save_pickle(ctx, name, data):
    path = os.path.join(ctx.paths.results_dir, name)
    tmp = atomic_path(path)
    with open(tmp, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp, path)


def save_figure(ctx, name):
    path = os.path.join(ctx.paths.results_dir, name)
    tmp = atomic_path(path)
    plt.savefig(tmp, dpi=150, bbox_inches='tight')
    os.replace(tmp, path)
    plt.close()


def run_probe_metrics(ctx: RunContext, shard: int):
    """Scores each probe on all labeled test frames."""
    import torch
    index = np.load(ctx.paths.test_index)
    labels = index['labels']
    logits = np.load(ctx.paths.test_logits)
    metrics = {}
    for action_id, action in enumerate(ctx.manifest['actions']):
        stats = torch.load(probe_path(ctx, action), weights_only=False)['stats']
        labeled = labels[action_id] != NO_LABEL
        action_labels, action_logits = labels[action_id, labeled], logits[labeled, action_id]
        metrics[action] = {
            'train': stats['train'],
            'test': binary_metrics(action_logits, action_labels),
            'n_test_pos': int((action_labels == 1).sum()),
            'n_test_neg': int((action_labels == 0).sum()),
            'train_PR_curve': stats['train_PR_curve'],
            'losses': stats['losses'],
        }
        has_both_classes = 0 < metrics[action]['n_test_pos'] and 0 < metrics[action]['n_test_neg']
        metrics[action]['test_PR_curve'] = (
            precision_recall_curve(action_labels, action_logits, drop_intermediate=True) if has_both_classes else None)

        plt.figure(figsize=(12, 5))
        plt.suptitle(action)
        for i, split in enumerate(['train', 'test']):
            plt.subplot(1, 2, i + 1)
            curve = metrics[action][f'{split}_PR_curve']
            if curve is not None:
                plt.plot(curve[1], curve[0], '.', markersize=2)
            score = metrics[action][split]
            plt.title(f"{split}: F1={score['f1']:.3f}, BA={score['balanced_accuracy']:.3f}")
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.axis([0, 1, 0, 1])
        save_figure(ctx, f"probe_{action}.png")
    save_pickle(ctx, 'probe_metrics.pkl', metrics)


def _action_classifications(ctx, sim_dataset, sim_data):
    """Loads probe outputs and adds the heuristic walk detector as the last action."""
    from apf.evaluation import detect_walk
    from flyllm.config import ARENA_RADIUS_PX, PXPERMM
    with open(ctx.paths.classifications, 'rb') as f:
        classifications = pickle.load(f)
    sources = {'test': (ctx.test_data['track'], ctx.test_dataset.sessions),
               'sim': (sim_data['track'], sim_dataset.sessions)}
    for name, (track, sessions) in sources.items():
        classific = classifications[name]
        walking = detect_walk(track, sessions, PXPERMM, ARENA_RADIUS_PX, kp=7, dur_thresh=20, log_vel_thresh=0)
        walking = walking[classific['flies'], classific['frames']]
        classific['logits'] = np.concatenate([classific['logits'], walking[:, None]], axis=1)
    return classifications, ctx.manifest['actions'] + ['walking']


def _n_bouts(predictions, flies, frames, action_id, min_dur):
    from apf.evaluation import get_bouts
    if predictions[:, action_id].sum() == 0:  # get_bouts requires at least one positive frame
        return 0
    return len(get_bouts(predictions, flies, frames, action_id, min_dur=min_dur)[0])


def run_action_stats(ctx: RunContext, shard: int):
    """Compares how often each action occurs in real and simulated data, and plots example locomotion."""
    from apf.evaluation import plot_locomotions
    params = ctx.params
    sim_dataset, sim_data = ctx.sim_dataset()
    sim_frame = sim_data['sim_frame']
    classifications, all_actions = _action_classifications(ctx, sim_dataset, sim_data)
    datanames = list(classifications)

    for max_dist in params.max_dists_to_prompt:
        bout_summary = np.zeros((len(datanames), len(all_actions)), int)
        frame_summary = np.zeros((len(datanames), len(all_actions)), int)
        for i, name in enumerate(datanames):
            classific = classifications[name]
            valid = sim_frame[classific['flies'], classific['frames']] <= max_dist
            predictions = classific['logits'][valid] > 0.5
            for action_id in range(len(all_actions)):
                bout_summary[i, action_id] = _n_bouts(predictions, classific['flies'][valid],
                                                      classific['frames'][valid], action_id, params.min_bout_dur)
                frame_summary[i, action_id] = predictions[:, action_id].sum()
        save_pickle(ctx, f"action_classification_results_max_dist_{max_dist}.pkl", {
            'frame_counts': frame_summary,
            'bout_counts': bout_summary,
            'datanames': datanames,
            'actions': all_actions,
        })

        plt.figure(figsize=(12, 3))
        plt.suptitle('Real vs. sim classification')
        for i, (title, summary) in enumerate([('frame count', frame_summary), ('bout count', bout_summary)]):
            plt.subplot(1, 2, i + 1)
            plt.plot(summary[0], '.', label='real', markersize=5)
            plt.plot(summary[1], '.', label='sim', markersize=3)
            for action_id in range(len(all_actions)):
                plt.plot([action_id, action_id], summary[:, action_id], '-k', linewidth=.1)
            plt.xticks(np.arange(len(all_actions)), all_actions, rotation=90)
            plt.title(title)
        save_figure(ctx, f"label_stats_max_dist_{max_dist}.png")

    tracks = {'test': ctx.test_data['track'], 'sim': sim_data['track']}
    for action in ['walking', 'perframe_wingext']:
        if action not in all_actions:
            continue
        for name in datanames:
            classific = classifications[name]
            predictions = classific['logits'] > 0.5
            # plot_locomotions selects bouts with its default minimum duration of 40 frames
            if _n_bouts(predictions, classific['flies'], classific['frames'], all_actions.index(action), 40) == 0:
                print(f"No {action} bouts in {name}, skipping locomotion figure")
                continue
            plot_locomotions(tracks[name], classifications, all_actions, sim_frame, action, name,
                             plot_with_features=action == 'walking')
            save_figure(ctx, f"locomotion_{action}_{name}.png")


def run_action_videos(ctx: RunContext, shard: int):
    from apf.evaluation import video_of_action_bouts
    params = ctx.params
    sim_dataset, sim_data = ctx.sim_dataset()
    sim_frame = sim_data['sim_frame']
    classifications, all_actions = _action_classifications(ctx, sim_dataset, sim_data)
    categories = ctx.manifest['categories']
    is_female = ctx.test_data['labels'][categories.index('female')].T

    video_actions = params.video_actions
    if video_actions is None:
        video_actions = [name for name in all_actions if name.startswith('perframe')] + ['walking']
    tracks = {'test': ctx.test_data['track'], 'sim': sim_data['track']}
    work = [(action, name) for action in video_actions if action in all_actions for name in tracks]
    for _, (action, name) in job_items(work, shard, params.n_video_jobs):
        classific = classifications[name]
        predictions = classific['logits'] > 0.5
        if _n_bouts(predictions, classific['flies'], classific['frames'], all_actions.index(action), 40) == 0:
            print(f"No {action} bouts in {name}, skipping video")
            continue
        output_path = os.path.join(ctx.paths.results_dir, f"action_bouts_{action}_{name}.avi")
        video_of_action_bouts(tracks[name], classifications, all_actions, sim_frame, action, name, output_path,
                              is_female, n_samples=params.video_samples, min_dist=1, max_dist=None, bg_img=None,
                              overwrite=False)


def _feature_names(ctx):
    from experiments.flyllm import BodyCentricKP
    from flyllm.config import keypointnames, posenames
    if isinstance(ctx.test_data['pose'].operations[0], BodyCentricKP):
        return list(posenames[:3]) + [f"{name}_{axis}" for name in keypointnames for axis in 'xy']
    return posenames


def run_distributions(ctx: RunContext, shard: int):
    from apf.evaluation import compute_feature_distributions, plot_distributions
    sim_dataset, sim_data = ctx.sim_dataset()
    valid_frames = ctx.valid_frames(sim_data['sim_frame'])
    feature_names = _feature_names(ctx)

    all_distributions = {}
    for data_name, use_log in [('pose', False), ('velocity', True)]:
        n_features = sim_data[data_name].array.shape[-1]
        feat_ids = [i for i in [0, 1, 2, 27, 28] + list(range(3, 27)) if i < n_features]
        all_distributions[data_name] = {}
        for frame_label, frames in valid_frames.items():
            distributions = compute_feature_distributions(
                ctx.test_data[data_name].array[frames], sim_data[data_name].array[frames],
                use_log=use_log, n_bins=1000, feat_ids=feat_ids)
            mean_kl = np.mean([value[-1] for value in distributions.values()])
            log_str = 'log ' if use_log else ''
            plot_distributions(distributions, posenames=feature_names, pdf=None, symlog=False,
                               title=f'{log_str}{data_name}, {frame_label} frames : mean KL = {mean_kl:.3f}')
            save_figure(ctx, f"distribution_{data_name}_{frame_label}.png")
            all_distributions[data_name][frame_label] = distributions
    save_pickle(ctx, 'all_distributions.pkl', all_distributions)


def run_discrimination(ctx: RunContext, shard: int):
    """Trains classifiers to tell real from simulated frames, with a real-vs-real control."""
    sim_dataset, sim_data = ctx.sim_dataset()
    valid_frames = ctx.valid_frames(sim_data['sim_frame'])
    layer_types = [[128], [512, 256, 128]]
    n_epochs = 3
    results = {}
    for input_type in ['pose', 'velocity']:
        if input_type not in ctx.test_dataset.inputs:
            print(f"Model has no {input_type} input, skipping")
            continue
        real = ctx.test_dataset.inputs[input_type].array
        sim = sim_dataset.inputs[input_type].array
        results[input_type] = np.zeros((len(layer_types), len(valid_frames), 2))
        for j, mlp_layers in enumerate(layer_types):
            for i, frames in enumerate(valid_frames.values()):
                results[input_type][j, i, 0] = classify_real_fake(
                    real[frames], sim[frames], n_epochs=n_epochs, mlp_layers=mlp_layers, device=ctx.device)[0]
                results[input_type][j, i, 1] = classify_real_fake(
                    real[frames], sim[frames], n_epochs=n_epochs * 2, control=True, mlp_layers=mlp_layers,
                    device=ctx.device)[0]
    save_pickle(ctx, 'real_sim_discrimination.pkl', {
        'discr_result': results,
        'first_dim': layer_types,
        'second_dim': list(valid_frames),
        'third_dim': ['accuracy', 'control accuracy'],
    })


def run_sim_videos(ctx: RunContext, shard: int):
    from apf.evaluation import get_multi_sessions, make_dataset_video
    sim_dataset, sim_data = ctx.sim_dataset()
    if len(get_multi_sessions(sim_dataset)) <= 80:  # make_dataset_video picks sessions starting from index 80
        print("Too few sessions for sample videos, skipping")
        return
    for name, track in [('simulation_samples', sim_data['track']), ('simulation_samples_real', ctx.test_data['track'])]:
        if not os.path.exists(os.path.join(ctx.paths.results_dir, f"{name}.mp4")):
            make_dataset_video(sim_dataset, track, ctx.paths.results_dir, name)


def run_report(ctx: RunContext, shard: int):
    """Writes what is available so far. The worker marks this done only if every other task finished."""
    row, per_action = summarize_run(ctx.paths.results_dir)
    write_json(os.path.join(ctx.paths.results_dir, 'summary.json'), {
        'configfile': ctx.configfile,
        'modelfile': ctx.modelfile,
        'metrics': row,
        'action_freq_deviation': {name: float(value) for name, value in per_action.items()},
    })
