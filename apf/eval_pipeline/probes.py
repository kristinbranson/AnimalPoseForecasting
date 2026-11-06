"""Per-category probes: choosing what to train on, training on stored features, applying them."""
import os
import pickle
import shutil

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm.auto import tqdm

from apf.eval_pipeline.context import RunContext, job_items, unique_categories, window_rows
from apf.eval_pipeline.hidden import (FeatureStore, iter_hidden, unit_path, unit_ranges, write_feature_unit)
from apf.eval_pipeline.params import atomic_path, write_json
from apf.eval_pipeline.simulate import enumerate_chunks
from apf.ml import MLP

NO_LABEL = -1


def select_actions(labels, categories, agents, frames, params):
    """Picks categories with enough labeled train frames. Returns the names and per-category counts."""
    actions = []
    counts = {}
    for i, name in enumerate(categories):
        if name == 'BadTracking' or (params.categories is not None and name not in params.categories):
            continue
        values = labels[i, frames, agents]
        n_pos = int((values == 1).sum())
        n_neg = int((values == 0).sum())
        if params.min_labeled < n_pos and params.min_labeled < n_neg:
            actions.append(name)
            counts[name] = {'n_pos': n_pos, 'n_neg': n_neg}
    return (actions if params.max_actions is None else actions[:params.max_actions]), counts


def compact_labels(labels, categories, actions, agents, frames):
    """Returns int8 labels of shape (n_actions, n_rows), with NO_LABEL where there is no label."""
    out = np.full((len(actions), len(frames)), NO_LABEL, np.int8)
    for i, action in enumerate(actions):
        values = labels[categories.index(action), frames, agents]
        out[i, values == 1] = 1
        out[i, values == 0] = 0
    return out


def _subsample(rng, rows, n_max, keep=None):
    selected = rows if n_max is None or rows.sum() <= n_max else None
    if selected is None:
        selected = np.zeros_like(rows)
        selected[rng.choice(np.where(rows)[0], n_max, replace=False)] = True
    return selected if keep is None else selected | keep


def run_prepare(ctx: RunContext, shard: int):
    """Decides what the rest of the run works on and saves it, so that array jobs only need these small files."""
    params = ctx.params
    paths = ctx.paths
    train_data = ctx.train_data
    test_data = ctx.test_data
    contextl = ctx.train_dataset.context_length
    assert list(train_data['categories']) == list(test_data['categories'])
    categories = unique_categories(train_data['categories'])
    rng = np.random.default_rng(0)

    chunks = enumerate_chunks(ctx.test_dataset, params.sim_len, params.first_sim_frame, params.stride,
                              params.max_chunks)

    # Train: every frame with a valid label in a category that is being trained
    agents, frames = window_rows(ctx.train_dataset, contextl)
    valid = train_data['isdata'][frames, agents]
    agents, frames = agents[valid], frames[valid]
    actions, counts = select_actions(train_data['labels'], categories, agents, frames, params)
    if len(actions) == 0:
        raise ValueError(f"No category has more than min_labeled={params.min_labeled} positive and negative frames")
    train_labels = compact_labels(train_data['labels'], categories, actions, agents, frames)
    labeled = (train_labels != NO_LABEL).any(axis=0)
    labeled = _subsample(rng, labeled, params.max_frames_per_split)
    np.savez(_npz_tmp(paths.train_index), agents=agents[labeled], frames=frames[labeled],
             labels=train_labels[:, labeled])
    os.replace(_npz_tmp(paths.train_index), paths.train_index)
    n_train_rows = int(labeled.sum())

    # Test: every frame with a full window. These are only kept as probe outputs.
    agents, frames = window_rows(ctx.test_dataset, contextl)
    valid = test_data['isdata'][frames, agents]
    agents, frames = agents[valid], frames[valid]
    keep = None
    if params.max_frames_per_split is not None:
        # Limited runs still need the real frames that correspond to the simulated ones
        in_sim = np.zeros(test_data['isdata'].shape[::-1], bool)
        for chunk in chunks:
            in_sim[np.array(chunk['agents'])[:, None], chunk['start']:chunk['start'] + params.sim_len] = True
        keep = in_sim[agents, frames]
    selected = _subsample(rng, np.ones(len(frames), bool), params.max_frames_per_split, keep)
    agents, frames = agents[selected], frames[selected]
    test_labels = compact_labels(test_data['labels'], categories, actions, agents, frames)
    np.savez(_npz_tmp(paths.test_index), agents=agents, frames=frames, labels=test_labels)
    os.replace(_npz_tmp(paths.test_index), paths.test_index)

    write_json(paths.manifest, {
        'categories': categories,
        'actions': actions,
        'train_counts': counts,
        'chunks': chunks,
        'n_train_rows': n_train_rows,
        'n_test_rows': len(frames),
    })


def _npz_tmp(path):
    return path + '.tmp.npz'


def run_train_features(ctx: RunContext, shard: int):
    params = ctx.params
    index = np.load(ctx.paths.train_index)
    agents, frames = index['agents'], index['frames']
    units = unit_ranges(len(frames), params.frames_per_unit)
    model = None
    for unit_id, (start, end) in job_items(units, shard, params.n_feature_jobs):
        path = unit_path(ctx.paths.feature_dir, unit_id)
        if os.path.exists(path):
            continue
        model = model or ctx.load_model(ctx.train_dataset)
        write_feature_unit(path, model, ctx.train_dataset, agents[start:end], frames[start:end],
                           params.probe_layer, params.feature_dtype, batch_size=params.hidden_batch_size,
                           causal=params.causal_hidden, num_workers=params.hidden_num_workers,
                           mixed_precision=params.hidden_mixed_precision)


class BalancedBlocks(torch.utils.data.Dataset):
    """Blocks of positive and negative features. Reading a block is what the loader workers parallelize."""

    def __init__(self, store, pos_rows, neg_rows, half_block):
        self.store = store
        self.pos_rows = pos_rows
        self.neg_rows = neg_rows
        self.half_block = half_block

    def __len__(self):
        return int(np.ceil(len(self.pos_rows) / self.half_block))

    def __getitem__(self, i):
        start, end = i * self.half_block, (i + 1) * self.half_block
        return self.store.read(self.pos_rows[start:end]), self.store.read(self.neg_rows[start:end])


def binary_metrics(probabilities, labels):
    """Same definitions as apf.ml.apply_binary_classifier."""
    predictions = probabilities > 0.5
    tp = int(np.logical_and(predictions, labels == 1).sum())
    fp = int(np.logical_and(predictions, labels == 0).sum())
    tn = int(np.logical_and(~predictions, labels == 0).sum())
    fn = int(np.logical_and(~predictions, labels == 1).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
    return {'balanced_accuracy': (recall + specificity) / 2, 'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def train_probe(store, labels, device, *, mlp_layers=(), lr=0.0001, min_epochs=5, max_epochs=100,
                block_rows=65536, num_workers=6, max_eval_rows=1000000, seed=None):
    """Trains a classifier on class-balanced batches.

    Batch sizes and the number of epochs follow apf.ml.train_binary_classifier. Each epoch the larger class is drawn
    anew from all of its frames. Performance is only measured after training, on the training set.
    """
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n_min = min(len(pos), len(neg))
    n_frames = n_min * 2

    batch_size = 32
    n_epochs = max_epochs
    if n_frames < 5000:
        batch_size = 8
    if n_frames > 100000:
        batch_size = 128
    half_batch_size = batch_size // 2
    n_batches = n_min // half_batch_size
    if n_frames > 100000:
        n_epochs = int(np.clip(25000 // n_batches, min_epochs, max_epochs))
    half_block = max(block_rows // 2 // half_batch_size, 1) * half_batch_size
    print(f"batch_size={batch_size}, n_epochs={n_epochs}, n_pos={len(pos)}, n_neg={len(neg)}", flush=True)

    classifier = MLP(store.d_feature, mlp_layers).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    losses = np.zeros(n_epochs)
    for epoch in tqdm(range(n_epochs), mininterval=30):
        classifier.train()
        epoch_loss = 0
        dataset = BalancedBlocks(store, rng.permutation(pos)[:n_min], rng.permutation(neg)[:n_min], half_block)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=num_workers)
        for block_pos, block_neg in loader:
            block_pos = block_pos.to(device)[torch.randperm(len(block_pos))]
            block_neg = block_neg.to(device)[torch.randperm(len(block_neg))]
            for i in range(0, len(block_pos), half_batch_size):
                batch_x = torch.cat([block_pos[i:i + half_batch_size], block_neg[i:i + half_batch_size]])
                batch_y = torch.zeros((len(batch_x), 1), device=device)
                batch_y[:len(block_pos[i:i + half_batch_size])] = 1
                optimizer.zero_grad()
                loss = criterion(classifier(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        losses[epoch] = epoch_loss / max(n_batches, 1)

    classifier.eval()
    eval_rows = np.sort(np.concatenate([pos, neg]))
    if len(eval_rows) > max_eval_rows:
        eval_rows = np.sort(rng.choice(eval_rows, max_eval_rows, replace=False))
    probabilities = np.zeros(len(eval_rows))
    with torch.no_grad():
        for i in range(0, len(eval_rows), block_rows):
            batch_x = torch.from_numpy(store.read(eval_rows[i:i + block_rows])).to(device)
            probabilities[i:i + block_rows] = classifier(batch_x).cpu().numpy()[:, 0]
    stats = {
        'losses': losses,
        'n_pos': len(pos),
        'n_neg': len(neg),
        'train': binary_metrics(probabilities, labels[eval_rows]),
        'train_PR_curve': precision_recall_curve(labels[eval_rows], probabilities, drop_intermediate=True),
    }
    return classifier, stats


def probe_path(ctx: RunContext, action: str):
    return os.path.join(ctx.paths.probe_dir, f"{action}.pth")


def run_probes(ctx: RunContext, shard: int):
    params = ctx.params
    manifest = ctx.manifest
    index = np.load(ctx.paths.train_index)
    labels = index['labels']
    store = FeatureStore(ctx.paths.feature_dir, manifest['n_train_rows'], params.frames_per_unit)
    device = ctx.device
    for action_id, action in job_items(manifest['actions'], shard, params.n_probe_jobs):
        path = probe_path(ctx, action)
        if os.path.exists(path):
            continue
        print(f"Training probe for {action}", flush=True)
        classifier, stats = train_probe(
            store, labels[action_id], device, mlp_layers=params.probe_mlp_layers, lr=params.probe_lr,
            min_epochs=params.probe_min_epochs, max_epochs=params.probe_max_epochs,
            block_rows=params.probe_block_rows, num_workers=params.probe_num_workers)
        tmp = atomic_path(path)
        torch.save({'model_state_dict': classifier.state_dict(), 'stats': stats}, tmp)
        os.replace(tmp, path)


def load_probes(ctx: RunContext, device, d_model):
    probes = []
    for action in ctx.manifest['actions']:
        classifier = MLP(d_model, ctx.params.probe_mlp_layers)
        classifier.load_state_dict(torch.load(probe_path(ctx, action), weights_only=False)['model_state_dict'])
        probes.append(classifier.to(device).eval())
    return probes


def apply_probes(probes, hidden, device):
    with torch.no_grad():
        hidden = torch.from_numpy(hidden).to(device)
        return torch.cat([probe(hidden) for probe in probes], dim=-1).cpu().numpy().astype(np.float32)


def run_logits(ctx: RunContext, shard: int):
    """Applies the probes to test frames (hidden states computed here) and simulated frames (hidden states on disk)."""
    params = ctx.params
    paths = ctx.paths
    manifest = ctx.manifest
    index = np.load(paths.test_index)
    agents, frames = index['agents'], index['frames']
    units = [('test', unit_id, unit) for unit_id, unit in
             enumerate(unit_ranges(len(frames), params.frames_per_unit))]
    work = units + [('sim', chunk_id, chunk) for chunk_id, chunk in enumerate(manifest['chunks'])]

    probes = None
    model = None
    for _, (kind, item_id, item) in job_items(work, shard, params.n_logit_jobs):
        if kind == 'test':
            path = unit_path(paths.test_logit_dir, item_id)
            if os.path.exists(path):
                continue
            model = model or ctx.load_model(ctx.test_dataset)
            device = ctx.device
            probes = probes or load_probes(ctx, device, model.d_model)
            start, end = item
            logits = np.zeros((end - start, len(probes)), np.float32)
            for row, hidden in iter_hidden(model, ctx.test_dataset, agents[start:end], frames[start:end],
                                           params.probe_layer, batch_size=params.hidden_batch_size,
                                           causal=params.causal_hidden, num_workers=params.hidden_num_workers,
                                           mixed_precision=params.hidden_mixed_precision):
                logits[row:row + len(hidden)] = apply_probes(probes, hidden, device)
        else:
            path = os.path.join(paths.sim_logit_dir, item['file'])
            if os.path.exists(path):
                continue
            device = ctx.device
            hidden = np.load(os.path.join(paths.sim_hidden_dir, item['file'])).astype(np.float32)
            probes = probes or load_probes(ctx, device, hidden.shape[-1])
            shape = hidden.shape
            logits = apply_probes(probes, hidden.reshape(-1, shape[-1]), device)
            logits = logits.reshape(shape[0], shape[1], -1)
        tmp = atomic_path(path)
        np.save(tmp, logits)
        os.replace(tmp, path)


def run_merge_logits(ctx: RunContext, shard: int):
    """Combines probe outputs into test_logits.npy and classifications.pkl, then removes simulated hidden states."""
    params = ctx.params
    paths = ctx.paths
    manifest = ctx.manifest
    index = np.load(paths.test_index)
    agents, frames = index['agents'], index['frames']
    n_actions = len(manifest['actions'])
    sim_frame = np.load(paths.sim_frame)

    test_logits = np.zeros((len(frames), n_actions), np.float32)
    for unit_id, (start, end) in enumerate(unit_ranges(len(frames), params.frames_per_unit)):
        test_logits[start:end] = np.load(unit_path(paths.test_logit_dir, unit_id))
    tmp = atomic_path(paths.test_logits)
    np.save(tmp, test_logits)
    os.replace(tmp, paths.test_logits)

    # Row of each (agent, frame) in the sim frame arrays
    sim_flies, sim_frames = np.where(sim_frame > 0)
    row_of = np.full(sim_frame.shape, -1, np.int64)
    row_of[sim_flies, sim_frames] = np.arange(len(sim_frames))

    sim_logits = np.full((len(sim_frames), n_actions), np.nan, np.float32)
    for chunk in manifest['chunks']:
        chunk_logits = np.load(os.path.join(paths.sim_logit_dir, chunk['file']))
        for agent_i, agent in enumerate(chunk['agents']):
            rows = row_of[agent, chunk['start']:chunk['start'] + params.sim_len]
            sim_logits[rows[rows >= 0]] = chunk_logits[agent_i][rows >= 0]

    # The same frames from the real data, so that real and simulated are compared over the same frames
    real_row_of = np.full(sim_frame.shape, -1, np.int64)
    real_row_of[agents, frames] = np.arange(len(frames))
    rows = real_row_of[sim_flies, sim_frames]
    assert np.all(rows >= 0), "Simulated frames are missing from the real test rows"

    classifications = {
        'test': {'logits': test_logits[rows], 'frames': sim_frames, 'flies': sim_flies},
        'sim': {'logits': sim_logits, 'frames': sim_frames, 'flies': sim_flies},
    }
    tmp = atomic_path(paths.classifications)
    with open(tmp, 'wb') as f:
        pickle.dump(classifications, f)
    os.replace(tmp, paths.classifications)

    if not ctx.keep_features:
        shutil.rmtree(paths.sim_hidden_dir, ignore_errors=True)
