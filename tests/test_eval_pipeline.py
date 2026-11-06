"""Tests of the evaluation pipeline that need no trained model, data or cluster.

Run from the repository root with
    python -m pytest tests/test_eval_pipeline.py
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from apf.eval_pipeline import lsf
from apf.eval_pipeline.context import job_items, unique_categories, use_test_split, window_rows
from apf.eval_pipeline.graph import Task, build_graph, is_done, mark_done, missing_dependencies, missing_shards
from apf.eval_pipeline.hidden import (FeatureStore, WindowBatches, iter_hidden, unit_path, unit_ranges,
                                      write_feature_unit)
from apf.eval_pipeline.params import EvalParams, Resources, RunPaths
from apf.eval_pipeline.probes import binary_metrics, compact_labels, select_actions, train_probe
from apf.eval_pipeline.simulate import enumerate_chunks, enumerate_sessions
from apf.models import TransformerModel


@pytest.fixture
def paths(tmp_path):
    paths = RunPaths(str(tmp_path / 'run'), str(tmp_path / 'cache'))
    paths.make_dirs()
    return paths


def make_dataset(sessions, n_agents=3, n_frames=40, d_input=4, contextl=8):
    """A stand-in for Dataset where input values encode (agent, frame), split over two input arrays."""
    velocity = np.zeros((n_agents, n_frames, d_input - 1), np.float32)
    pose = np.zeros((n_agents, n_frames, 1), np.float32)
    for agent in range(n_agents):
        velocity[agent] = (agent * 1000 + np.arange(n_frames))[:, None]
        pose[agent] = -(agent * 1000 + np.arange(n_frames))[:, None]
    return SimpleNamespace(
        inputs={'velocity': SimpleNamespace(array=velocity), 'pose': SimpleNamespace(array=pose)},
        context_length=contextl,
        d_input=d_input,
        sessions=[SimpleNamespace(agent_id=a, start_frame=s, duration=d) for a, s, d in sessions],
    )


# --- Parameters and paths ---

def test_overrides_parse_values_and_reject_unknown_keys():
    params = EvalParams().with_overrides(['max_chunks=2', 'categories=["a"]', 'feature_dtype=float32'])
    assert params.max_chunks == 2 and params.categories == ['a'] and params.feature_dtype == 'float32'
    with pytest.raises(ValueError):
        EvalParams().with_overrides(['not_a_param=1'])


def test_params_survive_saving():
    params = EvalParams().with_overrides(['max_actions=3'])
    assert EvalParams.from_dict(params.to_dict()).to_dict() == params.to_dict()


def test_use_test_split_requires_test_set():
    config = {'invalfilestr': 'testtrain_v3.npz', 'invalfile': '/data/testtrain_v3.npz'}
    assert use_test_split(config)['invalfile'] == '/data/test2_v3.npz'
    with pytest.raises(ValueError):
        use_test_split({'invalfilestr': 'other.npz', 'invalfile': '/data/other.npz'})


def test_unique_categories_handles_repeats():
    assert unique_categories(['control', 'a', 'control', 'control']) == ['control', 'a', 'control_', 'control__']


# --- Task graph ---

def test_graph_is_ordered_with_dependencies_first():
    seen = set()
    for task in build_graph(EvalParams()):
        assert set(task.deps) <= seen, task.name
        seen.add(task.name)


def test_completion_is_tracked_per_shard(paths):
    task = Task('simulate', 3, True)
    assert missing_shards(paths, task) == [1, 2, 3]
    mark_done(paths, 'simulate', 2)
    assert is_done(paths, 'simulate', 2)
    assert missing_shards(paths, task) == [1, 3]


def test_missing_dependencies(paths):
    graph = [Task('a', 2, False), Task('b', 1, False, ('a',))]
    assert missing_dependencies(paths, graph[1], graph) == ['a']
    mark_done(paths, 'a', 1)
    assert missing_dependencies(paths, graph[1], graph) == ['a']
    mark_done(paths, 'a', 2)
    assert missing_dependencies(paths, graph[1], graph) == []


def test_job_items_partition_everything_once():
    items = list(range(10))
    assigned = [i for shard in range(1, 4) for i, _ in job_items(items, shard, 3)]
    assert sorted(assigned) == items


# --- LSF ---

def test_array_spec():
    assert lsf.array_spec([1, 2, 3, 7]) == '[1-3,7]'
    assert lsf.array_spec([4]) == '[4]'
    assert lsf.array_spec([1, 3, 4, 6]) == '[1,3-4,6]'


def test_parse_job_elements():
    output = "11 RUN ev_ab_simulate[3]\n11 PEND ev_ab_simulate[4]\n12 DONE ev_ab_prepare\n13 RUN something_else\n"
    assert lsf.parse_job_elements(output, 'ev_ab') == {
        'simulate': {3: ('RUN', '11[3]'), 4: ('PEND', '11[4]')},
        'prepare': {1: ('DONE', '12')},
    }


GRAPH = [Task('a', 3, True), Task('b', 1, False, ('a',)), Task('c', 1, False, ('b',))]


def test_first_submission_includes_everything(paths):
    plan = lsf.plan_submission(paths, GRAPH, {})
    assert [(t.name, shards, deps) for t, shards, deps in plan] == [
        ('a', [1, 2, 3], []), ('b', [1], ['a']), ('c', [1], ['b'])]


def test_resume_only_submits_what_is_missing(paths):
    mark_done(paths, 'a', 1)
    plan = lsf.plan_submission(paths, GRAPH, {})
    assert plan[0][1] == [2, 3]
    assert plan[1][2] == ['a']


def test_resume_leaves_running_elements_alone(paths):
    mark_done(paths, 'a', 1)
    plan = lsf.plan_submission(paths, GRAPH, {'a': {2: ('RUN', '9[2]')}})
    assert plan[0][1] == [3]
    assert plan[1][2] == ['a']


def test_wait_on_running_task_even_if_nothing_new_is_submitted_for_it(paths):
    for shard in [1, 2]:
        mark_done(paths, 'a', shard)
    plan = lsf.plan_submission(paths, GRAPH, {'a': {3: ('RUN', '9[3]')}})
    assert [t.name for t, _, _ in plan] == ['b', 'c']
    assert plan[0][2] == ['a']


def test_no_dependency_on_finished_tasks(paths):
    for shard in [1, 2, 3]:
        mark_done(paths, 'a', shard)
    plan = lsf.plan_submission(paths, GRAPH, {})
    assert plan[0][0].name == 'b' and plan[0][2] == []


def test_nothing_to_submit_when_done(paths):
    for task in GRAPH:
        for shard in range(1, task.n_jobs + 1):
            mark_done(paths, task.name, shard)
    assert lsf.plan_submission(paths, GRAPH, {}) == []


def test_finished_or_failed_jobs_do_not_block_resubmission(paths):
    plan = lsf.plan_submission(paths, GRAPH, {'a': {1: ('EXIT', '9[1]'), 2: ('DONE', '9[2]')}})
    assert plan[0][1] == [1, 2, 3]


def test_bsub_command(paths):
    cmd = lsf.bsub_command(paths, GRAPH[0], [1, 3], Resources(queue='gpu_x', gpu_slots=7), ['prior'])
    prefix = paths.job_prefix()
    assert f"{prefix}_a[1,3]" in cmd
    assert "-q gpu_x" in cmd and "-n 7" in cmd and "-gpu num=1" in cmd
    assert f"ended({prefix}_prior)" in cmd
    assert "$LSB_JOBINDEX" in cmd
    assert "a.%I.log" in cmd

    cmd = lsf.bsub_command(paths, GRAPH[1], [1], Resources(cpu_queue='cpus', overrides={'b': {'slots': 2}}), [])
    assert "-q cpus" in cmd and "-n 2" in cmd and "-gpu" not in cmd and "-w" not in cmd.split()


def test_submit_resumes_through_fake_scheduler(paths, monkeypatch):
    commands = []
    bjobs_output = {'text': ''}

    def fake_run_lsf(cmd):
        commands.append(cmd)
        return bjobs_output['text'] if cmd.startswith('bjobs') else 'Job <1> is submitted'

    monkeypatch.setattr(lsf, 'run_lsf', fake_run_lsf)
    prefix = paths.job_prefix()

    assert [name for name, _ in lsf.submit(paths, GRAPH, Resources())] == ['a', 'b', 'c']

    # Second attempt: a[1] finished, a[2] is running, a[3] is pending along with the downstream tasks
    commands.clear()
    mark_done(paths, 'a', 1)
    bjobs_output['text'] = (f"5 RUN {prefix}_a[2]\n5 PEND {prefix}_a[3]\n6 PEND {prefix}_b\n7 PEND {prefix}_c\n")
    submitted = lsf.submit(paths, GRAPH, Resources())
    assert submitted == [('a', [3]), ('b', [1]), ('c', [1])]
    assert sorted(c for c in commands if c.startswith('bkill')) == ["bkill '5[3]'", "bkill '6'", "bkill '7'"]
    assert not any(c.startswith('bkill') and '5[2]' in c for c in commands)

    # Dry runs change nothing
    commands.clear()
    lsf.submit(paths, GRAPH, Resources(), dry_run=True)
    assert all(c.startswith('bjobs') for c in commands)


def test_status_shows_progress_and_failed_logs(paths):
    mark_done(paths, 'a', 1)
    with open(os.path.join(paths.log_dir, 'a.2.log'), 'w') as f:
        f.write('Traceback: something broke\n')
    lines = lsf.status_lines(paths, GRAPH, {'a': {3: ('RUN', '9[3]')}})
    text = '\n'.join(lines)
    assert 'done   1/3' in text and 'running   1' in text
    assert 'something broke' in text


# --- Choosing frames and simulation chunks ---

def test_window_rows_only_has_full_windows_and_is_sorted():
    dataset = make_dataset([(1, 20, 10), (0, 0, 10), (0, 20, 9)])
    agents, frames = window_rows(dataset, 8)
    assert list(zip(agents, frames)) == [(0, 7), (0, 8), (0, 9), (0, 27), (0, 28), (1, 27), (1, 28), (1, 29)]


def test_session_zero_is_simulated():
    # Regression test: session index 0 used to be treated as "no session", dropping its agent
    dataset = make_dataset([(0, 0, 40), (1, 0, 40)], n_agents=2)
    sessions = enumerate_sessions(dataset, sim_len=4, first_sim_frame=8)
    assert len(sessions) == 1
    assert list(sessions[0][2]) == [0, 1]


def test_sessions_split_when_agents_change():
    dataset = make_dataset([(0, 0, 40), (1, 20, 20)], n_agents=2)
    sessions = enumerate_sessions(dataset, sim_len=4, first_sim_frame=8)
    assert [(s, d, list(a)) for s, d, a in sessions] == [(0, 20, [0]), (20, 20, [0, 1])]


def test_chunks_cover_session_with_stride():
    dataset = make_dataset([(0, 0, 40)], n_agents=1)
    chunks = enumerate_chunks(dataset, sim_len=4, first_sim_frame=8)
    assert [c['start'] for c in chunks] == list(range(8, 36, 4))
    assert chunks[0]['agents'] == [0]
    assert chunks[0]['file'] == 'session_0_startsimframe_8_agentid_0_simlen_4.npy'
    assert len(enumerate_chunks(dataset, sim_len=4, first_sim_frame=8, max_chunks=2)) == 2


def test_select_and_compact_labels():
    labels = np.full((3, 10, 1), np.nan)
    labels[0, :5, 0] = 1
    labels[0, 5:, 0] = 0
    labels[1, 0, 0] = 1  # Too few
    labels[2] = 1  # All positive
    agents, frames = np.zeros(10, int), np.arange(10)
    params = EvalParams(min_labeled=2)
    actions, counts = select_actions(labels, ['x', 'y', 'z'], agents, frames, params)
    assert actions == ['x'] and counts['x'] == {'n_pos': 5, 'n_neg': 5}
    assert compact_labels(labels, ['x', 'y', 'z'], ['y'], agents, frames)[0, :2].tolist() == [1, -1]


# --- Hidden states ---

def test_windows_match_get_chunk_semantics():
    dataset = make_dataset([(0, 0, 40), (1, 0, 40)], n_agents=2)
    agents = np.array([0, 0, 1])
    frames = np.array([7, 12, 30])
    batch = WindowBatches(dataset, agents, frames, batch_size=3)[0]
    assert batch.shape == (3, 8, 4)
    for i, (agent, frame) in enumerate(zip(agents, frames)):
        expected = agent * 1000 + np.arange(frame - 7, frame + 1)
        assert np.array_equal(batch[i, :, 0], expected)  # Window ends at the frame
        assert np.array_equal(batch[i, :, 3], -expected)  # Second input array is concatenated last


def make_model(d_input=4, d_model=16, n_layers=4):
    torch.manual_seed(0)
    model = TransformerModel(d_input, 3, d_model=d_model, nhead=2, d_hid=32, nlayers=n_layers, dropout=0.0)
    return model.eval()


@pytest.mark.parametrize('causal', [True, False])
def test_hidden_pass_matches_full_forward(causal):
    """The fast pass has to give what running the whole model and keeping the last position gives."""
    dataset = make_dataset([(0, 0, 40), (1, 0, 40), (2, 0, 40)])
    agents, frames = window_rows(dataset, 8)
    model = make_model()
    fast = np.zeros((len(frames), 16))
    for row, hidden in iter_hidden(model, dataset, agents, frames, 1, batch_size=7, causal=causal):
        fast[row:row + len(hidden)] = hidden

    mask = torch.nn.Transformer.generate_square_subsequent_mask(8) if causal else None
    full = np.zeros_like(fast)
    windows = WindowBatches(dataset, agents, frames, 7)
    with torch.no_grad():
        for i in range(len(windows)):
            out = model.forward(torch.from_numpy(windows[i]), mask=mask, is_causal=causal, return_hidden=True)
            full[i * 7:i * 7 + len(out['hidden'][1])] = out['hidden'][1][:, -1]
    assert np.allclose(fast, full, atol=1e-5)


def test_causal_mask_changes_hidden_states():
    """Guards the reason the mask is passed at all: without it the states differ from training and simulation."""
    dataset = make_dataset([(0, 0, 40)], n_agents=1)
    agents, frames = window_rows(dataset, 8)
    model = make_model()
    causal = next(iter_hidden(model, dataset, agents, frames, 1, batch_size=8, causal=True))[1]
    open_ = next(iter_hidden(model, dataset, agents, frames, 1, batch_size=8, causal=False))[1]
    assert not np.allclose(causal, open_)


def test_forward_defaults_are_unchanged():
    model = make_model()
    src = torch.rand(2, 8, 4)
    with torch.no_grad():
        out = model.forward(src, return_hidden=True)
        selected = model.forward(src, return_hidden=True, hidden_layers=[2], hidden_last_only=True)
    assert len(out['hidden']) == 4 and out['hidden'][0].shape == (2, 8, 16)
    assert 'continuous' in out
    assert np.allclose(selected['hidden'][0], out['hidden'][2][:, -1])
    assert np.allclose(selected['continuous'].numpy(), out['continuous'].numpy())


def test_feature_store_round_trip(tmp_path):
    dataset = make_dataset([(0, 0, 40), (1, 0, 40)], n_agents=2)
    agents, frames = window_rows(dataset, 8)
    model = make_model()
    units = unit_ranges(len(frames), 10)
    assert units[-1][1] == len(frames)
    for unit_id, (start, end) in enumerate(units):
        write_feature_unit(unit_path(tmp_path, unit_id), model, dataset, agents[start:end], frames[start:end], 1,
                           'float32', batch_size=4)
    assert not [f for f in os.listdir(tmp_path) if f.startswith('.tmp')]

    store = FeatureStore(str(tmp_path), len(frames), 10)
    expected = np.concatenate([h for _, h in iter_hidden(model, dataset, agents, frames, 1, batch_size=4)])
    rows = np.array([len(frames) - 1, 0, 15, 3, 25])
    assert np.allclose(store.read(rows), expected[rows], atol=1e-5)  # Order of the request is kept


# --- Probes ---

def _separable_store(tmp_path, n=4000, d=8):
    rng = np.random.default_rng(0)
    labels = (rng.random(n) < 0.2).astype(np.int8)  # Imbalanced
    features = rng.normal(size=(n, d)).astype(np.float32)
    features[:, 0] += 4 * (labels * 2 - 1)
    for unit_id, (start, end) in enumerate(unit_ranges(n, 1000)):
        np.save(unit_path(tmp_path, unit_id), features[start:end])
    return FeatureStore(str(tmp_path), n, 1000), labels


def test_probe_learns_and_uses_only_labeled_rows(tmp_path):
    store, labels = _separable_store(tmp_path)
    labels = labels.copy()
    labels[::10] = -1
    _, stats = train_probe(store, labels, torch.device('cpu'), lr=0.01, min_epochs=2, max_epochs=8,
                           block_rows=256, num_workers=0, seed=0)
    assert stats['train']['balanced_accuracy'] > 0.95
    assert stats['n_pos'] + stats['n_neg'] == (labels != -1).sum()
    assert stats['train']['tp'] + stats['train']['fp'] + stats['train']['tn'] + stats['train']['fn'] == \
        (labels != -1).sum()


def test_probe_with_loader_workers(tmp_path):
    store, labels = _separable_store(tmp_path)
    _, stats = train_probe(store, labels, torch.device('cpu'), lr=0.01, min_epochs=2, max_epochs=4,
                           block_rows=256, num_workers=2, seed=0)
    assert stats['train']['balanced_accuracy'] > 0.9


def test_binary_metrics():
    metrics = binary_metrics(np.array([0.9, 0.8, 0.1, 0.7]), np.array([1, 1, 0, 0]))
    assert metrics['tp'] == 2 and metrics['fp'] == 1 and metrics['tn'] == 1 and metrics['fn'] == 0
    assert metrics['balanced_accuracy'] == pytest.approx(0.75)
    assert metrics['f1'] == pytest.approx(0.8)
    # No positives predicted or present must not crash
    assert np.isnan(binary_metrics(np.array([0.1]), np.array([0]))['f1'])
