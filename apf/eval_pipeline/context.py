"""Shared state of a task: run settings, config, datasets and model."""
import os
from functools import cached_property

import numpy as np
import torch

from apf.eval_pipeline.params import EvalParams, RunPaths, read_json


def use_test_split(config):
    """Points the validation entries of the config at the test set. All evaluation goes through this."""
    for key in ['invalfilestr', 'invalfile']:
        config[key] = config[key].replace('testtrain', 'test2')
    if 'test2' not in config['invalfile']:
        raise ValueError(f"Could not find the test set from invalfile={config['invalfile']!r}")
    return config


def unique_categories(categories):
    """Makes category names unique, some appear twice (e.g. 'control'), by appending underscores."""
    unique = []
    for name in categories:
        name = str(name)
        while name in unique:
            name += '_'
        unique.append(name)
    return unique


def window_rows(dataset, contextl):
    """Returns (agents, frames), sorted by agent then frame, of all frames that end a full context window."""
    agents = []
    frames = []
    for session in dataset.sessions:
        session_frames = np.arange(session.start_frame + contextl - 1, session.start_frame + session.duration)
        agents.append(np.full(len(session_frames), session.agent_id))
        frames.append(session_frames)
    agents = np.concatenate(agents)
    frames = np.concatenate(frames)
    order = np.lexsort((frames, agents))
    return agents[order], frames[order]


def job_items(items, shard, n_jobs):
    """Deterministically assigns items to an array element (1-indexed). Yields (index, item)."""
    for i, item in enumerate(items):
        if i % n_jobs == shard - 1:
            yield i, item


class RunContext:
    def __init__(self, run_dir):
        run = read_json(os.path.join(run_dir, 'run.json'))
        self.configfile = run['configfile']
        self.modelfile = run['modelfile']
        self.params = EvalParams.from_dict(run['params'])
        self.keep_features = run.get('keep_features', False)
        self.paths = RunPaths(run_dir, run['cache_dir'])

    @cached_property
    def config(self):
        # Imported here so that the scheduler side of the pipeline does not need the full stack
        from apf.io import load_config_from_model_file
        from flyllm.config import read_config
        config = read_config(self.configfile)
        load_config_from_model_file(loadmodelfile=self.modelfile, config=config, weights_only=False)
        return use_test_split(config)

    @cached_property
    def device(self):
        return torch.device(self.config['device'])

    def _load_split(self, train):
        """Loads one split. The test set gets its normalization from the model file, so it doesn't need train."""
        from flyllm.prepare import init_datasets
        return init_datasets(self.config, needtraindata=train, needvaldata=not train, res={'config': self.config},
                             debug_uselessdata=False)

    @cached_property
    def _train(self):
        return self._load_split(train=True)

    @cached_property
    def _test(self):
        if 'dataset_params' not in self.config:
            raise ValueError(f"{self.modelfile} does not store dataset_params, needed to normalize the test set")
        return self._load_split(train=False)

    @property
    def train_dataset(self):
        return self._train['train_dataset']

    @property
    def train_data(self):
        return self._train['train_data']

    @property
    def test_dataset(self):
        return self._test['val_dataset']

    @property
    def test_data(self):
        return self._test['val_data']

    def load_model(self, dataset):
        """Loads the model, using dataset only for its input and output sizes."""
        from apf.io import load_model
        from apf.models import initialize_model
        model, _ = initialize_model(self.config, dataset, self.device)
        load_model(self.modelfile, model, self.device, weights_only=False)
        model.eval()
        return model

    @cached_property
    def manifest(self):
        return read_json(self.paths.manifest)

    def sim_track_data(self):
        """Returns the test track with the simulated frames written in, and the frame-since-prompt array."""
        import copy
        sim_track = copy.deepcopy(self.test_data['track'])
        sim_track.array[:] = np.load(self.paths.sim_track)
        return sim_track, np.load(self.paths.sim_frame)

    def sim_dataset(self):
        from apf.evaluation import create_dataset_from_ref_data
        sim_track, sim_frame = self.sim_track_data()
        return create_dataset_from_ref_data(self.test_dataset, self.test_data, sim_track, sim_frame)

    def valid_frames(self, sim_frame):
        chunk = self.params.frames_chunk_for_eval
        return {
            'first': np.logical_and(sim_frame > 0, sim_frame <= chunk),
            'last': sim_frame > self.params.sim_len - chunk,
            'all': sim_frame > 0,
        }
