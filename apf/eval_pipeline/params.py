"""Evaluation parameters and on-disk layout of one evaluation run."""
import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field

DEFAULT_RESULTS_ROOT = "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/eval_results"
DEFAULT_CACHE_ROOT = "/nrs/branson/AnimalPoseForecasting/eval_cache"


@dataclass
class EvalParams:
    """Everything that affects the results of a run. A run refuses to resume if any of these change."""
    sim_len: int = 512
    first_sim_frame: int = 512
    stride: int | None = None  # None uses sim_len
    probe_layer: int = 5
    min_labeled: int = 500  # A category is trained if it has more than this many positive and negative train frames
    categories: list[str] | None = None  # Restrict the probes to these categories
    frames_chunk_for_eval: int = 64  # Size of the 'first' and 'last' frame groups
    min_bout_dur: int = 20
    max_dists_to_prompt: tuple[int, ...] = (64, 512)
    video_actions: list[str] | None = None  # None uses perframe categories and walking
    video_samples: int = 10

    # Probe training (same defaults as the earlier evaluation scripts)
    probe_mlp_layers: tuple[int, ...] = ()
    probe_lr: float = 0.0001
    probe_min_epochs: int = 5
    probe_max_epochs: int = 100
    probe_block_rows: int = 65536  # Rows read from disk at once by a loader worker
    probe_num_workers: int = 6

    # Hidden state pass
    causal_hidden: bool = True  # Must be True to match training and simulation; False only reproduces old caches
    hidden_batch_size: int = 128  # Larger gave no speedup on a TITAN RTX and 512 ran out of its 24 GB
    hidden_num_workers: int = 0
    hidden_mixed_precision: bool = True  # Real data only. Simulation always uses full precision.
    frames_per_unit: int = 1000000  # Resume granularity of the hidden state passes
    feature_dtype: str = 'float16'

    # Number of array elements per task
    n_simulate_jobs: int = 16
    n_feature_jobs: int = 12
    n_probe_jobs: int = 16
    n_logit_jobs: int = 8
    n_video_jobs: int = 8

    # Limits for smoke tests
    max_chunks: int | None = None
    max_actions: int | None = None
    max_frames_per_split: int | None = None

    def to_dict(self):
        return json.loads(json.dumps(dataclasses.asdict(self)))

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def with_overrides(self, assignments):
        """Applies a list of 'key=value' strings, with values parsed as JSON where possible."""
        names = {f.name for f in dataclasses.fields(self)}
        values = dataclasses.asdict(self)
        for assignment in assignments:
            key, sep, raw = assignment.partition('=')
            if not sep or key not in names:
                raise ValueError(f"Invalid --set {assignment!r}. Valid keys: {sorted(names)}")
            try:
                values[key] = json.loads(raw)
            except json.JSONDecodeError:
                values[key] = raw
        return EvalParams(**values)


@dataclass
class Resources:
    """LSF resources per task. These can change between runs without invalidating results."""
    queue: str = 'gpu_a100'
    cpu_queue: str = 'normal'
    gpu_slots: int = 12
    cpu_slots: int = 24
    walltime: str = '24:00'
    overrides: dict = field(default_factory=dict)  # task name -> {'queue': ..., 'slots': ..., 'walltime': ...}


def run_name(configfile, modelfile):
    configname = os.path.splitext(os.path.basename(configfile))[0]
    modelname = os.path.splitext(os.path.basename(modelfile))[0]
    return f"{configname}_{modelname}"


def model_fingerprint(modelfile):
    stat = os.stat(modelfile)
    return {'size': stat.st_size, 'mtime': stat.st_mtime}


class RunPaths:
    """Files of one run. Results are small and kept in run_dir, hidden states go in cache_dir."""

    def __init__(self, run_dir, cache_dir):
        self.run_dir = run_dir
        self.cache_dir = cache_dir
        self.run_file = os.path.join(run_dir, 'run.json')
        self.manifest = os.path.join(run_dir, 'manifest.json')
        self.done_dir = os.path.join(run_dir, 'done')
        self.log_dir = os.path.join(run_dir, 'logs')
        self.train_index = os.path.join(run_dir, 'train_index.npz')
        self.test_index = os.path.join(run_dir, 'test_index.npz')
        self.sim_track_dir = os.path.join(run_dir, 'sim', 'tracks')
        self.sim_track = os.path.join(run_dir, 'sim', 'sim_track.npy')
        self.sim_frame = os.path.join(run_dir, 'sim', 'sim_frame.npy')
        self.probe_dir = os.path.join(run_dir, 'probes')
        self.test_logit_dir = os.path.join(run_dir, 'logits', 'test')
        self.sim_logit_dir = os.path.join(run_dir, 'logits', 'sim')
        self.classifications = os.path.join(run_dir, 'classifications.pkl')
        self.test_logits = os.path.join(run_dir, 'test_logits.npy')
        self.results_dir = os.path.join(run_dir, 'results')
        self.feature_dir = os.path.join(cache_dir, 'train_features')
        self.sim_hidden_dir = os.path.join(cache_dir, 'sim_hidden')

    def make_dirs(self):
        for path in [self.done_dir, self.log_dir, self.sim_track_dir, self.probe_dir, self.test_logit_dir,
                     self.sim_logit_dir, self.results_dir, self.feature_dir, self.sim_hidden_dir]:
            os.makedirs(path, exist_ok=True)

    def job_prefix(self):
        return 'ev_' + hashlib.sha1(self.run_dir.encode()).hexdigest()[:8]


def atomic_path(path):
    """Returns a temporary path next to path, to be finished with os.replace once fully written.

    The suffix is kept since numpy and matplotlib pick their behaviour from it.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    suffix = os.path.splitext(path)[1]
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.tmp_', suffix=suffix)
    os.close(fd)
    return tmp


def write_json(path, data):
    tmp = atomic_path(path)
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)
