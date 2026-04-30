Code for generating data from the RatInABox synthetic rat model and feeding
it into APF.

## End-to-end workflow

Two scripts in this directory produce the saved trajectories that downstream
APF training consumes:

1. **`freeze_reinforcement_learning_example.py`** — runs the RatInABox RL
   example notebook (a TD-learning agent that finds a hidden reward),
   snapshots the live Env / Agent / Inputs / Reward / ValueNeuron / Sensory
   state via `apf_ratinabox.get_*_info`, and pickles it to
   `data/ratinabox_rl_state_<timestamp>.pkl`.

2. **`generate_data.py`** — loads that state, rehydrates the live objects,
   rolls out N evaluation episodes (parallel via `multiprocessing`), and
   pickles them to `data/ratinabox_rl_traindata_<timestamp>.pkl` and
   `data/ratinabox_rl_valdata_<timestamp>.pkl`. Each episode is `{'pos':
   (T, 2), 'head_direction': (T, 2), 'vel': (T, 2)}`. The same script also
   plots example trajectories and a velocity histogram for quality control.

## `data/` layout

`data/` is gitignored — has to be (re-)generated locally via the two scripts
above. Filenames embed the timestamp so multiple runs coexist; downstream
code reads whichever one your config points at.

## `apf_ratinabox.py` entry points

The module is the glue between RatInABox and APF. Functions you'll typically
call from a notebook or script:

- `make_dataset(config, filename, ...)` — build an APF
  `apf.dataset.Dataset` from a saved trajectory pkl. Wraps pose into
  `GlobalVelocity`, computes sensory features via the `Sensory` operation,
  applies `Zscore` / `Discretize` per the config, and returns the Dataset
  ready for a DataLoader. Pass `cached_sensory_array=` to skip the slow
  sensory pass when you already have the firing-rate array.
- `compute_sensory(track, Sensory=...|info=...)` — per-trajectory firing
  rates for one episode, returned as `{pop_name: (T, n_cells)}`. Either
  pass live `Sensory` (dict of Neurons) or `info` (dict containing
  `env_info`/`agent_info`/`sensory_info`) and let it rehydrate.
- `init_sensory(Ag, Env, cell_config)` — instantiate live Neurons
  populations from a config dict (`{name: {type: ..., **kwargs}}`).
- `rehydrate_data(data)` / `rehydrate_env` / `rehydrate_agent` /
  `rehydrate_value_neuron` / `rehydrate_sensory` — rebuild live RatInABox
  objects from saved info dicts (the inverse of `get_*_info`).
- `generate_episodes(...)` — batch eval rollouts with optional
  multiprocessing.
- `rect_polar_grid` — `cell_arrangement` callable for vector cells. Lays
  out a fixed `n_angles`-cells-per-ring polar grid using diverging-manifold
  (Hartley) ring spacing; firing-rate arrays then reshape cleanly to
  `(T, n_rings, n_angles)`.
- `Sensory` — an `apf.dataset.Operation` that bakes sensory firing-rate
  computation into a Dataset's preprocessing chain.
- `debug_plot_sample` / `visualize_sensory` / `plot_episode` —
  visualization helpers (trajectory + sensory firing rates, with optional
  predicted-pose overlay).
- `get_feature_names(neuron|info)` / `get_all_feature_names(Sensory|info)`
  — human-readable cell names per population, with `_ring{i}_ang{j}`
  suffixes when the layout is a `rect_polar_grid`.
- `convert_torch_to_numpy` (re-exported from `apf.utils`) — recurse into
  nested dicts/lists/tuples and convert torch tensors to numpy.

## Performance notes

Sensory feature computation is the dominant cost in `make_dataset`. For a
2.6M-frame train file with `FieldOfViewBVCs` (128 cells, ±150° FoV,
`dtheta=2°`), expect roughly:

- ~95 min serial.
- ~55 min serial after the broadcast / `is_array` rewrite (default).
- ~12 min with `n_workers=8` (the wired-in default in
  `_firingrate_over_trajectory`).

Tuning knobs live on `BoundaryVectorCells.get_state` in the submodule:
`chunk_size` (default 50000), `n_workers` (default 8 via apf_ratinabox),
`parallel_threshold` (default 200000). Smaller `chunk_size` keeps peak
memory bounded; larger `n_workers` helps up to ~8 on most hardware
(memory-bandwidth bound after that).

For repeated runs, cache the sensory firing-rate array to disk and pass it
back into `make_dataset(cached_sensory_array=...)` on the next load —
turns ~12 min into seconds.

## Citation

If you use the synthetic-rat data in a publication, please cite RatInABox:

> George, T.M., Rastogi, M., de Cothi, W., Clopath, C., Stachenfeld, K., &
> Barry, C. (2024). RatInABox, a toolkit for modelling locomotion and
> neuronal activity in continuous environments. *eLife*, 13, e85274.

## ratinabox_repo (submodule)

`ratinabox_repo/` is the [RatInABox](https://github.com/RatInABox-Lab/RatInABox)
library, tracked as a git submodule pinned to our fork
[kristinbranson/RatInABox @ speedup](https://github.com/kristinbranson/RatInABox/tree/speedup).
The fork carries perf work on `FieldOfViewBVCs.get_state` (broadcast operands,
chunking, multiprocessing pool, tqdm progress) plus `is_array=True` paths for
HDC / VelocityCells / SpeedCell so they can be called once per trajectory.

### First-time setup after cloning

```bash
git clone --recurse-submodules git@github.com:kristinbranson/AnimalPoseForecasting.git
# or, if you already cloned without submodules:
git submodule update --init --recursive
```

Install the submodule as an editable package (so local edits take effect on import):

```bash
pip install -e synthrat/ratinabox_repo
```

### Recommended local git config

Set once per clone so submodules stay in sync automatically:

```bash
git config submodule.recurse true             # pull/checkout recurse into submodules
git config push.recurseSubmodules on-demand   # push pushes submodule commits first
```

After this, plain `git pull` / `git push` / `git checkout` Just Work.

### Editing the submodule

The submodule is its own git repo. After `git submodule update`, it's in
**detached HEAD** state — always check out the branch first or your commits
won't belong to anything:

```bash
cd synthrat/ratinabox_repo
git checkout speedup           # attach to the branch (the .gitmodules-tracked one)
# ...edit, test...
git add -u && git commit -m "..."
git push                       # to kristinbranson/RatInABox

cd ../..
git add synthrat/ratinabox_repo   # bump the pin in the parent repo
git commit -m "Bump ratinabox_repo to <new SHA>"
git push                       # parent (auto-pushes submodule first if needed)
```
