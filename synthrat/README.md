# synthrat: a RatInABox synthetic rat for APF

A synthetic dataset where the ground truth is known exactly: a simulated rat moves under
RatInABox's value-function-guided policy toward a hidden reward, and an APF model is
trained to forecast its trajectory from what the rat senses. Because the policy, the environment
and the sensory neurons are all known, a forecast distribution can be compared against the
groundtruth distribution of trajectories. 

## Structure

Generation of synthetic data from RatInABox is separate from APF model training and evalution. 

**Shared code**: 
- `synthrat/sensory.py`: rebuilds the RatInABox objects from the dicts stored in a pickle and 
   computes the sensory firing rates that are the APF model inputs.
- `synthrat/plotting.py`: plotting helpers

**Generation**:
- `synthrat/freeze_reinforcement_learning_example.py`: jupytext notebook that uses RatInAbox 
  to train an RL agent, based on one of the RatInABox examples, outputs to 
  `{datadir}/ratinabox_rl_state_{timestamp}.pkl`
- `synthrat/generate_data.py`: generate synthetic trajectories from the RatInABox RL agent frozen 
   in `{datadir}/ratinabox_rl_state_{timestamp}.pkl` and saves the training data to 
   `{datadir}/ratinabox_rl_traindata_{timestamp}.pkl` and the validation data to 
   `{datadir}/ratinabox_rl_valdata_{timestamp}.pkl`. 
- `synthrat/inspect_synthrat_data.py`: visualizations of trajectory data written by 
   `synthrat.generate_data`: what the agent sees, what the trajectories look like, and how the 
   velocities are distributed. 

**APF modeling**:
- `experiments/synthrat.py`: APF code for synthrat trajectories, defining the `Operation`, `Data`,
  and `Dataset` objects needed for this data, and defines the `simulate()` function. 
- `notebooks/agent_synthrat.py`: jupytext notebook demonstrating how to train and evaluate the APF
  forecasting model on trajectories loaded from `{datadir}/ratinabox_rl_{split}data_{timestamp}.pkl`
- `synthrat/config.py`: reads the configuration for the APF model. 

## Workflow

**1. Train and freeze the policy.** `synthrat/freeze_reinforcement_learning_example.py`,
adapted from RatInABox's reinforcement-learning demo, trains a `ValueNeuron` by TD learning
until the agent reliably finds a reward hidden behind a wall, then snapshots the
environment, agent, place-cell inputs, reward population, value neuron and sensory
configuration as plain dicts to `synthrat/data/ratinabox_rl_state_<timestamp>.pkl`. Only
needed when the environment, the policy or the sensory populations change; otherwise reuse
an existing state file.

**2. Generate trajectories.** Given a saved policy state — environment, agent, place-cell
inputs, reward population and value neuron, captured as plain dicts by the `get_*_info`
functions:

```bash
python -m synthrat.generate_data --state synthrat/data/ratinabox_rl_state_20260423.pkl \
    --out-dir synthrat/data --timestamp 20260423 \
    --train-episodes 10000 --val-episodes 1000 --workers 32
```

Episodes are independent, so `--workers` scales nearly linearly. Each episode is
`{'pos': (T, 2), 'head_direction': (T, 2), 'vel': (T, 2)}`; the written pickle is the state
plus this split's `track` and `hidden`. To change what the rat senses, pass
`--sensory-config <cell_config.json>`: the populations are rebuilt, the state pickle is
rewritten, and the episodes are generated against the new configuration.

**3. Visualize the generated trajectories.** `synthrat/inspect_synthrat_data.py` draws what the 
agent sees at a few timepoints, some trajectories, and the velocity distribution.

**4. Train and evaluate.** `notebooks/agent_synthrat.py` builds the datasets, trains, rolls
the model out open-loop from held-out initial conditions, and compares those rollouts
against the original policy run from the same burn-in
(`generate_data.run_policy_from_burn_in`).

## Requirements

RatInABox must be importable. This was developed against the fork
[kristinbranson/RatInABox @ speedup](https://github.com/kristinbranson/RatInABox/tree/speedup),
which carries performance work on `FieldOfViewBVCs.get_state` (broadcast operands,
chunking, a multiprocessing pool) plus `is_array=True` paths for the head-direction,
velocity and speed cells, so they can be computed once per trajectory instead of per frame.
Install it editable:

```bash
git clone -b speedup git@github.com:kristinbranson/RatInABox.git
pip install -e RatInABox
```

The repository does not track it as a submodule; any RatInABox providing those entry points
will do, but the timings below assume the fork.

## Performance

Computing sensory features dominates dataset construction. For a 2.6M-frame training file
with `FieldOfViewBVCs` (128 cells, ±150° field of view, 2° resolution):

- ~95 min serial, before the broadcast rewrite
- ~55 min serial, after it
- ~12 min with `n_workers=8`, the default in `_firingrate_over_trajectory`

The knobs are on `BoundaryVectorCells.get_state` in the fork: `chunk_size` (default 50,000)
bounds peak memory, `n_workers` helps to about 8 before memory bandwidth limits it, and
`parallel_threshold` (default 200,000) decides when the pool is worth starting.

For repeated runs, cache the firing-rate array and hand it back:
`experiments.synthrat.make_dataset(..., cached_sensory_array=...)` turns those 12 minutes
into seconds. `make_dataset_cache_wrapper` does this bookkeeping for you.

## Citation

If you use the synthetic-rat data in a publication, please cite RatInABox:

> George, T.M., Rastogi, M., de Cothi, W., Clopath, C., Stachenfeld, K., & Barry, C. (2024).
> RatInABox, a toolkit for modelling locomotion and neuronal activity in continuous
> environments. *eLife*, 13, e85274.
