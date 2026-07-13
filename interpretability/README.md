# Interpretability

Tools for asking whether the FlyLLM pose-forecasting transformer internally represents a
human-interpretable behavioral concept — and if so, where.

Everything lives in one module, `interpret.py`, with three subcommands.

> **You need the config, the model weights and the training data before any of this runs**
> — about 49 GB, none of it in git. The defaults point at a Janelia cluster path. See
> [`external_data/README.md`](../external_data/README.md).

## The concept

Concepts are defined in `apf/concepts.py` as a Dataset `Operation` (`HumanConcept`), computed
from **raw velocity, before z-scoring or discretization**, so thresholds stay in physical mm/s.
Each frame gets one label:

| Label | Name | Meaning |
|-------|------|---------|
| `+1` | `start_walking` | Stopped for the last `tstopped` (0.5 s), will exceed `thresh_walking` (15 mm/s) within `tfuture` (1.0 s) |
| `-1` | `stays_stopped` | Same stopped history, but stays below `thresh_stopped` (5 mm/s) for the whole future window |
| `0` | `neutral` | Neither |
| `NaN` | — | Invalid data (missing velocity, or a track boundary) |

Concepts are switched on via `compute_concepts` in the config, stored on `Dataset.concepts`,
and never reach the loss — they are analysis-only. `interpret.py` sets this override for you.

## Trajectory anchoring

This convention is shared by all three subcommands and is the thing to understand before
reading any result:

> A concept frame `f` for agent `a` becomes the model input window starting at `f - contextl`,
> so the concept event lands at the **final** position of the context window.

The model therefore sees only the run-up to the event, never the event itself. A probe that
succeeds is reading an *anticipatory* representation, not an observation of walking.

Binary probes use `start_walking` (+1) as the positive class and `stays_stopped` (-1) as the
negative. `neutral` is sampled but unused.

## Usage

```bash
# probe the residual stream at layer 9
python -m interpretability.interpret probe-residual --layer 9 --pooling mean --probe_type linear

# probe attention maps at layer 9
python -m interpretability.interpret probe-attention --layer 9 --max-samples 1000

# roll the model out around concept events and animate it
python -m interpretability.interpret simulate --n-samples 5 --output-dir animations/
```

Sweep layers:

```bash
for layer in $(seq 0 10); do
    python -m interpretability.interpret probe-residual --layer $layer --output-dir results/
done
```

### Shared arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (see `--help`) | Checkpoint to interpret. **Always loaded** unless `--random-init` |
| `--mode` | `test` | `init_flyllm` mode. `test` also loads the config from the checkpoint |
| `--random-init` | off | Interpret an **untrained** model. Control baseline only |
| `--n-samples` | 100 | Trajectories sampled per concept |
| `--sampling-method` | `random` | `random`: any frame with the label. `boundary`: frames at the transition |
| `--layer` | 9 | Layer to probe (probes only) |
| `--pooling` | `mean` | `mean`, `max`, `last`, `cnn` (residual only) |
| `--probe_type` | `linear` | `linear`, `2layer`, `deep` (residual only) |
| `--pca-components` | off | Reduce the residual stream to N principal components first (residual only) |

### PCA-reduced probes

```bash
python -m interpretability.interpret probe-residual --layer 2 --pca-components 8 --probe_type 2layer
```

PCA is fit on the **training split only** — fitting on both splits would leak validation
structure into the probe's input and inflate accuracy. Each timestep is a PCA sample, so the
fit sees `(n_samples × seq_len, hidden_dim)` and the sequence dimension survives. Explained
variance is logged, and the fitted `PCA` object is saved in the checkpoint so the probe's
input can be reproduced at inference time. The solver is seeded from `--seed`: at this
dimensionality sklearn picks the randomized SVD solver, which is nondeterministic if unseeded.

### The untrained control

**Always run this before believing a probe result.** A probe has its own capacity, and a
sufficiently expressive one can reach high accuracy by fitting input statistics that have
nothing to do with what the model learned. The control tells you which you are looking at:

```bash
python -m interpretability.interpret probe-residual --layer 9 --random-init
```

Accuracy should land near chance (0.5). If an untrained model scores as well as the trained
one, the probe is reading the *input*, not the *representation*. Control runs are written with
a `_randominit` suffix so they cannot overwrite a real result.

## Residual stream layers

- **Layer 0** — after positional encoding (the input to the transformer)
- **Layers 1–10** — after each transformer block

Rough expectation: early layers carry low-level motion features, late layers carry
higher-level behavioral structure. That is a hypothesis to test with a layer sweep, not an
assumption to build on.

## Notes

- **Causal masking.** The probes call the model outside the training loop, which normally
  supplies `src_mask`. `apf/models.py` generates the causal mask itself when `is_causal=True`
  and no mask is given. Without that, captured internals would be contaminated by future
  frames — every result here depends on it.
- **Attention extraction mutates the model.** PyTorch's fused attention path won't return
  per-head weights, so `ModelInterpreter.enable_attention_output()` monkey-patches each
  `MultiheadAttention.forward`. Use `ModelInterpreter` as a context manager (as the
  subcommands do) so the patch is always reverted.
- **Dropped samples are logged.** Anchors too close to the start of a recording, rollouts that
  diverge to NaN, and NaN attention maps are all skipped, with counts reported — a silently
  truncated run would otherwise read as full coverage.

## Related files

- `apf/concepts.py` — `HumanConcept` operation
- `apf/models.py` — model architecture, causal mask generation
- `apf/dataset.py` — `Dataset.concepts`
- `experiments/flyllm.py` — where concepts are computed during dataset construction
- `notebooks/agent_fly.py` — demo notebook showing the config override
