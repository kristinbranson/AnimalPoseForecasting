# External data

Nothing in this repository can run without four files that are **not in git** — they total
about 49 GB. This directory is where they go.

The code does not yet read from here: the paths below are currently hardcoded to a Janelia
cluster location (see [Current state](#current-state)). If you are outside Janelia, download
the artifacts into this directory and pass the two you need explicitly:

```bash
python -m interpretability.interpret probe-residual \
    --config external_data/config_fly_llm_predvel_optimalbinning_20251113.json \
    --model  external_data/flypredvel_20251007_20251114T194024_bestepoch200.pth \
    --layer 9
```

The dataset files are not passed on the command line — they are found via `datadir` +
`intrainfilestr` / `invalfilestr` **inside the config JSON**, which you must edit to point at
wherever you put them.

## What you need

| File | Size | What it is |
|------|------|------------|
| `config_fly_llm_predvel_optimalbinning_20251113.json` | 9 KB | Model + data + training config. Also names the two dataset files below, and the bin edges used to discretize velocity. Everything reads this first. |
| `flypredvel_20251007_20251114T194024_bestepoch200.pth` | 2.4 GB | Trained transformer weights (best epoch 200). What the probes and rollouts interpret. |
| `usertrain_v3.npz` | 31 GB | Training set: fly tracking data. Loaded by `init_flyllm(needtraindata=True)`, which every entry point calls. |
| `testtrain_v3.npz` | 16 GB | Validation set, same format. |

## Which code needs which

| | config | weights | `usertrain_v3` | `testtrain_v3` |
|---|:---:|:---:|:---:|:---:|
| `interpretability/interpret.py probe-residual` | ✅ | ✅ | ✅ | — |
| `interpretability/interpret.py probe-attention` | ✅ | ✅ | ✅ | — |
| `interpretability/interpret.py simulate` | ✅ | ✅ | ✅ | — |
| `notebooks/agent_fly.py` | ✅ | ✅ | ✅ | — |
| `notebooks/train_fly_llm.py` (training from scratch) | ✅ | — | ✅ | ✅ |

The interpretability tools only touch the **training** split: they probe what the model
learned, using `dataset.concepts` computed over the training data. `testtrain_v3.npz` is only
needed if you are training or evaluating a model, not probing one.

There is no small fixture and no smoke-test path — `needtraindata=True` means even a
one-trajectory probe pays the full 31 GB load. Budget for that.

## Current state

The paths are hardcoded, and valid **only on the Janelia cluster**:

| Artifact | Hardcoded location | Set in |
|---|---|---|
| config | `/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/` | `interpretability/interpret.py` (`DEFAULT_CONFIG`), `notebooks/agent_fly.py` |
| weights | `/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/notebooks/flyllm_models/` | `interpretability/interpret.py` (`DEFAULT_MODEL`), `notebooks/agent_fly.py` |
| datasets | `/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/` | `datadir` inside the config JSON |

Note the config lives under **`bransonk`'s working copy of this repo**, not in the repo
itself — cloning the repo does not get you the config, even on the cluster. That is the first
thing to fix if you want a clone to run.

Both `--config` and `--model` can be overridden on the command line, so the hardcoded values
are defaults, not requirements. The dataset paths cannot: they come from the config.

## TODO

- [ ] Upload the four artifacts and add download links / checksums here.
- [ ] Point the defaults at this directory (or make `--config` / `--model` required) so a
      clone fails with a legible error instead of a path that exists on one person's disk.
- [ ] Commit the config JSON to the repo — it is 9 KB and is the one artifact small enough
      to version, yet its absence blocks everything else.
