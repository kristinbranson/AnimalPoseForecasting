# Containers

`conda_environment_linux.yml`: Direct dump of the environment that we are using on Linux.
`conda_environment_minimal.yml`: Manually created environment with the packages installed, may work better cross-platform, but untested.

# Older versions:
`previous/conda_environment_linux_<timestamp>.yml`
`previous/conda_environment_minimal_<timestamp>.yml`

# Updates

## 20251124

Updated to torch 2.9.1 and python 3.13
# RatInABox (synthrat experiment only)

The synthetic-rat code (`synthrat/`, `experiments/synthrat.py`) needs a patched
RatInABox that is not on PyPI and is not in the environment files:

```bash
git clone -b speedup git@github.com:kristinbranson/RatInABox.git
pip install -e RatInABox
```

The fork adds `is_array=True` batch evaluation for head-direction / velocity / speed /
egocentric boundary cells, plus `chunk_size`, `n_workers` and `parallel_threshold` on
`BoundaryVectorCells.get_state`. Stock upstream RatInABox fails **silently** in two ways,
so check this first if sensory features look wrong: the performance arguments are
swallowed by `**kwargs` (out-of-memory, or roughly 8x slower), and `SpeedCell` returns
shape `(1,)` instead of `(1, T)`.
