#!/bin/bash
# Run synthrat/scripts/samestart_rollouts.py with the transformer312 conda env.
# All arguments are passed through to the Python script; see its docstring.
# Works on a Janelia cluster node or a workstation ($HOME resolves to the
# local miniforge3 in either place).
#
# Example: submit from the workstation as a GPU job on the cluster. The log
# directory must exist before submitting. The model needs ~2 GB of GPU
# memory, so any GPU queue fits; gpu_l4 is the cheapest.
#   ssh login1 'bash -l -c "cd <repository root> && \
#       bsub -n 8 -gpu num=1 -q gpu_l4 -W 240 -J synthrat_samestart_0_1000 \
#       -o synthrat/data/logs/samestart_%J.log \
#       synthrat/scripts/run_samestart_rollouts.sh --episode-start 0 --episode-stop 1000"'
# To split the episodes over an array job, pass --episode-start/--episode-stop
# computed from $LSB_JOBINDEX in a small submit script.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Cluster jobs can fail on a missing /run/user/<uid>; nothing here needs it.
unset XDG_RUNTIME_DIR

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate transformer312
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

# Record which GPU (if any) the job got; failure here is not fatal.
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

python -m synthrat.scripts.samestart_rollouts "$@"
