# Model Interpretation & Human Concepts

This directory contains tools for interpreting transformer model representations and simulating behavior based on human-interpretable concepts.

## Overview

### Human Concepts Framework

The `HumanConcept` operation (`apf/concepts.py`) computes behavioral labels from raw velocity data:

| Concept | Label | Description |
|---------|-------|-------------|
| `start_walking` | +1 | Fly was stopped and is about to start walking |
| `stays_stopped` | -1 | Fly was stopped and will remain stopped |
| `neutral` | 0 | Neither condition applies |

### Model Interpretation Tools

| Tool | Target | Method |
|------|--------|--------|
| `train_residual_classifier.py` | Residual stream hidden states | Linear/MLP probes |
| `run_attn_classifier.py` | Attention patterns | CNN classifier |
| `simulate_human_concepts.py` | Model predictions | Trajectory simulation |

## Human Concept Computation

The `HumanConcept` operation (`apf/concepts.py`) detects behavioral transitions:

```python
from apf.concepts import HumanConcept

concepts = HumanConcept(
    concept_type="start_walking",
    fps=150.0,
    sigma=2,  # Gaussian smoothing in frames
    concept_params={
        'thresh_stopped': 5.0,   # Velocity threshold for "stopped" (mm/s)
        'thresh_walking': 15.0,  # Velocity threshold for "walking" (mm/s)
        'tstopped': 0.5,         # Time fly must have been stopped (seconds)
        'tfuture': 1.0,          # Time window to look into future (seconds)
    }
)(velocity, isstart=isstart)
```

### How `start_walking` is Computed

1. **Velocity smoothing**: Gaussian filter with `sigma` frames
2. **Was stopped**: Velocity ≤ `thresh_stopped` for at least `tstopped` seconds
3. **Will walk**: Velocity ≥ `thresh_walking` at some point in next `tfuture` seconds
4. **Will stop**: Velocity ≤ `thresh_stopped` for entire `tfuture` window

Labels:
- `+1`: `was_stopped` AND `will_walk` (start_walking)
- `-1`: `was_stopped` AND `will_stop` (stays_stopped)
- `0`: Neither condition
- `NaN`: Invalid data

## Simulating with Human Concepts

**File:** `simulate_human_concepts.py`

Run model simulations on trajectories labeled with behavioral concepts:

```python
# Sample trajectories by concept
sampled_trajectories = sample_trajectories_by_concept(
    concept_labels=train_dataset.concepts.array,
    n_samples=100
)

# Simulate for each concept
for concept_name, trajectories in sampled_trajectories.items():
    for agent_id, start_frame in trajectories:
        gt_track, pred_track = simulate(
            dataset=train_dataset,
            model=model,
            track_len=track_len,
            agent_ids=[agent_id],
            start_frame=start_frame - config['contextl']
        )
```

This enables analysis of how well the model predicts behavior around key behavioral transitions.

## Residual Stream Probing

**File:** `train_residual_classifier.py`

Train classifiers to detect behavioral concepts from transformer hidden states.

### Quick Start

```bash
python train_residual_classifier.py \
    --layer 9 \
    --pooling mean \
    --probe_type linear \
    --n_samples 1000 \
    --output_dir results/
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--layer` | 9 | Residual stream layer (0-10) |
| `--pooling` | mean | Aggregation: `mean`, `max`, `last`, `cnn` |
| `--probe_type` | linear | Probe: `linear`, `2layer`, `deep` |
| `--sampling_method` | random | How to sample: `random` or `boundary` |
| `--n_samples` | 100 | Trajectories per concept |

### Residual Stream Layers

- **Layer 0**: After positional encoding (input to transformer)
- **Layers 1-10**: After each transformer layer

Interpretation:
- Early (0-3): Low-level motion features
- Middle (4-7): Mid-level behavioral patterns
- Late (8-10): High-level concepts

### Probe Types

- **`linear`**: Single linear layer - tests linear separability
- **`2layer`**: 2-layer MLP with dropout
- **`deep`**: 3-layer MLP for complex boundaries

### Example: Sweep Layers

```bash
for layer in 0 3 6 9; do
    python train_residual_classifier.py \
        --layer $layer \
        --probe_type linear \
        --n_samples 500 \
        --output_dir layer_sweep/
done
```

## Attention Pattern Classification

**File:** `run_attn_classifier.py`

Train CNN classifiers on attention maps to understand what the model attends to.

### Quick Start

```bash
python run_attn_classifier.py \
    --config /path/to/config.json \
    --model /path/to/model.pth \
    --layer 9 \
    --max-samples 1000 \
    --output-dir results/
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--layer` | 9 | Transformer layer |
| `--max-samples` | 1000 | Max samples per class |
| `--lr` | 1e-4 | Learning rate |
| `--lr-scheduler` | plateau | LR scheduling |
| `--clip-grad` | 1.0 | Gradient clipping |
| `--wandb` | False | W&B logging |

### Architecture

```
Input: (batch, num_heads, seq_len, seq_len)
    ↓
Conv2D + MaxPool layers
    ↓
AdaptiveAvgPool2d
    ↓
FC layers with dropout
    ↓
Output: Binary logits
```

## Output Files

Both classifiers generate:

1. **Model checkpoint** (`.pth`): Weights, hyperparameters, history
2. **Training curves** (`_history.png`): Loss/accuracy over epochs
3. **Confusion matrix** (`_confusion_matrix.png`)

## Loading Trained Models

### Residual Stream Classifier

```python
import torch
from train_residual_classifier import ResidualStreamClassifier

checkpoint = torch.load('residual_classifier_layer9_mean_linear_random.pth')

model = ResidualStreamClassifier(
    seq_len=checkpoint['seq_len'],
    hidden_dim=checkpoint['hidden_dim'],
    pooling=checkpoint['pooling'],
    probe_type=checkpoint['probe_type']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best val accuracy: {max(checkpoint['history']['val_acc']):.4f}")
```

### Attention Classifier

```python
import torch
from run_attn_classifier import AttentionCNN

checkpoint = torch.load('attention_classifier_layer9.pth')

model = AttentionCNN(
    num_heads=checkpoint['num_heads'],
    seq_len=checkpoint['seq_len']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Configuration

Both tools use the FlyLLM initialization with concept computation:

```python
overrideconfig = {
    'compute_concepts': True,
    'concept_params': {
        'concept_type': 'start_walking',
        'sigma': 2,
        'thresh_stopped': 5.0,
        'thresh_walking': 15.0,
        'tstopped': 0.5,
        'tfuture': 1.0,
    }
}

res = init_flyllm(
    configfile=configfile,
    mode='test',
    loadmodelfile=model_path,
    needtraindata=True,
    overrideconfig=overrideconfig
)
```

## Tips

### Memory Management

```bash
python train_residual_classifier.py --batch_size 16 --max_samples 500
```

### Quick Testing

```bash
python train_residual_classifier.py \
    --n_samples 50 --max_samples 50 --epochs 10 --patience 3
```

### Reproducibility

```bash
python train_residual_classifier.py --seed 42
```

## Related Files

- `apf/concepts.py` - HumanConcept operation
- `apf/models.py` - Model architecture
- `apf/dataset.py` - Dataset with concept computation
- `flyllm/prepare.py` - Initialization utilities
