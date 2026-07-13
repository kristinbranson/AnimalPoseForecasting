"""Interpretability tools for the FlyLLM pose-forecasting transformer.

See interpretability/interpret.py. Run as:

    python -m interpretability.interpret probe-residual --layer 9
    python -m interpretability.interpret probe-attention --layer 9
    python -m interpretability.interpret simulate
"""

from interpretability.interpret import (
    AttentionProbe,
    ModelInterpreter,
    ResidualProbe,
    attention_layer_name,
    extract_dataset,
    load_flyllm,
    residual_layer_name,
    sample_trajectories_by_concept,
    train_probe,
)

__all__ = [
    'AttentionProbe',
    'ModelInterpreter',
    'ResidualProbe',
    'attention_layer_name',
    'extract_dataset',
    'load_flyllm',
    'residual_layer_name',
    'sample_trajectories_by_concept',
    'train_probe',
]
