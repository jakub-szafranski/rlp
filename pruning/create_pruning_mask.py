import json
import numpy as np
from pathlib import Path

_NEURON_IMPORTANCE_PATH = Path(__file__).parent / "neuron_importance_up_proj.json"

with open(_NEURON_IMPORTANCE_PATH, 'r') as f:
    _SORTED_INDICES = json.load(f)


def get_pruning_mask(fraction: float, layer: str) -> np.ndarray:
    """
    Generates pruning masks for the specified layer based on fraction to prune.
    Prunes the least important neurons (bottom fraction).
    """
    if layer in {"0", "1", "30", "31"}:
        return np.ones((14336,), dtype=bool)  # No pruning for first and last layers
    sorted_indices = _SORTED_INDICES[layer]
    total = len(sorted_indices)
    keep_count = int((1 - fraction) * total)
    keep_indices = set(sorted_indices[:keep_count])
    mask = np.zeros(total, dtype=bool)
    mask[list(keep_indices)] = True
    return mask

