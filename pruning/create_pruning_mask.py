import json
import numpy as np
from pathlib import Path

_CLUSTER_MASKS_PATH = Path(__file__).parent / "cluster_masks_up_2l_16c.json"
_CLUSTER_LAYER_MAPPING_PATH = Path(__file__).parent / "cluster_layer_mapping_up_2l_16c.json"

with open(_CLUSTER_MASKS_PATH, 'r') as f:
    _CLUSTER_MASKS = json.load(f)

with open(_CLUSTER_LAYER_MAPPING_PATH, 'r') as f:
    _CLUSTER_LAYER_MAPPING = json.load(f)


def _reduce_logical_or(array: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce(array, axis=0)


def get_pruning_mask(clusters_to_prune: list[str], layer: str) -> np.ndarray:
    """
    Generates pruning masks for the specified layer based on clusters to prune.
    For now only 'up_proj' masks are considered.
    """
    if layer in {"0", "1", "30", "31"}:
        return np.ones((14_336,))  # No pruning for first and last layers
    up_projs = []
    # down_projs = []
    for cluster in clusters_to_prune:
        if layer in _CLUSTER_LAYER_MAPPING[cluster]:
            up_projs.append(_CLUSTER_MASKS["up_proj"][layer][cluster])
            # down_projs.append(_CLUSTER_MASKS["down_proj"][layer][cluster])
    if not up_projs:
        return np.ones((14_336,))
    up_proj = 1 - _reduce_logical_or(np.stack(up_projs, axis=0))
    # down_proj = 1 - _reduce_logical_or(np.stack(down_projs, axis=0)) - -- IGNORE ---
    return up_proj #, down_proj -- IGNORE ---

