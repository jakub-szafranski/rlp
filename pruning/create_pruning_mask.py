import json
import numpy as np
import torch
from pathlib import Path
from typing import Tuple

_NEURON_IMPORTANCE_PATH = Path(__file__).parent / "neuron_importance_with_means.json"

with open(_NEURON_IMPORTANCE_PATH, 'r') as f:
    _NEURON_IMPORTANCE_DATA = json.load(f)


def get_pruning_mask_and_means(
    fraction: float, 
    layer: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate pruning mask and means for FLAP bias compensation.
    
    Args:
        fraction: Fraction of neurons to prune (0.0 to 1.0)
        layer: Layer index (int)
        device: Device for torch tensors
        
    Returns:
        mask: Boolean torch tensor (True = keep, False = prune)
        prune_indices: Torch tensor of neuron indices to prune
        prune_means: Torch tensor of activation means for pruned neurons (for bias compensation)
    """
    layer_str = str(layer)
    layer_data = _NEURON_IMPORTANCE_DATA[layer_str]
    sorted_indices = layer_data["indices"]  # Most important first
    sorted_means = layer_data["means"]
    total = layer_data["total_neurons"]

    if len(sorted_indices) != total or len(sorted_means) != total:
        raise ValueError(f"Data length mismatch for layer {layer}: expected {total}, got {len(sorted_indices)} indices and {len(sorted_means)} means.")

    # Keep top (1-fraction), prune bottom fraction
    keep_count = int((1 - fraction) * total)
    keep_idx_list = sorted_indices[:keep_count]
    prune_idx_list = sorted_indices[keep_count:]
    prune_means_list = sorted_means[keep_count:]
    # Create torch bool mask on device
    mask = torch.zeros(total, dtype=torch.bool, device=device)
    mask[torch.tensor(keep_idx_list, dtype=torch.long, device=device)] = True
    # Convert to torch tensors for FLAP compensation
    prune_indices = torch.tensor(prune_idx_list, dtype=torch.long, device=device)
    prune_means = torch.tensor(prune_means_list, dtype=torch.float32, device=device)
    return mask, prune_indices, prune_means



def make_mask_fn(fractions: list[float], device: str = "cuda"):
    """
    Create a mask_fn compatible with prune_with_undo_pack.
    
    Args:
        fraction: Fraction of neurons to prune (0.0 to 1.0)
        device: Device for torch tensors
        
    Returns:
        Function that takes layer_idx and returns (mask, prune_indices, prune_means)
    """
    def mask_fn(layer_idx: int):
        fraction = fractions[layer_idx]
        return get_pruning_mask_and_means(fraction, layer_idx, device)
    
    return mask_fn