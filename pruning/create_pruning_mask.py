import json
import torch
from pathlib import Path
from typing import Tuple

### Cluster-based pruning ###
_NEURON_IMPORTANCE_CLUSTER_PATH = Path(__file__).parent / "cluster_neuron_importance_64.json"
_LAYER_TO_CLUSTER_CLUSTER_PATH = Path(__file__).parent / "layer_to_cluster_mapping_64.json"

with open(_NEURON_IMPORTANCE_CLUSTER_PATH, 'r') as f:
    _CLUSTER_IMPORTANCE_DATA = json.load(f)

with open(_LAYER_TO_CLUSTER_CLUSTER_PATH, 'r') as f:
    _LAYER_TO_CLUSTER_MAPPING = json.load(f)

def get_pruning_mask_cluster_pruning(
    cluster: int,
    fraction: float, 
    layer_idx: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate pruning mask and means for FLAP bias compensation.
    Args:
        cluster: Cluster index (int)
        fraction: Fraction of neurons to prune (0.0 to 1.0)
        layer_idx: Layer index (int)
        device: Device for torch tensors
    Returns:
        mask: Boolean torch tensor (True = keep, False = prune)
        prune_indices: Torch tensor of neuron indices to prune
        prune_means: Torch tensor of activation means for pruned neurons (for bias compensation)
    """    
    cluster_layer_data = _CLUSTER_IMPORTANCE_DATA[str(cluster)][str(layer_idx)]
    
    sorted_indices = cluster_layer_data["indices"]  # Most important first
    sorted_means = cluster_layer_data["means"]
    
    # Keep top (1-fraction), prune bottom fraction
    keep_count = int((1 - fraction) * len(sorted_indices))
    keep_idx_list = sorted_indices[:keep_count]
    prune_idx_list = sorted_indices[keep_count:]
    prune_means_list = sorted_means[keep_count:]
    
    mask = torch.zeros(14336, dtype=torch.bool, device=device)
    mask[torch.tensor(keep_idx_list, dtype=torch.long, device=device)] = True
    
    prune_indices = torch.tensor(prune_idx_list, dtype=torch.long, device=device)
    prune_means = torch.tensor(prune_means_list, dtype=torch.float32, device=device)
    
    return mask, prune_indices, prune_means

### Layer-based pruning ###
_NEURON_IMPORTANCE_LAYER_PATH = Path(__file__).parent / "neuron_importance_layer.json"

with open(_NEURON_IMPORTANCE_LAYER_PATH, 'r') as f:
    _NEURON_IMPORTANCE_LAYER_DATA = json.load(f)

def get_pruning_mask_layer_pruning(
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
    layer_data = _NEURON_IMPORTANCE_LAYER_DATA[layer_str]
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



def make_mask_fn(fractions: list[float], device: str = "cuda", pruning_type: str = "cluster64"):
    """
    Create a mask_fn compatible with prune_with_undo_pack.
    Args:
        fractions: List of fractions of neurons to prune (0.0 to 1.0) for each cluster
        device: Device for torch tensors
    Returns:
        Function that takes layer_idx and returns (mask, prune_indices, prune_means)
    """
    if pruning_type in ["cluster64", "cluster128"]:
        def mask_fn(layer_idx: int):
            clusters = _LAYER_TO_CLUSTER_MAPPING[str(layer_idx)]
            
            prune_indices = []
            prune_means = []
            masks = []
            
            for cluster in clusters:
                fraction = fractions[cluster]
                mask, p_indices, p_means = get_pruning_mask_cluster_pruning(
                    cluster, fraction, layer_idx, device=device
                )
                masks.append(mask)
                prune_indices.append(p_indices)
                prune_means.append(p_means)
            
            prune_indices = torch.cat(prune_indices)
            prune_means = torch.cat(prune_means)
            mask = torch.stack(masks).any(dim=0)
            
            return mask, prune_indices, prune_means
    elif pruning_type == "layer":
        def mask_fn(layer_idx: int):
            fraction = fractions[layer_idx]
            return get_pruning_mask_layer_pruning(fraction, layer_idx, device)
    else:
        raise ValueError(f"Unknown pruning_type: {pruning_type}")
    
    return mask_fn