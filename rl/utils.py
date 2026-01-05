import numpy as np
from typing import Callable
import torch
from pruning import PrunableLLM, get_pruning_mask


class MaskFunctionAdapter:
    """
    Adapts action (binary cluster vector) to mask_fn for PrunableLLM.
    
    Maps: action[cluster_idx] = 1 means "prune this cluster"
          -> mask_fn(layer_idx) returns boolean tensor (True = KEEP)
    """
    
    def __init__(self, cluster_names: list[str]):
        """
        Args:
            cluster_names: List of all cluster names in order matching action indices
        """
        self.cluster_names = cluster_names
    
    def get_mask_fn(self, action: np.ndarray) -> Callable[[int], torch.Tensor]:
        """
        Create a mask_fn from binary action vector.
        
        Args:
            action: Binary array of shape (num_clusters,) where 1 = prune
            
        Returns:
            mask_fn(layer_idx) -> torch.Tensor of shape (intermediate_size,) 
                                  where True = keep, False = prune
        """
        if action.shape[0] != len(self.cluster_names):
            raise ValueError("Action length does not match number of clusters.")
        
        clusters_to_prune = [
            self.cluster_names[i] 
            for i in range(len(action)) 
            if action[i] == 1
        ]

        def mask_fn(layer_idx: int) -> torch.Tensor:
            layer_str = str(layer_idx)
            mask = get_pruning_mask(clusters_to_prune, layer_str)
            return torch.from_numpy(mask).bool()
        
        return mask_fn
