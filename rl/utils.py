import numpy as np
from typing import Callable
import torch
from pruning import PrunableLLM, get_pruning_mask_and_means


class FractionMaskAdapter:
    """
    Adapts action (fraction vector) to mask_fn for PrunableLLM.
    
    Maps: action[layer_idx] = fraction to prune (0-1)
          -> mask_fn(layer_idx) returns boolean tensor (True = KEEP)
    """
    
    def __init__(self):
        pass
    
    def get_mask_fn(self, action: np.ndarray) -> Callable[[int], torch.Tensor]:
        """
        Create a mask_fn from fraction action vector.
        
        Args:
            action: Float array of shape (32,) with fractions 0-1
            
        Returns:
            mask_fn(layer_idx) -> torch.Tensor of shape (intermediate_size,) 
                                  where True = keep, False = prune
        """
        if action.shape[0] != 32:
            raise ValueError("Action must be length 32 for 32 layers.")

        def mask_fn(layer_idx: int) -> torch.Tensor:
            fraction = action[layer_idx]
            layer_str = str(layer_idx)
            mask, _ = get_pruning_mask_and_means(fraction, layer_str)
            return torch.from_numpy(mask).bool()
        
        return mask_fn
