from typing import Callable, Literal, Optional
import torch
from transformers import PreTrainedModel

from .utils import UndoPack, prune_with_undo_pack, unprune_from_undo_pack


class PrunableLLM:
    """
    A proxy wrapper over a Hugging Face PreTrainedModel.
    - Keeps the full HF API (forward, generate, .config, etc.).
    - Adds prune(mask_fn, ...) and undo_prune().
    - Stores the UndoPack in self._undo_pack.
    - Exposes sparsity as a property after pruning.
    """
    def __init__(self, model: PreTrainedModel):
        self.model: PreTrainedModel = model
        self._undo_pack: Optional[UndoPack] = None
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    @property
    def sparsity(self) -> float:
        """Returns current sparsity ratio (0-1). 0 if not pruned."""
        if self._undo_pack is None:
            return 0.0
        return self._undo_pack.sparsity
    
    @property
    def is_pruned(self) -> bool:
        """Returns True if model is currently pruned."""
        return self._undo_pack is not None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __repr__(self):
        return f"PrunableLLM wrapping:\n{self.model.__repr__()}"

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @property
    def config(self):
        return self.model.config

    @torch.no_grad()
    def prune(
        self,
        mask_fn: Callable[[int], torch.Tensor],
        cast_dtype: torch.dtype = torch.float16,
    ) -> None:
        self._undo_pack = prune_with_undo_pack(
            self.model, mask_fn=mask_fn, cast_dtype=cast_dtype
        )

    @torch.no_grad()
    def undo_prune(self, device: Optional[torch.device] = None) -> None:
        if self._undo_pack is None:
            raise RuntimeError("No undo pack found. Call prune() first.")
        unprune_from_undo_pack(self.model, self._undo_pack, device=device)
        self._undo_pack = None
