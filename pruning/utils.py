from __future__ import annotations

from typing import Callable, Dict, Literal, Optional, Tuple
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, ConfigDict
from transformers import PreTrainedModel


class Dims(BaseModel):
    hidden: int
    inter: int


class LayerUndoEntry(BaseModel):
    keep_idx: torch.Tensor
    rem_idx: torch.Tensor
    gate_removed: torch.Tensor
    up_removed: torch.Tensor
    down_removed: torch.Tensor
    bias_delta: torch.Tensor 
    dtype_gate: str
    dtype_up: str
    dtype_down: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UndoPack(BaseModel):
    storage: Literal["gpu"] = "gpu"
    dims: Dims
    layers: Dict[int, LayerUndoEntry] = Field(default_factory=dict)
    sparsity: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _dtype_from_str(s: str) -> torch.dtype:
    return getattr(torch, s.replace("torch.", ""))


@torch.no_grad()
def prune_with_undo_pack(
    model: PreTrainedModel,
    mask_fn: Callable[[int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cast_dtype: torch.dtype = torch.float16,
) -> UndoPack:
    """
    Prune MLP neurons with FLAP bias compensation (improved version).
    """
    mlp0 = model.model.layers[0].mlp
    hidden = int(mlp0.gate_proj.in_features)
    inter = int(mlp0.gate_proj.out_features)
    pack = UndoPack(storage="gpu", dims=Dims(hidden=hidden, inter=inter))

    total_model_params = sum(int(p.numel()) for p in model.parameters())
    pruned_params = 0

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        device = mlp.gate_proj.weight.device
        
        # Get mask and FLAP data
        keep, prune_indices, prune_means = mask_fn(layer_idx)
        remove = ~keep
        
        gate_w = mlp.gate_proj.weight.data
        up_w = mlp.up_proj.weight.data
        down_w = mlp.down_proj.weight.data
        weight_dtype = down_w.dtype

        if prune_indices.numel() > 0:
            pruned_weights_fp32 = down_w[:, prune_indices].float()
            means_fp32 = prune_means.float()
            bias_delta_fp32 = torch.matmul(pruned_weights_fp32, means_fp32)
        else:
            bias_delta_fp32 = torch.zeros(down_w.shape[0], device=device, dtype=torch.float32)

        bias_delta = bias_delta_fp32.to(weight_dtype)
        if mlp.down_proj.bias is None:
            mlp.down_proj.bias = torch.nn.Parameter(bias_delta.clone())
        else:
            mlp.down_proj.bias.data.add_(bias_delta)

        # --- Physical Pruning ---
        w_gate = gate_w[keep, :].contiguous()
        w_up = up_w[keep, :].contiguous()
        w_down = down_w[:, keep].contiguous()

        mlp.gate_proj.weight.data = w_gate
        mlp.up_proj.weight.data = w_up
        mlp.down_proj.weight.data = w_down

        new_size = int(keep.sum().item())
        mlp.gate_proj.out_features = new_size
        mlp.up_proj.out_features = new_size
        mlp.down_proj.in_features = new_size

        # Store removed slices
        gate_removed = gate_w[remove, :].contiguous()
        up_removed = up_w[remove, :].contiguous()
        down_removed = down_w[:, remove].contiguous()

        pruned_params += int(gate_removed.numel() + up_removed.numel() + down_removed.numel())

        # Store indices directly on GPU as long (no CPU conversion)
        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        rem_idx = torch.nonzero(remove, as_tuple=False).squeeze(1)

        entry = LayerUndoEntry(
            keep_idx=keep_idx,
            rem_idx=rem_idx,
            gate_removed=gate_removed.to(dtype=cast_dtype),
            up_removed=up_removed.to(dtype=cast_dtype),
            down_removed=down_removed.to(dtype=cast_dtype),
            bias_delta=bias_delta.to(dtype=cast_dtype),  # store the applied delta
            dtype_gate=str(gate_w.dtype),
            dtype_up=str(up_w.dtype),
            dtype_down=str(down_w.dtype),
        )

        pack.layers[layer_idx] = entry

    pack.sparsity = pruned_params / total_model_params if total_model_params > 0 else 0.0

    return pack


@torch.no_grad()
def unprune_from_undo_pack(
    model: PreTrainedModel,
    pack: UndoPack,
    device: Optional[torch.device] = None
) -> None:
    hidden = int(pack.dims.hidden)
    inter = int(pack.dims.inter)

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        dev = device or mlp.gate_proj.weight.device

        entry = pack.layers[layer_idx]
        keep_idx = entry.keep_idx.to(device=dev)
        rem_idx = entry.rem_idx.to(device=dev)

        gate_dtype = _dtype_from_str(entry.dtype_gate)
        up_dtype = _dtype_from_str(entry.dtype_up)
        down_dtype = _dtype_from_str(entry.dtype_down)

        # Current (pruned) weights
        gate_cur = mlp.gate_proj.weight.data
        up_cur = mlp.up_proj.weight.data
        down_cur = mlp.down_proj.weight.data

        # Removed slices
        gate_removed = entry.gate_removed.to(dev, gate_dtype, non_blocking=True)
        up_removed = entry.up_removed.to(dev, up_dtype, non_blocking=True)
        down_removed = entry.down_removed.to(dev, down_dtype, non_blocking=True)

        # Allocate full tensors
        full_gate = torch.empty((inter, hidden), dtype=gate_dtype, device=dev)
        full_up = torch.empty((inter, hidden), dtype=up_dtype, device=dev)
        full_down = torch.empty((hidden, inter), dtype=down_dtype, device=dev)

        # Restore weights
        full_gate[keep_idx, :] = gate_cur
        full_up[keep_idx, :] = up_cur
        full_down[:, keep_idx] = down_cur

        full_gate[rem_idx, :] = gate_removed
        full_up[rem_idx, :] = up_removed
        full_down[:, rem_idx] = down_removed

        mlp.gate_proj.weight.data = full_gate
        mlp.gate_proj.out_features = inter
        mlp.gate_proj.in_features = hidden

        mlp.up_proj.weight.data = full_up
        mlp.up_proj.out_features = inter
        mlp.up_proj.in_features = hidden

        mlp.down_proj.weight.data = full_down
        mlp.down_proj.in_features = inter
        mlp.down_proj.out_features = hidden

        # --- Reverse FLAP bias compensation ---
        if mlp.down_proj.bias is not None:
            bias_delta = entry.bias_delta.to(dev, mlp.down_proj.bias.dtype, non_blocking=True)
            mlp.down_proj.bias.data.sub_(bias_delta)
            
            if torch.allclose(mlp.down_proj.bias.data, torch.zeros_like(mlp.down_proj.bias.data), atol=1e-5):
                mlp.down_proj.bias = None

        entry.gate_removed = torch.empty(0, device="cpu")
        entry.up_removed = torch.empty(0, device="cpu")
        entry.down_removed = torch.empty(0, device="cpu")
        entry.bias_delta = torch.empty(0, device="cpu")
        entry.keep_idx = torch.empty(0, dtype=torch.long, device="cpu")
        entry.rem_idx = torch.empty(0, dtype=torch.long, device="cpu")

        pack.layers.pop(layer_idx, None)

        del gate_cur, up_cur, down_cur
        del gate_removed, up_removed, down_removed
        del full_gate, full_up, full_down
        del keep_idx, rem_idx

    del pack