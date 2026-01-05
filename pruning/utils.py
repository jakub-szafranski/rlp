from __future__ import annotations

from typing import Callable, Dict, Literal, Optional
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
    dtype_gate: str
    dtype_up: str
    dtype_down: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UndoPack(BaseModel):
    storage: Literal["cpu", "gpu"] = "cpu"
    dims: Dims
    layers: Dict[int, LayerUndoEntry] = Field(default_factory=dict)
    sparsity: float = 0.0  # Fraction of neurons pruned (0-1)

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _cpu_int(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", torch.int32).contiguous()

def _dtype_from_str(s: str) -> torch.dtype:
    return getattr(torch, s.replace("torch.", ""))


@torch.no_grad()
def prune_with_undo_pack(
    model: PreTrainedModel,
    mask_fn: Callable[[int], torch.Tensor],
    storage: Literal["cpu", "gpu"] = "cpu",
    cast_dtype: torch.dtype = torch.float16,
) -> UndoPack:
    if storage not in {"gpu", "cpu"}:
        raise ValueError("storage must be 'cpu' | 'gpu'")

    mlp0 = model.model.layers[0].mlp
    hidden = int(mlp0.gate_proj.in_features)
    inter = int(mlp0.gate_proj.out_features)
    pack = UndoPack(storage=storage, dims=Dims(hidden=hidden, inter=inter))

    pin_ok = torch.cuda.is_available()
    
    # Track sparsity
    total_neurons = 0
    pruned_neurons = 0

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        keep = mask_fn(layer_idx).bool()
        remove = ~keep
        
        # Accumulate for sparsity
        total_neurons += len(keep)
        pruned_neurons += int(remove.sum().item())

        gate_w = mlp.gate_proj.weight.data
        up_w   = mlp.up_proj.weight.data
        down_w = mlp.down_proj.weight.data

        # Prune
        w_gate = gate_w[keep, :].contiguous()
        w_up   = up_w[keep,   :].contiguous()
        w_down = down_w[:, keep].contiguous()

        mlp.gate_proj.weight.data = w_gate
        mlp.up_proj.weight.data   = w_up
        mlp.down_proj.weight.data = w_down

        new_size = int(keep.sum().item())
        mlp.gate_proj.out_features = new_size
        mlp.up_proj.out_features   = new_size
        mlp.down_proj.in_features  = new_size

        # Removed slices
        gate_removed = gate_w[remove, :].contiguous()
        up_removed   = up_w[remove,   :].contiguous()
        down_removed = down_w[:, remove].contiguous()

        keep_idx = _cpu_int(torch.nonzero(keep).view(-1))
        rem_idx  = _cpu_int(torch.nonzero(remove).view(-1))

        if storage == "gpu":
            entry = LayerUndoEntry(
                keep_idx=keep_idx, rem_idx=rem_idx,
                gate_removed=gate_removed.to(gate_w.device, dtype=cast_dtype),
                up_removed=up_removed.to(up_w.device, dtype=cast_dtype),
                down_removed=down_removed.to(down_w.device, dtype=cast_dtype),
                dtype_gate=str(gate_w.dtype),
                dtype_up=str(up_w.dtype),
                dtype_down=str(down_w.dtype),
            )
        else:
            gr = gate_removed.detach().to(device="cpu", dtype=cast_dtype).contiguous()
            ur = up_removed.detach().to(device="cpu", dtype=cast_dtype).contiguous()
            dr = down_removed.detach().to(device="cpu", dtype=cast_dtype).contiguous()
            if pin_ok:
                gr = gr.pin_memory(); ur = ur.pin_memory(); dr = dr.pin_memory()
            entry = LayerUndoEntry(
                keep_idx=keep_idx, rem_idx=rem_idx,
                gate_removed=gr, up_removed=ur, down_removed=dr,
                dtype_gate=str(gate_w.dtype),
                dtype_up=str(up_w.dtype),
                dtype_down=str(down_w.dtype),
            )

        pack.layers[layer_idx] = entry
    
    # Compute sparsity ratio
    pack.sparsity = pruned_neurons / total_neurons if total_neurons > 0 else 0.0

    return pack


@torch.no_grad()
def unprune_from_undo_pack(
    model: PreTrainedModel,
    pack: UndoPack,
    device: Optional[torch.device] = None
) -> None:
    hidden = int(pack.dims.hidden)
    inter  = int(pack.dims.inter)

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        dev = device or mlp.gate_proj.weight.device

        entry = pack.layers[layer_idx]
        keep_idx = entry.keep_idx.to(torch.long, device=dev)
        rem_idx  = entry.rem_idx.to(torch.long, device=dev)

        gate_dtype = _dtype_from_str(entry.dtype_gate)
        up_dtype   = _dtype_from_str(entry.dtype_up)
        down_dtype = _dtype_from_str(entry.dtype_down)

        # Current (pruned) weights – already on GPU
        gate_cur = mlp.gate_proj.weight.data
        up_cur   = mlp.up_proj.weight.data
        down_cur = mlp.down_proj.weight.data

        # Removed slices are on GPU already if storage="gpu"
        gate_removed = entry.gate_removed.to(dev, gate_dtype, non_blocking=True)
        up_removed   = entry.up_removed.to(dev,   up_dtype,   non_blocking=True)
        down_removed = entry.down_removed.to(dev, down_dtype, non_blocking=True)

        # Allocate full tensors ONCE per layer
        full_gate = torch.empty((inter, hidden), dtype=gate_dtype, device=dev)
        full_up   = torch.empty((inter, hidden), dtype=up_dtype,   device=dev)
        full_down = torch.empty((hidden, inter), dtype=down_dtype, device=dev)

        # Fill kept rows/cols from current (pruned) weights
        full_gate[keep_idx, :] = gate_cur
        full_up[keep_idx,   :] = up_cur
        full_down[:, keep_idx] = down_cur

        # Fill removed rows/cols from undo pack
        full_gate[rem_idx, :] = gate_removed
        full_up[rem_idx,   :] = up_removed
        full_down[:, rem_idx] = down_removed

        # Reuse Parameter objects
        mlp.gate_proj.weight.data = full_gate
        mlp.gate_proj.out_features = inter
        mlp.gate_proj.in_features = hidden

        mlp.up_proj.weight.data = full_up
        mlp.up_proj.out_features = inter
        mlp.up_proj.in_features = hidden

        mlp.down_proj.weight.data = full_down
        mlp.down_proj.in_features = inter
        mlp.down_proj.out_features = hidden

        # Drop **this layer's** undo tensors so they no longer pin GPU memory
        entry.gate_removed = torch.empty(0, device="cpu")
        entry.up_removed   = torch.empty(0, device="cpu")
        entry.down_removed = torch.empty(0, device="cpu")
        entry.keep_idx = torch.empty(0, dtype=torch.int32, device="cpu")
        entry.rem_idx  = torch.empty(0, dtype=torch.int32, device="cpu")

        # Remove entry from pack so Python GC can release it
        pack.layers.pop(layer_idx, None)

        # Delete references in this scope
        del gate_cur, up_cur, down_cur
        del gate_removed, up_removed, down_removed
        del full_gate, full_up, full_down
        del keep_idx, rem_idx

    # After all layers, drop the pack object itself
    del pack
