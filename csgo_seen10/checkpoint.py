"""Atomic adapter checkpoints, resume state and late/best checkpoint links."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


FORMAT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_world() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if _distributed() else (0, 1)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _update_link(checkpoint_dir: Path, name: str, target: Path) -> None:
    link = checkpoint_dir / name
    temporary = checkpoint_dir / f".{name}.{os.getpid()}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.name)
    os.replace(temporary, link)


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int,
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_validation_loss: float,
    validation_loss: float,
    config: dict[str, Any],
    is_best: bool,
) -> Path:
    """Save a rank-synchronous optimizer/RNG checkpoint and update links on rank zero."""

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"step_{step:08d}.pt"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")

    local_rng = capture_rng_state()
    rank, world_size = _rank_world()
    if world_size > 1:
        rng_by_rank: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rng_by_rank, local_rng)
    else:
        rng_by_rank = [local_rng]

    if rank == 0:
        adapter_state = {
            name: value.detach().cpu()
            for name, value in adapter.state_dict().items()
        }
        payload = {
            "format_version": FORMAT_VERSION,
            "step": int(step),
            "adapter": adapter_state,
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": float(best_validation_loss),
            "validation_loss": float(validation_loss),
            "rng_by_rank": rng_by_rank,
            "world_size": world_size,
            "config": config,
        }
        _atomic_torch_save(payload, target)
        _update_link(directory, "late.pt", target)
        if is_best:
            _update_link(directory, "best.pt", target)
    if _distributed():
        dist.barrier()
    return target


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported Seen-10 checkpoint format: {path}")
    adapter.load_state_dict(payload["adapter"], strict=True)
    if optimizer is not None:
        if "optimizer" not in payload:
            raise ValueError(f"Checkpoint has no optimizer state: {path}")
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        rng_states = payload.get("rng_by_rank", [])
        rank, world_size = _rank_world()
        if int(payload.get("world_size", 1)) != world_size or len(rng_states) != world_size:
            raise ValueError(
                f"Checkpoint world size {payload.get('world_size')} cannot resume with world size {world_size}"
            )
        restore_rng_state(rng_states[rank])
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["capture_rng_state", "restore_rng_state", "save_checkpoint", "load_checkpoint", "sha256_file"]
