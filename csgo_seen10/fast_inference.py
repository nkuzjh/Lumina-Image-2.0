"""Fixed-shape NextDiT inference helpers for CSGO Seen-10 generation.

The optimized route is intentionally inference-only. It specializes the native
NextDiT at 448px, 23 condition tokens and 784 image tokens. Training and the
legacy eager inference route continue to use the upstream variable-length
forward implementation.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


IMAGE_SIZE = 448
PATCH_SIZE = 2
CAP_TOKEN_COUNT = 23
IMAGE_TOKEN_COUNT = (IMAGE_SIZE // 16) ** 2
FULL_TOKEN_COUNT = CAP_TOKEN_COUNT + IMAGE_TOKEN_COUNT
SEED_POLICY = "sha256(seed\\0task\\0sample_id) mod (2^63-1)"
COMPILE_MODE = "reduce-overhead/fullgraph"
COMPILE_MODES = ("reduce-overhead", "default")
_FLASH_ATTN_VARLEN_FUNC = None


def compile_mode_label(mode: str) -> str:
    """Return the stable run-metadata label for a fixed-shape compile mode."""

    if mode not in COMPILE_MODES:
        raise ValueError(f"Unsupported compile mode {mode!r}; expected one of {COMPILE_MODES}")
    return f"{mode}/fullgraph"


def compile_mode_uses_cudagraphs(mode: str) -> bool:
    if mode not in COMPILE_MODES:
        raise ValueError(f"Unsupported compile mode {mode!r}; expected one of {COMPILE_MODES}")
    return mode == "reduce-overhead"


def torch_compile_kwargs(mode: str) -> dict[str, Any]:
    """Map CLI modes to explicit Inductor settings without requiring a GPU."""

    if mode not in COMPILE_MODES:
        raise ValueError(f"Unsupported compile mode {mode!r}; expected one of {COMPILE_MODES}")
    if mode == "default":
        # torch.compile rejects mode and options together. Omitting mode selects
        # Inductor's default mode while keeping CUDA Graphs explicitly disabled.
        return {"fullgraph": True, "options": {"triton.cudagraphs": False}}
    return {"mode": mode, "fullgraph": True}


def fixed_position_ids(
    image_size: int = IMAGE_SIZE,
    cap_tokens: int = CAP_TOKEN_COUNT,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Return the upstream NextDiT axis positions for one fixed-size sample."""

    if image_size != IMAGE_SIZE or patch_size != PATCH_SIZE or cap_tokens != CAP_TOKEN_COUNT:
        raise ValueError("The fixed Seen-10 forward supports only 448px, patch-2 and 23 condition tokens")
    side = image_size // 16
    image_tokens = side * side
    ids = torch.zeros((1, cap_tokens + image_tokens, 3), dtype=torch.int32)
    ids[0, :cap_tokens, 0] = torch.arange(cap_tokens, dtype=torch.int32)
    ids[0, cap_tokens:, 0] = cap_tokens
    row_ids = torch.arange(side, dtype=torch.int32).view(-1, 1).expand(side, side).reshape(-1)
    col_ids = torch.arange(side, dtype=torch.int32).view(1, -1).expand(side, side).reshape(-1)
    ids[0, cap_tokens:, 1] = row_ids
    ids[0, cap_tokens:, 2] = col_ids
    return ids


def fixed_sequence_cu_seqlens(batch_size: int, sequence_length: int, *, device: str | torch.device = "cpu"):
    """Fixed FlashAttention varlen boundaries with one equal-length segment per row."""

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    return torch.arange(batch_size + 1, dtype=torch.int32, device=device) * int(sequence_length)


def euler_time_grid(num_steps: int, time_shifting_factor: float, *, device: str | torch.device = "cpu"):
    """Match the repository's torchdiffeq Euler sampling grid and time shift."""

    if num_steps < 2:
        raise ValueError("Euler sampling requires at least two grid points")
    # The upstream ODE sampler builds and shifts this grid on CPU, then moves it
    # to the model device. Preserve that order and its float32 rounding.
    times = torch.linspace(0.0, 1.0, int(num_steps), dtype=torch.float32, device="cpu")
    if time_shifting_factor:
        factor = float(time_shifting_factor)
        times = times / (times + factor - factor * times)
    return times.to(device=device)


def stable_seed(seed: int, task: str, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{task}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def make_latent_batch(
    sample_ids: Iterable[str],
    *,
    seed: int,
    task: str,
    image_size: int = IMAGE_SIZE,
    channels: int = 16,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Create one independently seeded initial latent per manifest identity."""

    latents = []
    shape = (1, channels, image_size // 8, image_size // 8)
    for sample_id in sample_ids:
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(seed, task, str(sample_id)))
        latents.append(torch.randn(shape, generator=generator, device=device, dtype=dtype))
    if not latents:
        raise ValueError("Cannot create a latent batch without sample ids")
    return torch.cat(latents, dim=0)


def pad_collated_batch(batch: dict[str, Any], target_size: int) -> tuple[dict[str, Any], int]:
    """Pad a short DataLoader batch by repeating its final item for fixed shapes."""

    real_size = len(batch["sample_id"])
    if real_size <= 0 or target_size < real_size:
        raise ValueError(f"Invalid collated batch size {real_size} for target {target_size}")
    if real_size == target_size:
        return batch, real_size
    padding = target_size - real_size
    padded: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            expanded_shape = (padding,) + tuple(value.shape[1:])
            extra = value[-1:].expand(expanded_shape)
            padded[key] = torch.cat((value, extra), dim=0)
        elif isinstance(value, list):
            padded[key] = value + [value[-1]] * padding
        else:
            raise TypeError(f"Unsupported collated batch field {key}: {type(value).__name__}")
    return padded, real_size


def _fixed_flash_attention_forward(self, x, x_mask, freqs_cis):
    """JointAttention forward with precomputed equal-length FlashAttention segments."""

    if _FLASH_ATTN_VARLEN_FUNC is None:
        raise RuntimeError("Fixed FlashAttention kernel was not initialized before inference")

    batch_size, sequence_length, _ = x.shape
    if sequence_length != self._seen10_sequence_length:
        raise RuntimeError(
            f"Fixed NextDiT attention expected sequence {self._seen10_sequence_length}, got {sequence_length}"
        )
    if batch_size != self._seen10_batch_size:
        raise RuntimeError(f"Fixed NextDiT attention expected batch {self._seen10_batch_size}, got {batch_size}")

    query, key, value = torch.split(
        self.qkv(x),
        [
            self.n_local_heads * self.head_dim,
            self.n_local_kv_heads * self.head_dim,
            self.n_local_kv_heads * self.head_dim,
        ],
        dim=-1,
    )
    query = query.view(batch_size, sequence_length, self.n_local_heads, self.head_dim)
    key = key.view(batch_size, sequence_length, self.n_local_kv_heads, self.head_dim)
    value = value.view(batch_size, sequence_length, self.n_local_kv_heads, self.head_dim)
    query = self.q_norm(query)
    key = self.k_norm(key)
    query = self.apply_rotary_emb(query, freqs_cis=freqs_cis)
    key = self.apply_rotary_emb(key, freqs_cis=freqs_cis)
    query, key = query.contiguous(), key.contiguous()
    value = value.contiguous()

    output = _FLASH_ATTN_VARLEN_FUNC(
        query.reshape(batch_size * sequence_length, self.n_local_heads, self.head_dim),
        key.reshape(batch_size * sequence_length, self.n_local_kv_heads, self.head_dim),
        value.reshape(batch_size * sequence_length, self.n_local_kv_heads, self.head_dim),
        cu_seqlens_q=self._seen10_cu_seqlens,
        cu_seqlens_k=self._seen10_cu_seqlens,
        max_seqlen_q=self._seen10_sequence_length,
        max_seqlen_k=self._seen10_sequence_length,
        dropout_p=0.0,
        causal=False,
        softmax_scale=math.sqrt(1.0 / self.head_dim),
    )
    return self.out(output.reshape(batch_size, sequence_length, -1))


def install_fixed_attention(dit: nn.Module, batch_size: int, *, device: torch.device) -> None:
    """Replace dynamic unpadding attention with presegmented fixed-length kernels."""

    global _FLASH_ATTN_VARLEN_FUNC
    if _FLASH_ATTN_VARLEN_FUNC is None:
        from flash_attn import flash_attn_varlen_func

        _FLASH_ATTN_VARLEN_FUNC = flash_attn_varlen_func
    from models.model import JointAttention

    groups = (
        (dit.context_refiner, CAP_TOKEN_COUNT),
        (dit.noise_refiner, IMAGE_TOKEN_COUNT),
        (dit.layers, FULL_TOKEN_COUNT),
    )
    for blocks, sequence_length in groups:
        for block in blocks:
            attention = block.attention
            if not isinstance(attention, JointAttention):
                raise TypeError(f"Expected native JointAttention, got {type(attention).__name__}")
            cu_seqlens = fixed_sequence_cu_seqlens(batch_size, sequence_length, device=device)
            attention.register_buffer("_seen10_cu_seqlens", cu_seqlens, persistent=False)
            attention._seen10_sequence_length = int(sequence_length)
            attention._seen10_batch_size = int(batch_size)
            attention.forward = MethodType(_fixed_flash_attention_forward, attention)


class FixedNextDiTInference(nn.Module):
    """448px native NextDiT path with condition/context work cached per batch."""

    def __init__(self, model: nn.Module, *, batch_size: int, image_size: int = IMAGE_SIZE):
        super().__init__()
        if image_size != IMAGE_SIZE:
            raise ValueError(f"Fixed NextDiT supports image_size={IMAGE_SIZE}, got {image_size}")
        dit = model.dit
        if dit.patch_size != PATCH_SIZE or dit.in_channels != 16:
            raise ValueError("Fixed NextDiT requires the native patch-2, 16-channel Lumina model")
        if dit.dim != 2304 or len(dit.layers) != 26 or len(dit.context_refiner) != 2 or len(dit.noise_refiner) != 2:
            raise ValueError("Fixed NextDiT only supports NextDiT_2B_GQA_patch2_Adaln_Refiner")
        if model.adapter.token_count != CAP_TOKEN_COUNT:
            raise ValueError(f"Fixed NextDiT requires exactly {CAP_TOKEN_COUNT} condition tokens")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dit = dit
        self.adapter = model.adapter
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.patch_side = image_size // (8 * PATCH_SIZE)
        self.image_token_count = self.patch_side**2
        if self.image_token_count != IMAGE_TOKEN_COUNT:
            raise ValueError(f"Expected {IMAGE_TOKEN_COUNT} image tokens, got {self.image_token_count}")

        device = next(dit.parameters()).device
        ids = fixed_position_ids(image_size=image_size).to(device=device)
        # This is the exact cap/image axis layout built by native patchify:
        # cap axis0 = 0..22; image axis0 = 23, with 28x28 row/column ids.
        all_freqs = dit.rope_embedder(ids)
        self.register_buffer("cap_freqs", all_freqs[:, :CAP_TOKEN_COUNT].contiguous(), persistent=False)
        self.register_buffer("image_freqs", all_freqs[:, CAP_TOKEN_COUNT:].contiguous(), persistent=False)
        self.register_buffer("full_freqs", all_freqs.contiguous(), persistent=False)
        self.register_buffer("cap_mask", torch.ones((1, CAP_TOKEN_COUNT), dtype=torch.bool, device=device), persistent=False)
        self.register_buffer("image_mask", torch.ones((1, IMAGE_TOKEN_COUNT), dtype=torch.bool, device=device), persistent=False)
        self.register_buffer("full_mask", torch.ones((1, FULL_TOKEN_COUNT), dtype=torch.bool, device=device), persistent=False)
        install_fixed_attention(dit, self.batch_size, device=device)

    def prepare_condition(self, radar, pose, map_id, task_id):
        cap_features, _ = self.adapter(radar, pose, map_id, task_id)
        cap_features = self.dit.cap_embedder(cap_features)
        cap_freqs = self.cap_freqs.expand(self.batch_size, -1, -1)
        cap_mask = self.cap_mask.expand(self.batch_size, -1)
        for layer in self.dit.context_refiner:
            cap_features = layer(cap_features, cap_mask, cap_freqs)
        return cap_features

    def forward(self, latent: torch.Tensor, timesteps: torch.Tensor, cap_features: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        if batch_size != self.batch_size:
            raise RuntimeError(f"Fixed NextDiT expected batch {self.batch_size}, got {batch_size}")
        if latent.shape[1:] != (16, 56, 56):
            raise RuntimeError(f"Fixed NextDiT expected latent [B,16,56,56], got {tuple(latent.shape)}")

        adaln_input = self.dit.t_embedder(timesteps)
        channels = latent.shape[1]
        image = (
            latent.reshape(batch_size, channels, 28, PATCH_SIZE, 28, PATCH_SIZE)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(batch_size, IMAGE_TOKEN_COUNT, PATCH_SIZE * PATCH_SIZE * channels)
        )
        image = self.dit.x_embedder(image)
        image_freqs = self.image_freqs.expand(batch_size, -1, -1)
        image_mask = self.image_mask.expand(batch_size, -1)
        for layer in self.dit.noise_refiner:
            image = layer(image, image_mask, image_freqs, adaln_input)

        # Native patchify stores the concatenated sequence in the latent dtype;
        # torchdiffeq promotes the ODE state to float32 before its first call.
        full = torch.cat((cap_features.to(dtype=latent.dtype), image.to(dtype=latent.dtype)), dim=1)
        full_mask = self.full_mask.expand(batch_size, -1)
        full_freqs = self.full_freqs.expand(batch_size, -1, -1)
        for layer in self.dit.layers:
            full = layer(full, full_mask, full_freqs, adaln_input)
        prediction = self.dit.final_layer(full, adaln_input)[:, CAP_TOKEN_COUNT:]
        return (
            prediction.reshape(batch_size, self.patch_side, self.patch_side, PATCH_SIZE, PATCH_SIZE, 16)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch_size, 16, self.image_size // 8, self.image_size // 8)
        )


class FixedEulerStep(nn.Module):
    """One flow-velocity Euler update, compiled independently of the step loop."""

    def __init__(self, denoiser: FixedNextDiTInference):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, state: torch.Tensor, timesteps: torch.Tensor, dt: torch.Tensor, cap_features: torch.Tensor):
        velocity = self.denoiser(state, timesteps, cap_features)
        return state + velocity.float() * dt


def validate_fixed_sampler(training_config: dict[str, Any], inference_config: dict[str, Any]) -> None:
    if inference_config.get("sampler", "euler").lower() != "euler":
        raise ValueError("The fixed/compiled inference engine supports only the Euler sampler")
    if training_config.get("path_type", "Linear") != "Linear":
        raise ValueError("The fixed/compiled inference engine requires a Linear flow path")
    if training_config.get("prediction", "velocity") != "velocity":
        raise ValueError("The fixed/compiled inference engine requires velocity prediction")


def _valid_rgb_jpeg(path: Path, size: int) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.format == "JPEG" and image.mode == "RGB" and image.size == (size, size)
    except (OSError, ValueError):
        return False


def _atomic_write_jpeg(path: Path, pixels: np.ndarray, *, size: int, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels).convert("RGB")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.jpg", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="JPEG", quality=quality, optimize=True)
        if not _valid_rgb_jpeg(temporary, size):
            raise RuntimeError(f"Generated output did not validate as {size}x{size} RGB JPEG: {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AsyncJpegWriter:
    """Bounded asynchronous atomic JPEG writer that propagates worker errors."""

    def __init__(self, *, workers: int = 2, max_pending: int = 8, quality: int = 95):
        if workers <= 0 or max_pending <= 0:
            raise ValueError("JPEG workers and queue bound must be positive")
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="seen10-jpeg")
        self.max_pending = int(max_pending)
        self.quality = int(quality)
        self.pending: set[Future] = set()
        self.closed = False

    def _reap(self, *, block: bool) -> None:
        if not self.pending:
            return
        completed = {future for future in self.pending if future.done()}
        if not completed and block:
            completed = {next(iter(self.pending))}
        for future in completed:
            self.pending.remove(future)
            future.result()

    def submit(self, path: Path, pixels: np.ndarray, *, size: int) -> None:
        if self.closed:
            raise RuntimeError("JPEG writer is already closed")
        self._reap(block=False)
        if len(self.pending) >= self.max_pending:
            self._reap(block=True)
        self.pending.add(
            self.executor.submit(_atomic_write_jpeg, path, pixels, size=size, quality=self.quality)
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        first_error: BaseException | None = None
        while self.pending:
            try:
                self._reap(block=True)
            except BaseException as exc:  # finish other atomic writes, then propagate one failure
                if first_error is None:
                    first_error = exc
        self.executor.shutdown(wait=True)
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self.closed = True
            self.executor.shutdown(wait=True, cancel_futures=True)
        return False


__all__ = [
    "CAP_TOKEN_COUNT",
    "COMPILE_MODE",
    "FixedEulerStep",
    "FixedNextDiTInference",
    "IMAGE_TOKEN_COUNT",
    "SEED_POLICY",
    "AsyncJpegWriter",
    "euler_time_grid",
    "fixed_position_ids",
    "fixed_sequence_cu_seqlens",
    "make_latent_batch",
    "pad_collated_batch",
    "stable_seed",
    "validate_fixed_sampler",
]
