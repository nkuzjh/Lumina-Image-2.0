"""Radar, numeric-pose and map/task conditioning for the native Lumina NextDiT."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


SEEN_MAPS = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)

OFFICIAL_STATE_DICT_BYTES = 10_439_286_770


class RadarPoseMapAdapter(nn.Module):
    """Encode all non-text conditions as a short caption-feature sequence."""

    def __init__(
        self,
        cap_feat_dim: int = 2304,
        radar_grid: tuple[int, int] = (4, 4),
        fourier_frequencies: int = 8,
        task_count: int = 1,
    ):
        super().__init__()
        if cap_feat_dim <= 0 or min(radar_grid) <= 0 or fourier_frequencies <= 0:
            raise ValueError("Adapter dimensions must be positive")
        self.cap_feat_dim = int(cap_feat_dim)
        self.radar_grid = tuple(int(x) for x in radar_grid)
        self.fourier_frequencies = int(fourier_frequencies)
        radar_tokens = self.radar_grid[0] * self.radar_grid[1]
        if radar_tokens + 5 + 1 + task_count >= 300:
            raise ValueError("Condition token count must remain strictly below NextDiT cap RoPE limit 300")

        self.radar_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(12, 192),
            nn.SiLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(self.radar_grid),
            nn.Conv2d(256, self.cap_feat_dim, kernel_size=1),
        )
        self.radar_positional_embedding = nn.Parameter(torch.empty(1, radar_tokens, self.cap_feat_dim))
        nn.init.normal_(self.radar_positional_embedding, std=0.02)

        fourier_dim = self.fourier_frequencies * 2
        self.pose_mlp = nn.Sequential(
            nn.Linear(fourier_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.cap_feat_dim),
        )
        self.map_embedding = nn.Embedding(len(SEEN_MAPS), self.cap_feat_dim)
        self.task_embedding = nn.Embedding(task_count, self.cap_feat_dim)
        nn.init.normal_(self.map_embedding.weight, std=0.02)
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(self.fourier_frequencies, dtype=torch.float32),
            persistent=False,
        )

    @property
    def token_count(self) -> int:
        return self.radar_grid[0] * self.radar_grid[1] + 5 + 2

    def forward(
        self,
        radar: torch.Tensor,
        pose: torch.Tensor,
        map_id: torch.Tensor,
        task_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if radar.ndim != 4 or radar.shape[1] != 3:
            raise ValueError(f"Expected radar tensor [B,3,H,W], got {tuple(radar.shape)}")
        if pose.ndim != 2 or pose.shape[1] != 5:
            raise ValueError(f"Expected normalized [x,y,z,pitch,yaw] pose [B,5], got {tuple(pose.shape)}")
        batch = radar.shape[0]
        if pose.shape[0] != batch or map_id.shape != (batch,):
            raise ValueError("Radar, pose and map identity batch dimensions do not match")
        if task_id is None:
            task_id = torch.zeros(batch, dtype=torch.long, device=radar.device)
        if task_id.shape != (batch,):
            raise ValueError("task_id must have one integer per sample")

        radar_features = self.radar_encoder(radar)
        radar_tokens = radar_features.flatten(2).transpose(1, 2)
        radar_tokens = radar_tokens + self.radar_positional_embedding.to(radar_tokens.dtype)

        angles = 2.0 * math.pi * pose.float().unsqueeze(-1) * self.frequencies.float()
        pose_fourier = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        pose_tokens = self.pose_mlp(pose_fourier.to(self.pose_mlp[0].weight.dtype))

        map_tokens = self.map_embedding(map_id).unsqueeze(1)
        task_tokens = self.task_embedding(task_id).unsqueeze(1)
        dtype = radar_tokens.dtype
        cap_feats = torch.cat(
            (
                radar_tokens,
                pose_tokens.to(dtype),
                map_tokens.to(dtype),
                task_tokens.to(dtype),
            ),
            dim=1,
        )
        if cap_feats.shape[1] >= 300:
            raise RuntimeError(f"Condition sequence exceeds NextDiT cap RoPE limit: {cap_feats.shape[1]}")
        # Every valid context token occupies a continuous true prefix.
        cap_mask = torch.ones(batch, cap_feats.shape[1], dtype=torch.bool, device=cap_feats.device)
        return cap_feats, cap_mask


class LuminaSeen10Model(nn.Module):
    """Frozen native NextDiT plus trainable condition adapters."""

    def __init__(self, dit: nn.Module, adapter: RadarPoseMapAdapter):
        super().__init__()
        self.dit = dit
        self.adapter = adapter
        for parameter in self.dit.parameters():
            parameter.requires_grad_(False)
        self.dit.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The base remains deterministic and frozen while adapter modules train.
        self.dit.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        radar: torch.Tensor,
        pose: torch.Tensor,
        map_id: torch.Tensor,
        task_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cap_feats, cap_mask = self.adapter(radar, pose, map_id, task_id)
        # Do not wrap the frozen DiT in no_grad: gradients must pass through it
        # to the radar and pose adapters.
        return self.dit(x, t, cap_feats=cap_feats, cap_mask=cap_mask)


def load_native_dit(
    base_model_dir: str | Path,
    *,
    model_name: str = "NextDiT_2B_GQA_patch2_Adaln_Refiner",
    configured_cap_feat_dim: int = 2304,
    qk_norm: bool = True,
) -> tuple[nn.Module, int]:
    """Strictly load the official pure NextDiT state_dict and infer its cap width."""

    base_dir = Path(base_model_dir).expanduser().resolve()
    if base_dir.is_file():
        checkpoint_path = base_dir
        base_dir = base_dir.parent
    else:
        checkpoint_path = base_dir / "consolidated.00-of-01.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Official Lumina state_dict not found: {checkpoint_path}")
    aria_marker = checkpoint_path.with_name(checkpoint_path.name + ".aria2")
    if aria_marker.exists():
        raise RuntimeError(f"Official Lumina state_dict is still downloading: {aria_marker}")
    actual_size = checkpoint_path.stat().st_size
    if actual_size != OFFICIAL_STATE_DICT_BYTES:
        raise ValueError(
            f"Official Lumina state_dict has size {actual_size}, expected {OFFICIAL_STATE_DICT_BYTES}: "
            f"{checkpoint_path}"
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Expected a pure NextDiT state_dict at {checkpoint_path}")
    cap_weight = state_dict.get("cap_embedder.0.weight")
    if not isinstance(cap_weight, torch.Tensor) or cap_weight.ndim != 1:
        raise ValueError("Official state_dict has no recognizable cap_embedder.0.weight")
    cap_feat_dim = int(cap_weight.shape[0])
    if configured_cap_feat_dim and cap_feat_dim != int(configured_cap_feat_dim):
        raise ValueError(
            f"Configured cap_feat_dim={configured_cap_feat_dim} disagrees with official state_dict width={cap_feat_dim}"
        )

    args_path = base_dir / "model_args.pth"
    if args_path.is_file():
        saved_args = torch.load(args_path, map_location="cpu", weights_only=False)
        saved_model = getattr(saved_args, "model", model_name)
        if saved_model != model_name:
            raise ValueError(f"Configured model {model_name!r} differs from model_args.pth ({saved_model!r})")
        qk_norm = bool(getattr(saved_args, "qk_norm", qk_norm))

    # Import here so pure adapter/data CPU checks do not require FlashAttention.
    import models

    try:
        model_factory = getattr(models, model_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown native Lumina model factory: {model_name}") from exc
    dit = model_factory(in_channels=16, qk_norm=qk_norm, cap_feat_dim=cap_feat_dim)
    # The official checkpoint is a plain model state_dict: key and shape mismatches fail.
    dit.load_state_dict(state_dict, strict=True)
    del state_dict
    return dit, cap_feat_dim


__all__ = ["RadarPoseMapAdapter", "LuminaSeen10Model", "load_native_dit", "SEEN_MAPS"]
