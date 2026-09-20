"""Strict manifest-driven Seen-10 dataset adapter.

The benchmark protocol implementation is imported from the shared evaluator;
this module does not rediscover samples or create splits from the image tree.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_MAPS = (
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
_TRAIN_SPLITS = {"seen_train", "seen_validation"}
_GENERATION_SPLITS = {"seen_discrete_test", "seen_continuous"}


@lru_cache(maxsize=4)
def _benchmark_reader(data_root: str, shared_eval_dir: str):
    protocol_path = Path(shared_eval_dir) / "protocol.py"
    if not protocol_path.is_file():
        raise FileNotFoundError(
            "The shared evaluator protocol.py is required for manifest-driven reads: "
            f"{protocol_path}"
        )
    spec = importlib.util.spec_from_file_location("lumina_seen10_shared_protocol", protocol_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the shared benchmark protocol at {protocol_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BenchmarkData(data_root)


def benchmark_data(data_root: str | Path, shared_eval_dir: str | Path):
    """Return the canonical reader from the shared, model-independent evaluator."""

    return _benchmark_reader(str(Path(data_root).expanduser().resolve()), str(Path(shared_eval_dir).expanduser().resolve()))


def _image_tensor(path: str | Path, size: int, *, normalise: bool) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div_(255.0)
    if normalise:
        tensor.mul_(2.0).sub_(1.0)
    return tensor


class Seen10Dataset(Dataset):
    """Rows are read from published manifest/splits in fixed Seen-10 order.

    ``load_targets`` can only be enabled for training and validation. Test
    generation instances contain no target image tensor and never call
    ``Image.open`` on an FPV path.
    """

    def __init__(
        self,
        data_root: str | Path,
        shared_eval_dir: str | Path,
        split: str,
        *,
        load_targets: bool | None = None,
        image_size: int = 448,
        radar_size: int = 224,
        max_samples: int | None = None,
        max_clips: int | None = None,
    ):
        if split not in _TRAIN_SPLITS | _GENERATION_SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        if image_size <= 0 or radar_size <= 0:
            raise ValueError("image_size and radar_size must be positive")

        is_training_split = split in _TRAIN_SPLITS
        if load_targets is None:
            load_targets = is_training_split
        if load_targets and not is_training_split:
            raise ValueError("Ground-truth FPV images may only be loaded for train/validation")
        if not load_targets and is_training_split:
            raise ValueError("Training/validation datasets must load their FPV targets")

        self.split = split
        self.load_targets = bool(load_targets)
        self.image_size = int(image_size)
        self.radar_size = int(radar_size)
        self.map_to_id = {name: index for index, name in enumerate(_MAPS)}
        self.reader = benchmark_data(data_root, shared_eval_dir)
        self.clips: list[dict[str, Any]] = []
        if split == "seen_continuous":
            self.clips = self.reader.clips("seen_continuous", max_clips=max_clips)
            rows = [row for clip in self.clips for row in clip["rows"]]
            self.rows = rows if max_samples is None else rows[:max_samples]
        else:
            self.rows = self.reader.rows(split, max_samples=max_samples)
        self._radar_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _radar(self, path: str) -> torch.Tensor:
        if path not in self._radar_cache:
            self._radar_cache[path] = _image_tensor(path, self.radar_size, normalise=True)
        return self._radar_cache[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "file_frame": row["file_frame"],
            "map_name": row["map_name"],
            "map_id": torch.tensor(self.map_to_id[row["map_name"]], dtype=torch.long),
            "pose": torch.tensor(row["pose"], dtype=torch.float32),
            "radar": self._radar(row["radar_path"]),
            "clip_id": row.get("clip_id") or "",
            "frame_index": -1 if row.get("frame_index") is None else int(row["frame_index"]),
        }
        if self.load_targets:
            item["target"] = _image_tensor(row["image_path"], self.image_size, normalise=True)
        return item


def collate_seen10(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty Seen-10 batch")
    result: dict[str, Any] = {
        "radar": torch.stack([item["radar"] for item in batch]),
        "pose": torch.stack([item["pose"] for item in batch]),
        "map_id": torch.stack([item["map_id"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "file_frame": [item["file_frame"] for item in batch],
        "map_name": [item["map_name"] for item in batch],
        "clip_id": [item["clip_id"] for item in batch],
        "frame_index": [item["frame_index"] for item in batch],
    }
    if "target" in batch[0]:
        result["target"] = torch.stack([item["target"] for item in batch])
    return result


__all__ = ["Seen10Dataset", "benchmark_data", "collate_seen10"]
