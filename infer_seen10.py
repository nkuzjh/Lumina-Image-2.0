"""Manifest-driven discrete and continuous Seen-10 generation with frozen best adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from csgo_seen10.checkpoint import load_checkpoint, sha256_file
from csgo_seen10.data import Seen10Dataset
from csgo_seen10.model import LuminaSeen10Model, RadarPoseMapAdapter, load_native_dit
from train_seen10 import _load_vae, _precision_dtype


def _read_config(path: str | Path) -> dict:
    with Path(path).expanduser().open(encoding="utf-8") as source:
        return json.load(source)


def _distributed_context() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_jpeg(path: Path, size: int) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.format == "JPEG" and image.mode == "RGB" and image.size == (size, size)
    except (OSError, ValueError):
        return False


def _sample_seed(seed: int, task: str, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{task}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _identity_indices(dataset: Seen10Dataset, task: str, rank: int, world_size: int) -> list[int]:
    if task == "discrete":
        return list(range(rank, len(dataset), world_size))
    selected: list[int] = []
    offset = 0
    for clip_index, clip in enumerate(dataset.clips):
        clip_length = len(clip["rows"])
        if clip_index % world_size == rank:
            # Keep every clip's frames in exact manifest order on its worker.
            end = min(offset + clip_length, len(dataset))
            selected.extend(range(offset, end))
        offset += clip_length
    return selected


def _infer_one(
    row: dict,
    item: dict,
    task: str,
    model: LuminaSeen10Model,
    vae,
    sampler,
    *,
    device: torch.device,
    dtype: torch.dtype,
    config: dict,
    seed: int,
    gen_root: Path,
) -> None:
    size = int(config["inference"]["image_size"])
    output_path = gen_root / row["map_name"] / f"{row['file_frame']}.jpg"
    if output_path.exists() and _valid_jpeg(output_path, size):
        return

    # The dataset item contains only radar, normalized pose and map identity;
    # it never opens row["image_path"].
    radar = item["radar"].unsqueeze(0).to(device=device, non_blocking=True)
    pose = item["pose"].unsqueeze(0).to(device=device, non_blocking=True)
    map_id = item["map_id"].view(1).to(device=device, non_blocking=True)
    task_id = torch.zeros(1, dtype=torch.long, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(_sample_seed(seed, task, row["sample_id"]))
    latent = torch.randn(
        (1, 16, size // 8, size // 8),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    kwargs = {"radar": radar, "pose": pose, "map_id": map_id, "task_id": task_id}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        # cfg_scale=1 uses one conditioned NextDiT forward path; no text/null
        # branch is duplicated or fed the same condition as an unconditional one.
        samples = sampler(latent, model, **kwargs)[-1]
        decoded = vae.decode(samples / float(config["model"]["vae_scale"]) + float(config["model"]["vae_shift"])).sample
    image_tensor = decoded[0].float().add(1.0).div(2.0).clamp_(0.0, 1.0)
    pixels = image_tensor.mul(255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(pixels, mode="RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".tmp.jpg", dir=output_path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="JPEG", quality=int(config["inference"]["jpeg_quality"]), optimize=True)
        if not _valid_jpeg(temporary, size):
            raise RuntimeError(f"Generated output did not validate as {size}x{size} RGB JPEG: {temporary}")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_provenance(
    task_root: Path,
    task: str,
    rows: list[dict],
    *,
    size: int,
    seed: int,
    checkpoint_path: Path,
    checkpoint_sha: str,
) -> None:
    records = []
    gen_root = task_root / "gen_imgs"
    expected: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["map_name"], row["file_frame"])
        if key in expected:
            raise RuntimeError(f"Duplicate manifest identity while writing provenance: {key}")
        expected.add(key)
        image_path = gen_root / row["map_name"] / f"{row['file_frame']}.jpg"
        if not _valid_jpeg(image_path, size):
            raise RuntimeError(f"Missing or invalid generated image: {image_path}")
        record = {
            "task": task,
            "sample_id": row["sample_id"],
            "map_name": row["map_name"],
            "file_frame": row["file_frame"],
            "clip_id": row.get("clip_id"),
            "frame_index": row.get("frame_index"),
            "stable_seed": _sample_seed(seed, task, row["sample_id"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "width": size,
            "height": size,
            "mode": "RGB",
            "format": "JPEG",
        }
        records.append(record)

    actual: set[tuple[str, str]] = set()
    if gen_root.exists():
        for path in gen_root.rglob("*.jpg"):
            actual.add((path.parent.name, path.stem))
    if actual != expected:
        missing, extra = sorted(expected - actual)[:5], sorted(actual - expected)[:5]
        raise RuntimeError(f"Generated image coverage differs from manifest; missing={missing}, extra={extra}")

    provenance = task_root / "provenance.jsonl"
    if provenance.exists():
        existing = [json.loads(line) for line in provenance.read_text(encoding="utf-8").splitlines() if line.strip()]
        old = {(record.get("map_name"), record.get("file_frame"), record.get("checkpoint_sha256")) for record in existing}
        new = {(record["map_name"], record["file_frame"], record["checkpoint_sha256"]) for record in records}
        if len(existing) != len(records) or old != new:
            raise FileExistsError(f"Existing provenance differs from the selected checkpoint/manifest: {provenance}")
        return

    task_root.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".provenance.", suffix=".tmp", dir=task_root)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, provenance)
    finally:
        temporary.unlink(missing_ok=True)


def _task_rows(config: dict, task: str, *, smoke: bool) -> Seen10Dataset:
    split = "seen_discrete_test" if task == "discrete" else "seen_continuous"
    return Seen10Dataset(
        config["data_root"],
        config["shared_eval_dir"],
        split,
        load_targets=False,
        image_size=int(config["inference"]["image_size"]),
        radar_size=int(config["adapter"]["radar_size"]),
        max_samples=1 if smoke else None,
        max_clips=1 if smoke and task == "continuous" else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/csgo_seen10.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--task", choices=("all", "discrete", "continuous"), default="all")
    parser.add_argument("--smoke", action="store_true", help="Generate one discrete and one continuous image")
    parser.add_argument("--checkpoint", default=None, help="Default is this seed's best.pt symlink")
    args = parser.parse_args()

    config = _read_config(args.config)
    seed = int(config["inference"].get("seed", 0) if args.seed is None else args.seed)
    if float(config["inference"].get("cfg_scale", 1.0)) != 1.0:
        raise ValueError("Seen-10 uses direct native NextDiT conditioning; cfg_scale must remain 1.0")

    rank, world_size, local_rank, initialized_here = _distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("Lumina Seen-10 inference requires a CUDA GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = _precision_dtype(config["model"].get("precision", "bf16"))

    seed_root = Path(config["output_root"]).expanduser().resolve() / f"seed_{seed}"
    run_root = seed_root / "smoke" if args.smoke else seed_root
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else run_root / "checkpoints" / "best.pt"
    )
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected adapter checkpoint not found: {checkpoint_path}")
    checkpoint_sha = sha256_file(checkpoint_path)

    dit, cap_feat_dim = load_native_dit(
        config["model"]["base_model_dir"],
        model_name=config["model"]["name"],
        configured_cap_feat_dim=int(config["model"]["cap_feat_dim"]),
        qk_norm=bool(config["model"].get("qk_norm", True)),
    )
    dit.to(device=device, dtype=dtype)
    adapter_config = config["adapter"]
    adapter = RadarPoseMapAdapter(
        cap_feat_dim=cap_feat_dim,
        radar_grid=tuple(adapter_config["radar_grid"]),
        fourier_frequencies=int(adapter_config["fourier_frequencies"]),
        task_count=int(adapter_config["task_count"]),
    )
    model = LuminaSeen10Model(dit, adapter).to(device).eval()
    load_checkpoint(checkpoint_path, adapter=model.adapter)
    vae, _, _ = _load_vae(config, dtype, device)

    from transport import Sampler, create_transport

    transport_config = config["training"]
    transport = create_transport(
        transport_config.get("path_type", "Linear"),
        transport_config.get("prediction", "velocity"),
        transport_config.get("loss_weight"),
        snr_type=transport_config.get("snr_type", "uniform"),
        do_shift=bool(transport_config.get("do_shift", True)),
        seq_len=(int(config["inference"]["image_size"]) // 16) ** 2,
    )
    sampling_steps = 2 if args.smoke else int(config["inference"]["sampling_steps"])
    sampler = Sampler(transport).sample_ode(
        sampling_method=config["inference"].get("sampler", "euler"),
        num_steps=sampling_steps,
        atol=1e-6,
        rtol=1e-3,
        time_shifting_factor=float(config["inference"].get("time_shifting_factor", 6)),
    )

    tasks = ("discrete", "continuous") if args.task == "all" else (args.task,)
    for task in tasks:
        dataset = _task_rows(config, task, smoke=args.smoke)
        if not dataset.rows:
            raise RuntimeError(f"No manifest rows found for {task}")
        task_root = run_root / task
        gen_root = task_root / "gen_imgs"
        marker_path = task_root / "inference_run.json"
        marker = {
            "model_name": config["model_name"],
            "task": task,
            "seed": seed,
            "smoke": bool(args.smoke),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "manifest": str(Path(config["data_root"]).expanduser().resolve() / "benchmark_manifest.json"),
            "expected_rows": len(dataset.rows),
            "resolution": int(config["inference"]["image_size"]),
            "sampling_steps": sampling_steps,
            "format": "RGB JPEG",
        }
        if rank == 0:
            task_root.mkdir(parents=True, exist_ok=True)
            if marker_path.exists():
                previous = json.loads(marker_path.read_text(encoding="utf-8"))
                if previous != marker:
                    raise FileExistsError(
                        f"Inference output belongs to a different checkpoint/protocol: {marker_path}; "
                        "choose an empty seed/task directory"
                    )
            elif gen_root.exists() and any(gen_root.rglob("*.jpg")):
                raise FileExistsError(f"Existing images have no checkpoint provenance: {gen_root}")
            else:
                _atomic_json(marker_path, marker)
        if dist.is_initialized():
            dist.barrier()
        if not marker_path.is_file():
            raise RuntimeError(f"Inference marker was not initialized: {marker_path}")

        indices = _identity_indices(dataset, task, rank, world_size)
        image_size = int(config["inference"]["image_size"])
        for count, row_index in enumerate(indices, start=1):
            row = dataset.rows[row_index]
            item = dataset[row_index]
            # Adapter inputs contain only radar, normalized pose and map id.
            _infer_one(
                row,
                item,
                task,
                model,
                vae,
                sampler,
                device=device,
                dtype=dtype,
                config=config,
                seed=seed,
                gen_root=gen_root,
            )
            if rank == 0 and (count == 1 or count % 100 == 0):
                print(f"{task}: generated/verified {count}/{len(indices)} assigned rows on rank 0")

        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            _write_provenance(
                task_root,
                task,
                dataset.rows,
                size=image_size,
                seed=seed,
                checkpoint_path=checkpoint_path,
                checkpoint_sha=checkpoint_sha,
            )
            print(f"{task} output: {gen_root}")
    if initialized_here:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
