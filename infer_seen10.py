"""Manifest-driven discrete and continuous Seen-10 generation with frozen best adapter."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader

from csgo_seen10.checkpoint import load_checkpoint, sha256_file
from csgo_seen10.data import Seen10Dataset, collate_seen10
from csgo_seen10.fast_inference import (
    SEED_POLICY,
    AsyncJpegWriter,
    FixedEulerStep,
    FixedNextDiTInference,
    compile_mode_label,
    compile_mode_uses_cudagraphs,
    euler_time_grid,
    make_latent_batch,
    pad_collated_batch,
    stable_seed,
    torch_compile_kwargs,
    validate_fixed_sampler,
)
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
    return stable_seed(seed, task, sample_id)


class _Rank0StageError(RuntimeError):
    """A rank-0-only stage failed in a distributed inference run."""

    def __init__(self, payload: dict[str, str]):
        self.stage = payload["stage"]
        self.original_type = payload["exception_type"]
        self.original_message = payload["message"]
        super().__init__(
            f"{self.stage} failed on rank 0 ({self.original_type}): {self.original_message}"
        )


def _rank0_exception_payload(stage: str, error: Exception) -> dict[str, str]:
    return {
        "stage": stage,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }


def _run_rank0_stage(stage: str, action) -> None:
    """Run rank-0 work and make distributed ranks observe its failure together."""

    if not dist.is_initialized():
        # Keep the original exception type and traceback for ordinary single-card use.
        action()
        return

    error_payload = [None]
    original_error = None
    if dist.get_rank() == 0:
        try:
            action()
        except Exception as exc:
            original_error = exc
            error_payload[0] = _rank0_exception_payload(stage, exc)

    dist.broadcast_object_list(error_payload, src=0)
    if error_payload[0] is not None:
        propagated_error = _Rank0StageError(error_payload[0])
        if original_error is not None:
            raise propagated_error from original_error
        raise propagated_error


_LEGACY_MARKER_KEYS = (
    "model_name",
    "task",
    "seed",
    "smoke",
    "checkpoint",
    "checkpoint_sha256",
    "manifest",
    "expected_rows",
    "resolution",
    "sampling_steps",
    "format",
)


def _marker_compatible(previous: dict[str, Any], current: dict[str, Any], *, accept_legacy: bool) -> bool:
    if previous == current:
        return True
    if (
        current.get("engine") == "compiled"
        and current.get("compile_mode") in {"reduce-overhead/fullgraph", "default/fullgraph"}
        and "cudagraphs" in current
        and "cudagraphs" not in previous
    ):
        # The first compiled-run schema already recorded the exact compile mode,
        # but predated the redundant explicit CUDA Graph flag. Accept only that
        # one-field schema migration so an interrupted run can use the same
        # command/output root; all remaining provenance must still match.
        previous_schema = dict(current)
        previous_schema.pop("cudagraphs")
        if previous == previous_schema:
            return True
    if not accept_legacy or not all(key in current for key in _LEGACY_MARKER_KEYS):
        return False
    old_marker = {key: current[key] for key in _LEGACY_MARKER_KEYS}
    return previous == old_marker


def _provenance_compatible(
    existing: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    inference_metadata: dict[str, Any],
    accept_legacy: bool,
) -> bool:
    old = {(record.get("map_name"), record.get("file_frame"), record.get("checkpoint_sha256")) for record in existing}
    new = {(record["map_name"], record["file_frame"], record["checkpoint_sha256"]) for record in expected}
    def metadata_matches(record: dict[str, Any]) -> bool:
        if all(record.get(key) == value for key, value in inference_metadata.items()):
            return True
        if (
            inference_metadata.get("engine") == "compiled"
            and inference_metadata.get("compile_mode") in {"reduce-overhead/fullgraph", "default/fullgraph"}
            and "cudagraphs" in inference_metadata
            and "cudagraphs" not in record
        ):
            previous_schema = {
                key: value for key, value in inference_metadata.items() if key != "cudagraphs"
            }
            if all(record.get(key) == value for key, value in previous_schema.items()):
                return True
        return accept_legacy and all(key not in record for key in inference_metadata)

    all_metadata_matches = all(metadata_matches(record) for record in existing)
    return len(existing) == len(expected) and old == new and all_metadata_matches


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


class _IndexBatchSampler:
    """Yield stable, manifest-ordered index blocks for one distributed shard."""

    def __init__(self, indices: list[int], batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.indices = list(indices)
        self.batch_size = int(batch_size)

    def __iter__(self):
        for offset in range(0, len(self.indices), self.batch_size):
            yield self.indices[offset : offset + self.batch_size]

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def _validate_benchmark_options(
    *,
    batches: int | None,
    output_root: str | Path | None,
    configured_root: str | Path,
    task: str,
    smoke: bool,
    world_size: int,
    sampling_steps: int,
) -> None:
    if batches is None:
        return
    if batches <= 0:
        raise ValueError("--benchmark-batches must be a positive integer")
    if output_root is None:
        raise ValueError("--benchmark-batches requires an explicit, independent --output-root")
    if Path(output_root).expanduser().resolve() == Path(configured_root).expanduser().resolve():
        raise ValueError("Benchmark --output-root must differ from the configured formal output root")
    if task == "all":
        raise ValueError("Benchmark one task at a time with --task discrete or --task continuous")
    if smoke:
        raise ValueError("--benchmark-batches cannot be combined with --smoke")
    if world_size != 1:
        raise ValueError("--benchmark-batches requires WORLD_SIZE=1")
    if sampling_steps != 28:
        raise ValueError("Benchmark mode requires the formal 28 sampling steps")


def _benchmark_report(
    *,
    task: str,
    engine: str,
    compile_mode: str,
    cudagraphs: bool,
    batch_size: int,
    vae_batch_size: int,
    sampling_steps: int,
    batch_seconds: list[float],
    total_seconds: float,
    model_initialization_seconds: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> dict[str, Any]:
    batch_rows = []
    for index, seconds in enumerate(batch_seconds, start=1):
        batch_rows.append(
            {
                "batch": index,
                "phase": "cold" if index == 1 else "steady",
                "images": batch_size,
                "seconds": float(seconds),
                "images_per_second": batch_size / max(float(seconds), 1e-12),
                "includes_lazy_compile": bool(index == 1 and engine == "compiled"),
            }
        )
    rows = len(batch_rows) * batch_size
    return {
        "report_schema": 1,
        "benchmark_only": True,
        "task": task,
        "engine": engine,
        "compile_mode": compile_mode,
        "cudagraphs": bool(cudagraphs),
        "batch_size": int(batch_size),
        "vae_batch_size": int(vae_batch_size),
        "sampling_steps": int(sampling_steps),
        "batches": len(batch_rows),
        "rows": rows,
        "model_initialization_seconds": float(model_initialization_seconds),
        "first_batch_includes_lazy_compile": bool(engine == "compiled"),
        "batch_timings": batch_rows,
        "total_seconds": float(total_seconds),
        "total_images_per_second": rows / max(float(total_seconds), 1e-12),
        "cuda_peak_allocated_bytes": int(peak_allocated_bytes),
        "cuda_peak_reserved_bytes": int(peak_reserved_bytes),
    }


def _euler_grid(transport, num_steps: int, time_shifting_factor: float, device: torch.device) -> torch.Tensor:
    t0, t1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False,
        eval=True,
        reverse=False,
        last_step_size=0.0,
    )
    if (float(t0), float(t1)) == (0.0, 1.0):
        return euler_time_grid(num_steps, time_shifting_factor, device=device)
    times = torch.linspace(float(t0), float(t1), num_steps, dtype=torch.float32, device="cpu")
    if time_shifting_factor:
        factor = float(time_shifting_factor)
        times = times / (times + factor - factor * times)
    return times.to(device=device)


def _eager_euler(
    initial: torch.Tensor,
    model: LuminaSeen10Model,
    transport,
    model_kwargs: dict[str, torch.Tensor],
    *,
    sampling_steps: int,
    time_shifting_factor: float,
    device: torch.device,
) -> torch.Tensor:
    """Streaming Euler loop matching transport.ode.sample without retaining all states."""

    state = initial.float()
    times = _euler_grid(transport, sampling_steps, time_shifting_factor, device)
    drift = transport.get_drift()
    for step_index in range(sampling_steps - 1):
        timestep = torch.ones(state.shape[0], dtype=times.dtype, device=device) * times[step_index]
        velocity = drift(state, timestep, model, **model_kwargs).float()
        state = state + velocity * (times[step_index + 1] - times[step_index])
    return state


def _write_provenance(
    task_root: Path,
    task: str,
    rows: list[dict],
    *,
    size: int,
    seed: int,
    checkpoint_path: Path,
    checkpoint_sha: str,
    inference_metadata: dict[str, Any],
    accept_legacy: bool,
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
        record.update(inference_metadata)
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
        if not _provenance_compatible(
            existing,
            records,
            inference_metadata=inference_metadata,
            accept_legacy=accept_legacy,
        ):
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
    parser.add_argument("--inference-engine", "--engine", dest="engine", choices=("eager", "compiled"), default="eager")
    parser.add_argument(
        "--compile-mode",
        choices=("reduce-overhead", "default"),
        default="reduce-overhead",
        help="Compiled engine mode; the default preserves the previous reduce-overhead behavior",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vae-batch-size", type=int, default=1)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override the output base directory (the seed_<N> subdirectory is appended)",
    )
    parser.add_argument("--jpeg-workers", type=int, default=2)
    parser.add_argument("--jpeg-queue-size", type=int, default=8)
    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=None,
        help="Generate this many full formal-step batches into a fresh benchmark-only output root",
    )
    args = parser.parse_args()

    config = _read_config(args.config)
    configured_root = Path(config["output_root"]).expanduser().resolve()
    seed = int(config["inference"].get("seed", 0) if args.seed is None else args.seed)
    if args.batch_size <= 0 or args.vae_batch_size <= 0:
        raise ValueError("--batch-size and --vae-batch-size must be positive")
    if args.vae_batch_size > args.batch_size:
        raise ValueError("--vae-batch-size must not exceed --batch-size")
    if args.jpeg_workers <= 0 or args.jpeg_queue_size <= 0:
        raise ValueError("--jpeg-workers and --jpeg-queue-size must be positive")
    if float(config["inference"].get("cfg_scale", 1.0)) != 1.0:
        raise ValueError("Seen-10 uses direct native NextDiT conditioning; cfg_scale must remain 1.0")

    sampling_steps = 2 if args.smoke else int(config["inference"]["sampling_steps"])
    _validate_benchmark_options(
        batches=args.benchmark_batches,
        output_root=args.output_root,
        configured_root=configured_root,
        task=args.task,
        smoke=args.smoke,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        sampling_steps=sampling_steps,
    )

    rank, world_size, local_rank, initialized_here = _distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("Lumina Seen-10 inference requires a CUDA GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = _precision_dtype(config["model"].get("precision", "bf16"))

    checkpoint_seed_root = configured_root / f"seed_{seed}"
    checkpoint_run_root = checkpoint_seed_root / "smoke" if args.smoke else checkpoint_seed_root
    output_base = Path(args.output_root).expanduser().resolve() if args.output_root else configured_root
    output_seed_root = output_base / f"seed_{seed}"
    run_root = output_seed_root / "smoke" if args.smoke else output_seed_root
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else checkpoint_run_root / "checkpoints" / "best.pt"
    )
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected adapter checkpoint not found: {checkpoint_path}")
    checkpoint_sha = sha256_file(checkpoint_path)

    model_initialization_started = time.monotonic() if args.benchmark_batches is not None else None
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
    model_initialization_seconds = (
        time.monotonic() - model_initialization_started
        if model_initialization_started is not None
        else 0.0
    )

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
    sampler_name = str(config["inference"].get("sampler", "euler"))
    sampling_method = sampler_name.lower()
    time_shifting_factor = float(config["inference"].get("time_shifting_factor", 6))
    fast_denoiser = None
    compiled_step = None
    sampler = None
    if args.engine == "compiled":
        validate_fixed_sampler(transport_config, config["inference"])
        if int(config["inference"]["image_size"]) != 448:
            raise ValueError("The compiled Seen-10 engine requires 448x448 generation")
        fast_denoiser = FixedNextDiTInference(
            model,
            batch_size=args.batch_size,
            image_size=int(config["inference"]["image_size"]),
        )
        try:
            compiled_step = torch.compile(
                FixedEulerStep(fast_denoiser),
                **torch_compile_kwargs(args.compile_mode),
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize the compiled fixed-shape Euler step; compiled mode never falls back to eager"
            ) from exc
    elif sampling_method != "euler":
        sampler = Sampler(transport).sample_ode(
            sampling_method=sampler_name,
            num_steps=sampling_steps,
            atol=1e-6,
            rtol=1e-3,
            time_shifting_factor=time_shifting_factor,
        )

    compiled_cudagraphs = (
        compile_mode_uses_cudagraphs(args.compile_mode) if args.engine == "compiled" else False
    )
    inference_metadata = {
        "engine": args.engine,
        "batch_size": int(args.batch_size),
        "vae_batch_size": int(args.vae_batch_size),
        "compile_mode": compile_mode_label(args.compile_mode) if args.engine == "compiled" else "none",
        "sampler": sampler_name,
        "sampling_steps": int(sampling_steps),
        "time_shifting_factor": time_shifting_factor,
        "seed_policy": SEED_POLICY,
        "jpeg_quality": int(config["inference"]["jpeg_quality"]),
    }
    if args.engine == "compiled":
        inference_metadata["cudagraphs"] = compiled_cudagraphs
    legacy_default = (
        args.engine == "eager"
        and args.batch_size == 1
        and args.vae_batch_size == 1
        and args.output_root is None
        and sampling_method == "euler"
        and time_shifting_factor == 6.0
        and int(config["inference"]["jpeg_quality"]) == 95
    )

    tasks = ("discrete", "continuous") if args.task == "all" else (args.task,)
    benchmark_only = args.benchmark_batches is not None
    for task in tasks:
        dataset = _task_rows(config, task, smoke=args.smoke)
        if not dataset.rows:
            raise RuntimeError(f"No manifest rows found for {task}")
        manifest_rows = len(dataset.rows)
        indices = _identity_indices(dataset, task, rank, world_size)
        benchmark_rows = 0
        if benchmark_only:
            benchmark_rows = int(args.benchmark_batches) * int(args.batch_size)
            if len(indices) < benchmark_rows:
                raise ValueError(
                    f"Benchmark requested {benchmark_rows} rows, but {task} has only {len(indices)} assigned rows"
                )
            indices = indices[:benchmark_rows]
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
            "expected_rows": benchmark_rows if benchmark_only else manifest_rows,
            "resolution": int(config["inference"]["image_size"]),
            "sampling_steps": sampling_steps,
            "engine": args.engine,
            "batch_size": int(args.batch_size),
            "vae_batch_size": int(args.vae_batch_size),
            "compile_mode": inference_metadata["compile_mode"],
            "sampler": sampler_name,
            "time_shifting_factor": time_shifting_factor,
            "seed_policy": SEED_POLICY,
            "output_root": str(output_base),
            "jpeg_quality": int(config["inference"]["jpeg_quality"]),
            "format": "RGB JPEG",
        }
        if args.engine == "compiled":
            marker["cudagraphs"] = compiled_cudagraphs
        if benchmark_only:
            marker.update(
                {
                    "benchmark_only": True,
                    "benchmark_batches": int(args.benchmark_batches),
                    "benchmark_rows": benchmark_rows,
                    "manifest_rows": manifest_rows,
                }
            )

        def initialize_marker() -> None:
            task_root.mkdir(parents=True, exist_ok=True)
            if benchmark_only:
                report_path = task_root / "benchmark_report.json"
                provenance_path = task_root / "provenance.jsonl"
                has_images = gen_root.exists() and any(gen_root.rglob("*.jpg"))
                if marker_path.exists() or report_path.exists() or provenance_path.exists() or has_images:
                    raise FileExistsError(
                        f"Benchmark output must be fresh and benchmark-only: {task_root}; "
                        "choose a new --output-root"
                    )
                _atomic_json(marker_path, marker)
                return
            if marker_path.exists():
                previous = json.loads(marker_path.read_text(encoding="utf-8"))
                if not _marker_compatible(previous, marker, accept_legacy=legacy_default):
                    raise FileExistsError(
                        f"Inference output belongs to a different checkpoint/protocol: {marker_path}; "
                        "choose an empty seed/task directory"
                    )
            elif gen_root.exists() and any(gen_root.rglob("*.jpg")):
                raise FileExistsError(f"Existing images have no checkpoint provenance: {gen_root}")
            else:
                _atomic_json(marker_path, marker)

        _run_rank0_stage("marker initialization/compatibility check", initialize_marker)
        if not marker_path.is_file():
            raise RuntimeError(f"Inference marker was not initialized: {marker_path}")

        image_size = int(config["inference"]["image_size"])
        rows_by_id = {row["sample_id"]: row for row in dataset.rows}
        worker_count = int(config["inference"].get("num_workers", config["training"].get("num_workers", 4)))
        loader = DataLoader(
            dataset,
            batch_sampler=_IndexBatchSampler(indices, args.batch_size),
            num_workers=worker_count,
            pin_memory=True,
            persistent_workers=worker_count > 0,
            collate_fn=collate_seen10,
        )
        processed = 0
        batch_seconds: list[float] = []
        task_started = time.monotonic()
        if benchmark_only:
            torch.cuda.reset_peak_memory_stats(device)
            print(
                f"{task}: benchmark config engine={args.engine} "
                f"compile_mode={inference_metadata['compile_mode']} cudagraphs={compiled_cudagraphs} "
                f"batch={args.batch_size} vae_batch={args.vae_batch_size} steps={sampling_steps} "
                f"model_init={model_initialization_seconds:.2f}s rows={benchmark_rows}"
            )
        with AsyncJpegWriter(
            workers=args.jpeg_workers,
            max_pending=args.jpeg_queue_size,
            quality=int(config["inference"]["jpeg_quality"]),
        ) as jpeg_writer:
            if benchmark_only:
                def batches():
                    loader_iterator = iter(loader)
                    for next_batch_index in range(1, int(args.benchmark_batches) + 1):
                        torch.cuda.synchronize(device)
                        batch_started = time.monotonic()
                        yield next_batch_index, next(loader_iterator), batch_started
            else:
                def batches():
                    for next_batch_index, next_batch in enumerate(loader, start=1):
                        yield next_batch_index, next_batch, None

            for batch_index, short_batch, batch_started in batches():
                batch, real_size = pad_collated_batch(short_batch, args.batch_size)
                batch_rows = [rows_by_id[sample_id] for sample_id in batch["sample_id"]]
                needs_write: list[int] = []
                for position, row in enumerate(batch_rows[:real_size]):
                    output_path = gen_root / row["map_name"] / f"{row['file_frame']}.jpg"
                    if not (output_path.is_file() and _valid_jpeg(output_path, image_size)):
                        needs_write.append(position)
                processed += real_size
                if benchmark_only and len(needs_write) != real_size:
                    raise RuntimeError("Benchmark batches must generate every selected row from a fresh output root")
                if not needs_write:
                    if rank == 0 and (batch_index == 1 or batch_index % 100 == 0):
                        elapsed = time.monotonic() - task_started
                        rate = processed / max(elapsed, 1e-9)
                        eta = (len(indices) - processed) / max(rate, 1e-9)
                        print(
                            f"{task}: generated/verified {processed}/{len(indices)} assigned rows on rank 0; "
                            f"elapsed={elapsed:.1f}s rate={rate:.3f} img/s shard-eta={eta:.1f}s"
                        )
                    continue

                radar = batch["radar"].to(device=device, non_blocking=True)
                pose = batch["pose"].to(device=device, non_blocking=True)
                map_id = batch["map_id"].to(device=device, non_blocking=True)
                task_id = torch.zeros(args.batch_size, dtype=torch.long, device=device)
                latent = make_latent_batch(
                    batch["sample_id"],
                    seed=seed,
                    task=task,
                    image_size=image_size,
                    channels=int(config["model"].get("in_channels", 16)),
                    dtype=dtype,
                    device=device,
                )
                kwargs = {"radar": radar, "pose": pose, "map_id": map_id, "task_id": task_id}

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                    if args.engine == "compiled":
                        assert fast_denoiser is not None and compiled_step is not None
                        cap_features = fast_denoiser.prepare_condition(radar, pose, map_id, task_id)
                        times = euler_time_grid(sampling_steps, time_shifting_factor, device=device)
                        state = latent.float()
                        if compiled_cudagraphs:
                            try:
                                torch.compiler.cudagraph_mark_step_begin()
                            except Exception as exc:
                                raise RuntimeError(
                                    "Could not mark the beginning of a reduce-overhead CUDA Graph batch"
                                ) from exc
                        for step_index in range(sampling_steps - 1):
                            timesteps = torch.ones(args.batch_size, dtype=times.dtype, device=device) * times[step_index]
                            dt = times[step_index + 1] - times[step_index]
                            try:
                                state = compiled_step(state, timesteps, dt, cap_features)
                            except Exception as exc:
                                raise RuntimeError(
                                    "Compiled fixed-shape Euler step failed; no eager fallback is enabled"
                                ) from exc
                    elif sampling_method == "euler":
                        state = _eager_euler(
                            latent,
                            model,
                            transport,
                            kwargs,
                            sampling_steps=sampling_steps,
                            time_shifting_factor=time_shifting_factor,
                            device=device,
                        )
                    else:
                        state = sampler(latent, model, **kwargs)[-1]

                    selected_positions = torch.tensor(needs_write, dtype=torch.long, device=device)
                    selected_state = state.index_select(0, selected_positions)
                    decoded_rows: list[torch.Tensor] = []
                    for decode_start in range(0, len(needs_write), args.vae_batch_size):
                        decode_chunk = selected_state[decode_start : decode_start + args.vae_batch_size]
                        chunk_count = decode_chunk.shape[0]
                        if chunk_count < args.vae_batch_size:
                            extra = decode_chunk[-1:].expand(args.vae_batch_size - chunk_count, *decode_chunk.shape[1:])
                            decode_chunk = torch.cat((decode_chunk, extra), dim=0)
                        decoded = vae.decode(
                            decode_chunk / float(config["model"]["vae_scale"])
                            + float(config["model"]["vae_shift"])
                        ).sample
                        decoded_rows.append(decoded[:chunk_count])
                    decoded = torch.cat(decoded_rows, dim=0)
                    pixels_batch = (
                        decoded.float()
                        .add(1.0)
                        .div(2.0)
                        .clamp_(0.0, 1.0)
                        .mul(255.0)
                        .round()
                        .to(torch.uint8)
                        .permute(0, 2, 3, 1)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )

                for row, pixels in zip((batch_rows[position] for position in needs_write), pixels_batch):
                    output_path = gen_root / row["map_name"] / f"{row['file_frame']}.jpg"
                    jpeg_writer.submit(output_path, pixels, size=image_size)
                if benchmark_only:
                    torch.cuda.synchronize(device)
                    batch_elapsed = time.monotonic() - float(batch_started)
                    batch_seconds.append(batch_elapsed)
                    batch_phase = "cold" if batch_index == 1 else "steady"
                    lazy_compile_text = ", includes lazy compile" if batch_index == 1 and args.engine == "compiled" else ""
                    print(
                        f"{task}: benchmark batch {batch_index}/{args.benchmark_batches} "
                        f"({batch_phase}{lazy_compile_text}) {real_size} images in {batch_elapsed:.2f}s "
                        f"({real_size / max(batch_elapsed, 1e-9):.3f} img/s)"
                    )
                if not benchmark_only and rank == 0 and (batch_index == 1 or batch_index % 100 == 0):
                    elapsed = time.monotonic() - task_started
                    rate = processed / max(elapsed, 1e-9)
                    eta = (len(indices) - processed) / max(rate, 1e-9)
                    print(
                        f"{task}: generated/verified {processed}/{len(indices)} assigned rows on rank 0; "
                        f"elapsed={elapsed:.1f}s rate={rate:.3f} img/s shard-eta={eta:.1f}s"
                    )

        elapsed = time.monotonic() - task_started
        rate = processed / max(elapsed, 1e-9)
        if benchmark_only:
            torch.cuda.synchronize(device)
            report = _benchmark_report(
                task=task,
                engine=args.engine,
                compile_mode=inference_metadata["compile_mode"],
                cudagraphs=compiled_cudagraphs,
                batch_size=args.batch_size,
                vae_batch_size=args.vae_batch_size,
                sampling_steps=sampling_steps,
                batch_seconds=batch_seconds,
                total_seconds=elapsed,
                model_initialization_seconds=model_initialization_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
            report["seed"] = seed
            report["checkpoint_sha256"] = checkpoint_sha
            report["manifest_rows"] = manifest_rows
            report["output_root"] = str(output_base)
            report_path = task_root / "benchmark_report.json"
            _atomic_json(report_path, report)
            print(
                f"{task}: benchmark completed {report['rows']} images in {elapsed:.2f}s "
                f"({report['total_images_per_second']:.3f} img/s); "
                f"compile_mode={report['compile_mode']} cudagraphs={report['cudagraphs']} "
                f"batch={report['batch_size']} vae_batch={report['vae_batch_size']} "
                f"steps={report['sampling_steps']}; "
                f"peak CUDA allocated={report['cuda_peak_allocated_bytes']} bytes, "
                f"reserved={report['cuda_peak_reserved_bytes']} bytes; report={report_path}"
            )
            continue

        if rank == 0:
            print(
                f"{task}: rank 0 processed {processed} assigned rows in {elapsed:.1f}s "
                f"({rate:.3f} img/s average)"
            )

        if dist.is_initialized():
            dist.barrier()

        def write_task_provenance() -> None:
            _write_provenance(
                task_root,
                task,
                dataset.rows,
                size=image_size,
                seed=seed,
                checkpoint_path=checkpoint_path,
                checkpoint_sha=checkpoint_sha,
                inference_metadata=inference_metadata,
                accept_legacy=legacy_default,
            )
            print(f"{task} output: {gen_root}")

        _run_rank0_stage("provenance write", write_task_provenance)
    if initialized_here:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
