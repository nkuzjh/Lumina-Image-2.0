"""Adapter-only flow-matching fine-tuning for CSGO Benchmark v2 Seen-10."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from csgo_seen10.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from csgo_seen10.data import Seen10Dataset, collate_seen10
from csgo_seen10.model import LuminaSeen10Model, RadarPoseMapAdapter, load_native_dit


def _read_config(path: str | Path) -> dict:
    with Path(path).expanduser().open(encoding="utf-8") as source:
        return json.load(source)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distributed_context() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


class EpochShardSampler(Sampler[int]):
    """A deterministic permutation split evenly across ranks and restartable by batch."""

    def __init__(self, size: int, rank: int, world_size: int, seed: int):
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0

    @property
    def samples_per_rank(self) -> int:
        return self.size // self.world_size

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(self.size, generator=generator)
        shard = permutation[self.rank :: self.world_size][: self.samples_per_rank]
        return iter(shard[self.start_index :].tolist())

    def __len__(self) -> int:
        return max(0, self.samples_per_rank - self.start_index)


class SequentialShardSampler(Sampler[int]):
    """Validation shard without DistributedSampler's duplicate padding rows."""

    def __init__(self, size: int, rank: int, world_size: int):
        self.indices = list(range(rank, size, world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _precision_dtype(name: str) -> torch.dtype:
    values = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported precision {name!r}; choose bf16, fp16, or fp32") from exc


def _load_vae(config: dict, dtype: torch.dtype, device: torch.device):
    from diffusers import AutoencoderKL

    model_config = config["model"]
    base_dir = Path(model_config["base_model_dir"]).expanduser().resolve()
    vae = AutoencoderKL.from_pretrained(
        str(base_dir),
        subfolder=model_config.get("vae_subfolder", "vae"),
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    vae.eval().requires_grad_(False)
    configured_scale = float(model_config["vae_scale"])
    configured_shift = float(model_config["vae_shift"])
    vae_scale = float(getattr(vae.config, "scaling_factor", configured_scale))
    vae_shift = float(getattr(vae.config, "shift_factor", configured_shift) or 0.0)
    if not math.isclose(vae_scale, configured_scale, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"FLUX VAE scaling_factor mismatch: config={configured_scale}, VAE={vae_scale}")
    if not math.isclose(vae_shift, configured_shift, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"FLUX VAE shift_factor mismatch: config={configured_shift}, VAE={vae_shift}")
    return vae, vae_scale, vae_shift


def _make_model(config: dict, device: torch.device, dtype: torch.dtype) -> LuminaSeen10Model:
    model_config = config["model"]
    dit, cap_feat_dim = load_native_dit(
        model_config["base_model_dir"],
        model_name=model_config["name"],
        configured_cap_feat_dim=int(model_config["cap_feat_dim"]),
        qk_norm=bool(model_config.get("qk_norm", True)),
    )
    dit.to(device=device, dtype=dtype)
    adapter_config = config["adapter"]
    adapter = RadarPoseMapAdapter(
        cap_feat_dim=cap_feat_dim,
        radar_grid=tuple(adapter_config["radar_grid"]),
        fourier_frequencies=int(adapter_config["fourier_frequencies"]),
        task_count=int(adapter_config["task_count"]),
    )
    return LuminaSeen10Model(dit, adapter).to(device)


def _loader(
    dataset: Seen10Dataset,
    *,
    batch_size: int,
    num_workers: int,
    sampler: Sampler[int],
    seed: int,
    drop_last: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        collate_fn=collate_seen10,
        generator=generator,
    )


def _autocast(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)


def _encode_targets(batch: dict, vae, *, device: torch.device, dtype: torch.dtype, scale: float, shift: float):
    images = batch["target"].to(device=device, dtype=dtype, non_blocking=True)
    with torch.no_grad():
        latent = vae.encode(images).latent_dist.mode()
    return (latent.float() - shift) * scale


def _model_kwargs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    radar = batch["radar"].to(device=device, non_blocking=True)
    return {
        "radar": radar,
        "pose": batch["pose"].to(device=device, non_blocking=True),
        "map_id": batch["map_id"].to(device=device, non_blocking=True),
        "task_id": torch.zeros(radar.shape[0], dtype=torch.long, device=device),
    }


def _validate(
    model: torch.nn.Module,
    vae,
    loader: DataLoader,
    transport,
    *,
    device: torch.device,
    dtype: torch.dtype,
    vae_scale: float,
    vae_shift: float,
    seed: int,
    rank: int,
) -> float:
    rng_state = capture_rng_state()
    _seed_everything(seed + 104729 + rank)
    model.eval()
    loss_sum = torch.zeros(2, dtype=torch.float64, device=device)
    try:
        with torch.no_grad():
            for batch in loader:
                latent = _encode_targets(batch, vae, device=device, dtype=dtype, scale=vae_scale, shift=vae_shift)
                with _autocast(device, dtype):
                    loss_pack = transport.training_losses(model, latent, _model_kwargs(batch, device))
                loss_sum[0] += loss_pack["loss"].double().sum()
                loss_sum[1] += loss_pack["loss"].numel()
        if dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        if loss_sum[1].item() <= 0:
            raise RuntimeError("Validation produced no samples")
        value = (loss_sum[0] / loss_sum[1]).item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite Seen-10 validation loss: {value}")
        return float(value)
    finally:
        restore_rng_state(rng_state)
        model.train()


def _save_loss_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _plot_loss(path: Path, loss_log: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps: list[int] = []
    losses: list[float] = []
    if loss_log.is_file():
        with loss_log.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
    if not steps:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(steps, losses, linewidth=0.8)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Flow-matching loss")
    axis.set_title("Lumina-Image-2.0 Seen-10 training loss")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _resolve_resume(value: str | None, checkpoint_dir: Path) -> Path | None:
    if value is None:
        return None
    if value == "auto":
        candidate = checkpoint_dir / "late.pt"
        return candidate if candidate.is_file() else None
    return Path(value).expanduser().resolve()


def _smoke_reload(adapter: RadarPoseMapAdapter, checkpoint_path: Path, config: dict) -> None:
    test_adapter = RadarPoseMapAdapter(
        cap_feat_dim=int(config["model"]["cap_feat_dim"]),
        radar_grid=tuple(config["adapter"]["radar_grid"]),
        fourier_frequencies=int(config["adapter"]["fourier_frequencies"]),
        task_count=int(config["adapter"]["task_count"]),
    )
    payload = load_checkpoint(checkpoint_path, adapter=test_adapter)
    expected = adapter.state_dict()
    actual = test_adapter.state_dict()
    if expected.keys() != actual.keys() or any(not torch.equal(expected[name].cpu(), actual[name].cpu()) for name in expected):
        raise RuntimeError("Adapter checkpoint did not reload bit-exactly")
    print(f"strict adapter checkpoint reload OK (step={payload['step']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/csgo_seen10.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one real train step and save/reload its adapter")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume from a checkpoint, or auto from late.pt")
    args = parser.parse_args()

    config = _read_config(args.config)
    training = config["training"]
    seed = int(training.get("seed", 0) if args.seed is None else args.seed)
    max_steps = int(1 if args.smoke else (args.max_steps or training["max_steps"]))
    eval_every = int(training["eval_every"])
    save_every = int(training["save_every"])
    if not args.smoke:
        if max_steps <= 0 or max_steps % 5 != 0:
            raise ValueError("Formal max_steps must be positive and divisible by five")
        if eval_every != max_steps // 5 or save_every != max_steps // 5:
            raise ValueError("Formal eval/save intervals must both equal max_steps / 5 (exactly five milestones)")

    rank, world_size, local_rank, initialized_here = _distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("Lumina Seen-10 training requires a CUDA GPU")
    if args.smoke and world_size != 1:
        raise ValueError("Smoke mode uses one GPU/process so its single-batch evidence is unambiguous")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = _precision_dtype(config["model"].get("precision", "bf16"))
    _seed_everything(seed)

    output_root = Path(config["output_root"]).expanduser().resolve() / f"seed_{seed}"
    run_dir = output_root / "smoke" if args.smoke else output_root
    checkpoint_dir = run_dir / "checkpoints"
    loss_log = run_dir / "training_loss.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_path = _resolve_resume(args.resume, checkpoint_dir)
    existing_outputs = [
        path for path in run_dir.iterdir()
        if not (not args.smoke and path.name == "smoke")
    ]
    if resume_path is None and existing_outputs:
        raise FileExistsError(f"Run output already contains files: {run_dir}; pass --resume auto or choose another seed")
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Requested resume checkpoint does not exist: {resume_path}")

    data_config = config
    train_dataset = Seen10Dataset(
        data_config["data_root"],
        data_config["shared_eval_dir"],
        training["train_split"],
        load_targets=True,
        image_size=int(config["inference"]["image_size"]),
        radar_size=int(config["adapter"]["radar_size"]),
        max_samples=1 if args.smoke else None,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("The Seen-10 training split is empty")

    train_sampler = EpochShardSampler(len(train_dataset), rank, world_size, seed)
    batch_size = int(training["batch_size_per_gpu"])
    sampler_samples = train_sampler.samples_per_rank
    steps_per_epoch = sampler_samples // batch_size
    if steps_per_epoch <= 0:
        raise RuntimeError("The training split is too small for this world size and batch size")
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0 if args.smoke else int(training["num_workers"]),
        sampler=train_sampler,
        seed=seed + 1,
        drop_last=True,
    )

    validation_loader = None
    if not args.smoke:
        valid_dataset = Seen10Dataset(
            data_config["data_root"],
            data_config["shared_eval_dir"],
            training["validation_split"],
            load_targets=True,
            image_size=int(config["inference"]["image_size"]),
            radar_size=int(config["adapter"]["radar_size"]),
        )
        valid_sampler = SequentialShardSampler(len(valid_dataset), rank, world_size)
        validation_loader = _loader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=int(training["num_workers"]),
            sampler=valid_sampler,
            seed=seed + 2,
            drop_last=False,
        )

    model = _make_model(config, device, dtype)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    adapter = model.module.adapter if isinstance(model, DistributedDataParallel) else model.adapter
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        eps=1e-8,
        betas=(0.9, 0.95),
    )
    vae, vae_scale, vae_shift = _load_vae(config, dtype, device)
    from transport import create_transport

    transport = create_transport(
        training.get("path_type", "Linear"),
        training.get("prediction", "velocity"),
        training.get("loss_weight"),
        snr_type=training.get("snr_type", "uniform"),
        do_shift=bool(training.get("do_shift", True)),
        seq_len=(int(config["inference"]["image_size"]) // 16) ** 2,
    )

    start_step = 0
    best_validation_loss = float("inf")
    if resume_path is not None:
        payload = load_checkpoint(resume_path, adapter=adapter, optimizer=optimizer, restore_rng=True)
        saved_steps = int(payload["step"])
        saved_max_steps = int(payload.get("config", {}).get("training", {}).get("max_steps", max_steps))
        if saved_max_steps != max_steps and not args.smoke:
            raise ValueError(f"Cannot resume max_steps={max_steps} from checkpoint configured for {saved_max_steps}")
        start_step = saved_steps
        best_validation_loss = float(payload.get("best_validation_loss", float("inf")))
        if rank == 0:
            print(f"resuming from {resume_path} at completed step {start_step}")
    elif args.resume is not None and args.resume != "auto":
        raise FileNotFoundError(f"Requested resume checkpoint does not exist: {args.resume}")
    else:
        # Keep initial adapter weights synchronized, then give each rank an
        # independent transport/noise stream for its distinct data shard.
        _seed_everything(seed + rank)

    if start_step > max_steps:
        raise ValueError(f"Resume step {start_step} is past requested max_steps={max_steps}")

    epoch = -1
    train_iterator = None
    model.train()
    for step_index in range(start_step, max_steps):
        current_epoch = step_index // steps_per_epoch
        batch_offset = step_index % steps_per_epoch
        if current_epoch != epoch:
            epoch = current_epoch
            train_sampler.set_epoch(epoch, start_index=batch_offset * batch_size)
            train_iterator = iter(train_loader)
        try:
            batch = next(train_iterator)
        except StopIteration as exc:
            raise RuntimeError("Training sampler ended before its configured epoch boundary") from exc

        latent = _encode_targets(batch, vae, device=device, dtype=dtype, scale=vae_scale, shift=vae_shift)
        kwargs = _model_kwargs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, dtype):
            loss_pack = transport.training_losses(model, latent, kwargs)
            loss = loss_pack["loss"].mean()
        loss.backward()

        if args.smoke:
            base_has_grad = any(parameter.grad is not None for parameter in model.dit.parameters())
            radar_grad = sum(
                float(parameter.grad.detach().abs().sum().item())
                for parameter in adapter.radar_encoder.parameters()
                if parameter.grad is not None
            )
            pose_grad = sum(
                float(parameter.grad.detach().abs().sum().item())
                for parameter in adapter.pose_mlp.parameters()
                if parameter.grad is not None
            )
            if base_has_grad or radar_grad <= 0.0 or pose_grad <= 0.0:
                raise RuntimeError(
                    "Adapter backward check failed: expected nonzero radar/pose gradients and no base-model gradients "
                    f"(radar={radar_grad}, pose={pose_grad}, base_has_grad={base_has_grad})"
                )
            if rank == 0:
                print(f"adapter backward OK (radar_grad={radar_grad:.4g}, pose_grad={pose_grad:.4g}, base frozen)")

        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=float(training["grad_clip"]),
        )
        optimizer.step()
        step = step_index + 1
        train_loss = float(loss.detach().item())
        if dist.is_initialized():
            global_loss = torch.tensor(train_loss, dtype=torch.float64, device=device)
            dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
            train_loss = float((global_loss / world_size).item())
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}: {train_loss}")
        if rank == 0:
            _save_loss_record(loss_log, {"step": step, "loss": train_loss})
            if step % int(training["log_every"]) == 0 or args.smoke:
                print(f"step={step}/{max_steps} loss={train_loss:.7g}")

        if args.smoke:
            checkpoint_path = save_checkpoint(
                checkpoint_dir,
                step=step,
                adapter=adapter,
                optimizer=optimizer,
                best_validation_loss=-1.0,
                validation_loss=-1.0,
                config={**config, "training": {**training, "max_steps": 1, "smoke": True}},
                is_best=True,
            )
            break

        if step % eval_every == 0 and step % save_every == 0:
            assert validation_loader is not None
            validation_loss = _validate(
                model,
                vae,
                validation_loader,
                transport,
                device=device,
                dtype=dtype,
                vae_scale=vae_scale,
                vae_shift=vae_shift,
                # Keep validation noise fixed so milestone losses are directly
                # comparable when selecting best.pt.
                seed=seed,
                rank=rank,
            )
            is_best = validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
            if rank == 0:
                print(
                    f"validation step={step}/{max_steps} loss={validation_loss:.7g} "
                    f"best={best_validation_loss:.7g}"
                )
            checkpoint_path = save_checkpoint(
                checkpoint_dir,
                step=step,
                adapter=adapter,
                optimizer=optimizer,
                best_validation_loss=best_validation_loss,
                validation_loss=validation_loss,
                config={**config, "training": {**training, "max_steps": max_steps}},
                is_best=is_best,
            )
            if rank == 0:
                print(f"saved {checkpoint_path}")

    if rank == 0:
        _plot_loss(run_dir / "loss_curve.png", loss_log)
        if args.smoke:
            _smoke_reload(adapter, checkpoint_dir / "best.pt", config)
            print(f"smoke checkpoint: {checkpoint_dir / 'best.pt'}")
        else:
            checkpoint_steps = sorted(checkpoint_dir.glob("step_*.pt"))
            if len(checkpoint_steps) != 5:
                raise RuntimeError(f"Formal training must produce exactly five checkpoints, found {len(checkpoint_steps)}")
            if not (checkpoint_dir / "late.pt").is_symlink() or not (checkpoint_dir / "best.pt").is_symlink():
                raise RuntimeError("Formal training did not create late.pt and best.pt links")
        print(f"loss curve: {run_dir / 'loss_curve.png'}")
    if initialized_here:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
