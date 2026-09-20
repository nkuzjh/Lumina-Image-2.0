# Lumina-Image-2.0 × CSGO Benchmark v2 Seen-10

This project produces the two Table 1 generation outputs: `seen_discrete_test`
and `seen_continuous`. It uses the shared evaluator's `protocol.py` to read the
verified report, manifest, fixed split files, mapped radar images and published
per-map Z calibration. No split is discovered from `images/`.

## Environment and weights

The model environment is project-local at `.venv`; the shared evaluator remains
in the separate UniLIP Python environment. On this server the setup script
clones the already Blackwell-tested `ControlAR/.venv` torch/FlashAttention base,
installs only the Seen-10 extras, and downloads the pinned public Lumina
checkpoint and its bundled FLUX VAE into `checkpoints/Lumina-Image-2.0`:

```bash
bash scripts/setup_csgo_seen10.sh
```

On another machine, set `LUMINA_CLONE_ENV` to a compatible environment. If it
does not exist, set `LUMINA_SETUP_PYTHON` to Python 3.11; the fallback installs
PyTorch 2.8/CUDA 12.8 and FlashAttention 2.8.3. Set `HF_TOKEN` only if needed.
The script validates CUDA architecture support and never installs into the
UniLIP conda environment. Setup is process-locked; an interrupted partial
`.venv` is moved to a timestamped `.venv.incomplete.*` backup before retry.

## Run

The first real-model smoke runs one training step, checks radar/pose gradients
and the frozen base, saves and strictly reloads the adapter, emits one image
for each generation split, then asks the shared evaluator to read both:

```bash
bash scripts/run_csgo_seen10.sh smoke --seed 0
```

Formal training uses 50,000 steps. Validation and adapter checkpoint saves
occur at steps 10,000, 20,000, 30,000, 40,000 and 50,000; `best.pt` selects
the lowest full `seen_validation` flow loss and `late.pt` points to the final
step. The main training loss curve is written to `loss_curve.png`.

```bash
bash scripts/run_csgo_seen10.sh train --seed 0
bash scripts/run_csgo_seen10.sh train --seed 0 --resume auto
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all
bash scripts/run_csgo_seen10.sh eval --seed 0 --task all
```

`infer` loads the same `best.pt` once for both tasks. Continuous clips are
generated frame-by-frame in manifest clip/frame order, using only each row's
radar, normalized `[x,y,z,pitch,yaw]` and map ID. The inference dataset has no
FPV target field and never opens target images. Resume skips only validated
448×448 RGB JPEGs under a run marker tied to the selected checkpoint hash.
`torchrun --nproc_per_node=N train_seen10.py ...` and the matching inference
entry are supported for multi-GPU sharding.

## Paths

- Adapter checkpoints and training logs: `outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0/seed_<seed>/checkpoints/`
- Formal predictions: `outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0/seed_<seed>/{discrete,continuous}/gen_imgs/<map>/<file_frame>.jpg`
- Shared evaluator results: `outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0/seed_<seed>/evaluation_shared/{discrete,continuous}/`
- Smoke-only predictions and evaluator input: the same seed's `smoke/` subtree

The shared evaluator is called from
`/home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py`; no evaluator
code is copied into this project.

## Verified smoke evidence

The seed-0 smoke completed on 2026-09-20 with Python 3.11, PyTorch
`2.11.0.dev20260124+cu128`, FlashAttention `2.8.3` and an RTX PRO 6000
Blackwell. The official base state dict matched SHA-256
`a7c09ebae62996a8289782161338a3cdba58c11d2d849c50b2d6502e152b0d6d`.
The real 2B step reported loss `0.487984`, nonzero radar/pose gradients and no
base gradients; the adapter was then strictly reloaded. Discrete
`cs_agency/file_num68_frame_421.jpg` and continuous
`cs_agency/file_num3_frame_205.jpg` were both verified as 448×448 RGB JPEGs.
The shared evaluator read each smoke prediction with expected/predicted/common
coverage `1/1/1`. Smoke metrics are diagnostic only and are not formal Table 1
results; `RUN_FULL=0`, so the 50,000-step run was intentionally not started.
