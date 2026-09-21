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

## Inference acceleration plan

The inference path is layered so the existing commands above remain valid.
With no new options, inference still selects `eager`, batch 1, VAE batch 1 and
the configured output directory. Existing legacy markers are accepted for
that default run. Accelerated runs should use an independent `--output-root`
so their different batch/engine provenance cannot be mixed with earlier JPEGs.

### P0: fixed manifest batches

Inference uses a sequential batch sampler over each rank's existing manifest
indices. It keeps discrete identity order and continuous clip/frame order;
each row gets its original `sha256(seed, task, sample_id)` latent before the
latents are stacked. The final short batch repeats its last row to keep the
DiT shape fixed, then discards padding when decoding/writing. Resume validates
each JPEG; if a batch contains any missing/invalid output, it recomputes that
whole fixed block and writes only the missing/invalid members. Both generation
split sizes (20,000 discrete and 12,800 continuous) divide evenly by 16.

### P1: fixed NextDiT forward and compile

`--inference-engine compiled` enables an inference-only native NextDiT fast path at
448px: 23 condition tokens, 784 image tokens and an 807-token joint sequence.
It precomputes the native RoPE axes and fixed FlashAttention `cu_seqlens` for
per-sample segments of length 23, 784 and 807. The static attention path
bypasses `.tolist()`, `.item()`, `nonzero`, and dynamic pad/unpad work. The
radar/pose/map adapter, cap embedder and two context-refiner blocks run once
per batch; each of the 27 Euler updates then runs the two noise-refiner blocks
and 26 main blocks. The Euler time grid preserves the configured
`t / (t + 6 - 6*t)` shift and float32 state/update semantics, while retaining
only the final latent. It does not use the ControlAR autoregressive KV cache.

The compiled engine applies Inductor `fullgraph=True` to one fixed-shape
denoiser-plus-Euler step, reused over the time grid. For command compatibility,
omitting `--compile-mode` still selects `reduce-overhead` and its CUDA Graphs.
The runner calls `torch.compiler.cudagraph_mark_step_begin()` once after
preparing each generated batch and before its 27 Euler updates, so all updates
in that batch share one CUDA Graph inference step. `--compile-mode default`
keeps default Inductor compilation but explicitly sets
`options={"triton.cudagraphs": False}` (and therefore does not pass a conflicting
`mode` argument to `torch.compile`). Its marker uses
`default/fullgraph`; reduce-overhead uses `reduce-overhead/fullgraph`, and the
two modes cannot share an output directory. Both support the configured
Linear velocity path and Euler sampler. Unsupported shapes/settings or a
compile failure stop with an error; there is no silent eager fallback.

### P2: VAE and JPEG pipeline

`--vae-batch-size` decodes selected final latents in microbatches. A bounded
JPEG worker pool encodes while subsequent GPU batches run. Each result is
written to a temporary file, validated as 448×448 RGB JPEG, and atomically
renamed; worker exceptions are propagated before inference exits. VAE batch
size and queue settings should be tuned against available CPU/GPU memory.

### Accelerated commands

Batch 16 with eager NextDiT is available for an intermediate comparison:

```bash
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all \
  --inference-engine eager --batch-size 16 --vae-batch-size 4 \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_eager_b16
```

The measured production recommendation is default Inductor without CUDA
Graphs. Use an explicit mode and a fresh output root:

```bash
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all \
  --inference-engine compiled --compile-mode default --batch-size 16 --vae-batch-size 4 \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_default_b16
bash scripts/run_csgo_seen10.sh eval --seed 0 --task all \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_default_b16
```

The repaired CUDA Graph route remains available. Omitting `--compile-mode`
continues to mean `reduce-overhead`, and an interrupted pre-fix compiled marker
that differs only by the previously absent `cudagraphs` field is accepted:

```bash
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all \
  --inference-engine compiled --compile-mode reduce-overhead --batch-size 16 --vae-batch-size 4 \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_compiled_b16
```

The isolated benchmark path runs full 28-step sampling on the first N complete
manifest batches. It requires one task, `WORLD_SIZE=1`, an explicit output root
different from the configured formal root, and a fresh task output directory.
It writes a `benchmark_only` marker and `benchmark_report.json`, does not write
formal provenance, and cannot be used as evaluator input:

```bash
bash scripts/run_csgo_seen10.sh infer --seed 0 --task discrete \
  --inference-engine compiled --compile-mode default --batch-size 16 \
  --vae-batch-size 4 --benchmark-batches 3 \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_default_b16_bench

bash scripts/run_csgo_seen10.sh infer --seed 0 --task discrete \
  --inference-engine compiled --compile-mode reduce-overhead --batch-size 16 \
  --vae-batch-size 4 --benchmark-batches 3 \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_reduce_b16_bench
```

The report records model initialization time, every batch's elapsed seconds
and images/s (batch 1 is cold and includes lazy compilation for compiled
runs), total elapsed time and throughput, compile mode, CUDA Graph use, batch
sizes, step count and CUDA peak allocated/reserved bytes. Timing synchronizes
CUDA only for benchmark runs and includes data loading in each batch time; the
final total includes draining the JPEG writer.

### Measured result and decision (2026-09-21)

The tests used the RTX PRO 6000 Blackwell, the same checkpoint and first 48
discrete rows, 28 sampling points, DiT batch 16 and VAE batch 4. Other GPU jobs,
including the ongoing eager run, remained active. The eager reference is the
median of 11 consecutive completed-batch intervals before the compile tests;
the next two uncontended intervals after the tests were 61.68 and 61.58 seconds,
confirming that it returned to the same baseline.

| Engine | Cold batch | Steady batch | Steady rate | Relative to eager | Peak CUDA reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager, batch 16 | n/a | 61.60 s median | 0.260 img/s | 1.00× | not reset on the live run |
| default/fullgraph, no CUDA Graph | 71.98 s | 35.82 s mean | 0.447 img/s | 1.72× | 9.67 GiB |
| reduce-overhead/fullgraph, CUDA Graph | 48.31 s | 36.24 s mean | 0.442 img/s | 1.70× | 10.73 GiB |

Each compiled route completed three consecutive full batches and produced 48
valid JPEGs. The two steady values are effectively tied under co-tenancy;
default was 1.2% faster in this sample and reserved 1.07 GiB less memory.
Cold times are recorded but are not directly comparable because the modes ran
sequentially and shared Inductor caches. The 48-image decoded-output comparison
gave 47.56 dB PSNR between the compiled modes and about 44.9 dB between either
compiled mode and eager; the files are not byte-identical because the kernel
paths have small numerical differences.

For the 32,800-image workload (2,050 batches), a linear steady-state projection
is about 20.4 hours for default compile versus 35.1 hours for eager. This is a
shared-GPU observation, excludes initialization/cold compile overhead, and is
not a completion-time guarantee. The ControlAR 9-hour result remains a
different-model measurement. Because CUDA Graphs showed no steady throughput
gain here while adding memory and lifecycle complexity, the final operational
choice is explicit `--compile-mode default`; repaired `reduce-overhead` remains
available for future isolated measurements.

Normal marker and provenance rows record engine, DiT batch, VAE batch, compile
mode, sampler, sampling steps, time shift, seed policy and checkpoint hash.
Compiled runs also record their `cudagraphs` setting. Eager marker and
provenance schemas remain unchanged for default-command resume compatibility.
`--compile-mode` is meaningful for the compiled engine; omitting it retains
`reduce-overhead` for command compatibility, although the measured recommendation
above explicitly selects `default`. `--output-root` overrides the base output directory;
`seed_<N>` and the task subdirectories are appended. The same option on `eval` points the shared
evaluator at that prediction root. It redirects predictions/evaluation only;
unless `--checkpoint` is supplied, inference still loads the configured
`seed_<N>/checkpoints/best.pt`.

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
