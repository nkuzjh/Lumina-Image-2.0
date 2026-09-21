# Lumina-Image-2.0 接入 CSGO Benchmark v2 Seen-10

本项目只承担 Table 1 的 generation 行：使用同一个 validation 选出的冻结 checkpoint，分别生成 `seen_discrete_test` 和 `seen_continuous` 的 448×448 RGB 结果。`RUN_FULL=0`，本轮以真实数据/模型 smoke 为验收边界，不启动 50,000 样本的完整训练。

## 变更方案

- `csgo_seen10/data.py`：通过已有共享评测器的 manifest protocol 读取 `minimal_dataset_report.json`、manifest、splits、radar 映射和发布的 Z calibration；固定 Seen-10 地图顺序，保留 `sample_id/clip_id/frame_index`。训练/验证才打开目标 FPV，推理不打开 GT 图像。
- `csgo_seen10/model.py`：保留原生 `NextDiT` 和 flow-matching 路径。radar 由轻量 CNN 转成空间 condition tokens；归一化 `[x,y,z,pitch,yaw]` 由显式 Fourier feature + MLP 转成数值 tokens；另加地图/任务 token。条件 token 复用 Lumina 的 caption feature 入口，不把 pose 仅写成自然语言。
- `configs/csgo_seen10.json`：固定数据根、原生基座 checkpoint、448 分辨率、训练预算、采样器和条件结构。正式预算的 eval/save interval 为总 step 的 1/5，只产生 5 个里程碑。
- `train_seen10.py`：复用 Flux VAE、Lumina transport loss、AdamW 和原生 DiT checkpoint 加载；训练 radar/pose/map adapter，用完整 `seen_validation` loss 选最优里程碑，保存可恢复 optimizer/RNG 状态，建立 `late.pt`/`best.pt` 链接，并画制主 loss 曲线。
- `infer_seen10.py`：从同一 `best.pt` 推理 discrete/continuous，continuous 严格按 manifest clip/frame 顺序逐帧独立生成；以稳定 sample seed 支持中断续跑，只跳过经尺寸/模式验证的既有图片，并写入 provenance manifest。
- `scripts/run_csgo_seen10.sh`：统一 `smoke|train|infer|eval --seed` 入口；模型环境与 `UNILIP_PYTHON` 评测环境分离。
- `scripts/setup_csgo_seen10.sh`：创建项目独立 `.venv`、安装依赖并获取官方 Lumina checkpoint；不改 UniLIP 环境。
- `CSGO_SEEN10.md`：记录实际环境、直接命令、checkpoint/预测/评测路径和 smoke 证据。

`BUILD_SHARED_EVALUATOR=0`：不修改 `/home/jiahao/task/csgo_benchmark_v2_eval_general`，只用其 `run_eval.py smoke discrete/continuous` 验证标准输出。DATA_ROOT 严格只读。

## 数据和模型 I/O

- 训练：`seen_train` 的 radar + normalized 5DoF + map id -> 目标 FPV latent；验证仅用 `seen_validation`。
- 推理：只读 radar + normalized 5DoF + identity，不打开目标/历史/未来 FPV。
- 输出：`outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0/seed_<seed>/{discrete,continuous}/gen_imgs/<map>/<file_frame>.jpg`。

## 运行入口

```bash
bash scripts/setup_csgo_seen10.sh
bash scripts/run_csgo_seen10.sh smoke --seed 0
bash scripts/run_csgo_seen10.sh train --seed 0
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all
bash scripts/run_csgo_seen10.sh eval --seed 0 --task all
```

Smoke 必须证明：manifest dataset 可读 batch；真实 Lumina DiT 有一次 forward/backward；adapter checkpoint 保存后可严格 reload；discrete 和 continuous 各生成一张 448×448 RGB JPEG；共享评测器均能读取。

## 推理加速实施方案

保留原始入口 `bash scripts/run_csgo_seen10.sh infer --seed 0 --task all`，默认值仍为 eager、DiT batch 1、VAE batch 1，输出仍落在配置中的 seed/task 目录。Python CLI 同时接受 `--inference-engine eager|compiled` 和短别名 `--engine`。加速试验应使用独立 `--output-root`，避免把不同 batch/引擎的图片写入同一套结果。旧默认 run marker/provenance 可继续续跑；参数不同则由 marker 阻止混写。

- **P0：固定 manifest batch。** 各 rank 对原有 manifest index 使用顺序 batch sampler；离散样本按原 identity 顺序，连续样本仍按 clip/frame 顺序。每个 `sample_id` 先按原稳定 seed 独立生成初始 latent，再堆叠为 batch。尾批复制最后一个样本填满静态 batch，生成后丢弃填充位。若一个块中有缺图/坏图，重算整块，只原子写缺失/坏图。可先用 eager batch 做吞吐比较；默认 eager batch 1 对旧命令兼容。
- **P1：固定 NextDiT 路径与单步编译。** 仅优化配置所用的 448px、23 条 condition、784 个 image token、807 条 joint sequence。预计算 native RoPE（condition axis0 为 0..22；image axis0 为 23，row/col 为 28×28）和 FlashAttention varlen 分段边界，每个样本分别形成长度 23、784、807 的 segment。绕过动态 `.tolist()`、`.item()`、`nonzero` 与 pad/unpad；adapter、cap embedder、context-refiner 每 batch 执行一次。ODE 使用原 28 个时间点、27 次 Euler 更新及唯一的 `t/(t+6-6*t)` shift，state/drift 保持 FP32，只保留最终 latent。`torch.compile(fullgraph=True)` 仅编译一个 denoiser+Euler step 并重复调用；它不采用 ControlAR 的自回归 KV cache。为兼容旧启动方式，省略 mode 时仍为 `reduce-overhead`；每个真实 batch 在准备好 condition 和 state 后、Euler 循环前调用一次 `torch.compiler.cudagraph_mark_step_begin()`，27 个 step 内不重复调用。`--compile-mode default` 不传与 options 冲突的 mode 参数，使用 Inductor 默认模式、`fullgraph=True` 并显式传 `options={"triton.cudagraphs": False}`。marker/provenance 仅 compiled 新增 `cudagraphs`，reduce 标签沿用 `reduce-overhead/fullgraph`，default 为 `default/fullgraph`，两者不可混写；只缺旧版 `cudagraphs` 字段且其他字段完全相同的中断 compiled run 可继续。原 eager marker/provenance schema 不变。compiled 当前要求 Linear velocity + Euler，fullgraph 不支持或 compile 失败时明确退出，不回退 eager。
- **P2：VAE 与 JPEG 流水。** `--vae-batch-size` 将最终 latent 分成 microbatch 解码。2 个 JPEG worker 和有界队列与下一批 GPU 工作重叠；JPEG 经临时文件写入、RGB/分辨率校验后原子重命名，后台异常必须回传主进程。VAE microbatch、worker 数和队列长度都受机器显存/CPU 约束。

新增入口示例：

```bash
# eager batch 对照
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all \
  --inference-engine eager --batch-size 16 --vae-batch-size 4 \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_eager_b16

# 最终推荐：默认 Inductor，明确关闭 CUDA Graph
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all \
  --inference-engine compiled --compile-mode default --batch-size 16 --vae-batch-size 4 \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_default_b16
bash scripts/run_csgo_seen10.sh eval --seed 0 --task all \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_default_b16

# 已修复且兼容旧命令：reduce-overhead + CUDA Graph
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all \
  --inference-engine compiled --compile-mode reduce-overhead --batch-size 16 --vae-batch-size 4 \
  --output-root outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0_compiled_b16
```

正式推理前可先运行 `--benchmark-batches N`：要求单 task、`WORLD_SIZE=1`、显式且不同于配置正式输出根的 `--output-root`、非 smoke、原正式 28 steps 和空白 task 输出目录；按 manifest 顺序只生成前 `N*batch_size` 个样本，即 N 个完整 batch。该模式写入 `benchmark_only` marker 和 `seed_<N>/<task>/benchmark_report.json`，不写正式 provenance，结果不用于 evaluator。分别测试两个 compile mode 时必须使用不同且新建的 output root。例如：

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

报告包含模型初始化秒数、每个 batch 的耗时与 img/s（首批标记 cold；compiled 首批包含 lazy compile）、总耗时/吞吐、compile mode、CUDAGraph 开关、DiT/VAE batch、steps 和 CUDA peak allocated/reserved。只有 benchmark 会为每 batch 显式 CUDA synchronize，普通正式推理不增加该同步。

### 2026-09-21 共卡实测与最终选择

RTX PRO 6000 Blackwell 上保持当前 eager 及其他 GPU 作业运行，使用同一 checkpoint、discrete 前 48 条、28 个采样点、DiT batch16、VAE batch4。eager 对照取 compile 测试前 11 个连续 batch 间隔的中位数；测试结束后两个间隔恢复为 61.68/61.58 秒，确认 eager 未停止且基线稳定。

| 方案 | 首批（含 lazy compile） | 稳态 batch | 稳态 img/s | 相对 eager | CUDA peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager batch16 | 不适用 | 61.60 秒（中位数） | 0.260 | 1.00× | 运行中未重置统计 |
| default/fullgraph，无 CUDA Graph | 71.98 秒 | 35.82 秒（后两批均值） | 0.447 | 1.72× | 9.67 GiB |
| reduce-overhead/fullgraph，CUDA Graph | 48.31 秒 | 36.24 秒（后两批均值） | 0.442 | 1.70× | 10.73 GiB |

两个 compile 方案均连续完成 3 个 batch/48 张有效 JPEG；`reduce-overhead` 未再出现 output overwrite。两者稳态在共卡噪声下可视为持平，default 本次快约 1.2%，且少 reserve 1.07 GiB。首批因顺序执行和共享 Inductor cache 不做横向判断。48 张解码图的 default/reduce PSNR 为 47.56 dB，两者相对 eager 约 44.9 dB；不同 kernel 路径存在小数值差异，因此 JPEG 不逐字节相同。

32,800 张共 2,050 个 batch，按当前稳态线性外推，default compile 约 20.4 小时，eager 约 35.1 小时；该数字不含初始化/冷编译，且共卡负载会变化，不是完成时间承诺。ControlAR 的约 9 小时仍是另一模型的测量。最终运行方案选择显式 `--compile-mode default`：当前没有观察到 CUDA Graph 稳态收益，同时降低显存和生命周期风险；修复后的 reduce-overhead 保留作兼容及后续独占 GPU 复测。

普通 marker 与 provenance row 保存 engine、DiT/VAE batch、compile mode、sampler、step 数、time shift、seed policy、JPEG quality 与 checkpoint hash；compiled 另记录 cudagraphs。默认 eager 字段保持兼容。省略 `--compile-mode` 仍选 reduce-overhead 以兼容旧命令，但正式推荐命令显式写 `default`。`--output-root` 只重定向预测与评测输出，并作为 seed 目录的父目录；未显式传 `--checkpoint` 时仍从配置原目录的 `seed_<N>/checkpoints/best.pt` 加载 adapter。eval 会对同一输出路径写评测结果。
