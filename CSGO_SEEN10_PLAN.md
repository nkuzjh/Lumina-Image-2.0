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
