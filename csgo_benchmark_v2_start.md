你负责将当前已 clone 的独立模型项目接入 CSGO Benchmark v2 Seen-10。

【项目参数】
MODEL_NAME=Lumina-Image-2.0
MODEL_TYPE=GENERATION
PROJECT_ROOT=/home/jiahao/task/Lumina-Image-2.0
UNILIP_ROOT=/home/jiahao/task/UniLIP
DATA_ROOT=/home/jiahao/task/UniLIP/data/csgo_benchmark_v2
UNILIP_PYTHON=/home/jiahao/miniconda3/envs/UniLIP/bin/python
BUILD_SHARED_EVALUATOR=0
SHARED_EVAL_DIR=/home/jiahao/task/csgo_benchmark_v2_eval_general
RUN_FULL=0
TRAIN_SEEDS="0"  # 只需要单种子结果即可

Table 1 的任务分工如下：

- VLA：X-VLA、RDT、OpenVLA-OFT、pi0.5，只接入 localization。
- GENERATION：OmniGen、ControlAR、Lumina-Image-2.0、Show-o2-1.5B、Puffin、Janus-Pro-1B，接入 discrete generation 和 continuous generation。
- 不要为了补齐三列而给模型实现其在 Table 1 中不承担的任务。这里的“三个任务”是 Table 1 整体的 localization、discrete generation、continuous generation。

目标不是做大规模框架重构或公平性审计，而是以最小改动打通当前模型在 Seen-10 上的训练、推理和统一评测，并得到可填入 Table 1 的结果。不要停留在方案阶段，也不要反复请求审批。

一、阅读与方案

1. 完整阅读当前项目的 README、环境文件、训练入口、推理入口、dataset/dataloader、processor、模型构造、checkpoint 保存/加载和配置系统。
2. 找出最适合复用的原生训练与推理路径，优先保留项目原有 Trainer、optimizer、distributed launcher、LoRA/finetune 方式和 checkpoint 格式。
3. 阅读以下 UniLIP 参考文件：
   - ${UNILIP_ROOT}/AGENT.md
   - ${UNILIP_ROOT}/CSGO_BENCHMARK_V2_NEW_SERVER_MIGRATION_CHECKLIST.md
   - ${UNILIP_ROOT}/CSGO_BENCHMARK_METRICS_ZH.md
   - ${UNILIP_ROOT}/benchmark_csgo_v1.py
   - ${UNILIP_ROOT}/benchmark_csgo_v1_conti.py
   - ${UNILIP_ROOT}/eval_csgo_loc.py
   - ${UNILIP_ROOT}/scripts/aggregate_csgo_benchmark_v2_metrics.py
   - ${UNILIP_ROOT}/csgo_configs/benchmark_v2.yaml
4. 先简要列出需要修改/新增的文件、数据接入点、模型输入输出适配方式和运行命令，给出变更方案并保存在当前项目根目录下。
5. 然后根据变更方案直接实施，不等待确认。
5. 只改当前独立项目以及 SHARED_EVAL_DIR；不要修改、移动或复制 DATA_ROOT 中的数据。

二、Seen-10 数据合同

必须由以下文件驱动数据读取，不能扫描 images/ 后自行重新划分：

- ${DATA_ROOT}/minimal_dataset_report.json
- ${DATA_ROOT}/benchmark_manifest.json
- ${DATA_ROOT}/splits/
- ${DATA_ROOT}/images/<map>/<frame>.jpg
- ${DATA_ROOT}/radars/
- ${DATA_ROOT}/calibration/

Seen-10 地图固定顺序：

cs_agency
cs_italy
de_ancient
de_anubis
de_dust2
de_inferno
de_mirage
de_nuke
de_overpass
de_train

数据量：

- seen_train：每地图5,000，共50,000。
- seen_validation：每地图500，共5,000。
- seen_discrete_test：每地图2,000，共20,000。
- seen_continuous：每地图20个clip，每个64帧；共200个clip、12,800帧。

使用 benchmark manifest/split row 给出的 sample ID、图像和 radar 映射。不要把 DATA_ROOT/images 直接当成旧式 data_dir，也不要依赖
<data_dir>/<map>/imgs/<frame>.jpg。

三、任务适配

如果 MODEL_TYPE=VLA：

1. 输入为当前第一视角图像、对应 radar/map 图像以及固定任务 instruction/map name。
2. 机器人 proprio/state 不适用于该任务，设为 zero/masked，禁止填入 GT pose。
3. 将动作输出以最小改动替换或适配为 horizon=1 的绝对5DoF：
   [x, y, z, pitch, yaw]。
4. 标签和预测统一使用 Benchmark v2 的归一化定义；Z 范围必须读取发布 calibration/manifest，不得从 test 重新估计。
5. 尽量复用原模型的视觉、语言和 action head；仅新增必要的双图输入适配、5D head或投影层。
6. 使用 seen_train 训练、seen_validation 选 checkpoint、seen_discrete_test 推理。
7. 输出：
        outputs/csgo_benchmark_v2_seen10/${MODEL_NAME}/seed_<seed>/localization/predictions.jsonl
   每行至少包含：
        sample_id、map_name、pred_x、pred_y、pred_z、pred_pitch、pred_yaw。
   不要在模型输入文件中放置 GT pose。
9. 训练阶段eval interval和checkpoint save interval设置为总训练steps的 1/5；
   即训练阶段只eval和save五次，且使用late和best链接到checkpoints的最后一次保存结果和最优保存结果。
10. 训练结束后根据训练日志的主loss绘制loss曲线图。
11. 在训练时的eval和推理时增加样本可视化功能，可视化功能的主要行为有：
    - 每张地图固定随机选择 10 个样本，保证各次 eval 可横向比较。
    - Radar 上：
        - GT：同色实心圆。
        - Prediction：同色大号空心圆。
        - GT 与预测之间绘制连线。
        - 越界预测贴边显示。
    - 右侧 10 张 FPV 竖排，左上角显示对应颜色圆点。
    - FPV 顶部居中显示：
        - gt_xyzhw
        - pred_xyzhw
    - 数值使用物理坐标，xyzhw = x,y,z,pitch,yaw，角度单位为度。

如果 MODEL_TYPE=GENERATION：

1. 输入为对应 radar/map 图像和数值5DoF pose，输出对应的448×448 RGB第一视角图像。
2. 优先使用项目已有的 image condition、control image、multimodal或camera condition接口。
3. 如果原模型只有文本条件，以最小方式增加 radar encoder/adapter 和数值 pose 投影；pose 至少通过 numeric tokens、MLP、FiLM或等价数值条件接入，不能只把坐标拼成自然语言。
4. 使用 seen_train 训练、seen_validation 选 checkpoint。
5. 同一个冻结 checkpoint 分别推理：
   - seen_discrete_test
   - seen_continuous
6. continuous 按 manifest 中的 clip_id 和 frame 顺序逐帧生成，不允许读取目标帧、历史/未来GT帧。
7. 输出文件名必须保持 manifest 中的 sample/frame identity：
   outputs/csgo_benchmark_v2_seen10/${MODEL_NAME}/seed_<seed>/discrete/gen_imgs/<map>/<frame>.jpg
   outputs/csgo_benchmark_v2_seen10/${MODEL_NAME}/seed_<seed>/continuous/gen_imgs/<map>/<frame>.jpg
8. 保存图像时复用 UniLIP 当前输出尺寸、RGB转换和编码方式。
9. 训练结束后根据训练日志的主loss绘制loss曲线图。
10. 训练阶段eval interval和checkpoint save interval设置为总训练steps的 1/5；
   即训练阶段只eval和save五次，且使用late和best链接到checkpoints的最后一次保存结果和最优保存结果。

不要求不同模型使用完全相同的 optimizer、native resolution、LoRA策略或可训练参数量。优先选择当前项目最稳定、改动最小的官方训练路径，但数据 split、任务输入输出和最终 metric 必须一致。

四、需要落地的最小文件

在当前项目内新增尽可能少的内容，例如：

- 一个 manifest-driven Seen-10 dataset adapter。
- 一个模型任务适配模块。
- 一份 localization 或 generation 配置。
- train_seen10.py 或原训练入口的轻量 wrapper。
- infer_seen10.py 或原推理入口的轻量 wrapper。
- scripts/run_csgo_seen10.sh，支持 train、infer、eval 和 --seed。
- 简短的 CSGO_SEEN10.md，只记录环境、实际命令和输出路径。

不要复制完整 UniLIP 训练代码，不要重写当前项目已有的训练框架。

五、首个项目建立一次通用评测器

当 BUILD_SHARED_EVALUATOR=1 时，在 SHARED_EVAL_DIR 中实现一个与模型项目无关的评测目录。后续项目只需手动同步整个目录和配置，不再重复开发。

通用评测器应：

1. 只接收标准化的 localization JSONL 或 generation gen_imgs 目录，不 import 任何外部模型代码。
2. 从 DATA_ROOT 的 manifest、splits 和 calibration join GT。
3. 参考并复用 UniLIP 当前 metric 的算法和默认参数，只保留 Table 1 必需指标：
   - Localization：XY_Dist、Z_Dist、Pitch_Dist、Yaw_Dist。
   - Discrete：PSNR、SSIM、LPIPS、Boundary_F1、FID。
   - Continuous：PSNR、SSIM、LPIPS、TWE、TDE、FVD。
4. yaw 使用正确的最短圆周距离：
   abs(((pred - gt + period/2) % period) - period/2)
   不要照搬可能在预测越界时产生负误差的旧 min(d, period-d) 写法。
5. 连续评测固定：
   clip_length=16、clip_stride=16、fvd_size=224、
   frame_diff_threshold=2、min_track_len=4。
6. 每地图分别计算，再按上述固定地图顺序输出 equal-map macro。
7. 可以移除 Table 1 不使用的 external locator、IS、CLIP和Aesthetic，减少依赖，但不能改变保留指标的计算实现。
8. 至少提供以下统一入口：
   python run_eval.py localization --pred-root ... --data-root ... --output ...
   python run_eval.py discrete --pred-root ... --data-root ... --output ...
   python run_eval.py continuous --pred-root ... --data-root ... --output ...
9. 默认使用 UNILIP_PYTHON 运行 metric，并提供唯一一份 requirements/config。
10. 输出 per-map JSON 和 summary_equal_map.json。

当 BUILD_SHARED_EVALUATOR=0 时：

- 不重新实现评测器。
- 尽量减少修改通用评测器，如必需修改则说明并记录修改理由和内容。
- 使用已手动同步到 SHARED_EVAL_DIR 的通用代码。
- 只保证当前模型的预测满足上述标准输出格式，然后直接运行统一评测命令。
- 如果为了当前模型的预测满足上述标准输出格式而修改了 SHARED_EVAL_DIR 的通用代码，
  需要确保修改后的代码兼容之前所有 MODEL_TYPE=VLA 和 MODEL_TYPE=GENERATION 的模型预测输出格式。

六、执行要求

1. 使用当前模型自己的独立环境训练和推理；不要把它的依赖安装进 UniLIP conda 环境。
2. 先完成最小 smoke：
   - dataset能读取一个batch；
   - 模型能执行一次forward/backward；
   - 能加载保存的checkpoint；
   - 能生成一个标准预测文件或一张448×448图片；
   - 通用评测器能读取该输出。
3. smoke 失败时直接定位并修复，不能只报告问题。
4. RUN_FULL=1 时，在 smoke 后直接按 TRAIN_SEEDS 依次执行完整训练、验证、推理和评测。
5. 默认先跑 seed=0 得到完整结果；但脚本必须支持以后不改代码直接追加 seed=1、2。
6. 不进行额外审批、许可证调研、大规模数据审计、无关重构、参数量统计、资源公平性分析或长篇实验总结。
7. 不使用 incomplete/debug coverage 生成正式结果；缺文件时应补推理对应样本。
8. 不覆盖已有输出。每个 seed 使用独立目录，训练应支持从项目原生 checkpoint 恢复。

最终只需返回：

- 实际修改/新增的文件。
- 环境创建与运行命令。
- train/infer/eval 的直接命令。
- checkpoint 和结果路径。
- 若 RUN_FULL=1，给出三项任务中该模型负责部分的 per-map 与 equal-map 结果。
- 尚未完成的唯一真实阻塞项。

不要只给建议或伪代码；需要把当前项目改到能够实际运行。