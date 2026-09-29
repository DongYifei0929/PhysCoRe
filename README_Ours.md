# PhysCoRe：本地补充文档

本仓库的上游使用说明保留在 `README.md`，未作修改。

## 新增文档

- `PhysCoRe_论文代码讲解.md`：基于官方仓库提交 `ba7ddf7` 和论文 v2 的架构级代码导读。重点解释 MfM、MLS-MPM、RfD 的数据流、张量形状和调用关系，并记录当前代码中的硬编码、无效配置、默认隐藏行为、未发布功能与复现边界。

该文档是代码阅读结果，不改变训练、验证或推理行为。

## process from scratch

下面的流程从一组**已经从 3 台 RealSense 导出为同步 RGB-D 序列**的视频开始，
使用仓库发布的 MfM/RfD checkpoint 完成处理和推理。当前仓库没有把 RealSense
`.bag` 文件直接转换为输入目录的脚本；需要先用 RealSense SDK 或自己的导出程序
完成时间同步、彩色/深度对齐和相机标定。

### 0. 准备输入目录

仓库固定读取 3 个视角，目录必须整理成：

```text
$RAW_ROOT/$CASE/
├── calibrate.pkl             # 长度为 3 的 camera-to-world 4x4 矩阵列表
├── metadata.json             # intrinsics、WH、fps、frame_num 等
├── color/
│   ├── 0.mp4                 # 第 0 个视角的 RGB 视频
│   ├── 1.mp4
│   ├── 2.mp4
│   ├── 0/000000.png          # 与视频逐帧对应的 RGB PNG
│   ├── 1/000000.png
│   └── 2/000000.png
└── depth/
    ├── 0/000000.npy          # uint16 深度，单位为毫米
    ├── 1/000000.npy
    └── 2/000000.npy
```

三路数据需要帧数、帧率和时间起点一致。`metadata.json` 中的内参必须对应
RGB 图像分辨率；`calibrate.pkl` 必须使用代码期望的 camera-to-world 变换。
如果只有 PNG 而没有 MP4，可以先用 FFmpeg 生成视频，例如：

```bash
ffmpeg -y -framerate 30 \
  -i "$RAW_ROOT/$CASE/color/0/%06d.png" \
  -c:v libx264 -pix_fmt yuv420p "$RAW_ROOT/$CASE/color/0.mp4"
```

对视角 `1`、`2` 重复执行。深度 `.npy` 会在 3D 处理阶段除以 `1000`，因此不要
把单位为米的浮点深度直接当作毫米保存。

### 1. 环境和路径

```bash
cd /nvme0/zhangruiying/PhysCoRe
conda activate /nvme0/zhangruiying/anaconda3/envs/physcore

export CASE=my_realsense_case
export RAW_ROOT=/path/to/raw_cases
export EPISODE_ROOT=/path/to/physcore_episodes
export OUTPUT_ROOT=/path/to/physcore_outputs

# 目标物体的 GroundingDINO 文本提示，以及控制器类别。
export CATEGORY="yellow cloth"
export CONTROLLER="hand"             # 或 "robot gripper"

# 目录内必须包含这三个文件：
# GroundingDINO_SwinT_OGC.py
# groundingdino_swint_ogc.pth
# sam2.1_hiera_large.pt
export CHECKPOINT_DIR=$PWD/checkpoints
mkdir -p "$EPISODE_ROOT/$CASE" "$OUTPUT_ROOT"
```

如果尚未安装 2D 依赖和下载 GroundingDINO/SAM2 权重，先按上游
`README.md` 的 “Environment setup” 和 “Install GroundingDINO and SAM 2
dependencies” 执行；发布数据虽然跳过 2D 阶段，但自采 RealSense 数据不能跳过。

### 2. 2D segmentation、tracking 和 3D lifting

这个入口会依次运行：GroundingDINO+SAM2 分割、CoTracker 稠密追踪、深度反投影、
mask 后处理、物体/控制器追踪以及采样轨迹生成。

```bash
python datagen/process2d/process_data.py \
  --base_path "$RAW_ROOT" \
  --case_name "$CASE" \
  --category "$CATEGORY" \
  --controller "$CONTROLLER" \
  --headless
```

完成后重点检查以下文件：

```text
$RAW_ROOT/$CASE/mask/
$RAW_ROOT/$CASE/pcd/
$RAW_ROOT/$CASE/cotracker/
$RAW_ROOT/$CASE/sampled_tracks.pkl
```

如果 `mask_info_*.json` 中没有目标物体或控制器标签，先调整
`CATEGORY`/`CONTROLLER` 或检查第一帧的 GroundingDINO 检测结果，再继续后面的步骤。

### 3. 3D episode construction

根据物体类型选择粒子分辨率。绳子、塑料泥等默认使用 8 mm；薄布使用 4 mm
并限制内部填充距离。

```bash
# rope / plasticine / 一般物体
python datagen/convert3d/convert_to_episode.py \
  --source_dir "$RAW_ROOT/$CASE" \
  --output_dir "$EPISODE_ROOT/$CASE" \
  --controller_mask_label "$CONTROLLER" \
  --particle_voxel_size 0.008 \
  --target_total_particles 8000 \
  --mask_erode_pixels 2 \
  --dpsr_device cuda
```

布料可以改用：

```bash
python datagen/convert3d/convert_to_episode.py \
  --source_dir "$RAW_ROOT/$CASE" \
  --output_dir "$EPISODE_ROOT/$CASE" \
  --controller_mask_label "$CONTROLLER" \
  --particle_voxel_size 0.004 \
  --target_total_particles 16000 \
  --mask_erode_pixels 2 \
  --interior_to_shell_max_distance 0.004 \
  --dpsr_device cuda
```

输出应包含：

```text
$EPISODE_ROOT/$CASE/episode_0000/episode_data.pt
```

### 4. MfM inference

MfM 在观测窗口上估计逐粒子的 `log_E`、`nu` 和 confidence，并运行 MPM
rollout。`--out` 非空且 `save_rollout` 开启时，会额外保存后续渲染需要的
trajectory。

```bash
mkdir -p "$OUTPUT_ROOT/MfM/$CASE"

python validate_MfM.py \
  --config data/configs/validate_MfM.yaml \
  --checkpoint data/checkpoints/MfM_checkpoint.pt \
  --root "$EPISODE_ROOT/$CASE/episode_0000" \
  --out "$OUTPUT_ROOT/MfM/$CASE/validation.json" \
  --save-rollout
```

主要输出包括：

```text
$OUTPUT_ROOT/MfM/$CASE/validation.json
$OUTPUT_ROOT/MfM/$CASE/validation_episode_0000_trajectory.pt
```

如果要做固定材料先验消融，可在验证命令中同时传入 `--prior-log-E` 和
`--prior-nu`。这两个值会在每个 MfM 更新后覆盖所有粒子的材料，并实际用于
后续 MPM rollout；取值必须分别落在 `model.log_E_min/max` 和
`model.nu_min/max` 范围内。例如：

```bash
python validate_MfM.py \
  --config data/configs/validate_MfM.yaml \
  --checkpoint data/checkpoints/MfM_checkpoint.pt \
  --root "$EPISODE_ROOT/$CASE/episode_0000" \
  --out "$OUTPUT_ROOT/MfM/$CASE/prior_validation.json" \
  --prior-log-E 9.0 \
  --prior-nu 0.30
```

该覆盖发生在 MfM 输出端，而不是只修改初始 `material_guess`；因此不会在下一
个观测窗口被网络预测值覆盖。未传入这两个参数时，原有 MfM 推理行为不变。

### 5. MfM + MPM + RfD inference

`validate_RfD.py` 只扫描 `checkpoints/gvc_epoch_*.pt`，所以先为发布的
`RfD_checkpoint.pt` 建立兼容文件名：

```bash
export RFD_RUN=$OUTPUT_ROOT/RfD/$CASE
mkdir -p "$RFD_RUN/checkpoints"
ln -sfn "$PWD/data/checkpoints/RfD_checkpoint.pt" \
  "$RFD_RUN/checkpoints/gvc_epoch_0001.pt"

python validate_RfD.py \
  --config data/configs/validate_RfD.yaml \
  --refiner-checkpoint data/checkpoints/MfM_checkpoint.pt \
  --epoch 1 \
  --save-trajectory \
  "dataset.validation_roots=[$EPISODE_ROOT/$CASE/episode_0000]" \
  "train.output_dir=$RFD_RUN"
```

加入 `--save-trajectory` 后，每个 checkpoint 和 episode 会额外保存：

```text
$RFD_RUN/trajectories/epoch_0001/<episode_label>_trajectory.pt
```

文件包含 `predicted`、`predicted_frames`、`use_gvc`、`correction_mode`
和 `checkpoint_epoch`。轨迹覆盖帧 0 到最后一个完整 `update_every` 窗口；
尾部不足一个窗口的帧与验证指标一样不参与 rollout。

预测视频通常位于：

```text
$RFD_RUN/videos/val/epoch_0001/
```

加入 `--skip-render` 可以只计算指标而不生成视频。该推理流程使用 episode 中
已有的控制器轨迹作为未来控制条件；RfD 预测的是 MPM 网格速度残差，不会额外
预测未知的机器人动作或控制器外力。

以上命令使用的是仓库发布权重，不需要运行数据增强、`train_MfM.py` 或
`train_RfD.py`。如果要用自采数据重新训练模型，还需要先把多个 episode 加入
训练配置，再运行 `datagen/augment/run_recipe.py`、`train_MfM.py` 和
`train_RfD.py`；这些训练配置中的数据路径和输出路径不能直接使用占位符。

## 官方数据到发布权重推理（本机实测）

当前 Hugging Face 数据仓库下载在项目的 `data/` 子目录，因此真实路径比
上游 README 多一层：

```text
data/data/physcore/different_types/<case>/
data/checkpoints/{MfM_checkpoint.pt,RfD_checkpoint.pt}
data/configs/{validate_MfM.yaml,validate_RfD.yaml}
```

发布数据已经包含 `mask/` 和 `sampled_tracks.pkl`，不要重复运行 2D 分割与追踪；
直接从 3D episode 转换开始。输出 episode 很大（实测 `single_lift_rope` 为
434 MB），应把 `EPISODE_ROOT` 放到有足够空间的持久存储，而不是会被清理的
`/tmp`。当前项目内的 `data` 是指向 `/data5/public_data/PhysCoRe/data` 的软链接，
以下相对路径仍然有效；生成的 episode 和输出不应写回这个公共数据目录。

```bash
cd /nvme0/zhangruiying/PhysCoRe
conda activate /nvme0/zhangruiying/anaconda3/envs/physcore

export CASE=single_lift_cloth
export RAW_ROOT=$PWD/data/data/physcore/different_types
export EPISODE_ROOT=/data5/public_data/PhysCoRe/medium
export OUTPUT_ROOT=/data5/public_data/PhysCoRe/result

# 注意不同材料不同参数！
python datagen/convert3d/convert_to_episode.py \
  --source_dir "$RAW_ROOT/$CASE" \
  --output_dir "$EPISODE_ROOT/$CASE" \
  --controller_mask_label hand \
  --particle_voxel_size 0.004 \
  --target_total_particles 16000 \
  --mask_erode_pixels 2 \
  --interior_to_shell_max_distance 0.004

python validate_MfM.py \
  --config data/configs/validate_MfM.yaml \
  --checkpoint data/checkpoints/MfM_checkpoint.pt \
  --root "$EPISODE_ROOT/$CASE/episode_0000" \
  --out "$OUTPUT_ROOT/MfM/$CASE/validation.json"
```

`tools/pipeline.sh` 要求将 `CASE` 作为第一个参数传入，例如
`bash tools/pipeline.sh single_lift_cloth`；未传参数时会显示用法并退出。
脚本会按 `CASE` 名称自动选择转换参数：名称中包含
`cloth` 时使用 4 mm、16000 粒子上限和 4 mm 内部到表面距离；
其他 case 统一使用 8 mm、12000 粒子上限。两个分支均使用 2 像素
mask erosion。这是该 pipeline 的两档策略；上游 README 仍保留下述
rope/plasticine 的 8000 粒子三档预设。

`validate_RfD.py` 只扫描训练输出式的 `checkpoints/gvc_epoch_*.pt`，而发布文件名
是 `RfD_checkpoint.pt`。先将其复制或软链接成该名字，再执行验证即为
MfM + MPM + RfD 推理：

```bash
export RFD_RUN=$OUTPUT_ROOT/RfD/released
mkdir -p "$RFD_RUN/checkpoints"
cp data/checkpoints/RfD_checkpoint.pt "$RFD_RUN/checkpoints/gvc_epoch_0001.pt"

python validate_RfD.py \
  --config data/configs/validate_RfD.yaml \
  --refiner-checkpoint data/checkpoints/MfM_checkpoint.pt \
  --epoch 1 \
  "dataset.validation_roots=[$EPISODE_ROOT/$CASE/episode_0000]" \
  "train.output_dir=$RFD_RUN"
```

结果在 `$RFD_RUN/videos/val/epoch_0001/`；只计算指标、不渲染视频时增加
`--skip-render`。批量转换时按上游预设选择参数：rope/plasticine 用 8 mm、8000；
bear 用 8 mm、12000；cloth 用 4 mm、16000，并设置
`--interior_to_shell_max_distance 0.004`。这些数量是粒子上限，过滤后的实际数量
可以更少。

2026-09-24 冒烟验证使用 `single_lift_rope`：3D 转换得到 1902 个粒子；MfM
验证 `avg_recon_loss=0.00610338`；RfD 验证 `val_loss=0.0220189`，并成功生成
75 帧 H.264 视频。后续已补装 `simple_knn`，并完成下面的 3DGS 渲染验证。

## 3DGS Appearance 模式

`render_MfM_confidence_3dgs.py` 现在支持两种兼容模式：默认的 `confidence`
用色图替换 Gaussian 外观；`appearance` 保留发布 3DGS 的原始 SH 颜色，只按
保存的 MPM 轨迹更新 Gaussian 位置和旋转。Appearance 不加载 MfM checkpoint，
但仍需要 `validate_MfM.py --save-rollout` 产生的 trajectory。

```bash
export CASE=single_lift_rope
export GS_RUN='init=pcd_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

python render_MfM_confidence_3dgs.py \
  --config configs/render_MfM_confidence_3dgs.yaml \
  --mode appearance \
  --episode-root "$EPISODE_ROOT/$CASE/episode_0000" \
  --gs-dir "$PWD/data/gaussian_output/physcore/$CASE/$GS_RUN" \
  --traj-path "$OUTPUT_ROOT/MfM/$CASE/validation_episode_0000_trajectory.pt" \
  --out-dir "$OUTPUT_ROOT/MfM_appearance/$CASE" \
  --name "$CASE"
```

每个相机输出 `<case>_<cam>_appearance.mp4`；启用默认的
`overlay_on_rgb: true` 时还输出 `_appearance_overlay.mp4`。逐帧 PNG 分别位于
`appearance/<camera>/` 和 `overlay/appearance/<camera>/`。独立 appearance
视频以黑色为背景，overlay 使用原始 RGB 作为真实背景。

渲染器的 `rasterizer: auto` 优先使用 gsplat，从而读取真实相机主点并直接返回
alpha；gsplat 不可用时回退到 `diff_gaussian_rasterization`，后者以额外白色 pass
重建 alpha，但只能使用居中主点。脚本会自动发现 Conda 环境内的 CUDA toolkit，
设置 `MAX_JOBS`（默认 1）、系统 GCC/G++ 和当前 GPU 架构，并优先直接加载已有的
PyTorch extension cache，避免 gsplat 1.4 重复 JIT。

本机已用 RTX PRO 6000、CUDA 12.8 和 `single_lift_rope` 实测 gsplat：8664 个
Gaussian、3 个真实相机、原始 SH appearance、RGB overlay、MP4 和 PNG 均成功
输出；原 confidence 模式和 diff fallback 也分别完成回归测试。当前形变只更新
Gaussian 的中心和旋转，不更新 scale，也不会把 SH 方向随局部旋转重定向；它适合
发布案例的外观回放，但不是完整的可重光照材质模型。

## sampled_tracks 三维追踪可视化

`sampled_tracks.pkl` 不是原始二维 CoTracker 输出，而是源视频经过分割、
稠密二维追踪、RGB-D 反投影、多相机合并、可见性/运动过滤和体素降采样后
得到的三维轨迹。新增 `tools/visualize_sampled_tracks.py` 可将物体轨迹和红色
控制器点导出为 MP4/GIF。

```bash
conda activate /nvme0/zhangruiying/anaconda3/envs/physcore
python tools/visualize_sampled_tracks.py
```

默认读取 `single_lift_cloth/sampled_tracks.pkl`，并写入：

```text
outputs/track_visualization/single_lift_cloth_sampled_tracks.mp4
```

默认使用根据第 0 帧 Y 坐标生成的稳定轨迹颜色，同一个点在所有帧中
颜色不变；`--color-mode rgb` 则显示从源 RGB 视频采样的颜色。脚本默认将
原始 Z 方向翻转为 Z-up 仅供显示，`--no-flip-z` 可保留源坐标。自定义例子：

```bash
python tools/visualize_sampled_tracks.py \
  --input data/data/physcore/different_types/single_lift_cloth/sampled_tracks.pkl \
  --output /tmp/single_lift_cloth_tracks.mp4 \
  --color-mode rgb \
  --azimuth 90 \
  --elevation 20 \
  --point-radius 2
```

`--frame-step` 可跳帧，`--max-frames` 可用于快速冒烟测试。当前
`single_lift_cloth` 文件包含 164 帧、3250 条物体轨迹和 30 条控制器轨迹。

## 粒子轨迹动态可视化

新增 `tools/visualize_particle_trajectory.py`，用于将 validation trajectory
中的 `predicted` 和 `ground_truth` 粒子轨迹导出为左右对比视频。脚本会使用
`predicted_frames` 将预测帧与完整真值帧对齐；当前 `single_lift_cloth` 的
`predicted` 是 `[161, 11820, 3]`，真值是 `[164, 11820, 3]`，因此不会直接按数组
下标截断真值。

```bash
conda activate /nvme0/zhangruiying/anaconda3/envs/physcore
python tools/visualize_particle_trajectory.py
```

默认读取：

```text
/data5/public_data/PhysCoRe/result/MfM/single_lift_cloth/
validation_episode_0000_trajectory.pt
```

默认输出：

```text
outputs/trajectory_visualization/
validation_episode_0000_trajectory_predicted_vs_ground_truth.mp4
```

也可以指定输入、输出、帧率、分辨率和视角：

```bash
python tools/visualize_particle_trajectory.py \
  --input /data5/public_data/PhysCoRe/result/MfM/single_lift_cloth/validation_episode_0000_trajectory.pt \
  --output /tmp/single_lift_cloth_particles.mp4 \
  --fps 30 \
  --width 480 \
  --height 360 \
  --point-radius 1 \
  --azimuth 45 \
  --elevation 25
```

画面左侧为蓝色 predicted，右侧为红色 ground truth。脚本依赖项目环境中的
PyTorch、NumPy、Pillow、imageio/imageio-ffmpeg。

## 多操作 MfM → 全程 RfD pipeline

新增 `tools/run_multiaction_rfd_pipeline.py`，用于验证以下两阶段方案：先让冻结的
MfM 读取识别 episode 的所有完整窗口并保持 recurrent state，取最后一个窗口的
逐粒子 `log_E`、`nu` 和 confidence；再冻结该材料场，从目标操作的第一个仿真
substep 起开启 RfD，不使用前半段 PID/observation correction。

如果多种已知操作和陌生操作属于一条**连续拍摄且连续追踪**的 episode，可在同一
文件中指定分界帧。直接拼接独立 MP4 并不够：RGB、深度、三路相机时间轴、mask、
控制器点和粒子 ID 都必须同步连续，否则应分别转换 episode。

```bash
python tools/run_multiaction_rfd_pipeline.py \
  --identify-root /path/to/combined/episode_0000 \
  --target-root /path/to/combined/episode_0000 \
  --identify-end-frame 80 \
  --target-start-frame 80 \
  --transfer-mode direct \
  --output-dir outputs/multiaction_rfd/continuous \
  --render-video
```

若识别和目标是分别转换的 episode，二者通常没有逐粒子对应。默认
`--transfer-mode auto` 会改用 confidence 加权的全局 `log_E/nu`；只有两份 canonical
粒子几何已经对齐时才应显式使用 `--transfer-mode nearest`。`nearest` 只做质心平移
对齐，不估计旋转或非刚性配准。目标 episode 仍须提供完整的未来控制器轨迹；脚本
不会预测控制器动作或真实外力。

输出目录包含 `trajectory.pt` 和 `report.json`；加 `--render-video` 会额外生成
`predicted_vs_ground_truth.mp4`。`--deactivate-rfd` 使用同一 MfM 材料场运行 MPM-only
消融，`--max-identify-windows` 和 `--max-target-windows` 可用于短冒烟测试。

本机在 `single_lift_cloth`（11,820 粒子）上完成了代码路径验证：MfM 使用帧 1–160
的 16 个窗口，目标 rollout 为帧 161–163。RfD 的 Chamfer / tracked L2 分别为
`0.007115 / 0.018933`，同条件 MPM-only 为 `0.008550 / 0.023606`，即这个 3 帧样例中
分别下降约 16.8% / 19.8%。结果保存在
`outputs/multiaction_rfd/full_identification_smoke/`。该测试仍沿用同一个 lift 控制器，
不能证明对陌生控制器的泛化；正式结论至少需要按物体划分 familiar-controller ID 段
和 held-out-controller target 段，并在多条长 rollout 上比较 MPM-only、现有半程 RfD
和全程 RfD。
