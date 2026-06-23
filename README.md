# FastUMI Retarget Tool

UMI 采集数据 → 给定 URDF 下的全身关节轨迹数据集。

## 功能

- **Kinematic Retarget**: FastUMI SLAM 位姿 → pyroki 全局轨迹优化 → 全身关节角
- **质量评估**: 自动检测异常 + 多维度评分 → FAIL/PASS/GOOD 分级
- **数据集合并**: 质量筛选 + 统一 normalization + train/val 划分
- **Action-Pose Alignment** (可选): GBT 残差预测 + pyroki 全局平滑, 让真机在正确位置执行正确动作
- **本体设计评估**：来衡量本体设计能否完美实现whole-body-tast，参阅embodiment_evaluation_guide.txt

## 安装

```bash
pip install numpy scipy h5py pyyaml scikit-learn
pip install jax[cuda] jaxls pyroki yourdfpy
```

## 快速开始

```bash
# 一键处理
python3 -m fastumi_retarget auto \
  --urdf ./urdf/r1_pro.urdf \
  --input ./raw_umi_data/ \
  --output ./dataset/
```

## 分步使用

### Step 1: 配置

```bash
python3 -m fastumi_retarget init \
  --urdf ./urdf/r1_pro.urdf \
  --input ./raw_umi_data/
```

交互式选择每个子集的 mode (local/mobile) 和 hand (left/right/both)，生成 `config.yaml`。

### Step 2: 调参 (可选)

```bash
python3 -m fastumi_retarget tune --config config.yaml --preset balanced
```

### Step 3: 批量 Retarget

```bash
python3 -m fastumi_retarget batch \
  --config config.yaml \
  --input ./raw_umi_data/ \
  --output ./retargeted/
```

支持断点续传 (中断后重跑自动跳过已处理的 episode)，`--force` 强制覆盖。

### Step 4: Action-Pose Alignment (可选)

```bash
# 需要先有已训练的 APA 模型 (在 config.yaml 中配置 alignment.model_path)
python3 -m fastumi_retarget align \
  --config config.yaml \
  --input ./retargeted/
```

让真机 replay 时手臂位置和夹爪动作在空间上对齐。验证指标：轨迹重合度 + 夹爪-位姿匹配度。

### Step 5: 质量评估

```bash
python3 -m fastumi_retarget evaluate \
  --input ./retargeted/ \
  --config config.yaml
```

输出三级分类 (FAIL/PASS/GOOD) + 汇总报告。

### Step 6: 可视化 (开发中)

```bash
python3 -m fastumi_retarget visualize \
  --input ./retargeted/ \
  --show typical
```

### Step 7: 合并数据集

```bash
python3 -m fastumi_retarget merge \
  --input ./retargeted/ \
  --output ./final_dataset/ \
  --min-grade PASS
```

筛选 PASS 以上的数据，统一 normalization，划分 train/val。

## 输出格式

```
final_dataset/
├── local_standard/episode_01/dataset.hdf5
├── mobile_standard/episode_01/dataset.hdf5
├── ...
├── normalization.json       # 全量统一
├── train_episodes.txt       # 训练集 (相对路径)
├── val_episodes.txt         # 验证集
└── merge_report.txt         # 合并统计
```

每个 `dataset.hdf5`:

```
data/demo_0/
├── obs/joint_positions  (N, 20)  # 18关节 + 2夹爪
├── obs/eef_pose_left    (N, 7)
├── obs/base_position    (N, 3)
├── actions              (N, 20)
├── quality/
│   ├── keyframe_mask    (N,) bool
│   └── tracking_*       (N,) float
└── attrs: grade=GOOD/PASS/FAIL, quality_score=0.xx
```

## config.yaml

```yaml
urdf_path: ./urdf/r1_pro.urdf
preset: balanced
subsets:
  local_standard:  {mode: local, hand: left}
  mobile_standard: {mode: mobile, hand: left}
weights:
  pos_weight: 20
  ori_weight: 0.3
  kf_pos_weight: 200
  smooth_weight: 5
  collision_weight: 100
  # ... (完整参数见 config.py 的 PRESETS)
alignment:
  model_path: null   # 设为 ./calib/robot.pkl 启用 Action-Pose Alignment
evaluation:
  good_threshold: 0.7
```

## 依赖

| 包 | 用途 |
|---|---|
| numpy, scipy, h5py | 基础 |
| pyyaml | config.yaml |
| jax[cuda], jaxls | GPU 优化 |
| pyroki, yourdfpy | 轨迹优化 + URDF |
| scikit-learn | GBT 残差模型 (补偿用) |
