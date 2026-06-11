# Basketball Camera Calibration

基于 `2022-winners-camera-calibration-challenge` 中的 **KaliCalib** 方法，对篮球比赛图片和视频进行相机标定（单应矩阵估计 + 场地线可视化）。

---

## 目录结构

```text
basketball/
├── process_image.py                            # 单张图片推理脚本
├── process_image.sh                            # 示例调用脚本
├── process_video.py                            # 视频推理脚本（抽帧 + 汇总 JSON）
├── dino/                                       # ConvNeXt 升级模型（训练 + 定义）
│   ├── model_convnext.py                       #   KaliCalibConvNeXt 模型定义
│   ├── train.py                                #   微调训练脚本
│   ├── load_model.py                           #   backbone 下载辅助脚本
│   ├── checkpoints/                            #   训练产生的 checkpoint
│   ├── models/hub/                             #   timm 本地权重缓存
│   └── README.md                               #   dino 模块说明
├── 2022-winners-camera-calibration-challenge/  # KaliCalib 获胜方案（原始代码 + 模型）
│   ├── models/
│   │   ├── model_challenge.pth                 #   官方全量数据训练模型
│   │   └── model_test.pth                      #   官方 train-only 训练模型
│   ├── kalicalib/                              #   推理核心代码
│   └── configs/                               #   训练配置文件
├── camera-calibration-challenge/              # 官方 challenge 基线代码（参考）
├── images/                                    # 测试图片（image1.png … image5.png）
├── videos/                                    # 测试视频
├── results/                                   # 推理输出目录
├── keypoints_topdown.png                      # 91 个关键点俯视图（编号可视化）
├── visualize_keypoints_topdown.py             # 生成 keypoints_topdown.png 的脚本
└── requirements.txt                           # pip 依赖
```

---

## 环境准备

```bash
conda create -n task python=3.10
conda activate task
pip install -r requirements.txt
```

所有脚本均在 **`basketball/` 根目录**下运行。

---

## 模型下载

由于 github 的单文件限制，需要在 https://huggingface.co/Jinqi-T/Basketball-Calibration/resolve/main/model_tiny_final.pth 下载 tiny 权重，并保存为 ```dino/checkpoints/tiny_finetune/model_final.pth```
(small 权重仍在训练当中)


## 单张图片推理 — `process_image.py`

### 快速开始
```bash
./process_image.sh
```

```bash 
python process_image.py images/image1.png \
  --output-dir results \
  --model dino/checkpoints/tiny_finetune/model_final.pth \
  --valid-model 2022-winners-camera-calibration-challenge/models/model_challenge.pth \
  --valid-diff-threshold 80 \
  --threshold 0.9 \
  --check 10 \
  --clahe --draw-keypoints --print-coords \
  --verbose
```

### 输出文件

每次推理在 `--output-dir` 目录下生成两个文件：

| 文件 | 说明 |
|------|------|
| `<image>_calibrated.png` | 标定可视化图（红色场地线 + 可选关键点叠加） |
| `<image>_result.json` | 完整结果 JSON（关键点坐标/置信度、单应矩阵、验证结果等） |

### 参数说明

| 参数 | 是否必填 | 默认值 | 说明 |
|------|----------|--------|------|
| `input` | ✅ | — | 输入图片路径 |
| `--output-dir` | ✅ | — | 输出目录 |
| `--model` | ✅ | — | 模型权重路径（`.pth`） |
| `--threshold` | — | `0.0` | 置信度阈值；低于该值的关键点不参与计算 |
| `--top-k` | — | 全部 | 阈值过滤后取置信度最高的 K 个点计算单应矩阵 |
| `--check` | — | — | 最少有效关键点数量；不足时输出原始图像并标记 `check_failed` |
| `--valid-model` | — | — | 验证模型路径；差异过大时输出原始图像并标记 `validation_failed` |
| `--valid-diff-threshold` | — | `20.0` | 两模型关键点平均像素距离上限（px） |
| `--clahe` | — | — | 开启 CLAHE 预处理（LAB 色彩空间 L 通道均衡化，改善光照差异） |
| `--draw-keypoints` | — | — | 在输出图上叠加 91 个关键点并标注编号 |
| `--print-coords` | — | — | 配合 `--draw-keypoints` + `--verbose`，将坐标打印到终端 |
| `--verbose` | — | — | 打印进度信息；默认完全静默 |

### 示例

```bash
# 基础推理（静默）
python process_image.py images/image1.png \
  --output-dir results \
  --model 2022-winners-camera-calibration-challenge/models/model_challenge.pth

# 开启详细输出 + 关键点可视化
python process_image.py images/image1.png \
  --output-dir results \
  --model 2022-winners-camera-calibration-challenge/models/model_challenge.pth \
  --draw-keypoints --print-coords --verbose

# 置信度过滤 + top-k + CLAHE
python process_image.py images/image1.png \
  --output-dir results \
  --model 2022-winners-camera-calibration-challenge/models/model_challenge.pth \
  --threshold 0.5 --top-k 30 --clahe --verbose

# 使用 dino 微调模型 + 最少关键点检查
python process_image.py images/image1.png \
  --output-dir results \
  --model dino/checkpoints/tiny_finetune/model_500.pth \
  --threshold 0.9 --check 10 --verbose

# 双模型交叉验证（平均偏差 > 80 px 则输出原始图像）
python process_image.py images/image1.png \
  --output-dir results \
  --model dino/checkpoints/tiny_finetune/model_final.pth \
  --valid-model 2022-winners-camera-calibration-challenge/models/model_challenge.pth \
  --valid-diff-threshold 80 \
  --threshold 0.9 --check 10 --clahe --draw-keypoints --print-coords --verbose
```

### `_result.json` 结构

```json
{
  "timestamp": "2026-06-10T14:32:00",
  "input": "/abs/path/to/image1.png",
  "output_dir": "/abs/path/to/results",
  "settings": {
    "model": "...", "threshold": 0.9, "top_k": null,
    "check": 10, "clahe": true, "valid_diff_threshold": 80.0
  },
  "status": "success",
  "model_arch": "convnext_tiny.dinov3_lvd1689m",
  "homography": [[...], [...], [...]],
  "keypoints": {
    "0":  {"img_x": 123.4, "img_y": 456.7, "conf": 0.023456},
    "13": {"img_x": 200.1, "img_y": 300.5, "conf": 0.987654},
    "...": {}
  },
  "valid_model_arch": "resnet18 (original)",
  "valid_keypoints": {"...": {}},
  "check_result":      {"valid_count": 15, "required": 10, "passed": true},
  "validation_result": {"ok": true, "mean_dist_px": 5.2, "max_dist_px": 12.1,
                        "n_common": 45, "n_main": 60, "n_valid": 58}
}
```

`status` 取值：`"success"` / `"check_failed"` / `"validation_failed"` / `"insufficient_points"`

---

## 视频推理 — `process_video.py`

对视频执行**逐帧（或抽帧）**推理，输出叠加了场地线的标定视频和逐帧汇总 JSON。

### 快速开始
```bash
./process_video.sh
```

```bash
python process_video.py videos/video1.mp4 \
  --output-dir results \
  --model dino/checkpoints/tiny_finetune/model_final.pth \
  --valid-model 2022-winners-camera-calibration-challenge/models/model_challenge.pth \
  --valid-diff-threshold 80 \
  --threshold 0.9 \
  --check 10 \
  --frame-step 1 \
  --clahe \
  --verbose
```

### 抽帧逻辑

- **关键帧**（`frame_idx % frame_step == 0`）：执行完整推理流程，更新单应矩阵缓存。
- **中间帧**（仅在 `frame_step > 1` 时存在）：复用上一次成功的单应矩阵重绘场地线，保持视觉连贯，**不写入 JSON**。

### 输出文件

| 文件 | 说明 |
|------|------|
| `<stem>_calibrated.mp4` | 完整视频（所有帧都写入，中间帧复用上次叠加） |
| `<stem>_results.json` | 被推理的关键帧结果列表（每条含 `frame_idx`） |

### 参数说明

| 参数 | 是否必填 | 默认值 | 说明 |
|------|----------|--------|------|
| `video` | ✅ | — | 输入视频路径 |
| `--output-dir` | ✅ | — | 输出目录 |
| `--model` | ✅ | — | 模型权重路径（`.pth`） |
| `--frame-step` | — | `1` | 每隔 N 帧推理一次（`1` = 每帧都推理） |
| `--threshold` | — | `0.0` | 置信度阈值 |
| `--top-k` | — | 全部 | 取置信度最高的 K 个点计算单应矩阵 |
| `--check` | — | — | 最少有效关键点数量；不足时该帧标记为 `check_failed` |
| `--clahe` | — | — | 开启 CLAHE 预处理 |
| `--valid-model` | — | — | 验证模型路径 |
| `--valid-diff-threshold` | — | `20.0` | 两模型关键点平均像素距离上限（px） |
| `--verbose` | — | — | 打印进度信息 |

### 示例

```bash
# 每帧推理（默认）
python process_video.py videos/game.mp4 \
  --output-dir results \
  --model 2022-winners-camera-calibration-challenge/models/model_challenge.pth \
  --verbose

# 每 10 帧推理一次，中间帧复用
python process_video.py videos/game.mp4 \
  --output-dir results \
  --model dino/checkpoints/tiny_finetune/model_final.pth \
  --frame-step 10 \
  --threshold 0.9 --check 10 --clahe --verbose

# 双模型验证 + 抽帧
python process_video.py videos/game.mp4 \
  --output-dir results \
  --model dino/checkpoints/tiny_finetune/model_final.pth \
  --valid-model 2022-winners-camera-calibration-challenge/models/model_challenge.pth \
  --valid-diff-threshold 80 \
  --frame-step 5 --threshold 0.9 --check 10 --clahe --verbose
```

### `_results.json` 结构

视频 JSON 是一个列表，每个元素对应一个被推理的关键帧，字段与 `_result.json` 一致，额外含 `frame_idx`：

```json
[
  {
    "frame_idx": 0,
    "status": "success",
    "model_arch": "convnext_tiny.dinov3_lvd1689m",
    "homography": [[...], [...], [...]],
    "keypoints": {"0": {"img_x": 123.4, "img_y": 456.7, "conf": 0.98}, "...": {}},
    "valid_model_arch": null,
    "valid_keypoints": null,
    "check_result": {"valid_count": 23, "required": 10, "passed": true},
    "validation_result": null
  },
  {
    "frame_idx": 10,
    "status": "check_failed",
    "homography": null,
    "keypoints": {"...": {}},
    "check_result": {"valid_count": 3, "required": 10, "passed": false},
    "...": {}
  }
]
```

`status` 取值同 `process_image.py`：`"success"` / `"check_failed"` / `"validation_failed"` / `"insufficient_points"`

---

## 训练新模型 — `dino/train.py`

使用 **ConvNeXt（DINOv3 预训练）+ U-Net 解码器** 微调篮球场关键点检测模型，输出与原始 KaliCalib 格式完全兼容的 checkpoint。

### 模型架构

```
输入 (B, 3, 540, 960)
        │
┌───────┴──────────────────┐
│  ConvNeXt Backbone        │  DINOv3 (LVD-1689M) 预训练
│  f0: H/4  × W/4  × C96   │
│  f1: H/8  × W/8  × C192  │
│  f2: H/16 × W/16 × C384  │
│  f3: H/32 × W/32 × C768  │
└───────┬──────────────────┘
        │  skip connections (U-Net)
┌───────┴──────────────────┐
│  U-Net Decoder (3 stage)  │
│  1×1 Conv head            │
└───────┬──────────────────┘
        │
输出 (B, 94, 135, 240)  ← 与原始 ResNet-18 输出格式完全一致
```

### 参数说明

| 参数 | 是否必填 | 默认值 | 说明 |
|------|----------|--------|------|
| `--checkpoint-dir` | ✅ | — | checkpoint 保存目录（如 `dino/checkpoints/exp1`） |
| `--arch` | — | `convnext_tiny.dinov3_lvd1689m` | 模型架构（tiny / small） |
| `--epochs` | — | `500` | 总训练轮数 |
| `--lr` | — | `1e-4` | 初始学习率（AdamW） |
| `--batch-size` | — | 配置文件值 | 覆盖配置文件的 batch size |
| `--freeze-backbone-epochs` | — | `0` | 前 N epoch 冻结 backbone；0 表示全程微调 |
| `--save-every` | — | `10` | 每隔 N epoch 保存一次 |
| `--resume` | — | — | 从指定 checkpoint 继续训练 |
| `--config-file` | — | `train_sviewds_full_dataset.yml` | 数据集配置文件（默认与 `model_challenge.pth` 训练条件一致） |

### 示例

```bash
# 基础微调（全程训练 backbone）
python dino/train.py \
  --checkpoint-dir dino/checkpoints/tiny_full \
  --epochs 500

# 推荐：两阶段训练（先冻结 backbone 50 epoch，再解冻）
python dino/train.py \
  --checkpoint-dir dino/checkpoints/tiny_finetune \
  --epochs 500 \
  --freeze-backbone-epochs 50 \
  --lr 1e-4

# 使用 ConvNeXt-Small（精度更高）
python dino/train.py \
  --arch convnext_small.dinov3_lvd1689m \
  --checkpoint-dir dino/checkpoints/small_finetune \
  --epochs 500 --freeze-backbone-epochs 50

# 断点续训
python dino/train.py \
  --checkpoint-dir dino/checkpoints/tiny_finetune \
  --resume dino/checkpoints/tiny_finetune/model_200.pth \
  --epochs 500
```

### Checkpoint 格式

```python
{
    'arch':       'convnext_tiny.dinov3_lvd1689m',
    'state_dict': { ... },
    'epoch':      500,
    'loss':       0.00012345,
}
```

`process_image.py` / `process_video.py` 通过检测 `'arch'` 键自动识别新旧格式，无需手动配置。

---

## 可用模型

| 模型文件 | 架构 | 预训练 | 说明 |
|----------|------|--------|------|
| `2022-winners-camera-calibration-challenge/models/model_challenge.pth` | ResNet-18 | ImageNet-1K | 官方全量数据训练，MSE 73.16 mm |
| `2022-winners-camera-calibration-challenge/models/model_test.pth` | ResNet-18 | ImageNet-1K | 官方 train-only，MSE 107.78 mm |
| `dino/checkpoints/<exp>/model_<epoch>.pth` | ConvNeXt-Tiny/Small | DINOv3 LVD-1689M | 自训练微调模型 |

| 模型 | 参数量 | 推理速度（960×540, GPU） | 跨场馆泛化 |
|------|--------|--------------------------|------------|
| ResNet-18（原始） | ~14M | ~80 fps | 一般 |
| ConvNeXt-Tiny | ~35M | ~50 fps | 较好 |
| ConvNeXt-Small | ~57M | ~35 fps | 最好 |

---

## 关键点编号

模型检测 `0`–`90` 共 91 个地面关键点，按 **7 行 × 13 列**排列，编号规则：

```
index = 13 × row + col
```

行 0 为底线（最靠近左下角），列 0 为左侧边线。详见 `keypoints_topdown.png`。

![keypoints_topdown](keypoints_topdown.png)

---

## 注意事项

- 所有脚本默认优先使用 CUDA；无 GPU 时自动退回 CPU（速度较慢）。
- `process_image.py` 模块加载时**不会**修改工作目录；所有命令行传入的相对路径均基于用户运行脚本时的目录解析。
- `dino/models/hub/` 存放 timm 本地权重缓存，若目录为空可运行 `python dino/load_model.py` 触发下载。
- `.gitignore` 已忽略数据集、压缩包、批量输出文件和缓存，保留 `images/`、`videos/` 样本、`models/*.pth` 和 `results/`。
