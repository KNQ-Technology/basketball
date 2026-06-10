# dino — 基于 ConvNeXt (DINOv3) 的篮球场标定模型

本目录提供用 **ConvNeXt backbone（DINOv3 预训练权重）+ U-Net 解码器** 对篮球场关键点检测任务进行微调的完整流程，作为原始 ResNet-18 模型的升级方案。

---

## 目录结构

```
dino/
├── model_convnext.py   # 模型定义：KaliCalibConvNeXt
├── train.py            # 微调训练脚本
├── load_model.py       # 单独加载预训练 backbone 的辅助脚本（调试用）
├── checkpoints/        # 训练产生的 checkpoint（.pth）保存在此
└── models/
    └── hub/            # timm 本地权重缓存（HuggingFace 格式）
        ├── models--timm--convnext_tiny.dinov3_lvd1689m/
        └── models--timm--convnext_small.dinov3_lvd1689m/
```

---

## 模型架构

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
│  U-Net Decoder            │
│  deconv + cat skip × 3    │
│  1×1 Conv head            │
└───────┬──────────────────┘
        │
输出 (B, 94, 135, 240)   ← H/4 × W/4，与原模型完全一致
  通道 softmax + clamp[1e-4, 1-1e-4]
```

输出格式与 `estimateCalibHM()` 完全兼容，可直接接入 `process_image.py` / `process_video.py`。

---

## 快速开始

### 1. 确认权重已缓存

```bash
ls dino/models/hub/
# 应看到：
# models--timm--convnext_tiny.dinov3_lvd1689m
# models--timm--convnext_small.dinov3_lvd1689m
```

若缺少，可运行 `dino/load_model.py` 触发下载（需联网或配置镜像）：

```bash
python dino/load_model.py
```

### 2. 开始训练

所有命令**在项目根目录 `basketball/` 下运行**。

#### 基础训练（全程微调）

```bash
python dino/train.py \
  --arch convnext_tiny.dinov3_lvd1689m \
  --checkpoint-dir dino/checkpoints/tiny_full \
  --epochs 500
```

> 默认配置文件为 `train_sviewds_full_dataset.yml`（使用 train+val+test 全量数据集），与官方 `model_challenge.pth` 训练条件一致。

#### 推荐：两阶段训练（先冻结 backbone）

前 N epoch 只训练解码器（收敛快、不破坏预训练特征），之后解冻全网络精细微调：

```bash
python dino/train.py \
  --arch convnext_tiny.dinov3_lvd1689m \
  --checkpoint-dir dino/checkpoints/tiny_finetune \
  --epochs 500 \
  --freeze-backbone-epochs 50 \
  --lr 1e-4
```

#### 使用 ConvNeXt-Small（精度更高，速度略慢）

```bash
python dino/train.py \
  --arch convnext_small.dinov3_lvd1689m \
  --checkpoint-dir dino/checkpoints/small_finetune \
  --epochs 500 \
  --freeze-backbone-epochs 50
```

#### 仅使用 train split（对应 model_test.pth 的训练条件）

```bash
python dino/train.py \
  --arch convnext_tiny.dinov3_lvd1689m \
  --checkpoint-dir dino/checkpoints/tiny_trainonly \
  --config-file 2022-winners-camera-calibration-challenge/configs/train_sviewds.yml \
  --epochs 500
```

#### 断点续训

```bash
python dino/train.py \
  --arch convnext_tiny.dinov3_lvd1689m \
  --checkpoint-dir dino/checkpoints/tiny_finetune \
  --resume dino/checkpoints/tiny_finetune/model_200.pth \
  --epochs 500
```

### 3. 推理

训练产生的 checkpoint 可直接通过 `process_image.py` 的 `--model` 参数使用，无需任何额外配置：

```bash
python process_image.py images/image1.png \
  --output-dir results \
  --model dino/checkpoints/tiny_finetune/model_500.pth \
  --threshold 0.9 --top-k 20 --verbose
```

`process_image.py` 会自动识别 checkpoint 格式，选择正确的模型类（ResNet-18 原始模型或 ConvNeXt 新模型）。

---

## 训练参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--arch` | `convnext_tiny.dinov3_lvd1689m` | 模型架构，支持 tiny / small |
| `--epochs` | `500` | 总训练轮数 |
| `--lr` | `1e-4` | 初始学习率（AdamW） |
| `--batch-size` | 配置文件中的值（默认 2） | 覆盖配置文件的 batch size |
| `--freeze-backbone-epochs` | `0` | 前 N epoch 冻结 backbone；0 表示全程微调 |
| `--save-every` | `10` | 每隔 N epoch 保存一次 checkpoint |
| `--resume` | — | 从指定 checkpoint 继续训练 |
| `--config-file` | `...configs/train_sviewds.yml` | 数据集配置文件路径 |
| `--checkpoint-dir` | **必填** | checkpoint 保存目录（如 `dino/checkpoints/exp1`） |

---

## Checkpoint 格式

新模型的 checkpoint 为 Python dict，包含以下字段：

```python
{
    'arch':       'convnext_tiny.dinov3_lvd1689m',  # 模型架构名
    'state_dict': { ... },                           # 模型权重
    'epoch':      100,                               # 已训练 epoch 数
    'loss':       0.00012345,                        # 最后一 epoch 的平均损失
}
```

`infer.py` 通过检测 `'arch'` 键来区分新旧格式：有 `'arch'` 键则加载 `KaliCalibConvNeXt`，否则加载原始 ResNet-18 模型。

---

## 与原始模型的对比

| 项目 | 原始模型 | ConvNeXt-Tiny | ConvNeXt-Small |
|---|---|---|---|
| Backbone | ResNet-18 (11M) | ConvNeXt-Tiny (28M) | ConvNeXt-Small (50M) |
| 预训练数据 | ImageNet-1K | DINOv3 LVD-1689M | DINOv3 LVD-1689M |
| 参数量（含 decoder） | ~14M | ~35M | ~57M |
| 推理速度（960×540, GPU） | ~80 fps | ~50 fps | ~35 fps |
| 跨场馆泛化 | 一般 | 更好 | 最好 |
