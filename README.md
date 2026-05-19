# Basketball Camera Calibration Demo

本项目基于 `2022-winners-camera-calibration-challenge` 中的 KaliCalib 方法，对单张篮球比赛图片进行相机标定可视化。

当前根目录下的 `script.sh` 会读取 `image.png`，使用 challenge 模型权重推理，并把带关键点序号和球场标定线的结果保存为 `output.png`。

## 目录说明

```text
basketball/
├── 2022-winners-camera-calibration-challenge/  # KaliCalib 获胜方案代码和模型
├── camera-calibration-challenge/               # 官方 challenge 基线代码
├── image.png                                   # 默认输入图片
├── output.png                                  # 默认输出可视化图片
├── script.sh                                   # 单图推理脚本
├── requirements.txt                            # task 环境导出的 pip 依赖
└── .gitignore                                  # 忽略数据集、压缩包、生成图片等大文件
```

## 环境准备

推荐使用 Conda 创建环境，然后安装依赖：

```bash
conda create -n task python=3.10
conda activate task
pip install -r requirements.txt
```

当前脚本默认调用：

```bash
/home/tangjinqi/miniconda3/envs/task/bin/python
```

如果你的 Conda 安装路径或环境名不同，需要修改 `script.sh` 第 12 行的 Python 路径。

## 运行单图推理

在项目根目录运行：

```bash
cd /home/tangjinqi/basketball
./script.sh
```

运行成功后会生成：

```text
/home/tangjinqi/basketball/output.png
```

这张图片会包含：

- 蓝色关键点位置
- 红色关键点编号
- 黄色球场标定轮廓线

## 默认使用的模型

脚本默认使用：

```text
2022-winners-camera-calibration-challenge/models/model_challenge.pth
```

这是作者提供的 challenge 模型权重。

如果想改用 test 模型，把 `script.sh` 里的：

```bash
MODEL_PATH="${REPO_DIR}/models/model_challenge.pth"
```

以及 Python 代码中的：

```python
MODEL_PATH = os.path.join(REPO_DIR, "models", "model_challenge.pth")
```

改成：

```text
models/model_test.pth
```

## 更换输入图片

默认输入图片是：

```text
image.png
```

如果想换成别的图片，有两种方式：

1. 直接把新的图片重命名为 `image.png`，放在项目根目录。
2. 修改 `script.sh` 中的输入路径：

```bash
INPUT_IMAGE="${ROOT_DIR}/image.png"
```

以及 Python 代码中的：

```python
INPUT_IMAGE = os.path.join(ROOT_DIR, "image.png")
```

## 更改输出位置

默认输出图片是：

```text
output.png
```

如果想保存到其他位置，修改 `script.sh` 中：

```bash
OUTPUT_IMAGE="${ROOT_DIR}/output.png"
```

以及 Python 代码中的：

```python
OUTPUT_IMAGE = os.path.join(ROOT_DIR, "output.png")
```

## 更改输入尺寸

模型当前按 challenge 设置使用：

```python
IMG_WIDTH = 960
IMG_HEIGHT = 540
```

如果你想测试其他分辨率，可以修改 `script.sh` 中这两个变量。不过该模型是在 `960x540` 相关设置下使用的，建议默认保持不变。

## 关键点编号说明

模型输出的关键点来自篮球场模板点阵。主要使用 `0` 到 `90` 共 91 个地面点，按 `7` 行乘 `13` 列排列：

```text
index = 13 * row + col
```

可视化图片中的红色数字就是这些关键点编号。

## 注意事项

- `script.sh` 会自动选择 CUDA；如果没有 GPU，会退回 CPU，但速度会慢一些。
- `requirements.txt` 是当前 `task` 环境的完整导出，可能包含一些与本项目无关的包。
- `.gitignore` 已忽略数据集、压缩包、批量输出图片和缓存，但保留根目录的 `image.png`、`output.png` 以及 `models/` 下的模型权重。
