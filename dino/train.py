#!/usr/bin/env python3
"""
dino/train.py — 基于 ConvNeXt (DINOv3 预训练) 的篮球场关键点检测微调脚本

用法示例（在 basketball/ 目录下运行）：
  python dino/train.py \\
    --arch convnext_tiny.dinov3_lvd1689m \\
    --config-file 2022-winners-camera-calibration-challenge/configs/train_sviewds_full_dataset.yml \\
    --epochs 500 \\
    --freeze-backbone-epochs 50
"""
import sys
import os
import argparse
import random
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 路径设置：将 challenge 仓库加入 sys.path，保持工作目录不变
# ──────────────────────────────────────────────────────────────
DINO_DIR  = os.path.dirname(os.path.abspath(__file__))
BBALL_DIR = os.path.dirname(DINO_DIR)
REPO_DIR  = os.path.join(BBALL_DIR, '2022-winners-camera-calibration-challenge')

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# 让 timm 使用本地 hub 缓存（避免重复下载）
os.environ['HF_HOME'] = os.path.join(DINO_DIR, 'models')

from config import cfg                          # challenge 仓库的 yacs config
from data import make_data_loader               # challenge 仓库的数据加载
from kalicalib.losses import KeypointsCrossEntropyLoss

from model_convnext import KaliCalibConvNeXt    # 同目录

# ──────────────────────────────────────────────────────────────
# 可复现性
# ──────────────────────────────────────────────────────────────
random.seed(4212)
np.random.seed(4212)
torch.manual_seed(4212)


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────
def save_checkpoint(path: str, model: torch.nn.Module, arch: str,
                    epoch: int, avg_loss: float):
    """保存格式可被 infer.py 直接加载的 checkpoint。"""
    torch.save({
        'arch':       arch,
        'state_dict': model.state_dict(),
        'epoch':      epoch,
        'loss':       avg_loss,
    }, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    device: torch.device) -> int:
    """从 checkpoint 恢复模型权重，返回已训练的 epoch 数。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt.get('epoch', 0) if isinstance(ckpt, dict) else 0


# ──────────────────────────────────────────────────────────────
# 主训练函数
# ──────────────────────────────────────────────────────────────
def train(cfg, args):
    n_epoch   = args.epochs
    ckpt_dir  = args.checkpoint_dir
    arch      = args.arch
    freeze_bb = args.freeze_backbone_epochs

    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] device={device}  arch={arch}  epochs={n_epoch}")

    # ── 模型 ──────────────────────────────────────────────────
    model = KaliCalibConvNeXt(
        arch=arch,
        pretrained=True,
        hub_dir=os.path.join(DINO_DIR, 'models'),
        freeze_backbone=(freeze_bb > 0),
    ).to(device)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, device)
        print(f"[Train] 从 {args.resume} 恢复，已完成 {start_epoch} epoch")

    # ── 损失函数（与原脚本完全一致）──────────────────────────
    hm_h = cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[1] // 4
    hm_w = cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[0] // 4
    weights = torch.ones((94, hm_h, hm_w), device=device)
    weights[:-1] *= 3750
    loss_fn = KeypointsCrossEntropyLoss(weights)

    # ── 优化器 & 调度器 ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    # 在 2/3 处将 lr 下降一次（与原脚本策略一致）
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2 * n_epoch // 3)

    # ── 数据加载 ──────────────────────────────────────────────
    if args.batch_size:
        cfg.defrost()
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
        cfg.freeze()
    if args.workers is not None:
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = args.workers
        cfg.freeze()

    data_loader = make_data_loader(cfg, is_train=True)

    # ── 训练循环 ──────────────────────────────────────────────
    model.train()

    for e in range(start_epoch, n_epoch):
        # 在 freeze_bb epoch 结束后解冻 backbone
        if e == freeze_bb and freeze_bb > 0:
            print(f"[Train] Epoch {e+1}: 解冻 backbone，开始全网络微调")
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            # 重建 optimizer，让 backbone 参数也被优化
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, max(1, (n_epoch - freeze_bb) * 2 // 3)
            )

        pbar = tqdm(data_loader, desc=f"Epoch {e+1}/{n_epoch}")
        total_loss, n_iter = 0.0, 0

        for imgs, data in pbar:
            imgs     = imgs.to(device)
            heatmaps = data['heatmaps'].to(device)

            out  = model(imgs)
            loss = loss_fn(out, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_iter     += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        lr_scheduler.step()
        avg_loss = total_loss / max(n_iter, 1)
        tqdm.write(f"Epoch {e+1}/{n_epoch}  avg_loss={avg_loss:.8e}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if (e + 1) % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_{e+1}.pth")
            save_checkpoint(ckpt_path, model, arch, e + 1, avg_loss)
            tqdm.write(f"  → saved: {ckpt_path}")

    # 训练结束后保存最终模型
    final_path = os.path.join(ckpt_dir, "model_final.pth")
    save_checkpoint(final_path, model, arch, n_epoch, avg_loss)
    print(f"[Train] 完成，最终模型: {final_path}")


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ConvNeXt 篮球场关键点微调训练脚本")

    # 模型
    parser.add_argument(
        '--arch', default='convnext_tiny.dinov3_lvd1689m',
        choices=['convnext_tiny.dinov3_lvd1689m', 'convnext_small.dinov3_lvd1689m'],
        help="timm 模型名（models/hub 中需已缓存），默认 convnext_tiny"
    )
    parser.add_argument(
        '--resume', default='', type=str,
        help="从已有 checkpoint 继续训练（传入 .pth 路径）"
    )

    # 训练超参
    parser.add_argument('--epochs',     type=int,   default=500,    help="总训练 epoch 数")
    parser.add_argument('--lr',         type=float, default=1e-4,   help="初始学习率")
    parser.add_argument('--batch-size', type=int,   default=None,   help="覆盖 config 中的 batch size")
    parser.add_argument('--workers',    type=int,   default=None,   help="DataLoader worker 数")
    parser.add_argument('--save-every', type=int,   default=10,     help="每隔 N epoch 保存一次 checkpoint")
    parser.add_argument(
        '--freeze-backbone-epochs', type=int, default=0, dest='freeze_backbone_epochs',
        help="前 N epoch 冻结 backbone，仅训练解码器；0 表示全程微调（默认）"
    )

    # 路径
    parser.add_argument(
        '--config-file',
        default=os.path.join(REPO_DIR, 'configs', 'train_sviewds_full_dataset.yml'),
        help="challenge 仓库的 YAML 配置文件（默认使用全量数据集，与 model_challenge.pth 训练条件一致）"
    )
    parser.add_argument(
        '--checkpoint-dir',
        required=True,
        metavar='DIR',
        help="checkpoint 保存目录（必填，例如 dino/checkpoints/exp1）"
    )

    args = parser.parse_args()

    # 加载 challenge 配置
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    train(cfg, args)
