"""
ConvNeXt backbone + U-Net decoder for basketball court keypoint heatmap prediction.

输出格式与原 KaliCalib (model_resnet.py) 完全一致：
  (B, 94, H/4, W/4)，通道维 softmax，值域 [1e-4, 1-1e-4]
可直接被 estimateCalibHM() 使用。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_backbone(arch: str, pretrained: bool, hub_dir: str | None) -> nn.Module:
    """加载 timm ConvNeXt backbone，支持本地 hub 缓存。"""
    if hub_dir is not None:
        os.environ['HF_HOME'] = hub_dir

    import timm
    backbone = timm.create_model(
        arch,
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    return backbone


class _DecBlock(nn.Module):
    """转置卷积上采样 + BN + ReLU。"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.up(x)))


class KaliCalibConvNeXt(nn.Module):
    """
    ConvNeXt backbone（DINOv3 / ImageNet-22k 预训练）+ 3 级 U-Net 解码器。

    Args:
        arch:      timm 模型名，如 'convnext_tiny.dinov3_lvd1689m'
        pretrained: 是否在构造时下载/加载预训练权重（训练时 True，推理时 False）
        hub_dir:   HuggingFace 本地缓存目录（None 则用默认 ~/.cache/huggingface）
        num_classes: 输出通道数，默认 94（91 关键点 + 2 篮筐占位 + 1 背景）
        freeze_backbone: 是否冻结 backbone（仅训练解码器）
    """

    def __init__(
        self,
        arch: str = 'convnext_tiny.dinov3_lvd1689m',
        pretrained: bool = True,
        hub_dir: str | None = None,
        num_classes: int = 94,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.arch = arch

        self.backbone = _build_backbone(arch, pretrained, hub_dir)

        # 从 timm feature_info 自动读取各阶段通道数（C0/C1/C2/C3）
        chs = [info['num_chs'] for info in self.backbone.feature_info]
        c0, c1, c2, c3 = chs  # e.g. [96, 192, 384, 768] for ConvNeXt-Tiny/Small

        # 解码器：stage3(H/32) → stage2(H/16) → stage1(H/8) → stage0(H/4)
        self.dec3 = _DecBlock(c3,      c2)        # 768 → 384
        self.dec2 = _DecBlock(c2 + c2, c1)        # 384+384 → 192
        self.dec1 = _DecBlock(c1 + c1, c0)        # 192+192 → 96
        self.head = nn.Conv2d(c0 + c0, num_classes, kernel_size=1)  # 96+96 → 94

        self._init_decoder()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _init_decoder(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """对 x 做双线性插值，使空间尺寸匹配 ref（处理奇数尺寸的误差）。"""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0, f1, f2, f3 = self.backbone(x)   # H/4, H/8, H/16, H/32

        x = self.dec3(f3)                    # → H/16
        x = self._match_size(x, f2)
        x = torch.cat([x, f2], dim=1)

        x = self.dec2(x)                     # → H/8
        x = self._match_size(x, f1)
        x = torch.cat([x, f1], dim=1)

        x = self.dec1(x)                     # → H/4
        x = self._match_size(x, f0)
        x = torch.cat([x, f0], dim=1)

        x = self.head(x)                     # → (B, 94, H/4, W/4)

        # 与原模型完全一致的输出变换
        x = x.permute(0, 2, 3, 1)
        x = torch.softmax(x, dim=3)
        x = x.permute(0, 3, 1, 2)
        x = torch.clamp(x, min=1e-4, max=1 - 1e-4)

        return x
