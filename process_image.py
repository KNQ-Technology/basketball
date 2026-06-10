#!/usr/bin/env python3
import os
import sys
import argparse
import json
import io
import contextlib
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

BBALL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR  = os.path.join(BBALL_DIR, "2022-winners-camera-calibration-challenge")
sys.path.insert(0, REPO_DIR)
if BBALL_DIR not in sys.path:
    sys.path.insert(1, BBALL_DIR)
# 记录调用时的工作目录，用于解析命令行传入的相对路径
_ORIG_CWD = os.getcwd()

from kalicalib.model_resnet import makeModel
from kalicalib.estimateHomography import estimateCalibHM, getFieldPoints2d
from data.datasets.viewds import getFieldPoints

IMG_WIDTH = 960
IMG_HEIGHT = 540


def preprocess(image_path, clahe=False):
    """读取图片并做分辨率和颜色预处理，返回 RGB numpy array (H, W, 3) uint8。

    clahe: 在 LAB 色彩空间的 L 通道上做自适应直方图均衡化，
           缓解不同场馆光照/对比度差异。
    """
    img = Image.open(image_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT),
                                                        Image.BILINEAR)
    img_np = np.array(img, dtype=np.uint8)

    if clahe:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe_obj.apply(l)
        lab = cv2.merge([l, a, b])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img_np


def load_model(model_path, device, verbose=True):
    """加载模型权重。

    支持两种 checkpoint 格式：
    1. 原始格式（纯 state_dict，由 kalicalib/train.py 生成）→ 加载 ResNet-18 KaliCalib 模型
    2. 新格式（dict with 'arch' key，由 dino/train.py 生成）→ 加载 ConvNeXt KaliCalib 模型

    返回 (model, arch_info: str)
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'arch' in ckpt:
        from dino.model_convnext import KaliCalibConvNeXt
        model = KaliCalibConvNeXt(
            arch=ckpt['arch'],
            pretrained=False,
        ).to(device)
        model.load_state_dict(ckpt['state_dict'])
        arch_info = ckpt['arch']
    else:
        model = makeModel().to(device)
        model.load_state_dict(ckpt)
        arch_info = "resnet18 (original)"

    model.eval()
    if verbose:
        print(f"  模型架构: {arch_info}")
    return model, arch_info


def infer(image_rgb, model, device):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(image_rgb).to(device).unsqueeze(0)
    with torch.no_grad():
        heatmaps = model(tensor)
    return heatmaps


def visualize(image_rgb, heatmaps, top_k=None, threshold=0.0):
    """在图片上绘制红色场地线。

    Returns:
        (vis_bgr, success: bool, homography: np.ndarray | None)
        success=False 时 vis_bgr 为未修改的原始图像，homography 为 None。
    """
    vis_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    calib, Hest = estimateCalibHM(
        heatmaps,
        vis_bgr,
        getFieldPoints2d(),
        getFieldPoints(),
        IMG_HEIGHT,
        IMG_WIDTH,
        False,
        draw_points=False,
        draw_calib_court=False,
        top_k=top_k,
        threshold=threshold,
    )
    success = calib is not None
    return vis_bgr, success, Hest


def _extract_keypoints(heatmaps):
    """从 heatmaps 中提取所有检测到的关键点位置。

    Returns:
        dict {idx: (img_x, img_y, conf)}，仅包含有有效质心的关键点。
        坐标已缩放回图像分辨率（heatmap 分辨率 × 4）。
    """
    out = heatmaps[0].cpu().numpy()
    nb_points = out.shape[0] - 1 - 2  # 91

    pixelScores = np.swapaxes(out, 0, 2)
    pixelMaxScores = np.max(pixelScores, axis=2, keepdims=True)
    pixelMax = (pixelScores == pixelMaxScores)
    pixelMap = np.swapaxes(pixelMax, 0, 2).astype(np.uint8)
    pixelMap = (out > 0) * pixelMap

    kpts = {}
    for i in range(nb_points):
        M = cv2.moments(pixelMap[i])
        if M["m00"] == 0:
            continue
        img_x = M["m10"] / M["m00"] * 4
        img_y = M["m01"] / M["m00"] * 4
        conf  = float(out[i].max())
        kpts[i] = (img_x, img_y, conf)
    return kpts


def validate_keypoints(heatmaps_main, heatmaps_valid,
                       diff_threshold=20.0, min_common=4):
    """比较两个模型的关键点检测结果。

    Returns:
        (ok: bool, message: str, details: dict)
        details 包含 n_main, n_valid, n_common, mean_dist_px, max_dist_px
    """
    kpts_main  = _extract_keypoints(heatmaps_main)
    kpts_valid = _extract_keypoints(heatmaps_valid)

    common_ids = set(kpts_main) & set(kpts_valid)
    n_common   = len(common_ids)
    details = {
        "n_main":       len(kpts_main),
        "n_valid":      len(kpts_valid),
        "n_common":     n_common,
        "mean_dist_px": None,
        "max_dist_px":  None,
    }

    if n_common < min_common:
        msg = (
            f"验证失败：两模型共同检测到的关键点仅有 {n_common} 个"
            f"（需至少 {min_common} 个），差异过大，输出原始图像。"
            f" 主模型 {len(kpts_main)} 个，验证模型 {len(kpts_valid)} 个。"
        )
        return False, msg, details

    dists = []
    for idx in common_ids:
        x1, y1, _ = kpts_main[idx]
        x2, y2, _ = kpts_valid[idx]
        dists.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    mean_dist = float(np.mean(dists))
    max_dist  = float(np.max(dists))
    details["mean_dist_px"] = round(mean_dist, 2)
    details["max_dist_px"]  = round(max_dist, 2)

    if mean_dist > diff_threshold:
        msg = (
            f"验证失败：{n_common} 个共同关键点平均偏差 {mean_dist:.1f} px"
            f"（阈值 {diff_threshold} px，最大 {max_dist:.1f} px），输出原始图像。"
        )
        return False, msg, details

    return True, (
        f"验证通过：{n_common} 个共同关键点平均偏差 {mean_dist:.1f} px"
        f"（阈值 {diff_threshold} px）。"
    ), details


# 7 行关键点对应的 BGR 颜色（HSV 均匀分布，高饱和度）
_ROW_COLORS = [
    cv2.cvtColor(np.uint8([[[int(h * 180 / 7), 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
    for h in range(7)
]


def overlay_keypoints(vis_bgr, heatmaps, print_coords=False, threshold=0.0):
    """在已有的 BGR 图像上叠加 91 个关键点（就地修改，同时返回该图像）。

    每行（共 7 行，每行 13 个点）用不同颜色绘制，点旁标注点序号。
    print_coords: 同时将各点的图像坐标与场地坐标打印到标准输出。
    """
    out = heatmaps[0].cpu().numpy()
    nb_points = out.shape[0] - 1 - 2  # 91 = 7×13

    pixelScores = np.swapaxes(out, 0, 2)
    pixelMaxScores = np.max(pixelScores, axis=2, keepdims=True)
    pixelMax = (pixelScores == pixelMaxScores)
    pixelMap = np.swapaxes(pixelMax, 0, 2).astype(np.uint8)
    pixelMap = (out > 0) * pixelMap

    field_pts = getFieldPoints()[:nb_points, :2]  # (91, 2)  单位 mm

    if print_coords:
        print(f"\n{'idx':>4}  {'img_x':>6} {'img_y':>6}  {'field_x(mm)':>12} {'field_y(mm)':>12}  {'conf':>8}")
        print("-" * 62)

    for i in range(nb_points):
        conf = float(out[i].max())
        M = cv2.moments(pixelMap[i])
        if M["m00"] == 0 or conf < threshold:
            if print_coords:
                reason = "(not detected)" if M["m00"] == 0 else f"(conf {conf:.4f} < threshold)"
                print(f"{i:>4}  {reason:>35}  {conf:>8.4f}")
            continue

        # heatmap 分辨率是图像的 1/4，×4 还原到图像坐标
        img_y = M["m01"] / M["m00"] * 4
        img_x = M["m10"] / M["m00"] * 4
        cx, cy = round(img_x), round(img_y)

        color = _ROW_COLORS[i // 13]

        cv2.circle(vis_bgr, (cx, cy), 6, (0, 0, 0), -1)
        cv2.circle(vis_bgr, (cx, cy), 5, color, -1)

        label = str(i)
        cv2.putText(vis_bgr, label, (cx + 7, cy - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis_bgr, label, (cx + 7, cy - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, color, 1, cv2.LINE_AA)

        if print_coords:
            fx, fy = field_pts[i]
            print(f"{i:>4}  {cx:>6} {cy:>6}  {fx:>12.1f} {fy:>12.1f}  {conf:>8.4f}")

    return vis_bgr


def _abs(path):
    """将相对路径转为基于 _ORIG_CWD 的绝对路径。"""
    return os.path.join(_ORIG_CWD, path) if not os.path.isabs(path) else path


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _kpts_to_dict(kpts):
    """将 _extract_keypoints 的结果转为可序列化的 dict。"""
    return {
        str(idx): {"img_x": round(x, 2), "img_y": round(y, 2), "conf": round(conf, 6)}
        for idx, (x, y, conf) in kpts.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="篮球场景相机标定推理：输出标定图和结果 JSON")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("--output-dir", required=True, metavar="DIR",
                        help="输出目录（必填）；结果文件命名为 <image>_calibrated.png 和 <image>_result.json")
    parser.add_argument("--model", required=True, metavar="MODEL",
                        help="模型权重路径（必填）")
    parser.add_argument("--clahe", action="store_true",
                        help="对 L 通道做自适应直方图均衡化，缓解光照差异")
    parser.add_argument("--draw-keypoints", action="store_true",
                        help="在输出图片上叠加 91 个关键点并标注编号")
    parser.add_argument("--print-coords", action="store_true",
                        help="配合 --draw-keypoints，将关键点坐标打印到终端（需同时开启 --verbose）")
    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help="取置信度最高的 K 个关键点计算单应矩阵；不指定则使用所有高于阈值的点")
    parser.add_argument("--threshold", type=float, default=0.0, metavar="T",
                        help="置信度阈值，低于该值的关键点不参与计算（默认 0.0）")
    parser.add_argument("--check", type=int, default=None, metavar="N",
                        help="最少有效关键点数量（置信度 ≥ threshold）；不足时输出原始图像")
    parser.add_argument("--valid-model", default=None, metavar="MODEL",
                        help="验证模型路径；差异过大时输出原始图像")
    parser.add_argument("--valid-diff-threshold", type=float, default=20.0, metavar="PX",
                        help="两模型关键点平均像素距离上限（默认 20 px）")
    parser.add_argument("--verbose", action="store_true",
                        help="开启后才会在终端打印进度和调试信息；默认静默运行")
    args = parser.parse_args()

    # ── 路径规范化 ────────────────────────────────────────────────
    args.input      = _abs(args.input)
    args.output_dir = _abs(args.output_dir)
    args.model      = _abs(args.model)
    if args.valid_model:
        args.valid_model = _abs(args.valid_model)

    verbose = args.verbose
    vprint  = print if verbose else (lambda *a, **k: None)

    # ── 输出路径 ──────────────────────────────────────────────────
    stem           = Path(args.input).stem
    calibrated_path = os.path.join(args.output_dir, f"{stem}_calibrated.png")
    json_path       = os.path.join(args.output_dir, f"{stem}_result.json")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 初始化 result_data ────────────────────────────────────────
    result_data = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "input":            args.input,
        "output_dir":       args.output_dir,
        "settings": {
            "model":                args.model,
            "valid_model":          args.valid_model,
            "threshold":            args.threshold,
            "top_k":                args.top_k,
            "check":                args.check,
            "clahe":                args.clahe,
            "valid_diff_threshold": args.valid_diff_threshold,
        },
        "status":           "success",
        "model_arch":       None,
        "keypoints":        {},
        "valid_model_arch": None,
        "valid_keypoints":  None,
        "check_result":     None,
        "validation_result": None,
    }

    def _bail(status, image_rgb):
        """失败时：保存原始图像 + JSON，然后退出。"""
        result_data["status"] = status
        orig_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(calibrated_path, orig_bgr)
        _save_json(json_path, result_data)
        vprint(f"已保存至: {calibrated_path}  {json_path}")
        sys.exit(0)

    # ── 文件存在性检查 ────────────────────────────────────────────
    for label, path in [("输入文件", args.input),
                        ("模型文件", args.model)]:
        if not os.path.exists(path):
            vprint(f"Error: {label}不存在: {path}")
            sys.exit(1)
    if args.valid_model and not os.path.exists(args.valid_model):
        vprint(f"Error: 验证模型文件不存在: {args.valid_model}")
        sys.exit(1)

    # ── 推理 ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vprint(f"使用设备: {device}")

    vprint("加载模型...")
    model, arch_info = load_model(args.model, device, verbose=verbose)
    result_data["model_arch"] = arch_info

    vprint("预处理图片...")
    image_rgb = preprocess(args.input, args.clahe)

    vprint("推理中...")
    heatmaps = infer(image_rgb, model, device)

    # 提取并记录所有关键点（不做阈值过滤，完整保存）
    kpts_main = _extract_keypoints(heatmaps)
    result_data["keypoints"] = _kpts_to_dict(kpts_main)

    # ── --check 检查 ──────────────────────────────────────────────
    if args.check is not None:
        valid_count = sum(1 for (_, _, c) in kpts_main.values() if c >= args.threshold)
        passed = valid_count >= args.check
        result_data["check_result"] = {
            "valid_count": valid_count,
            "required":    args.check,
            "passed":      passed,
        }
        if not passed:
            vprint(
                f"Error: 置信度 ≥ {args.threshold} 的有效关键点数量为 {valid_count}，"
                f"少于 --check {args.check}，输出原始图像。"
            )
            _bail("check_failed", image_rgb)
        vprint(f"[Check] 有效关键点 {valid_count} ≥ {args.check}，通过。")

    # ── 验证模型检查 ──────────────────────────────────────────────
    if args.valid_model:
        vprint("加载验证模型...")
        valid_model, valid_arch = load_model(args.valid_model, device, verbose=verbose)
        result_data["valid_model_arch"] = valid_arch

        vprint("验证模型推理中...")
        heatmaps_valid = infer(image_rgb, valid_model, device)

        kpts_valid = _extract_keypoints(heatmaps_valid)
        result_data["valid_keypoints"] = _kpts_to_dict(kpts_valid)

        ok, msg, val_details = validate_keypoints(
            heatmaps, heatmaps_valid,
            diff_threshold=args.valid_diff_threshold,
        )
        result_data["validation_result"] = {"ok": ok, "message": msg, **val_details}
        vprint(f"[验证] {msg}")
        if not ok:
            _bail("validation_failed", image_rgb)

    # ── 可视化 ────────────────────────────────────────────────────
    vprint("生成可视化...")
    # 当不 verbose 时，静默掉 challenge 仓库内部的 print（如 "Default calib"）
    ctx = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()
    with ctx:
        result, success, Hest = visualize(image_rgb, heatmaps,
                                          top_k=args.top_k, threshold=args.threshold)

    # 单应矩阵写入 JSON（3×3 列表，或 null）
    result_data["homography"] = (
        Hest.tolist() if Hest is not None else None
    )

    if not success:
        result_data["status"] = "insufficient_points"
        vprint("警告：有效关键点不足，已输出原始图像（不含场地线）。")

    if args.draw_keypoints:
        overlay_keypoints(result, heatmaps,
                          print_coords=(args.print_coords and verbose),
                          threshold=args.threshold)

    # ── 保存输出 ──────────────────────────────────────────────────
    cv2.imwrite(calibrated_path, result)
    _save_json(json_path, result_data)
    vprint(f"已保存至: {calibrated_path}")
    vprint(f"结果 JSON: {json_path}")


if __name__ == "__main__":
    main()
