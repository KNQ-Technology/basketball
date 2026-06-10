#!/usr/bin/env python3
"""
process_video.py — 篮球视频标定：抽帧推理，输出标定视频和逐帧 JSON

用法示例（在 basketball/ 目录下运行）：
  python process_video.py videos/game.mp4 \
    --output-dir results \
    --model dino/checkpoints/tiny_finetune/model_final.pth \
    --frame-step 10 \
    --threshold 0.9 --check 10 \
    --clahe --verbose

输出（--output-dir 目录下）：
  <stem>_calibrated.mp4   叠加场地线的完整视频（中间帧沿用上次单应矩阵）
  <stem>_results.json     被推理帧的结果列表（含 frame_idx，格式与 process_image.py 一致）
"""
import os as _os
_SCRIPT_CWD = _os.getcwd()  # 记录调用时的工作目录，用于解析相对路径

import sys
import argparse
import json
import io
import contextlib
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 把 basketball 目录加入 sys.path，以便 import process_image
_BBALL_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _BBALL_DIR not in sys.path:
    sys.path.insert(0, _BBALL_DIR)

# 导入 process_image 中的推理函数
# 注意：process_image 模块加载时会执行 os.chdir(REPO_DIR)，之后 cwd 变为 challenge 目录
from process_image import (       # noqa: E402
    load_model,
    infer          as _run_infer,
    visualize,
    _extract_keypoints,
    _kpts_to_dict,
    validate_keypoints,
    IMG_WIDTH,
    IMG_HEIGHT,
)

# process_image 加载后 cwd 已切换；drawField 在同一 sys.path 下可直接 import
from kalicalib.estimateHomography import drawField  # noqa: E402


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _abs(path: str) -> str:
    """把相对路径解析为基于调用脚本时工作目录的绝对路径。"""
    return _os.path.join(_SCRIPT_CWD, path) if not _os.path.isabs(path) else path


def _bgr_to_rgb_clahe(frame_bgr: np.ndarray, clahe: bool) -> np.ndarray:
    """缩放 BGR 帧至 960×540，可选 CLAHE，返回 RGB array。"""
    resized = cv2.resize(frame_bgr, (IMG_WIDTH, IMG_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if clahe:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe_obj.apply(l)
        rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return rgb


def _process_one_frame(
    frame_idx: int,
    image_rgb: np.ndarray,
    model,
    valid_model,
    device,
    args,
    verbose: bool,
    vprint,
    arch_info: str,
    valid_arch: str | None,
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    """
    对单帧执行完整推理流程（与 process_image.py 的 main() 逻辑一致）。

    Returns:
        frame_result  写入 JSON 的 dict（含 frame_idx）
        vis_bgr       标定后 BGR 图像；None 表示该帧失败，调用方应用原图
        Hest          单应矩阵（3×3 ndarray）；None 表示未估计成功
    """
    frame_result = {
        "frame_idx":         frame_idx,
        "status":            "success",
        "model_arch":        arch_info,
        "homography":        None,
        "keypoints":         {},
        "valid_model_arch":  valid_arch,
        "valid_keypoints":   None,
        "check_result":      None,
        "validation_result": None,
    }

    # 推理
    heatmaps  = _run_infer(image_rgb, model, device)
    kpts_main = _extract_keypoints(heatmaps)
    frame_result["keypoints"] = _kpts_to_dict(kpts_main)

    # --check
    if args.check is not None:
        valid_count = sum(1 for (_, _, c) in kpts_main.values() if c >= args.threshold)
        passed = valid_count >= args.check
        frame_result["check_result"] = {
            "valid_count": valid_count,
            "required":    args.check,
            "passed":      passed,
        }
        if not passed:
            vprint(f"  [frame {frame_idx}] check 失败：有效关键点 {valid_count} < {args.check}")
            frame_result["status"] = "check_failed"
            return frame_result, None, None

    # --valid-model
    if valid_model is not None:
        heatmaps_valid = _run_infer(image_rgb, valid_model, device)
        kpts_valid = _extract_keypoints(heatmaps_valid)
        frame_result["valid_keypoints"] = _kpts_to_dict(kpts_valid)
        ok, msg, val_details = validate_keypoints(
            heatmaps, heatmaps_valid,
            diff_threshold=args.valid_diff_threshold,
        )
        frame_result["validation_result"] = {"ok": ok, "message": msg, **val_details}
        if not ok:
            vprint(f"  [frame {frame_idx}] 验证失败：{msg}")
            frame_result["status"] = "validation_failed"
            return frame_result, None, None

    # 可视化（静默 challenge 内部 print）
    ctx = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()
    with ctx:
        vis_bgr, success, Hest = visualize(
            image_rgb, heatmaps,
            top_k=args.top_k,
            threshold=args.threshold,
        )

    frame_result["homography"] = Hest.tolist() if Hest is not None else None
    if not success:
        frame_result["status"] = "insufficient_points"
        vprint(f"  [frame {frame_idx}] 关键点不足，输出原图。")

    return frame_result, vis_bgr, Hest


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def process_video(args, vprint):
    stem            = Path(args.video).stem
    calibrated_path = _os.path.join(args.output_dir, f"{stem}_calibrated.mp4")
    json_path       = _os.path.join(args.output_dir, f"{stem}_results.json")
    _os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vprint(f"使用设备: {device}")

    vprint("加载主模型...")
    model, arch_info = load_model(args.model, device, verbose=args.verbose)

    valid_model = None
    valid_arch  = None
    if args.valid_model:
        vprint("加载验证模型...")
        valid_model, valid_arch = load_model(args.valid_model, device, verbose=args.verbose)

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        vprint(f"Error: 无法打开视频: {args.video}")
        sys.exit(1)

    orig_fps     = cap.get(cv2.CAP_PROP_FPS)
    orig_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vprint(f"视频: {orig_width}×{orig_height} @ {orig_fps:.2f}fps，共 {total_frames} 帧")
    vprint(f"抽帧步长: {args.frame_step}（每 {args.frame_step} 帧推理一次，中间帧沿用上次单应矩阵）")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(calibrated_path, fourcc, orig_fps, (IMG_WIDTH, IMG_HEIGHT))

    results   = []    # 仅包含被推理的关键帧
    last_hest = None  # 最近一次有效单应矩阵，供中间帧复用

    pbar = tqdm(total=total_frames, desc="Processing", disable=not args.verbose)

    for frame_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        image_rgb = _bgr_to_rgb_clahe(frame_bgr, args.clahe)

        if frame_idx % args.frame_step == 0:
            # ── 关键帧：执行完整推理 ──────────────────────────────────
            frame_result, vis_bgr, Hest = _process_one_frame(
                frame_idx, image_rgb, model, valid_model, device,
                args, args.verbose, vprint, arch_info, valid_arch,
            )
            results.append(frame_result)

            if vis_bgr is not None:
                last_hest = Hest      # 更新缓存的单应矩阵
                out_frame = vis_bgr
            else:
                # 推理失败：写入无叠加的原图（last_hest 不更新）
                out_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            # ── 中间帧：沿用上次单应矩阵重绘场地线 ──────────────────
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if last_hest is not None:
                ctx = contextlib.redirect_stdout(io.StringIO()) if not args.verbose \
                      else contextlib.nullcontext()
                with ctx:
                    drawField(bgr, last_hest, (0, 0, 255), (0, 0, 255), 2)
            out_frame = bgr

        writer.write(out_frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    vprint(f"\n处理完成！推理帧数: {len(results)} / {total_frames}")
    vprint(f"  标定视频: {calibrated_path}")
    vprint(f"  结果 JSON: {json_path}")


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="篮球视频标定：抽帧推理，输出标定视频和逐帧 JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video",
                        help="输入视频路径")
    parser.add_argument("--output-dir", required=True, metavar="DIR",
                        help="输出目录（必填）；文件命名为 <stem>_calibrated.mp4 / <stem>_results.json")
    parser.add_argument("--model", required=True, metavar="MODEL",
                        help="模型权重路径（必填）")
    parser.add_argument("--frame-step", type=int, default=1, metavar="N",
                        help="每隔 N 帧推理一次；中间帧沿用上次单应矩阵（默认 1 = 每帧推理）")
    parser.add_argument("--threshold", type=float, default=0.0, metavar="T",
                        help="置信度阈值，低于该值的关键点不参与计算")
    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help="取置信度最高的 K 个关键点计算单应矩阵")
    parser.add_argument("--check", type=int, default=None, metavar="N",
                        help="最少有效关键点数量；不足时该帧标记为 check_failed")
    parser.add_argument("--clahe", action="store_true",
                        help="开启 CLAHE 预处理（缓解光照差异）")
    parser.add_argument("--valid-model", default=None, metavar="MODEL",
                        help="验证模型路径；差异过大时该帧标记为 validation_failed")
    parser.add_argument("--valid-diff-threshold", type=float, default=20.0, metavar="PX",
                        help="两模型关键点平均像素距离上限（px）")
    parser.add_argument("--verbose", action="store_true",
                        help="打印进度信息（默认完全静默）")

    args = parser.parse_args()
    args.video      = _abs(args.video)
    args.output_dir = _abs(args.output_dir)
    args.model      = _abs(args.model)
    if args.valid_model:
        args.valid_model = _abs(args.valid_model)

    vprint = print if args.verbose else (lambda *a, **k: None)

    for label, path in [("视频文件", args.video), ("模型文件", args.model)]:
        if not _os.path.exists(path):
            vprint(f"Error: {label}不存在: {path}")
            sys.exit(1)
    if args.valid_model and not _os.path.exists(args.valid_model):
        vprint(f"Error: 验证模型不存在: {args.valid_model}")
        sys.exit(1)

    process_video(args, vprint)


if __name__ == "__main__":
    main()
