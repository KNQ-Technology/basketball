#!/usr/bin/env python3
"""以指定关键点为中心，在原图上绘制边长为 --pixel 的正方形。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_WIDTH = 960
IMG_HEIGHT = 540
DEFAULT_KPT = 85


def load_image(path: str) -> np.ndarray:
    return np.array(
        Image.open(path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    )


def load_keypoint(json_path: str, kpt_id: int) -> tuple[float, float]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    kpt = data["keypoints"][str(kpt_id)]
    return float(kpt["img_x"]), float(kpt["img_y"])


def square_bounds(cx: float, cy: float, side: int) -> tuple[int, int, int, int]:
    """返回以 (cx, cy) 为中心、边长 side 的正方形整数边界 (x0, y0, x1, y1)。"""
    half = side / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + side
    y1 = y0 + side
    return x0, y0, x1, y1


def draw_square(
    img_rgb: np.ndarray,
    cx: float,
    cy: float,
    side: int,
    kpt_id: int,
) -> np.ndarray:
    out = img_rgb.copy()
    x0, y0, x1, y1 = square_bounds(cx, cy, side)

    cv2.rectangle(out, (x0, y0), (x1, y1), color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    center = (int(round(cx)), int(round(cy)))
    cv2.circle(out, center, 5, (255, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(out, center, 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(
        out,
        str(kpt_id),
        (center[0] + 8, center[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        str(kpt_id),
        (center[0] + 8, center[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="以关键点为中心，在原图上绘制正方形区域"
    )
    parser.add_argument(
        "--pixel",
        type=int,
        required=True,
        metavar="N",
        help="正方形边长（像素）",
    )
    parser.add_argument("--image", default="images/image5.png", help="输入图像路径")
    parser.add_argument(
        "--result-json",
        default="results/image5_result.json",
        help="标定结果 JSON",
    )
    parser.add_argument(
        "--keypoint",
        type=int,
        default=DEFAULT_KPT,
        help=f"关键点编号（默认 {DEFAULT_KPT}）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出路径（默认 results/<image>_kpt<K>_square<P>.png）",
    )
    args = parser.parse_args()

    if args.pixel <= 0:
        raise SystemExit("Error: --pixel 必须为正整数")

    root = Path(__file__).resolve().parent
    image_path = root / args.image
    json_path = root / args.result_json

    if not image_path.exists():
        raise SystemExit(f"Error: 图像不存在: {image_path}")
    if not json_path.exists():
        raise SystemExit(f"Error: JSON 不存在: {json_path}")

    cx, cy = load_keypoint(str(json_path), args.keypoint)
    img = load_image(str(image_path))
    vis = draw_square(img, cx, cy, args.pixel, args.keypoint)

    if args.output:
        output_path = root / args.output
    else:
        stem = image_path.stem
        output_path = (
            root / "results" / f"{stem}_kpt{args.keypoint}_square{args.pixel}.png"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    x0, y0, x1, y1 = square_bounds(cx, cy, args.pixel)
    print(f"关键点 #{args.keypoint}: ({cx:.2f}, {cy:.2f}) px")
    print(f"正方形边长: {args.pixel} px, 边界: ({x0}, {y0}) – ({x1}, {y1})")
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
