#!/usr/bin/env python3
"""两阶段关键点 RGB/value 筛选，全图伪标签 + RGB 软间隔 SVM，绘制决策边界。

流程：
1. 第一组关键点按像素 RGB 迭代剔除离群点，得到 class0 原型
2. 第二组关键点在正方形内计算 value，迭代剔除离群点，得到 class1 原型
3. 全图每个像素按与两原型的 RGB 欧式距离打伪标签
4. 在 RGB 空间训练软间隔线性 SVM，提取并绘制 decision=0 决策边界
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.svm import LinearSVC
from skimage import measure

IMG_WIDTH = 960
IMG_HEIGHT = 540

KEYPOINT_IDS = [
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    55, 56, 57, 59, 60, 61,
    42, 43, 44, 46, 47, 48,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
]

SECOND_KEYPOINT_IDS = [
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 77,
    64, 51, 38, 25, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    13, 26, 39, 52, 65,
]


@dataclass
class KeypointSample:
    kpt_id: int
    x: float
    y: float
    rgb: np.ndarray


def load_image(path: str) -> np.ndarray:
    return np.array(
        Image.open(path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    )


def resolve_image_path(root: Path, image_arg: str) -> Path:
    path = root / image_arg
    if path.exists():
        return path
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = root / f"{image_arg}{ext}"
        if candidate.exists():
            return candidate
    return path


def load_keypoints(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("keypoints", {})


def load_keypoint_samples(
    keypoints: dict, img: np.ndarray, kpt_ids: list[int]
) -> list[KeypointSample]:
    samples: list[KeypointSample] = []
    for kpt_id in kpt_ids:
        key = str(kpt_id)
        if key not in keypoints:
            continue
        kpt = keypoints[key]
        x = float(kpt["img_x"])
        y = float(kpt["img_y"])
        px = int(round(x))
        py = int(round(y))
        px = max(0, min(IMG_WIDTH - 1, px))
        py = max(0, min(IMG_HEIGHT - 1, py))
        rgb = img[py, px].astype(np.float64)
        samples.append(KeypointSample(kpt_id=kpt_id, x=x, y=y, rgb=rgb))
    return samples


def mean_rgb(samples: list[KeypointSample]) -> np.ndarray:
    return np.mean([s.rgb for s in samples], axis=0)


def farthest_index(samples: list[KeypointSample], center: np.ndarray) -> int:
    distances = [float(np.linalg.norm(s.rgb - center)) for s in samples]
    return int(np.argmax(distances))


def select_keypoint_by_rgb(samples: list[KeypointSample]) -> KeypointSample:
    """迭代剔除与当前 RGB 均值欧式距离最远的点，直到只剩一个。"""
    remaining = list(samples)
    if len(remaining) == 1:
        return remaining[0]

    while len(remaining) > 1:
        center = mean_rgb(remaining)
        idx = farthest_index(remaining, center)
        remaining.pop(idx)
    return remaining[0]


def square_bounds(cx: float, cy: float, side: int) -> tuple[int, int, int, int]:
    half = side / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + side
    y1 = y0 + side
    return x0, y0, x1, y1


def clip_square_to_image(
    x0: int, y0: int, x1: int, y1: int
) -> tuple[int, int, int, int]:
    x0 = max(0, min(IMG_WIDTH, x0))
    x1 = max(0, min(IMG_WIDTH, x1))
    y0 = max(0, min(IMG_HEIGHT, y0))
    y1 = max(0, min(IMG_HEIGHT, y1))
    return x0, y0, x1, y1


def compute_square_value(
    img: np.ndarray,
    cx: float,
    cy: float,
    side: int,
    ref_rgb: np.ndarray,
    threshold: float,
) -> np.ndarray | None:
    """正方形内，与 ref_rgb 距离大于 threshold 的像素 RGB 均值。"""
    x0, y0, x1, y1 = square_bounds(cx, cy, side)
    x0, y0, x1, y1 = clip_square_to_image(x0, y0, x1, y1)
    if x1 <= x0 or y1 <= y0:
        return None

    patch = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    distances = np.linalg.norm(patch - ref_rgb, axis=1)
    mask = distances > threshold
    if not np.any(mask):
        return None
    return np.mean(patch[mask], axis=0)


def load_value_samples(
    keypoints: dict,
    img: np.ndarray,
    kpt_ids: list[int],
    side: int,
    ref_rgb: np.ndarray,
    threshold: float,
) -> list[KeypointSample]:
    samples: list[KeypointSample] = []
    for kpt_id in kpt_ids:
        key = str(kpt_id)
        if key not in keypoints:
            continue
        kpt = keypoints[key]
        x = float(kpt["img_x"])
        y = float(kpt["img_y"])
        value = compute_square_value(img, x, y, side, ref_rgb, threshold)
        if value is None:
            continue
        samples.append(KeypointSample(kpt_id=kpt_id, x=x, y=y, rgb=value))
    return samples


def pseudo_label_pixels(
    pixels: np.ndarray,
    class0_rgb: np.ndarray,
    class1_rgb: np.ndarray,
) -> np.ndarray:
    """距 class0_rgb 更近为 0 类，距 class1_rgb 更近为 1 类。"""
    dist0 = np.sum((pixels - class0_rgb) ** 2, axis=1)
    dist1 = np.sum((pixels - class1_rgb) ** 2, axis=1)
    return (dist1 < dist0).astype(np.int32)


def fit_rgb_linear_svm(pixels: np.ndarray, labels: np.ndarray, C: float) -> LinearSVC:
    clf = LinearSVC(C=C, max_iter=20_000, dual="auto", random_state=0)
    clf.fit(pixels, labels)
    return clf


def decision_map_from_clf(img: np.ndarray, clf: LinearSVC) -> np.ndarray:
    pixels = img.reshape(-1, 3).astype(np.float64)
    return clf.decision_function(pixels).reshape(img.shape[:2])


def extract_decision_contours(decision: np.ndarray, level: float = 0.0) -> list[np.ndarray]:
    contours = measure.find_contours(decision, level)
    out = []
    for contour in contours:
        xs = np.clip(contour[:, 1], 0, decision.shape[1] - 1)
        ys = np.clip(contour[:, 0], 0, decision.shape[0] - 1)
        if len(xs) >= 2:
            out.append(np.stack([xs, ys], axis=1))
    return out


def draw_selected_point(
    img_rgb: np.ndarray,
    sample: KeypointSample,
    color: tuple[int, int, int],
    label_prefix: str = "",
) -> None:
    center = (int(round(sample.x)), int(round(sample.y)))
    cv2.circle(img_rgb, center, 8, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img_rgb, center, 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    label = f"{label_prefix}{sample.kpt_id}"
    cv2.putText(
        img_rgb,
        label,
        (center[0] + 10, center[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_rgb,
        label,
        (center[0] + 10, center[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_result(
    img_rgb: np.ndarray,
    selected: KeypointSample,
    selected_second: KeypointSample,
    contours: list[np.ndarray],
    show_contours: bool,
) -> np.ndarray:
    out = img_rgb.copy()

    if show_contours:
        for contour in contours:
            pts = contour.astype(np.int32)
            cv2.polylines(
                out,
                [pts],
                isClosed=False,
                color=(255, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    draw_selected_point(out, selected, (255, 0, 0), "P1:")
    draw_selected_point(out, selected_second, (0, 255, 0), "P2:")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="两阶段关键点筛选 + RGB 软间隔 SVM 决策边界"
    )
    parser.add_argument(
        "--image",
        default="images/image5",
        help="输入图像路径（默认 images/image5）",
    )
    parser.add_argument(
        "--result-json",
        default="results/image5_result.json",
        help="标注点 JSON（默认 results/image5_result.json）",
    )
    parser.add_argument(
        "--pixel",
        type=int,
        default=40,
        help="第二组关键点正方形边长（像素，默认 40）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        help="正方形内像素与第一阶段选中点 RGB 距离阈值（默认 40）",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="软间隔 SVM 惩罚系数（默认 1.0）",
    )
    parser.add_argument(
        "--no-contours",
        action="store_true",
        help="不绘制全图 decision=0 决策边界",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出路径（默认 results/<image>_rgb_svm_boundary.png）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="在控制台打印运行详情（默认静默）",
    )
    args = parser.parse_args()

    if args.pixel <= 0:
        raise SystemExit("Error: --pixel 必须为正整数")

    root = Path(__file__).resolve().parent
    image_path = resolve_image_path(root, args.image)
    json_path = root / args.result_json

    if not image_path.exists():
        raise SystemExit(f"Error: 图像不存在: {image_path}")
    if not json_path.exists():
        raise SystemExit(f"Error: JSON 不存在: {json_path}")

    img = load_image(str(image_path))
    keypoints = load_keypoints(json_path)

    samples = load_keypoint_samples(keypoints, img, KEYPOINT_IDS)
    if not samples:
        raise SystemExit("Error: 第一组关键点在 JSON 中均未识别到")

    selected = select_keypoint_by_rgb(samples)
    class0_rgb = selected.rgb

    value_samples = load_value_samples(
        keypoints,
        img,
        SECOND_KEYPOINT_IDS,
        args.pixel,
        class0_rgb,
        args.threshold,
    )
    if not value_samples:
        raise SystemExit(
            "Error: 第二组关键点均无有效 value（未识别或正方形内无满足阈值的像素）"
        )
    selected_second = select_keypoint_by_rgb(value_samples)
    class1_rgb = selected_second.rgb

    pixels = img.reshape(-1, 3).astype(np.float64)
    labels = pseudo_label_pixels(pixels, class0_rgb, class1_rgb)
    clf = fit_rgb_linear_svm(pixels, labels, C=args.C)

    decision = decision_map_from_clf(img, clf)
    contours = extract_decision_contours(decision, level=0.0)

    vis = draw_result(
        img,
        selected,
        selected_second,
        contours,
        show_contours=not args.no_contours,
    )

    if args.output:
        output_path = root / args.output
    else:
        output_path = root / "results" / f"{image_path.stem}_rgb_svm_boundary.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    wr, wg, wb = clf.coef_[0]
    b = float(clf.intercept_[0])

    if args.verbose:
        print(f"输入图像: {image_path}")
        print(f"标注 JSON: {json_path}")
        print(f"正方形边长: {args.pixel} px, RGB 距离阈值: {args.threshold}")
        print(f"第一组可用关键点: {len(samples)} / {len(KEYPOINT_IDS)}")
        print(
            f"第一阶段选中 #{selected.kpt_id}: "
            f"({selected.x:.2f}, {selected.y:.2f}), "
            f"class0 RGB=({class0_rgb[0]:.0f}, {class0_rgb[1]:.0f}, {class0_rgb[2]:.0f})"
        )
        print(f"第二组有效 value 关键点: {len(value_samples)} / {len(SECOND_KEYPOINT_IDS)}")
        print(
            f"第二阶段选中 #{selected_second.kpt_id}: "
            f"({selected_second.x:.2f}, {selected_second.y:.2f}), "
            f"class1 value=({class1_rgb[0]:.0f}, {class1_rgb[1]:.0f}, {class1_rgb[2]:.0f})"
        )
        print(f"全图伪标签: class0={n0}, class1={n1}")
        print(f"SVM 权重 RGB: [{wr:.4f}, {wg:.4f}, {wb:.4f}], intercept={b:.4f}")
        print(f"决策边界段数: {len(contours)}")
        print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
