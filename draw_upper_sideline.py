#!/usr/bin/env python3
"""利用标定结果中 84 号点与上边线像素特征，启发式搜索并绘制球场上边线。

不使用单应矩阵。以 84 号点为锚，在斜率空间中搜索使边线像素得分最高的直线，
再沿该直线逐列精修后重新拟合，最终绘制直线段。
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_WIDTH = 960
IMG_HEIGHT = 540
ANCHOR_KPT = 84


def load_image(path: str) -> np.ndarray:
    return np.array(
        Image.open(path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    )


def load_anchor(json_path: str) -> tuple[float, float]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    kpt = data["keypoints"][str(ANCHOR_KPT)]
    return float(kpt["img_x"]), float(kpt["img_y"])


def _rgb_luminance(rgb: np.ndarray) -> float:
    return float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])


def _lab_mean(patch: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.mean(axis=(0, 1))


def _lab_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def sideline_pixel_score(img: np.ndarray, x: int, y: int, half_w: int = 3) -> float:
    """边线像素鲁棒得分：依赖上/线/下三段的颜色差异，而非绝对颜色。

    真实上边线典型特征：
    - 线上方（红色地板）与线位、线下方（木地板）在 Lab 空间差异明显
    - 线上方偏红，整体比下方更红
    - 线位相对更亮，且在局部竖直剖面里接近亮度峰值
  非边线区域（纯木地板、球员阴影、广告牌等）因颜色相近或过暗而被抑制。
    """
    if y < 12 or y >= IMG_HEIGHT - 12 or x < half_w or x >= IMG_WIDTH - half_w:
        return -1e9

    above = img[y - 10 : y - 4, x - half_w : x + half_w + 1]
    line = img[y - 3 : y + 3, x - half_w : x + half_w + 1]
    below = img[y + 4 : y + 11, x - half_w : x + half_w + 1]

    rgb_above = above.mean(axis=(0, 1))
    rgb_line = line.mean(axis=(0, 1))
    rgb_below = below.mean(axis=(0, 1))
    lab_above = _lab_mean(above)
    lab_line = _lab_mean(line)
    lab_below = _lab_mean(below)

    dist_above_line = _lab_dist(lab_above, lab_line)
    dist_line_below = _lab_dist(lab_line, lab_below)
    dist_above_below = _lab_dist(lab_above, lab_below)

    # 三段两两都要有可见色差，且上下差异是边线存在的前提
    contrast = min(dist_above_line, dist_line_below) + 0.35 * dist_above_below
    if contrast < 15.0 or dist_above_below < 10.0:
        return -1e9

    lum_above = _rgb_luminance(rgb_above)
    lum_line = _rgb_luminance(rgb_line)
    lum_below = _rgb_luminance(rgb_below)
    if min(lum_above, lum_line, lum_below) < 35.0:
        return -1e9

    # 上方比下方更红；线位比上下邻居更亮
    red_above = float(rgb_above[0] - 0.5 * (rgb_above[1] + rgb_above[2]))
    red_gap = float(rgb_above[0] - rgb_below[0])
    bright_line = lum_line - max(lum_above, lum_below)

    column = img[y - 10 : y + 11, x - half_w : x + half_w + 1].astype(np.float32)
    profile = (
        0.299 * column[:, :, 0].mean(axis=1)
        + 0.587 * column[:, :, 1].mean(axis=1)
        + 0.114 * column[:, :, 2].mean(axis=1)
    )
    local_peak = float(profile[10] - 0.5 * (profile[7] + profile[13]))

    # 线位过暗多为球员/阴影误检
    dark_line_penalty = max(0.0, 55.0 - lum_line) * 1.5

    return (
        0.45 * contrast
        + 0.20 * max(0.0, red_above)
        + 0.15 * max(0.0, red_gap)
        + 0.12 * max(0.0, bright_line)
        + 0.08 * max(0.0, local_peak)
        - dark_line_penalty
    )


def score_line_slope(
    img: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    slope: float,
    x_min: int = 0,
    x_max: int = IMG_WIDTH - 1,
    step: int = 3,
) -> float:
    """过锚点的直线 y = anchor_y + slope*(x - anchor_x) 的边线像素平均得分。"""
    total, count = 0.0, 0
    for x in range(x_min, x_max + 1, step):
        y = anchor_y + slope * (x - anchor_x)
        yi = int(round(y))
        if not 0 <= yi < IMG_HEIGHT:
            continue
        s = sideline_pixel_score(img, x, yi)
        if s > -1e8:
            total += s
            count += 1
    return total / max(count, 1)


def search_best_slope(
    img: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    slope_min: float = -0.12,
    slope_max: float = 0.02,
    coarse_step: float = 0.002,
    fine_half_window: float = 0.006,
    fine_step: float = 0.0002,
) -> float:
    """两阶段启发式搜索：粗搜斜率 → 细搜斜率。"""
    best_slope, best_score = 0.0, -1e9
    s = slope_min
    while s <= slope_max + 1e-9:
        sc = score_line_slope(img, anchor_x, anchor_y, s)
        if sc > best_score:
            best_score, best_slope = sc, s
        s += coarse_step

    lo = best_slope - fine_half_window
    hi = best_slope + fine_half_window
    s = lo
    while s <= hi + 1e-9:
        sc = score_line_slope(img, anchor_x, anchor_y, s)
        if sc > best_score:
            best_score, best_slope = sc, s
        s += fine_step

    return best_slope


def column_best_score(
    img: np.ndarray,
    x: int,
    y_init: float,
    search_radius: int,
) -> tuple[int, float]:
    y_lo = int(max(0, round(y_init) - search_radius))
    y_hi = int(min(IMG_HEIGHT - 1, round(y_init) + search_radius))
    best_y, best_score = int(round(y_init)), -1e9
    for y in range(y_lo, y_hi + 1):
        sc = sideline_pixel_score(img, x, y)
        if sc > best_score:
            best_score, best_y = sc, y
    return best_y, best_score


def expand_line_from_anchor(
    img: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    slope: float,
    search_radius: int = 10,
    score_threshold: float = 38.0,
    max_gap: int = 30,
) -> list[tuple[int, int]]:
    """从锚点沿直线向两侧扩展，允许短暂遮挡造成的小间断。"""
    ax = int(round(anchor_x))
    pts: dict[int, int] = {}

    for direction in (-1, 1):
        gap = 0
        x = ax
        while 0 <= x < IMG_WIDTH:
            y_init = anchor_y + slope * (x - anchor_x)
            best_y, best_score = column_best_score(img, x, y_init, search_radius)
            if best_score >= score_threshold:
                pts[x] = best_y
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            x += direction

    return [(x, pts[x]) for x in sorted(pts)]


def fit_line_through_anchor(
    pts: list[tuple[int, int]],
    anchor_x: float,
    anchor_y: float,
) -> tuple[float, float]:
    """最小二乘拟合直线，并平移使直线经过锚点。"""
    if len(pts) < 2:
        return anchor_x, anchor_y

    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)

    # 平移截距，使 (anchor_x, anchor_y) 在直线上
    intercept = anchor_y - slope * anchor_x
    return float(slope), float(intercept)


def trim_right_by_score_run(
    img: np.ndarray,
    slope: float,
    intercept: float,
    x_right: int,
    anchor_x: float,
    search_radius: int = 10,
    min_score: float = 42.0,
    run: int = 18,
) -> int:
    """从右端向左收缩，要求末端连续若干列的边线得分足够高。"""
    x = x_right
    ax = int(round(anchor_x))
    while x > ax:
        ok = True
        for i in range(run):
            xi = x - i
            if xi < 0:
                ok = False
                break
            _, sc = column_best_score(img, xi, slope * xi + intercept, search_radius)
            if sc < min_score:
                ok = False
                break
        if ok:
            return x
        x -= 1
    return x_right


def line_endpoints(
    img: np.ndarray,
    slope: float,
    intercept: float,
    edge_pts: list[tuple[int, int]],
    anchor_x: float,
    search_radius: int = 10,
    margin: int = 5,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """根据有效边缘点确定直线在图像内的起止端点。"""
    if not edge_pts:
        return None

    xs = [p[0] for p in edge_pts]
    x0 = max(0, min(xs) - margin)
    x1 = min(IMG_WIDTH - 1, max(xs) + margin)
    x1 = trim_right_by_score_run(
        img, slope, intercept, x1, anchor_x, search_radius=search_radius
    )

    y0 = int(round(slope * x0 + intercept))
    y1 = int(round(slope * x1 + intercept))
    y0 = int(np.clip(y0, 0, IMG_HEIGHT - 1))
    y1 = int(np.clip(y1, 0, IMG_HEIGHT - 1))
    return (x0, y0), (x1, y1)


def detect_upper_sideline(
    img: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    search_radius: int = 10,
) -> tuple[float, float, list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]] | None]:
    """启发式检测上边线：搜斜率 → 从锚点扩展采点 → 拟合直线。"""
    slope = search_best_slope(img, anchor_x, anchor_y)
    edge_pts = expand_line_from_anchor(
        img, anchor_x, anchor_y, slope, search_radius=search_radius
    )
    slope, intercept = fit_line_through_anchor(edge_pts, anchor_x, anchor_y)
    endpoints = line_endpoints(
        img, slope, intercept, edge_pts, anchor_x, search_radius=search_radius
    )
    return slope, intercept, edge_pts, endpoints


def draw_upper_sideline(
    img_rgb: np.ndarray,
    endpoints: tuple[tuple[int, int], tuple[int, int]] | None,
    anchor_x: float,
    anchor_y: float,
    edge_pts: list[tuple[int, int]] | None = None,
    show_samples: bool = False,
) -> np.ndarray:
    out = img_rgb.copy()

    if endpoints is not None:
        cv2.line(
            out,
            endpoints[0],
            endpoints[1],
            color=(0, 255, 255),
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    if show_samples and edge_pts:
        for x, y in edge_pts[::8]:
            cv2.circle(out, (x, y), 2, (0, 200, 255), -1, cv2.LINE_AA)

    ax, ay = int(round(anchor_x)), int(round(anchor_y))
    cv2.circle(out, (ax, ay), 7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(out, (ax, ay), 5, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"#{ANCHOR_KPT}",
        (ax + 8, ay - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"#{ANCHOR_KPT}",
        (ax + 8, ay - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="启发式搜索上边线（不使用单应矩阵）并绘制"
    )
    parser.add_argument("--image", default="images/image5.png")
    parser.add_argument("--result-json", default="results/image5_result.json")
    parser.add_argument("--output", default="results/image5_upper_sideline.png")
    parser.add_argument(
        "--search-radius",
        type=int,
        default=10,
        help="逐列精修时相对初始直线的纵向搜索半径（像素）",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="叠加显示采样到的边缘点",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    image_path = root / args.image
    json_path = root / args.result_json
    output_path = root / args.output

    if not image_path.exists():
        raise SystemExit(f"图像不存在: {image_path}")
    if not json_path.exists():
        raise SystemExit(f"JSON 不存在: {json_path}")

    img = load_image(str(image_path))
    anchor_x, anchor_y = load_anchor(str(json_path))

    slope, intercept, edge_pts, endpoints = detect_upper_sideline(
        img, anchor_x, anchor_y, search_radius=args.search_radius
    )

    vis = draw_upper_sideline(
        img,
        endpoints,
        anchor_x,
        anchor_y,
        edge_pts=edge_pts,
        show_samples=args.show_samples,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"锚点 #{ANCHOR_KPT}: ({anchor_x:.1f}, {anchor_y:.1f}) px")
    print(f"启发式搜索斜率: {slope:.5f}  (y = {slope:.5f} * x + {intercept:.2f})")
    print(f"边缘采样点: {len(edge_pts)}")
    if endpoints:
        print(f"直线端点: {endpoints[0]} → {endpoints[1]}")
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
