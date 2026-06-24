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

CM_PER_METER = 100.0
CM_PER_FOOT = 30.48
NBA_FIELD_LENGTH = 28.65 * CM_PER_METER
NBA_FIELD_WIDTH = 15.24 * CM_PER_METER
NBA_FREE_THROW_X = 5.80 * CM_PER_METER
NBA_LANE_WIDTH = 4.90 * CM_PER_METER

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


@dataclass
class SidelineDetection:
    side: str
    keypoint_ids: list[int]
    anchor_points: np.ndarray
    candidate_points: np.ndarray
    line_points: np.ndarray
    endpoints: tuple[tuple[int, int], tuple[int, int]] | None
    line: np.ndarray | None
    score: float


SIDELINE_KEYPOINT_IDS = {
    "upper": list(range(78, 91)),
    "lower": list(range(0, 13)),
    "left": [0, 13, 26, 39, 52, 65, 78],
    "right": [12, 25, 38, 51, 64, 77, 90],
}

SIDELINE_COLORS = {
    "upper": (255, 255, 0),
    "lower": (0, 255, 255),
    "left": (0, 255, 0),
    "right": (255, 80, 80),
}

PAINT_KEYPOINT_GROUPS = {
    "left_paint": {
        "upper": [39, 40, 41, 42],
        "lower": [52, 53, 54, 55],
        "left": [39, 52],
        "right": [42, 55],
    },
    "right_paint": {
        "upper": [48, 49, 50, 51],
        "lower": [61, 62, 63, 64],
        "left": [48, 61],
        "right": [51, 64],
    },
}

PAINT_CORNER_GROUPS = {
    "left_paint": {
        "upper": [91, 92],
        "lower": [94, 93],
        "left": [91, 94],
        "right": [92, 93],
    },
    "right_paint": {
        "upper": [96, 95],
        "lower": [97, 98],
        "left": [96, 97],
        "right": [95, 98],
    },
}

PAINT_COLORS = {
    "upper": (255, 160, 0),
    "lower": (255, 0, 160),
    "left": (80, 255, 80),
    "right": (80, 160, 255),
}


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


def collect_contour_points(contours: list[np.ndarray]) -> np.ndarray:
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(contours).astype(np.float64)


def load_sideline_anchor_points(keypoints: dict, kpt_ids: list[int]) -> np.ndarray:
    points = []
    for kpt_id in kpt_ids:
        kpt = keypoints.get(str(kpt_id))
        if not kpt:
            continue
        points.append([float(kpt["img_x"]), float(kpt["img_y"])])
    return np.array(points, dtype=np.float64)


def get_field_points() -> np.ndarray:
    original_length = 2800.0
    original_width = 1500.0
    original_row_y = np.array([1500.0, 1325.0, 1120.0, 885.0, 620.0, 325.0, 0.0])
    row_y = original_row_y / original_width * NBA_FIELD_WIDTH

    points = []
    for y in row_y:
        for i in range(13):
            points.append([i * NBA_FIELD_LENGTH / 12.0, y, 0.0])

    lane_top_y = NBA_FIELD_WIDTH / 2 - NBA_LANE_WIDTH / 2
    lane_bottom_y = NBA_FIELD_WIDTH / 2 + NBA_LANE_WIDTH / 2
    points.extend(
        [
            [0.0, lane_top_y, 0.0],
            [NBA_FREE_THROW_X, lane_top_y, 0.0],
            [NBA_FREE_THROW_X, lane_bottom_y, 0.0],
            [0.0, lane_bottom_y, 0.0],
            [NBA_FIELD_LENGTH, lane_top_y, 0.0],
            [NBA_FIELD_LENGTH - NBA_FREE_THROW_X, lane_top_y, 0.0],
            [NBA_FIELD_LENGTH - NBA_FREE_THROW_X, lane_bottom_y, 0.0],
            [NBA_FIELD_LENGTH, lane_bottom_y, 0.0],
        ]
    )
    return np.array(points, dtype=np.float64)


def estimate_coarse_homography_from_keypoints(keypoints: dict) -> np.ndarray | None:
    field_points = get_field_points()[:91, :2]
    src = []
    dst = []
    for idx in range(91):
        kpt = keypoints.get(str(idx))
        if not kpt:
            continue
        src.append(field_points[idx])
        dst.append([float(kpt["img_x"]), float(kpt["img_y"])])

    if len(src) < 4:
        return None

    H, inlier_mask = cv2.findHomography(
        np.array(src, dtype=np.float64),
        np.array(dst, dtype=np.float64),
        cv2.RANSAC,
        ransacReprojThreshold=10.0,
        maxIters=3000,
        confidence=0.995,
    )
    if H is None or inlier_mask is None or int(np.count_nonzero(inlier_mask)) < 4:
        return None
    return H.astype(np.float64)


def project_field_points(H: np.ndarray, point_ids: list[int]) -> np.ndarray:
    field_points = get_field_points()[point_ids, :2].astype(np.float32)
    projected = cv2.perspectiveTransform(field_points.reshape(1, -1, 2), H)[0]
    return projected.astype(np.float64)


def fit_line_pca(points: np.ndarray) -> np.ndarray | None:
    """拟合归一化直线 ax + by + c = 0。"""
    if len(points) < 2:
        return None

    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    normal /= norm
    c = -float(np.dot(normal, center))
    return np.array([normal[0], normal[1], c], dtype=np.float64)


def line_distances(line: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.abs(points @ line[:2] + line[2])


def line_signed_values(line: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ line[:2] + line[2]


def robust_fit_line(points: np.ndarray, trim_px: float = 35.0) -> np.ndarray | None:
    if len(points) < 2:
        return None

    active = points.astype(np.float64)
    line = fit_line_pca(active)
    if line is None:
        return None

    for _ in range(3):
        distances = line_distances(line, active)
        if len(active) <= 3:
            break
        cutoff = max(trim_px, float(np.percentile(distances, 70)))
        keep = distances <= cutoff
        if np.count_nonzero(keep) < 2 or np.all(keep):
            break
        active = active[keep]
        next_line = fit_line_pca(active)
        if next_line is None:
            break
        line = next_line
    return line


def line_from_hough(rho: float, theta: float) -> np.ndarray:
    normal = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    return np.array([normal[0], normal[1], -rho], dtype=np.float64)


def line_score_by_candidate_count(
    line: np.ndarray,
    points: np.ndarray,
    inlier_px: float,
) -> tuple[int, float, float, np.ndarray]:
    distances = line_distances(line, points)
    mask = distances <= inlier_px
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0, 0.0, float("inf"), mask

    inliers = points[mask]
    span = float(np.ptp(inliers @ line_direction(line))) if count >= 2 else 0.0
    residual = float(np.mean(distances[mask]))
    return count, span, residual, mask


def rasterize_candidate_points(points: np.ndarray) -> np.ndarray:
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    xy = np.rint(points).astype(np.int32)
    xy[:, 0] = np.clip(xy[:, 0], 0, IMG_WIDTH - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, IMG_HEIGHT - 1)
    mask[xy[:, 1], xy[:, 0]] = 255
    return mask


def hough_candidate_lines(points: np.ndarray, max_lines: int = 240) -> list[np.ndarray]:
    if len(points) < 2:
        return []

    mask = rasterize_candidate_points(points)
    unique_points = int(np.count_nonzero(mask))
    if unique_points < 2:
        return []

    thresholds = [
        max(18, min(120, unique_points // 8)),
        max(14, min(90, unique_points // 12)),
        max(10, min(70, unique_points // 18)),
    ]

    lines: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for threshold in thresholds:
        raw_lines = cv2.HoughLines(mask, 1, np.pi / 720.0, threshold)
        if raw_lines is None:
            continue
        for raw_line in raw_lines[:max_lines]:
            rho, theta = (float(v) for v in raw_line[0])
            key = (int(round(rho)), int(round(theta * 1000)))
            if key in seen:
                continue
            seen.add(key)
            lines.append(line_from_hough(rho, theta))
            if len(lines) >= max_lines:
                return lines
        if lines:
            break
    return lines


def ransac_candidate_lines(
    points: np.ndarray,
    iterations: int = 300,
    min_pair_dist: float = 40.0,
) -> list[np.ndarray]:
    if len(points) < 2:
        return []

    rng = np.random.default_rng(0)
    lines = []
    n = len(points)
    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False)
        p0 = points[i]
        p1 = points[j]
        if np.linalg.norm(p1 - p0) < min_pair_dist:
            continue
        line = fit_line_pca(np.array([p0, p1], dtype=np.float64))
        if line is not None:
            lines.append(line)
    return lines


def fit_line_from_candidate_points(
    points: np.ndarray,
    anchors: np.ndarray,
    inlier_px: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray]:
    """选择覆盖候选点最多的直线；关键点不参与最终打分。"""
    _ = anchors
    if len(points) < 2:
        return None, np.empty((0, 2), dtype=np.float64)

    candidate_lines = hough_candidate_lines(points)
    if not candidate_lines:
        candidate_lines = ransac_candidate_lines(points)

    best_line = None
    best_mask = None
    best_key = (-1, -1.0, -float("inf"))

    for line in candidate_lines:
        count, span, residual, mask = line_score_by_candidate_count(
            line, points, inlier_px
        )
        if count < 2:
            continue
        score_key = (count, span, -residual)
        if score_key > best_key:
            best_key = score_key
            best_line = line
            best_mask = mask

    if best_line is None or best_mask is None:
        fallback = robust_fit_line(points, trim_px=8.0)
        if fallback is None:
            return None, np.empty((0, 2), dtype=np.float64)
        mask = line_distances(fallback, points) <= inlier_px
        return fallback, points[mask]

    inliers = points[best_mask]
    refined = robust_fit_line(inliers, trim_px=max(inlier_px, 3.0))
    if refined is not None:
        refined_count, refined_span, refined_residual, refined_mask = (
            line_score_by_candidate_count(refined, points, inlier_px)
        )
        refined_key = (refined_count, refined_span, -refined_residual)
        if refined_key >= best_key:
            best_line = refined
            inliers = points[refined_mask]

    return best_line, inliers


def _line_alignment(line_a: np.ndarray, line_b: np.ndarray) -> float:
    return float(abs(np.dot(line_direction(line_a), line_direction(line_b))))


def _guided_line_score(
    line: np.ndarray,
    points: np.ndarray,
    anchors: np.ndarray,
    expected_line: np.ndarray,
    inlier_px: float,
) -> tuple[float, np.ndarray]:
    count, span, residual, mask = line_score_by_candidate_count(line, points, inlier_px)
    if count < 2:
        return -float("inf"), mask

    anchor_distances = line_distances(line, anchors)
    anchor_mean = float(np.mean(anchor_distances)) if len(anchor_distances) else 0.0
    anchor_max = float(np.max(anchor_distances)) if len(anchor_distances) else 0.0
    alignment = _line_alignment(line, expected_line)

    score = (
        count
        + 0.25 * span
        - 4.0 * residual
        - 7.0 * anchor_mean
        - 3.0 * anchor_max
        - 120.0 * (1.0 - alignment)
    )
    return float(score), mask


def fit_line_from_candidate_points_guided(
    points: np.ndarray,
    anchors: np.ndarray,
    expected_line: np.ndarray,
    inlier_px: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray]:
    if len(anchors) < 2:
        return fit_line_from_candidate_points(points, anchors, inlier_px=inlier_px)

    candidate_lines = hough_candidate_lines(points)
    candidate_lines.extend(ransac_candidate_lines(points))
    candidate_lines.append(expected_line)

    best_line = None
    best_mask = None
    best_key = (-1, -1.0, -float("inf"))

    for line in candidate_lines:
        if line is None:
            continue
        count, span, residual, mask = line_score_by_candidate_count(
            line,
            points,
            inlier_px,
        )
        if count < 2:
            continue

        score_key = (count, span, -residual)
        if score_key > best_key:
            best_key = score_key
            best_line = line
            best_mask = mask

    if best_line is None or best_mask is None:
        mask = line_distances(expected_line, points) <= max(inlier_px, 4.0)
        return expected_line, points[mask]

    inliers = points[best_mask]
    if len(inliers) >= 2:
        refined = robust_fit_line(inliers, trim_px=max(inlier_px, 4.0))
        if refined is not None:
            refined_count, refined_span, refined_residual, refined_mask = line_score_by_candidate_count(
                refined,
                points,
                inlier_px,
            )
            refined_key = (refined_count, refined_span, -refined_residual)
            if refined_key >= best_key:
                best_line = refined
                inliers = points[refined_mask]

    return best_line, inliers


def line_direction(line: np.ndarray) -> np.ndarray:
    direction = np.array([line[1], -line[0]], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.array([1.0, 0.0], dtype=np.float64)
    return direction / norm


def points_near_line_span(
    points: np.ndarray,
    line: np.ndarray,
    span_points: np.ndarray,
    distance_px: float = 12.0,
    margin_px: float = 80.0,
) -> np.ndarray:
    if len(points) == 0 or len(span_points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    direction = line_direction(line)
    span_t = span_points @ direction
    point_t = points @ direction
    distances = line_distances(line, points)
    mask = (
        (distances <= distance_px)
        & (point_t >= float(span_t.min()) - margin_px)
        & (point_t <= float(span_t.max()) + margin_px)
    )
    return points[mask]


def sample_line_points(
    line: np.ndarray,
    span_points: np.ndarray,
    margin_px: float = 20.0,
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]] | None]:
    if len(span_points) < 2:
        return np.empty((0, 2), dtype=np.int32), None

    normal = line[:2]
    direction = line_direction(line)
    base = -line[2] * normal
    span_t = span_points @ direction
    lo = float(span_t.min()) - margin_px
    hi = float(span_t.max()) + margin_px
    steps = max(2, int(round(hi - lo)) + 1)
    ts = np.linspace(lo, hi, steps)
    pts = base[None, :] + ts[:, None] * direction[None, :]
    pts = np.rint(pts).astype(np.int32)

    in_image = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < IMG_WIDTH)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < IMG_HEIGHT)
    )
    pts = pts[in_image]
    if len(pts) == 0:
        return pts, None

    _, unique_idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(unique_idx)]
    if len(pts) < 2:
        return pts, None

    endpoints = (
        (int(pts[0, 0]), int(pts[0, 1])),
        (int(pts[-1, 0]), int(pts[-1, 1])),
    )
    return pts, endpoints


def trim_side_detection_against_upper(
    detection: SidelineDetection,
    upper_detection: SidelineDetection,
    side_anchors: np.ndarray,
    margin_px: float = 3.0,
) -> SidelineDetection:
    if (
        detection.line is None
        or upper_detection.line is None
        or len(detection.candidate_points) < 2
        or len(side_anchors) == 0
    ):
        return detection

    anchor_values = line_signed_values(upper_detection.line, side_anchors)
    nonzero_values = anchor_values[np.abs(anchor_values) > margin_px]
    if len(nonzero_values) == 0:
        return detection

    field_side_sign = 1.0 if float(np.median(nonzero_values)) >= 0 else -1.0
    candidate_values = line_signed_values(upper_detection.line, detection.candidate_points)
    keep = candidate_values * field_side_sign >= -margin_px
    filtered = detection.candidate_points[keep]
    if len(filtered) < 2:
        return detection

    line, inliers = fit_line_from_candidate_points(filtered, side_anchors)
    if line is None or len(inliers) < 2:
        line = robust_fit_line(filtered, trim_px=5.0)
        if line is None:
            return detection
        inliers = filtered[line_distances(line, filtered) <= 3.0]
        if len(inliers) < 2:
            inliers = filtered

    line_points, endpoints = sample_line_points(line, inliers, margin_px=3.0)
    residual = float(np.mean(line_distances(line, inliers))) if len(inliers) else 0.0
    score = float(len(inliers)) / max(residual + 1.0, 1.0)

    return SidelineDetection(
        side=detection.side,
        keypoint_ids=detection.keypoint_ids,
        anchor_points=detection.anchor_points,
        candidate_points=inliers,
        line_points=line_points,
        endpoints=endpoints,
        line=line,
        score=score,
    )


def detect_sidelines(
    keypoints: dict,
    contours: list[np.ndarray],
) -> dict[str, SidelineDetection]:
    contour_points = collect_contour_points(contours)
    detections: dict[str, SidelineDetection] = {}

    for side, kpt_ids in SIDELINE_KEYPOINT_IDS.items():
        anchors = load_sideline_anchor_points(keypoints, kpt_ids)
        expected_line = robust_fit_line(anchors, trim_px=30.0)
        line = None
        candidates = np.empty((0, 2), dtype=np.float64)
        line_points = np.empty((0, 2), dtype=np.int32)
        endpoints = None
        score = 0.0

        if expected_line is not None:
            search_candidates = points_near_line_span(
                contour_points,
                expected_line,
                anchors,
                distance_px=28.0,
                margin_px=100.0,
            )
            line, candidates = fit_line_from_candidate_points(search_candidates, anchors)

            if line is not None and len(candidates) >= 2:
                line_points, endpoints = sample_line_points(line, candidates)
                residual = float(np.mean(line_distances(line, candidates)))
                score = float(len(candidates)) / max(residual + 1.0, 1.0)

        detections[side] = SidelineDetection(
            side=side,
            keypoint_ids=kpt_ids,
            anchor_points=anchors,
            candidate_points=candidates,
            line_points=line_points,
            endpoints=endpoints,
            line=line,
            score=score,
        )

    upper_detection = detections.get("upper")
    if upper_detection is not None and upper_detection.line is not None:
        for side in ("left", "right"):
            detection = detections.get(side)
            if detection is None:
                continue
            detections[side] = trim_side_detection_against_upper(
                detection,
                upper_detection,
                detection.anchor_points,
            )

    return detections


def _detect_line_group(
    side: str,
    contour_points: np.ndarray,
    anchors: np.ndarray,
    kpt_ids: list[int],
    distance_px: float = 28.0,
    margin_px: float = 80.0,
) -> SidelineDetection:
    expected_line = robust_fit_line(anchors, trim_px=20.0)
    line = None
    candidates = np.empty((0, 2), dtype=np.float64)
    line_points = np.empty((0, 2), dtype=np.int32)
    endpoints = None
    score = 0.0

    if expected_line is not None:
        search_candidates = points_near_line_span(
            contour_points,
            expected_line,
            anchors,
            distance_px=distance_px,
            margin_px=margin_px,
        )
        line, candidates = fit_line_from_candidate_points_guided(
            search_candidates,
            anchors,
            expected_line,
        )

        if line is not None and len(candidates) >= 2:
            line_points, endpoints = sample_line_points(line, anchors, margin_px=12.0)
            residual = float(np.mean(line_distances(line, candidates)))
            score = float(len(candidates)) / max(residual + 1.0, 1.0)

    return SidelineDetection(
        side=side,
        keypoint_ids=kpt_ids,
        anchor_points=anchors,
        candidate_points=candidates,
        line_points=line_points,
        endpoints=endpoints,
        line=line,
        score=score,
    )


def detect_paint_boundaries(
    keypoints: dict,
    contours: list[np.ndarray],
) -> dict[str, SidelineDetection]:
    contour_points = collect_contour_points(contours)
    H = estimate_coarse_homography_from_keypoints(keypoints)
    best_detections: dict[str, SidelineDetection] = {}
    best_key = (-1, -1.0)

    for paint_name, keypoint_groups in PAINT_KEYPOINT_GROUPS.items():
        corner_groups = PAINT_CORNER_GROUPS[paint_name]
        detections = {}
        for side, kpt_ids in keypoint_groups.items():
            if H is None:
                anchors = load_sideline_anchor_points(keypoints, kpt_ids)
                distance_px = 24.0
                margin_px = 70.0
            else:
                anchors = project_field_points(H, corner_groups[side])
                distance_px = 34.0
                margin_px = 45.0

            detections[side] = _detect_line_group(
                side,
                contour_points,
                anchors,
                kpt_ids,
                distance_px=distance_px,
                margin_px=margin_px,
            )

        detected_count = sum(1 for detection in detections.values() if detection.line is not None)
        total_score = float(sum(detection.score for detection in detections.values()))
        score_key = (detected_count, total_score)
        if score_key > best_key:
            best_key = score_key
            best_detections = {
                side: SidelineDetection(
                    side=f"{paint_name}_{side}",
                    keypoint_ids=detection.keypoint_ids,
                    anchor_points=detection.anchor_points,
                    candidate_points=detection.candidate_points,
                    line_points=detection.line_points,
                    endpoints=detection.endpoints,
                    line=detection.line,
                    score=detection.score,
                )
                for side, detection in detections.items()
            }

    return best_detections


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


def draw_sidelines_result(
    img_rgb: np.ndarray,
    detections: dict[str, SidelineDetection],
) -> np.ndarray:
    out = img_rgb.copy()
    for side in ("upper", "left", "right"):
        detection = detections.get(side)
        if detection is None:
            continue
        if len(detection.line_points) < 2:
            continue
        color = SIDELINE_COLORS[side]
        cv2.polylines(
            out,
            [detection.line_points.astype(np.int32)],
            isClosed=False,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        label_at = tuple(int(v) for v in detection.line_points[len(detection.line_points) // 2])
        cv2.putText(
            out,
            side,
            (label_at[0] + 6, label_at[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            side,
            (label_at[0] + 6, label_at[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    draw_line_intersection(out, detections, "upper", "left", "court_upleft", (0, 255, 255))
    draw_line_intersection(out, detections, "upper", "right", "court_upright", (0, 255, 255))
    return out


def draw_paint_result(
    img_rgb: np.ndarray,
    detections: dict[str, SidelineDetection],
) -> np.ndarray:
    out = img_rgb.copy()
    for side, detection in detections.items():
        if len(detection.line_points) < 2:
            continue
        label = side.rsplit("_", 1)[-1]
        color = PAINT_COLORS[label]
        cv2.polylines(
            out,
            [detection.line_points.astype(np.int32)],
            isClosed=False,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        label_at = tuple(int(v) for v in detection.line_points[len(detection.line_points) // 2])
        cv2.putText(
            out,
            label,
            (label_at[0] + 6, label_at[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (label_at[0] + 6, label_at[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    draw_paint_corners(out, detections)
    return out


def intersection_from_detections(
    detections: dict[str, SidelineDetection],
    side_a: str,
    side_b: str,
) -> np.ndarray | None:
    det_a = detections.get(side_a)
    det_b = detections.get(side_b)
    if det_a is None or det_b is None or det_a.line is None or det_b.line is None:
        return None

    a1, b1, c1 = det_a.line
    a2, b2, c2 = det_b.line
    denom = a1 * b2 - a2 * b1
    if abs(float(denom)) < 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / denom
    y = (c1 * a2 - c2 * a1) / denom
    return np.array([x, y], dtype=np.float64)


def draw_corner_point(
    img_rgb: np.ndarray,
    point: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y = point
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    if not (0 <= cx < IMG_WIDTH and 0 <= cy < IMG_HEIGHT):
        return
    cv2.circle(img_rgb, (cx, cy), 7, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(img_rgb, (cx, cy), 5, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        img_rgb,
        label,
        (cx + 8, cy - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_rgb,
        label,
        (cx + 8, cy - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_line_intersection(
    img_rgb: np.ndarray,
    detections: dict[str, SidelineDetection],
    side_a: str,
    side_b: str,
    label: str,
    color: tuple[int, int, int],
) -> None:
    point = intersection_from_detections(detections, side_a, side_b)
    if point is not None:
        draw_corner_point(img_rgb, point, label, color)


def draw_paint_corners(
    img_rgb: np.ndarray,
    detections: dict[str, SidelineDetection],
) -> None:
    corner_specs = [
        ("upper", "left", "paint_ul"),
        ("upper", "right", "paint_ur"),
        ("lower", "right", "paint_lr"),
        ("lower", "left", "paint_ll"),
    ]
    for side_a, side_b, label in corner_specs:
        draw_line_intersection(img_rgb, detections, side_a, side_b, label, (255, 128, 0))


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
        "--sideline",
        action="store_true",
        help="绘制上/左/右边线和球场上角点；永远不绘制下边线",
    )
    parser.add_argument(
        "--paint",
        action="store_true",
        help="绘制从 SVM 决策边界候选点识别出的三秒区上下左右边界和角点",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出路径（默认分割标注为 results/<image>_segmented.png，否则为 results/<image>_rgb_svm_boundary.png）",
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

    sideline_detections: dict[str, SidelineDetection] | None = None
    paint_detections: dict[str, SidelineDetection] | None = None
    if args.sideline or args.paint:
        vis = img.copy()
        if args.sideline:
            sideline_detections = detect_sidelines(keypoints, contours)
            vis = draw_sidelines_result(vis, sideline_detections)
        if args.paint:
            paint_detections = detect_paint_boundaries(keypoints, contours)
            vis = draw_paint_result(vis, paint_detections)
    else:
        vis = draw_result(
            img,
            selected,
            selected_second,
            contours,
            show_contours=not args.no_contours,
        )

    if args.output:
        output_path = root / args.output
    elif args.sideline or args.paint:
        output_path = root / "results" / f"{image_path.stem}_segmented.png"
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
        for group_label, detections in (
            ("边线", sideline_detections),
            ("三秒区边界", paint_detections),
        ):
            if detections is None:
                continue
            for side, detection in detections.items():
                if group_label == "边线" and side == "lower":
                    continue
                endpoint_text = "None"
                if detection.endpoints is not None:
                    endpoint_text = f"{detection.endpoints[0]} -> {detection.endpoints[1]}"
                print(
                    f"{side} {group_label}: anchors={len(detection.anchor_points)}, "
                    f"candidates={len(detection.candidate_points)}, "
                    f"line_points={len(detection.line_points)}, "
                    f"score={detection.score:.2f}, endpoints={endpoint_text}"
                )
        print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
