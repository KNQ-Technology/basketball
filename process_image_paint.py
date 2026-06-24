#!/usr/bin/env python3
"""模型关键点 + 三秒区角点加权 + OpenCV 单应矩阵标定。"""
import argparse
import contextlib
import io
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from sklearn.svm import LinearSVC
from skimage import measure

from process_image import (
    IMG_HEIGHT,
    IMG_WIDTH,
    _extract_keypoints,
    _kpts_to_dict,
    infer,
    load_model,
    preprocess,
    validate_keypoints,
)


CM_PER_METER = 100.0
CM_PER_FOOT = 30.48
NBA_FIELD_LENGTH = 28.65 * CM_PER_METER
NBA_FIELD_WIDTH = 15.24 * CM_PER_METER
NBA_LINE_WIDTH = 0.05 * CM_PER_METER
NBA_BASKET_X = 5.25 * CM_PER_FOOT
NBA_FREE_THROW_X = 5.80 * CM_PER_METER
NBA_LANE_WIDTH = 4.90 * CM_PER_METER
NBA_CIRCLE_RADIUS = 1.80 * CM_PER_METER
NBA_THREE_POINT_RADIUS = 7.24 * CM_PER_METER
NBA_THREE_POINT_CORNER_DISTANCE = 6.70 * CM_PER_METER
NBA_THREE_POINT_SIDELINE_MARGIN = NBA_FIELD_WIDTH / 2 - NBA_THREE_POINT_CORNER_DISTANCE
NBA_THREE_LINE_Y = NBA_THREE_POINT_SIDELINE_MARGIN
NBA_THREE_LINE_X = NBA_BASKET_X + np.sqrt(
    NBA_THREE_POINT_RADIUS ** 2 - (NBA_FIELD_WIDTH / 2 - NBA_THREE_LINE_Y) ** 2
)
NBA_THREE_ARC_STOP_ANGLE = np.arcsin(
    (NBA_FIELD_WIDTH / 2 - NBA_THREE_LINE_Y) / NBA_THREE_POINT_RADIUS
)

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
UPPER_IDS = list(range(78, 91))
LEFT_IDS = [13, 26, 39, 52, 65, 78]
RIGHT_IDS = [25, 38, 51, 64, 77, 90]
LINE_KEYPOINT_IDS = {"upper": UPPER_IDS, "left": LEFT_IDS, "right": RIGHT_IDS}
SIDELINE_KEYPOINT_IDS = {
    "upper": list(range(78, 91)),
    "lower": list(range(0, 13)),
    "left": [0, 13, 26, 39, 52, 65, 78],
    "right": [12, 25, 38, 51, 64, 77, 90],
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
PAINT_CORNER_IDS = list(range(91, 99))


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


def _abs(path: str) -> str:
    return os.path.abspath(path)


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_nba_field_points() -> np.ndarray:
    original_width = 1500.0
    original_row_y = np.array([1500.0, 1325.0, 1120.0, 885.0, 620.0, 325.0, 0.0])
    row_y = original_row_y / original_width * NBA_FIELD_WIDTH
    points = []
    for y in row_y:
        for i in range(13):
            points.append([i * NBA_FIELD_LENGTH / 12.0, y, 0.0])
    lane_top_y = NBA_FIELD_WIDTH / 2 - NBA_LANE_WIDTH / 2
    lane_bottom_y = NBA_FIELD_WIDTH / 2 + NBA_LANE_WIDTH / 2
    points.extend([
        [0.0, lane_top_y, 0.0],
        [NBA_FREE_THROW_X, lane_top_y, 0.0],
        [NBA_FREE_THROW_X, lane_bottom_y, 0.0],
        [0.0, lane_bottom_y, 0.0],
        [NBA_FIELD_LENGTH, lane_top_y, 0.0],
        [NBA_FIELD_LENGTH - NBA_FREE_THROW_X, lane_top_y, 0.0],
        [NBA_FIELD_LENGTH - NBA_FREE_THROW_X, lane_bottom_y, 0.0],
        [NBA_FIELD_LENGTH, lane_bottom_y, 0.0],
    ])
    return np.array(points, dtype=np.float64)


def _draw_nba_field_circle(img, H, mul_coef, translate_coef, color, thickness, center, radius, start_angle, stop_angle):
    field_points = [
        center + np.array([radius * np.cos(theta), radius * np.sin(theta)])
        for theta in np.arange(start_angle, stop_angle, 0.2 * np.pi / 180)
    ]
    if not field_points:
        return
    pts = np.array([field_points], dtype=np.float32)
    pts *= mul_coef
    pts += translate_coef
    points_to_draw = cv2.perspectiveTransform(pts, H)[0].astype(int)
    for p0, p1 in zip(points_to_draw[:-1], points_to_draw[1:]):
        cv2.line(img, tuple(p0), tuple(p1), color, thickness)


def _draw_nba_quarter_field(img, H, line_color, curve_color, thickness, mul_coef, translate_coef):
    lane_y = NBA_FIELD_WIDTH / 2 - NBA_LANE_WIDTH / 2
    field_points = np.float32([[
        [0, 0],
        [NBA_FIELD_LENGTH / 2, 0],
        [0, NBA_FIELD_WIDTH / 2],
        [0, lane_y],
        [NBA_FREE_THROW_X, lane_y],
        [NBA_FREE_THROW_X, NBA_FIELD_WIDTH / 2],
        [0, NBA_THREE_LINE_Y],
        [NBA_THREE_LINE_X, NBA_THREE_LINE_Y],
        [NBA_FIELD_LENGTH / 2, NBA_FIELD_WIDTH / 2],
    ]])
    field_points *= mul_coef
    field_points += translate_coef
    points = cv2.perspectiveTransform(field_points, H)[0].astype(int)
    for a, b in ((0, 1), (2, 0), (3, 4), (4, 5), (6, 7), (1, 8)):
        cv2.line(img, tuple(points[a]), tuple(points[b]), line_color, thickness)
    _draw_nba_field_circle(img, H, mul_coef, translate_coef, curve_color, thickness, np.array([NBA_FREE_THROW_X, NBA_FIELD_WIDTH / 2]), NBA_CIRCLE_RADIUS, 0, np.pi / 2)
    _draw_nba_field_circle(img, H, mul_coef, translate_coef, curve_color, thickness, np.array([NBA_BASKET_X, NBA_FIELD_WIDTH / 2]), NBA_THREE_POINT_RADIUS, 0, NBA_THREE_ARC_STOP_ANGLE)
    _draw_nba_field_circle(img, H, mul_coef, translate_coef, curve_color, thickness, np.array([NBA_FIELD_LENGTH / 2, NBA_FIELD_WIDTH / 2]), NBA_CIRCLE_RADIUS, np.pi / 2, np.pi)


def draw_nba_field(img, H, line_color, curve_color, thickness) -> None:
    mul_coefs = [
        np.array([1, 1], dtype=np.float32),
        np.array([1, -1], dtype=np.float32),
        np.array([-1, -1], dtype=np.float32),
        np.array([-1, 1], dtype=np.float32),
    ]
    translate_coefs = [
        np.array([0, 0], dtype=np.float32),
        np.array([0, NBA_FIELD_WIDTH], dtype=np.float32),
        np.array([NBA_FIELD_LENGTH, NBA_FIELD_WIDTH], dtype=np.float32),
        np.array([NBA_FIELD_LENGTH, 0], dtype=np.float32),
    ]
    for mul_coef, translate_coef in zip(mul_coefs, translate_coefs):
        _draw_nba_quarter_field(img, H, line_color, curve_color, thickness, mul_coef, translate_coef)


def _keypoints_for_heuristic(kpts: dict[int, tuple[float, float, float]]) -> dict:
    return {str(idx): {"img_x": float(x), "img_y": float(y), "conf": float(conf)} for idx, (x, y, conf) in kpts.items()}


def _keypoints_to_json(kpts: dict[int, tuple[float, float, float]]) -> dict:
    return {str(idx): {"img_x": round(x, 2), "img_y": round(y, 2), "conf": round(conf, 6)} for idx, (x, y, conf) in sorted(kpts.items())}


def _draw_adjusted_keypoints(image_bgr, adjusted_kpts, threshold, print_coords) -> None:
    field_points = get_nba_field_points()[:, :2]
    row_colors = [
        cv2.cvtColor(np.uint8([[[int(h * 180 / 7), 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
        for h in range(7)
    ]
    if print_coords:
        print(f"\n{'idx':>4}  {'img_x':>7} {'img_y':>7}  {'field_x':>10} {'field_y':>10}  {'conf':>8}")
        print("-" * 64)
    for idx, (x, y, conf) in sorted(adjusted_kpts.items()):
        if conf < threshold:
            continue
        cx, cy = int(round(x)), int(round(y))
        if not (0 <= cx < IMG_WIDTH and 0 <= cy < IMG_HEIGHT):
            continue
        color = row_colors[idx // 13] if idx < 91 else (0, 165, 255)
        cv2.circle(image_bgr, (cx, cy), 6, (0, 0, 0), -1)
        cv2.circle(image_bgr, (cx, cy), 5, color, -1)
        cv2.putText(image_bgr, str(idx), (cx + 7, cy - 3), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image_bgr, str(idx), (cx + 7, cy - 3), cv2.FONT_HERSHEY_PLAIN, 0.85, color, 1, cv2.LINE_AA)
        if print_coords:
            fx, fy = field_points[idx]
            print(f"{idx:>4}  {x:>7.2f} {y:>7.2f}  {fx:>10.1f} {fy:>10.1f}  {conf:>8.4f}")


def load_keypoint_samples(keypoints: dict, img: np.ndarray, kpt_ids: list[int]) -> list[KeypointSample]:
    samples = []
    for kpt_id in kpt_ids:
        kpt = keypoints.get(str(kpt_id))
        if not kpt:
            continue
        x, y = float(kpt["img_x"]), float(kpt["img_y"])
        px = max(0, min(IMG_WIDTH - 1, int(round(x))))
        py = max(0, min(IMG_HEIGHT - 1, int(round(y))))
        samples.append(KeypointSample(kpt_id, x, y, img[py, px].astype(np.float64)))
    return samples


def select_keypoint_by_rgb(samples: list[KeypointSample]) -> KeypointSample:
    remaining = list(samples)
    while len(remaining) > 1:
        center = np.mean([s.rgb for s in remaining], axis=0)
        distances = [float(np.linalg.norm(s.rgb - center)) for s in remaining]
        remaining.pop(int(np.argmax(distances)))
    return remaining[0]


def _square_value(img, cx, cy, side, ref_rgb, threshold) -> np.ndarray | None:
    half = side / 2.0
    x0 = max(0, min(IMG_WIDTH, int(round(cx - half))))
    y0 = max(0, min(IMG_HEIGHT, int(round(cy - half))))
    x1 = max(0, min(IMG_WIDTH, x0 + side))
    y1 = max(0, min(IMG_HEIGHT, y0 + side))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    mask = np.linalg.norm(patch - ref_rgb, axis=1) > threshold
    if not np.any(mask):
        return None
    return np.mean(patch[mask], axis=0)


def load_value_samples(keypoints, img, kpt_ids, side, ref_rgb, threshold) -> list[KeypointSample]:
    samples = []
    for kpt_id in kpt_ids:
        kpt = keypoints.get(str(kpt_id))
        if not kpt:
            continue
        x, y = float(kpt["img_x"]), float(kpt["img_y"])
        value = _square_value(img, x, y, side, ref_rgb, threshold)
        if value is not None:
            samples.append(KeypointSample(kpt_id, x, y, value))
    return samples


def pseudo_label_pixels(pixels: np.ndarray, class0_rgb: np.ndarray, class1_rgb: np.ndarray) -> np.ndarray:
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


def _collect_contour_points(contours: list[np.ndarray]) -> np.ndarray:
    return np.vstack(contours).astype(np.float64) if contours else np.empty((0, 2), dtype=np.float64)


def load_sideline_anchor_points(keypoints: dict, kpt_ids: list[int]) -> np.ndarray:
    points = []
    for kpt_id in kpt_ids:
        kpt = keypoints.get(str(kpt_id))
        if kpt:
            points.append([float(kpt["img_x"]), float(kpt["img_y"])])
    return np.array(points, dtype=np.float64)


def fit_line_pca(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 2:
        return None
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    normal /= norm
    return np.array([normal[0], normal[1], -float(np.dot(normal, center))], dtype=np.float64)


def line_distances(line: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.abs(points @ line[:2] + line[2])


def line_direction(line: np.ndarray) -> np.ndarray:
    direction = np.array([line[1], -line[0]], dtype=np.float64)
    norm = np.linalg.norm(direction)
    return direction / norm if norm else np.array([1.0, 0.0], dtype=np.float64)


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
        if next_line is not None:
            line = next_line
    return line


def points_near_line_span(points, line, span_points, distance_px=12.0, margin_px=80.0):
    if len(points) == 0 or len(span_points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    direction = line_direction(line)
    span_t = span_points @ direction
    point_t = points @ direction
    distances = line_distances(line, points)
    return points[(distances <= distance_px) & (point_t >= span_t.min() - margin_px) & (point_t <= span_t.max() + margin_px)]


def _line_from_hough(rho: float, theta: float) -> np.ndarray:
    normal = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    return np.array([normal[0], normal[1], -rho], dtype=np.float64)


def _candidate_lines(points: np.ndarray, max_lines: int = 240) -> list[np.ndarray]:
    lines = []
    if len(points) < 2:
        return lines
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    xy = np.rint(points).astype(np.int32)
    xy[:, 0] = np.clip(xy[:, 0], 0, IMG_WIDTH - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, IMG_HEIGHT - 1)
    mask[xy[:, 1], xy[:, 0]] = 255
    raw_lines = cv2.HoughLines(mask, 1, np.pi / 720.0, max(10, min(80, int(np.count_nonzero(mask)) // 12)))
    if raw_lines is not None:
        for raw_line in raw_lines[:max_lines]:
            rho, theta = (float(v) for v in raw_line[0])
            lines.append(_line_from_hough(rho, theta))
    rng = np.random.default_rng(0)
    for _ in range(300):
        i, j = rng.choice(len(points), size=2, replace=False)
        if np.linalg.norm(points[i] - points[j]) >= 40.0:
            line = fit_line_pca(np.array([points[i], points[j]], dtype=np.float64))
            if line is not None:
                lines.append(line)
    return lines


def _line_score(line: np.ndarray, points: np.ndarray, inlier_px: float):
    distances = line_distances(line, points)
    mask = distances <= inlier_px
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0, 0.0, float("inf"), mask
    inliers = points[mask]
    span = float(np.ptp(inliers @ line_direction(line))) if count >= 2 else 0.0
    residual = float(np.mean(distances[mask]))
    return count, span, residual, mask


def fit_line_from_candidate_points(points: np.ndarray, anchors: np.ndarray, inlier_px: float = 3.0):
    _ = anchors
    if len(points) < 2:
        return None, np.empty((0, 2), dtype=np.float64)
    best_line = None
    best_mask = None
    best_key = (-1, -1.0, -float("inf"))
    for line in _candidate_lines(points):
        count, span, residual, mask = _line_score(line, points, inlier_px)
        key = (count, span, -residual)
        if count >= 2 and key > best_key:
            best_key, best_line, best_mask = key, line, mask
    if best_line is None or best_mask is None:
        fallback = robust_fit_line(points, trim_px=8.0)
        if fallback is None:
            return None, np.empty((0, 2), dtype=np.float64)
        return fallback, points[line_distances(fallback, points) <= inlier_px]
    inliers = points[best_mask]
    refined = robust_fit_line(inliers, trim_px=max(inlier_px, 3.0))
    if refined is not None:
        count, span, residual, mask = _line_score(refined, points, inlier_px)
        if (count, span, -residual) >= best_key:
            return refined, points[mask]
    return best_line, inliers


def sample_line_points(line: np.ndarray, span_points: np.ndarray, margin_px: float = 20.0):
    if len(span_points) < 2:
        return np.empty((0, 2), dtype=np.int32), None
    normal = line[:2]
    direction = line_direction(line)
    base = -line[2] * normal
    span_t = span_points @ direction
    lo, hi = float(span_t.min()) - margin_px, float(span_t.max()) + margin_px
    pts = base[None, :] + np.linspace(lo, hi, max(2, int(round(hi - lo)) + 1))[:, None] * direction[None, :]
    pts = np.rint(pts).astype(np.int32)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < IMG_WIDTH) & (pts[:, 1] >= 0) & (pts[:, 1] < IMG_HEIGHT)]
    if len(pts) < 2:
        return pts, None
    _, unique_idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(unique_idx)]
    if len(pts) < 2:
        return pts, None
    return pts, ((int(pts[0, 0]), int(pts[0, 1])), (int(pts[-1, 0]), int(pts[-1, 1])))


def detect_sidelines(keypoints: dict, contours: list[np.ndarray]) -> dict[str, SidelineDetection]:
    contour_points = _collect_contour_points(contours)
    detections = {}
    for side, kpt_ids in SIDELINE_KEYPOINT_IDS.items():
        anchors = load_sideline_anchor_points(keypoints, kpt_ids)
        expected = robust_fit_line(anchors, trim_px=30.0)
        line = None
        candidates = np.empty((0, 2), dtype=np.float64)
        line_points = np.empty((0, 2), dtype=np.int32)
        endpoints = None
        score = 0.0
        if expected is not None:
            search = points_near_line_span(contour_points, expected, anchors, distance_px=28.0, margin_px=100.0)
            line, candidates = fit_line_from_candidate_points(search, anchors)
            if line is not None and len(candidates) >= 2:
                line_points, endpoints = sample_line_points(line, candidates)
                residual = float(np.mean(line_distances(line, candidates)))
                score = float(len(candidates)) / max(residual + 1.0, 1.0)
        detections[side] = SidelineDetection(side, kpt_ids, anchors, candidates, line_points, endpoints, line, score)
    return detections


def project_field_points(H: np.ndarray, point_ids: list[int]) -> np.ndarray:
    field_points = get_nba_field_points()[point_ids, :2].astype(np.float32)
    return cv2.perspectiveTransform(field_points.reshape(1, -1, 2), H)[0].astype(np.float64)


def estimate_coarse_homography_from_keypoints(keypoints: dict) -> np.ndarray | None:
    field_points = get_nba_field_points()[:91, :2]
    src = []
    dst = []
    for idx in range(91):
        kpt = keypoints.get(str(idx))
        if kpt:
            src.append(field_points[idx])
            dst.append([float(kpt["img_x"]), float(kpt["img_y"])])
    if len(src) < 4:
        return None
    H, mask = cv2.findHomography(np.array(src, dtype=np.float64), np.array(dst, dtype=np.float64), cv2.RANSAC, 10.0)
    if H is None or mask is None or int(np.count_nonzero(mask)) < 4:
        return None
    return H.astype(np.float64)


def detect_paint_boundaries(keypoints: dict, contours: list[np.ndarray]) -> dict[str, SidelineDetection]:
    contour_points = _collect_contour_points(contours)
    H = estimate_coarse_homography_from_keypoints(keypoints)
    best_detections = {}
    best_key = (-1, -1.0)
    for paint_name, keypoint_groups in PAINT_KEYPOINT_GROUPS.items():
        detections = {}
        corner_groups = PAINT_CORNER_GROUPS[paint_name]
        for side, kpt_ids in keypoint_groups.items():
            if H is None:
                anchors = load_sideline_anchor_points(keypoints, kpt_ids)
                distance_px, margin_px = 24.0, 70.0
            else:
                anchors = project_field_points(H, corner_groups[side])
                distance_px, margin_px = 34.0, 45.0
            expected = robust_fit_line(anchors, trim_px=20.0)
            line = None
            candidates = np.empty((0, 2), dtype=np.float64)
            line_points = np.empty((0, 2), dtype=np.int32)
            endpoints = None
            score = 0.0
            if expected is not None:
                search = points_near_line_span(contour_points, expected, anchors, distance_px=distance_px, margin_px=margin_px)
                line, candidates = fit_line_from_candidate_points(search, anchors)
                if line is not None and len(candidates) >= 2:
                    line_points, endpoints = sample_line_points(line, anchors, margin_px=12.0)
                    residual = float(np.mean(line_distances(line, candidates)))
                    score = float(len(candidates)) / max(residual + 1.0, 1.0)
            detections[side] = SidelineDetection(f"{paint_name}_{side}", kpt_ids, anchors, candidates, line_points, endpoints, line, score)
        key = (sum(1 for d in detections.values() if d.line is not None), float(sum(d.score for d in detections.values())))
        if key > best_key:
            best_key, best_detections = key, detections
    return best_detections


def _line_intersection(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    denom = a1 * b2 - a2 * b1
    if abs(float(denom)) < 1e-9:
        return None
    return np.array([(b1 * c2 - b2 * c1) / denom, (c1 * a2 - c2 * a1) / denom], dtype=np.float64)


def _project_point_to_line(point: np.ndarray, line: np.ndarray) -> np.ndarray:
    return point - float(np.dot(line[:2], point) + line[2]) * line[:2]


def _adjust_keypoints(kpts: dict[int, tuple[float, float, float]], sideline_detections: dict):
    adjusted = dict(kpts)
    details = {}
    for side, ids in LINE_KEYPOINT_IDS.items():
        det = sideline_detections.get(side)
        if det is None or det.line is None:
            details[side] = {"detected": False, "adjusted_ids": []}
            continue
        adjusted_ids = []
        for idx in ids:
            if idx in adjusted:
                x, y, conf = adjusted[idx]
                p = _project_point_to_line(np.array([x, y], dtype=np.float64), det.line)
                adjusted[idx] = (float(p[0]), float(p[1]), conf)
                adjusted_ids.append(idx)
        details[side] = {"detected": True, "adjusted_ids": adjusted_ids, "line": [float(v) for v in det.line], "candidate_points": int(len(det.candidate_points)), "line_points": int(len(det.line_points)), "score": float(det.score), "endpoints": det.endpoints}
    intersections = {}
    for idx, a, b in ((78, "upper", "left"), (90, "upper", "right")):
        da, db = sideline_detections.get(a), sideline_detections.get(b)
        if da is not None and db is not None and da.line is not None and db.line is not None and idx in adjusted:
            p = _line_intersection(da.line, db.line)
            if p is not None:
                _, _, conf = adjusted[idx]
                adjusted[idx] = (float(p[0]), float(p[1]), conf)
                intersections[str(idx)] = [float(p[0]), float(p[1])]
    details["intersections"] = intersections
    return adjusted, details


def _paint_prefix(paint_detections: dict) -> str | None:
    for detection in paint_detections.values():
        if detection.side.startswith("left_paint_"):
            return "left_paint"
        if detection.side.startswith("right_paint_"):
            return "right_paint"
    return None


def _add_paint_corners(adjusted_kpts, paint_detections):
    adjusted = dict(adjusted_kpts)
    prefix = _paint_prefix(paint_detections)
    details = {"detected": False, "paint": prefix, "corner_ids": [], "corners": {}}
    if prefix is None or any(side not in paint_detections or paint_detections[side].line is None for side in ("upper", "lower", "left", "right")):
        return adjusted, details
    upper, lower = paint_detections["upper"].line, paint_detections["lower"].line
    left, right = paint_detections["left"].line, paint_detections["right"].line
    specs = [(91, left, upper), (92, right, upper), (93, right, lower), (94, left, lower)] if prefix == "left_paint" else [(95, right, upper), (96, left, upper), (97, left, lower), (98, right, lower)]
    for idx, line_a, line_b in specs:
        p = _line_intersection(line_a, line_b)
        if p is None:
            continue
        adjusted[idx] = (float(p[0]), float(p[1]), 1.0)
        details["corner_ids"].append(idx)
        details["corners"][str(idx)] = [float(p[0]), float(p[1])]
    details["detected"] = len(details["corner_ids"]) == 4
    return adjusted, details


def _run_heuristic_sidelines(image_rgb, keypoints, pixel, rgb_threshold, svm_c):
    samples = load_keypoint_samples(keypoints, image_rgb, KEYPOINT_IDS)
    if not samples:
        raise RuntimeError("第一组关键点在模型输出中均未识别到")
    selected = select_keypoint_by_rgb(samples)
    class0_rgb = selected.rgb
    value_samples = load_value_samples(keypoints, image_rgb, SECOND_KEYPOINT_IDS, pixel, class0_rgb, rgb_threshold)
    if not value_samples:
        raise RuntimeError("第二组关键点均无有效 RGB value")
    selected_second = select_keypoint_by_rgb(value_samples)
    class1_rgb = selected_second.rgb
    pixels = image_rgb.reshape(-1, 3).astype(np.float64)
    labels = pseudo_label_pixels(pixels, class0_rgb, class1_rgb)
    clf = fit_rgb_linear_svm(pixels, labels, C=svm_c)
    decision = decision_map_from_clf(image_rgb, clf)
    contours = extract_decision_contours(decision)
    sideline_detections = detect_sidelines(keypoints, contours)
    paint_detections = detect_paint_boundaries(keypoints, contours)
    details = {
        "class0": {"keypoint_id": selected.kpt_id, "x": round(selected.x, 2), "y": round(selected.y, 2), "rgb": [round(float(v), 2) for v in class0_rgb]},
        "class1": {"keypoint_id": selected_second.kpt_id, "x": round(selected_second.x, 2), "y": round(selected_second.y, 2), "rgb": [round(float(v), 2) for v in class1_rgb]},
        "decision_contours": len(contours),
        "svm": {"coef": [float(v) for v in clf.coef_[0]], "intercept": float(clf.intercept_[0])},
    }
    return sideline_detections, paint_detections, details


def _draw_detection_line(img_rgb: np.ndarray, detection: SidelineDetection, label: str, color: tuple[int, int, int]) -> None:
    if len(detection.line_points) < 2:
        return
    cv2.polylines(img_rgb, [detection.line_points.astype(np.int32)], False, color, 2, cv2.LINE_AA)
    p = tuple(int(v) for v in detection.line_points[len(detection.line_points) // 2])
    cv2.putText(img_rgb, label, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_rgb, label, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_corner(img_rgb: np.ndarray, point: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    if not (0 <= x < IMG_WIDTH and 0 <= y < IMG_HEIGHT):
        return
    cv2.circle(img_rgb, (x, y), 7, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img_rgb, (x, y), 5, color, -1, cv2.LINE_AA)
    cv2.putText(img_rgb, label, (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_rgb, label, (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_intersection(img_rgb: np.ndarray, detections: dict, a: str, b: str, label: str, color: tuple[int, int, int]) -> None:
    da, db = detections.get(a), detections.get(b)
    if da is None or db is None or da.line is None or db.line is None:
        return
    point = _line_intersection(da.line, db.line)
    if point is not None:
        _draw_corner(img_rgb, point, label, color)


def save_segmented_image(
    image_rgb: np.ndarray,
    output_path: str,
    sideline_detections: dict,
    paint_detections: dict,
) -> None:
    vis = image_rgb.copy()
    sideline_colors = {"upper": (255, 255, 0), "left": (0, 255, 0), "right": (255, 80, 80)}
    for side in ("upper", "left", "right"):
        det = sideline_detections.get(side)
        if det is not None:
            _draw_detection_line(vis, det, side, sideline_colors[side])
    _draw_intersection(vis, sideline_detections, "upper", "left", "court_upleft", (0, 255, 255))
    _draw_intersection(vis, sideline_detections, "upper", "right", "court_upright", (0, 255, 255))

    paint_colors = {"upper": (255, 160, 0), "lower": (255, 0, 160), "left": (80, 255, 80), "right": (80, 160, 255)}
    for side in ("upper", "lower", "left", "right"):
        det = paint_detections.get(side)
        if det is not None:
            _draw_detection_line(vis, det, side, paint_colors[side])
    for a, b, label in (("upper", "left", "paint_ul"), ("upper", "right", "paint_ur"), ("lower", "right", "paint_lr"), ("lower", "left", "paint_ll")):
        _draw_intersection(vis, paint_detections, a, b, label, (255, 128, 0))

    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def _corner_correspondences(sideline_detections: dict) -> list[tuple[np.ndarray, np.ndarray, str]]:
    specs = [
        ("upper", "left", np.array([0.0, 0.0], dtype=np.float64), "court_upper_left"),
        (
            "upper",
            "right",
            np.array([NBA_FIELD_LENGTH, 0.0], dtype=np.float64),
            "court_upper_right",
        ),
        (
            "lower",
            "left",
            np.array([0.0, NBA_FIELD_WIDTH], dtype=np.float64),
            "court_lower_left",
        ),
        (
            "lower",
            "right",
            np.array([NBA_FIELD_LENGTH, NBA_FIELD_WIDTH], dtype=np.float64),
            "court_lower_right",
        ),
    ]

    out = []
    for side_a, side_b, court_point, name in specs:
        det_a = sideline_detections.get(side_a)
        det_b = sideline_detections.get(side_b)
        if det_a is None or det_b is None or det_a.line is None or det_b.line is None:
            continue
        image_point = _line_intersection(det_a.line, det_b.line)
        if image_point is None:
            continue
        out.append((court_point, image_point, name))
    return out


def _repeat_correspondence(
    src: list[np.ndarray],
    dst: list[np.ndarray],
    src_point: np.ndarray,
    dst_point: np.ndarray,
    weight: int,
) -> None:
    for _ in range(max(1, int(weight))):
        src.append(src_point.astype(np.float64))
        dst.append(dst_point.astype(np.float64))


def _estimate_homography_opencv(
    adjusted_kpts: dict[int, tuple[float, float, float]],
    sideline_detections: dict,
    threshold: float,
    top_k: int | None,
    keypoint_weight: int,
    court_corner_weight: int,
    paint_corner_weight: int,
) -> tuple[np.ndarray | None, list[int], int, str, dict]:
    field_points = get_nba_field_points()[:, :2].astype(np.float64)
    use_ids = [
        idx
        for idx, (_, _, conf) in adjusted_kpts.items()
        if idx < len(field_points) and conf >= threshold
    ]

    if top_k is not None:
        use_ids.sort(key=lambda idx: adjusted_kpts[idx][2], reverse=True)
        use_ids = use_ids[:top_k]
    else:
        use_ids.sort()

    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    weighted_counts = {
        "model_keypoints": 0,
        "paint_corners": 0,
        "court_corners": 0,
    }

    for idx in use_ids:
        x, y, _ = adjusted_kpts[idx]
        weight = paint_corner_weight if idx in PAINT_CORNER_IDS else keypoint_weight
        _repeat_correspondence(src, dst, field_points[idx], np.array([x, y]), weight)
        if idx in PAINT_CORNER_IDS:
            weighted_counts["paint_corners"] += weight
        else:
            weighted_counts["model_keypoints"] += weight

    court_corners = []
    for court_point, image_point, name in _corner_correspondences(sideline_detections):
        _repeat_correspondence(src, dst, court_point, image_point, court_corner_weight)
        weighted_counts["court_corners"] += court_corner_weight
        court_corners.append(
            {
                "name": name,
                "field": [float(v) for v in court_point],
                "image": [float(v) for v in image_point],
                "weight": court_corner_weight,
            }
        )

    details = {
        "method": "cv2.findHomography",
        "weighting": "integer correspondence repetition",
        "keypoint_weight": keypoint_weight,
        "court_corner_weight": court_corner_weight,
        "paint_corner_weight": paint_corner_weight,
        "weighted_counts": weighted_counts,
        "weighted_correspondences": len(src),
        "court_corners": court_corners,
    }

    if len(src) < 4:
        return None, use_ids, 0, "opencv_weighted_findHomography", details

    H, _ = cv2.findHomography(
        np.array(src, dtype=np.float64),
        np.array(dst, dtype=np.float64),
        0,
    )
    if H is None:
        return None, use_ids, 0, "opencv_weighted_findHomography", details

    inliers = _count_forward_inliers(H.astype(np.float64), adjusted_kpts, use_ids, field_points)
    return H.astype(np.float64), use_ids, inliers, "opencv_weighted_findHomography", details


def _count_forward_inliers(
    H: np.ndarray,
    adjusted_kpts: dict[int, tuple[float, float, float]],
    use_ids: list[int],
    field_points: np.ndarray,
    threshold_px: float = 10.0,
) -> int:
    if not use_ids:
        return 0
    src = np.array([field_points[idx] for idx in use_ids], dtype=np.float64)
    dst = np.array([[adjusted_kpts[idx][0], adjusted_kpts[idx][1]] for idx in use_ids], dtype=np.float64)
    projected = cv2.perspectiveTransform(src.reshape(1, -1, 2).astype(np.float64), H)[0]
    errors = np.linalg.norm(projected - dst, axis=1)
    return int(np.count_nonzero(errors <= threshold_px))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="模型关键点 + 三秒区角点高权重 + OpenCV 单应矩阵篮球场标定"
    )
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("--output-dir", required=True, metavar="DIR", help="输出目录（必填）")
    parser.add_argument("--model", required=True, metavar="MODEL", help="模型权重路径（必填）")
    parser.add_argument("--clahe", action="store_true", help="对 L 通道做自适应直方图均衡化")
    parser.add_argument("--draw-keypoints", action="store_true", help="在输出图片上叠加调整后的关键点")
    parser.add_argument("--print-coords", action="store_true", help="配合 --draw-keypoints 和 --verbose 打印坐标")
    parser.add_argument("--top-k", type=int, default=None, metavar="K", help="取置信度最高的 K 个模型关键点")
    parser.add_argument("--threshold", type=float, default=0.0, metavar="T", help="模型关键点置信度阈值")
    parser.add_argument("--check", type=int, default=None, metavar="N", help="最少有效模型关键点数量")
    parser.add_argument("--valid-model", default=None, metavar="MODEL", help="验证模型路径")
    parser.add_argument("--valid-diff-threshold", type=float, default=20.0, metavar="PX", help="两模型关键点平均像素距离上限")
    parser.add_argument("--pixel", type=int, default=40, help="启发式第二阶段正方形边长")
    parser.add_argument("--rgb-threshold", type=float, default=40.0, help="RGB 原型距离阈值")
    parser.add_argument("--C", type=float, default=1.0, help="RGB 线性 SVM 惩罚系数")
    parser.add_argument("--keypoint-weight", type=int, default=1, help="普通模型关键点权重")
    parser.add_argument("--court-corner-weight", type=int, default=40, help="球场角点权重")
    parser.add_argument("--paint-corner-weight", type=int, default=40, help="三秒区角点权重")
    parser.add_argument("--save_segment", action="store_true", help="保存 <image>_segmented.png 分割标注图")
    parser.add_argument("--verbose", action="store_true", help="打印调试信息")
    args = parser.parse_args()

    args.input = _abs(args.input)
    args.model = _abs(args.model)
    args.output_dir = _abs(args.output_dir)
    if args.valid_model:
        args.valid_model = _abs(args.valid_model)
    os.makedirs(args.output_dir, exist_ok=True)

    stem = Path(args.input).stem
    calibrated_path = os.path.join(args.output_dir, f"{stem}_calibrated_paint.png")
    json_path = os.path.join(args.output_dir, f"{stem}_result_paint.json")
    vprint = print if args.verbose else (lambda *a, **k: None)

    result_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": args.input,
        "output_dir": args.output_dir,
        "settings": {
            "model": args.model,
            "valid_model": args.valid_model,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "check": args.check,
            "pixel": args.pixel,
            "rgb_threshold": args.rgb_threshold,
            "C": args.C,
            "clahe": args.clahe,
            "valid_diff_threshold": args.valid_diff_threshold,
            "keypoint_weight": args.keypoint_weight,
            "court_corner_weight": args.court_corner_weight,
            "paint_corner_weight": args.paint_corner_weight,
            "court_template": {
                "type": "NBA",
                "unit": "cm",
                "length": NBA_FIELD_LENGTH,
                "width": NBA_FIELD_WIDTH,
                "line_width": NBA_LINE_WIDTH,
                "basket_x": NBA_BASKET_X,
                "free_throw_x": NBA_FREE_THROW_X,
                "lane_width": NBA_LANE_WIDTH,
                "circle_radius": NBA_CIRCLE_RADIUS,
                "three_point_radius": NBA_THREE_POINT_RADIUS,
                "three_point_corner_distance": NBA_THREE_POINT_CORNER_DISTANCE,
                "three_point_sideline_margin": NBA_THREE_POINT_SIDELINE_MARGIN,
            },
        },
        "status": "success",
        "model_arch": None,
        "valid_model_arch": None,
        "keypoints": {},
        "valid_keypoints": None,
        "check_result": None,
        "validation_result": None,
        "adjusted_keypoints": {},
        "heuristic": None,
        "sidelines": None,
        "paint_boundaries": None,
        "paint_corners": None,
        "adjustments": None,
        "homography": None,
        "homography_details": None,
        "homography_keypoint_ids": [],
        "homography_inliers": 0,
        "homography_mode": None,
    }

    if not os.path.exists(args.input):
        raise SystemExit(f"Error: 图像不存在: {args.input}")
    if not os.path.exists(args.model):
        raise SystemExit(f"Error: 模型不存在: {args.model}")
    if args.valid_model and not os.path.exists(args.valid_model):
        raise SystemExit(f"Error: 验证模型不存在: {args.valid_model}")
    if args.pixel <= 0:
        raise SystemExit("Error: --pixel 必须为正整数")
    if args.top_k is not None and args.top_k <= 0:
        raise SystemExit("Error: --top-k 必须为正整数")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vprint(f"使用设备: {device}")
    model, arch_info = load_model(args.model, device, verbose=args.verbose)
    result_data["model_arch"] = arch_info

    image_rgb = preprocess(args.input, args.clahe)
    heuristic_rgb = preprocess(args.input, clahe=False)
    heatmaps = infer(image_rgb, model, device)
    kpts = _extract_keypoints(heatmaps)
    keypoints = _keypoints_for_heuristic(kpts)
    result_data["keypoints"] = _kpts_to_dict(kpts)

    def _bail(status: str) -> None:
        result_data["status"] = status
        cv2.imwrite(calibrated_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        _save_json(json_path, result_data)
        vprint(f"已保存: {calibrated_path}")
        vprint(f"结果 JSON: {json_path}")

    if args.check is not None:
        valid_count = sum(1 for (_, _, conf) in kpts.values() if conf >= args.threshold)
        passed = valid_count >= args.check
        result_data["check_result"] = {
            "valid_count": valid_count,
            "required": args.check,
            "passed": passed,
        }
        if not passed:
            _bail("check_failed")
            return

    if args.valid_model:
        valid_model, valid_arch = load_model(args.valid_model, device, verbose=args.verbose)
        result_data["valid_model_arch"] = valid_arch
        heatmaps_valid = infer(image_rgb, valid_model, device)
        kpts_valid = _extract_keypoints(heatmaps_valid)
        result_data["valid_keypoints"] = _kpts_to_dict(kpts_valid)
        ok, msg, val_details = validate_keypoints(
            heatmaps,
            heatmaps_valid,
            diff_threshold=args.valid_diff_threshold,
        )
        result_data["validation_result"] = {"ok": ok, "message": msg, **val_details}
        vprint(f"[验证] {msg}")
        if not ok:
            _bail("validation_failed")
            return

    try:
        sideline_detections, paint_detections, heuristic_details = _run_heuristic_sidelines(
            heuristic_rgb,
            keypoints,
            pixel=args.pixel,
            rgb_threshold=args.rgb_threshold,
            svm_c=args.C,
        )
    except RuntimeError as exc:
        result_data["status"] = "heuristic_failed"
        result_data["heuristic"] = {"error": str(exc)}
        cv2.imwrite(calibrated_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        _save_json(json_path, result_data)
        vprint(f"启发式失败: {exc}")
        return

    if args.save_segment:
        segmented_path = os.path.join(args.output_dir, f"{stem}_segmented.png")
        save_segmented_image(
            image_rgb,
            segmented_path,
            sideline_detections,
            paint_detections,
        )
        vprint(f"分割标注图: {segmented_path}")

    adjusted_kpts, adjustment_details = _adjust_keypoints(kpts, sideline_detections)
    adjusted_kpts, paint_corner_details = _add_paint_corners(adjusted_kpts, paint_detections)
    H, used_ids, inliers, homography_mode, homography_details = _estimate_homography_opencv(
        adjusted_kpts,
        sideline_detections,
        args.threshold,
        args.top_k,
        keypoint_weight=args.keypoint_weight,
        court_corner_weight=args.court_corner_weight,
        paint_corner_weight=args.paint_corner_weight,
    )

    result_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if H is not None:
        ctx = contextlib.redirect_stdout(io.StringIO()) if not args.verbose else contextlib.nullcontext()
        with ctx:
            draw_nba_field(result_bgr, H, (0, 0, 255), (0, 0, 255), 2)
    else:
        result_data["status"] = "insufficient_points"
        vprint("警告：有效对应点少于 4，无法估计单应矩阵。")

    if args.draw_keypoints:
        _draw_adjusted_keypoints(
            result_bgr,
            adjusted_kpts,
            args.threshold,
            print_coords=(args.print_coords and args.verbose),
        )

    result_data["adjusted_keypoints"] = _keypoints_to_json(adjusted_kpts)
    result_data["heuristic"] = heuristic_details
    result_data["sidelines"] = {
        side: {
            "detected": detection.line is not None,
            "line": [float(v) for v in detection.line] if detection.line is not None else None,
            "candidate_points": int(len(detection.candidate_points)),
            "line_points": int(len(detection.line_points)),
            "score": float(detection.score),
            "endpoints": detection.endpoints,
        }
        for side, detection in sideline_detections.items()
    }
    result_data["paint_boundaries"] = {
        side: {
            "detected": detection.line is not None,
            "name": detection.side,
            "line": [float(v) for v in detection.line] if detection.line is not None else None,
            "candidate_points": int(len(detection.candidate_points)),
            "line_points": int(len(detection.line_points)),
            "score": float(detection.score),
            "endpoints": detection.endpoints,
        }
        for side, detection in paint_detections.items()
    }
    result_data["paint_corners"] = paint_corner_details
    result_data["adjustments"] = adjustment_details
    result_data["homography"] = H.tolist() if H is not None else None
    result_data["homography_details"] = homography_details
    result_data["homography_keypoint_ids"] = used_ids
    result_data["homography_inliers"] = inliers
    result_data["homography_mode"] = homography_mode

    cv2.imwrite(calibrated_path, result_bgr)
    _save_json(json_path, result_data)
    vprint(f"已保存: {calibrated_path}")
    vprint(f"结果 JSON: {json_path}")


if __name__ == "__main__":
    main()
