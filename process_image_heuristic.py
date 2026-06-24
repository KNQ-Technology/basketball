#!/usr/bin/env python3
"""模型关键点 + RGB/SVM 启发式边线修正 + OpenCV 单应矩阵标定。"""
import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares

from process_image_paint import (
    KEYPOINT_IDS,
    SECOND_KEYPOINT_IDS,
    detect_paint_boundaries,
    detect_sidelines,
    decision_map_from_clf,
    extract_decision_contours,
    fit_rgb_linear_svm,
    line_distances,
    load_keypoint_samples,
    load_value_samples,
    pseudo_label_pixels,
    select_keypoint_by_rgb,
)
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

BBALL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BBALL_DIR, "2022-winners-camera-calibration-challenge")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

UPPER_IDS = list(range(78, 91))
LEFT_IDS = [13, 26, 39, 52, 65, 78]
RIGHT_IDS = [25, 38, 51, 64, 77, 90]
LINE_KEYPOINT_IDS = {
    "upper": UPPER_IDS,
    "left": LEFT_IDS,
    "right": RIGHT_IDS,
}

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


def _abs(path: str) -> str:
    return os.path.abspath(path)


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_nba_field_points() -> np.ndarray:
    """Return the court keypoint template in standard NBA court coordinates (cm)."""
    original_length = 2800.0
    original_width = 1500.0
    original_row_y = np.array([1500.0, 1325.0, 1120.0, 885.0, 620.0, 325.0, 0.0])
    row_y = original_row_y / original_width * NBA_FIELD_WIDTH

    points = []
    for y in row_y:
        for i in range(13):
            x = i * original_length / 12.0 / original_length * NBA_FIELD_LENGTH
            points.append([x, y, 0.0])

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


def _draw_nba_field_circle(
    img: np.ndarray,
    H: np.ndarray,
    mul_coef: np.ndarray,
    translate_coef: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    center: np.ndarray,
    radius: float,
    start_angle: float,
    stop_angle: float,
) -> None:
    field_points = []
    for theta in np.arange(start_angle, stop_angle, 0.2 * np.pi / 180):
        field_points.append(center + np.array([radius * np.cos(theta), radius * np.sin(theta)]))
    if not field_points:
        return

    pts = np.array([field_points], dtype=np.float32)
    pts *= mul_coef
    pts += translate_coef
    points_to_draw = cv2.perspectiveTransform(pts, H)[0].astype(int)

    last_point = None
    for point in points_to_draw:
        if last_point is not None:
            cv2.line(img, tuple(last_point), tuple(point), color, thickness)
        last_point = point


def _draw_nba_quarter_field(
    img: np.ndarray,
    H: np.ndarray,
    line_color: tuple[int, int, int],
    curve_color: tuple[int, int, int],
    thickness: int,
    mul_coef: np.ndarray,
    translate_coef: np.ndarray,
) -> None:
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
    points_to_draw = cv2.perspectiveTransform(field_points, H)[0].astype(int)

    cv2.line(img, tuple(points_to_draw[0]), tuple(points_to_draw[1]), line_color, thickness)
    cv2.line(img, tuple(points_to_draw[2]), tuple(points_to_draw[0]), line_color, thickness)
    cv2.line(img, tuple(points_to_draw[3]), tuple(points_to_draw[4]), line_color, thickness)
    cv2.line(img, tuple(points_to_draw[4]), tuple(points_to_draw[5]), line_color, thickness)
    cv2.line(img, tuple(points_to_draw[6]), tuple(points_to_draw[7]), line_color, thickness)
    cv2.line(img, tuple(points_to_draw[1]), tuple(points_to_draw[8]), line_color, thickness)

    _draw_nba_field_circle(
        img,
        H,
        mul_coef,
        translate_coef,
        curve_color,
        thickness,
        np.array([NBA_FREE_THROW_X, NBA_FIELD_WIDTH / 2]),
        NBA_CIRCLE_RADIUS,
        0,
        np.pi / 2,
    )
    _draw_nba_field_circle(
        img,
        H,
        mul_coef,
        translate_coef,
        curve_color,
        thickness,
        np.array([NBA_BASKET_X, NBA_FIELD_WIDTH / 2]),
        NBA_THREE_POINT_RADIUS,
        0,
        NBA_THREE_ARC_STOP_ANGLE,
    )
    _draw_nba_field_circle(
        img,
        H,
        mul_coef,
        translate_coef,
        curve_color,
        thickness,
        np.array([NBA_FIELD_LENGTH / 2, NBA_FIELD_WIDTH / 2]),
        NBA_CIRCLE_RADIUS,
        np.pi / 2,
        np.pi,
    )


def draw_nba_field(
    img: np.ndarray,
    H: np.ndarray,
    line_color: tuple[int, int, int],
    curve_color: tuple[int, int, int],
    thickness: int,
) -> None:
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
        _draw_nba_quarter_field(
            img,
            H,
            line_color,
            curve_color,
            thickness,
            mul_coef,
            translate_coef,
        )


def _keypoints_for_heuristic(kpts: dict[int, tuple[float, float, float]]) -> dict:
    return {
        str(idx): {"img_x": float(x), "img_y": float(y), "conf": float(conf)}
        for idx, (x, y, conf) in kpts.items()
    }


def _run_heuristic_sidelines(
    image_rgb: np.ndarray,
    keypoints: dict,
    pixel: int,
    rgb_threshold: float,
    svm_c: float,
) -> tuple[dict, dict, dict]:
    samples = load_keypoint_samples(keypoints, image_rgb, KEYPOINT_IDS)
    if not samples:
        raise RuntimeError("第一组关键点在模型输出中均未识别到")

    selected = select_keypoint_by_rgb(samples)
    class0_rgb = selected.rgb

    value_samples = load_value_samples(
        keypoints,
        image_rgb,
        SECOND_KEYPOINT_IDS,
        pixel,
        class0_rgb,
        rgb_threshold,
    )
    if not value_samples:
        raise RuntimeError("第二组关键点均无有效 RGB value")

    selected_second = select_keypoint_by_rgb(value_samples)
    class1_rgb = selected_second.rgb

    pixels = image_rgb.reshape(-1, 3).astype(np.float64)
    labels = pseudo_label_pixels(pixels, class0_rgb, class1_rgb)
    clf = fit_rgb_linear_svm(pixels, labels, C=svm_c)
    decision = decision_map_from_clf(image_rgb, clf)
    contours = extract_decision_contours(decision, level=0.0)
    sideline_detections = detect_sidelines(keypoints, contours)
    paint_detections = detect_paint_boundaries(keypoints, contours)

    details = {
        "class0": {
            "keypoint_id": selected.kpt_id,
            "x": round(selected.x, 2),
            "y": round(selected.y, 2),
            "rgb": [round(float(v), 2) for v in class0_rgb],
        },
        "class1": {
            "keypoint_id": selected_second.kpt_id,
            "x": round(selected_second.x, 2),
            "y": round(selected_second.y, 2),
            "rgb": [round(float(v), 2) for v in class1_rgb],
        },
        "decision_contours": len(contours),
        "svm": {
            "coef": [float(v) for v in clf.coef_[0]],
            "intercept": float(clf.intercept_[0]),
        },
    }
    return sideline_detections, paint_detections, details


def _project_point_to_line(point: np.ndarray, line: np.ndarray) -> np.ndarray:
    signed_distance = float(np.dot(line[:2], point) + line[2])
    return point - signed_distance * line[:2]


def _line_intersection(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    denom = a1 * b2 - a2 * b1
    if abs(float(denom)) < 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / denom
    y = (c1 * a2 - c2 * a1) / denom
    return np.array([x, y], dtype=np.float64)


def _adjust_keypoints(
    kpts: dict[int, tuple[float, float, float]],
    sideline_detections: dict,
) -> tuple[dict[int, tuple[float, float, float]], dict]:
    adjusted = dict(kpts)
    adjustment_details = {}

    for side, ids in LINE_KEYPOINT_IDS.items():
        detection = sideline_detections.get(side)
        if detection is None or detection.line is None:
            adjustment_details[side] = {"detected": False, "adjusted_ids": []}
            continue

        adjusted_ids = []
        for idx in ids:
            if idx not in adjusted:
                continue
            x, y, conf = adjusted[idx]
            projected = _project_point_to_line(np.array([x, y], dtype=np.float64), detection.line)
            adjusted[idx] = (float(projected[0]), float(projected[1]), conf)
            adjusted_ids.append(idx)

        adjustment_details[side] = {
            "detected": True,
            "adjusted_ids": adjusted_ids,
            "line": [float(v) for v in detection.line],
            "candidate_points": int(len(detection.candidate_points)),
            "line_points": int(len(detection.line_points)),
            "score": float(detection.score),
            "endpoints": detection.endpoints,
        }

    upper = sideline_detections.get("upper")
    left = sideline_detections.get("left")
    right = sideline_detections.get("right")

    intersections = {}
    if upper is not None and left is not None and upper.line is not None and left.line is not None:
        pt = _line_intersection(upper.line, left.line)
        if pt is not None and 78 in adjusted:
            _, _, conf = adjusted[78]
            adjusted[78] = (float(pt[0]), float(pt[1]), conf)
            intersections["78"] = [float(pt[0]), float(pt[1])]

    if upper is not None and right is not None and upper.line is not None and right.line is not None:
        pt = _line_intersection(upper.line, right.line)
        if pt is not None and 90 in adjusted:
            _, _, conf = adjusted[90]
            adjusted[90] = (float(pt[0]), float(pt[1]), conf)
            intersections["90"] = [float(pt[0]), float(pt[1])]

    adjustment_details["intersections"] = intersections
    return adjusted, adjustment_details


def _paint_prefix(paint_detections: dict) -> str | None:
    for detection in paint_detections.values():
        side_name = getattr(detection, "side", "")
        if side_name.startswith("left_paint_"):
            return "left_paint"
        if side_name.startswith("right_paint_"):
            return "right_paint"
    return None


def _add_paint_corners(
    adjusted_kpts: dict[int, tuple[float, float, float]],
    paint_detections: dict,
) -> tuple[dict[int, tuple[float, float, float]], dict]:
    adjusted = dict(adjusted_kpts)
    prefix = _paint_prefix(paint_detections)
    details = {
        "detected": False,
        "paint": prefix,
        "corner_ids": [],
        "corners": {},
    }
    if prefix is None:
        return adjusted, details

    required = ("upper", "lower", "left", "right")
    if any(
        side not in paint_detections or paint_detections[side].line is None
        for side in required
    ):
        return adjusted, details

    upper = paint_detections["upper"].line
    lower = paint_detections["lower"].line
    left = paint_detections["left"].line
    right = paint_detections["right"].line

    if prefix == "left_paint":
        corner_specs = [
            (91, left, upper),
            (92, right, upper),
            (93, right, lower),
            (94, left, lower),
        ]
    else:
        corner_specs = [
            (95, right, upper),
            (96, left, upper),
            (97, left, lower),
            (98, right, lower),
        ]

    corner_ids = []
    corners = {}
    for idx, line_a, line_b in corner_specs:
        pt = _line_intersection(line_a, line_b)
        if pt is None:
            continue
        adjusted[idx] = (float(pt[0]), float(pt[1]), 1.0)
        corner_ids.append(idx)
        corners[str(idx)] = [float(pt[0]), float(pt[1])]

    details.update(
        {
            "detected": len(corner_ids) == 4,
            "corner_ids": corner_ids,
            "corners": corners,
        }
    )
    return adjusted, details


def _homography_points(
    adjusted_kpts: dict[int, tuple[float, float, float]],
    sideline_detections: dict,
    threshold: float,
    top_k: int | None,
) -> tuple[np.ndarray, np.ndarray, list[int], str]:
    _ = sideline_detections
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

    src = np.array([field_points[idx] for idx in use_ids], dtype=np.float64)
    dst = np.array(
        [[adjusted_kpts[idx][0], adjusted_kpts[idx][1]] for idx in use_ids],
        dtype=np.float64,
    )
    return src, dst, use_ids, "constrained_optimization_all_threshold_points"


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    shifted = points - center
    mean_dist = float(np.mean(np.linalg.norm(shifted, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-9 else 1.0
    transform = np.array(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    points_h = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    normalized = (transform @ points_h.T).T[:, :2]
    return normalized, transform


def _dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    if len(src) < 4:
        return None

    src_norm, src_t = _normalize_points(src)
    dst_norm, dst_t = _normalize_points(dst)
    rows = []
    for (x, y), (u, v) in zip(src_norm, dst_norm):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.array(rows, dtype=np.float64))
    h_norm = vh[-1].reshape(3, 3)
    H = np.linalg.inv(dst_t) @ h_norm @ src_t

    if abs(float(H[2, 2])) > 1e-12:
        H = H / H[2, 2]
    else:
        norm = np.linalg.norm(H)
        if norm <= 1e-12:
            return None
        H = H / norm
    return H


def _score_initial_homography(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    try:
        inv_h = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return float("inf")
    court_pred = _transform_points(inv_h, dst)
    errors = np.linalg.norm(court_pred - src, axis=1)
    finite_errors = errors[np.isfinite(errors)]
    if len(finite_errors) != len(errors):
        return float("inf")
    return float(np.median(errors) + 0.25 * np.mean(errors))


def _robust_initial_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    full_h = _dlt_homography(src, dst)
    best_h = full_h
    best_score = (
        _score_initial_homography(full_h, src, dst)
        if full_h is not None
        else float("inf")
    )

    if len(src) < 4:
        return best_h

    rng = np.random.default_rng(0)
    iterations = 800 if len(src) > 4 else 1
    tried: set[tuple[int, ...]] = set()
    for _ in range(iterations):
        ids = tuple(sorted(int(i) for i in rng.choice(len(src), size=4, replace=False)))
        if ids in tried:
            continue
        tried.add(ids)
        H = _dlt_homography(src[list(ids)], dst[list(ids)])
        if H is None:
            continue
        score = _score_initial_homography(H, src, dst)
        if score < best_score:
            best_score = score
            best_h = H

    return best_h


def _params_to_homography(params: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=np.float64,
    )


def _homography_to_params(H: np.ndarray) -> np.ndarray:
    H = H / H[2, 2] if abs(float(H[2, 2])) > 1e-12 else H
    return np.array(
        [H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]],
        dtype=np.float64,
    )


def _transform_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = (H @ points_h.T).T
    denom = projected[:, 2:3]
    safe_denom = np.where(np.abs(denom) < 1e-9, np.sign(denom) * 1e-9 + 1e-9, denom)
    return projected[:, :2] / safe_denom


def _normalize_homogeneous_line(line: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(line[:2]))
    if norm <= 1e-12 or not np.all(np.isfinite(line)):
        return None
    return line / norm


def _court_line(side: str) -> np.ndarray | None:
    if side == "upper":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if side == "left":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if side == "right":
        return np.array([1.0, 0.0, -NBA_FIELD_LENGTH], dtype=np.float64)
    return None


def _project_court_line_to_image(H: np.ndarray, court_line: np.ndarray) -> np.ndarray | None:
    try:
        image_line = np.linalg.inv(H).T @ court_line
    except np.linalg.LinAlgError:
        return None
    return _normalize_homogeneous_line(image_line)


def _line_to_line_residual(
    H: np.ndarray,
    court_line: np.ndarray,
    image_line: np.ndarray,
) -> np.ndarray | None:
    projected_line = _project_court_line_to_image(H, court_line)
    target_line = _normalize_homogeneous_line(image_line)
    if projected_line is None or target_line is None:
        return None
    if float(np.dot(projected_line, target_line)) < 0:
        projected_line = -projected_line
    # One logical line constraint: normalized homogeneous line equality.
    delta = projected_line - target_line
    return np.array([delta[0] * 10.0, delta[1] * 10.0, delta[2] / 25.0])


def _homography_residuals(
    params: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    sideline_detections: dict,
) -> np.ndarray:
    H = _params_to_homography(params)
    try:
        inv_h = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(src) * 2 + 64, 1e6, dtype=np.float64)

    residuals = []

    # Main objective: image points reprojected back to NBA court coordinates.
    court_pred = _transform_points(inv_h, dst)
    residuals.extend(((court_pred - src) / 25.0).ravel())

    # Stabilizer: keep the forward projection close in image space as well.
    image_pred = _transform_points(H, src)
    residuals.extend(((image_pred - dst) / 10.0).ravel())

    # Three line constraints at most: upper, left, right.
    for side in ("upper", "left", "right"):
        detection = sideline_detections.get(side)
        if detection is None or detection.line is None:
            continue
        court_line = _court_line(side)
        if court_line is None:
            continue
        line_residual = _line_to_line_residual(H, court_line, detection.line)
        if line_residual is not None:
            residuals.extend(line_residual)

    # Two corner constraints at most: upper-left and upper-right.
    corner_specs = [
        ("upper", "left", np.array([0.0, 0.0], dtype=np.float64)),
        ("upper", "right", np.array([NBA_FIELD_LENGTH, 0.0], dtype=np.float64)),
    ]
    for side_a, side_b, court_corner in corner_specs:
        det_a = sideline_detections.get(side_a)
        det_b = sideline_detections.get(side_b)
        if det_a is None or det_b is None or det_a.line is None or det_b.line is None:
            continue
        image_corner = _line_intersection(det_a.line, det_b.line)
        if image_corner is None:
            continue
        projected_corner = _transform_points(H, court_corner.reshape(1, 2))[0]
        residuals.extend(((projected_corner - image_corner) / 25.0).ravel())

    return np.array(residuals, dtype=np.float64)


def _point_objective_residuals(
    H: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    try:
        inv_h = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(src) * 4, 1e6, dtype=np.float64)

    residuals = []

    court_pred = _transform_points(inv_h, dst)
    residuals.extend(((court_pred - src) / 25.0).ravel())

    image_pred = _transform_points(H, src)
    residuals.extend(((image_pred - dst) / 10.0).ravel())

    return np.array(residuals, dtype=np.float64)


def _homography_objective(params: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    residuals = _point_objective_residuals(_params_to_homography(params), src, dst)
    return float(np.dot(residuals, residuals))


def _detection_line_points(detection, count: int = 2) -> np.ndarray:
    if detection.endpoints is not None:
        points = np.array(detection.endpoints, dtype=np.float64)
    elif len(detection.line_points) >= 2:
        points = np.array([detection.line_points[0], detection.line_points[-1]], dtype=np.float64)
    elif len(detection.candidate_points) >= 2:
        points = np.array([detection.candidate_points[0], detection.candidate_points[-1]], dtype=np.float64)
    else:
        return np.empty((0, 2), dtype=np.float64)

    if count >= len(points):
        return points
    return points[:count]


def _farthest_detection_line_point(detection, point: np.ndarray) -> np.ndarray | None:
    points = _detection_line_points(detection, count=2)
    if len(points) == 0:
        return None
    distances = np.linalg.norm(points - point.reshape(1, 2), axis=1)
    return points[int(np.argmax(distances))]


def _line_point_constraint(H: np.ndarray, side: str, image_point: np.ndarray) -> float:
    court_line = _court_line(side)
    if court_line is None:
        return 1e6
    projected_line = _project_court_line_to_image(H, court_line)
    if projected_line is None:
        return 1e6
    return float(np.dot(projected_line[:2], image_point) + projected_line[2])


def _hard_constraint_values(params: np.ndarray, sideline_detections: dict) -> np.ndarray:
    H = _params_to_homography(params)
    values = []

    upper = sideline_detections.get("upper")
    upper_detected = upper is not None and upper.line is not None

    if upper_detected:
        for point in _detection_line_points(upper, count=2):
            values.append(_line_point_constraint(H, "upper", point))

        for side, court_corner in (
            ("left", np.array([0.0, 0.0], dtype=np.float64)),
            ("right", np.array([NBA_FIELD_LENGTH, 0.0], dtype=np.float64)),
        ):
            detection = sideline_detections.get(side)
            if detection is None or detection.line is None:
                continue

            image_corner = _line_intersection(upper.line, detection.line)
            if image_corner is None:
                continue

            projected_corner = _transform_points(H, court_corner.reshape(1, 2))[0]
            values.extend(projected_corner - image_corner)

            far_point = _farthest_detection_line_point(detection, image_corner)
            if far_point is not None:
                values.append(_line_point_constraint(H, side, far_point))

        return np.array(values, dtype=np.float64)

    for side in ("left", "right"):
        detection = sideline_detections.get(side)
        if detection is None or detection.line is None:
            continue
        for point in _detection_line_points(detection, count=2):
            values.append(_line_point_constraint(H, side, point))

    return np.array(values, dtype=np.float64)


def _constraints_satisfied(
    params: np.ndarray,
    sideline_detections: dict,
    tolerance: float = 1e-2,
) -> bool:
    values = _hard_constraint_values(params, sideline_detections)
    return len(values) == 0 or bool(np.max(np.abs(values)) <= tolerance)


def _line_incidence_row(image_line: np.ndarray, court_point: np.ndarray) -> np.ndarray:
    a, b, c = image_line
    x, y = court_point
    return np.array(
        [a * x, a * y, a, b * x, b * y, b, c * x, c * y, c],
        dtype=np.float64,
    )


def _point_correspondence_rows(court_point: np.ndarray, image_point: np.ndarray) -> list[np.ndarray]:
    x, y = court_point
    u, v = image_point
    return [
        np.array([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v], dtype=np.float64),
    ]


def _hard_constraint_matrix(sideline_detections: dict) -> np.ndarray:
    rows = []
    upper = sideline_detections.get("upper")
    upper_detected = upper is not None and upper.line is not None

    if upper_detected:
        rows.append(_line_incidence_row(upper.line, np.array([0.0, 0.0], dtype=np.float64)))
        rows.append(_line_incidence_row(upper.line, np.array([NBA_FIELD_LENGTH, 0.0], dtype=np.float64)))

        side_specs = [
            ("left", np.array([0.0, 0.0], dtype=np.float64), np.array([0.0, NBA_FIELD_WIDTH], dtype=np.float64)),
            (
                "right",
                np.array([NBA_FIELD_LENGTH, 0.0], dtype=np.float64),
                np.array([NBA_FIELD_LENGTH, NBA_FIELD_WIDTH], dtype=np.float64),
            ),
        ]
        for side, court_corner, side_far_point in side_specs:
            detection = sideline_detections.get(side)
            if detection is None or detection.line is None:
                continue

            image_corner = _line_intersection(upper.line, detection.line)
            if image_corner is None:
                continue

            rows.extend(_point_correspondence_rows(court_corner, image_corner))
            rows.append(_line_incidence_row(detection.line, side_far_point))

        return np.array(rows, dtype=np.float64)

    for side, court_points in (
        ("left", (np.array([0.0, 0.0], dtype=np.float64), np.array([0.0, NBA_FIELD_WIDTH], dtype=np.float64))),
        (
            "right",
            (
                np.array([NBA_FIELD_LENGTH, 0.0], dtype=np.float64),
                np.array([NBA_FIELD_LENGTH, NBA_FIELD_WIDTH], dtype=np.float64),
            ),
        ),
    ):
        detection = sideline_detections.get(side)
        if detection is None or detection.line is None:
            continue
        for court_point in court_points:
            rows.append(_line_incidence_row(detection.line, court_point))

    return np.array(rows, dtype=np.float64)


def _hard_constraint_nullspace(sideline_detections: dict) -> np.ndarray | None:
    matrix = _hard_constraint_matrix(sideline_detections)
    if len(matrix) == 0:
        return None

    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    nullspace = vh[rank:].T
    if nullspace.shape[1] == 0:
        return None
    return nullspace


def _linear_constraints_satisfied(
    H: np.ndarray,
    sideline_detections: dict,
    tolerance: float = 1e-5,
) -> bool:
    matrix = _hard_constraint_matrix(sideline_detections)
    if len(matrix) == 0:
        return True
    values = matrix @ H.reshape(-1)
    return bool(np.max(np.abs(values)) <= tolerance)


def _homography_from_nullspace(nullspace: np.ndarray, coeffs: np.ndarray) -> np.ndarray | None:
    h = nullspace @ coeffs
    norm = float(np.linalg.norm(h))
    if norm <= 1e-12 or not np.all(np.isfinite(h)):
        return None
    H = h.reshape(3, 3)
    if abs(float(H[2, 2])) > 1e-12:
        return H / H[2, 2]
    return H / norm


def _nullspace_residuals(
    coeffs: np.ndarray,
    nullspace: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    H = _homography_from_nullspace(nullspace, coeffs)
    if H is None:
        return np.full(len(src) * 2, 1e6, dtype=np.float64)

    image_pred = _transform_points(H, src)
    if not np.all(np.isfinite(image_pred)):
        return np.full(len(src) * 2, 1e6, dtype=np.float64)
    return ((image_pred - dst) / 10.0).ravel()


def _score_homography_image_fit(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    image_pred = _transform_points(H, src)
    if not np.all(np.isfinite(image_pred)):
        return float("inf")
    errors = np.linalg.norm(image_pred - dst, axis=1)
    if not np.all(np.isfinite(errors)):
        return float("inf")
    return float(np.median(errors) + 0.25 * np.mean(errors))


def _optimize_homography(
    src: np.ndarray,
    dst: np.ndarray,
    sideline_detections: dict,
) -> np.ndarray | None:
    initial_h = _robust_initial_homography(src, dst)
    if initial_h is None:
        return None

    nullspace = _hard_constraint_nullspace(sideline_detections)
    if nullspace is None:
        return initial_h

    initial_h_vec = initial_h.reshape(-1)
    initial_coeffs = nullspace.T @ initial_h_vec
    if float(np.linalg.norm(initial_coeffs)) <= 1e-12:
        initial_coeffs = np.zeros(nullspace.shape[1], dtype=np.float64)
        initial_coeffs[-1] = 1.0

    starts = [initial_coeffs]
    rng = np.random.default_rng(0)
    base_norm = float(np.linalg.norm(initial_coeffs))
    if base_norm > 1e-12:
        for scale in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            for _ in range(8):
                starts.append(
                    initial_coeffs
                    + rng.normal(size=nullspace.shape[1]) * base_norm * scale
                )

    best_h = None
    best_score = float("inf")
    matrix = _hard_constraint_matrix(sideline_detections)
    for start in starts:
        result = least_squares(
            _nullspace_residuals,
            start,
            args=(nullspace, src, dst),
            method="trf",
            loss="soft_l1",
            f_scale=2.0,
            x_scale="jac",
            max_nfev=3000,
        )
        H = _homography_from_nullspace(nullspace, result.x)
        if H is None or not _linear_constraints_satisfied(H, sideline_detections):
            continue
        constraint_error = float(np.max(np.abs(matrix @ H.reshape(-1)))) if len(matrix) else 0.0
        score = _score_homography_image_fit(H, src, dst) + 1000.0 * constraint_error
        if score < best_score:
            best_score = score
            best_h = H

    if best_h is None:
        return None

    return best_h / best_h[2, 2] if abs(float(best_h[2, 2])) > 1e-12 else best_h


def _count_court_reprojection_inliers(
    H: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    threshold_cm: float = 50.0,
) -> int:
    try:
        inv_h = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 0
    court_pred = _transform_points(inv_h, dst)
    errors = np.linalg.norm(court_pred - src, axis=1)
    return int(np.count_nonzero(errors <= threshold_cm))


def _estimate_homography(
    adjusted_kpts: dict[int, tuple[float, float, float]],
    sideline_detections: dict,
    threshold: float,
    top_k: int | None,
) -> tuple[np.ndarray | None, list[int], int, str]:
    src, dst, used_ids, mode = _homography_points(
        adjusted_kpts, sideline_detections, threshold, top_k
    )
    if len(used_ids) < 4:
        return None, used_ids, 0, mode

    H = _optimize_homography(src, dst, sideline_detections)
    if H is None:
        return None, used_ids, 0, mode

    inliers = _count_court_reprojection_inliers(H, src, dst)
    return H, used_ids, inliers, mode


def _keypoints_to_json(kpts: dict[int, tuple[float, float, float]]) -> dict:
    return {
        str(idx): {"img_x": round(x, 2), "img_y": round(y, 2), "conf": round(conf, 6)}
        for idx, (x, y, conf) in sorted(kpts.items())
    }


def _draw_adjusted_keypoints(
    image_bgr: np.ndarray,
    adjusted_kpts: dict[int, tuple[float, float, float]],
    threshold: float,
    print_coords: bool,
) -> None:
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
        cx = int(round(x))
        cy = int(round(y))
        if not (0 <= cx < IMG_WIDTH and 0 <= cy < IMG_HEIGHT):
            continue
        color = row_colors[idx // 13] if idx < 91 else (0, 165, 255)
        cv2.circle(image_bgr, (cx, cy), 6, (0, 0, 0), -1)
        cv2.circle(image_bgr, (cx, cy), 5, color, -1)
        cv2.putText(
            image_bgr,
            str(idx),
            (cx + 7, cy - 3),
            cv2.FONT_HERSHEY_PLAIN,
            0.85,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            str(idx),
            (cx + 7, cy - 3),
            cv2.FONT_HERSHEY_PLAIN,
            0.85,
            color,
            1,
            cv2.LINE_AA,
        )
        if print_coords:
            fx, fy = field_points[idx]
            print(f"{idx:>4}  {x:>7.2f} {y:>7.2f}  {fx:>10.1f} {fy:>10.1f}  {conf:>8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="模型关键点 + 启发式边线修正后的篮球场标定"
    )
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="输出目录（必填）；结果文件命名为 <image>_calibrated_heuristic.png 和 <image>_result_heuristic.json",
    )
    parser.add_argument("--model", required=True, metavar="MODEL", help="模型权重路径（必填）")
    parser.add_argument("--clahe", action="store_true", help="对 L 通道做自适应直方图均衡化")
    parser.add_argument("--draw-keypoints", action="store_true", help="在输出图片上叠加调整后的关键点")
    parser.add_argument("--print-coords", action="store_true", help="配合 --draw-keypoints 和 --verbose 打印调整后关键点坐标")
    parser.add_argument("--top-k", type=int, default=None, metavar="K", help="取置信度最高的 K 个边线关键点计算单应矩阵")
    parser.add_argument("--threshold", type=float, default=0.0, metavar="T", help="置信度阈值，低于该值的关键点不参与计算")
    parser.add_argument("--check", type=int, default=None, metavar="N", help="最少有效关键点数量（置信度 ≥ threshold）")
    parser.add_argument("--valid-model", default=None, metavar="MODEL", help="验证模型路径；差异过大时输出原始图像")
    parser.add_argument("--valid-diff-threshold", type=float, default=20.0, metavar="PX", help="两模型关键点平均像素距离上限")
    parser.add_argument("--pixel", type=int, default=40, help="启发式第二阶段正方形边长")
    parser.add_argument(
        "--rgb-threshold",
        type=float,
        default=40.0,
        help="正方形内像素与地板 RGB 原型的距离阈值",
    )
    parser.add_argument("--C", type=float, default=1.0, help="RGB 线性 SVM 惩罚系数")
    parser.add_argument("--verbose", action="store_true", help="打印调试信息")
    args = parser.parse_args()

    args.input = _abs(args.input)
    args.model = _abs(args.model)
    args.output_dir = _abs(args.output_dir)
    if args.valid_model:
        args.valid_model = _abs(args.valid_model)
    os.makedirs(args.output_dir, exist_ok=True)

    stem = Path(args.input).stem
    calibrated_path = os.path.join(args.output_dir, f"{stem}_calibrated_heuristic.png")
    json_path = os.path.join(args.output_dir, f"{stem}_result_heuristic.json")
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
            vprint(
                f"Error: 置信度 ≥ {args.threshold} 的有效关键点数量为 {valid_count}，"
                f"少于 --check {args.check}，输出原始图像。"
            )
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

    adjusted_kpts, adjustment_details = _adjust_keypoints(kpts, sideline_detections)
    adjusted_kpts, paint_corner_details = _add_paint_corners(
        adjusted_kpts,
        paint_detections,
    )
    H, used_ids, inliers, homography_mode = _estimate_homography(
        adjusted_kpts,
        sideline_detections,
        args.threshold,
        args.top_k,
    )

    result_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if H is not None:
        ctx = contextlib.redirect_stdout(io.StringIO()) if not args.verbose else contextlib.nullcontext()
        with ctx:
            draw_nba_field(result_bgr, H, (0, 0, 255), (0, 0, 255), 2)
    else:
        result_data["status"] = "insufficient_points"
        vprint("警告：上/左/右边线上的有效关键点少于 4，无法估计单应矩阵。")

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
    result_data["homography_keypoint_ids"] = used_ids
    result_data["homography_inliers"] = inliers
    result_data["homography_mode"] = homography_mode

    cv2.imwrite(calibrated_path, result_bgr)
    _save_json(json_path, result_data)
    vprint(f"已保存: {calibrated_path}")
    vprint(f"结果 JSON: {json_path}")


if __name__ == "__main__":
    main()
