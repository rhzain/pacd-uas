from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    points: np.ndarray | None
    debug_image: np.ndarray
    edge_image: np.ndarray
    message: str


def order_points(points: np.ndarray) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Exactly four 2D points are required.")

    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def destination_size(ordered_points: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = ordered_points
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = max(1, int(round(max(width_top, width_bottom))))
    height = max(1, int(round(max(height_left, height_right))))
    return width, height


def correct_perspective(image_rgb: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = order_points(points)
    width, height = destination_size(src)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(image_rgb, matrix, (width, height))
    return corrected, matrix


def project_image(
    background_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    destination_points: np.ndarray,
    opacity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    dst = order_points(destination_points)
    overlay_h, overlay_w = overlay_rgb.shape[:2]
    src = np.array(
        [[0, 0], [overlay_w - 1, 0], [overlay_w - 1, overlay_h - 1], [0, overlay_h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    bg_h, bg_w = background_rgb.shape[:2]
    warped = cv2.warpPerspective(overlay_rgb, matrix, (bg_w, bg_h))

    mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * float(np.clip(opacity, 0.0, 1.0))

    blended = background_rgb.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
    return blended.clip(0, 255).astype(np.uint8), matrix


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    return float(
        0.5
        * abs(
            np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
            - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
        )
    )


def _candidate_score(
    points: np.ndarray,
    image_shape: tuple[int, int, int],
    min_area_ratio: float,
) -> float:
    image_h, image_w = image_shape[:2]
    image_area = image_h * image_w
    area = polygon_area(points)

    # Reject too small or too large (likely the image border itself)
    if area < image_area * min_area_ratio or area > image_area * 0.92:
        return -1.0
    if not cv2.isContourConvex(points.astype(np.int32).reshape(-1, 1, 2)):
        return -1.0

    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    if w < image_w * 0.05 or h < image_h * 0.05:
        return -1.0

    # --- Rectangularity: how much of the bounding rect is filled ---
    bounding_area = w * h
    solidity = area / bounding_area if bounding_area > 0 else 0.0
    # Perfect rectangle = 1.0; penalise very skewed quads
    rect_score = solidity  # range [0, 1]

    # --- Aspect ratio reasonableness (prefer common doc/billboard ratios) ---
    aspect = max(w, h) / max(1, min(w, h))
    # Reasonable ratios: 1.0 to ~2.5 (A4 ≈ 1.41, letter ≈ 1.29, billboard ≈ 2.0)
    if aspect > 5.0:
        aspect_score = 0.3
    elif aspect > 3.0:
        aspect_score = 0.6
    else:
        aspect_score = 1.0

    # --- Centre proximity: prefer objects near the image centre ---
    cx, cy = x + w / 2, y + h / 2
    dist_x = abs(cx - image_w / 2) / (image_w / 2)
    dist_y = abs(cy - image_h / 2) / (image_h / 2)
    center_score = 1.0 - 0.5 * (dist_x + dist_y) / 2  # range [0.5, 1.0]

    # --- Area ratio (normalised, but don't let huge areas dominate) ---
    area_ratio = area / image_area
    # Sweet-spot: 5%-70% of image area. Penalise extremes.
    if area_ratio > 0.85:
        area_score = 0.2
    elif area_ratio > 0.70:
        area_score = 0.6
    else:
        area_score = min(area_ratio * 2.0, 1.0)  # gentle ramp from 0 to 1

    # --- Border penalty ---
    touches_border = x <= 2 or y <= 2 or x + w >= image_w - 2 or y + h >= image_h - 2
    border_penalty = 0.40 if touches_border else 0.0

    # Weighted combination
    score = (
        0.30 * area_score
        + 0.30 * rect_score
        + 0.20 * aspect_score
        + 0.20 * center_score
        - border_penalty
    )
    return score


def _edge_maps(
    gray: np.ndarray,
    canny_low: int,
    canny_high: int,
    kernel_size: int,
    close_iterations: int,
    dilate_iterations: int,
    use_adaptive: bool,
) -> list[tuple[str, np.ndarray]]:
    # --- Pre-processing variants ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)

    # CLAHE for better local contrast (crucial for uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blurred)

    # Bilateral filter: smooth noise but keep edges sharp
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    bilateral_clahe = clahe.apply(bilateral)

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def apply_morph(edges: np.ndarray) -> np.ndarray:
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=max(1, close_iterations))
        if dilate_iterations > 0:
            closed = cv2.dilate(closed, kernel, iterations=dilate_iterations)
        return closed

    maps: list[tuple[str, np.ndarray]] = []

    # --- Canny variants on different pre-processed images ---
    sources = [
        ("equalised", equalized),
        ("CLAHE", clahe_img),
        ("bilateral+CLAHE", bilateral_clahe),
    ]
    threshold_pairs = [
        (max(1, int(canny_low * 0.6)), max(2, int(canny_high * 0.6))),
        (canny_low, canny_high),
        (min(254, int(canny_low * 1.4)), min(255, int(canny_high * 1.4))),
    ]
    for src_name, src_img in sources:
        for lower, upper in threshold_pairs:
            edges = cv2.Canny(src_img, lower, upper)
            maps.append((f"Canny {lower}-{upper} ({src_name})", apply_morph(edges)))

    # --- Morphological gradient edge detection ---
    grad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_grad = cv2.morphologyEx(clahe_img, cv2.MORPH_GRADIENT, grad_kernel)
    _, morph_grad_bin = cv2.threshold(morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maps.append(("morphological gradient", apply_morph(morph_grad_bin)))

    # --- Otsu thresholding + Canny ---
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_edges = cv2.Canny(otsu, 50, 150)
    maps.append(("Otsu+Canny", apply_morph(otsu_edges)))

    if not use_adaptive:
        return maps

    # --- Adaptive threshold variants ---
    for block_size in (11, 21, 31):
        for constant in (2, 5, 10):
            adaptive = cv2.adaptiveThreshold(
                clahe_img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                constant,
            )
            adaptive_closed = apply_morph(adaptive)
            maps.append((f"adaptive B{block_size} C{constant}", adaptive_closed))

    # --- Combined edge map: union of best Canny + adaptive ---
    combined = cv2.Canny(clahe_img, canny_low, canny_high)
    adaptive_main = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5,
    )
    combined = cv2.bitwise_or(combined, adaptive_main)
    maps.append(("combined Canny+adaptive", apply_morph(combined)))

    return maps


def auto_detect_quadrilateral(
    image_rgb: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    kernel_size: int = 5,
    close_iterations: int = 1,
    dilate_iterations: int = 0,
    min_area_ratio: float = 0.01,
    use_adaptive: bool = True,
) -> DetectionResult:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    best_points: np.ndarray | None = None
    best_score = -1.0
    best_method = ""
    best_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), canny_low, canny_high)
    debug = image_rgb.copy()

    for method, edge_map in _edge_maps(
        gray,
        canny_low,
        canny_high,
        kernel_size,
        close_iterations,
        dilate_iterations,
        use_adaptive,
    ):
        # Try both retrieval modes: LIST finds all, EXTERNAL finds outermost only
        for retr_mode, retr_name in [
            (cv2.RETR_LIST, "list"),
            (cv2.RETR_EXTERNAL, "ext"),
        ]:
            contours, _ = cv2.findContours(edge_map, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours[:100]:
                if cv2.contourArea(contour) < image_rgb.shape[0] * image_rgb.shape[1] * min_area_ratio:
                    continue

                perimeter = cv2.arcLength(contour, True)
                for epsilon_ratio in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
                    approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
                    if len(approx) == 4:
                        points = order_points(approx.reshape(4, 2).astype(np.float32))
                        score = _candidate_score(points, image_rgb.shape, min_area_ratio)
                        if score > best_score:
                            best_score = score
                            best_points = points
                            best_method = f"{method} ({retr_name}), approx {epsilon_ratio:.3f}"
                            best_edges = edge_map

                # Convex hull fallback: if the contour has >4 points, try hull
                hull = cv2.convexHull(contour)
                if len(hull) >= 4:
                    hull_perim = cv2.arcLength(hull, True)
                    for eps in (0.02, 0.04, 0.06, 0.08):
                        hull_approx = cv2.approxPolyDP(hull, eps * hull_perim, True)
                        if len(hull_approx) == 4:
                            points = order_points(hull_approx.reshape(4, 2).astype(np.float32))
                            score = _candidate_score(points, image_rgb.shape, min_area_ratio) * 0.85
                            if score > best_score:
                                best_score = score
                                best_points = points
                                best_method = f"{method} ({retr_name}), hull approx {eps:.2f}"
                                best_edges = edge_map

                # Min-area rectangle fallback
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect).astype(np.float32)
                points = order_points(box)
                score = _candidate_score(points, image_rgb.shape, min_area_ratio) * 0.75
                if score > best_score:
                    best_score = score
                    best_points = points
                    best_method = f"{method} ({retr_name}), min-area rect"
                    best_edges = edge_map

    if best_points is not None:
        cv2.polylines(debug, [best_points.astype(np.int32)], True, (30, 220, 80), 4)
        for index, point in enumerate(best_points.astype(np.int32)):
            cv2.circle(debug, tuple(point), 8, (255, 80, 40), -1)
            cv2.putText(
                debug,
                str(index + 1),
                tuple(point + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 80, 40),
                2,
                cv2.LINE_AA,
            )
        return DetectionResult(
            points=best_points,
            debug_image=debug,
            edge_image=cv2.cvtColor(best_edges, cv2.COLOR_GRAY2RGB),
            message=f"Area otomatis ditemukan dengan {best_method}.",
        )

    return DetectionResult(
        points=None,
        debug_image=cv2.cvtColor(best_edges, cv2.COLOR_GRAY2RGB),
        edge_image=cv2.cvtColor(best_edges, cv2.COLOR_GRAY2RGB),
        message=(
            "Area otomatis belum ditemukan. Kemungkinan tepi billboard putus, terlalu menyatu "
            "dengan background, atau kontur tulisan lebih dominan. Gunakan mode manual sebagai fallback."
        ),
    )


def draw_polygon(image_rgb: np.ndarray, points: np.ndarray | None) -> np.ndarray:
    preview = image_rgb.copy()
    if points is None or len(points) == 0:
        return preview

    pts = np.asarray(points, dtype=np.int32)
    for index, point in enumerate(pts):
        cv2.circle(preview, tuple(point), 8, (255, 80, 40), -1)
        cv2.putText(
            preview,
            str(index + 1),
            tuple(point + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 80, 40),
            2,
            cv2.LINE_AA,
        )

    if len(pts) >= 2:
        cv2.polylines(preview, [pts], len(pts) == 4, (40, 160, 255), 3)
    return preview


