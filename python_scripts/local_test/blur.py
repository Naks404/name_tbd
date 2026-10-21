from __future__ import annotations

from typing import Callable, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
import time
import tkinter as tk


def _to_gray_float64(image: np.ndarray) -> np.ndarray:
    """
    Convert input image to a contiguous grayscale float64 array.

    Accepts:
      - HxW grayscale
      - HxWx3 BGR/RGB-like arrays
      - HxWx4 BGRA/RGBA-like arrays

    For 3+ channel inputs, OpenCV-style BGR conversion is used for 3/4-channel
    images, otherwise channels are averaged.
    """
    arr = np.asarray(image)

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.shape[2] == 4:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        else:
            gray = arr.mean(axis=2)
    else:
        raise ValueError("image must be a 2D grayscale array or a 3D color array")

    return np.ascontiguousarray(gray.astype(np.float64, copy=False))


def _check_window_size(window_size: int) -> int:
    """
    Window size must be a positive odd integer so the window is centered.
    """
    if not isinstance(window_size, (int, np.integer)):
        raise TypeError("window_size must be an integer")
    window_size = int(window_size)
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    return window_size


def _box_sum(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Fast summed window response with reflect padding.
    """
    return cv2.boxFilter(
        arr,
        ddepth=cv2.CV_64F,
        ksize=(window_size, window_size),
        normalize=False,
        borderType=cv2.BORDER_REFLECT101,
    )


def lap1_energy_of_laplacian(
    image: np.ndarray,
    window_size: int,
) -> Tuple[float, np.ndarray]:
    """
    LAP1: energy of the Laplacian.

    Returns:
        score: mean of the local response map
        response_map: per-pixel local focus map
    """
    img = _to_gray_float64(image)
    window_size = _check_window_size(window_size)

    # Standard discrete Laplacian stage for this family
    lap_kernel = np.array(
        [[0.0, -1.0, 0.0],
         [-1.0,  4.0, -1.0],
         [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )

    lap = cv2.filter2D(
        img,
        ddepth=cv2.CV_64F,
        kernel=lap_kernel,
        borderType=cv2.BORDER_REFLECT101,
    )

    response_map = _box_sum(lap * lap, window_size)
    score = float(response_map.mean())
    return score, response_map


def lap2_modified_laplacian(
    image: np.ndarray,
    window_size: int,
) -> Tuple[float, np.ndarray]:
    """
    LAP2: modified Laplacian.

    Pertuz et al. use:
        DmI = |I * [1, -2, 1]| + |I * [1, -2, 1]^T|
    accumulated over the local window.
    """
    img = _to_gray_float64(image)
    window_size = _check_window_size(window_size)
    noise_threshold = 5

    kx = np.array([[-1.0, 2.0, -1.0]], dtype=np.float64)
    ky = kx.T

    mx = cv2.filter2D(
        img,
        ddepth=cv2.CV_64F,
        kernel=kx,
        borderType=cv2.BORDER_REFLECT101,
    )
    my = cv2.filter2D(
        img,
        ddepth=cv2.CV_64F,
        kernel=ky,
        borderType=cv2.BORDER_REFLECT101,
    )

    ml_map = np.abs(mx) + np.abs(my)
    ml_map[ml_map < noise_threshold] = 0
    response_map = _box_sum(ml_map, window_size)
    score = float(response_map.mean())
    return score, response_map


def lap4_variance_of_laplacian(
    image: np.ndarray,
    window_size: int,
) -> Tuple[float, np.ndarray]:
    """
    LAP4: variance of the Laplacian.

    The local response is the sum of squared deviations of the Laplacian
    inside the window, matching the Pertuz et al. SFF adaptation.
    """
    img = _to_gray_float64(image)
    window_size = _check_window_size(window_size)

    lap_kernel = np.array(
        [[0.0, -1.0, 0.0],
         [-1.0,  4.0, -1.0],
         [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )

    lap = cv2.filter2D(
        img,
        ddepth=cv2.CV_64F,
        kernel=lap_kernel,
        borderType=cv2.BORDER_REFLECT101,
    )

    area = float(window_size * window_size)
    sum_lap = cv2.boxFilter(
        lap,
        ddepth=cv2.CV_64F,
        ksize=(window_size, window_size),
        normalize=False,
        borderType=cv2.BORDER_REFLECT101,
    )
    sum_lap2 = cv2.boxFilter(
        lap * lap,
        ddepth=cv2.CV_64F,
        ksize=(window_size, window_size),
        normalize=False,
        borderType=cv2.BORDER_REFLECT101,
    )

    # sum((x - mean)^2) = sum(x^2) - N * mean^2
    response_map = sum_lap2 - (sum_lap * sum_lap) / area
    response_map = np.maximum(response_map, 0.0)

    score = float(response_map.mean())
    return score, response_map


def gra6_tenengrad(
    image: np.ndarray,
    window_size: int,
) -> Tuple[float, np.ndarray]:
    """
    GRA6: Tenengrad.

    Uses Sobel gradients and accumulates Gx^2 + Gy^2 in the local window.
    """
    img = _to_gray_float64(image)
    window_size = _check_window_size(window_size)

    gx = cv2.Sobel(
        img,
        ddepth=cv2.CV_64F,
        dx=1,
        dy=0,
        ksize=3,
        borderType=cv2.BORDER_REFLECT101,
    )
    gy = cv2.Sobel(
        img,
        ddepth=cv2.CV_64F,
        dx=0,
        dy=1,
        ksize=3,
        borderType=cv2.BORDER_REFLECT101,
    )

    response_map = _box_sum(gx * gx + gy * gy, window_size)
    score = float(response_map.mean())
    return score, response_map


def mis2_brenner(
    image: np.ndarray,
    window_size: int,
) -> Tuple[float, np.ndarray]:
    """
    MIS2: Brenner's measure.

    Pertuz et al. use the horizontal 2-pixel second-difference form:
        |I(i, j + 2) - I(i, j)|^2
    accumulated over the local window.

    Padding is reflect-based so the output map keeps the same size as input.
    """
    img = _to_gray_float64(image)
    window_size = _check_window_size(window_size)

    # Right-pad by 2 so the horizontal 2-pixel difference is defined for all pixels.
    pad = np.pad(img, ((0, 0), (0, 2)), mode="reflect")
    diff = pad[:, 2:] - pad[:, :-2]
    response_map = _box_sum(diff * diff, window_size)

    score = float(response_map.mean())
    return score, response_map

def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Normalize a floating-point response map to 8-bit for display only."""
    disp = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return disp.astype(np.uint8)


def make_display_tile(
    image_bgr: np.ndarray,
    title: str,
    score: float,
    tile_size: Tuple[int, int] | None = None,
) -> np.ndarray:

    tile = image_bgr.copy()

    # resize ONLY the image itself
    if tile_size is not None:
        tile = cv2.resize(tile, tile_size, interpolation=cv2.INTER_AREA)

    text = f"{title} | score: {score:.6f}"

    # translucent dark overlay INSIDE the image
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tile.shape[1], 36), (0, 0, 0), -1)

    alpha = 0.45
    tile = cv2.addWeighted(overlay, alpha, tile, 1 - alpha, 0)

    # text directly on image
    cv2.putText(
        tile,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return tile

@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: Callable[[np.ndarray, int], Tuple[float, np.ndarray]]


METHODS = [
    MethodSpec("LAP1", lap1_energy_of_laplacian),
    MethodSpec("LAP2", lap2_modified_laplacian),
    MethodSpec("LAP4", lap4_variance_of_laplacian),
    MethodSpec("GRA6", gra6_tenengrad),
    MethodSpec("MIS2", mis2_brenner),
]


def compute_grid(frame_bgr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Build a 2x3 grid:
      [ original | LAP1 | LAP2 ]
      [ LAP4     | GRA6 | MIS2 ]
    """
    h, w = frame_bgr.shape[:2]
    target_size = (w, h)

    tiles = []
    tiles.append(make_display_tile(frame_bgr, "ORIGINAL", 0.0, tile_size=target_size))

    for spec in METHODS:
        score, response_map = spec.fn(frame_bgr, window_size)
        disp = normalize_for_display(response_map)
        disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        tiles.append(make_display_tile(disp_bgr, spec.name, score, tile_size=target_size))

    # Pad to 6 tiles if needed
    if len(tiles) < 6:
        empty = np.zeros_like(tiles[0])
        tiles.append(make_display_tile(empty, "EMPTY", 0.0, tile_size=target_size))

    top = np.hstack(tiles[0:3])
    bottom = np.hstack(tiles[3:6])
    return np.vstack([top, bottom])



def main() -> None:
    camera_index = 0
    window_size = 9  # paper-like values: 5, 9, 13, 17, 21

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    last_time = time.time()
    fps = 0.0
    saved_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            grid = compute_grid(frame, window_size)

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            cv2.putText(
                grid,
                f"window={window_size}   fps={fps:.1f}   q/ESC=quit   s=save",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Resize whole grid to fit screen while preserving aspect ratio
            screen_max_w = 1600
            screen_max_h = 900

            h, w = grid.shape[:2]



            display_grid = cv2.resize(grid, (1920, 720), interpolation=cv2.INTER_AREA)
            cv2.imshow("Blur benchmark grid", display_grid)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord("s"):
                out_name = f"benchmark_grid_{saved_idx:03d}.png"
                cv2.imwrite(out_name, display_grid)
                print(f"Saved {out_name}")
                saved_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()