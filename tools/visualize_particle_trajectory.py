#!/usr/bin/env python3
"""Export a predicted-vs-ground-truth particle trajectory animation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_INPUT = Path(
    "/data5/public_data/PhysCoRe/result/MfM/single_lift_cloth/"
    "validation_episode_0000_trajectory.pt"
)


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"trajectory file does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dictionary")
    return payload


def _trajectory_array(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [T, N, 3], got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one frame and particle")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array.astype(np.float32, copy=False)


def _frame_indices(value: Any, frame_count: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    frames = np.asarray(value).reshape(-1)
    if len(frames) != frame_count:
        raise ValueError(
            f"predicted_frames has {len(frames)} entries, "
            f"but predicted has {frame_count} frames"
        )
    return frames.astype(np.int64, copy=False)


def _load_aligned_trajectories(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    payload = _load_payload(path)
    if "predicted" not in payload or "ground_truth" not in payload:
        keys = ", ".join(sorted(payload))
        raise KeyError(
            f"{path} must contain 'predicted' and 'ground_truth'. "
            f"Available keys: {keys}"
        )

    predicted = _trajectory_array(payload["predicted"], "predicted")
    ground_truth = _trajectory_array(payload["ground_truth"], "ground_truth")
    if predicted.shape[1] != ground_truth.shape[1]:
        raise ValueError(
            "predicted and ground_truth must have the same particle count: "
            f"{predicted.shape[1]} vs {ground_truth.shape[1]}"
        )

    frames = _frame_indices(payload.get("predicted_frames"), len(predicted))
    if frames is not None:
        if frames.min() < 0 or frames.max() >= len(ground_truth):
            raise IndexError(
                "predicted_frames contains an index outside ground_truth: "
                f"[{frames.min()}, {frames.max()}] vs {len(ground_truth)} frames"
            )
        ground_truth = ground_truth[frames]
    else:
        frame_count = min(len(predicted), len(ground_truth))
        predicted = predicted[:frame_count]
        ground_truth = ground_truth[:frame_count]

    return predicted, ground_truth, frames


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _project_points(
    points: np.ndarray,
    *,
    center: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
    width: int,
    height: int,
    fov: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    camera_position = center + distance * np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ],
        dtype=np.float32,
    )

    forward = center - camera_position
    forward /= np.linalg.norm(forward) + 1e-8
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)

    relative = points - camera_position
    depth = relative @ forward
    camera_x = relative @ right
    camera_y = relative @ up
    focal = 0.5 * width / np.tan(np.radians(fov / 2.0))
    valid = depth > 1e-4

    pixels = np.zeros((len(points), 2), dtype=np.float32)
    pixels[valid, 0] = focal * camera_x[valid] / depth[valid] + width / 2.0
    pixels[valid, 1] = height / 2.0 - focal * camera_y[valid] / depth[valid]
    return pixels, depth


def _render_particle_frame(
    particles: np.ndarray,
    *,
    center: np.ndarray,
    distance: float,
    width: int,
    height: int,
    azimuth: float,
    elevation: float,
    color: np.ndarray,
    point_radius: int,
) -> np.ndarray:
    image = np.full((height, width, 3), 247, dtype=np.uint8)
    pixels, depth = _project_points(
        particles,
        center=center,
        distance=distance,
        azimuth=azimuth,
        elevation=elevation,
        width=width,
        height=height,
    )
    valid_depth = depth > 1e-4
    if not valid_depth.any():
        return image

    depth_values = depth[valid_depth]
    depth_min = depth_values.min()
    depth_max = depth_values.max()
    depth_range = max(float(depth_max - depth_min), 1e-6)
    shade = np.where(
        valid_depth,
        1.0 - 0.35 * (depth - depth_min) / depth_range,
        0.0,
    )
    colors = np.clip(color[None, :].astype(np.float32) * shade[:, None], 0, 255)
    order = np.argsort(-depth)

    for point_index in order:
        if not valid_depth[point_index]:
            continue
        x = int(round(float(pixels[point_index, 0])))
        y = int(round(float(pixels[point_index, 1])))
        if x < -point_radius or x >= width + point_radius:
            continue
        if y < -point_radius or y >= height + point_radius:
            continue
        particle_color = colors[point_index].astype(np.uint8)
        for offset_x in range(-point_radius, point_radius + 1):
            for offset_y in range(-point_radius, point_radius + 1):
                if offset_x * offset_x + offset_y * offset_y > point_radius * point_radius:
                    continue
                pixel_x = x + offset_x
                pixel_y = y + offset_y
                if 0 <= pixel_x < width and 0 <= pixel_y < height:
                    image[pixel_y, pixel_x] = particle_color
    return image


def _label_frame(
    frame: np.ndarray,
    label: str,
    frame_index: int,
    color: Tuple[int, int, int],
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 30), fill=(248, 248, 248))
    draw.text(
        (10, 6),
        f"{label}  |  frame {frame_index}",
        fill=color,
        font=_font(16),
    )
    return np.asarray(image)


def _render_comparison_frames(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    width: int,
    height: int,
    azimuth: float,
    elevation: float,
    point_radius: int,
    frame_indices: Optional[np.ndarray],
) -> np.ndarray:
    all_points = np.concatenate([predicted.reshape(-1, 3), ground_truth.reshape(-1, 3)])
    bounds_min = all_points.min(axis=0)
    bounds_max = all_points.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    extent = float(np.max(bounds_max - bounds_min))
    distance = max(2.5 * extent, 0.5)
    separator = np.full((height, 6, 3), 215, dtype=np.uint8)
    frames = []

    for time_index, (predicted_frame, ground_truth_frame) in enumerate(
        zip(predicted, ground_truth)
    ):
        source_frame = (
            int(frame_indices[time_index])
            if frame_indices is not None
            else time_index
        )
        rendered_predicted = _render_particle_frame(
            predicted_frame,
            center=center,
            distance=distance,
            color=np.array([55, 120, 235], dtype=np.uint8),
            width=width,
            height=height,
            azimuth=azimuth,
            elevation=elevation,
            point_radius=point_radius,
        )
        rendered_ground_truth = _render_particle_frame(
            ground_truth_frame,
            center=center,
            distance=distance,
            color=np.array([225, 85, 65], dtype=np.uint8),
            width=width,
            height=height,
            azimuth=azimuth,
            elevation=elevation,
            point_radius=point_radius,
        )
        labeled_predicted = _label_frame(
            rendered_predicted, "Predicted", source_frame, (35, 85, 180)
        )
        labeled_ground_truth = _label_frame(
            rendered_ground_truth, "Ground truth", source_frame, (180, 55, 45)
        )
        frames.append(
            np.concatenate(
                [labeled_predicted, separator, labeled_ground_truth],
                axis=1,
            )
        )

    return np.stack(frames)


def _write_video_frames(frames: np.ndarray, output_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required for video export") from exc

    macro_block_size = 16
    pad_height = (-frames.shape[1]) % macro_block_size
    pad_width = (-frames.shape[2]) % macro_block_size
    if pad_height or pad_width:
        frames = np.pad(
            frames,
            ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=247,
        )

    try:
        imageio.mimwrite(str(output_path), list(frames), fps=fps)
    except Exception as exc:
        gif_path = output_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), list(frames), duration=1.0 / fps, loop=0)
        print(f"video writer failed ({exc}); wrote GIF instead: {gif_path}")
        return
    print(f"video saved to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize predicted and ground-truth particle trajectories."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="validation trajectory .pt containing predicted and ground_truth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output .mp4 or .gif path",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--azimuth", type=float, default=45.0)
    parser.add_argument("--elevation", type=float, default=25.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("--fps, --width, and --height must be positive")
    if args.point_radius < 1:
        raise ValueError("--point-radius must be at least 1")

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else ROOT
        / "outputs"
        / "trajectory_visualization"
        / f"{input_path.stem}_predicted_vs_ground_truth.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted, ground_truth, frame_indices = _load_aligned_trajectories(input_path)
    print(f"input: {input_path}")
    print(f"predicted: {tuple(predicted.shape)}")
    print(f"ground_truth aligned: {tuple(ground_truth.shape)}")
    if frame_indices is not None:
        print(f"frame range: {int(frame_indices[0])}..{int(frame_indices[-1])}")

    frames = _render_comparison_frames(
        predicted,
        ground_truth,
        width=args.width,
        height=args.height,
        azimuth=args.azimuth,
        elevation=args.elevation,
        point_radius=args.point_radius,
        frame_indices=frame_indices,
    )
    _write_video_frames(frames, output_path=output_path, fps=args.fps)


if __name__ == "__main__":
    main()
