#!/usr/bin/env python3
"""Render sampled 3D object and controller tracks as an animation."""

from __future__ import annotations

import argparse
import colorsys
import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "data"
    / "physcore"
    / "different_types"
    / "single_lift_cloth"
    / "sampled_tracks.pkl"
)


def _load_tracks(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"sampled track file does not exist: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dictionary")

    required = ("object_points", "object_visibilities", "controller_points")
    missing = [name for name in required if name not in payload]
    if missing:
        raise KeyError(f"{path} is missing required keys: {', '.join(missing)}")

    arrays = {name: np.asarray(value) for name, value in payload.items()}
    object_points = arrays["object_points"]
    visibilities = arrays["object_visibilities"]
    controller_points = arrays["controller_points"]
    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError(
            f"object_points must have shape [T, N, 3], got {object_points.shape}"
        )
    if visibilities.shape != object_points.shape[:2]:
        raise ValueError(
            "object_visibilities must have shape [T, N], got "
            f"{visibilities.shape} for points {object_points.shape}"
        )
    if controller_points.ndim != 3 or controller_points.shape[-1] != 3:
        raise ValueError(
            "controller_points must have shape [T, M, 3], got "
            f"{controller_points.shape}"
        )
    if controller_points.shape[0] != object_points.shape[0]:
        raise ValueError(
            "object and controller frame counts differ: "
            f"{object_points.shape[0]} vs {controller_points.shape[0]}"
        )
    if not np.isfinite(object_points).all() or not np.isfinite(controller_points).all():
        raise ValueError("track coordinates contain NaN or Inf")

    colors = arrays.get("object_colors")
    if colors is not None and colors.shape != object_points.shape:
        raise ValueError(
            f"object_colors must have shape {object_points.shape}, got {colors.shape}"
        )
    arrays["object_points"] = object_points.astype(np.float32, copy=False)
    arrays["object_visibilities"] = visibilities.astype(bool, copy=False)
    arrays["controller_points"] = controller_points.astype(np.float32, copy=False)
    if colors is not None:
        arrays["object_colors"] = colors.astype(np.float32, copy=False)
    return arrays


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _persistent_track_colors(points_frame0: np.ndarray) -> np.ndarray:
    """Assign a stable rainbow color using each track's initial Y coordinate."""
    values = points_frame0[:, 1]
    value_min = float(values.min())
    value_max = float(values.max())
    scale = max(value_max - value_min, 1.0e-8)
    normalized = (values - value_min) / scale
    rgb = [
        colorsys.hsv_to_rgb(float((0.75 * (1.0 - t)) % 1.0), 0.82, 0.92)
        for t in normalized
    ]
    return np.rint(255.0 * np.asarray(rgb, dtype=np.float32)).astype(np.uint8)


def _camera_parameters(
    object_points: np.ndarray,
    visibilities: np.ndarray,
    controller_points: np.ndarray,
) -> Tuple[np.ndarray, float]:
    visible_points = object_points[visibilities]
    if visible_points.size == 0:
        raise ValueError("object_visibilities does not contain any visible point")
    all_points = np.concatenate(
        [visible_points.reshape(-1, 3), controller_points.reshape(-1, 3)], axis=0
    )
    bounds_min = all_points.min(axis=0)
    bounds_max = all_points.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    extent = float(np.max(bounds_max - bounds_min))
    return center, max(2.5 * extent, 0.5)


def _project_points(
    points: np.ndarray,
    *,
    center: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
    width: int,
    height: int,
    fov: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    forward /= np.linalg.norm(forward) + 1.0e-8
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1.0e-6:
        raise ValueError("camera elevation is too close to +/-90 degrees")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    relative = points - camera_position
    depth = relative @ forward
    camera_x = relative @ right
    camera_y = relative @ up
    focal = 0.5 * width / np.tan(np.radians(fov / 2.0))
    valid = depth > 1.0e-4
    pixels = np.zeros((len(points), 2), dtype=np.float32)
    pixels[valid, 0] = focal * camera_x[valid] / depth[valid] + width / 2.0
    pixels[valid, 1] = height / 2.0 - focal * camera_y[valid] / depth[valid]
    return pixels, depth, valid


def _draw_points(
    draw: ImageDraw.ImageDraw,
    pixels: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    colors: np.ndarray,
    *,
    radius: int,
    y_offset: int,
    width: int,
    height: int,
    outline: Optional[Tuple[int, int, int]] = None,
) -> None:
    indices = np.where(valid)[0]
    if indices.size == 0:
        return
    order = indices[np.argsort(-depth[indices])]
    depth_min = float(depth[indices].min())
    depth_range = max(float(depth[indices].max()) - depth_min, 1.0e-6)
    for index in order:
        x = int(round(float(pixels[index, 0])))
        y = int(round(float(pixels[index, 1]))) + y_offset
        if (
            x < -radius
            or x >= width + radius
            or y < y_offset - radius
            or y >= height + radius
        ):
            continue
        shade = 1.0 - 0.28 * (float(depth[index]) - depth_min) / depth_range
        color = tuple(
            np.clip(colors[index].astype(np.float32) * shade, 0, 255).astype(np.uint8)
        )
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(box, fill=color, outline=outline)


def _render_frame(
    frame_index: int,
    *,
    object_points: np.ndarray,
    object_visibilities: np.ndarray,
    controller_points: np.ndarray,
    stored_colors: Optional[np.ndarray],
    track_colors: np.ndarray,
    color_mode: str,
    center: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
    fov: float,
    width: int,
    height: int,
    point_radius: int,
    controller_radius: int,
) -> np.ndarray:
    header_height = 38
    viewport_height = height - header_height
    image = Image.new("RGB", (width, height), (247, 247, 247))
    draw = ImageDraw.Draw(image)

    visible = object_visibilities[frame_index]
    points = object_points[frame_index]
    pixels, depth, in_front = _project_points(
        points,
        center=center,
        distance=distance,
        azimuth=azimuth,
        elevation=elevation,
        width=width,
        height=viewport_height,
        fov=fov,
    )
    if color_mode == "rgb":
        assert stored_colors is not None
        colors = stored_colors[frame_index]
        if float(colors.max(initial=0.0)) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = track_colors
    _draw_points(
        draw,
        pixels,
        depth,
        in_front & visible,
        colors,
        radius=point_radius,
        y_offset=header_height,
        width=width,
        height=height,
    )

    controllers = controller_points[frame_index]
    controller_pixels, controller_depth, controller_valid = _project_points(
        controllers,
        center=center,
        distance=distance,
        azimuth=azimuth,
        elevation=elevation,
        width=width,
        height=viewport_height,
        fov=fov,
    )
    controller_colors = np.tile(
        np.array([[230, 55, 45]], dtype=np.uint8), (len(controllers), 1)
    )
    _draw_points(
        draw,
        controller_pixels,
        controller_depth,
        controller_valid,
        controller_colors,
        radius=controller_radius,
        y_offset=header_height,
        width=width,
        height=height,
        outline=(135, 25, 20),
    )

    draw.rectangle((0, 0, width, header_height), fill=(250, 250, 250))
    draw.text(
        (10, 8),
        f"sampled 3D tracks  |  frame {frame_index}  |  visible {int(visible.sum())}/{len(visible)}",
        fill=(30, 30, 30),
        font=_font(16),
    )
    legend_x = max(width - 145, 10)
    draw.ellipse((legend_x, 13, legend_x + 10, 23), fill=(230, 55, 45))
    draw.text((legend_x + 15, 8), "controller", fill=(90, 35, 30), font=_font(14))
    return np.asarray(image)


def _write_animation(
    output_path: Path,
    frame_indices: np.ndarray,
    render: Callable[[int], np.ndarray],
    fps: int,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required for animation export") from exc

    if output_path.suffix.lower() == ".gif":
        frames = [render(int(index)) for index in frame_indices]
        imageio.mimsave(str(output_path), frames, duration=1.0 / fps, loop=0)
        print(f"animation saved to {output_path}")
        return

    try:
        with imageio.get_writer(
            str(output_path), fps=fps, macro_block_size=16
        ) as writer:
            for index in frame_indices:
                writer.append_data(render(int(index)))
    except Exception as exc:
        gif_path = output_path.with_suffix(".gif")
        frames = [render(int(index)) for index in frame_indices]
        imageio.mimsave(str(gif_path), frames, duration=1.0 / fps, loop=0)
        print(f"video writer failed ({exc}); wrote GIF instead: {gif_path}")
        return
    print(f"animation saved to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize sampled 3D object and controller tracks."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--controller-radius", type=int, default=5)
    parser.add_argument("--azimuth", type=float, default=45.0)
    parser.add_argument("--elevation", type=float, default=25.0)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument(
        "--color-mode",
        choices=("track", "rgb"),
        default="track",
        help="stable per-track colors or RGB sampled from the source videos",
    )
    parser.add_argument(
        "--no-flip-z",
        dest="flip_z",
        action="store_false",
        help="keep the raw capture Z direction instead of displaying it as Z-up",
    )
    parser.set_defaults(flip_z=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="optional limit after applying --frame-step",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 38:
        raise ValueError("--fps and --width must be positive; --height must exceed 38")
    if args.point_radius < 1 or args.controller_radius < 1:
        raise ValueError("point radii must be at least 1")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    if not 1.0 < args.fov < 179.0:
        raise ValueError("--fov must be between 1 and 179 degrees")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")

    input_path = args.input.expanduser().resolve()
    tracks = _load_tracks(input_path)
    object_points = tracks["object_points"].copy()
    controller_points = tracks["controller_points"].copy()
    if args.flip_z:
        object_points[..., 2] *= -1.0
        controller_points[..., 2] *= -1.0

    stored_colors = tracks.get("object_colors")
    if args.color_mode == "rgb" and stored_colors is None:
        raise KeyError("--color-mode rgb requires object_colors in the input file")
    track_colors = _persistent_track_colors(object_points[0])
    center, distance = _camera_parameters(
        object_points,
        tracks["object_visibilities"],
        controller_points,
    )
    frame_indices = np.arange(0, len(object_points), args.frame_step, dtype=np.int64)
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else ROOT
        / "outputs"
        / "track_visualization"
        / f"{input_path.parent.name}_sampled_tracks.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"input: {input_path}")
    print(f"object tracks: {tuple(object_points.shape)}")
    print(f"controller tracks: {tuple(controller_points.shape)}")
    print(f"rendered frames: {len(frame_indices)} / {len(object_points)}")
    print(f"color mode: {args.color_mode}; display Z-up: {args.flip_z}")

    def render(index: int) -> np.ndarray:
        return _render_frame(
            index,
            object_points=object_points,
            object_visibilities=tracks["object_visibilities"],
            controller_points=controller_points,
            stored_colors=stored_colors,
            track_colors=track_colors,
            color_mode=args.color_mode,
            center=center,
            distance=distance,
            azimuth=args.azimuth,
            elevation=args.elevation,
            fov=args.fov,
            width=args.width,
            height=args.height,
            point_radius=args.point_radius,
            controller_radius=args.controller_radius,
        )

    _write_animation(output_path, frame_indices, render, args.fps)


if __name__ == "__main__":
    main()
