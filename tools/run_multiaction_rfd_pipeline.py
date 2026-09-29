#!/usr/bin/env python3
"""Identify material on one episode, then run RfD for a new operation.

The identification episode may be a continuous recording containing several
known operations.  MfM consumes every complete observation window and keeps
its recurrent state across those windows.  The last per-particle material
field is then frozen and used by MPM while RfD is enabled from the first
target substep.

For a single continuous episode, use the same path for ``--identify-root``
and ``--target-root``; particle identity is preserved and direct transfer is
safe.  Separately converted episodes generally do *not* share particle
indices, so ``auto`` deliberately falls back to a confidence-weighted global
material instead of silently making a false correspondence assumption.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physcore.model_RfD import GridVelocityCorrector  # noqa: E402
from physcore.particle_flow import episode_runtime as ert  # noqa: E402
from physcore.particle_flow.dataset import (  # noqa: E402
    ParticleFlowEpisodeDataset,
    _resolve_episode_roots,
)
from physcore.particle_flow.mfm_training import defaults, material_guess  # noqa: E402
from physcore.particle_flow.mfm_validation import _tracked_l2_at_obs  # noqa: E402
from physcore.particle_flow.rfd_runtime import (  # noqa: E402
    FrozenRefiner,
    build_engine,
    chamfer_gt_to_pred,
    load_refiner,
)


def _resolved_episode_root(value: str) -> str:
    roots = _resolve_episode_roots([str(value)])
    if len(roots) != 1:
        raise ValueError(
            f"expected exactly one episode under {value!r}, found {len(roots)}: {roots}"
        )
    return str(Path(roots[0]).resolve())


def _load_episode(root: str, cfg, device: torch.device) -> Dict[str, Any]:
    dataset = ParticleFlowEpisodeDataset(
        [root],
        cache_size=1,
        real_world_domain_center=cfg.dataset.get("real_world_domain_center", None),
        observation_views=cfg.dataset.get("observation_views", None),
        observation_view_index=int(cfg.dataset.get("observation_view_index", 0)),
    )
    return ert._load_episode_tensors(dataset[0], device)


def _load_config(path: str, overrides: list[str]):
    cfg = defaults(OmegaConf.load(path))
    cfg.model.input_mode = "observed_control"
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


@torch.no_grad()
def identify_material(
    refiner: FrozenRefiner,
    ep: Dict[str, Any],
    cfg,
    *,
    end_frame: Optional[int],
    max_windows: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Run MfM over all requested complete windows and return its final field."""
    coords = ep["coords"]
    total_frames = int(coords.shape[0])
    k = int(cfg.train.update_every)
    requested_end = total_frames - 1 if end_frame is None else int(end_frame)
    if requested_end < k or requested_end >= total_frames:
        raise ValueError(
            f"--identify-end-frame must be in [{k}, {total_frames - 1}], got {requested_end}"
        )
    windows = requested_end // k
    if max_windows is not None:
        if int(max_windows) <= 0:
            raise ValueError("--max-identify-windows must be positive")
        windows = min(windows, int(max_windows))
    if windows < 1:
        raise ValueError("identification range contains no complete MfM window")

    effective_end = windows * k
    refiner.reset(coords[0:1])
    material = material_guess(ep, cfg, 0, seed=int(cfg.train.get("seed", 0)))
    confidence = material.new_ones(material.shape[0], material.shape[1], 2)
    history = []
    for window_idx in range(windows):
        f_start = window_idx * k
        f_end = f_start + k
        material, confidence = refiner.predict_window(
            ep,
            start_frame=f_start,
            end_frame=f_end,
            material=material,
        )
        row = {
            "window": window_idx,
            "start_frame": f_start,
            "end_frame": f_end,
            "log_E_mean": float(material[..., 0].mean().item()),
            "nu_mean": float(material[..., 1].mean().item()),
            "confidence_mean": float(confidence.mean().item()),
        }
        history.append(row)
        print(
            f"[MfM] window {window_idx + 1}/{windows} frames={f_start + 1}..{f_end} "
            f"log_E={row['log_E_mean']:.4f} nu={row['nu_mean']:.4f} "
            f"confidence={row['confidence_mean']:.4f}",
            flush=True,
        )
    return material, confidence, {
        "window_size": k,
        "windows": windows,
        "requested_end_frame": requested_end,
        "effective_end_frame": effective_end,
        "history": history,
    }


def _confidence_weighted_global(
    material: torch.Tensor,
    confidence: torch.Tensor,
    target_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    conf = confidence
    if conf.shape[-1] == 1:
        conf = conf.expand(*conf.shape[:-1], 2)
    conf = conf[..., :2].clamp_min(1.0e-6)
    mean_material = (material * conf).sum(dim=1, keepdim=True) / conf.sum(
        dim=1, keepdim=True
    )
    mean_confidence = conf.mean(dim=1, keepdim=True)
    return (
        mean_material.expand(-1, int(target_count), -1).contiguous(),
        mean_confidence.expand(-1, int(target_count), -1).contiguous(),
    )


@torch.no_grad()
def _nearest_transfer(
    material: torch.Tensor,
    confidence: torch.Tensor,
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    neighbors: int,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centroid-aligned inverse-distance transfer; no rotation is estimated."""
    source = source_points - source_points.mean(dim=0, keepdim=True)
    target = target_points - target_points.mean(dim=0, keepdim=True)
    source_count = int(source.shape[0])
    k = min(max(int(neighbors), 1), source_count)
    out_material = []
    out_confidence = []
    for start in range(0, int(target.shape[0]), int(chunk_size)):
        query = target[start : start + int(chunk_size)]
        distances, indices = torch.cdist(query, source).topk(k, largest=False)
        weights = distances.clamp_min(1.0e-6).reciprocal()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        gathered_material = material[0, indices]
        gathered_confidence = confidence[0, indices]
        out_material.append((gathered_material * weights[..., None]).sum(dim=1))
        out_confidence.append((gathered_confidence * weights[..., None]).sum(dim=1))
    return torch.cat(out_material, dim=0)[None], torch.cat(out_confidence, dim=0)[None]


def transfer_material(
    material: torch.Tensor,
    confidence: torch.Tensor,
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    mode: str,
    same_episode: bool,
    neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Map an identified material field onto the target particle set."""
    resolved_mode = str(mode)
    if resolved_mode == "auto":
        resolved_mode = "direct" if same_episode else "global"
    if resolved_mode == "direct":
        if material.shape[1] != target_points.shape[0]:
            raise ValueError(
                "direct transfer requires identical particle counts; use --transfer-mode "
                "global, or nearest only when the canonical geometries are aligned"
            )
        return material.clone(), confidence.clone(), resolved_mode
    if resolved_mode == "global":
        m, c = _confidence_weighted_global(material, confidence, target_points.shape[0])
        return m, c, resolved_mode
    if resolved_mode == "nearest":
        m, c = _nearest_transfer(
            material,
            confidence,
            source_points,
            target_points,
            neighbors=neighbors,
        )
        return m, c, resolved_mode
    raise ValueError(f"unknown transfer mode: {mode}")


def _load_corrector(checkpoint_path: str, cfg, device: torch.device):
    gvc_cfg = cfg.get("gvc", {})
    corrector = GridVelocityCorrector(
        in_channels=19,
        cond_dim=16,
        base_channels=int(gvc_cfg.get("base_channels", 32)),
        levels=int(gvc_cfg.get("levels", 2)),
        max_delta_v=float(gvc_cfg.get("max_delta_v", 0.05)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    corrector.load_state_dict(state, strict=True)
    corrector.requires_grad_(False)
    corrector.eval()
    return corrector


@torch.no_grad()
def _rollout_chunk(
    *,
    engine,
    ep: Dict[str, Any],
    f_start: int,
    f_end: int,
    rollout_steps: int,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    deformation: torch.Tensor,
    affine: Optional[torch.Tensor],
    material: torch.Tensor,
    confidence: torch.Tensor,
    chunk_dt: float,
    ground_height: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    frame_count = int(f_end - f_start)
    outer_steps = frame_count * int(rollout_steps)
    rigid_window = ert._piecewise_catmull_rom_window(
        ep["r_coords"], f_start, f_end, steps_per_frame=rollout_steps
    )
    controller_window = ert._piecewise_catmull_rom_window(
        ep.get("controller_grid_points", None),
        f_start,
        f_end,
        steps_per_frame=rollout_steps,
    )
    kinematic_ids = ert._kinematic_contact_ids(ep)
    contact_window = (
        ert._hold_index_window(kinematic_ids, f_start, outer_steps)
        if torch.is_tensor(kinematic_ids)
        else None
    )
    rigid_in = rigid_window[None] if rigid_window is not None else None
    controller_in = controller_window[None] if controller_window is not None else None
    delta_v = torch.zeros(
        outer_steps,
        positions.shape[0],
        positions.shape[1],
        3,
        device=positions.device,
        dtype=positions.dtype,
    )
    with ert._temporary_rollout_timestep(
        engine, dt=chunk_dt, ground_height=ground_height
    ):
        out = engine(
            positions,
            velocities,
            deformation,
            delta_v,
            material[..., 0],
            material[..., 1],
            C=affine,
            material_model_info=ep.get("particle_material_models", None),
            rigid_points=rigid_in,
            rigid_collision_cfg=ep.get("rigid_collision_cfg", {}),
            rigid_body_primitives=ep.get("rigid_body_primitives", []),
            manipulation_indicator=ep.get("manipulation_flag", None),
            manipulation_contact_particle_ids=contact_window,
            controller_grid_points=controller_in,
            material_confidence=confidence,
        )
    camera_step_indices = (
        torch.arange(1, frame_count + 1, device=positions.device) * int(rollout_steps) - 1
    )
    camera_positions = out["predicted_positions"].index_select(
        0, camera_step_indices
    )[:, 0]
    return (
        out["predicted_positions"][-1].detach(),
        out["final_velocity"].detach(),
        out["final_deformation_gradient"].detach(),
        out["final_C"].detach() if out.get("final_C", None) is not None else None,
        camera_positions.detach(),
    )


@torch.no_grad()
def rollout_target(
    *,
    engine,
    ep: Dict[str, Any],
    material: torch.Tensor,
    confidence: torch.Tensor,
    cfg,
    start_frame: int,
    end_frame: Optional[int],
    max_windows: Optional[int],
    skip_metrics: bool,
    use_rfd: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    coords = ep["coords"]
    total_frames = int(coords.shape[0])
    k = int(cfg.train.update_every)
    rollout_steps = int(cfg.dataset.get("rollout_steps", 25))
    if start_frame < 0 or start_frame >= total_frames - 1:
        raise ValueError(
            f"--target-start-frame must be in [0, {total_frames - 2}], got {start_frame}"
        )
    requested_end = total_frames - 1 if end_frame is None else int(end_frame)
    if requested_end <= start_frame or requested_end >= total_frames:
        raise ValueError(
            f"--target-end-frame must be in [{start_frame + 1}, {total_frames - 1}], "
            f"got {requested_end}"
        )
    if max_windows is not None:
        if int(max_windows) <= 0:
            raise ValueError("--max-target-windows must be positive")
        requested_end = min(requested_end, start_frame + int(max_windows) * k)

    positions = coords[start_frame : start_frame + 1]
    velocities = ep["particle_v"][start_frame : start_frame + 1]
    deformation = ep["particle_F"][start_frame : start_frame + 1]
    affine = (
        ep["particle_C"][start_frame : start_frame + 1]
        if ep.get("particle_C", None) is not None
        else None
    )
    chunk_dt = ert._real_world_chunk_dt(ep, engine, rollout_steps)
    ground_height = float(ep.get("ground_height", engine.ground_height))
    engine.enable_corrector(bool(use_rfd))

    predicted = [positions[0].detach().cpu()]
    predicted_frames = [start_frame]
    chamfer_sum = 0.0
    tracked_l2_sum = 0.0
    metric_frames = 0
    current = start_frame
    chunk_index = 0
    while current < requested_end:
        chunk_end = min(current + k, requested_end)
        positions, velocities, deformation, affine, camera_positions = _rollout_chunk(
            engine=engine,
            ep=ep,
            f_start=current,
            f_end=chunk_end,
            rollout_steps=rollout_steps,
            positions=positions,
            velocities=velocities,
            deformation=deformation,
            affine=affine,
            material=material,
            confidence=confidence,
            chunk_dt=chunk_dt,
            ground_height=ground_height,
        )
        for local_index, frame in enumerate(range(current + 1, chunk_end + 1)):
            prediction = camera_positions[local_index]
            predicted.append(prediction.cpu())
            predicted_frames.append(frame)
            if not skip_metrics:
                obs = ep.get("observation_data", None)
                if obs is not None:
                    chamfer_sum += float(
                        chamfer_gt_to_pred(
                            obs["object_points_clean"][:, frame],
                            obs["object_valid_mask"][:, frame],
                            prediction,
                        ).item()
                    )
                tracked_l2_sum += float(
                    _tracked_l2_at_obs(prediction, ep, frame).item()
                )
                metric_frames += 1
        chunk_index += 1
        print(
            f"[RfD] chunk {chunk_index} frames={current + 1}..{chunk_end} "
            f"(residual {'enabled' if use_rfd else 'disabled'})",
            flush=True,
        )
        current = chunk_end

    metrics = (
        {
            "chamfer": None,
            "tracked_l2": None,
            "frames": 0,
            "skipped": True,
        }
        if skip_metrics
        else {
            "chamfer": chamfer_sum / max(metric_frames, 1),
            "tracked_l2": tracked_l2_sum / max(metric_frames, 1),
            "frames": metric_frames,
            "skipped": False,
        }
    )
    payload = {
        "predicted": torch.stack(predicted),
        "ground_truth": coords.detach().cpu(),
        "predicted_frames": torch.tensor(predicted_frames, dtype=torch.long),
        "material": material[0].detach().cpu(),
        "material_confidence": confidence[0].detach().cpu(),
    }
    return payload, metrics


def _render_video(trajectory_path: Path, output_path: Path, fps: int) -> None:
    command = [
        sys.executable,
        str(ROOT / "tools" / "visualize_particle_trajectory.py"),
        "--input",
        str(trajectory_path),
        "--output",
        str(output_path),
        "--fps",
        str(int(fps)),
    ]
    subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config, args.cli_overrides)
    identify_root = _resolved_episode_root(args.identify_root)
    target_root = _resolved_episode_root(args.target_root)
    mfm_checkpoint = str(Path(args.mfm_checkpoint).expanduser().resolve())
    rfd_checkpoint = str(Path(args.rfd_checkpoint).expanduser().resolve())
    for checkpoint in (mfm_checkpoint, rfd_checkpoint):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)

    same_episode = identify_root == target_root
    plan = {
        "identify_root": identify_root,
        "target_root": target_root,
        "same_episode": same_episode,
        "transfer_mode_requested": args.transfer_mode,
        "mfm_checkpoint": mfm_checkpoint,
        "rfd_checkpoint": rfd_checkpoint,
        "config": str(Path(args.config).resolve()),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    device = torch.device(
        cfg.train.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError(
            "this pipeline requires CUDA: RfD uses spconv and the full particle graph is "
            "not a practical CPU workload"
        )
    torch.manual_seed(int(cfg.train.get("seed", 0)))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    identify_ep = _load_episode(identify_root, cfg, device)
    target_ep = identify_ep if same_episode else _load_episode(target_root, cfg, device)

    refiner_model = load_refiner(mfm_checkpoint, cfg, device)
    refiner = FrozenRefiner(refiner_model, cfg, device)
    source_material, source_confidence, identification = identify_material(
        refiner,
        identify_ep,
        cfg,
        end_frame=args.identify_end_frame,
        max_windows=args.max_identify_windows,
    )
    if args.target_start_frame is None:
        target_start = (
            int(identification["effective_end_frame"]) if same_episode else 0
        )
    else:
        target_start = int(args.target_start_frame)
    target_frame_count = int(target_ep["coords"].shape[0])
    if target_start < 0 or target_start >= target_frame_count - 1:
        raise ValueError(
            f"target start {target_start} leaves no rollout frames in a "
            f"{target_frame_count}-frame episode; shorten --identify-end-frame or "
            "set --target-start-frame explicitly"
        )
    if same_episode and target_start < int(identification["effective_end_frame"]):
        raise ValueError(
            "target range overlaps frames already consumed by MfM; choose a target "
            "start at or after identification.effective_end_frame to avoid leakage"
        )

    target_material, target_confidence, resolved_transfer = transfer_material(
        source_material,
        source_confidence,
        identify_ep["coords"][0],
        target_ep["coords"][target_start],
        mode=args.transfer_mode,
        same_episode=same_episode,
        neighbors=args.nearest_neighbors,
    )
    print(
        f"[transfer] mode={resolved_transfer} source_particles={source_material.shape[1]} "
        f"target_particles={target_material.shape[1]}",
        flush=True,
    )

    corrector = _load_corrector(rfd_checkpoint, cfg, device)
    engine = build_engine(cfg, device, target_root)
    engine.attach_corrector(corrector, h=int(cfg.gvc.get("h", 10)))
    trajectory, metrics = rollout_target(
        engine=engine,
        ep=target_ep,
        material=target_material,
        confidence=target_confidence,
        cfg=cfg,
        start_frame=target_start,
        end_frame=args.target_end_frame,
        max_windows=args.max_target_windows,
        skip_metrics=args.skip_metrics,
        use_rfd=not args.deactivate_rfd,
    )

    elapsed = time.perf_counter() - t0
    plan.update(
        {
            "device": str(device),
            "transfer_mode": resolved_transfer,
            "identification": identification,
            "target_start_frame": target_start,
            "target_end_frame": int(trajectory["predicted_frames"][-1]),
            "source_particles": int(source_material.shape[1]),
            "target_particles": int(target_material.shape[1]),
            "metrics": metrics,
            "elapsed_s": elapsed,
            "rfd_enabled_from_first_target_substep": not args.deactivate_rfd,
        }
    )
    trajectory["source_material"] = source_material[0].detach().cpu()
    trajectory["source_material_confidence"] = source_confidence[0].detach().cpu()
    trajectory["metadata"] = plan
    trajectory_path = output_dir / "trajectory.pt"
    report_path = output_dir / "report.json"
    torch.save(trajectory, trajectory_path)
    report_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] trajectory: {trajectory_path}", flush=True)
    print(f"[done] report: {report_path}", flush=True)
    if args.render_video:
        video_path = output_dir / "predicted_vs_ground_truth.mp4"
        _render_video(trajectory_path, video_path, args.video_fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-sequence MfM material identification, transfer the frozen "
            "material to a target operation, and enable RfD for the whole target rollout."
        )
    )
    parser.add_argument("--identify-root", required=True, help="episode used by MfM")
    parser.add_argument("--target-root", required=True, help="episode supplying the new controller trajectory")
    parser.add_argument("--mfm-checkpoint", default="data/checkpoints/MfM_checkpoint.pt")
    parser.add_argument("--rfd-checkpoint", default="data/checkpoints/RfD_checkpoint.pt")
    parser.add_argument("--config", default="configs/validate_RfD.yaml")
    parser.add_argument("--output-dir", default="outputs/multiaction_rfd")
    parser.add_argument(
        "--transfer-mode",
        choices=["auto", "direct", "global", "nearest"],
        default="auto",
        help=(
            "auto=direct only for the same episode, otherwise confidence-weighted "
            "global; nearest assumes centroid-aligned canonical geometry"
        ),
    )
    parser.add_argument("--nearest-neighbors", type=int, default=4)
    parser.add_argument("--identify-end-frame", type=int, default=None)
    parser.add_argument("--target-start-frame", type=int, default=None)
    parser.add_argument("--target-end-frame", type=int, default=None)
    parser.add_argument(
        "--max-identify-windows",
        type=int,
        default=None,
        help="debug/smoke-test limit; default uses every complete MfM window",
    )
    parser.add_argument(
        "--max-target-windows",
        type=int,
        default=None,
        help="debug/smoke-test limit; the final window may be partial",
    )
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument(
        "--deactivate-rfd",
        action="store_true",
        help="MPM-only ablation using the same frozen MfM material field",
    )
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    bad = [item for item in overrides if "=" not in item or item.startswith("-")]
    if bad:
        parser.error(f"unrecognized argument(s): {bad}")
    args.cli_overrides = overrides
    return args


if __name__ == "__main__":
    run(parse_args())
