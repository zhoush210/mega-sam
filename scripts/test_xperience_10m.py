#!/usr/bin/env python3
"""Run MegaSaM on one xperience-10m episode and evaluate camera trajectory."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts import test_egodex as egodex_utils  # pylint: disable=g-import-not-at-top


CAMERA_FILES = {
    "stereo_left": "stereo_left.mp4",
    "stereo_right": "stereo_right.mp4",
    "fisheye_cam0": "fisheye_cam0.mp4",
    "fisheye_cam1": "fisheye_cam1.mp4",
    "fisheye_cam2": "fisheye_cam2.mp4",
    "fisheye_cam3": "fisheye_cam3.mp4",
}

SUPPORTED_GT_CAMERAS = {"stereo_left"}


@dataclass
class EpisodePaths:
  episode_dir: Path
  video_path: Path
  annotation_path: Path


@dataclass
class XperienceGroundTruth:
  poses: np.ndarray
  intrinsic: np.ndarray


def _require_h5py():
  try:
    import h5py  # pylint: disable=import-outside-toplevel
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Reading xperience-10m annotation.hdf5 files requires h5py. "
        "Install it in the MegaSaM environment, for example: pip install h5py"
    ) from exc
  return h5py


def resolve_episode_paths(
    dataset_path: Path | str, episode: int | str, camera: str
) -> EpisodePaths:
  """Resolve xperience-10m episode folder, video, and annotation paths."""
  dataset_path = Path(dataset_path)
  episode_name = str(episode)
  if not episode_name.startswith("ep"):
    episode_name = f"ep{episode_name}"

  episode_dir = dataset_path / episode_name
  if not episode_dir.exists():
    raise FileNotFoundError(f"Missing episode folder: {episode_dir}")

  annotation_path = episode_dir / "annotation.hdf5"
  if not annotation_path.exists():
    raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

  video_name = CAMERA_FILES.get(camera, camera)
  video_path = episode_dir / video_name
  if not video_path.exists():
    raise FileNotFoundError(f"Missing video file: {video_path}")

  return EpisodePaths(
      episode_dir=episode_dir,
      video_path=video_path,
      annotation_path=annotation_path,
  )


def intrinsic_vector_to_matrix(values: np.ndarray) -> np.ndarray:
  """Convert xperience-10m [fx, fy, cx, cy] intrinsics to a 3x3 matrix."""
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  if values.shape[0] != 4:
    raise ValueError("intrinsic vector must have four values: fx fy cx cy")
  fx, fy, cx, cy = values.tolist()
  intrinsic = np.eye(3, dtype=np.float64)
  intrinsic[0, 0] = fx
  intrinsic[1, 1] = fy
  intrinsic[0, 2] = cx
  intrinsic[1, 2] = cy
  return intrinsic


def trans_quat_wxyz_to_matrix(
    translations: np.ndarray, quaternions_wxyz: np.ndarray
) -> np.ndarray:
  """Convert translations and wxyz quaternions to homogeneous matrices."""
  translations = np.asarray(translations, dtype=np.float64)
  quaternions_wxyz = np.asarray(quaternions_wxyz, dtype=np.float64)
  if translations.ndim != 2 or translations.shape[1] != 3:
    raise ValueError("translations must have shape (N, 3)")
  if quaternions_wxyz.ndim != 2 or quaternions_wxyz.shape[1] != 4:
    raise ValueError("quaternions_wxyz must have shape (N, 4)")
  if translations.shape[0] != quaternions_wxyz.shape[0]:
    raise ValueError("translations and quaternions must have matching length")

  q = quaternions_wxyz.copy()
  norms = np.linalg.norm(q, axis=1, keepdims=True)
  if np.any(norms <= 1e-12):
    raise ValueError("quaternions must be non-zero")
  q /= norms
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

  poses = np.repeat(np.eye(4, dtype=np.float64)[None], q.shape[0], axis=0)
  poses[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
  poses[:, 0, 1] = 2.0 * (x * y - z * w)
  poses[:, 0, 2] = 2.0 * (x * z + y * w)
  poses[:, 1, 0] = 2.0 * (x * y + z * w)
  poses[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
  poses[:, 1, 2] = 2.0 * (y * z - x * w)
  poses[:, 2, 0] = 2.0 * (x * z - y * w)
  poses[:, 2, 1] = 2.0 * (y * z + x * w)
  poses[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
  poses[:, :3, 3] = translations
  return poses


def body_pose_to_camera_pose(
    body_c2w: np.ndarray, T_camera_body: np.ndarray
) -> np.ndarray:
  """Convert body camera-to-world poses to camera camera-to-world poses."""
  body_c2w = np.asarray(body_c2w, dtype=np.float64)
  T_camera_body = np.asarray(T_camera_body, dtype=np.float64)
  if body_c2w.ndim != 3 or body_c2w.shape[1:] != (4, 4):
    raise ValueError("body_c2w must have shape (N, 4, 4)")
  if T_camera_body.shape != (4, 4):
    raise ValueError("T_camera_body must have shape (4, 4)")
  camera_c2w = body_c2w @ np.linalg.inv(T_camera_body)
  camera_c2w[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
  return camera_c2w


def load_xperience_ground_truth(
    annotation_path: Path,
    camera: str = "stereo_left",
    slam_pose_convention: str = "body_c2w",
    intrinsic_key: str = "calibration/cam01/K",
    camera_body_key: str = "calibration/cam01/T_c0_b",
) -> XperienceGroundTruth:
  """Load xperience-10m SLAM poses and stereo-left intrinsics."""
  if camera not in SUPPORTED_GT_CAMERAS:
    raise ValueError(
        "Only stereo_left GT camera conversion is currently supported; "
        f"got {camera!r}"
    )
  if slam_pose_convention not in {
      "body_c2w",
      "body_w2c",
      "camera_c2w",
      "camera_w2c",
  }:
    raise ValueError(
        "slam_pose_convention must be one of: body_c2w, body_w2c, "
        "camera_c2w, camera_w2c"
    )

  h5py = _require_h5py()
  annotation_path = Path(annotation_path)
  with h5py.File(annotation_path, "r") as f:
    for key in ("slam/trans_xyz", "slam/quat_wxyz", intrinsic_key):
      if key not in f:
        raise KeyError(f"{annotation_path} is missing HDF5 dataset '{key}'")

    translations = np.asarray(f["slam/trans_xyz"][()], dtype=np.float64)
    quaternions = np.asarray(f["slam/quat_wxyz"][()], dtype=np.float64)
    intrinsic = intrinsic_vector_to_matrix(f[intrinsic_key][()])

    T_camera_body = None
    if slam_pose_convention.startswith("body_"):
      if camera_body_key not in f:
        raise KeyError(
            f"{annotation_path} is missing HDF5 dataset '{camera_body_key}'"
        )
      T_camera_body = np.asarray(f[camera_body_key][()], dtype=np.float64)

  poses = trans_quat_wxyz_to_matrix(translations, quaternions)
  if slam_pose_convention in {"body_w2c", "camera_w2c"}:
    poses = np.linalg.inv(poses)
  if slam_pose_convention.startswith("body_"):
    poses = body_pose_to_camera_pose(poses, T_camera_body)

  if poses.shape[0] < 2:
    raise ValueError("xperience-10m episode must contain at least two poses")

  return XperienceGroundTruth(poses=poses, intrinsic=intrinsic)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run MegaSaM on one xperience-10m episode and compare to SLAM GT."
  )
  parser.add_argument("--dataset_path", type=Path, required=True)
  parser.add_argument("--episode", type=str, required=True)
  parser.add_argument(
      "--camera",
      default="stereo_left",
      choices=sorted(CAMERA_FILES.keys()),
      help="Input video stream. GT conversion currently supports stereo_left.",
  )
  parser.add_argument("--scene_name", type=str, default=None)
  parser.add_argument(
      "--work_dir", type=Path, default=Path("outputs_xperience_10m/work")
  )
  parser.add_argument(
      "--result_dir", type=Path, default=Path("outputs_xperience_10m/results")
  )
  parser.add_argument(
      "--weights", type=Path, default=Path("checkpoints/megasam_final.pth")
  )
  parser.add_argument(
      "--depth_anything_checkpoint",
      type=Path,
      default=Path("Depth-Anything/checkpoints/depth_anything_vitl14.pth"),
  )
  parser.add_argument("--mono_depth_path", type=Path, default=None)
  parser.add_argument("--metric_depth_path", type=Path, default=None)
  parser.add_argument("--frame_stride", type=int, default=1)
  parser.add_argument("--max_frames", type=int, default=None)
  parser.add_argument(
      "--slam_pose_convention",
      choices=["body_c2w", "body_w2c", "camera_c2w", "camera_w2c"],
      default="body_c2w",
  )
  parser.add_argument("--intrinsic_key", default="calibration/cam01/K")
  parser.add_argument("--camera_body_key", default="calibration/cam01/T_c0_b")
  parser.add_argument("--prediction_path", type=Path, default=None)
  parser.add_argument("--cuda_visible_devices", default="0")
  parser.add_argument("--opt_focal", action="store_true")
  parser.add_argument("--skip_frame_extract", action="store_true")
  parser.add_argument("--skip_mono_depth", action="store_true")
  parser.add_argument("--skip_tracking", action="store_true")
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--dry_run", action="store_true")
  return parser.parse_args(argv)


def _resolve_path(repo_root: Path, path: Path) -> Path:
  return path if path.is_absolute() else (repo_root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  repo_root = Path(__file__).resolve().parents[1]
  dataset_path = args.dataset_path.resolve()
  paths = resolve_episode_paths(dataset_path, args.episode, args.camera)
  gt = load_xperience_ground_truth(
      paths.annotation_path,
      camera=args.camera,
      slam_pose_convention=args.slam_pose_convention,
      intrinsic_key=args.intrinsic_key,
      camera_body_key=args.camera_body_key,
  )

  scene_base = (
      args.scene_name
      or f"xperience_10m_{dataset_path.name}_ep{str(args.episode).removeprefix('ep')}"
  )
  scene_name = egodex_utils.sanitize_scene_name(scene_base)
  work_dir = _resolve_path(repo_root, args.work_dir)
  result_root = _resolve_path(repo_root, args.result_dir)
  dataset_root = work_dir / "datasets"
  scene_root = dataset_root / scene_name
  rgb_dir = scene_root / "rgb"
  calibration_path = scene_root / "calibration.txt"
  mono_depth_parent = (
      work_dir / "Depth-Anything" / "video_visualization"
      if args.mono_depth_path is None
      else _resolve_path(repo_root, args.mono_depth_path)
  )
  metric_depth_parent = (
      work_dir / "UniDepth" / "outputs"
      if args.metric_depth_path is None
      else _resolve_path(repo_root, args.metric_depth_path)
  )

  usable_frames = min(
      egodex_utils.get_video_frame_count(paths.video_path), gt.poses.shape[0]
  )
  frame_indices = egodex_utils.select_frame_indices(
      usable_frames,
      frame_stride=args.frame_stride,
      max_frames=args.max_frames,
  )
  if not frame_indices:
    raise ValueError("No frames selected for evaluation")

  print(f"Scene: {scene_name}")
  print(f"Episode directory: {paths.episode_dir}")
  print(f"Episode video: {paths.video_path}")
  print(f"Episode annotation: {paths.annotation_path}")
  print(f"Selected frames: {len(frame_indices)} / {usable_frames}")

  if not args.skip_frame_extract:
    egodex_utils.extract_video_frames(
        paths.video_path, rgb_dir, frame_indices, force=args.force
    )
  else:
    print("Frame extraction skipped by --skip_frame_extract")
  egodex_utils.write_calibration(calibration_path, gt.intrinsic)

  if not args.skip_mono_depth:
    egodex_utils.run_depth_precomputation(
        repo_root=repo_root,
        rgb_dir=rgb_dir,
        scene_name=scene_name,
        mono_depth_parent=mono_depth_parent,
        metric_depth_parent=metric_depth_parent,
        depth_anything_checkpoint=_resolve_path(
            repo_root, args.depth_anything_checkpoint
        ),
        frame_count=len(frame_indices),
        force=args.force,
        dry_run=args.dry_run,
    )
  else:
    print("Depth precomputation skipped by --skip_mono_depth")

  prediction_path = args.prediction_path
  if not args.skip_tracking and prediction_path is None:
    prediction_path = egodex_utils.run_camera_tracking(
        repo_root=repo_root,
        dataset_root=dataset_root,
        scene_name=scene_name,
        weights=_resolve_path(repo_root, args.weights),
        mono_depth_parent=mono_depth_parent,
        metric_depth_parent=metric_depth_parent,
        cuda_visible_devices=args.cuda_visible_devices,
        opt_focal=args.opt_focal,
        force=args.force,
        dry_run=args.dry_run,
    )
  elif prediction_path is None:
    prediction_path = repo_root / "outputs" / f"{scene_name}_droid.npz"
    print(f"Camera tracking skipped by --skip_tracking; using {prediction_path}")

  if args.dry_run:
    print("Dry run finished before evaluation.")
    return 0

  pred_poses = egodex_utils.load_prediction_poses(prediction_path)
  num_eval = min(pred_poses.shape[0], len(frame_indices))
  if pred_poses.shape[0] != len(frame_indices):
    print(
        "Warning: prediction/GT frame count mismatch; "
        f"using first {num_eval} frames "
        f"(prediction={pred_poses.shape[0]}, gt={len(frame_indices)})"
    )
  if num_eval < 2:
    raise ValueError("Need at least two matched frames for evaluation")

  eval_frame_indices = frame_indices[:num_eval]
  result = egodex_utils.evaluate_trajectory(
      gt.poses[eval_frame_indices],
      pred_poses[:num_eval],
  )

  output_dir = result_root / scene_name
  metrics_json, metrics_csv = egodex_utils.write_metrics(
      output_dir, result.metrics
  )
  traj_path = egodex_utils.write_trajectories(
      output_dir, eval_frame_indices, result
  )
  plot_path = egodex_utils.plot_trajectories(
      output_dir / "trajectory.png",
      result.gt_cam_c2w,
      result.pred_aligned_cam_c2w,
      eval_frame_indices,
      title="xperience_10m camera trajectory",
  )

  print("Metrics:")
  for key, value in result.metrics.items():
    print(f"  {key}: {value:.8f}")
  print(f"Saved metrics JSON: {metrics_json}")
  print(f"Saved metrics CSV: {metrics_csv}")
  print(f"Saved trajectories: {traj_path}")
  print(f"Saved trajectory plot: {plot_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
