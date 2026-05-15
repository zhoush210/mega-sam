#!/usr/bin/env python3
"""Run MegaSaM on one EgoDex episode and evaluate camera trajectory."""

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence

import numpy as np


METERS_TO_MILLIMETERS = 1000.0


@dataclass
class EgoDexGroundTruth:
  poses: np.ndarray
  intrinsic: np.ndarray


@dataclass
class SimilarityAlignment:
  rotation: np.ndarray
  translation: np.ndarray
  scale: float
  errors: np.ndarray


@dataclass
class EvaluationResult:
  metrics: dict[str, float]
  gt_cam_c2w: np.ndarray
  pred_cam_c2w: np.ndarray
  pred_aligned_cam_c2w: np.ndarray
  alignment: SimilarityAlignment


def sanitize_scene_name(name: str) -> str:
  """Return a filesystem- and MegaSaM-friendly scene name."""
  name = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
  name = re.sub(r"_+", "_", name).strip("_")
  if not name:
    raise ValueError("scene name is empty after sanitization")
  return name


def select_frame_indices(
    num_frames: int, frame_stride: int = 1, max_frames: int | None = None
) -> list[int]:
  """Select frame indices from an episode."""
  if num_frames < 0:
    raise ValueError("num_frames must be non-negative")
  if frame_stride < 1:
    raise ValueError("frame_stride must be >= 1")
  if max_frames is not None and max_frames < 1:
    raise ValueError("max_frames must be >= 1")

  indices = list(range(0, num_frames, frame_stride))
  if max_frames is not None:
    indices = indices[:max_frames]
  return indices


def _require_h5py():
  try:
    import h5py  # pylint: disable=import-outside-toplevel
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Reading EgoDex .hdf5 files requires h5py. Install it in the "
        "MegaSaM environment, for example: pip install h5py"
    ) from exc
  return h5py


def load_egodex_ground_truth(
    hdf5_path: Path,
    pose_key: str = "transforms/camera",
    intrinsic_key: str = "camera/intrinsic",
    pose_convention: str = "c2w",
) -> EgoDexGroundTruth:
  """Load EgoDex camera poses and intrinsics from one episode HDF5 file."""
  hdf5_path = Path(hdf5_path)
  if not hdf5_path.exists():
    raise FileNotFoundError(hdf5_path)
  if pose_convention not in {"c2w", "w2c"}:
    raise ValueError("pose_convention must be 'c2w' or 'w2c'")

  h5py = _require_h5py()
  with h5py.File(hdf5_path, "r") as f:
    if pose_key not in f:
      raise KeyError(f"{hdf5_path} is missing HDF5 dataset '{pose_key}'")
    if intrinsic_key not in f:
      raise KeyError(f"{hdf5_path} is missing HDF5 dataset '{intrinsic_key}'")
    poses = np.asarray(f[pose_key][()], dtype=np.float64)
    intrinsic = np.asarray(f[intrinsic_key][()], dtype=np.float64)

  if poses.ndim != 3 or poses.shape[1:] != (4, 4):
    raise ValueError(f"{pose_key} must have shape (N, 4, 4)")
  if intrinsic.shape != (3, 3):
    raise ValueError(f"{intrinsic_key} must have shape (3, 3)")
  if poses.shape[0] < 2:
    raise ValueError("EgoDex episode must contain at least two camera poses")

  poses = poses.copy()
  poses[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
  if pose_convention == "w2c":
    poses = np.linalg.inv(poses)

  return EgoDexGroundTruth(poses=poses, intrinsic=intrinsic)


def get_video_frame_count(video_path: Path) -> int:
  """Return the number of frames in a video file."""
  try:
    import cv2  # pylint: disable=import-outside-toplevel
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Extracting EgoDex frames requires opencv-python-headless."
    ) from exc

  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise ValueError(f"Failed to open video: {video_path}")
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cap.release()
  if frame_count <= 0:
    raise ValueError(f"Video has no readable frames: {video_path}")
  return frame_count


def _existing_png_count(path: Path) -> int:
  return len(list(path.glob("*.png")))


def extract_video_frames(
    video_path: Path,
    rgb_dir: Path,
    frame_indices: Sequence[int],
    force: bool = False,
) -> None:
  """Extract selected video frames to PNG files named 00000.png, ..."""
  if not frame_indices:
    raise ValueError("frame_indices must not be empty")
  rgb_dir.mkdir(parents=True, exist_ok=True)
  if not force and _existing_png_count(rgb_dir) == len(frame_indices):
    print(f"Frame extraction skipped; found {len(frame_indices)} PNGs in {rgb_dir}")
    return

  for path in rgb_dir.glob("*.png"):
    path.unlink()

  import cv2  # pylint: disable=import-outside-toplevel

  selected = set(frame_indices)
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise ValueError(f"Failed to open video: {video_path}")

  output_index = 0
  frame_index = 0
  while output_index < len(frame_indices):
    ret, frame = cap.read()
    if not ret:
      break
    if frame_index in selected:
      out_path = rgb_dir / f"{output_index:05d}.png"
      if not cv2.imwrite(str(out_path), frame):
        cap.release()
        raise IOError(f"Failed to write frame: {out_path}")
      output_index += 1
    frame_index += 1

  cap.release()
  if output_index != len(frame_indices):
    raise ValueError(
        f"Only extracted {output_index} frames from {video_path}, "
        f"expected {len(frame_indices)}"
    )


def write_calibration(path: Path, intrinsic: np.ndarray) -> None:
  """Write MegaSaM/Sintel-style fx fy cx cy calibration file."""
  path.parent.mkdir(parents=True, exist_ok=True)
  fx = float(intrinsic[0, 0])
  fy = float(intrinsic[1, 1])
  cx = float(intrinsic[0, 2])
  cy = float(intrinsic[1, 2])
  path.write_text(f"{fx:.9g} {fy:.9g} {cx:.9g} {cy:.9g}\n", encoding="utf-8")


def _count_files(path: Path, suffix: str) -> int:
  return len(list(path.glob(f"*{suffix}")))


def _clear_files(path: Path, suffix: str) -> None:
  if not path.exists():
    return
  for item in path.glob(f"*{suffix}"):
    item.unlink()


def _run_command(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
  printable = " ".join(str(x) for x in command)
  print(f"$ {printable}")
  if dry_run:
    return
  subprocess.run(command, cwd=str(cwd), env=env, check=True)


def _with_pythonpath(env: dict[str, str], path: Path) -> dict[str, str]:
  env = env.copy()
  existing = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = str(path) if not existing else f"{existing}:{path}"
  return env


def run_depth_precomputation(
    repo_root: Path,
    rgb_dir: Path,
    scene_name: str,
    mono_depth_parent: Path,
    metric_depth_parent: Path,
    depth_anything_checkpoint: Path,
    frame_count: int,
    force: bool = False,
    dry_run: bool = False,
) -> None:
  """Run Depth-Anything and UniDepth unless outputs already exist."""
  mono_scene_dir = mono_depth_parent / scene_name
  metric_scene_dir = metric_depth_parent / scene_name

  if force or _count_files(mono_scene_dir, ".npy") != frame_count:
    mono_scene_dir.mkdir(parents=True, exist_ok=True)
    _clear_files(mono_scene_dir, ".npy")
    _run_command(
        [
            sys.executable,
            "Depth-Anything/run_videos.py",
            "--encoder",
            "vitl",
            "--load-from",
            str(depth_anything_checkpoint),
            "--img-path",
            str(rgb_dir),
            "--outdir",
            str(mono_scene_dir),
        ],
        cwd=repo_root,
        dry_run=dry_run,
    )
  else:
    print(f"Depth-Anything skipped; found {frame_count} files in {mono_scene_dir}")

  if force or _count_files(metric_scene_dir, ".npz") != frame_count:
    metric_scene_dir.mkdir(parents=True, exist_ok=True)
    _clear_files(metric_scene_dir, ".npz")
    env = _with_pythonpath(os.environ.copy(), repo_root / "UniDepth")
    _run_command(
        [
            sys.executable,
            "UniDepth/scripts/demo_mega-sam.py",
            "--scene-name",
            scene_name,
            "--img-path",
            str(rgb_dir),
            "--outdir",
            str(metric_depth_parent),
        ],
        cwd=repo_root,
        env=env,
        dry_run=dry_run,
    )
  else:
    print(f"UniDepth skipped; found {frame_count} files in {metric_scene_dir}")


def run_camera_tracking(
    repo_root: Path,
    dataset_root: Path,
    scene_name: str,
    weights: Path,
    mono_depth_parent: Path,
    metric_depth_parent: Path,
    cuda_visible_devices: str,
    opt_focal: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
  """Run MegaSaM camera tracking through the existing Sintel-style script."""
  prediction_path = repo_root / "outputs" / f"{scene_name}_droid.npz"
  if prediction_path.exists() and not force:
    print(f"Camera tracking skipped; found {prediction_path}")
    return prediction_path

  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
  command = [
      sys.executable,
      "camera_tracking_scripts/test_sintel.py",
      "--datapath",
      str(dataset_root),
      "--weights",
      str(weights),
      "--scene_name",
      scene_name,
      "--mono_depth_path",
      str(mono_depth_parent),
      "--metric_depth_path",
      str(metric_depth_parent),
      "--disable_vis",
  ]
  if opt_focal:
    command.append("--opt_focal")
  _run_command(command, cwd=repo_root, env=env, dry_run=dry_run)
  return prediction_path


def load_prediction_poses(prediction_path: Path) -> np.ndarray:
  """Load MegaSaM predicted camera-to-world poses from a .npz result."""
  prediction_path = Path(prediction_path)
  if not prediction_path.exists():
    raise FileNotFoundError(prediction_path)
  with np.load(prediction_path) as data:
    if "cam_c2w" not in data.files:
      raise ValueError(f"{prediction_path} is missing required key 'cam_c2w'")
    poses = np.asarray(data["cam_c2w"], dtype=np.float64)
  if poses.ndim != 3 or poses.shape[1:] != (4, 4):
    raise ValueError("cam_c2w must have shape (N, 4, 4)")
  if poses.shape[0] < 2:
    raise ValueError("Prediction must contain at least two poses")
  poses = poses.copy()
  poses[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
  return poses


def _trajectory_normalize_scale(gt_cam_c2w: np.ndarray) -> float:
  full_t = np.linalg.inv(gt_cam_c2w[-1]) @ gt_cam_c2w[0]
  endpoint = float(np.linalg.norm(full_t[:3, 3]))
  if endpoint > 1e-8:
    return endpoint
  path = np.diff(gt_cam_c2w[:, :3, 3], axis=0)
  path_length = float(np.sum(np.linalg.norm(path, axis=1)))
  return max(path_length, 1e-8)


def align_trajectories(model_xyz: np.ndarray, data_xyz: np.ndarray) -> SimilarityAlignment:
  """Align model points to data points with a similarity transform."""
  model_xyz = np.asarray(model_xyz, dtype=np.float64)
  data_xyz = np.asarray(data_xyz, dtype=np.float64)
  if model_xyz.shape != data_xyz.shape or model_xyz.ndim != 2 or model_xyz.shape[1] != 3:
    raise ValueError("model_xyz and data_xyz must both have shape (N, 3)")
  if model_xyz.shape[0] < 2:
    raise ValueError("Need at least two points for trajectory alignment")

  model_mean = model_xyz.mean(axis=0)
  data_mean = data_xyz.mean(axis=0)
  model_centered = model_xyz - model_mean
  data_centered = data_xyz - data_mean

  W = model_centered.T @ data_centered
  U, _, Vt = np.linalg.svd(W.T)
  correction = np.eye(3)
  if np.linalg.det(U @ Vt) < 0.0:
    correction[2, 2] = -1.0
  rotation = U @ correction @ Vt

  rotated_model = (rotation @ model_centered.T).T
  denom = float(np.sum(model_centered * model_centered))
  if denom <= 1e-12:
    raise ValueError("Predicted trajectory is degenerate; cannot align scale")
  scale = float(np.sum(data_centered * rotated_model) / denom)
  translation = data_mean - scale * (rotation @ model_mean)
  aligned = (scale * (rotation @ model_xyz.T)).T + translation
  errors = np.linalg.norm(aligned - data_xyz, axis=1)
  return SimilarityAlignment(
      rotation=rotation,
      translation=translation,
      scale=scale,
      errors=errors,
  )


def apply_alignment_to_poses(
    poses: np.ndarray, alignment: SimilarityAlignment
) -> np.ndarray:
  """Apply a trajectory similarity alignment to camera-to-world poses."""
  aligned = poses.copy()
  aligned[:, :3, 3] = (
      alignment.scale * (alignment.rotation @ poses[:, :3, 3].T)
  ).T + alignment.translation
  aligned[:, :3, :3] = alignment.rotation @ poses[:, :3, :3]
  aligned[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
  return aligned


def _rotation_angle(transform: np.ndarray) -> float:
  value = (np.trace(transform[:3, :3]) - 1.0) / 2.0
  value = float(np.clip(value, -1.0, 1.0))
  if value > 1.0 - 1e-12:
    return 0.0
  return float(np.arccos(value))


def relative_pose_errors(
    gt_cam_c2w: np.ndarray, pred_cam_c2w: np.ndarray, delta: int = 1
) -> tuple[np.ndarray, np.ndarray]:
  """Compute adjacent-frame relative translation and rotation errors."""
  if delta < 1:
    raise ValueError("delta must be >= 1")
  if gt_cam_c2w.shape != pred_cam_c2w.shape:
    raise ValueError("gt_cam_c2w and pred_cam_c2w must have matching shapes")
  if gt_cam_c2w.shape[0] <= delta:
    raise ValueError("Need more frames than delta for relative pose error")

  trans_errors = []
  rot_errors = []
  for i in range(gt_cam_c2w.shape[0] - delta):
    j = i + delta
    pred_rel = np.linalg.inv(pred_cam_c2w[j]) @ pred_cam_c2w[i]
    gt_rel = np.linalg.inv(gt_cam_c2w[j]) @ gt_cam_c2w[i]
    error = np.linalg.inv(pred_rel) @ gt_rel
    trans_errors.append(np.linalg.norm(error[:3, 3]))
    rot_errors.append(_rotation_angle(error))
  return np.asarray(trans_errors), np.asarray(rot_errors)


def evaluate_trajectory(
    gt_cam_c2w: np.ndarray, pred_cam_c2w: np.ndarray
) -> EvaluationResult:
  """Normalize GT, align prediction, and compute distance metrics in mm."""
  gt_cam_c2w = np.asarray(gt_cam_c2w, dtype=np.float64)
  pred_cam_c2w = np.asarray(pred_cam_c2w, dtype=np.float64)
  if gt_cam_c2w.shape != pred_cam_c2w.shape:
    raise ValueError("gt_cam_c2w and pred_cam_c2w must have matching shapes")
  if gt_cam_c2w.ndim != 3 or gt_cam_c2w.shape[1:] != (4, 4):
    raise ValueError("poses must have shape (N, 4, 4)")
  if gt_cam_c2w.shape[0] < 2:
    raise ValueError("Need at least two poses to evaluate a trajectory")

  normalize_scale = _trajectory_normalize_scale(gt_cam_c2w)
  gt_norm = gt_cam_c2w.copy()
  gt_norm[:, :3, 3] /= normalize_scale

  alignment = align_trajectories(pred_cam_c2w[:, :3, 3], gt_norm[:, :3, 3])
  pred_aligned = apply_alignment_to_poses(pred_cam_c2w, alignment)

  rte_errors, rre_errors = relative_pose_errors(gt_norm, pred_aligned, delta=1)
  distance_scale_mm = normalize_scale * METERS_TO_MILLIMETERS
  metrics = {
      "normalize_scale": float(distance_scale_mm),
      "ATE": float(np.sqrt(np.mean(alignment.errors**2)) * distance_scale_mm),
      "RTE": float(np.sqrt(np.mean(rte_errors**2)) * distance_scale_mm),
      "RRE": float(np.rad2deg(np.sqrt(np.mean(rre_errors**2)))),
  }

  return EvaluationResult(
      metrics=metrics,
      gt_cam_c2w=gt_norm,
      pred_cam_c2w=pred_cam_c2w,
      pred_aligned_cam_c2w=pred_aligned,
      alignment=alignment,
  )


def write_metrics(output_dir: Path, metrics: dict[str, float]) -> tuple[Path, Path]:
  """Write metrics to JSON and CSV files."""
  output_dir.mkdir(parents=True, exist_ok=True)
  metrics = {key: float(value) for key, value in metrics.items()}
  json_path = output_dir / "metrics.json"
  csv_path = output_dir / "metrics.csv"
  json_path.write_text(
      json.dumps(metrics, indent=2, sort_keys=False) + "\n",
      encoding="utf-8",
  )
  with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "value"])
    writer.writeheader()
    for key, value in metrics.items():
      writer.writerow({"metric": key, "value": value})
  return json_path, csv_path


def write_trajectories(
    output_dir: Path,
    frame_indices: Sequence[int],
    result: EvaluationResult,
) -> Path:
  """Save GT, raw prediction, aligned prediction, and frame indices."""
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / "trajectories.npz"
  np.savez(
      path,
      frame_indices=np.asarray(frame_indices, dtype=np.int64),
      gt_cam_c2w=result.gt_cam_c2w,
      pred_cam_c2w=result.pred_cam_c2w,
      pred_aligned_cam_c2w=result.pred_aligned_cam_c2w,
      alignment_rotation=result.alignment.rotation,
      alignment_translation=result.alignment.translation,
      alignment_scale=np.asarray(result.alignment.scale),
  )
  return path


def trajectory_plot_specs() -> list[tuple[str, str, int]]:
  """Return frame-coordinate plot specs for trajectory visualization."""
  return [("frame", "x", 0), ("frame", "y", 1), ("frame", "z", 2)]


def plot_trajectories(
    output_path: Path,
    gt_cam_c2w: np.ndarray,
    pred_aligned_cam_c2w: np.ndarray,
    frame_indices: Sequence[int] | None = None,
    title: str = "EgoDex camera trajectory",
) -> Path:
  """Save a trajectory plot comparing aligned prediction and GT."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault(
      "MPLCONFIGDIR", str(output_path.parent / "matplotlib_cache")
  )
  import matplotlib  # pylint: disable=import-outside-toplevel

  matplotlib.use("Agg", force=True)
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  gt_t = gt_cam_c2w[:, :3, 3]
  pred_t = pred_aligned_cam_c2w[:, :3, 3]
  frames = (
      np.arange(gt_t.shape[0], dtype=np.int64)
      if frame_indices is None
      else np.asarray(frame_indices, dtype=np.int64)
  )
  if frames.shape[0] != gt_t.shape[0]:
    raise ValueError("frame_indices must match trajectory length")

  fig, axes = plt.subplots(
      3, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True
  )
  for ax, (x_name, y_name, coord_i) in zip(axes, trajectory_plot_specs()):
    ax.plot(frames, gt_t[:, coord_i], "-o", markersize=2, label="GT")
    ax.plot(
        frames,
        pred_t[:, coord_i],
        "-o",
        markersize=2,
        label="MegaSaM",
    )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{x_name}-{y_name}")
    ax.grid(True, linewidth=0.4, alpha=0.4)
  axes[0].legend(loc="best")
  fig.suptitle(title)
  fig.savefig(output_path, dpi=180)
  plt.close(fig)
  return output_path


def _episode_paths(dataset_path: Path, episode: int) -> tuple[Path, Path]:
  video_path = dataset_path / f"{episode}.mp4"
  hdf5_path = dataset_path / f"{episode}.hdf5"
  if not video_path.exists():
    raise FileNotFoundError(video_path)
  if not hdf5_path.exists():
    raise FileNotFoundError(hdf5_path)
  return video_path, hdf5_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run MegaSaM on one EgoDex episode and compare to GT poses."
  )
  parser.add_argument("--dataset_path", type=Path, required=True)
  parser.add_argument("--episode", type=int, required=True)
  parser.add_argument("--scene_name", type=str, default=None)
  parser.add_argument("--work_dir", type=Path, default=Path("outputs_egodex/work"))
  parser.add_argument(
      "--result_dir", type=Path, default=Path("outputs_egodex/results")
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
  parser.add_argument("--pose_key", default="transforms/camera")
  parser.add_argument("--intrinsic_key", default="camera/intrinsic")
  parser.add_argument("--gt_pose_convention", choices=["c2w", "w2c"], default="c2w")
  parser.add_argument("--prediction_path", type=Path, default=None)
  parser.add_argument("--cuda_visible_devices", default="0")
  parser.add_argument("--opt_focal", action="store_true")
  parser.add_argument("--skip_frame_extract", action="store_true")
  parser.add_argument("--skip_mono_depth", action="store_true")
  parser.add_argument("--skip_tracking", action="store_true")
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--dry_run", action="store_true")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  repo_root = Path(__file__).resolve().parents[1]
  dataset_path = args.dataset_path.resolve()
  video_path, hdf5_path = _episode_paths(dataset_path, args.episode)
  gt = load_egodex_ground_truth(
      hdf5_path,
      pose_key=args.pose_key,
      intrinsic_key=args.intrinsic_key,
      pose_convention=args.gt_pose_convention,
  )

  scene_base = args.scene_name or f"egodex_{dataset_path.name}_{args.episode}"
  scene_name = sanitize_scene_name(scene_base)
  work_dir = (repo_root / args.work_dir).resolve() if not args.work_dir.is_absolute() else args.work_dir
  result_root = (repo_root / args.result_dir).resolve() if not args.result_dir.is_absolute() else args.result_dir
  dataset_root = work_dir / "datasets"
  scene_root = dataset_root / scene_name
  rgb_dir = scene_root / "rgb"
  calibration_path = scene_root / "calibration.txt"
  mono_depth_parent = (
      work_dir / "Depth-Anything" / "video_visualization"
      if args.mono_depth_path is None
      else args.mono_depth_path
  )
  metric_depth_parent = (
      work_dir / "UniDepth" / "outputs"
      if args.metric_depth_path is None
      else args.metric_depth_path
  )
  if not mono_depth_parent.is_absolute():
    mono_depth_parent = (repo_root / mono_depth_parent).resolve()
  if not metric_depth_parent.is_absolute():
    metric_depth_parent = (repo_root / metric_depth_parent).resolve()

  usable_frames = min(get_video_frame_count(video_path), gt.poses.shape[0])
  frame_indices = select_frame_indices(
      usable_frames,
      frame_stride=args.frame_stride,
      max_frames=args.max_frames,
  )
  if not frame_indices:
    raise ValueError("No frames selected for evaluation")

  print(f"Scene: {scene_name}")
  print(f"Episode video: {video_path}")
  print(f"Episode HDF5: {hdf5_path}")
  print(f"Selected frames: {len(frame_indices)} / {usable_frames}")

  if not args.skip_frame_extract:
    extract_video_frames(video_path, rgb_dir, frame_indices, force=args.force)
  else:
    print("Frame extraction skipped by --skip_frame_extract")
  write_calibration(calibration_path, gt.intrinsic)

  if not args.skip_mono_depth:
    run_depth_precomputation(
        repo_root=repo_root,
        rgb_dir=rgb_dir,
        scene_name=scene_name,
        mono_depth_parent=mono_depth_parent,
        metric_depth_parent=metric_depth_parent,
        depth_anything_checkpoint=(
            args.depth_anything_checkpoint
            if args.depth_anything_checkpoint.is_absolute()
            else repo_root / args.depth_anything_checkpoint
        ),
        frame_count=len(frame_indices),
        force=args.force,
        dry_run=args.dry_run,
    )
  else:
    print("Depth precomputation skipped by --skip_mono_depth")

  prediction_path = args.prediction_path
  if not args.skip_tracking and prediction_path is None:
    prediction_path = run_camera_tracking(
        repo_root=repo_root,
        dataset_root=dataset_root,
        scene_name=scene_name,
        weights=args.weights if args.weights.is_absolute() else repo_root / args.weights,
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

  pred_poses = load_prediction_poses(prediction_path)
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
  gt_eval = gt.poses[eval_frame_indices]
  pred_eval = pred_poses[:num_eval]
  result = evaluate_trajectory(gt_eval, pred_eval)

  output_dir = result_root / scene_name
  metrics_json, metrics_csv = write_metrics(output_dir, result.metrics)
  traj_path = write_trajectories(output_dir, eval_frame_indices, result)
  plot_path = plot_trajectories(
      output_dir / "trajectory.png",
      result.gt_cam_c2w,
      result.pred_aligned_cam_c2w,
      eval_frame_indices,
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
