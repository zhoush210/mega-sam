#!/usr/bin/env python3
"""Visualize MegaSaM demo RGB-D trajectory results.

The demo pipeline writes files such as:
  outputs/swing_droid.npz
  outputs_cvd/swing_sgd_cvd_hr.npz

Each file contains RGB images, per-frame depth, intrinsics, and camera-to-world
poses. This script back-projects the RGB-D frames into a sparse colored point
cloud and draws it together with the camera trajectory.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DemoResult:
  images: np.ndarray
  depths: np.ndarray
  intrinsic: np.ndarray
  cam_c2w: np.ndarray


def load_result(path):
  """Load and validate a MegaSaM demo result .npz file."""
  path = Path(path)
  required_keys = ("images", "depths", "intrinsic", "cam_c2w")

  with np.load(path) as data:
    missing = [key for key in required_keys if key not in data.files]
    if missing:
      raise ValueError(
          "{} is missing required keys: {}".format(path, ", ".join(missing))
      )

    images = np.array(data["images"])
    depths = np.array(data["depths"])
    intrinsic = np.array(data["intrinsic"], dtype=np.float32)
    cam_c2w = np.array(data["cam_c2w"], dtype=np.float32)

  if images.ndim != 4 or images.shape[-1] != 3:
    raise ValueError("images must have shape (N, H, W, 3)")
  if depths.ndim != 3:
    raise ValueError("depths must have shape (N, H, W)")
  if images.shape[:3] != depths.shape:
    raise ValueError(
        "images and depths must agree on (N, H, W); got {} and {}".format(
            images.shape, depths.shape
        )
    )
  if intrinsic.shape != (3, 3):
    raise ValueError("intrinsic must have shape (3, 3)")
  if cam_c2w.ndim != 3 or cam_c2w.shape[1:] != (4, 4):
    raise ValueError("cam_c2w must have shape (N, 4, 4)")
  if cam_c2w.shape[0] != images.shape[0]:
    raise ValueError(
        "cam_c2w and images must have the same frame count; got {} and {}".format(
            cam_c2w.shape[0], images.shape[0]
        )
    )

  return DemoResult(
      images=images,
      depths=depths,
      intrinsic=intrinsic,
      cam_c2w=cam_c2w,
  )


def select_frame_indices(num_frames, frame_stride=1, max_frames=None):
  """Return deterministic frame indices for visualization."""
  if num_frames <= 0:
    return []
  if frame_stride < 1:
    raise ValueError("frame_stride must be >= 1")

  indices = np.arange(0, num_frames, frame_stride, dtype=np.int64)

  if max_frames is not None:
    if max_frames < 1:
      raise ValueError("max_frames must be >= 1")
    if len(indices) > max_frames:
      keep = np.linspace(0, len(indices) - 1, max_frames).astype(np.int64)
      indices = indices[keep]

  return indices.tolist()


def _normalize_colors(colors):
  colors = colors.astype(np.float32)
  if colors.size and colors.max() > 1.0:
    colors = colors / 255.0
  return np.clip(colors, 0.0, 1.0)


def backproject_frames(
    images,
    depths,
    intrinsic,
    cam_c2w,
    frame_indices,
    pixel_stride=8,
    min_depth=0.01,
    max_depth=100.0,
    max_points=None,
):
  """Back-project selected RGB-D frames into world-space points."""
  if pixel_stride < 1:
    raise ValueError("pixel_stride must be >= 1")
  if min_depth <= 0 or max_depth <= min_depth:
    raise ValueError("depth range must satisfy 0 < min_depth < max_depth")

  fx = float(intrinsic[0, 0])
  fy = float(intrinsic[1, 1])
  cx = float(intrinsic[0, 2])
  cy = float(intrinsic[1, 2])
  if fx == 0.0 or fy == 0.0:
    raise ValueError("intrinsic focal lengths must be non-zero")

  height, width = depths.shape[1:]
  ys = np.arange(0, height, pixel_stride, dtype=np.float32)
  xs = np.arange(0, width, pixel_stride, dtype=np.float32)
  grid_x, grid_y = np.meshgrid(xs, ys)

  all_points = []
  all_colors = []

  for frame_index in frame_indices:
    depth = depths[frame_index, ::pixel_stride, ::pixel_stride].astype(
        np.float32
    )
    image = images[frame_index, ::pixel_stride, ::pixel_stride]

    valid = (
        np.isfinite(depth)
        & (depth >= min_depth)
        & (depth <= max_depth)
    )
    if not np.any(valid):
      continue

    z = depth[valid]
    x = (grid_x[valid] - cx) * z / fx
    y = (grid_y[valid] - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1)

    pose = cam_c2w[frame_index]
    points_world = points_cam @ pose[:3, :3].T + pose[:3, 3]

    all_points.append(points_world.astype(np.float32))
    all_colors.append(_normalize_colors(image[valid]))

  if not all_points:
    return (
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    )

  points = np.concatenate(all_points, axis=0)
  colors = np.concatenate(all_colors, axis=0)

  if max_points is not None and points.shape[0] > max_points:
    if max_points < 1:
      raise ValueError("max_points must be >= 1")
    keep = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    points = points[keep]
    colors = colors[keep]

  return points.astype(np.float32), colors.astype(np.float32)


def make_trajectory_lines(cam_c2w, frame_indices):
  """Create points and line indices for camera-center trajectory."""
  centers = cam_c2w[frame_indices, :3, 3].astype(np.float32)
  if len(frame_indices) < 2:
    lines = np.empty((0, 2), dtype=np.int32)
  else:
    lines = np.stack(
        [
            np.arange(0, len(frame_indices) - 1),
            np.arange(1, len(frame_indices)),
        ],
        axis=1,
    ).astype(np.int32)
  return centers, lines


def make_camera_frustum_lines(
    intrinsic,
    cam_c2w,
    frame_indices,
    image_shape,
    scale=0.1,
):
  """Create a merged camera-frustum LineSet as points and line indices."""
  height, width = image_shape
  fx = float(intrinsic[0, 0])
  fy = float(intrinsic[1, 1])
  cx = float(intrinsic[0, 2])
  cy = float(intrinsic[1, 2])
  if fx == 0.0 or fy == 0.0:
    raise ValueError("intrinsic focal lengths must be non-zero")

  corners_px = np.array(
      [
          [0.0, 0.0],
          [width - 1.0, 0.0],
          [width - 1.0, height - 1.0],
          [0.0, height - 1.0],
      ],
      dtype=np.float32,
  )
  corners_cam = np.column_stack(
      [
          (corners_px[:, 0] - cx) * scale / fx,
          (corners_px[:, 1] - cy) * scale / fy,
          np.full(4, scale, dtype=np.float32),
      ]
  )
  local_points = np.vstack(
      [np.zeros((1, 3), dtype=np.float32), corners_cam]
  )
  local_lines = np.array(
      [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]],
      dtype=np.int32,
  )

  all_points = []
  all_lines = []
  for frame_index in frame_indices:
    pose = cam_c2w[frame_index]
    offset = len(all_points) * local_points.shape[0]
    all_points.append(local_points @ pose[:3, :3].T + pose[:3, 3])
    all_lines.append(local_lines + offset)

  if not all_points:
    return (
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 2), dtype=np.int32),
    )

  return (
      np.concatenate(all_points, axis=0).astype(np.float32),
      np.concatenate(all_lines, axis=0).astype(np.int32),
  )


def write_ply(path, points, colors):
  """Write an ASCII PLY point cloud without requiring Open3D."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  colors_u8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)

  with path.open("w", encoding="utf-8") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex {}\n".format(points.shape[0]))
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for point, color in zip(points, colors_u8):
      f.write(
          "{:.6f} {:.6f} {:.6f} {} {} {}\n".format(
              point[0], point[1], point[2], color[0], color[1], color[2]
          )
      )


def _line_set(o3d, points, lines, color):
  line_set = o3d.geometry.LineSet()
  line_set.points = o3d.utility.Vector3dVector(points)
  line_set.lines = o3d.utility.Vector2iVector(lines)
  line_set.colors = o3d.utility.Vector3dVector(
      np.repeat(np.asarray(color, dtype=np.float32)[None], len(lines), axis=0)
  )
  return line_set


def draw_open3d(
    points,
    colors,
    trajectory_points,
    trajectory_lines,
    frustum_points,
    frustum_lines,
    point_size=2.0,
):
  """Open an Open3D window for the reconstructed point cloud and cameras."""
  try:
    import open3d as o3d
  except ImportError as exc:
    raise SystemExit(
        "Open3D is not installed in this environment. Install it, or run with "
        "--save-ply <path> --no-window to export a point cloud."
    ) from exc

  geometries = []
  if points.shape[0]:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(point_cloud)

  if trajectory_lines.shape[0]:
    geometries.append(
        _line_set(o3d, trajectory_points, trajectory_lines, (0.1, 0.8, 0.2))
    )

  if frustum_lines.shape[0]:
    geometries.append(
        _line_set(o3d, frustum_points, frustum_lines, (1.0, 0.15, 0.05))
    )

  geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

  visualizer = o3d.visualization.Visualizer()
  visualizer.create_window(window_name="MegaSaM Demo Result", width=1280, height=800)
  for geometry in geometries:
    visualizer.add_geometry(geometry)
  render_option = visualizer.get_render_option()
  render_option.point_size = float(point_size)
  render_option.background_color = np.asarray([0.02, 0.02, 0.02])
  visualizer.run()
  visualizer.destroy_window()


def parse_args():
  parser = argparse.ArgumentParser(
      description="Visualize a MegaSaM demo .npz result as point cloud + trajectory."
  )
  parser.add_argument("result_npz", help="Path to *_droid.npz or *_sgd_cvd_hr.npz")
  parser.add_argument("--frame-stride", type=int, default=1)
  parser.add_argument("--max-frames", type=int, default=None)
  parser.add_argument("--pixel-stride", type=int, default=8)
  parser.add_argument("--max-points", type=int, default=500000)
  parser.add_argument("--min-depth", type=float, default=0.01)
  parser.add_argument("--max-depth", type=float, default=100.0)
  parser.add_argument("--camera-every", type=int, default=10)
  parser.add_argument("--camera-scale", type=float, default=0.15)
  parser.add_argument("--point-size", type=float, default=2.0)
  parser.add_argument("--save-ply", default=None, help="Optional point-cloud export path")
  parser.add_argument(
      "--no-window",
      action="store_true",
      help="Do not open Open3D; useful with --save-ply on headless machines.",
  )
  parser.add_argument(
      "--trajectory-only",
      action="store_true",
      help="Draw only camera trajectory/frustums, without RGB-D points.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  result = load_result(args.result_npz)
  frame_indices = select_frame_indices(
      result.images.shape[0], args.frame_stride, args.max_frames
  )
  if not frame_indices:
    raise SystemExit("No frames selected for visualization.")

  print("Loaded {}".format(args.result_npz))
  print(
      "frames={} size={}x{} selected={}".format(
          result.images.shape[0],
          result.images.shape[2],
          result.images.shape[1],
          len(frame_indices),
      )
  )

  if args.trajectory_only:
    points = np.empty((0, 3), dtype=np.float32)
    colors = np.empty((0, 3), dtype=np.float32)
  else:
    points, colors = backproject_frames(
        result.images,
        result.depths,
        result.intrinsic,
        result.cam_c2w,
        frame_indices,
        pixel_stride=args.pixel_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_points=args.max_points,
    )
    print("points={}".format(points.shape[0]))

  trajectory_points, trajectory_lines = make_trajectory_lines(
      result.cam_c2w, frame_indices
  )

  if args.camera_every < 1:
    raise SystemExit("--camera-every must be >= 1")
  frustum_indices = frame_indices[::args.camera_every]
  if frustum_indices[-1] != frame_indices[-1]:
    frustum_indices.append(frame_indices[-1])
  frustum_points, frustum_lines = make_camera_frustum_lines(
      result.intrinsic,
      result.cam_c2w,
      frustum_indices,
      result.images.shape[1:3],
      scale=args.camera_scale,
  )

  if args.save_ply:
    if points.shape[0] == 0:
      print("No points to export; skipping PLY.")
    else:
      write_ply(args.save_ply, points, colors)
      print("Wrote {}".format(args.save_ply))

  if args.no_window:
    return

  draw_open3d(
      points,
      colors,
      trajectory_points,
      trajectory_lines,
      frustum_points,
      frustum_lines,
      point_size=args.point_size,
  )


if __name__ == "__main__":
  main()
