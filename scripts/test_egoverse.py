#!/usr/bin/env python3
"""Run MegaSaM on one egoVerse episode and evaluate camera trajectory."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts import test_egodex as egodex_utils  # pylint: disable=g-import-not-at-top


DEFAULT_IMAGE_KEY = "images.front_1"
DEFAULT_POSE_KEY = "obs_head_pose"
MISSING_SHARD_ENTRY = 2**64 - 1


@dataclass
class EgoVersePaths:
  dataset_path: Path
  image_array_path: Path
  pose_array_path: Path


@dataclass
class EgoVerseGroundTruth:
  poses: np.ndarray
  intrinsic: np.ndarray
  attrs: dict


def read_json(path: Path) -> dict:
  return json.loads(Path(path).read_text(encoding="utf-8"))


def root_attrs(dataset_path: Path) -> dict:
  metadata_path = Path(dataset_path) / "zarr.json"
  if not metadata_path.exists():
    raise FileNotFoundError(f"Missing egoVerse zarr metadata: {metadata_path}")
  return read_json(metadata_path).get("attributes", {})


def array_metadata(array_path: Path) -> dict:
  metadata_path = Path(array_path) / "zarr.json"
  if not metadata_path.exists():
    raise FileNotFoundError(f"Missing egoVerse array metadata: {metadata_path}")
  metadata = read_json(metadata_path)
  if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
    raise ValueError(f"Unsupported zarr array metadata: {metadata_path}")
  return metadata


def resolve_dataset_paths(
    dataset_path: Path | str,
    image_key: str = DEFAULT_IMAGE_KEY,
    pose_key: str = DEFAULT_POSE_KEY,
) -> EgoVersePaths:
  """Resolve the egoVerse zarr group and arrays needed by MegaSaM."""
  dataset_path = Path(dataset_path)
  if not (dataset_path / "zarr.json").exists():
    raise FileNotFoundError(f"Missing egoVerse zarr group: {dataset_path}")

  image_array_path = dataset_path / image_key
  if not (image_array_path / "zarr.json").exists():
    raise FileNotFoundError(f"Missing egoVerse image array: {image_array_path}")

  pose_array_path = dataset_path / pose_key
  if not (pose_array_path / "zarr.json").exists():
    raise FileNotFoundError(f"Missing egoVerse pose array: {pose_array_path}")

  return EgoVersePaths(
      dataset_path=dataset_path,
      image_array_path=image_array_path,
      pose_array_path=pose_array_path,
  )


def _dtype_from_zarr(data_type: str) -> np.dtype:
  if data_type == "float64":
    return np.dtype("<f8")
  if data_type == "float32":
    return np.dtype("<f4")
  if data_type in {"int64", "uint64", "int32", "uint32", "uint8"}:
    return np.dtype("<" + np.dtype(data_type).str[1:])
  raise ValueError(f"Unsupported egoVerse zarr data_type: {data_type}")


def _sharding_config(metadata: dict) -> dict:
  codecs = metadata.get("codecs") or []
  if len(codecs) != 1 or codecs[0].get("name") != "sharding_indexed":
    raise ValueError("Only zarr v3 sharding_indexed arrays are supported.")
  return codecs[0]["configuration"]


def _chunk_key(array_path: Path, outer_coords: tuple[int, ...], separator: str) -> Path:
  coord_parts = [str(int(coord)) for coord in outer_coords]
  if separator == "/":
    return Path(array_path, "c", *coord_parts)
  return Path(array_path, "c", separator.join(coord_parts))


def _inner_chunk_counts(
    outer_chunk_shape: tuple[int, ...], inner_chunk_shape: tuple[int, ...]
) -> tuple[int, ...]:
  return tuple(
      int(math.ceil(float(outer) / float(inner)))
      for outer, inner in zip(outer_chunk_shape, inner_chunk_shape)
  )


def _index_trailer_size(sharding_conf: dict, entry_count: int) -> int:
  index_size = int(entry_count) * 16
  for codec in sharding_conf.get("index_codecs", []):
    if codec.get("name") == "crc32c":
      index_size += 4
  return index_size


def _zstd_decompress(payload: bytes) -> bytes:
  try:
    import zstandard as zstd  # pylint: disable=import-outside-toplevel
  except ImportError:
    zstd_command = shutil.which("zstd")
    if zstd_command is None:
      raise ImportError(
          "Decoding egoVerse zarr chunks requires either the Python "
          "'zstandard' package or the 'zstd' command in PATH."
      )
    result = subprocess.run(
        [zstd_command, "-d", "-q", "-c"],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
      raise RuntimeError(
          "zstd failed while decoding an egoVerse zarr chunk: "
          f"{result.stderr.decode('utf-8', errors='replace')}"
      )
    return result.stdout
  return zstd.ZstdDecompressor().decompress(payload)


def _decode_payload(encoded: bytes, codecs: Sequence[dict]) -> bytes:
  payload = encoded
  for codec in reversed(codecs):
    name = codec.get("name")
    if name == "zstd":
      payload = _zstd_decompress(payload)
    elif name in {"bytes", "vlen-bytes"}:
      continue
    else:
      raise ValueError(f"Unsupported egoVerse zarr codec: {name}")
  return payload


def _decode_vlen_bytes(raw: bytes) -> list[bytes]:
  if len(raw) < 4:
    return []
  count = struct.unpack_from("<I", raw, 0)[0]
  offset = 4
  lengths = []
  for _ in range(count):
    if offset + 4 > len(raw):
      raise ValueError("Malformed vlen-bytes chunk in egoVerse zarr array.")
    lengths.append(struct.unpack_from("<I", raw, offset)[0])
    offset += 4

  values = []
  for length in lengths:
    end = offset + int(length)
    if end > len(raw):
      raise ValueError("Malformed vlen-bytes payload in egoVerse zarr array.")
    values.append(raw[offset:end])
    offset = end
  return values


def _entry_coordinates(entry_idx: int, inner_counts: tuple[int, ...]) -> tuple[int, ...]:
  coords = []
  remaining = int(entry_idx)
  for count in reversed(inner_counts):
    coords.append(remaining % count)
    remaining //= count
  return tuple(reversed(coords))


def _iter_outer_coords(shape: tuple[int, ...], outer_chunk_shape: tuple[int, ...]):
  counts = [
      int(math.ceil(float(dim) / float(chunk)))
      for dim, chunk in zip(shape, outer_chunk_shape)
  ]
  return np.ndindex(*counts)


def _reshape_numeric_chunk(
    raw: bytes, dtype: np.dtype, inner_chunk_shape: tuple[int, ...]
) -> np.ndarray:
  values = np.frombuffer(raw, dtype=dtype)
  expected = int(math.prod(inner_chunk_shape))
  if values.size == expected:
    return values.reshape(inner_chunk_shape)
  if len(inner_chunk_shape) == 1:
    return values.reshape(-1)

  trailing = int(math.prod(inner_chunk_shape[1:]))
  if trailing <= 0 or values.size % trailing != 0:
    raise ValueError(
        f"Cannot reshape egoVerse zarr chunk with {values.size} values "
        f"to inner shape {inner_chunk_shape}."
    )
  return values.reshape((values.size // trailing, *inner_chunk_shape[1:]))


def _fill_numeric_chunk(
    output: np.ndarray,
    chunk_values: np.ndarray,
    start: tuple[int, ...],
    inner_chunk_shape: tuple[int, ...],
) -> None:
  slices = []
  source_slices = []
  for dim_idx, start_idx in enumerate(start):
    dim_end = min(start_idx + inner_chunk_shape[dim_idx], output.shape[dim_idx])
    if dim_end <= start_idx:
      return
    slices.append(slice(start_idx, dim_end))
    source_slices.append(slice(0, dim_end - start_idx))
  output[tuple(slices)] = chunk_values[tuple(source_slices)]


def load_sharded_zarr_array(array_path: Path, limit: int | None = None):
  """Load a zarr v3 sharding_indexed array used by egoVerse."""
  array_path = Path(array_path)
  metadata = array_metadata(array_path)
  shape = tuple(int(value) for value in metadata["shape"])
  if limit is not None and shape:
    shape = (min(shape[0], int(limit)), *shape[1:])

  sharding_conf = _sharding_config(metadata)
  outer_chunk_shape = tuple(
      int(value)
      for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
  )
  inner_chunk_shape = tuple(int(value) for value in sharding_conf["chunk_shape"])
  separator = (
      metadata.get("chunk_key_encoding", {})
      .get("configuration", {})
      .get("separator", "/")
  )
  inner_counts = _inner_chunk_counts(outer_chunk_shape, inner_chunk_shape)
  entry_count = int(math.prod(inner_counts))
  index_size = _index_trailer_size(sharding_conf, entry_count)
  codecs = sharding_conf.get("codecs", [])

  if metadata["data_type"] == "variable_length_bytes":
    values = [b""] * (shape[0] if shape else 0)
    is_vlen = True
    dtype = None
  else:
    dtype = _dtype_from_zarr(metadata["data_type"])
    values = np.full(shape, metadata.get("fill_value", 0), dtype=dtype)
    is_vlen = False

  for outer_coords in _iter_outer_coords(shape, outer_chunk_shape):
    shard_path = _chunk_key(array_path, outer_coords, separator=separator)
    if not shard_path.exists():
      continue
    shard = shard_path.read_bytes()
    if len(shard) < index_size:
      raise ValueError(f"Malformed sharded zarr file: {shard_path}")
    index = shard[-index_size:]
    index_codecs = sharding_conf.get("index_codecs", [])
    if index_codecs and index_codecs[-1].get("name") == "crc32c":
      index = index[:-4]

    for entry_idx in range(entry_count):
      offset, length = struct.unpack_from("<QQ", index, entry_idx * 16)
      if offset == MISSING_SHARD_ENTRY and length == MISSING_SHARD_ENTRY:
        continue

      inner_coords = _entry_coordinates(entry_idx, inner_counts)
      start = tuple(
          outer_coords[dim] * outer_chunk_shape[dim]
          + inner_coords[dim] * inner_chunk_shape[dim]
          for dim in range(len(shape))
      )
      if any(start[dim] >= shape[dim] for dim in range(len(shape))):
        continue

      raw = _decode_payload(shard[offset : offset + length], codecs)
      if is_vlen:
        chunk_values = _decode_vlen_bytes(raw)
        for value_idx, value in enumerate(chunk_values):
          row_idx = start[0] + value_idx
          if row_idx < len(values):
            values[row_idx] = value
      else:
        chunk_values = _reshape_numeric_chunk(raw, dtype, inner_chunk_shape)
        _fill_numeric_chunk(values, chunk_values, start, inner_chunk_shape)

  return values


def zarr_array_length(array_path: Path) -> int:
  metadata = array_metadata(array_path)
  shape = metadata.get("shape") or []
  if not shape:
    raise ValueError(f"zarr array has empty shape: {array_path}")
  return int(shape[0])


def _first_present(mapping: dict, *keys: str):
  for key in keys:
    if key in mapping and mapping[key] is not None:
      return mapping[key]
  return None


def image_dimensions_from_attrs(
    attrs: dict, image_key: str = DEFAULT_IMAGE_KEY
) -> tuple[int | None, int | None]:
  feature = attrs.get("features", {}).get(image_key, {})
  shape = feature.get("shape")
  if shape and len(shape) >= 2:
    height, width = shape[:2]
    return int(width), int(height)
  return None, None


def _intrinsic_matrix_from_mapping(
    value: dict, width: int | None, height: int | None
) -> np.ndarray | None:
  if not isinstance(value, dict):
    return None

  fx = _first_present(value, "fx", "fl_x", "focal_x", "focal_length_x")
  fy = _first_present(value, "fy", "fl_y", "focal_y", "focal_length_y")
  if fx is None:
    return None
  if fy is None:
    fy = fx

  src_width = _first_present(value, "w", "width", "image_width")
  src_height = _first_present(value, "h", "height", "image_height")
  scale_x = 1.0
  scale_y = 1.0
  if width is not None and src_width not in (None, 0):
    scale_x = float(width) / float(src_width)
  if height is not None and src_height not in (None, 0):
    scale_y = float(height) / float(src_height)

  cx = _first_present(value, "cx", "principal_x", "principal_point_x")
  cy = _first_present(value, "cy", "principal_y", "principal_point_y")
  cx = float(cx) * scale_x if cx is not None else (
      float(width) / 2.0 if width else 0.0
  )
  cy = float(cy) * scale_y if cy is not None else (
      float(height) / 2.0 if height else 0.0
  )

  return np.array(
      [
          [float(fx) * scale_x, 0.0, cx],
          [0.0, float(fy) * scale_y, cy],
          [0.0, 0.0, 1.0],
      ],
      dtype=np.float64,
  )


def fallback_intrinsic(
    width: int | None, height: int | None, focal: float | None = None
) -> np.ndarray | None:
  if width is None or height is None:
    return None
  focal = float(focal) if focal is not None else max(600.0, float(width))
  return np.array(
      [
          [focal, 0.0, float(width) / 2.0],
          [0.0, focal, float(height) / 2.0],
          [0.0, 0.0, 1.0],
      ],
      dtype=np.float64,
  )


def intrinsic_from_attrs(
    attrs: dict,
    image_key: str = DEFAULT_IMAGE_KEY,
    fallback_focal: float | None = None,
) -> np.ndarray:
  width, height = image_dimensions_from_attrs(attrs, image_key=image_key)
  intrinsics = attrs.get("intrinsics")

  intrinsic = None
  if isinstance(intrinsics, dict) and intrinsics:
    intrinsic = _intrinsic_matrix_from_mapping(
        intrinsics, width=width, height=height
    )
    if intrinsic is None:
      for value in intrinsics.values():
        intrinsic = _intrinsic_matrix_from_mapping(
            value, width=width, height=height
        )
        if intrinsic is not None:
          break

  if intrinsic is None:
    intrinsic = fallback_intrinsic(width, height, focal=fallback_focal)
  if intrinsic is None:
    raise ValueError("egoVerse metadata does not contain usable intrinsics.")
  return intrinsic


def quaternion_wxyz_to_matrix(quaternions_wxyz: np.ndarray) -> np.ndarray:
  quaternions_wxyz = np.asarray(quaternions_wxyz, dtype=np.float64)
  if quaternions_wxyz.ndim != 2 or quaternions_wxyz.shape[1] != 4:
    raise ValueError("quaternions_wxyz must have shape (N, 4)")

  q = quaternions_wxyz.copy()
  norms = np.linalg.norm(q, axis=1, keepdims=True)
  if np.any(norms <= 1e-12):
    raise ValueError("quaternions must be non-zero")
  q /= norms
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

  matrix = np.empty((q.shape[0], 3, 3), dtype=np.float64)
  matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
  matrix[:, 0, 1] = 2.0 * (x * y - z * w)
  matrix[:, 0, 2] = 2.0 * (x * z + y * w)
  matrix[:, 1, 0] = 2.0 * (x * y + z * w)
  matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
  matrix[:, 1, 2] = 2.0 * (y * z - x * w)
  matrix[:, 2, 0] = 2.0 * (x * z - y * w)
  matrix[:, 2, 1] = 2.0 * (y * z + x * w)
  matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
  return matrix


def pose7_wxyz_to_matrices(
    poses_7d: np.ndarray, pose_convention: str = "c2w"
) -> np.ndarray:
  """Convert [x y z qw qx qy qz] egoVerse poses to 4x4 matrices."""
  if pose_convention not in {"c2w", "w2c"}:
    raise ValueError("pose_convention must be 'c2w' or 'w2c'")
  poses_7d = np.asarray(poses_7d, dtype=np.float64)
  if poses_7d.ndim != 2 or poses_7d.shape[1] != 7:
    raise ValueError(f"Expected pose array with shape (N, 7), got {poses_7d.shape}")

  poses = np.repeat(np.eye(4, dtype=np.float64)[None], poses_7d.shape[0], axis=0)
  poses[:, :3, :3] = quaternion_wxyz_to_matrix(poses_7d[:, 3:7])
  poses[:, :3, 3] = poses_7d[:, :3]
  if pose_convention == "w2c":
    poses = np.linalg.inv(poses)
  poses[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
  return poses


def load_egoverse_ground_truth(
    paths: EgoVersePaths,
    image_key: str = DEFAULT_IMAGE_KEY,
    pose_convention: str = "c2w",
    fallback_focal: float | None = None,
) -> EgoVerseGroundTruth:
  """Load egoVerse head/camera pose GT and scaled camera intrinsics."""
  attrs = root_attrs(paths.dataset_path)
  pose_values = load_sharded_zarr_array(paths.pose_array_path)
  poses = pose7_wxyz_to_matrices(pose_values, pose_convention=pose_convention)
  if poses.shape[0] < 2:
    raise ValueError("egoVerse episode must contain at least two poses")
  intrinsic = intrinsic_from_attrs(
      attrs, image_key=image_key, fallback_focal=fallback_focal
  )
  return EgoVerseGroundTruth(poses=poses, intrinsic=intrinsic, attrs=attrs)


def _existing_png_count(path: Path) -> int:
  return len(list(path.glob("*.png")))


def extract_zarr_frames(
    image_array_path: Path,
    rgb_dir: Path,
    frame_indices: Sequence[int],
    force: bool = False,
) -> None:
  """Decode selected egoVerse JPEG zarr frames to MegaSaM PNG images."""
  if not frame_indices:
    raise ValueError("frame_indices must not be empty")
  rgb_dir.mkdir(parents=True, exist_ok=True)
  if not force and _existing_png_count(rgb_dir) == len(frame_indices):
    print(f"Frame extraction skipped; found {len(frame_indices)} PNGs in {rgb_dir}")
    return

  for path in rgb_dir.glob("*.png"):
    path.unlink()

  import cv2  # pylint: disable=import-outside-toplevel

  frame_bytes = load_sharded_zarr_array(
      image_array_path, limit=max(frame_indices) + 1
  )
  for output_index, source_index in enumerate(frame_indices):
    encoded = np.frombuffer(frame_bytes[source_index], dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
      raise ValueError(f"Failed to decode egoVerse JPEG frame {source_index}")
    out_path = rgb_dir / f"{output_index:05d}.png"
    if not cv2.imwrite(str(out_path), image):
      raise IOError(f"Failed to write frame: {out_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run MegaSaM on one egoVerse zarr episode and compare to GT."
  )
  parser.add_argument("--dataset_path", type=Path, required=True)
  parser.add_argument("--image_key", type=str, default=DEFAULT_IMAGE_KEY)
  parser.add_argument("--pose_key", type=str, default=DEFAULT_POSE_KEY)
  parser.add_argument("--scene_name", type=str, default=None)
  parser.add_argument("--work_dir", type=Path, default=Path("outputs_egoverse/work"))
  parser.add_argument(
      "--result_dir", type=Path, default=Path("outputs_egoverse/results")
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
  parser.add_argument("--pose_convention", choices=["c2w", "w2c"], default="c2w")
  parser.add_argument("--fallback_focal", type=float, default=None)
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
  paths = resolve_dataset_paths(
      dataset_path, image_key=args.image_key, pose_key=args.pose_key
  )
  gt = load_egoverse_ground_truth(
      paths,
      image_key=args.image_key,
      pose_convention=args.pose_convention,
      fallback_focal=args.fallback_focal,
  )

  scene_base = args.scene_name or f"egoverse_{dataset_path.name}"
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

  usable_frames = min(zarr_array_length(paths.image_array_path), gt.poses.shape[0])
  frame_indices = egodex_utils.select_frame_indices(
      usable_frames,
      frame_stride=args.frame_stride,
      max_frames=args.max_frames,
  )
  if not frame_indices:
    raise ValueError("No frames selected for evaluation")

  print(f"Scene: {scene_name}")
  print(f"egoVerse dataset: {paths.dataset_path}")
  print(f"Image array: {paths.image_array_path}")
  print(f"Pose array: {paths.pose_array_path}")
  print(f"Selected frames: {len(frame_indices)} / {usable_frames}")

  if not args.skip_frame_extract:
    extract_zarr_frames(
        paths.image_array_path, rgb_dir, frame_indices, force=args.force
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

  prediction_path = (
      _resolve_path(repo_root, args.prediction_path)
      if args.prediction_path is not None
      else None
  )
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
      title="egoVerse camera trajectory",
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
