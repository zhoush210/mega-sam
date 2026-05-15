import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "visualize_demo_result.py"


def load_module():
  spec = importlib.util.spec_from_file_location(
      "visualize_demo_result", SCRIPT_PATH
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class VisualizeDemoResultTest(unittest.TestCase):

  def test_load_result_validates_required_keys_and_shapes(self):
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmpdir:
      result_path = Path(tmpdir) / "demo.npz"
      np.savez(
          result_path,
          images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
          depths=np.ones((1, 2, 2), dtype=np.float32),
          intrinsic=np.eye(3, dtype=np.float32),
          cam_c2w=np.eye(4, dtype=np.float32)[None],
      )

      result = mod.load_result(result_path)

    self.assertEqual(result.images.shape, (1, 2, 2, 3))
    self.assertEqual(result.depths.shape, (1, 2, 2))
    self.assertEqual(result.intrinsic.shape, (3, 3))
    self.assertEqual(result.cam_c2w.shape, (1, 4, 4))

  def test_backproject_frames_returns_world_points_and_unit_colors(self):
    mod = load_module()
    images = np.array(
        [[
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ]],
        dtype=np.uint8,
    )
    depths = np.ones((1, 2, 2), dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    cam_c2w = np.eye(4, dtype=np.float32)[None]

    points, colors = mod.backproject_frames(
        images,
        depths,
        intrinsic,
        cam_c2w,
        frame_indices=[0],
        pixel_stride=1,
        min_depth=0.01,
        max_depth=10.0,
    )

    np.testing.assert_allclose(
        points,
        np.array(
            [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        colors,
        np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
            dtype=np.float32,
        ),
    )

  def test_make_trajectory_lines_connects_camera_centers(self):
    mod = load_module()
    cam_c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    cam_c2w[:, 0, 3] = [0.0, 1.0, 3.0]

    points, lines = mod.make_trajectory_lines(cam_c2w, [0, 1, 2])

    np.testing.assert_allclose(
        points,
        np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(lines, np.array([[0, 1], [1, 2]]))


if __name__ == "__main__":
  unittest.main()
