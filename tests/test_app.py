"""Viewer integration against real Parquet, Gaussian PLY, MP4 and local Viser.

The fixture deliberately puts two episodes in each camera MP4 and gives the
second scene a distinct coordinate frame and colors. It catches decoder reuse,
wrong episode offsets, stale mosaics, swapped views, and accidental recentering.
No browser, network download, or private dataset is used.
"""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from plyfile import PlyData, PlyElement

from kiwi_viser.app import Viewer, trajectory_segments


CAMERAS = ("left_front", "right_front", "left_rear", "right_rear")
FPS = 4
LENGTH = 6


def create_delivery(root: Path) -> dict[tuple[str, str], list[np.ndarray]]:
    expected_images = {}
    for scene_number, name in enumerate(("world_alpha", "world_beta")):
        scene = root / "lerobot" / name
        (scene / "meta/episodes/chunk-000").mkdir(parents=True)
        (scene / "data/chunk-000").mkdir(parents=True)
        features = {
            "observation.state": {"dtype": "float32", "shape": [16]},
            "action": {"dtype": "float32", "shape": [16]},
            "observation.camera_pose.left": {"dtype": "float32", "shape": [7]},
            "observation.camera_pose.right": {"dtype": "float32", "shape": [7]},
            **{f"observation.images.{key}": {"dtype": "video", "shape": [32, 32, 3]} for key in CAMERAS},
        }
        info = {
            "codebase_version": "v3.0", "fps": FPS, "total_episodes": 2, "total_frames": 2 * LENGTH,
            "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
        (scene / "meta/info.json").write_text(json.dumps(info))
        metadata, records = [], []
        for episode in range(2):
            meta = {"episode_index": episode, "tasks": [f"{name} task {episode}"], "length": LENGTH,
                    "dataset_from_index": episode * LENGTH, "dataset_to_index": (episode + 1) * LENGTH,
                    "data/chunk_index": 0, "data/file_index": 0}
            for camera in CAMERAS:
                prefix = f"videos/observation.images.{camera}/"
                meta.update({prefix + "chunk_index": 0, prefix + "file_index": 0,
                             prefix + "from_timestamp": episode * LENGTH / FPS,
                             prefix + "to_timestamp": (episode + 1) * LENGTH / FPS})
            metadata.append(meta)
            def state_at(i):
                base = scene_number * 10 + episode
                half = math.sqrt(0.5)
                return [base + i / 10, 0.2, 0.3, 0, 0, half, half, i / (LENGTH - 1),
                        base - i / 10, 0.5, 0.6, half, 0, 0, half, 1 - i / (LENGTH - 1)]
            for i in range(LENGTH):
                state = state_at(i)
                records.append({"episode_index": episode, "frame_index": i, "index": episode * LENGTH + i,
                                "timestamp": i / FPS, "observation.state": state,
                                "action": state_at(min(i + 1, LENGTH - 1)),
                                "observation.camera_pose.left": [state[0], 0.7, 0.8, 0, 0, 0, 1],
                                "observation.camera_pose.right": [state[8], 0.9, 1.0, 0, 0, 0, 1],
                                "observation.gripper_filled.left": i == 2,
                                "observation.gripper_filled.right": False})
        pq.write_table(pa.Table.from_pylist(metadata), scene / "meta/episodes/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist(records), scene / "data/chunk-000/file-000.parquet")
        fields = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
        points = np.zeros(4, dtype=[(field, "<f4") for field in fields])
        points["x"] = np.array([0, 1, 0, 1]) + scene_number * 10
        points["y"] = [0, 0, 1, 1]
        points["rot_0"] = 1
        for axis in range(3):
            points[f"scale_{axis}"] = -3
        PlyData([PlyElement.describe(points, "vertex")], text=False).write(scene / "environment.ply")
        provenance = {
            "scene": {"scene_id": name, "world_frame": f"{name}_frame", "up_axis": "+Z"},
            "environment": {"files": [{"path": "environment.ply", "format": "PLY", "T_world_asset": np.eye(4).tolist()}]},
            "episodes": [{"episode_index": i, "episode_id": f"demo_{i}", "task": f"{name} task {i}"} for i in range(2)],
        }
        (scene / "meta/export_provenance.json").write_text(json.dumps(provenance))
        for camera_index, camera in enumerate(CAMERAS):
            path = scene / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
            path.parent.mkdir(parents=True)
            with av.open(str(path), "w") as output:
                stream = output.add_stream("libx264", rate=FPS)
                stream.width = stream.height = 32
                stream.pix_fmt = "yuv420p"
                stream.codec_context.gop_size = 4
                for i in range(2 * LENGTH):
                    # A whole-frame color uniquely identifies scene, camera and
                    # time, and remains unambiguous after lossy video encoding.
                    color = [20 + scene_number * 90, 30 + camera_index * 55, 10 + i * 20]
                    pixels = np.full((32, 32, 3), color, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    frame.pts = i
                    frame.time_base = Fraction(1, FPS)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            with av.open(str(path)) as source:
                expected_images[name, camera] = [frame.to_ndarray(format="rgb24") for frame in source.decode(video=0)]
    return expected_images


class TrajectoryTests(unittest.TestCase):
    def test_window_uses_timestamps_and_keeps_exact_edges(self) -> None:
        timestamps = np.array([0.0, 0.1, 0.4, 1.0, 1.2, 1.5, 2.0])
        positions = np.column_stack((np.arange(len(timestamps)), np.zeros((len(timestamps), 2))))
        segments = trajectory_segments(positions, timestamps, 3, half_window_seconds=0.6)
        np.testing.assert_array_equal(segments[:, 0, 0], [2, 3, 4])
        np.testing.assert_array_equal(segments[:, 1, 0], [3, 4, 5])
        first = trajectory_segments(positions, timestamps, 0, half_window_seconds=0.4)
        np.testing.assert_array_equal(first[:, 0, 0], [0, 1])
        np.testing.assert_array_equal(first[-1, 1], positions[2])
        last = trajectory_segments(positions, timestamps, 6, half_window_seconds=0.5)
        np.testing.assert_array_equal(last, [[positions[5], positions[6]]])
        self.assertEqual(trajectory_segments(positions, timestamps, 3, half_window_seconds=0).shape, (0, 2, 3))

    def test_full_path_decimation_preserves_endpoints_without_crossing_invalid_data(self) -> None:
        timestamps = np.arange(20) / 30
        positions = np.column_stack((np.arange(20), np.zeros((20, 2))))
        full = trajectory_segments(positions, timestamps, 10, max_points=5)
        self.assertEqual(full.shape, (4, 2, 3))
        np.testing.assert_array_equal(full[0, 0], positions[0])
        np.testing.assert_array_equal(full[-1, 1], positions[-1])
        positions[6, 0] = np.nan
        positions[14, 2] = np.inf
        broken = trajectory_segments(positions, timestamps, 10, max_points=5)
        self.assertTrue(np.isfinite(broken).all())
        self.assertEqual(broken.shape, (1, 2, 3))
        np.testing.assert_array_equal(broken[0], [positions[0], positions[4]])

    def test_decimated_nearby_path_retains_current_frame_and_validates_time(self) -> None:
        timestamps = np.arange(20, dtype=float)
        positions = np.column_stack((timestamps, np.zeros((20, 2))))
        segments = trajectory_segments(positions, timestamps, 8, half_window_seconds=20, max_points=4)
        self.assertIn(8, segments.reshape(-1, 3)[:, 0])
        np.testing.assert_array_equal(segments[0, 0], positions[0])
        np.testing.assert_array_equal(segments[-1, 1], positions[-1])
        with self.assertRaises(ValueError):
            trajectory_segments(positions, timestamps, 8, half_window_seconds=-1)
        with self.assertRaises(ValueError):
            trajectory_segments(positions, timestamps[::-1], 8)
        with self.assertRaises(IndexError):
            trajectory_segments(positions, timestamps, 20)


class ViewerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.expected_images = create_delivery(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def make_viewer(self, **kwargs) -> Viewer:
        # Viser and its actual GUI/scene handle serialization run on loopback;
        # port zero avoids colliding with a user's running viewer.
        viewer = Viewer(self.root, host="127.0.0.1", port=0, max_gaussians=4, **kwargs)
        self.addCleanup(viewer.server.stop)
        self.addCleanup(lambda: viewer.videos.close())
        return viewer

    def assert_frame_visible(self, viewer: Viewer, scene: str, episode: int, index: int) -> None:
        self.assertEqual(viewer.scene.name, scene)
        self.assertEqual(viewer.samples.episode.index, episode)
        self.assertEqual(viewer.frame_index, index)
        self.assertEqual(viewer.timeline.value, index)
        self.assertIn(f"{index / FPS:.2f} / {(LENGTH - 1) / FPS:.2f} s", viewer.status.content)
        self.assertIn(f"{scene} task {episode}", viewer.status.content)
        self.assertNotIn("Unable", viewer.status.content)
        image = viewer.mosaic.image
        self.assertEqual(image.shape, (750, 640, 3))
        for quadrant, camera in enumerate(CAMERAS):
            # Center pixels are image content, away from labels/opening bars.
            x = (quadrant % 2) * 320 + 160
            y = (quadrant // 2) * 348 + 28 + 160
            expected = self.expected_images[scene, camera][episode * LENGTH + index][16, 16]
            np.testing.assert_array_equal(image[y, x], expected)
        base = (10 if scene == "world_beta" else 0) + episode
        np.testing.assert_allclose(viewer.pose_handles["tcp_left"].position, [base + index / 10, 0.2, 0.3])
        np.testing.assert_allclose(viewer.pose_handles["tcp_right"].position, [base - index / 10, 0.5, 0.6])
        np.testing.assert_allclose(viewer.pose_handles["tcp_left"].wxyz, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
        np.testing.assert_allclose(viewer.pose_handles["camera_right"].position, [base - index / 10, 0.9, 1.0])
        np.testing.assert_allclose(viewer.pose_handles["action_left"].position, [base + min(index + 1, LENGTH - 1) / 10, 0.2, 0.3])

    def test_initial_episode_shared_video_switch_scene_and_seek(self) -> None:
        viewer = self.make_viewer()
        self.assert_frame_visible(viewer, "world_alpha", 0, 0)
        viewer._command("frame", LENGTH - 1)
        self.assert_frame_visible(viewer, "world_alpha", 0, LENGTH - 1)
        viewer._command("episode", viewer.scene.episodes[1].label)
        self.assert_frame_visible(viewer, "world_alpha", 1, 0)
        viewer._command("frame", 3)
        self.assert_frame_visible(viewer, "world_alpha", 1, 3)
        viewer._command("scene", "world_beta")
        self.assert_frame_visible(viewer, "world_beta", 0, 0)
        self.assertIn("world_beta_frame", viewer.scene_info.content)
        viewer._command("episode", viewer.scene.episodes[1].label)
        viewer._command("frame", 10_000)
        self.assert_frame_visible(viewer, "world_beta", 1, LENGTH - 1)
        viewer._command("frame", -10)
        self.assert_frame_visible(viewer, "world_beta", 1, 0)
        # Return to previously closed files and decode again successfully.
        viewer._command("scene", "world_alpha")
        self.assert_frame_visible(viewer, "world_alpha", 0, 0)

    def test_nearby_trajectory_follows_seek_playback_and_episode_boundaries(self) -> None:
        viewer = self.make_viewer()
        self.assertEqual(viewer.trajectory_range.value, "Nearby")
        self.assertEqual(viewer.trajectory_window.value, 2.0)
        viewer.trajectory_window.value = 0.5
        viewer._command("trajectory", None)
        original_handles = {side: viewer.pose_handles[f"trajectory_{side}"] for side in ("left", "right")}

        def assert_range(start, end):
            for side, offset in (("left", 0), ("right", 8)):
                points = viewer.samples.state[start:end + 1, offset:offset + 3]
                expected = np.stack((points[:-1], points[1:]), axis=1)
                np.testing.assert_allclose(viewer.pose_handles[f"trajectory_{side}"].points, expected)

        assert_range(0, 2)
        viewer._command("frame", 3)
        assert_range(1, 5)
        viewer._command("frame", LENGTH - 1)
        assert_range(3, 5)
        # The playback loop calls the same drawing path without stopping play.
        viewer._command("frame", 1)
        viewer._command("play", True)
        viewer._draw_frame(2)
        self.assertTrue(viewer.playing)
        assert_range(0, 4)
        for side in ("left", "right"):
            self.assertIs(viewer.pose_handles[f"trajectory_{side}"], original_handles[side])
        viewer.show_trajectories.value = False
        viewer._command("visibility", None)
        viewer._command("episode", viewer.scene.episodes[1].label)
        assert_range(0, 2)
        self.assertEqual(viewer.trajectory_window.value, 0.5)
        for side in ("left", "right"):
            self.assertFalse(viewer.pose_handles[f"trajectory_{side}"].visible)
        viewer._command("scene", "world_beta")
        assert_range(0, 2)
        self.assertEqual(viewer.frame_index, 0)

    def test_full_trajectory_mode_and_window_control_do_not_seek_or_decode(self) -> None:
        viewer = self.make_viewer()
        viewer._command("frame", 3)
        viewer._command("play", True)
        viewer.trajectory_range.value = "Full episode"
        # A trajectory-only edit should not decode the current video again.
        original_read = viewer.videos.read
        def unexpected_read(*args, **kwargs):
            self.fail("Trajectory controls decoded video")
        viewer.videos.read = unexpected_read
        viewer._command("trajectory", None)
        self.assertTrue(viewer.playing)
        self.assertEqual(viewer.frame_index, 3)
        self.assertTrue(viewer.trajectory_window.disabled)
        for side, offset in (("left", 0), ("right", 8)):
            points = viewer.samples.state[:, offset:offset + 3]
            np.testing.assert_allclose(viewer.pose_handles[f"trajectory_{side}"].points,
                                       np.stack((points[:-1], points[1:]), axis=1))
        viewer.videos.read = original_read
        viewer._command("episode", viewer.scene.episodes[1].label)
        self.assertEqual(viewer.trajectory_range.value, "Full episode")
        np.testing.assert_allclose(viewer.pose_handles["trajectory_left"].points[0, 0],
                                   viewer.samples.state[0, :3])
        np.testing.assert_allclose(viewer.pose_handles["trajectory_left"].points[-1, 1],
                                   viewer.samples.state[-1, :3])
        viewer._command("frame", 3)
        original_read = viewer.videos.read
        viewer.videos.read = unexpected_read
        viewer.trajectory_range.value = "Nearby"
        viewer.trajectory_window.value = 0.25
        viewer._command("trajectory", None)
        self.assertFalse(viewer.trajectory_window.disabled)
        self.assertEqual(viewer.pose_handles["trajectory_left"].points.shape[0], 2)
        viewer.videos.read = original_read
        viewer._command("frame", 0)
        self.assertEqual(viewer.pose_handles["trajectory_left"].points.shape[0], 1)

    def test_visibility_persists_while_switching_and_seek_pauses(self) -> None:
        viewer = self.make_viewer(scene_name="world_beta", episode_index=1, render="points")
        self.assert_frame_visible(viewer, "world_beta", 1, 0)
        viewer.show_environment.value = False
        viewer.show_trajectories.value = False
        viewer.show_cameras.value = True
        viewer.show_actions.value = True
        viewer._command("visibility", None)
        for scene in ("world_beta", "world_alpha"):
            if viewer.scene.name != scene:
                viewer._command("scene", scene)
            self.assertFalse(viewer.environment_handle.visible)
            for side in ("left", "right"):
                self.assertFalse(viewer.pose_handles[f"trajectory_{side}"].visible)
                self.assertTrue(viewer.pose_handles[f"camera_{side}"].visible)
                self.assertTrue(viewer.pose_handles[f"action_{side}"].visible)
                self.assertTrue(viewer.pose_handles[f"tcp_{side}"].visible)
            viewer._command("play", True)
            self.assertTrue(viewer.playing)
            viewer._command("frame", 2)
            self.assertFalse(viewer.playing)
            self.assertFalse(viewer.play.value)
            self.assertAlmostEqual(viewer.playhead, 2 / FPS)
            self.assert_frame_visible(viewer, scene, viewer.samples.episode.index, 2)


if __name__ == "__main__":
    unittest.main()
