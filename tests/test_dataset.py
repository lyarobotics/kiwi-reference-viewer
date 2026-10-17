"""Contract tests built entirely from small, independent delivery fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kiwi_viser.dataset import DatasetError, SceneDataset, discover_datasets


CAMERAS = ("left_front", "left_rear", "right_front", "right_rear")


def fixture_scene(root: Path, name: str = "scene_alpha") -> Path:
    scene = root / "lerobot" / name
    (scene / "meta/episodes/chunk-003").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [16]},
        "action": {"dtype": "float32", "shape": [16]},
        "observation.camera_pose.left": {"dtype": "float32", "shape": [7]},
        "observation.camera_pose.right": {"dtype": "float32", "shape": [7]},
        **{f"observation.images.{key}": {"dtype": "video", "shape": [16, 16, 3]} for key in CAMERAS},
    }
    info = {
        "codebase_version": "v3.0", "fps": 2.0, "total_episodes": 3, "total_frames": 9,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (scene / "meta/info.json").write_text(json.dumps(info))
    rows = []
    records = []
    provenance = {"scene": {"scene_id": name, "world_frame": f"{name}_world"}, "episodes": []}
    for episode in range(3):
        # The first two episodes share both data and video files. The last uses
        # another chunk and video file with a nonzero presentation-time origin.
        data_chunk, data_file = (5, 9) if episode < 2 else (6, 10)
        video_chunk, video_file = (3, 2) if episode < 2 else (4, 7)
        start = [0.0, 1.5, 5.0][episode]
        row = {"episode_index": episode, "tasks": [f"task {episode}"], "length": 3,
               "dataset_from_index": episode * 3, "dataset_to_index": episode * 3 + 3,
               "data/chunk_index": data_chunk, "data/file_index": data_file}
        for camera in CAMERAS:
            key = f"videos/observation.images.{camera}/"
            row.update({key + "chunk_index": video_chunk, key + "file_index": video_file,
                        key + "from_timestamp": start, key + "to_timestamp": start + 1.5})
        rows.append(row)
        sampling = f"meta/quality/{episode:06d}/export_samples.npz"
        provenance["episodes"].append({"episode_index": episode, "episode_id": f"capture_{episode + 10}",
                                        "task": f"delivered task {episode}", "sampling_file": sampling})
        sampling_path = scene / sampling
        sampling_path.parent.mkdir(parents=True)
        np.savez(sampling_path, source_global_timestamp_s=np.arange(3) / 2 + 100,
                 left_completion_method=np.array(["observed", "linear", "observed"]),
                 left_is_filled=np.array([False, True, False]))
        for i in range(3):
            state = np.arange(16, dtype=np.float32) + episode * 100 + i
            action = np.arange(16, dtype=np.float32) + episode * 100 + min(i + 1, 2)
            records.append({"episode_index": episode, "frame_index": i, "index": episode * 3 + i,
                            "timestamp": i / 2, "observation.state": state.tolist(), "action": action.tolist(),
                            "observation.camera_pose.left": [1, 2, 3, 0, 0, 0, 1],
                            "observation.camera_pose.right": [4, 5, 6, 0, 0, 0, 1],
                            "observation.gripper_filled.left": i == 1,
                            "observation.source_timestamp_s.left": 100 + i / 2,
                            "observation.pose_gap.left": 0.001})
    pq.write_table(pa.Table.from_pylist(rows), scene / "meta/episodes/chunk-003/file-008.parquet")
    for chunk, file, episode_ids in [(5, 9, (0, 1)), (6, 10, (2,))]:
        path = scene / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row for row in reversed(records) if row["episode_index"] in episode_ids]), path)
    (scene / "meta/export_provenance.json").write_text(json.dumps(provenance))
    return scene


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scene_path = fixture_scene(self.root)
        self.scene = SceneDataset(self.scene_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_discovery_release_lerobot_and_single_scene(self) -> None:
        fixture_scene(self.root, "scene_beta")
        self.assertEqual([s.name for s in discover_datasets(self.root)], ["scene_alpha", "scene_beta"])
        self.assertEqual(len(discover_datasets(self.root / "lerobot")), 2)
        self.assertEqual(len(discover_datasets(self.scene_path)), 1)
        self.assertEqual(self.scene.world_frame, "scene_alpha_world")
        self.assertEqual(self.scene.episodes[1].episode_id, "capture_11")

    def test_shared_data_filters_and_sorts_episode(self) -> None:
        samples = self.scene.load_episode(1)
        self.assertEqual(samples.state.shape, (3, 16))
        np.testing.assert_array_equal(samples.state[:, 0], [100, 101, 102])
        np.testing.assert_array_equal(samples.action[:-1], samples.state[1:])
        np.testing.assert_array_equal(samples.action[-1], samples.state[-1])
        self.assertEqual(samples.camera_poses["right"].shape, (3, 7))
        self.assertEqual(samples.sample(1).quality["left_completion_method"], "linear")
        self.assertTrue(samples.sample(1).quality["observation.gripper_filled.left"])
        self.assertEqual(samples.sample(1).timestamp, 0.5)
        self.assertEqual(samples.sample(1).quality["source_global_timestamp_s"], 100.5)
        with self.assertRaises(ValueError):
            samples.state[0, 0] = 0

    def test_shared_video_offset_and_later_chunk(self) -> None:
        first = self.scene.video_reference(0, "left_front", 2)
        second = self.scene.video_reference(1, "left_front", 0)
        self.assertEqual(first.path, second.path)
        self.assertEqual(first.timestamp, 1.0)
        self.assertEqual(second.timestamp, 1.5)
        self.assertTrue(str(second.path).endswith("observation.images.left_front/chunk-003/file-002.mp4"))
        last = self.scene.sample(2, 2)
        self.assertEqual(len(last.videos), 4)
        for reference in last.videos.values():
            self.assertEqual(reference.timestamp, 6.0)
            self.assertTrue(str(reference.path).endswith("chunk-004/file-007.mp4"))

    def test_invalid_index_and_camera(self) -> None:
        for i in [-1, 3]:
            with self.assertRaises(IndexError):
                self.scene.sample(0, i)
        with self.assertRaises(TypeError):
            self.scene.sample(0, 1.5)
        with self.assertRaises(KeyError):
            self.scene.load_episode(3)
        with self.assertRaises(KeyError):
            self.scene.video_reference(0, "missing", 0)
        other = SceneDataset(self.scene_path)
        with self.assertRaises(ValueError):
            self.scene.load_episode(other.episodes[0])

    def test_missing_rows_are_not_silently_misaligned(self) -> None:
        path = self.scene_path / "data/chunk-006/file-010.parquet"
        pq.write_table(pq.read_table(path).slice(0, 2), path)
        with self.assertRaisesRegex(DatasetError, "expected 3 rows"):
            self.scene.load_episode(2)

    def test_duplicate_frame_and_nonfinite_pose_fail(self) -> None:
        path = self.scene_path / "data/chunk-006/file-010.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["frame_index"] = rows[1]["frame_index"]
        pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaisesRegex(DatasetError, "missing or duplicate"):
            self.scene.load_episode(2)
        rows[0]["frame_index"] = 2
        rows[0]["observation.state"][0] = float("nan")
        pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaisesRegex(DatasetError, "nonfinite"):
            self.scene.load_episode(2)

    def test_metadata_path_cannot_escape_dataset(self) -> None:
        self.scene.metadata["data_path"] = "../../outside.parquet"
        with self.assertRaisesRegex(DatasetError, "escapes"):
            self.scene.load_episode(0)

    def test_quality_length_must_match_episode(self) -> None:
        np.savez(self.scene_path / "meta/quality/000000/export_samples.npz", left_is_filled=[False])
        with self.assertRaisesRegex(DatasetError, "wrong length"):
            self.scene.load_episode(0)

    def test_video_interval_must_contain_requested_frame(self) -> None:
        self.scene.episodes[0].metadata["videos/observation.images.left_front/to_timestamp"] = 0.7
        with self.assertRaisesRegex(DatasetError, "outside"):
            self.scene.video_reference(0, "left_front", 2)

    def test_missing_dataset_and_metadata_version(self) -> None:
        with self.assertRaises(FileNotFoundError):
            discover_datasets(self.root / "missing")
        info_path = self.scene_path / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["codebase_version"] = "v2.0"
        info_path.write_text(json.dumps(info))
        with self.assertRaisesRegex(DatasetError, "v3.0"):
            SceneDataset(self.scene_path)


if __name__ == "__main__":
    unittest.main()
