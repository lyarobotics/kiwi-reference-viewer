"""Read delivered LeRobot v3 files without importing the exporter or LeRobot.

All poses are returned exactly as stored: metres and Hamilton ``xyzw``
body-to-world quaternions. Each SceneDataset owns an independent world. Video
references use the *exported* episode timeline; quality NPZ timestamps describe
the original recordings and must not be used to seek the exported MP4s.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
import operator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class DatasetError(ValueError):
    """Delivered metadata or samples do not satisfy the declared dataset contract."""


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DatasetError(f"Expected a JSON object in {path}")
    return value


def _local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DatasetError(f"Metadata path escapes dataset root: {relative}")
    return path


def _index(index: int, length: int) -> int:
    try:
        result = operator.index(index)
    except TypeError as exc:
        raise TypeError("Frame index must be an integer") from exc
    if not 0 <= result < length:
        raise IndexError(f"Frame {result} is outside [0, {length})")
    return result


@dataclass(frozen=True)
class Episode:
    """Episode identity and file metadata; indices are local to its scene."""

    index: int
    episode_id: str
    task: str
    length: int
    fps: float
    metadata: dict[str, Any] = field(repr=False)
    provenance: dict[str, Any] = field(repr=False)

    @property
    def duration(self) -> float:
        """Playback extent in seconds, including the final frame interval."""
        return self.length / self.fps

    @property
    def label(self) -> str:
        return f"{self.episode_id} · {self.task}"


@dataclass(frozen=True)
class VideoReference:
    """An absolute MP4 path and a timestamp on that file's presentation clock."""

    path: Path
    timestamp: float
    from_timestamp: float
    to_timestamp: float
    camera_key: str
    frame_index: int


@dataclass(frozen=True)
class FrameSample:
    """One synchronized row, with RGB references decoded separately on demand."""

    frame_index: int
    timestamp: float
    state: np.ndarray
    action: np.ndarray
    camera_poses: dict[str, np.ndarray]
    quality: dict[str, Any]
    features: dict[str, Any]
    videos: dict[str, VideoReference]


@dataclass
class EpisodeSamples:
    """Read-only NumPy columns for one episode, in ascending frame order.

    ``arrays`` retains every Parquet feature under its delivered name.
    ``quality`` combines sampling NPZ arrays (their original names) and all
    scalar observation features. Access ``sample(i)`` for a synchronized row.
    """

    scene: SceneDataset = field(repr=False)
    episode: Episode
    arrays: dict[str, np.ndarray] = field(repr=False)
    quality: dict[str, np.ndarray] = field(repr=False)

    def __len__(self) -> int:
        return self.episode.length

    @property
    def state(self) -> np.ndarray:
        return self.arrays["observation.state"]

    @property
    def action(self) -> np.ndarray:
        return self.arrays["action"]

    @property
    def timestamps(self) -> np.ndarray:
        return self.arrays["timestamp"]

    @property
    def camera_poses(self) -> dict[str, np.ndarray]:
        prefix = "observation.camera_pose."
        return {key[len(prefix):]: value for key, value in self.arrays.items() if key.startswith(prefix)}

    def sample(self, frame_index: int) -> FrameSample:
        i = _index(frame_index, len(self))
        def row(value: np.ndarray) -> Any:
            result = value[i]
            return result.item() if isinstance(result, np.generic) else result
        timestamp = float(self.timestamps[i])
        return FrameSample(
            frame_index=i,
            timestamp=timestamp,
            state=self.state[i],
            action=self.action[i],
            camera_poses={side: values[i] for side, values in self.camera_poses.items()},
            quality={key: row(value) for key, value in self.quality.items()},
            features={key: row(value) for key, value in self.arrays.items()},
            videos={key: self.scene._video_reference(self.episode, key, i, timestamp) for key in self.scene.video_keys},
        )


@dataclass
class SceneDataset:
    """A local scene dataset, loaded solely from its supplied metadata.

    Construct with the scene root, or use ``discover_datasets`` with a release
    root. Episode arrays load lazily; the two most recently used episodes are
    cached by default. No image data is loaded by this class.
    """

    root: Path | str
    max_cached_episodes: int = 2
    metadata: dict[str, Any] = field(init=False)
    provenance: dict[str, Any] = field(init=False)
    episodes: list[Episode] = field(init=False)
    _cache: OrderedDict[int, EpisodeSamples] = field(init=False, repr=False, default_factory=OrderedDict)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.metadata = _json(self.root / "meta/info.json")
        if self.metadata.get("codebase_version") != "v3.0":
            raise DatasetError(f"Expected LeRobot v3.0 metadata at {self.root}")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise DatasetError("Dataset fps must be positive and finite")
        if self.max_cached_episodes < 0:
            raise ValueError("max_cached_episodes must be nonnegative")
        provenance_path = self.root / "meta/export_provenance.json"
        self.provenance = _json(provenance_path) if provenance_path.exists() else {}
        provenance_episodes = {int(row["episode_index"]): row for row in self.provenance.get("episodes", [])}
        metadata_files = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if not metadata_files:
            raise FileNotFoundError(f"No episode metadata found in {self.root / 'meta/episodes'}")
        rows = [row for path in metadata_files for row in pq.read_table(path).to_pylist()]
        self.episodes = []
        seen: set[int] = set()
        for row in sorted(rows, key=lambda value: value["episode_index"]):
            index = int(row["episode_index"])
            if index in seen:
                raise DatasetError(f"Duplicate episode index {index} in {self.root}")
            seen.add(index)
            length = int(row["length"])
            if length <= 0:
                raise DatasetError(f"Episode {index} has no frames")
            if int(row["dataset_to_index"]) - int(row["dataset_from_index"]) != length:
                raise DatasetError(f"Episode {index} length does not match dataset index range")
            provenance = provenance_episodes.get(index, {})
            tasks = row.get("tasks") or []
            self.episodes.append(Episode(
                index=index,
                episode_id=str(provenance.get("episode_id", f"episode_{index:06d}")),
                task=str(provenance.get("task") or "; ".join(tasks) or "Unspecified task"),
                length=length,
                fps=self.fps,
                metadata=row,
                provenance=provenance,
            ))
        if len(self.episodes) != int(self.metadata.get("total_episodes", len(self.episodes))):
            raise DatasetError(f"Episode metadata count does not match meta/info.json in {self.root}")

    @property
    def info(self) -> dict[str, Any]:
        return self.metadata

    @property
    def name(self) -> str:
        return str(self.provenance.get("scene", {}).get("scene_id", self.root.name))

    @property
    def world_frame(self) -> str:
        return str(self.provenance.get("scene", {}).get("world_frame", "unspecified"))

    @property
    def fps(self) -> float:
        return float(self.metadata["fps"])

    @property
    def video_keys(self) -> tuple[str, ...]:
        return tuple(key for key, feature in self.metadata["features"].items() if feature.get("dtype") == "video")

    @classmethod
    def discover(cls, root: Path | str) -> list[SceneDataset]:
        return discover_datasets(root)

    def episode(self, episode: int | Episode) -> Episode:
        index = episode.index if isinstance(episode, Episode) else operator.index(episode)
        for candidate in self.episodes:
            if candidate.index == index:
                if isinstance(episode, Episode) and candidate is not episode:
                    raise ValueError("Episode belongs to a different SceneDataset")
                return candidate
        raise KeyError(f"No episode {index} in {self.name}")

    def load_episode(self, episode: int | Episode) -> EpisodeSamples:
        ep = self.episode(episode)
        if ep.index in self._cache:
            self._cache.move_to_end(ep.index)
            return self._cache[ep.index]
        relative = self.metadata["data_path"].format(
            chunk_index=int(ep.metadata["data/chunk_index"]),
            file_index=int(ep.metadata["data/file_index"]),
            episode_index=ep.index,
        )
        path = _local_path(self.root, relative)
        table = pq.read_table(path, filters=[("episode_index", "=", ep.index)])
        if len(table) != ep.length:
            raise DatasetError(f"Episode {ep.index}: expected {ep.length} rows, found {len(table)} in {path}")
        if table["frame_index"].null_count:
            raise DatasetError(f"Episode {ep.index} contains null frame indices")
        order = np.argsort(table["frame_index"].to_numpy(), kind="stable")
        table = table.take(pa.array(order))
        arrays: dict[str, np.ndarray] = {}
        for key in table.column_names:
            column = table[key].combine_chunks()
            if column.null_count:
                raise DatasetError(f"Episode {ep.index}: null values in {key}")
            if pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type) or pa.types.is_large_list(column.type):
                value = np.asarray(column.to_pylist(), dtype=self.metadata.get("features", {}).get(key, {}).get("dtype"))
            else:
                value = column.to_numpy(zero_copy_only=False)
            value.setflags(write=False)
            arrays[key] = value
        if not np.array_equal(arrays["frame_index"], np.arange(ep.length)):
            raise DatasetError(f"Episode {ep.index} has missing or duplicate frame indices")
        for key, width in [("observation.state", 16), ("action", 16)]:
            if key not in arrays or arrays[key].shape != (ep.length, width):
                raise DatasetError(f"Episode {ep.index}: {key} must have shape ({ep.length}, {width})")
            if not np.isfinite(arrays[key]).all():
                raise DatasetError(f"Episode {ep.index}: {key} contains nonfinite values")
        for key, value in arrays.items():
            if key.startswith("observation.camera_pose.") and (value.shape != (ep.length, 7) or not np.isfinite(value).all()):
                raise DatasetError(f"Episode {ep.index}: invalid {key}")
        timestamps = arrays["timestamp"]
        if timestamps.shape != (ep.length,) or not np.isfinite(timestamps).all() or timestamps[0] < 0 or np.any(np.diff(timestamps) <= 0):
            raise DatasetError(f"Episode {ep.index}: timestamps must be finite, nonnegative and increasing")
        if "index" in arrays:
            expected = np.arange(int(ep.metadata["dataset_from_index"]), int(ep.metadata["dataset_to_index"]))
            if not np.array_equal(arrays["index"], expected):
                raise DatasetError(f"Episode {ep.index}: global indices disagree with metadata")
        quality = {key: value for key, value in arrays.items() if key.startswith("observation.") and value.ndim == 1}
        sampling_file = ep.provenance.get("sampling_file")
        if sampling_file:
            sampling_path = _local_path(self.root, sampling_file)
            with np.load(sampling_path, allow_pickle=False) as archive:
                for key in archive.files:
                    value = archive[key]
                    if value.ndim == 0 or len(value) != ep.length:
                        raise DatasetError(f"Episode {ep.index}: sampling array {key} has the wrong length")
                    value.setflags(write=False)
                    quality[key] = value
        samples = EpisodeSamples(self, ep, arrays, quality)
        if self.max_cached_episodes:
            self._cache[ep.index] = samples
            while len(self._cache) > self.max_cached_episodes:
                self._cache.popitem(last=False)
        return samples

    def sample(self, episode: int | Episode, frame_index: int) -> FrameSample:
        return self.load_episode(episode).sample(frame_index)

    def video_reference(self, episode: int | Episode, camera_key: str, frame_index: int) -> VideoReference:
        ep = self.episode(episode)
        i = _index(frame_index, ep.length)
        timestamp = float(self.load_episode(ep).timestamps[i])
        return self._video_reference(ep, camera_key, i, timestamp)

    def _video_reference(self, ep: Episode, camera_key: str, frame_index: int, timestamp: float) -> VideoReference:
        if camera_key not in self.video_keys:
            camera_key = "observation.images." + camera_key
        if camera_key not in self.video_keys:
            raise KeyError(f"Unknown camera {camera_key!r}; available: {self.video_keys}")
        prefix = f"videos/{camera_key}/"
        try:
            start = float(ep.metadata[prefix + "from_timestamp"])
            end = float(ep.metadata[prefix + "to_timestamp"])
            relative = self.metadata["video_path"].format(
                video_key=camera_key,
                chunk_index=int(ep.metadata[prefix + "chunk_index"]),
                file_index=int(ep.metadata[prefix + "file_index"]),
                episode_index=ep.index,
            )
        except KeyError as exc:
            raise DatasetError(f"Episode {ep.index}: missing video metadata for {camera_key}") from exc
        target = start + timestamp
        if not all(math.isfinite(value) for value in (start, end, target)) or start < 0 or not start <= target < end:
            raise DatasetError(f"Episode {ep.index}: frame timestamp is outside {camera_key}'s video interval")
        return VideoReference(_local_path(self.root, relative), target, start, end, camera_key, frame_index)


def discover_datasets(root: Path | str) -> list[SceneDataset]:
    """Discover scene roots under a release, its ``lerobot`` folder, or one scene.

    Only delivered ``meta/info.json`` files identify a dataset; no scene names,
    episode counts or private source-tree paths are assumed.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if (root / "meta/info.json").is_file():
        paths = [root]
    else:
        search = root / "lerobot" if (root / "lerobot").is_dir() else root
        paths = sorted({path.parent.parent for path in search.glob("*/meta/info.json")})
    if not paths:
        raise FileNotFoundError(f"No LeRobot scene datasets found under {root}")
    return [SceneDataset(path) for path in paths]
