"""Bounded PyAV RGB decoding using actual presentation timestamps.

Random access seeks to a keyframe and decodes to the nearest PTS. Nearby forward
requests reuse the decoder, while a small frame cache makes repeated requests
cheap. Explicit close/context managers release all video file handles.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Any, Iterator

import av
import numpy as np

from .dataset import VideoReference


@dataclass(frozen=True)
class DecodedFrame:
    image: np.ndarray
    timestamp: float
    pts: int


class VideoReader:
    """Decode one file by absolute presentation time in seconds.

    ``read`` returns RGB uint8 H×W×3. ``read_frame`` also returns the selected PTS.
    Targets outside the stream's [start_time, end_time) interval raise IndexError;
    exact nearest-frame ties select the earlier frame. Instances are thread-safe.
    """

    def __init__(self, path: Path | str, *, cache_frames: int = 8, sequential_window_s: float = 2.0) -> None:
        if cache_frames < 2:
            raise ValueError("cache_frames must be at least 2")
        if not math.isfinite(sequential_window_s) or sequential_window_s < 0:
            raise ValueError("sequential_window_s must be finite and nonnegative")
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._lock = threading.RLock()
        self._container = av.open(str(self.path))
        try:
            if not self._container.streams.video:
                raise ValueError(f"No video stream in {self.path}")
            self._stream = self._container.streams.video[0]
            if self._stream.time_base is None:
                raise ValueError(f"Video has no presentation time base: {self.path}")
            self._stream.thread_type = "AUTO"
            self._stream.codec_context.thread_count = 2
            self.time_base = float(self._stream.time_base)
            rate = self._stream.average_rate or self._stream.guessed_rate
            self.fps = float(rate) if rate else 30.0
            self.start_time = float(self._stream.start_time or 0) * self.time_base
            self.end_time = None if self._stream.duration is None else self.start_time + float(self._stream.duration) * self.time_base
        except Exception:
            self._container.close()
            raise
        self._cache: deque[DecodedFrame] = deque(maxlen=cache_frames)
        self._iterator: Iterator[Any] | None = None
        self._last_decoded: float | None = None
        self._eof = False
        self._closed = False
        self._sequential_window_s = sequential_window_s

    @property
    def closed(self) -> bool:
        return self._closed

    def _seek(self, timestamp: float) -> None:
        # Include a preceding frame even when the requested time is a keyframe:
        # choosing a future frame alone would be wrong for a nearby earlier PTS.
        seek_time = max(self.start_time, timestamp - 1.0 / self.fps)
        self._container.seek(math.floor(seek_time / self.time_base), stream=self._stream, backward=True, any_frame=False)
        self._iterator = iter(self._container.decode(self._stream))
        self._cache.clear()
        self._last_decoded = None
        self._eof = False

    @staticmethod
    def _nearest(frames: list[DecodedFrame], timestamp: float) -> DecodedFrame:
        # Roundoff at fractional frame rates must not flip an exact tie.
        best = frames[0]
        for frame in frames[1:]:
            if abs(frame.timestamp - timestamp) < abs(best.timestamp - timestamp) - 1e-12:
                best = frame
        return best

    def read(self, timestamp: float) -> np.ndarray:
        return self.read_frame(timestamp).image

    def read_frame(self, timestamp: float) -> DecodedFrame:
        timestamp = float(timestamp)
        with self._lock:
            if self._closed:
                raise RuntimeError("VideoReader is closed")
            if not math.isfinite(timestamp):
                raise ValueError("Video timestamp must be finite")
            if timestamp < self.start_time - 1e-9 or (self.end_time is not None and timestamp >= self.end_time):
                raise IndexError(f"Video timestamp {timestamp} outside [{self.start_time}, {self.end_time}) for {self.path.name}")
            if self._cache and self._cache[0].timestamp <= timestamp <= self._cache[-1].timestamp:
                return self._nearest(list(self._cache), timestamp)
            can_continue = self._last_decoded is not None and self._last_decoded <= timestamp and timestamp - self._last_decoded <= self._sequential_window_s
            if not can_continue:
                self._seek(timestamp)
            previous = self._cache[-1] if self._cache else None
            assert self._iterator is not None
            if not self._eof:
                for frame in self._iterator:
                    if frame.pts is None:
                        continue
                    actual_time = float(frame.pts * frame.time_base)
                    image = frame.to_ndarray(format="rgb24")
                    image.setflags(write=False)
                    decoded = DecodedFrame(image, actual_time, int(frame.pts))
                    self._cache.append(decoded)
                    self._last_decoded = actual_time
                    if actual_time >= timestamp:
                        candidates = [previous, decoded] if previous is not None else [decoded]
                        return self._nearest(candidates, timestamp)
                    previous = decoded
                self._eof = True
            if previous is None:
                raise ValueError(f"No decodable frame at {timestamp} in {self.path}")
            # Unknown-duration streams still must not silently return a stale frame.
            if self.end_time is None and timestamp >= previous.timestamp + 1.0 / self.fps:
                raise IndexError(f"Video timestamp {timestamp} exceeds the last frame in {self.path}")
            return previous

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._container.close()
                self._cache.clear()
                self._iterator = None
                self._closed = True

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class VideoPool:
    """LRU-bounded readers for scene/episode switching; defaults to eight files."""

    def __init__(self, max_open: int = 8, *, cache_frames: int = 8) -> None:
        if max_open < 1:
            raise ValueError("max_open must be positive")
        self.max_open = max_open
        self.cache_frames = cache_frames
        self._readers: OrderedDict[Path, VideoReader] = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False

    def read(self, reference: VideoReference) -> np.ndarray:
        return self.read_frame(reference).image

    def read_frame(self, reference: VideoReference) -> DecodedFrame:
        with self._lock:
            if self._closed:
                raise RuntimeError("VideoPool is closed")
            if not reference.from_timestamp <= reference.timestamp < reference.to_timestamp:
                raise IndexError("Video reference is outside its episode interval")
            path = reference.path.resolve()
            if path not in self._readers:
                if len(self._readers) >= self.max_open:
                    _, oldest = self._readers.popitem(last=False)
                    oldest.close()
                self._readers[path] = VideoReader(path, cache_frames=self.cache_frames)
            self._readers.move_to_end(path)
            return self._readers[path].read_frame(reference.timestamp)

    def close(self) -> None:
        with self._lock:
            for reader in self._readers.values():
                reader.close()
            self._readers.clear()
            self._closed = True

    def __enter__(self) -> VideoPool:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
