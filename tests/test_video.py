"""Small encoded movies test PTS, seeking, shared-file boundaries, and closure."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from kiwi_viser.dataset import VideoReference
from kiwi_viser.video import VideoPool, VideoReader


def make_video(path: Path, offset: int = 0) -> None:
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 4
        stream.codec_context.max_b_frames = 2
        for index in range(12):
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[:, :, 0] = index * 20
            image[:, :, 1] = 255 - index * 17
            image[:, index: index + 4, 2] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = offset + index
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def read_all(path: Path):
    with av.open(str(path)) as source:
        return [(float(frame.pts * frame.time_base), frame.to_ndarray(format="rgb24")) for frame in source.decode(video=0)]


class VideoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "offset.mp4"
        make_video(self.path, 50)
        self.frames = read_all(self.path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def assert_decodes_frame(self, reader: VideoReader, timestamp: float, expected_index: int) -> None:
        result = reader.read_frame(timestamp)
        self.assertAlmostEqual(result.timestamp, self.frames[expected_index][0], places=8)
        np.testing.assert_array_equal(result.image, self.frames[expected_index][1])
        self.assertEqual(result.image.dtype, np.uint8)
        self.assertEqual(result.image.shape, (24, 32, 3))

    def test_nonzero_pts_and_random_seek_nearest(self) -> None:
        self.assertEqual(self.frames[0][0], 5.0)
        with VideoReader(self.path) as reader:
            for target, expected in [(6.1, 11), (5.04, 0), (5.06, 1), (5.85, 8), (5.25, 2), (5.91, 9), (5.0, 0)]:
                with self.subTest(target=target):
                    self.assert_decodes_frame(reader, target, expected)

    def test_sequential_and_repeat_use_same_rgb(self) -> None:
        with VideoReader(self.path) as reader:
            for index in range(12):
                self.assert_decodes_frame(reader, 5.0 + index / 10, index)
            first = reader.read(6.1)
            second = reader.read(6.1)
            self.assertIs(first, second)
            self.assert_decodes_frame(reader, 6.19, 11)

    def test_bounds_and_closing(self) -> None:
        reader = VideoReader(self.path)
        for timestamp in [-1, 4.999, 6.2, 100]:
            with self.assertRaises(IndexError):
                reader.read(timestamp)
        for timestamp in [float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                reader.read(timestamp)
        reader.close()
        reader.close()
        self.assertTrue(reader.closed)
        with self.assertRaises(RuntimeError):
            reader.read(5.0)

    def test_shared_episode_video_references(self) -> None:
        # Episode A uses frames 0..5, B uses 6..11 of the same file.
        with VideoPool() as pool:
            for target, start, end, expected in [(5.5, 5, 5.6, 5), (5.6, 5.6, 6.2, 6), (6.1, 5.6, 6.2, 11)]:
                reference = VideoReference(self.path, target, start, end, "left_front", expected)
                np.testing.assert_array_equal(pool.read(reference), self.frames[expected][1])
            with self.assertRaises(IndexError):
                pool.read(VideoReference(self.path, 5.5, 5.6, 6.2, "left_front", 0))
        with self.assertRaises(RuntimeError):
            pool.read(reference)

    def test_pool_closes_evicted_and_remaining_readers(self) -> None:
        second_path = self.root / "second.mp4"
        make_video(second_path)
        created = []
        def create_reader(*args, **kwargs):
            reader = VideoReader(*args, **kwargs)
            created.append(reader)
            return reader
        with patch("kiwi_viser.video.VideoReader", side_effect=create_reader):
            with VideoPool(max_open=1) as pool:
                pool.read(VideoReference(self.path, 5, 5, 6.2, "left_front", 0))
                pool.read(VideoReference(second_path, 0, 0, 1.2, "left_front", 0))
                self.assertTrue(created[0].closed)
                self.assertFalse(created[1].closed)
            self.assertTrue(created[1].closed)

    def test_missing_or_invalid_files_and_configuration(self) -> None:
        with self.assertRaises(FileNotFoundError):
            VideoReader(self.root / "missing.mp4")
        broken = self.root / "broken.mp4"
        broken.write_text("not a video")
        with self.assertRaises(av.FFmpegError):
            VideoReader(broken)
        with self.assertRaises(ValueError):
            VideoPool(max_open=0)
        with self.assertRaises(ValueError):
            VideoReader(self.path, cache_frames=1)


if __name__ == "__main__":
    unittest.main()
