"""A local Viser session with one owner thread for decoding and scene changes."""
from __future__ import annotations

import logging
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import numpy as np
from PIL import Image, ImageDraw
import viser

from .dataset import discover_datasets
from .geometry import load_scene_from_metadata, suggest_camera, xyzw_to_wxyz
from .video import VideoPool

COLORS = {"left": (52, 211, 223), "right": (255, 158, 79)}
VIEWS = ("left_front", "right_front", "left_rear", "right_rear")


def trajectory_segments(positions: np.ndarray, timestamps: np.ndarray, frame_index: int,
                        *, half_window_seconds: float | None = None,
                        max_points: int = 5000) -> np.ndarray:
    """Build bounded episode-local segments, retaining both sampled endpoints.

    ``None`` selects the full episode; a window includes observations at both
    time boundaries and the current frame. Invalid positions create gaps rather
    than connecting unrelated parts of a trajectory, including when decimated.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        positions = np.asarray(positions, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or timestamps.shape != (len(positions),):
        raise ValueError("Trajectory positions and timestamps must have shapes (N, 3) and (N,)")
    if not 0 <= frame_index < len(positions):
        raise IndexError("Trajectory frame is outside the episode")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
        raise ValueError("Trajectory timestamps must be finite and sorted")
    if max_points < 2:
        raise ValueError("At least two trajectory points are required")
    start, stop = 0, len(positions)
    if half_window_seconds is not None:
        if not np.isfinite(half_window_seconds) or half_window_seconds < 0:
            raise ValueError("Trajectory window must be finite and nonnegative")
        center = timestamps[frame_index]
        start = int(np.searchsorted(timestamps, center - half_window_seconds, side="left"))
        stop = int(np.searchsorted(timestamps, center + half_window_seconds, side="right"))
    if stop - start < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    selected = np.linspace(start, stop - 1, min(stop - start, max_points), dtype=int)
    if half_window_seconds is not None:
        selected = np.unique(np.append(selected, frame_index))
    # Prefix sums prevent decimation from drawing through a missing sample.
    invalid = ~np.isfinite(positions[start:stop]).all(axis=1)
    invalid_prefix = np.concatenate(([0], np.cumsum(invalid)))
    valid = (invalid_prefix[selected[1:] - start + 1] - invalid_prefix[selected[:-1] - start]) == 0
    segments = np.stack((positions[selected[:-1]], positions[selected[1:]]), axis=1)
    return segments[valid]


def make_mosaic(images: dict[str, np.ndarray], frame, width: int = 640) -> np.ndarray:
    """Label RGB views and normalized opening without inventing jaw dimensions."""
    cell = width // 2
    band = 28
    canvas = Image.new("RGB", (width, 2 * (cell + band) + 54), (18, 23, 32))
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(VIEWS):
        x, y = (i % 2) * cell, (i // 2) * (cell + band)
        color = COLORS[name.split("_")[0]]
        draw.text((x + 10, y + 8), name.replace("_", " ").upper(), fill=color)
        if name in images:
            picture = Image.fromarray(images[name]).resize((cell, cell), Image.Resampling.BILINEAR)
            canvas.paste(picture, (x, y + band))
    y = 2 * (cell + band)
    for side, offset, x in (("left", 0, 0), ("right", 8, cell)):
        opening = float(frame.state[offset + 7])
        filled = any(side in key and "filled" in key and bool(np.any(value))
                     for key, value in frame.quality.items())
        label = f"{side.title()} opening {opening:.2f}" + ("  [filled]" if filled else "")
        draw.text((x + 10, y + 8), label, fill=COLORS[side])
        draw.rounded_rectangle((x + 10, y + 30, x + cell - 10, y + 41), radius=3, fill=(45, 53, 66))
        if opening > 0:
            draw.rounded_rectangle((x + 10, y + 30, x + 10 + (cell - 20) * np.clip(opening, 0, 1), y + 41),
                                   radius=3, fill=COLORS[side])
    return np.asarray(canvas)


def dataset_notes(scene) -> str:
    p = scene.provenance
    lines = [f"**{scene.world_frame}** · metres · {p.get('scene', {}).get('up_axis', 'unspecified')} up",
             p.get("scene", {}).get("coordinate_scope", ""),
             "**Pose convention**", p.get("state", {}).get("pose", "Not declared."),
             "**Action**", p.get("state", {}).get("action", "Not declared.")]
    for title, section, key in (("TCP", "tcp", "limitation"), ("Camera", "calibration", "limitation"),
                                ("Gripper", "gripper", "endpoint_limitation"),
                                ("Environment", "environment", "limitation")):
        if p.get(section, {}).get(key):
            lines += [f"**{title}**", p[section][key]]
    # Surface additional scene/episode caveats without hardcoding any dataset identity.
    def caveats(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                if any(term in key.lower() for term in ("alignment", "warning", "caveat", "known_issue")):
                    if isinstance(child, str):
                        lines.append(child)
                    elif isinstance(child, (dict, list)):
                        caveats(child, key)
                elif isinstance(child, (dict, list)):
                    caveats(child, key)
                elif prefix and isinstance(child, str) and "limitation" in key:
                    lines.append(child)
        elif isinstance(value, list):
            for child in value:
                caveats(child, prefix)
    caveats(p.get("episodes", []))
    return "\n\n".join(str(line) for line in lines if line)


class Viewer:
    def __init__(self, root: Path, *, scene_name: str | None = None, episode_index: int = 0,
                 host: str = "127.0.0.1", port: int = 8080, max_gaussians: int | None = None,
                 render: str = "splats"):
        self.scenes = {scene.name: scene for scene in discover_datasets(root)}
        if not self.scenes:
            raise ValueError("No datasets found")
        if scene_name is not None and scene_name not in self.scenes:
            raise ValueError(f"Unknown scene {scene_name}; choose from {', '.join(self.scenes)}")
        self.scene = self.scenes[scene_name or next(iter(self.scenes))]
        self.scene.episode(episode_index)  # Validate before starting a server.
        self.max_gaussians, self.render_mode = max_gaussians, render
        self.commands = SimpleQueue()
        self.videos = VideoPool(max_open=8)
        self.server = viser.ViserServer(host=host, port=port, label="KIWI Reference Viewer")
        self.server.scene.set_up_direction("+z")
        self.server.gui.configure_theme(control_layout="floating", control_width="large", dark_mode=True,
                                        show_logo=False, show_share_button=False, brand_color=(52, 211, 223))
        self.server.gui.main_panel.dock_right()
        self.frame_index = 0
        self.playing = False
        self.speed = 1.0
        self.last_tick = time.monotonic()
        self.playhead = 0.0
        self.scene_handles = []
        self.episode_handles = []
        self.pose_handles = {}
        self.environment_handle = None
        self.bounds = np.array([[-1., -1., -1.], [1., 1., 1.]])
        self.camera_view = None
        self._create_gui()
        self.server.on_client_connect(lambda client: self.commands.put(("camera", client)))
        self._load_scene(self.scene.name, episode_index)
        print(f"Viewer ready: http://{host}:{self.server.get_port()}", flush=True)

    def _user_update(self, handle, command):
        @handle.on_update
        def update(event):
            if event.client is not None:
                self.commands.put((command, handle.value))

    def _create_gui(self):
        gui = self.server.gui
        gui.add_markdown("### KIWI Reference Viewer\nScene-aligned demonstrations")
        self.scene_select = gui.add_dropdown("Scene", list(self.scenes), initial_value=self.scene.name)
        self.episode_select = gui.add_dropdown("Episode", [e.label for e in self.scene.episodes])
        self._user_update(self.scene_select, "scene")
        self._user_update(self.episode_select, "episode")
        tabs = gui.add_tab_group()
        with tabs.add_tab("Playback"):
            self.status = gui.add_markdown("Loading dataset…")
            self.timeline = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self._user_update(self.timeline, "frame")
            self.play = gui.add_checkbox("Playing", False)
            self._user_update(self.play, "play")
            self.speed_select = gui.add_dropdown("Speed", ["0.25×", "0.5×", "1×", "2×"], initial_value="1×")
            self._user_update(self.speed_select, "speed")
            self.mosaic = gui.add_image(np.zeros((750, 640, 3), dtype=np.uint8), format="jpeg", jpeg_quality=82)
            gui.add_markdown("**Cyan** left TCP · **Orange** right TCP\n\nOpening: 0 closed → 1 open")
        with tabs.add_tab("Scene"):
            reset = gui.add_button("Reset view")
            reset.on_click(lambda _: self.commands.put(("camera", None)))
            self.show_environment = gui.add_checkbox("Environment", True)
            self.show_trajectories = gui.add_checkbox("Trajectories", True)
            self.trajectory_range = gui.add_dropdown(
                "Trajectory range", ["Nearby", "Full episode"], initial_value="Nearby")
            self.trajectory_window = gui.add_slider(
                "Window ± seconds", min=0.25, max=10.0, step=0.25, initial_value=2.0)
            self._user_update(self.trajectory_range, "trajectory")
            self._user_update(self.trajectory_window, "trajectory")
            self.show_cameras = gui.add_checkbox("Front-camera axes", False)
            self.show_actions = gui.add_checkbox("Next-state targets", False)
            for handle in (self.show_environment, self.show_trajectories, self.show_cameras, self.show_actions):
                self._user_update(handle, "visibility")
            self.scene_info = gui.add_markdown("")
            gui.add_markdown("Drag to orbit · right drag to pan · scroll to zoom\n\nAxes: X red · Y green · Z blue")
        with tabs.add_tab("Dataset"):
            self.notes = gui.add_markdown("")

    def _load_scene(self, name, episode_index=None):
        self._pause()
        self.status.content = "Loading scene…"
        candidate = self.scenes[name]
        geometry = load_scene_from_metadata(candidate.root, candidate.provenance, max_gaussians=self.max_gaussians)
        self.scene = candidate
        for handle in self.scene_handles:
            handle.remove()
        self.scene_handles.clear()
        kwargs = dict(name="/environment", visible=self.show_environment.value)
        if self.render_mode == "splats":
            self.environment_handle = self.server.scene.add_gaussian_splats(
                **kwargs, centers=geometry.centers, covariances=geometry.covariances,
                rgbs=geometry.rgbs, opacities=geometry.opacities)
        else:
            self.environment_handle = self.server.scene.add_point_cloud(
                **kwargs, points=geometry.centers, colors=geometry.point_colors,
                point_size=0.008, point_shape="circle", precision="float32")
        self.scene_handles.append(self.environment_handle)
        self.bounds = geometry.bounds
        self.scene_centers = geometry.centers
        self.scene_opacities = geometry.opacities
        self.notes.content = dataset_notes(candidate)
        self.scene_info.content = (f"**{candidate.world_frame}**\n\n"
                                   f"{len(geometry.centers):,} / {geometry.source_count:,} Gaussians displayed\n\n"
                                   "Static environment · base colors")
        self.episode_select.options = [e.label for e in candidate.episodes]
        episode = candidate.episode(episode_index) if episode_index is not None else candidate.episodes[0]
        self.episode_select.value = episode.label
        self._load_episode(episode.index)

    def _pause(self):
        self.playing = False
        self.play.value = False

    def _load_episode(self, index):
        self._pause()
        samples = self.scene.load_episode(index)
        self.videos.close()
        self.videos = VideoPool(max_open=8)
        for handle in reversed(self.episode_handles):
            handle.remove()
        self.episode_handles.clear()
        self.pose_handles.clear()
        self.samples = samples
        self.frame_index = 0
        self.playhead = 0.0
        self.timeline.max = max(1, len(samples) - 1)
        self.timeline.value = 0
        self.full_trajectories = {}
        for side, offset in (("left", 0), ("right", 8)):
            positions = samples.state[:, offset:offset + 3]
            self.full_trajectories[side] = trajectory_segments(positions, samples.timestamps, 0)
            handle = self.server.scene.add_line_segments(
                f"/trajectory/{side}", points=np.empty((0, 2, 3), dtype=np.float32),
                colors=COLORS[side], thickness=1.5, thickness_units="screen",
                visible=self.show_trajectories.value)
            self.pose_handles[f"trajectory_{side}"] = handle
            self.episode_handles.append(handle)
            for kind, length in (("tcp", 0.09), ("camera", 0.05), ("action", 0.06)):
                handle = self.server.scene.add_frame(f"/{kind}/{side}", axes_length=length,
                    axes_radius=0.0025, origin_radius=0.012 if kind == "tcp" else 0.005, origin_color=COLORS[side],
                    visible=kind == "tcp")
                self.pose_handles[f"{kind}_{side}"] = handle
                self.episode_handles.append(handle)
            self.episode_handles.append(self.server.scene.add_label(
                f"/tcp/{side}/label", f"{side.title()} TCP", position=(0, 0, 0),
                font_screen_scale=0.8, anchor="bottom-left"))
        positions = np.concatenate((samples.state[:, :3], samples.state[:, 8:11]))
        self.camera_view = suggest_camera(
            self.bounds, trajectory_positions=positions,
            scene_centers=self.scene_centers, scene_opacities=self.scene_opacities)
        self._reset_camera()
        self._visibility()
        self._draw_frame(0)

    def _reset_camera(self, client=None):
        if self.camera_view is None:
            return
        clients = [client] if client is not None else list(self.server.get_clients().values())
        for entry in clients:
            with entry.atomic():
                entry.camera.position = self.camera_view.position
                entry.camera.look_at = self.camera_view.look_at
                entry.camera.up_direction = self.camera_view.up_direction

    def _visibility(self):
        if self.environment_handle:
            self.environment_handle.visible = self.show_environment.value
        for key, handle in self.pose_handles.items():
            if key.startswith("trajectory_"):
                handle.visible = self.show_trajectories.value
            elif key.startswith("camera_"):
                side = key.removeprefix("camera_")
                handle.visible = self.show_cameras.value and side in self.samples.camera_poses
            elif key.startswith("action_"):
                handle.visible = self.show_actions.value

    def _update_trajectories(self, index):
        nearby = self.trajectory_range.value == "Nearby"
        self.trajectory_window.disabled = not nearby
        for side, offset in (("left", 0), ("right", 8)):
            handle = self.pose_handles[f"trajectory_{side}"]
            points = (trajectory_segments(self.samples.state[:, offset:offset + 3],
                        self.samples.timestamps, index,
                        half_window_seconds=self.trajectory_window.value)
                      if nearby else self.full_trajectories[side])
            # Full paths do not change during playback; avoid resending them.
            if handle.points.shape != points.shape or not np.array_equal(handle.points, points):
                handle.points = points

    def _draw_frame(self, index):
        frame = self.samples.sample(index)
        images = {key.rsplit(".", 1)[-1]: self.videos.read(reference) for key, reference in frame.videos.items()}
        mosaic = make_mosaic(images, frame)
        with self.server.atomic():
            self._update_trajectories(index)
            for side, offset in (("left", 0), ("right", 8)):
                poses = {"tcp": frame.state[offset:offset + 7],
                         "action": frame.action[offset:offset + 7]}
                if side in frame.camera_poses:
                    poses["camera"] = frame.camera_poses[side]
                for kind, pose in poses.items():
                    handle = self.pose_handles[f"{kind}_{side}"]
                    handle.position = pose[:3]
                    handle.wxyz = xyzw_to_wxyz(pose[3:7])
            self.mosaic.image = mosaic
            self.timeline.value = index
            self.status.content = (f"**{self.samples.episode.task}**\n\n"
                f"{frame.timestamp:.2f} / {self.samples.timestamps[-1]:.2f} s · "
                f"frame {index:,} / {len(self.samples) - 1:,} · {self.scene.fps:g} Hz")
        self.frame_index = index

    def _command(self, name, value):
        if name == "scene":
            self._load_scene(value)
        elif name == "episode":
            episode = next(e for e in self.scene.episodes if e.label == value)
            self._load_episode(episode.index)
        elif name == "frame":
            self._pause()
            index = min(max(int(value), 0), len(self.samples) - 1)
            self.playhead = index / self.scene.fps
            self._draw_frame(index)
        elif name == "play":
            self.playing = bool(value)
            if self.playing and self.frame_index == len(self.samples) - 1:
                self.playhead = 0.0
            self.last_tick = time.monotonic()
        elif name == "speed":
            self.speed = float(value.removesuffix("×"))
        elif name == "visibility":
            self._visibility()
        elif name == "trajectory":
            with self.server.atomic():
                self._update_trajectories(self.frame_index)
        elif name == "camera":
            self._reset_camera(value)

    def run(self):
        try:
            while True:
                start = time.monotonic()
                try:
                    # Coalesce slider events so expensive video seeks don't accumulate.
                    pending = []
                    while True:
                        try:
                            command = self.commands.get_nowait()
                        except Empty:
                            break
                        if pending and command[0] == pending[-1][0] == "frame":
                            pending[-1] = command
                        else:
                            pending.append(command)
                    for command in pending:
                        self._command(*command)
                    now = time.monotonic()
                    elapsed, self.last_tick = now - self.last_tick, now
                    if self.playing:
                        self.playhead += elapsed * self.speed
                        index = min(int(self.playhead * self.scene.fps), len(self.samples) - 1)
                        if index != self.frame_index:
                            self._draw_frame(index)
                        if index == len(self.samples) - 1:
                            self._pause()
                except Exception as error:
                    self._pause()
                    self.status.content = f"**Unable to display selection:** {error}"
                    logging.exception("Viewer update failed")
                time.sleep(max(0.005, 1 / 15 - (time.monotonic() - start)))
        finally:
            self.videos.close()
            self.server.stop()
