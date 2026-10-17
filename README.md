# KIWI Reference Viewer

A local reference viewer for customer-delivered bimanual UMI demonstrations in LeRobot v3 format. It combines a static Gaussian scene, world-frame TCP trajectories, four RGB views, and normalized gripper openings in one browser window.

## Run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Tested with Python 3.12. The server needs no CUDA, PyTorch, or KIWI installation; the browser uses WebGL.

```sh
git clone https://github.com/lyarobotics/kiwi-reference-viewer.git
cd kiwi-reference-viewer
uv sync --locked
```

If your data is on a private Hugging Face dataset, authenticate with an account/token that has access, then download it. Replace `ORG/DATASET` with the repository you were given:

```sh
uv run hf auth login
uv run kiwi-viser download --repo-id ORG/DATASET --revision v1 --output .data/dataset
```

After downloading (or with an existing local release), inspect and start the viewer:

```sh
uv run kiwi-viser inspect .data/dataset
uv run kiwi-viser view .data/dataset
```

Open **http://127.0.0.1:8080**. A release root, its `lerobot/` directory, or one scene dataset root is accepted. Use `--scene scene_001 --episode 0` to choose a starting episode. `download --scene scene_001` fetches one scene only.

## Explore

- Choose a scene and episode, play/pause, scrub to an exact frame, or change playback speed.
- Cyan marks the left TCP; orange marks the right TCP. Each TCP shows its orientation axes.
- The four-view mosaic follows the selected frame. Video seeking uses the episode table's MP4 shard and start offset, including shared video files.
- Trajectories show the current frame ±2 seconds by default. In the Scene tab, adjust **Window ± seconds** or select **Full episode** under **Trajectory range**. The nearby window follows playback and seeking, stays inside the selected episode, and uses its timestamps.
- Toggle the static environment, trajectories, front-camera axes, and next-state targets. Reset view recenters the work area.
- Gripper bars show normalized opening (0 closed, 1 open). Filled estimates are identified when quality flags are present.
- The Dataset tab explains coordinate conventions and limitations supplied with the release.

The viewer preserves the delivered scene world, metric positions, and Hamilton `xyzw` pose convention. It applies the PLY asset's declared `T_world_asset` transform and converts quaternion order only at the Viser boundary. Different scenes remain separate worlds. It never subtracts the first frame or recalculates TCP calibration.

Playback follows the dataset clock; rendering is capped at 15 updates/s and may skip intermediate frames during playback. Pause and seek to inspect any individual dataset frame. Gaussian scenes load **all splats by default**, preserving their delivered covariance and opacity. `--max-gaussians all` makes this explicit. For a faster, lower-detail preview, use a positive cap such as `--max-gaussians 500000`; this deterministically omits splats and can make fine geometry look sparse or soft. Use `--render points` for a simpler rendering mode.

Viser displays degree-zero/base RGB only; the PLY’s higher-order, view-dependent spherical harmonics are not rendered. Full splat count therefore does not mean identical appearance to a full SH renderer. No extra Gaussian scale or opacity adjustment is applied. On a high-DPI display, `http://127.0.0.1:8080/?fixedDpr=2` fixes the browser render resolution at 2× if adaptive resolution is reducing detail; it costs more GPU work. The full scene can still contain reconstruction blur and the limitations described in the Dataset tab.

Camera axes are symbolic body frames. Accurate camera frusta, undistortion, and physical jaw width require additional calibration information. Action semantics and known reconstruction/calibration limitations are read from the delivered metadata.

## Assessing a sample

Start a tracking review with **Front-camera axes** and the videos. TCP poses add the camera-to-tool transform's uncertainty; a visual match to a reconstructed scene does not establish absolute accuracy. Check the delivered calibration status before interpreting TCP targets as calibrated physical motion.

Automated consumers, including AI tools, should read the dataset README and each scene's `meta/export_provenance.json`, including calibration-status flags, the TCP formula, availability of camera intrinsics, and scene-specific limitations. Coordinate conventions specify how to interpret values; they do not guarantee measurement accuracy.

## Dataset contract

Each scene contains standard LeRobot v3 `meta/info.json`, `meta/episodes/**/*.parquet`, `data/`, and `videos/`, plus customer scene metadata at `meta/export_provenance.json` and a local Gaussian PLY asset. The metadata describes the scene frame, asset transform, state layout, and sampling information. This first version supports the bimanual 16-value pose/opening layout exposed by that contract; it is not a universal LeRobot robot viewer. MCAP is not a viewer input yet.

Private data stays local: `.data/`, `.local/`, media, Parquet, PLY, NPZ, and credentials are excluded from Git. The server binds to `127.0.0.1` by default. `--host` explicitly changes that. A server is one shared playback session; multiple browser tabs share its controls.

## Development

```sh
uv run pytest
```

Tests use synthetic fixtures. No customer data or access token is needed.
