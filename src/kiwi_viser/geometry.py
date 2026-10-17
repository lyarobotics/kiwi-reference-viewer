"""Customer PLY geometry, without imports from the capture/processing projects.

The 3DGS PLY convention stores logarithmic axis scales, opacity logits, a
Hamilton wxyz rotation, and degree-zero SH color coefficients. Viser accepts
covariances and RGB colors, so higher-order view-dependent SH terms are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from plyfile import PlyData
from scipy.special import expit
from scipy.spatial.transform import Rotation

DEFAULT_MAX_GAUSSIANS: int | None = None
_DECODE_BATCH_SIZE = 65_536
SH_C0 = 0.28209479177387814
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class GaussianScene:
    """Sampled scene in the declared world frame, ready for Viser.

    ``bounds`` is a (2, 3) array of minimum/maximum retained centers; it does
    not include the infinite support of Gaussian densities. ``source_count``
    counts the original PLY vertices before deterministic sampling.
    """

    centers: FloatArray
    covariances: FloatArray
    rgbs: FloatArray
    opacities: FloatArray
    bounds: FloatArray
    source_count: int
    source_path: Path

    @property
    def count(self) -> int:
        return len(self.centers)

    @property
    def point_colors(self) -> NDArray[np.uint8]:
        """RGB colors usable with ``scene.add_point_cloud`` as a fallback."""
        return np.rint(self.rgbs * 255).astype(np.uint8)


@dataclass(frozen=True)
class CameraView:
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)


def xyzw_to_wxyz(quaternion: ArrayLike) -> np.ndarray:
    """Reorder Hamilton body-to-world quaternions for Viser; no axis change."""
    q = np.asarray(quaternion)
    if q.ndim == 0 or q.shape[-1] != 4:
        raise ValueError("A quaternion must have a final dimension of four (xyzw).")
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(quaternion: ArrayLike) -> np.ndarray:
    q = np.asarray(quaternion)
    if q.ndim == 0 or q.shape[-1] != 4:
        raise ValueError("A quaternion must have a final dimension of four (wxyz).")
    return q[..., [1, 2, 3, 0]]


def _affine_transform(value: ArrayLike | None) -> np.ndarray:
    transform = np.eye(4) if value is None else np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_world_asset must be a finite 4×4 matrix.")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("T_world_asset must use homogeneous column vectors.")
    if abs(np.linalg.det(transform[:3, :3])) < 1e-12:
        raise ValueError("T_world_asset must have an invertible linear part.")
    return transform


def transform_points(points: ArrayLike, transform: ArrayLike) -> np.ndarray:
    """Apply a declared homogeneous transform to points of shape (..., 3)."""
    points_array = np.asarray(points)
    if points_array.ndim == 0 or points_array.shape[-1] != 3:
        raise ValueError("Points must have a final dimension of three.")
    matrix = _affine_transform(transform)
    return points_array @ matrix[:3, :3].T + matrix[:3, 3]


def load_gaussian_scene(
    path: str | Path,
    *,
    T_world_asset: ArrayLike | None = None,
    max_gaussians: int | None = DEFAULT_MAX_GAUSSIANS,
) -> GaussianScene:
    """Read the reference 3DGS PLY and apply its declared asset transform.

    None loads every Gaussian. A positive cap selects a deterministic display
    sample. Binary PLY is memory-mapped and parameter decoding is batched so
    temporary float64 covariance arrays stay bounded even in full-scene mode.
    The delivered data is never rewritten. ASCII PLY is not supported.
    """
    asset_path = Path(path).expanduser()
    if asset_path.suffix.lower() != ".ply":
        raise ValueError(
            f"Unsupported scene format {asset_path.suffix!r}. Use the delivered "
            "reference .ply file; this viewer does not decode SPZ."
        )
    if max_gaussians is not None and (
        isinstance(max_gaussians, bool) or not isinstance(max_gaussians, int) or max_gaussians < 1
    ):
        raise ValueError("max_gaussians must be None (all) or a positive integer.")
    transform = _affine_transform(T_world_asset)
    # Inspect only the format line before plyfile can allocate an ASCII file.
    with asset_path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {asset_path}")
        format_line = stream.readline().strip()
    if not format_line.startswith((b"format binary_little_endian", b"format binary_big_endian")):
        raise ValueError("A binary 3DGS PLY is required for bounded-memory scene loading.")
    ply = PlyData.read(str(asset_path), mmap="r")
    if "vertex" not in ply:
        raise ValueError("The PLY has no vertex element.")
    vertices = ply["vertex"].data
    required = (
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )
    missing = set(required).difference(vertices.dtype.names or ())
    if missing:
        raise ValueError(f"Not a supported 3DGS PLY; missing properties: {', '.join(sorted(missing))}")
    source_count = len(vertices)
    if source_count == 0:
        raise ValueError("The scene PLY contains no Gaussians.")
    count = source_count if max_gaussians is None else min(source_count, max_gaussians)
    indices = None if count == source_count else np.linspace(0, source_count - 1, count, dtype=np.int64)
    centers32 = np.empty((count, 3), dtype=np.float32)
    covariances32 = np.empty((count, 3, 3), dtype=np.float32)
    rgbs = np.empty((count, 3), dtype=np.float32)
    opacities = np.empty((count, 1), dtype=np.float32)
    for begin in range(0, count, _DECODE_BATCH_SIZE):
        end = min(begin + _DECODE_BATCH_SIZE, count)
        selected = slice(begin, end) if indices is None else indices[begin:end]

        def columns(names: tuple[str, ...]) -> np.ndarray:
            return np.column_stack([vertices[name][selected] for name in names]).astype(np.float64)

        centers = columns(("x", "y", "z"))
        dc = columns(("f_dc_0", "f_dc_1", "f_dc_2"))
        logits = columns(("opacity",))
        log_scales = columns(("scale_0", "scale_1", "scale_2"))
        rotations_wxyz = columns(("rot_0", "rot_1", "rot_2", "rot_3"))
        for label, values in (
            ("centers", centers), ("colors", dc), ("opacity logits", logits),
            ("log scales", log_scales), ("rotations", rotations_wxyz),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"Scene contains non-finite {label} in the display selection.")
        if np.any(np.linalg.norm(rotations_wxyz, axis=1) < 1e-12):
            raise ValueError("Scene contains a zero-length Gaussian rotation quaternion.")
        rotations = Rotation.from_quat(wxyz_to_xyzw(rotations_wxyz)).as_matrix()
        # R diag(exp(log_scale)) is the Gaussian's oriented axis matrix.
        with np.errstate(over="ignore", invalid="ignore"):
            axes = rotations * np.exp(log_scales)[:, None, :]
            world_axes = transform[:3, :3] @ axes
            covariances32[begin:end] = world_axes @ world_axes.transpose(0, 2, 1)
            centers32[begin:end] = transform_points(centers, transform)
        if not np.isfinite(covariances32[begin:end]).all() or not np.isfinite(centers32[begin:end]).all():
            raise ValueError("Transformed scene geometry exceeds finite float32 coordinates/covariances.")
        rgbs[begin:end] = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
        opacities[begin:end] = expit(logits)
    return GaussianScene(
        centers=centers32, covariances=covariances32, rgbs=rgbs, opacities=opacities,
        bounds=np.stack([centers32.min(axis=0), centers32.max(axis=0)]),
        source_count=source_count,
        source_path=asset_path.resolve(),
    )


def load_scene_from_metadata(
    scene_root: str | Path,
    provenance: Mapping[str, Any],
    *,
    max_gaussians: int | None = DEFAULT_MAX_GAUSSIANS,
) -> GaussianScene:
    """Resolve a delivered PLY relative to a LeRobot scene dataset root.

    Accepts either the complete ``export_provenance.json`` object or its
    ``environment`` object. The PLY's own transform is mandatory and applied
    exactly once. Files outside the selected dataset root are rejected.
    """
    root = Path(scene_root).expanduser().resolve()
    environment = provenance.get("environment", provenance)
    candidates = [
        entry for entry in environment.get("files", [])
        if str(entry.get("format", "")).upper() == "PLY"
        or str(entry.get("local_path", entry.get("path", ""))).lower().endswith(".ply")
    ]
    if not candidates:
        raise ValueError("Scene metadata has no reference PLY; SPZ decoding is not supported.")
    missing_paths: list[str] = []
    for entry in candidates:
        relative_path = entry.get("local_path", entry.get("path"))
        if not relative_path:
            continue
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Scene asset path must stay inside its delivered dataset root.")
        if not path.is_file():
            missing_paths.append(str(path))
            continue
        if "T_world_asset" not in entry:
            raise ValueError("PLY metadata is missing its explicit T_world_asset transform.")
        return load_gaussian_scene(
            path, T_world_asset=entry["T_world_asset"], max_gaussians=max_gaussians,
        )
    raise FileNotFoundError("The scene PLY has not been downloaded: " + ", ".join(missing_paths))


def suggest_camera(
    scene_bounds: ArrayLike | None,
    trajectory_positions: ArrayLike | None = None,
    *,
    scene_centers: ArrayLike | None = None,
    scene_opacities: ArrayLike | None = None,
) -> CameraView:
    """Frame the task workspace with +Z up, avoiding distant scene outliers.

    Trajectory positions may have shape (N, 3) or (N, 2, 3). With a trajectory,
    use its robust task extent and add room for the surrounding environment.
    Geometry bounds provide the fallback if no valid trajectory is available.
    When scene centers are supplied, choose among twelve elevated viewpoints
    using at most 20,000 deterministic samples. Opacity-weighted sightline and
    camera-proximity penalties prefer open space over walls. This is a framing
    heuristic for a partial reconstruction, not a collision-free camera planner.
    """
    positions = np.empty((0, 3), dtype=np.float64)
    if trajectory_positions is not None:
        raw = np.asarray(trajectory_positions, dtype=np.float64)
        if raw.ndim == 0 or raw.shape[-1] != 3:
            raise ValueError("Trajectory positions must have a final dimension of three.")
        positions = raw.reshape(-1, 3)
        positions = positions[np.isfinite(positions).all(axis=1)]
    if len(positions):
        low, high = np.quantile(positions, [0.02, 0.98], axis=0)
        center = (low + high) / 2
        radius = max(float(np.linalg.norm(high - low)) * 0.65, 0.75)
    elif scene_bounds is not None:
        bounds = np.asarray(scene_bounds, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
            raise ValueError("Scene bounds must be finite minimum/maximum corners (2, 3).")
        center = bounds.mean(axis=0)
        radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 0.55, 0.75)
    else:
        center = np.zeros(3)
        radius = 1.0
    position = center + np.array([1.3, -1.6, 1.0]) * radius
    if scene_centers is not None:
        raw_centers = np.asarray(scene_centers)
        if raw_centers.ndim != 2 or raw_centers.shape[1] != 3:
            raise ValueError("Scene centers must have shape (N, 3).")
        count = len(raw_centers)
        selected = np.linspace(0, count - 1, min(count, 20_000), dtype=np.int64)
        points = np.asarray(raw_centers[selected], dtype=np.float64)
        if scene_opacities is None:
            weights = np.ones(len(points), dtype=np.float64)
        else:
            raw_weights = np.asarray(scene_opacities)
            if raw_weights.shape not in {(count,), (count, 1)}:
                raise ValueError("Scene opacities must have shape (N,) or (N, 1).")
            weights = np.asarray(raw_weights.reshape(-1)[selected], dtype=np.float64)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(weights)
        points = points[valid]
        weights = np.clip(weights[valid], 0.0, 1.0)
        if len(points) and np.any(weights > 0):
            relative = points - center
            distance = float(np.linalg.norm(position - center))
            # The default direction breaks ties, followed by a fixed circle.
            # Candidate distances retain the task-extent-based framing above.
            angles = np.deg2rad((np.arange(12) * 30 + 300) % 360)
            directions = np.column_stack((np.cos(angles), np.sin(angles), np.full(12, 0.75)))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            corridor_radius = np.clip(radius * 0.45, 0.25, 0.55)
            camera_clearance = np.clip(radius * 0.30, 0.20, 0.40)
            task_clearance = np.clip(radius * 0.45, 0.25, 0.50)
            best_score = float("inf")
            for direction in directions:
                depth = relative @ direction
                transverse = np.linalg.norm(relative - depth[:, None] * direction, axis=1)
                # Ignore the task itself; favor directions with little geometry
                # between the viewer and the workspace, including its edges.
                in_sightline = (depth > task_clearance) & (depth < distance)
                path_penalty = np.exp(-0.5 * (transverse / corridor_radius) ** 2) * in_sightline
                candidate = center + direction * distance
                near_camera = np.linalg.norm(points - candidate, axis=1)
                # Include geometry just behind and beside the eye, too: a clear
                # center ray alone can still put the camera into a wall.
                proximity_penalty = 4.0 * np.exp(-0.5 * (near_camera / camera_clearance) ** 2)
                score = float(weights @ (path_penalty + proximity_penalty))
                if score < best_score:
                    best_score = score
                    position = candidate
    return CameraView(tuple(float(x) for x in position), tuple(float(x) for x in center))
