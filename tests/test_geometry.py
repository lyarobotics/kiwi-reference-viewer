import numpy as np
import pytest
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from kiwi_viser.geometry import (
    SH_C0,
    load_gaussian_scene,
    load_scene_from_metadata,
    suggest_camera,
    xyzw_to_wxyz,
)


def write_gaussians(path, *, count=1, rotation=None, scales=(1, 2, 3)):
    names = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    vertices = np.zeros(count, dtype=[(name, "f4") for name in names])
    vertices["x"] = np.arange(count) + 1
    vertices["y"] = 2
    vertices["z"] = 3
    vertices["f_dc_0"] = (0.1 - 0.5) / SH_C0
    vertices["f_dc_1"] = 0
    vertices["f_dc_2"] = (0.9 - 0.5) / SH_C0
    vertices["opacity"] = np.log(0.8 / 0.2)
    for axis, scale in enumerate(scales):
        vertices[f"scale_{axis}"] = np.log(scale)
    quat = np.asarray([1, 0, 0, 0] if rotation is None else rotation)
    for axis in range(4):
        vertices[f"rot_{axis}"] = quat[axis]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], byte_order="<").write(path)


def test_gaussian_parameters_and_world_transform(tmp_path):
    # Anisotropic axes rotated 90° around Z; the asset then rotates 90° around X.
    asset_rotation = Rotation.from_euler("z", 90, degrees=True)
    path = tmp_path / "scene.ply"
    write_gaussians(path, rotation=xyzw_to_wxyz(asset_rotation.as_quat()))
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    transform[:3, 3] = [10, 20, 30]
    scene = load_gaussian_scene(path, T_world_asset=transform)
    np.testing.assert_allclose(scene.centers, [[11, 17, 32]], atol=1e-6)
    np.testing.assert_allclose(scene.covariances[0], np.diag([4, 9, 1]), atol=1e-6)
    np.testing.assert_allclose(scene.rgbs, [[0.1, 0.5, 0.9]], atol=1e-7)
    np.testing.assert_allclose(scene.opacities, [[0.8]], atol=1e-7)
    np.testing.assert_allclose(scene.bounds, [[11, 17, 32], [11, 17, 32]])
    assert scene.centers.dtype == np.float32
    assert scene.point_colors.dtype == np.uint8


def test_sampling_is_deterministic_and_bounded(tmp_path):
    path = tmp_path / "scene.ply"
    write_gaussians(path, count=103)
    first = load_gaussian_scene(path, max_gaussians=10)
    second = load_gaussian_scene(path, max_gaussians=10)
    assert first.count == 10
    assert first.source_count == 103
    np.testing.assert_array_equal(first.centers, second.centers)
    assert first.centers[0, 0] == 1
    assert first.centers[-1, 0] == 103
    assert load_gaussian_scene(path, max_gaussians=200).count == 103


def test_metadata_uses_ply_specific_transform_and_local_path(tmp_path):
    path = tmp_path / "scenes" / "scene_001" / "scene.ply"
    write_gaussians(path)
    matrix = np.eye(4)
    matrix[:3, 3] = [4, 5, 6]
    provenance = {"environment": {"files": [
        {"format": "SPZ", "local_path": "scene.spz", "T_world_asset": np.eye(4).tolist()},
        {"format": "PLY", "local_path": "scenes/scene_001/scene.ply", "T_world_asset": matrix.tolist()},
    ]}}
    scene = load_scene_from_metadata(tmp_path, provenance)
    np.testing.assert_allclose(scene.centers, [[5, 7, 9]])
    assert scene.source_path == path


def test_unsupported_and_unsafe_assets_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="does not decode SPZ"):
        load_gaussian_scene(tmp_path / "scene.spz")
    with pytest.raises(ValueError, match="inside its delivered dataset root"):
        load_scene_from_metadata(tmp_path, {"files": [{"format": "PLY", "local_path": "../outside.ply"}]})
    with pytest.raises(FileNotFoundError, match="has not been downloaded"):
        load_scene_from_metadata(tmp_path, {"files": [{"format": "PLY", "local_path": "missing.ply"}]})


def test_quaternion_conversion_does_not_rotate_or_normalize():
    source = np.array([[1, 2, 3, 4], [0, 0, 0, 1]])
    np.testing.assert_array_equal(xyzw_to_wxyz(source), [[4, 1, 2, 3], [1, 0, 0, 0]])
    np.testing.assert_array_equal(source[0], [1, 2, 3, 4])


def test_camera_prioritizes_task_over_distant_geometry():
    camera = suggest_camera([[-100, -100, -100], [100, 100, 100]], [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_allclose(camera.look_at, [1.5, 2.5, 3.5])
    assert np.linalg.norm(np.asarray(camera.position) - camera.look_at) < 5
    assert camera.up_direction == (0.0, 0.0, 1.0)


def test_camera_avoids_opaque_wall_and_remains_reproducible():
    # A tall wall east of the workspace should favor a western viewpoint.
    y, z = np.meshgrid(np.linspace(-2, 2, 65), np.linspace(0, 2, 50))
    wall = np.column_stack((np.full(y.size, 0.55), y.ravel(), z.ravel()))
    positions = [[-0.1, -0.1, 0], [0.1, 0.1, 0]]
    kwargs = dict(scene_centers=wall, scene_opacities=np.ones((len(wall), 1)))
    first = suggest_camera([[-2, -2, 0], [2, 2, 2]], positions, **kwargs)
    second = suggest_camera([[-2, -2, 0], [2, 2, 2]], positions, **kwargs)
    assert first == second
    assert first.position[0] < -0.5
    assert np.isfinite(first.position).all()
    np.testing.assert_allclose(first.look_at, [0, 0, 0])
    assert first.up_direction == (0.0, 0.0, 1.0)


def test_camera_proximity_penalty_avoids_surface_at_eye():
    # Put a cluster just behind the first candidate. It lies outside the center
    # sightline interval but must still prevent placing the eye on that surface.
    radius = 0.75 * np.linalg.norm([1.3, -1.6, 1])
    direction = np.array([np.cos(np.deg2rad(300)), np.sin(np.deg2rad(300)), 0.75])
    direction /= np.linalg.norm(direction)
    surface = np.tile(direction * (radius + 0.03), (100, 1))
    camera = suggest_camera(None, [[0, 0, 0]], scene_centers=surface)
    assert np.linalg.norm(np.asarray(camera.position) - surface[0]) > 0.5


def test_camera_empty_invisible_or_nonfinite_geometry_has_finite_fallback():
    original = suggest_camera(None, [[0, 0, 0]])
    assert suggest_camera(None, [[0, 0, 0]], scene_centers=np.empty((0, 3))) == original
    assert suggest_camera(None, [[0, 0, 0]], scene_centers=[[np.nan, 0, 0]]) == original
    assert suggest_camera(None, [[0, 0, 0]], scene_centers=[[1, 2, 3]], scene_opacities=[0]) == original
    assert np.isfinite(original.position).all()
    with pytest.raises(ValueError, match="Scene opacities"):
        suggest_camera(None, scene_centers=[[1, 2, 3]], scene_opacities=[1, 2])


def test_full_scene_retains_every_vertex_across_decode_batches(tmp_path, monkeypatch):
    import kiwi_viser.geometry as geometry
    path = tmp_path / "scene.ply"
    write_gaussians(path, count=103)
    monkeypatch.setattr(geometry, "_DECODE_BATCH_SIZE", 17)
    full = load_gaussian_scene(path)
    explicit = load_gaussian_scene(path, max_gaussians=None)
    capped = load_gaussian_scene(path, max_gaussians=103)
    assert full.count == full.source_count == 103
    np.testing.assert_array_equal(full.centers[:, 0], np.arange(1, 104))
    for key in ("centers", "covariances", "rgbs", "opacities"):
        np.testing.assert_array_equal(getattr(full, key), getattr(explicit, key))
        np.testing.assert_array_equal(getattr(full, key), getattr(capped, key))
    np.testing.assert_allclose(full.covariances, np.broadcast_to(np.diag([1, 4, 9]), (103, 3, 3)), atol=1e-6)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            load_gaussian_scene(path, max_gaussians=invalid)
