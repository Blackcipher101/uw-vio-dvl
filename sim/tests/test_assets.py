"""Generated meshes are closed, outward-oriented triangle meshes (Stonefish physics and the DVL ray casts need that)."""
from collections import Counter

import numpy as np
import pytest

import gen_assets as ga
import trajectory as tj


def _check_closed_outward(mesh: ga.Mesh):
    v = np.round(np.asarray(mesh.v), 5)  # weld duplicated vertices (box faces carry their own uv)
    f = np.asarray(mesh.f)
    ids = {}
    welded = np.array([ids.setdefault(tuple(p), len(ids)) for p in v])
    edges = Counter()
    for a, b, c in welded[f]:
        for e in ((a, b), (b, c), (c, a)):
            edges[e] += 1
    assert all(n == 1 and edges.get((e[1], e[0]), 0) == 1 for e, n in edges.items()), "open or inconsistently wound"
    vol = np.einsum("ij,ij->", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])) / 6.0
    assert vol > 0, "triangles wound inward"


def test_terrain_closed_and_top_faces_up():
    m = ga.terrain_mesh()
    _check_closed_outward(m)
    v, f = np.asarray(m.v), np.asarray(m.f)
    top = f[: 2 * (len(np.arange(tj.TERRAIN_X[0], tj.TERRAIN_X[1] + 1e-9, tj.TERRAIN_CELL_M)) - 1)
            * (len(np.arange(tj.TERRAIN_Y[0], tj.TERRAIN_Y[1] + 1e-9, tj.TERRAIN_CELL_M)) - 1)]
    n = np.cross(v[top[:, 1]] - v[top[:, 0]], v[top[:, 2]] - v[top[:, 0]])
    assert np.all(n[:, 2] < 0), "seabed surface normals must point up (-z in NED)"


def test_structures_and_rocks_closed():
    objs = tj.scene_objects()
    _check_closed_outward(ga.structure_mesh(objs))
    _check_closed_outward(ga.rocks_mesh(objs))


def test_terrain_mesh_matches_function_and_pit_is_deep():
    m = ga.terrain_mesh()
    v = np.asarray(m.v)[: 10 ** 9]
    sel = v[np.abs(v[:, 2] - tj.terrain_z(v[:, 0], v[:, 1])) < 1e-3]
    assert len(sel) > 20000
    assert tj.terrain_z(*tj.PIT_C) == pytest.approx(tj.PIT_FLOOR_Z_M, abs=1e-6)
    assert tj.terrain_z(30.0, 5.0) == pytest.approx(tj.SHELF_Z_M, abs=1.0)


def test_png_roundtrip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    img = (np.arange(64 * 48 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
    ga.write_png(tmp_path / "a.png", img)
    back = cv2.imread(str(tmp_path / "a.png"), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(back[..., ::-1], img)
    g = img[..., 0]
    ga.write_png(tmp_path / "g.png", g)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / "g.png"), cv2.IMREAD_GRAYSCALE), g)


def test_textures_are_contrasty():
    for tex in (ga.seabed_texture(256), ga.rock_texture(128), ga.metal_texture(128)):
        assert tex.shape[-1] == 3 and tex.std() > 15
