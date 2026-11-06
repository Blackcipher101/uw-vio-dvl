#!/usr/bin/env python3
"""Generate the procedural scene assets (OBJ meshes and PNG textures) into sim/assets/.

Why generated: the Stonefish data directory on the benchmark machine ships no seabed meshes or textures that we could
verify, so the terrain, structures and rocks are written here directly in world NED coordinates (z down) from the
functions in trajectory.py. Only numpy is needed (PNG is written with zlib). Meshes are closed (Stonefish's physics
and the DVL ray casts use them), triangulated, with outward counter-clockwise winding and per-vertex uv/normals.

  python3 gen_assets.py [--out sim/assets]
"""
from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path

import numpy as np

import trajectory as tj

TEX_PERIOD_M = 4.0  # one seabed texture tile covers this many metres (uv = metres / period, repeating)
BOTTOM_Z_M = 34.0  # underside of the terrain slab (below the pit floor)


def write_png(path: Path, img: np.ndarray) -> None:
    """8-bit gray (H,W) or RGB (H,W,3) PNG."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    ctype = 0 if img.ndim == 2 else 2
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def tileable_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Periodic power-law noise (spectral synthesis), zero mean, unit std."""
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.rfftfreq(n)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = 1.0
    spec = (f ** (-beta / 2.0)) * np.exp(2j * np.pi * rng.random(f.shape))
    spec[0, 0] = 0.0
    z = np.fft.irfft2(spec, s=(n, n))
    return (z - z.mean()) / z.std()


def seabed_texture(n: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(3)
    coarse, mid, fine = (tileable_noise(n, b, rng) for b in (3.2, 2.2, 1.2))
    y = np.arange(n)[:, None] / n
    x = np.arange(n)[None, :] / n
    ripples = np.sin(2 * np.pi * (6 * x + 2 * y + 0.25 * coarse))  # periodic: integer cycles per tile
    pebbles = (mid > 1.3).astype(float) * (0.5 + 0.5 * np.tanh(fine))
    shade = 0.55 + 0.12 * coarse + 0.10 * mid + 0.07 * fine + 0.06 * ripples - 0.25 * pebbles
    sand = np.array([0.74, 0.66, 0.48])
    img = np.clip(shade[..., None] * sand[None, None, :] * 1.15, 0, 1)
    return (255 * img).astype(np.uint8)


def rock_texture(n: int = 512) -> np.ndarray:
    rng = np.random.default_rng(4)
    a, b, c = (tileable_noise(n, be, rng) for be in (3.0, 2.0, 1.0))
    shade = np.clip(0.45 + 0.13 * a + 0.10 * b + 0.07 * c, 0, 1)
    col = np.array([0.55, 0.52, 0.47]) + 0.12 * np.stack([a, b, c], -1) * np.array([0.5, 0.4, 0.3])
    return (255 * np.clip(shade[..., None] * col * 1.6, 0, 1)).astype(np.uint8)


def metal_texture(n: int = 512) -> np.ndarray:
    rng = np.random.default_rng(5)
    a, b = tileable_noise(n, 2.5, rng), tileable_noise(n, 1.4, rng)
    rust = _smooth(a - 0.3)
    base = np.array([0.30, 0.38, 0.42])
    rustc = np.array([0.55, 0.28, 0.12])
    img = (1 - rust[..., None]) * base + rust[..., None] * rustc
    img = img * (0.8 + 0.12 * b[..., None])
    yy, xx = np.mgrid[0:n, 0:n]
    img[((xx // 64) + (yy // 64)) % 2 == 0] *= 0.82  # panel checker
    img[(xx % 128) < 3] *= 0.5  # seams
    img[(yy % 128) < 3] *= 0.5
    stripe = ((xx + yy) // 24) % 2 == 0
    img[(yy > 40) & (yy < 90) & stripe] = np.array([0.85, 0.75, 0.1])  # hazard stripe
    return (255 * np.clip(img, 0, 1)).astype(np.uint8)


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Mesh:
    def __init__(self):
        self.v, self.uv, self.f = [], [], []

    def add(self, verts, uvs, faces):
        base = len(self.v)
        self.v.extend(np.asarray(verts, dtype=float))
        self.uv.extend(np.asarray(uvs, dtype=float))
        self.f.extend((np.asarray(faces, dtype=int) + base).tolist())

    def orient_outward(self, ref_center_per_face=None):
        """Flip triangles whose normal points towards the centroid of their own closed component."""
        v = np.asarray(self.v)
        f = np.asarray(self.f)
        n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        ctr = v[f].mean(axis=1)
        ref = ref_center_per_face if ref_center_per_face is not None else v.mean(axis=0)[None, :]
        flip = np.einsum("ij,ij->i", n, ctr - ref) < 0
        f[flip] = f[flip][:, ::-1]
        self.f = f.tolist()

    def write(self, path: Path):
        v = np.asarray(self.v)
        f = np.asarray(self.f)
        n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # area weighted
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, f[:, k], n)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        with open(path, "w") as fh:
            fh.write("# generated by sim/gen_assets.py (world NED, metres)\n")
            for p in v:
                fh.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            for u in self.uv:
                fh.write(f"vt {u[0]:.4f} {u[1]:.4f}\n")
            for q in vn:
                fh.write(f"vn {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}\n")
            for a, b, c in (f + 1):
                fh.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")


def terrain_mesh() -> Mesh:
    x = np.arange(tj.TERRAIN_X[0], tj.TERRAIN_X[1] + 1e-9, tj.TERRAIN_CELL_M)
    y = np.arange(tj.TERRAIN_Y[0], tj.TERRAIN_Y[1] + 1e-9, tj.TERRAIN_CELL_M)
    nx, ny = len(x), len(y)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    gz = tj.terrain_z(gx, gy)
    top_v = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    idx = np.arange(nx * ny).reshape(nx, ny)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    top_f = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    m = Mesh()
    m.add(top_v, top_v[:, :2] / TEX_PERIOD_M, top_f)
    # perimeter (counter-clockwise in x,y), bottom ring + centre fan, side quads
    per = np.concatenate([idx[:, 0], idx[-1, 1:], idx[-2::-1, -1], idx[0, -2:0:-1]])
    bot_v = top_v[per].copy()
    bot_v[:, 2] = BOTTOM_Z_M
    nb = len(per)
    base = len(m.v)
    centre = np.array([[x.mean(), y.mean(), BOTTOM_Z_M]])
    side_f = [[per[i], per[(i + 1) % nb], base + (i + 1) % nb] for i in range(nb)]
    side_f += [[per[i], base + (i + 1) % nb, base + i] for i in range(nb)]
    fan_f = [[base + i, base + (i + 1) % nb, base + nb] for i in range(nb)]
    m.add(np.concatenate([bot_v, centre]), np.concatenate([bot_v, centre])[:, :2] / TEX_PERIOD_M, [])
    # orientation: top faces must have the normal pointing up (-z); the rest outward from the slab centre
    v = np.asarray(m.v)
    top_f = np.asarray(top_f)
    n = np.cross(v[top_f[:, 1]] - v[top_f[:, 0]], v[top_f[:, 2]] - v[top_f[:, 0]])
    top_f[n[:, 2] > 0] = top_f[n[:, 2] > 0][:, ::-1]
    rest = np.asarray(side_f + fan_f)
    ctr = v[rest].mean(axis=1)
    slab_c = np.array([x.mean(), y.mean(), 0.5 * (BOTTOM_Z_M + float(gz.min()))])
    n = np.cross(v[rest[:, 1]] - v[rest[:, 0]], v[rest[:, 2]] - v[rest[:, 0]])
    rest[np.einsum("ij,ij->i", n, ctr - slab_c) < 0] = rest[np.einsum("ij,ij->i", n, ctr - slab_c) < 0][:, ::-1]
    m.f = np.concatenate([top_f, rest]).tolist()
    return m


def box_mesh(m: Mesh, center, size, uv_scale=2.0):
    c = np.asarray(center, float)
    h = np.asarray(size, float) / 2.0
    for axis in range(3):
        for sgn in (-1, 1):
            a1, a2 = [(axis + 1) % 3, (axis + 2) % 3]
            quad = []
            for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = c.copy()
                p[axis] += sgn * h[axis]
                p[a1] += s1 * h[a1]
                p[a2] += s2 * h[a2]
                quad.append(p)
            uv = [((s1 * h[a1] + h[a1]) / uv_scale, (s2 * h[a2] + h[a2]) / uv_scale)
                  for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            v0 = len(m.v)
            m.add(quad, uv, [[0, 1, 2], [0, 2, 3]])
            tri = np.asarray(m.f[-2:])
            out = np.zeros(3)
            out[axis] = sgn
            v = np.asarray(m.v)
            nrm = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
            for k in range(2):
                if nrm[k] @ out < 0:
                    m.f[-2 + k] = tri[k][::-1].tolist()
            assert v0 + 4 == len(m.v)


def cylinder_mesh(m: Mesh, p0, p1, radius, n=32, uv_scale=2.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    length = np.linalg.norm(ax)
    ax /= length
    helper = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(ax, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = [radius * (math.cos(a) * e1 + math.sin(a) * e2) for a in ang]
    verts = [p0 + r for r in ring] + [p1 + r for r in ring] + [p0, p1]
    uvs = [(a * radius / uv_scale, 0.0) for a in ang] + [(a * radius / uv_scale, length / uv_scale) for a in ang] + [(0, 0), (0, 1)]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    base = len(m.f)
    m.add(verts, uvs, faces)
    v = np.asarray(m.v)
    centre = 0.5 * (p0 + p1)
    for k in range(base, len(m.f)):
        t = np.asarray(m.f[k])
        nrm = np.cross(v[t[1]] - v[t[0]], v[t[2]] - v[t[0]])
        if nrm @ (v[t].mean(axis=0) - centre) < 0:
            m.f[k] = t[::-1].tolist()


def icosphere(subdiv: int):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                cache[k] = len(v) - 1
            return cache[k]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f)


def rocks_mesh(objs: tj.SceneObjects) -> Mesh:
    m = Mesh()
    base_v, base_f = icosphere(3)
    for x, y, rad, seed in objs.rocks:
        rng = np.random.default_rng(seed)
        k = rng.normal(size=(6, 3))
        disp = 1.0 + 0.22 * np.sin(base_v @ k[:3].T * 2.3).sum(axis=1) / 3 + 0.08 * np.sin(base_v @ k[3:].T * 7.1).sum(axis=1) / 3
        sc = np.array([1.0, rng.uniform(0.7, 1.2), rng.uniform(0.5, 0.8)]) * rad
        v = base_v * disp[:, None] * sc
        ground = float(tj.terrain_z(x, y))
        v = v + np.array([x, y, ground - 0.45 * sc[2]])
        uv = v[:, :2] / 1.5 + v[:, 2:3] / 3.0
        m.add(v, uv, base_f)
        tri = np.asarray(m.f[-len(base_f):])
        vv = np.asarray(m.v)
        ctr = vv[len(vv) - len(base_v):].mean(axis=0)
        nrm = np.cross(vv[tri[:, 1]] - vv[tri[:, 0]], vv[tri[:, 2]] - vv[tri[:, 0]])
        flip = np.einsum("ij,ij->i", nrm, vv[tri].mean(axis=1) - ctr) < 0
        tri[flip] = tri[flip][:, ::-1]
        m.f[-len(base_f):] = tri.tolist()
    return m


def structure_mesh(objs: tj.SceneObjects) -> Mesh:
    m = Mesh()
    box_mesh(m, objs.container_center, objs.container_size, uv_scale=2.0)
    cylinder_mesh(m, objs.pipe_p0, objs.pipe_p1, objs.pipe_radius_m, 32, uv_scale=2.0)
    px = float(objs.pipe_p0[0])
    pz = float(objs.pipe_p0[2])
    for y in np.linspace(objs.pipe_p0[1] + 1.5, objs.pipe_p1[1] - 1.5, 3):
        g = float(tj.terrain_z(px, y))
        box_mesh(m, [px, y, 0.5 * (pz + g) + 0.15], [0.5, 0.35, g - pz + 0.2], uv_scale=1.0)
    box_mesh(m, [px, 0.5 * (objs.pipe_p0[1] + objs.pipe_p1[1]), pz - 0.45], [0.45, 0.45, 0.45], uv_scale=1.0)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "assets")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    write_png(a.out / "seabed_tex.png", seabed_texture())
    write_png(a.out / "rock_tex.png", rock_texture())
    write_png(a.out / "metal_tex.png", metal_texture())
    objs = tj.scene_objects()
    for name, mesh in (("terrain", terrain_mesh()), ("structures", structure_mesh(objs)), ("rocks", rocks_mesh(objs))):
        mesh.write(a.out / f"{name}.obj")
        print(f"{name}.obj: {len(mesh.v)} vertices, {len(mesh.f)} triangles")
    print(f"assets written to {a.out}")


if __name__ == "__main__":
    main()
