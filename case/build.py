"""
Сборка корпуса: STL для печати, GLB для просмотра, отчёт проверок.

    python3 case/build.py            # всё
    python3 case/build.py --no-glb   # только STL и проверки

Параметры правятся в scanner_case.Params или флагом --set имя=значение
(можно несколько раз), например:  --set top_h=31 --set btn_from_corner=11.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, fields

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import scanner_case as sc  # noqa: E402
from manifold3d import Manifold  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DENSITY = {"PLA": 1.24, "PETG": 1.27, "ASA": 1.07}

COLORS = {
    "bottom": ([0.055, 0.058, 0.062, 1.0], 0.0, 0.55),
    "ring": ([0.55, 0.38, 0.22, 1.0], 0.55, 0.38),
    "lid": ([0.50, 0.52, 0.54, 1.0], 0.25, 0.45),
    "pcb": ([0.10, 0.36, 0.72, 1.0], 0.1, 0.5),
    "heatsink": ([0.74, 0.76, 0.78, 1.0], 0.8, 0.35),
    "fan": ([0.12, 0.12, 0.13, 1.0], 0.2, 0.5),
    "connectors": ([0.70, 0.72, 0.74, 1.0], 0.9, 0.3),
    "nvme": ([0.08, 0.08, 0.09, 1.0], 0.2, 0.5),
    "button": ([0.05, 0.05, 0.05, 1.0], 0.1, 0.5),
}


def to_arrays(m: Manifold):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), np.asarray(mesh.tri_verts).astype(np.int64)


def write_stl(m: Manifold, path):
    v, f = to_arrays(m)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    data = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data["n"] = n
    data["v"] = tri
    with open(path, "wb") as fh:
        fh.write(b"scanner case".ljust(80, b" "))
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(data.tobytes())


def print_pose(name, m: Manifold):
    """Положение для печати: низ — дном на стол, крышка — верхом на стол."""
    if name == "lid":
        m = m.rotate([180, 0, 0])
    bb = m.bounding_box()
    return m.translate([-(bb[0] + bb[3]) / 2, -(bb[1] + bb[4]) / 2, -bb[2]])


def overhang_report(m: Manifold, limit_deg=45.0, bed_tol=0.25):
    """Площадь граней, которые смотрят вниз круче limit_deg от вертикали (кроме стола)."""
    v, f = to_arrays(m)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    nz = n[:, 2] / np.maximum(2 * area, 1e-12)
    zmin = v[:, 2].min()
    on_bed = tri[:, :, 2].max(axis=1) < zmin + bed_tol
    thr = -math.cos(math.radians(90 - limit_deg))  # грань вниз круче limit -> nz < -sin(45°)
    bad = (nz < thr - 1e-3) & ~on_bed & (area > 1e-3)
    # короткие «мосты» (горизонтальные потолки шириной до ~3 мм) отделяем отдельно
    flat = bad & (nz < -0.999)
    return {
        "overhang_mm2": round(float(area[bad].sum()), 1),
        "flat_ceiling_mm2": round(float(area[flat].sum()), 1),
        "worst_deg_from_vertical": round(float(math.degrees(math.acos(max(-1.0, min(1.0, -nz[~on_bed].min())))) if (~on_bed).any() else 0.0), 1),
        "bad_tri": int(bad.sum()),
        "bad_centroids": tri[bad].mean(axis=1)[:12].round(1).tolist(),
    }


def checks(p: sc.Params, parts):
    rep = {}
    assembled = parts["bottom"] + parts["lid"]
    zones = sc.keepout(p)
    clash = {}
    clamp = Manifold.batch_boolean([
        Manifold.cylinder(p.pcb_t + 1, p.post_r + 0.01, p.post_r + 0.01, 32).translate(
            [sx * (p.bx2 - p.hole_inset), sy * (p.by2 - p.hole_inset), 0])
        for sx in (-1, 1) for sy in (-1, 1)], sc.OpType.Add)
    for name, z in zones.items():
        vol = ((assembled - clamp) ^ z).volume()
        clash[name] = round(vol, 2)
    rep["пересечения_с_платой_мм3"] = clash
    rep["натяг_прижимных_стоек_мм3"] = round((parts["lid"] ^ clamp ^ zones["плата"]).volume(), 2)
    # путь установки платы сверху в нижнюю часть
    bx, by = p.bx2, p.by2
    hw = bx - p.conn_corner
    path = sc.box(-bx, bx, -by, by, 0, p.z_part + 1) + \
        sc.box(-hw, hw, -by - p.conn_overhang, by, p.pcb_t, p.z_part + 1) + \
        sc.box(-bx - p.btn_protrude, -bx, p.btn_y - 1.8, p.btn_y + 1.8, p.pcb_t, p.z_part + 1)
    rep["путь_установки_платы_мм3"] = round((parts["bottom"] ^ path).volume(), 2)
    rep["зазор_низ_крышка_мм"] = round(parts["bottom"].min_gap(parts["lid"], 2.0), 3)
    rep["пересечение_низ_крышка_мм3"] = round((parts["bottom"] ^ parts["lid"]).volume(), 3)
    # толкатель
    btn_face = -(bx + p.btn_protrude)
    btn_box = sc.box(btn_face - 0.01, btn_face, p.btn_y - 1.8, p.btn_y + 1.8, p.pcb_t, p.pcb_t + 2 * p.btn_z)
    rep["зазор_толкатель_кнопка_мм"] = round(parts["bottom"].min_gap(btn_box, 3.0), 3)
    # масса
    mass = {}
    for k, m in parts.items():
        mass[k] = {mat: round(m.volume() / 1000 * d, 1) for mat, d in DENSITY.items()}
    rep["масса_г_при_100%"] = mass
    rep["объём_см3"] = {k: round(m.volume() / 1000, 1) for k, m in parts.items()}
    # печатаемость
    rep["нависания"] = {k: overhang_report(print_pose(k, m)) for k, m in parts.items()}
    rep["габарит_мм"] = [round(v, 1) for v in np.subtract(assembled.bounding_box()[3:], assembled.bounding_box()[:3])]
    rep["печатный_габарит_мм"] = {k: [round(v, 1) for v in np.subtract(print_pose(k, m).bounding_box()[3:],
                                                                         print_pose(k, m).bounding_box()[:3])]
                                  for k, m in parts.items()}
    return rep


def write_glb(p: sc.Params, parts, path):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    scene = trimesh.Scene()
    # кольцо-акцент — верх нижней части, печатается другим цветом: показываем отдельно
    zr = p.z_part - p.ring_h
    big = 400.0
    allparts = {"bottom": parts["bottom"] ^ sc.box(-big, big, -big, big, -big, zr),
                "ring": parts["bottom"] ^ sc.box(-big, big, -big, big, zr, big),
                "lid": parts["lid"]}
    allparts.update(sc.board_dummy(p))
    for name, m in allparts.items():
        v, f = to_arrays(m)
        tm = trimesh.Trimesh(v, f, process=False)
        col, metal, rough = COLORS.get(name, ([0.6, 0.6, 0.6, 1], 0.0, 0.6))
        tm.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(name=name, baseColorFactor=col, metallicFactor=metal, roughnessFactor=rough))
        scene.add_geometry(tm, node_name=name, geom_name=name)
    with open(path, "wb") as fh:
        fh.write(scene.export(file_type="glb"))


def parse_set(items):
    p = sc.Params()
    types = {f.name: f.type for f in fields(sc.Params)}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in types:
            raise SystemExit(f"нет параметра {k}")
        cur = getattr(p, k)
        if isinstance(cur, tuple):
            val = tuple(float(x) for x in v.split(","))
        elif isinstance(cur, int):
            val = int(v)
        else:
            val = float(v)
        setattr(p, k, val)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append")
    ap.add_argument("--no-glb", action="store_true")
    ap.add_argument("--out", default=ROOT)
    a = ap.parse_args()
    p = parse_set(a.set)
    parts = sc.build(p)

    stl_dir = os.path.join(a.out, "stl")
    os.makedirs(stl_dir, exist_ok=True)
    names = {"bottom": "scanner_case_bottom.stl", "lid": "scanner_case_lid.stl"}
    for k, m in parts.items():
        write_stl(print_pose(k, m), os.path.join(stl_dir, names[k]))
    if not a.no_glb:
        vdir = os.path.join(a.out, "docs", "viewer")
        os.makedirs(vdir, exist_ok=True)
        write_glb(p, parts, os.path.join(vdir, "scanner_case.glb"))

    rep = checks(p, parts)
    rep["вентиляция"] = getattr(sc.build, "stats", {})
    rep["смена_цвета_для_кольца_мм_от_стола"] = round(p.z_part - p.ring_h - p.z_bot, 2)
    rep["параметры"] = asdict(p)
    with open(os.path.join(a.out, "docs", "report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    show = {k: v for k, v in rep.items() if k != "параметры"}
    print(json.dumps(show, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
