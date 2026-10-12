"""
Сборка корпуса: STL для печати, GLB для просмотра, отчёт проверок.

    python3 case/build.py            # всё
    python3 case/build.py --no-glb   # только STL, проект P1S и проверки

Параметры правятся в scanner_case.Params или флагом --set имя=значение
(можно несколько раз), например:  --set top_h=31 --set btn_from_corner=11.5
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from dataclasses import asdict, fields

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import scanner_case as sc  # noqa: E402
import scanner_rig as rg  # noqa: E402
import bambu_project as bp  # noqa: E402
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
    "handle": ([0.10, 0.105, 0.11, 1.0], 0.0, 0.85),
    "arm": ([0.22, 0.23, 0.24, 1.0], 0.1, 0.6),
    "plate": ([0.22, 0.23, 0.24, 1.0], 0.1, 0.6),
    "lidar": ([0.03, 0.03, 0.035, 1.0], 0.3, 0.25),
    "d455": ([0.70, 0.72, 0.74, 1.0], 0.85, 0.3),
    "d405": ([0.70, 0.72, 0.74, 1.0], 0.85, 0.3),
    "maixcam": ([0.05, 0.05, 0.06, 1.0], 0.3, 0.3),
}

STL_NAMES = {"bottom": "scanner_case_bottom.stl", "ring": "scanner_case_ring.stl", "lid": "scanner_case_lid.stl",
             "handle": "scanner_handle.stl", "arm": "scanner_arm.stl", "plate": "scanner_sensor_plate.stl"}


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
    """Положение для печати: низ — дном на стол, крышка — верхом на стол,
    кронштейн — на боку, площадка — тылом на стол, рукоять — стоя."""
    if name == "lid":
        m = m.rotate([180, 0, 0])
    elif name == "arm":
        m = m.rotate([90, 0, 0])
    elif name == "plate":
        m = m.rotate([0, -90, 0])
    bb = m.bounding_box()
    return m.translate([-(bb[0] + bb[3]) / 2, -(bb[1] + bb[4]) / 2, -bb[2]])


def overhang_report(m: Manifold, limit_deg=45.0, bed_tol=0.25, bridge_zone=None):
    """Площадь граней, которые смотрят вниз круче limit_deg от вертикали (кроме стола).

    bridge_zone(centroids) -> bool-маска: такие грани — потолки прорезей букв,
    их считаем мостами отдельно и меряем самый длинный пролёт.
    """
    v, f = to_arrays(m)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    nz = n[:, 2] / np.maximum(2 * area, 1e-12)
    zmin = v[:, 2].min()
    on_bed = tri[:, :, 2].max(axis=1) < zmin + bed_tol
    thr = -math.cos(math.radians(90 - limit_deg))  # грань вниз круче limit -> nz < -sin(45°)
    bad = (nz < thr - 1e-3) & ~on_bed & (area > 1e-3)
    out = {}
    if bridge_zone is not None:
        zone = bridge_zone(tri.mean(axis=1))
        br = bad & zone
        bad = bad & ~zone
        runs = []
        tb = tri[br]
        for zl in np.unique(np.round(tb[:, :, 2].mean(1), 1)):
            sel = np.abs(tb[:, :, 2].mean(1) - zl) < 0.06
            iv = sorted(zip(tb[sel][:, :, 0].min(1), tb[sel][:, :, 0].max(1)))
            cur = list(iv[0])
            for a, c in iv[1:]:
                if a <= cur[1] + 0.3:
                    cur[1] = max(cur[1], c)
                else:
                    runs.append(cur[1] - cur[0])
                    cur = [a, c]
            runs.append(cur[1] - cur[0])
        out = {"мосты_в_буквах_мм2": round(float(area[br].sum()), 1),
               "самый_длинный_мост_мм": round(float(max(runs)) if runs else 0.0, 1)}
    flat = bad & (nz < -0.999)
    return out | {
        "overhang_mm2": round(float(area[bad].sum()), 1),
        "flat_ceiling_mm2": round(float(area[flat].sum()), 1),
        "worst_deg_from_vertical": round(float(math.degrees(math.acos(max(-1.0, min(1.0, -nz[~on_bed].min())))) if (~on_bed).any() else 0.0), 1),
        "bad_tri": int(bad.sum()),
        "bad_centroids": tri[bad].mean(axis=1)[:12].round(1).tolist(),
    }


def checks(p: sc.Params, parts):
    rep = {}
    assembled = parts["bottom"] + parts["ring"] + parts["lid"]
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
    rep["пересечения_деталей_мм3"] = {
        "низ×пояс": round((parts["bottom"] ^ parts["ring"]).volume(), 3),
        "низ×крышка": round((parts["bottom"] ^ parts["lid"]).volume(), 3),
        "пояс×крышка": round((parts["ring"] ^ parts["lid"]).volume(), 3)}
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
    _, tw = sc.text_contours(p)
    y_face = sc.plan(p).poly(0.0)[:, 1].max()

    def zone_for(name):
        zc = {"lid": p.text_z_lid, "bottom": p.text_z_base}.get(name)
        if zc is None:
            return None
        # центроиды в положении печати -> в координатах сборки
        m = print_pose(name, parts[name])
        bb0, bb1 = parts[name].bounding_box(), m.bounding_box()

        def f(c):
            c = c.copy()
            if name == "lid":
                c = c - [(bb1[0] + bb1[3]) / 2, (bb1[1] + bb1[4]) / 2, bb1[2]]
                c = c * [1, -1, -1] + [(bb0[0] + bb0[3]) / 2, (bb0[1] + bb0[4]) / 2, bb0[5]]
            else:
                c = c - [(bb1[0] + bb1[3]) / 2, (bb1[1] + bb1[4]) / 2, bb1[2]] + \
                    [(bb0[0] + bb0[3]) / 2, (bb0[1] + bb0[4]) / 2, bb0[2]]
            return (np.abs(c[:, 0]) < tw / 2 + 1) & (np.abs(c[:, 2] - zc) < p.text_cap / 2 + 1) & \
                (c[:, 1] > y_face - p.wall - 1)
        return f

    rep["нависания"] = {k: overhang_report(print_pose(k, m), bridge_zone=zone_for(k)) for k, m in parts.items()}
    rep["габарит_мм"] = [round(v, 1) for v in np.subtract(assembled.bounding_box()[3:], assembled.bounding_box()[:3])]
    rep["печатный_габарит_мм"] = {k: [round(v, 1) for v in np.subtract(print_pose(k, m).bounding_box()[3:],
                                                                         print_pose(k, m).bounding_box()[:3])]
                                  for k, m in parts.items()}
    return rep


def rig_checks(p: sc.Params, r: rg.Rig, case_parts, rig_parts):
    import trimesh
    rep = {}
    case = case_parts["bottom"] + case_parts["ring"] + case_parts["lid"]
    sens = rg.sensors(p, r)
    sensors_all = Manifold.batch_boolean(list(sens.values()), sc.OpType.Add)
    f = rg.frame(p, r)
    # байонет: в замке лепестки лежат на кольце с натягом, при вставке проходят пазы
    rep["рукоять×корпус_в_замке_мм3"] = round((rig_parts["handle"] ^ case).volume(), 2)
    unlocked = rg.handle(p, r, locked=False)
    head = unlocked ^ sc.box(-60, 60, -60, 60, f["z_floor"] - 0.5, 0)
    shadow = head.project().extrude(30).translate([0, 0, f["z_floor"] - 10])
    rep["путь_вставки_байонета_мм3"] = round((shadow ^ case_parts["bottom"]).volume(), 2)
    pairs = {
        "кронштейн×корпус": (rig_parts["arm"], case),
        "кронштейн×рукоять": (rig_parts["arm"], rig_parts["handle"]),
        "площадка×корпус": (rig_parts["plate"], case),
        "площадка×кронштейн": (rig_parts["plate"], rig_parts["arm"]),
        "датчики×площадка": (sensors_all, rig_parts["plate"]),
        "датчики×корпус_и_кронштейн": (sensors_all, case + rig_parts["arm"]),
    }
    rep["пересечения_мм3"] = {k: round((a ^ b).volume(), 2) for k, (a, b) in pairs.items()}
    rep["поля_зрения"] = rg.fov_report(p, r)
    # масса и баланс: печатные детали — сплошные (верхняя оценка) и ≈ 45 % при 3 периметрах и 20 % заполнения
    items = []
    for k, m in list(case_parts.items()) + list(rig_parts.items()):
        v, fc = to_arrays(m)
        tm = trimesh.Trimesh(v, fc, process=False)
        k_fill = 1.0 if k in ("bottom", "ring", "lid") else 0.45
        items.append((k, m.volume() / 1000 * DENSITY["PETG"] * k_fill, tm.center_mass))
    items.append(("плата", 132.0, np.array([0.0, 0.0, 8.0])))
    for k, m in sens.items():
        bb = m.bounding_box()
        items.append((k, float(rg.SENSOR_MASS[k]), (np.array(bb[:3]) + np.array(bb[3:])) / 2))
    total = sum(w for _, w, _ in items)
    com = sum(w * np.asarray(c) for _, w, c in items) / total
    z_mid = f["z_tongue0"] - r.head_h - r.flare_h - r.grip_len / 2
    x_hand = -(r.head_h + r.flare_h + r.grip_len / 2 - r.head_h - r.flare_h) * math.tan(math.radians(r.grip_tilt))
    rep["масса_сборки_г"] = round(total)
    rep["масса_по_деталям_г"] = {k: round(w) for k, w, _ in items}
    rep["центр_масс_мм"] = [round(float(v), 1) for v in com]
    rep["вылет_центра_масс_от_середины_рукояти_мм"] = round(float(com[0] - x_hand), 1)
    rep["момент_на_кисть_Н·м"] = round(total / 1000 * 9.81 * float(com[0] - x_hand) / 1000, 2)
    rep["нависания"] = {k: overhang_report(print_pose(k, m)) for k, m in rig_parts.items()}
    rep["объём_см3"] = {k: round(m.volume() / 1000, 1) for k, m in rig_parts.items()}
    rep["печатный_габарит_мм"] = {k: [round(v, 1) for v in np.subtract(print_pose(k, m).bounding_box()[3:],
                                                                         print_pose(k, m).bounding_box()[:3])]
                                  for k, m in rig_parts.items()}
    allm = case + Manifold.batch_boolean(list(rig_parts.values()), sc.OpType.Add) + sensors_all
    rep["габарит_сканера_мм"] = [round(v, 1) for v in np.subtract(allm.bounding_box()[3:], allm.bounding_box()[:3])]
    return rep


def write_glb(p: sc.Params, parts, path, extra=None):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    scene = trimesh.Scene()
    # кольцо-акцент — верх нижней части, печатается другим цветом: показываем отдельно
    zr = p.z_part - p.ring_h
    big = 400.0
    allparts = dict(parts)
    allparts.update(sc.board_dummy(p))
    allparts.update(extra or {})
    for name, m in allparts.items():
        v, f = to_arrays(m)
        tm = trimesh.Trimesh(v, f, process=False)
        col, metal, rough = COLORS.get(name, ([0.6, 0.6, 0.6, 1], 0.0, 0.6))
        tm.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(name=name, baseColorFactor=col, metallicFactor=metal, roughnessFactor=rough))
        scene.add_geometry(tm, node_name=name, geom_name=name)
    with open(path, "wb") as fh:
        fh.write(scene.export(file_type="glb"))


def plate_thumbnails(render_dir):
    """Картинки столов для превью в Bambu Studio: docs/renders/p1s_plate_N.png, если они есть."""
    from PIL import Image
    out = {}
    for i in range(1, len(bp.PLATES) + 1):
        src = os.path.join(render_dir, f"p1s_plate_{i}.png")
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert("RGBA")
        side = min(im.size)
        im = im.crop(((im.width - side) // 2, (im.height - side) // 2, (im.width + side) // 2, (im.height + side) // 2))
        pngs = []
        for size in (512, 128):
            buf = io.BytesIO()
            im.resize((size, size), Image.LANCZOS).save(buf, "PNG", optimize=True)
            pngs.append(buf.getvalue())
        out[i] = tuple(pngs)
    return out


def parse_set(items):
    """--set имя=значение для корпуса, --set rig.имя=значение для рукояти и площадки."""
    p, r = sc.Params(), rg.Rig()
    for it in items or []:
        k, v = it.split("=", 1)
        target = p
        if k.startswith("rig."):
            target, k = r, k[4:]
        if k not in {f.name for f in fields(target)}:
            raise SystemExit(f"нет параметра {k}")
        cur = getattr(target, k)
        if isinstance(cur, tuple):
            val = tuple(float(x) for x in v.split(","))
        elif isinstance(cur, int):
            val = int(v)
        elif isinstance(cur, str):
            val = v
        else:
            val = float(v)
        setattr(target, k, val)
    return p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append")
    ap.add_argument("--no-glb", action="store_true")
    ap.add_argument("--out", default=ROOT)
    a = ap.parse_args()
    p, r = parse_set(a.set)
    parts = sc.build(p)
    rig_parts = rg.build_rig(p, r)

    stl_dir = os.path.join(a.out, "stl")
    os.makedirs(stl_dir, exist_ok=True)
    posed = {k: print_pose(k, m) for k, m in list(parts.items()) + list(rig_parts.items())}
    for k, m in posed.items():
        write_stl(m, os.path.join(stl_dir, STL_NAMES[k]))
    project = bp.write_project(posed, os.path.join(a.out, "print", "scanner_p1s.3mf"), "Сканер OPi6",
                               plate_thumbnails(os.path.join(a.out, "docs", "renders")))
    if not a.no_glb:
        vdir = os.path.join(a.out, "docs", "viewer")
        os.makedirs(vdir, exist_ok=True)
        extra = dict(rig_parts)
        extra.update(rg.sensors(p, r))
        write_glb(p, parts, os.path.join(vdir, "scanner_case.glb"), extra)

    rep = checks(p, parts)
    rep["вентиляция"] = getattr(sc.build, "stats", {})
    rep["оснастка"] = rig_checks(p, r, parts, rig_parts)
    rep["печать_P1S"] = dict(project, столы=bp.layout(posed))
    rep["параметры"] = asdict(p)
    rep["параметры_оснастки"] = asdict(r)
    with open(os.path.join(a.out, "docs", "report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    show = {k: v for k, v in rep.items() if k != "параметры"}
    print(json.dumps(show, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
