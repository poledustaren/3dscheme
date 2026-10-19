"""
Сборка органайзера: STL в положении для печати, GLB для 3D-просмотра, проект Bambu Studio
под P1S, проверки и отчёт docs/report.json.

    python3 cad/build.py                       # STL, GLB, проект, отчёт
    python3 cad/build.py --render              # плюс рендеры Blender в docs/renders и превью стола в проекте
    python3 cad/build.py --bambu bambu-studio  # полные профили в проекте и нарезка: время и граммы в отчёте
    python3 cad/build.py --set rib_n=72 --set word_center="сегодня!"
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, fields

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import organizer as og  # noqa: E402
import bambu_project as bp  # noqa: E402
from manifold3d import JoinType, Manifold  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLA = 1.24          # г/см³
PULL = 25.0         # насколько выдвинут верхний ящик на рендерах и в 3D-просмотре


def to_arrays(m: Manifold):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties, np.float64)[:, :3], np.asarray(mesh.tri_verts, np.int64)


def write_stl(m: Manifold, path):
    import trimesh
    v, f = to_arrays(m)
    trimesh.Trimesh(v, f, process=False).export(path)


def down_faces(m: Manifold, bed_tol=0.25, limit_deg=45.0):
    """Грани, смотрящие вниз круче limit_deg от вертикали, кроме лежащих на столе.

    Горизонтальные (мосты) группируются по высоте: площадь и габарит по X и Y.
    """
    v, f = to_arrays(m)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2
    nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    zmin = np.minimum(np.minimum(a[:, 2], b[:, 2]), c[:, 2])
    down = (nz < -math.cos(math.radians(limit_deg))) & (zmin > bed_tol)
    flat = down & (nz < -0.999)
    # связные куски горизонтальных потолков: большие — мосты, мелкие — верхние кромки букв и табличек
    import trimesh
    idx = np.flatnonzero(flat)
    sub = trimesh.Trimesh(v, f[idx], process=False)
    bridges, small = [], []
    for comp in trimesh.graph.connected_components(sub.face_adjacency, nodes=np.arange(len(idx))):
        pts = v[f[idx[comp]]].reshape(-1, 3)
        span = pts.max(0) - pts.min(0)
        row = {"z": round(float(pts[:, 2].min()), 2), "габарит_XY_мм": [round(float(span[0]), 1), round(float(span[1]), 1)],
               "площадь_мм2": round(float(area[idx[comp]].sum()), 1)}
        (bridges if min(span[0], span[1]) > 3.0 else small).append(row)
    other = down & ~flat
    return {"мосты": sorted(bridges, key=lambda r: r["z"]),
            "мелкие_потолки": {"штук": len(small),
                               "самый_широкий_мм": max((min(r["габарит_XY_мм"]) for r in small), default=0.0)},
            "наклонные_свесы_мм2": round(float(area[other].sum()), 1),
            "самый_пологий_свес_град_от_горизонтали": round(float(np.degrees(np.arccos(-nz[other].min()))) if other.any() else 0.0, 1)}


def checks(p: og.Params, parts: dict[str, Manifold]) -> dict:
    body, drawer = parts["body"], parts["drawer"]
    h = p.cup_pitch / 2
    y0, y1 = p.bay_y
    py0, py1 = p.block_front + p.pocket_wall, p.block_back - p.pocket_wall
    # полочки и карманчики: кратчайшая стенка до отверстия стакана по плану (ось стакана в (h, 0))
    from manifold3d import CrossSection
    bore = CrossSection.circle(p.bore_r, og.SEG).translate([h, 0])
    side_gap = {}
    for name, top in (("полочки", False), ("карманчики", True)):
        plan = og.side_plan(p, top)
        side_gap[name] = next((round(w, 2) for w in np.arange(0.0, 5.0, 0.05)
                               if (plan ^ bore.offset(w + 0.05, JoinType.Round, 2.0, 96)).area() > 1e-3), None)
    shelf = og.side_plan(p, False)
    front = shelf ^ CrossSection.square([h, 0.5]).translate([0, p.block_front])
    fb = front.bounds()
    # полочки не должны прорезать наружные стенки: всё, что дальше 1,5 мм от плана корпуса внутрь, кроме проёма спереди
    # выше верха корпуса фаска режет воздух — считаем только до z = height
    inner = og.outline(p).offset(-1.5, JoinType.Round, 2.0, 48).extrude(p.height)
    beyond = (og.side_cutters(p) - inner) ^ Manifold.cube([2 * h + 80, 200, p.height + 4.99]).translate([-h - 40, p.block_front + 2.0, -5])
    words = {}
    for side, word, depth in (("левый стакан", p.word_left, None), ("центр", p.word_center, p.panel_depth_center),
                              ("правый стакан", p.word_right, None)):
        _, width = og.word_contours(p, word)
        pw = width + 2 * p.panel_pad
        info = {"слово": word, "длина_слова_мм": round(width, 1), "табличка_мм": [round(pw, 1), p.panel_h]}
        if depth is None:
            half = math.degrees(pw / 2 / p.panel_r_cup)
            junction = 90 - math.degrees(math.asin(-p.block_front / p.cup_r))
            info["дуга_таблички_град"] = round(2 * half, 1)
            info["запас_до_стыка_с_блоком_град"] = round(junction - half, 1)
        words[side] = info
    clash = {f"ящик_{i + 1}_с_корпусом_мм3": round((body ^ og.drawer_in_body(p, i)).volume(), 3) for i in (0, 1)}
    w, d, hgt = p.drawer_size
    return {
        "габарит_мм": [round(x, 1) for x in np.subtract(body.bounding_box()[3:], body.bounding_box()[:3])],
        "стаканы": {"Ø_по_рёбрам": 2 * p.cup_r, "Ø_отверстия": round(2 * p.bore_r, 1), "рёбер": p.rib_n,
                    "шаг_рёбер_мм": round(2 * math.pi * p.cup_r / p.rib_n, 2), "глубина_мм": p.height - p.cup_floor},
        "карман_мм": [p.pocket_w, round(py1 - py0, 1), p.height - p.pocket_floor_z],
        "ящик_мм": [round(w, 1), round(d, 1), round(hgt, 1)],
        "ящик_внутри_мм": [round(w - 2 * p.drawer_wall, 1), round(d - p.drawer_front - p.drawer_wall, 1),
                           round(hgt - p.drawer_floor, 1)],
        "зазоры_ящика_мм": {"бок": p.gap_side, "верх": p.gap_top, "зад": p.gap_back},
        "пересечения": clash,
        "полочки": {"ярусов_с_каждой_стороны": p.side_tiers,
                    "высота_яруса_мм": round(og.side_tiers(p)[0][1] - og.side_tiers(p)[0][0], 1),
                    "проём_спереди_мм": round(fb[2] - fb[0], 1),
                    "глубина_мм": round(p.bay_y[1] - p.block_front, 1),
                    "площадь_яруса_мм2": round(shelf.area())},
        "карманчики": {"глубина_мм": p.height - p.pocket_floor_z,
                       "площадь_мм2": round(og.side_plan(p, True).area())},
        "стенки_мм": {
            "полочки_до_отверстия_стакана": side_gap["полочки"],
            "карманчики_до_отверстия_стакана": side_gap["карманчики"],
            "перегородки_ниши_и_кармана": p.side_wall,
            "полочки_и_карманчики_наружу_мм3": round(beyond.volume(), 3),
            "под_надписью_на_стакане": round(p.panel_r_cup - p.text_depth - p.bore_r, 2),
            "под_надписью_в_центре": round(p.pocket_wall - p.panel_depth_center - p.text_depth, 2),
        },
        "надписи": words,
        "свесы": {"корпус": down_faces(body), "ящик": down_faces(drawer)},
        "масса_сплошная_г_PLA": {"корпус": round(body.volume() / 1000 * PLA), "ящик": round(drawer.volume() / 1000 * PLA)},
    }


def assembly(p: og.Params) -> dict:
    """Положение ящиков в корпусе для рендера: сдвиг STL ящика (лицо в y = 0, дно в z = 0)."""
    return {"drawers": [[0.0, p.block_front - (PULL if i == 1 else 0.0), p.slot_z(i)[0]] for i in (0, 1)],
            "bore": [[-p.cup_pitch / 2, 0.0, p.cup_floor], [p.cup_pitch / 2, 0.0, p.cup_floor]],
            "bore_r": p.bore_r,
            "pocket": [0.0, (p.block_front + p.block_back) / 2, p.pocket_floor_z],
            "pull": PULL,
            "side_x": (p.bay_w / 2 + p.side_wall + p.cup_pitch / 2 - math.sqrt(p.cup_r ** 2 - p.block_front ** 2) - p.side_front_margin) / 2,
            "tiers": og.side_tiers(p),
            "plate": [[it.center[0], it.center[1], it.part] for it in bp.PLATES[0].items]}


def write_glb(p: og.Params, parts, path):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    scene = trimesh.Scene()
    mat = PBRMaterial(baseColorFactor=[0.33, 0.34, 0.35, 1.0], roughnessFactor=0.6, metallicFactor=0.0)
    meshes = [("корпус", parts["body"])] + [(f"ящик {i + 1}", og.drawer_in_body(p, i, PULL if i == 1 else 0.0)) for i in (0, 1)]
    for name, m in meshes:
        v, f = to_arrays(m)
        tm = trimesh.Trimesh(v[:, [0, 2, 1]] * [1, 1, -1], f, process=False)   # glTF: Y вверх
        tm.visual = trimesh.visual.TextureVisuals(material=mat)
        scene.add_geometry(tm, node_name=name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene.export(path)


def render(out_dir):
    blender = shutil.which("blender")
    if not blender:
        print("blender не найден — рендеры пропущены")
        return
    script = os.path.join(ROOT, "cad", "render_blender.py")
    r = subprocess.run([blender, "-b", "--factory-startup", "-P", script, "--", ROOT], capture_output=True, text=True)
    if r.returncode != 0 or "RENDER OK" not in r.stdout:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit("рендер не удался")


def plate_thumbnail(render_dir):
    from PIL import Image
    src = os.path.join(render_dir, "p1s_plate_1.png")
    if not os.path.exists(src):
        return {}
    im = Image.open(src).convert("RGBA")
    out = []
    for size in (512, 128):
        buf = io.BytesIO()
        im.resize((size, size), Image.LANCZOS).save(buf, "PNG")
        out.append(buf.getvalue())
    return {1: tuple(out)}


def parse_set(items):
    names = {f.name: f.type for f in fields(og.Params)}
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in names:
            raise SystemExit(f"нет параметра {k}")
        t = names[k]
        out[k] = int(v) if t in ("int", int) else float(v) if t in ("float", float) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--bambu", help="исполняемый файл Bambu Studio (CLI)")
    a = ap.parse_args()
    p = og.Params(**parse_set(a.set))
    parts = og.build(p)
    for k, m in parts.items():
        assert m.status().name == "NoError", (k, m.status())

    os.makedirs(os.path.join(ROOT, "stl"), exist_ok=True)
    write_stl(parts["body"], os.path.join(ROOT, "stl", "organizer_body.stl"))
    write_stl(parts["drawer"], os.path.join(ROOT, "stl", "organizer_drawer.stl"))
    write_glb(p, parts, os.path.join(ROOT, "docs", "viewer", "organizer.glb"))
    json.dump(assembly(p), open(os.path.join(ROOT, "docs", "assembly.json"), "w"), ensure_ascii=False, indent=2)

    render_dir = os.path.join(ROOT, "docs", "renders")
    if a.render:
        render(render_dir)

    posed = {"body": parts["body"], "drawer": parts["drawer"]}
    proj_path = os.path.join(ROOT, "print", "organizer_p1s.3mf")
    proj = bp.write_project(posed, proj_path,
                            "Органайзер «вчера · сегодня · завтра»", plate_thumbnail(render_dir))
    bambu = a.bambu or shutil.which("bambu-studio")
    if bambu:
        bp.embed_full_presets(proj_path, bambu, os.path.join(ROOT, "print", "presets"))
        proj["нарезка_Bambu_Studio"] = bp.slice_estimate(proj_path, a.bambu)
        proj["размер_МБ"] = round(os.path.getsize(proj_path) / 1e6, 2)
    else:
        print("ВНИМАНИЕ: Bambu Studio не найден (--bambu) — в проекте только имена профилей, "
              "Studio подставит значения по умолчанию. Для готового проекта запустите с --bambu.")
    report = {"параметры": asdict(p), "проверки": checks(p, parts), "печать_P1S": bp.layout(posed), "проект": proj}
    json.dump(report, open(os.path.join(ROOT, "docs", "report.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(report["проверки"], ensure_ascii=False, indent=1))
    print(json.dumps(report["печать_P1S"], ensure_ascii=False, indent=1))
    print(json.dumps(report["проект"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
