"""
Проект Bambu Studio (.3mf) под Bambu Lab P1S.

Детали уже повёрнуты для печати и разложены по четырём столам, у каждого стола
один пруток, поэтому AMS не нужен. Профили — системные: принтер
«Bambu Lab P1S 0.4 nozzle», процесс «0.20mm Standard @BBL X1C», пруток
«Generic PETG». Bambu Studio берёт их значения из своих профилей по имени,
в проекте хранятся только отличия по деталям: стенки, заполнение, кайма, шов.

Формат повторяет то, что пишет сам Bambu Studio (src/libslic3r/Format/bbs_3mf.cpp):
3D/3dmodel.model — сетки и сборка, Metadata/model_settings.config — детали и столы,
Metadata/project_settings.config — профили.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np
from manifold3d import Manifold

APP_VERSION = "02.00.00.95"
PRINTER = "Bambu Lab P1S 0.4 nozzle"
PRINTER_MODEL = "Bambu Lab P1S"
PROCESS = "0.20mm Standard @BBL X1C"
FILAMENT = "Generic PETG"
FILAMENT_ID = "GFG99"
BED = 256.0                       # printable_area P1S: 0x0..256x256
BED_EXCLUDE = (18.0, 28.0)        # bed_exclude_area: передний левый угол под срезку прутка
MAX_H = 250.0                     # printable_height
STRIDE = BED * (1 + 1 / 5)        # PartPlateList: шаг столов = ширина × (1 + LOGICAL_PART_PLATE_GAP)
BED_TYPE = "Textured PEI Plate"

# слоты прутка: имя, цвет
FILAMENTS = [
    ("PETG чёрный", "#1B1B1D"),
    ("PETG серебристый", "#B9BCC0"),
    ("PETG бронзовый", "#A8743F"),
]


@dataclass
class Item:
    part: str
    name: str
    filament: int                  # 1..len(FILAMENTS)
    center: tuple[float, float]    # центр габарита на столе, мм от переднего левого угла
    rot_z: float = 0.0             # поворот на столе, градусы
    settings: dict = field(default_factory=dict)


@dataclass
class Plate:
    name: str
    items: list[Item]


# настройки поверх системного процесса: ключи Bambu Studio уровня объекта
_CASE = {"wall_loops": "4", "seam_position": "back", "brim_type": "no_brim"}
_RIG = {"wall_loops": "4", "sparse_infill_density": "25%", "sparse_infill_pattern": "gyroid"}

PLATES = [
    Plate("Низ корпуса и рукоять", [
        # разъёмами к задней стенке принтера: шов «back» уходит на сторону разъёмов
        Item("bottom", "Низ корпуса", 1, (85.0, 128.0), 180.0, dict(_CASE)),
        Item("handle", "Рукоять", 1, (205.0, 128.0), 0.0,
             dict(_RIG, brim_type="outer_only", brim_width="5", seam_position="back")),
    ]),
    Plate("Крышка", [
        Item("lid", "Крышка", 2, (128.0, 128.0), 0.0, dict(_CASE)),
    ]),
    Plate("Пояс", [
        Item("ring", "Пояс", 3, (128.0, 128.0), 180.0,
             {"wall_loops": "2", "seam_position": "back",
              "brim_type": "outer_and_inner", "brim_width": "3"}),
    ]),
    Plate("Площадка и кронштейн", [
        Item("plate", "Площадка датчиков", 1, (128.0, 172.0), 0.0, dict(_RIG)),
        Item("arm", "Кронштейн", 1, (128.0, 62.0), 0.0,
             dict(_RIG, wall_loops="5", sparse_infill_density="40%",
                  brim_type="outer_only", brim_width="5")),
    ]),
]


def plate_origin(i: int, count: int) -> tuple[float, float]:
    """Положение i-го стола в мировых координатах Bambu Studio (PartPlateList::compute_shape_position)."""
    v = count ** 0.5
    cols = int(round(v)) + (1 if v > round(v) else 0)
    return (i % cols) * STRIDE, -(i // cols) * STRIDE


def _arrays(m: Manifold):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    return v, f


def placed(m: Manifold, rot_z: float) -> Manifold:
    """Деталь в положении для печати, повёрнутая на столе, центр габарита в (0, 0), низ на z = 0."""
    if rot_z:
        m = m.rotate([0, 0, rot_z])
    bb = m.bounding_box()
    return m.translate([-(bb[0] + bb[3]) / 2, -(bb[1] + bb[4]) / 2, -bb[2]])


def layout(posed: dict[str, Manifold]) -> list[dict]:
    """Проверяет раскладку и возвращает габариты деталей на столах (для отчёта)."""
    out = []
    for pi, pl in enumerate(PLATES):
        boxes = []
        for it in pl.items:
            m = placed(posed[it.part], it.rot_z)
            bb = m.bounding_box()
            brim = float(it.settings.get("brim_width", 0)) if it.settings.get("brim_type", "no_brim") != "no_brim" else 0.0
            x0, x1 = it.center[0] + bb[0] - brim, it.center[0] + bb[3] + brim
            y0, y1 = it.center[1] + bb[1] - brim, it.center[1] + bb[4] + brim
            boxes.append((it, (x0, y0, x1, y1), bb[5]))
        problems = []
        for it, (x0, y0, x1, y1), h in boxes:
            if x0 < 2 or y0 < 2 or x1 > BED - 2 or y1 > BED - 2:
                problems.append(f"{it.name}: выходит за стол")
            if x0 < BED_EXCLUDE[0] and y0 < BED_EXCLUDE[1]:
                problems.append(f"{it.name}: в зоне срезки прутка")
            if h > MAX_H:
                problems.append(f"{it.name}: выше {MAX_H} мм")
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                (_, ra, _), (_, rb, _) = boxes[a], boxes[b]
                gap = max(rb[0] - ra[2], ra[0] - rb[2], rb[1] - ra[3], ra[1] - rb[3])
                if gap < 5:
                    problems.append(f"{boxes[a][0].name} и {boxes[b][0].name}: зазор {gap:.1f} мм")
        out.append({
            "стол": pi + 1,
            "название": pl.name,
            "пруток": FILAMENTS[pl.items[0].filament - 1][0],
            "детали": {it.name: {"габарит_с_каймой_мм": [round(x1 - x0, 1), round(y1 - y0, 1), round(h, 1)],
                                 "на_столе_мм": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]}
                       for it, (x0, y0, x1, y1), h in boxes},
            "замечания": problems,
        })
    return out


def _fmt(a: np.ndarray) -> np.ndarray:
    s = np.char.mod("%.4f", a)
    s = np.char.rstrip(np.char.rstrip(s, "0"), ".")
    return np.where(s == "-0", "0", s)


def _mesh_xml(obj_id: int, v: np.ndarray, f: np.ndarray) -> str:
    sv = _fmt(v)
    verts = "".join(f'     <vertex x="{x}" y="{y}" z="{z}"/>\n' for x, y, z in sv)
    tris = "".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>\n' for a, b, c in f)
    return (f'  <object id="{obj_id}" type="model">\n   <mesh>\n    <vertices>\n{verts}    </vertices>\n'
            f'    <triangles>\n{tris}    </triangles>\n   </mesh>\n  </object>\n')


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&apos;"))


IDENTITY = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"


def write_project(posed: dict[str, Manifold], path: str, title: str,
                  thumbnails: dict[int, tuple[bytes, bytes]] | None = None) -> dict:
    """posed — детали в положении для печати (как в STL). thumbnails — {номер стола: (png 512, png 128)}."""
    thumbnails = thumbnails or {}
    n_plates = len(PLATES)

    objects_xml, components_xml, build_xml = [], [], []
    obj_cfg, plate_cfg = [], []
    next_id = 1
    identify = 100
    for pi, pl in enumerate(PLATES):
        ox, oy = plate_origin(pi, n_plates)
        instances = []
        for it in pl.items:
            m = placed(posed[it.part], it.rot_z)
            v, f = _arrays(m)
            mesh_id, obj_id = next_id, next_id + 1
            next_id += 2
            objects_xml.append(_mesh_xml(mesh_id, v, f))
            components_xml.append(
                f'  <object id="{obj_id}" type="model">\n   <components>\n'
                f'    <component objectid="{mesh_id}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n'
                f'   </components>\n  </object>\n')
            tx, ty = ox + it.center[0], oy + it.center[1]
            build_xml.append(f'  <item objectid="{obj_id}" transform="1 0 0 0 1 0 0 0 1 {tx:.4f} {ty:.4f} 0" printable="1"/>\n')

            meta = [("name", it.name), ("extruder", str(it.filament))] + sorted(it.settings.items())
            lines = "".join(f'    <metadata key="{k}" value="{_esc(str(val))}"/>\n' for k, val in meta)
            obj_cfg.append(
                f'  <object id="{obj_id}">\n{lines}'
                f'    <metadata face_count="{len(f)}"/>\n'
                f'    <part id="{mesh_id}" subtype="normal_part">\n'
                f'      <metadata key="name" value="{_esc(it.name)}"/>\n'
                f'      <metadata key="matrix" value="{IDENTITY}"/>\n'
                f'      <mesh_stat face_count="{len(f)}" edges_fixed="0" degenerate_facets="0" '
                f'facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
                f'    </part>\n  </object>\n')
            instances.append(
                f'    <model_instance>\n'
                f'      <metadata key="object_id" value="{obj_id}"/>\n'
                f'      <metadata key="instance_id" value="0"/>\n'
                f'      <metadata key="identify_id" value="{identify}"/>\n'
                f'    </model_instance>\n')
            identify += 1
        thumb = f'    <metadata key="thumbnail_file" value="Metadata/plate_{pi + 1}.png"/>\n' if pi + 1 in thumbnails else ""
        plate_cfg.append(
            f'  <plate>\n'
            f'    <metadata key="plater_id" value="{pi + 1}"/>\n'
            f'    <metadata key="plater_name" value="{_esc(pl.name)}"/>\n'
            f'    <metadata key="locked" value="false"/>\n'
            f'    <metadata key="bed_type" value="{BED_TYPE}"/>\n'
            f'{thumb}{"".join(instances)}  </plate>\n')

    model_meta = {
        "Application": f"BambuStudio-{APP_VERSION}",
        "BambuStudio:3mfVersion": "1",
        "Title": title,
        "Designer": "",
        "Description": "",
    }
    model = io.StringIO()
    model.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    model.write('<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
                'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">\n')
    for k, val in model_meta.items():
        model.write(f' <metadata name="{k}">{_esc(val)}</metadata>\n')
    model.write(" <resources>\n")
    for s in objects_xml:
        model.write(s)
    for s in components_xml:
        model.write(s)
    model.write(" </resources>\n <build>\n")
    for s in build_xml:
        model.write(s)
    model.write(" </build>\n</model>\n")

    settings = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
                + "".join(obj_cfg) + "".join(plate_cfg) + "</config>\n")

    nf = len(FILAMENTS)
    flush = []
    for a in range(nf):
        for b in range(nf):
            flush.append("0" if a == b else "280")
    project = {
        "name": "project_settings",
        "from": "project",
        "version": APP_VERSION,
        "printer_model": PRINTER_MODEL,
        "printer_settings_id": PRINTER,
        "print_settings_id": PROCESS,
        "filament_settings_id": [FILAMENT] * nf,
        "inherits": "",
        "inherits_group": [""] * (nf + 2),
        "different_settings_to_system": [""] * (nf + 2),
        "filament_colour": [c for _, c in FILAMENTS],
        "filament_type": ["PETG"] * nf,
        "filament_diameter": ["1.75"] * nf,
        "filament_ids": [FILAMENT_ID] * nf,
        "nozzle_diameter": ["0.4"],
        "printable_area": ["0x0", "256x0", "256x256", "0x256"],
        "printable_height": str(int(MAX_H)),
        "curr_bed_type": BED_TYPE,
        "flush_volumes_matrix": flush,
        "flush_volumes_vector": ["140"] * (2 * nf),
        "flush_multiplier": ["1"],
    }

    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n')
    if 1 in thumbnails:
        rels += (' <Relationship Target="/Metadata/plate_1.png" Id="rel-2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>\n'
                 ' <Relationship Target="/Metadata/plate_1.png" Id="rel-4" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-middle"/>\n'
                 ' <Relationship Target="/Metadata/plate_1_small.png" Id="rel-5" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-small"/>\n')
    rels += "</Relationships>"
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                     ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                     ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
                     ' <Default Extension="png" ContentType="image/png"/>\n'
                     ' <Default Extension="gcode" ContentType="text/x.gcode"/>\n'
                     '</Types>')

    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp = (2026, 1, 1, 0, 0, 0)     # постоянная дата в архиве: одинаковая сборка — одинаковый файл

    def add(z, name, data):
        info = zipfile.ZipInfo(name, date_time=stamp)
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, data)

    with zipfile.ZipFile(path, "w") as z:
        add(z, "[Content_Types].xml", content_types)
        add(z, "_rels/.rels", rels)
        add(z, "3D/3dmodel.model", model.getvalue())
        add(z, "Metadata/model_settings.config", settings)
        add(z, "Metadata/project_settings.config", json.dumps(project, ensure_ascii=False, indent=4))
        for i, (big, small) in sorted(thumbnails.items()):
            add(z, f"Metadata/plate_{i}.png", big)
            add(z, f"Metadata/plate_{i}_small.png", small)
    return {"файл": os.path.relpath(path), "размер_МБ": round(os.path.getsize(path) / 1e6, 2)}
