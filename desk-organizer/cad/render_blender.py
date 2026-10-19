"""
Рендеры органайзера в Blender (Cycles): blender -b --factory-startup -P cad/render_blender.py -- <папка проекта>

Берёт stl/organizer_body.stl, stl/organizer_drawer.stl и docs/assembly.json (положение ящиков),
пишет docs/renders/*.png. Ручки и стикеры в кадре — только для масштаба.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

ROOT = sys.argv[sys.argv.index("--") + 1]
OUT = os.path.join(ROOT, "docs", "renders")
MM = 0.001
os.makedirs(OUT, exist_ok=True)
asm = json.load(open(os.path.join(ROOT, "docs", "assembly.json")))


# ---------- сцена ----------

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            if any(d.type == kind for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == kind
                sc.cycles.device = "GPU"
                break
        except TypeError:
            continue
    sc.cycles.samples = 160
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.78, 0.8, 0.84, 1)
    bg.inputs[1].default_value = 0.2
    return sc


def material(name, color, rough=0.5, layer_lines=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    if layer_lines:
        # слои печати 0,2 мм: синус по высоте → bump
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 2 * math.pi / (0.2 * MM)
        sin = nt.nodes.new("ShaderNodeMath")
        sin.operation = "SINE"
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.12
        bump.inputs["Distance"].default_value = 0.02 * MM
        nt.links.new(tc.outputs["Object"], sep.inputs[0])
        nt.links.new(sep.outputs["Z"], mul.inputs[0])
        nt.links.new(mul.outputs[0], sin.inputs[0])
        nt.links.new(sin.outputs[0], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def wood():
    """Светлый дуб: шум, растянутый вдоль X, даёт волокна."""
    m = bpy.data.materials.new("wood")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.45
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.8, 30.0, 1.0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.62
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0.33, 0.19, 0.09, 1)
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = (0.52, 0.33, 0.17, 1)
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    nt.links.new(mp.outputs[0], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], bsdf.inputs["Base Color"])
    return m


def import_stl(path, name, mat, loc=(0, 0, 0)):
    bpy.ops.wm.stl_import(filepath=path, global_scale=MM)
    ob = bpy.context.selected_objects[0]
    ob.name = name
    ob.location = Vector(loc) * MM
    try:
        bpy.ops.object.shade_auto_smooth(angle=math.radians(35))
    except Exception:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
    ob.data.materials.append(mat)
    return ob


def box(name, size, loc, mat, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = Vector(size) * MM
    ob.location = Vector(loc) * MM
    ob.rotation_euler = rot
    ob.data.materials.append(mat)
    return ob


def pen(base, tilt_deg, azim_deg, length, r, mat, tip_mat):
    """Ручка: цилиндр с кнопкой вверху, низ в точке base (мм)."""
    t, a = math.radians(tilt_deg), math.radians(azim_deg)
    d = Vector((math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)))
    rot = Vector((0, 0, 1)).rotation_difference(d).to_euler()
    c = Vector(base) + d * (length / 2)
    bpy.ops.mesh.primitive_cylinder_add(radius=r * MM, depth=length * MM, vertices=48, location=c * MM, rotation=rot)
    body = bpy.context.active_object
    body.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    tip = Vector(base) + d * (length + 4)       # кнопка ручки
    bpy.ops.mesh.primitive_cylinder_add(radius=r * 0.55 * MM, depth=8 * MM, vertices=48,
                                        location=tip * MM, rotation=rot)
    cone = bpy.context.active_object
    cone.data.materials.append(tip_mat)
    bpy.ops.object.shade_smooth()


def area_light(name, loc, target, size, power, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="AREA", location=Vector(loc) * MM)
    li = bpy.context.active_object
    li.name = name
    li.data.size = size * MM
    li.data.energy = power
    li.data.color = color
    li.rotation_euler = (Vector(target) * MM - li.location).to_track_quat("-Z", "Y").to_euler()
    return li


def camera(loc, target, lens=50, ortho=None):
    bpy.ops.object.camera_add(location=Vector(loc) * MM)
    cam = bpy.context.active_object
    cam.rotation_euler = (Vector(target) * MM - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens
    cam.data.clip_start = 0.01
    cam.data.clip_end = 20
    if ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho * MM
    bpy.context.scene.camera = cam
    return cam


def shoot(name, w, h, transparent=False):
    only = os.environ.get("RENDER_ONLY")
    if only and name.split(".")[0] not in only.split(","):
        return
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.film_transparent = transparent
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    sc.render.filepath = os.path.join(OUT, name)
    bpy.ops.render.render(write_still=True)


# ---------- кадры ----------

def desk_scene(props=True):
    reset()
    pla = material("pla", (0.045, 0.047, 0.05), rough=0.5, layer_lines=True)
    import_stl(os.path.join(ROOT, "stl", "organizer_body.stl"), "body", pla)
    for i, loc in enumerate(asm["drawers"]):
        import_stl(os.path.join(ROOT, "stl", "organizer_drawer.stl"), f"drawer{i}", pla, loc)
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0.4, 0))
    bpy.context.active_object.data.materials.append(wood())
    wall = material("wall", (0.72, 0.70, 0.66), rough=0.9)
    box("wall", (3000, 10, 1500), (0, 700, 750), wall)
    if props:
        black = material("black", (0.012, 0.012, 0.014), rough=0.35)
        blue = material("blue", (0.02, 0.05, 0.32), rough=0.3)
        steel = material("steel", (0.8, 0.8, 0.82), rough=0.25)
        steel.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 1.0
        bx, by, bz = asm["bore"][1]
        for dx, dy, tilt, az, m in ((-14, -8, 13, 20, black), (-6, 10, 10, 50, blue), (8, -12, 9, 150, black),
                                    (-18, 6, 16, 0, blue)):
            pen((bx + dx, by + dy, bz), tilt, az, 138, 4.6, m, steel)
        yellow = material("notes", (0.95, 0.78, 0.12), rough=0.85)
        px, py, pz = asm["pocket"]
        box("notes", (51, 38, 30), (px, py, pz + 15), yellow, (0, 0, math.radians(-5)))
        # мелочь на боковых полочках: ластик слева, флешка справа
        sx, (lz0, lz1), (bz0, _) = asm["side_x"], asm["tiers"][1], asm["tiers"][0]
        box("eraser", (13, 24, 8), (-sx, -10, lz0 + 4), material("eraser", (0.85, 0.42, 0.45), 0.7), (0, 0, math.radians(4)))
        box("usb", (12, 30, 5), (sx, -8, bz0 + 2.5), material("usb", (0.03, 0.12, 0.35), 0.35), (0, 0, math.radians(-3)))
    area_light("window", (-900, -500, 700), (0, 0, 40), 900, 55, (1.0, 0.96, 0.9))
    area_light("fill", (700, -700, 300), (0, 0, 40), 600, 12, (0.92, 0.95, 1.0))
    area_light("top", (0, -100, 900), (0, 0, 0), 500, 10)


def main():
    desk_scene()
    camera((95, -400, 250), (0, -5, 40), lens=48)
    shoot("hero.png", 1800, 1100)
    camera((0, -560, 190), (0, 0, 52), lens=62)
    shoot("front.png", 1800, 900)
    camera((-150, -175, 120), (-84, -30, 80), lens=70)
    shoot("detail_left.png", 1400, 900)
    camera((40, -230, 140), (0, -22, 78), lens=60)
    shoot("detail_center.png", 1400, 900)

    desk_scene(props=False)
    camera((0, 0, 700), (0, 0, 0), ortho=300)
    shoot("top.png", 1500, 600)
    camera((-260, 330, 230), (0, 0, 40), lens=50)
    shoot("rear_34.png", 1500, 1000)

    # стол принтера: детали на текстурном PEI в положении для печати
    reset()
    pla = material("pla", (0.045, 0.047, 0.05), rough=0.5)
    pei = material("pei", (0.10, 0.085, 0.06), rough=0.85)
    box("bed", (256, 256, 2), (128, 128, -1), pei)
    for x, y, part in asm["plate"]:
        name = "organizer_body.stl" if part == "body" else "organizer_drawer.stl"
        ob = import_stl(os.path.join(ROOT, "stl", name), part, pla)
        bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        cx = (min(v.x for v in bb) + max(v.x for v in bb)) / 2
        cy = (min(v.y for v in bb) + max(v.y for v in bb)) / 2
        ob.location += Vector((x * MM - cx, y * MM - cy, 0))
    area_light("key", (-300, -400, 600), (128, 128, 0), 600, 40)
    area_light("fill", (500, -200, 400), (128, 128, 0), 500, 12)
    camera((128, -190, 380), (128, 120, 0), lens=40)
    shoot("p1s_plate_1.png", 1024, 1024, transparent=True)
    print("RENDER OK")


main()
