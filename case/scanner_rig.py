"""
Рукоять, кронштейн и площадка датчиков ручного сканера.

Координаты — те же, что у корпуса (scanner_case.py): Z = 0 — низ платы,
дно корпуса на Z = −20. Сканер смотрит в +X: там площадка с лидаром и камерами,
она стоит у правого борта корпуса, где нет ни вентиляции, ни кнопки.
Оператор — со стороны −X, туда смотрит кнопка питания корпуса.
Разъёмы платы оказываются справа (−Y), выдув WELL — слева (+Y).

Детали:
  * рукоять — встаёт в байонет дна корпуса, поворот на 45° до щелчка;
    внутри канал под кабель питания и провод кнопки, под указательным пальцем —
    гнездо под тактовую кнопку 12 × 12, в торце — вставка 1/4″-20 под штатив;
  * кронштейн — язык зажат между дном корпуса и рукоятью, на шейке рукояти
    лыски, поэтому язык поворачивается вместе с ней и встаёт строго вперёд;
    спереди фланец с четырьмя термовставками M4;
  * площадка датчиков — снизу вверх D405, D455, лидар Unitree L2,
    сверху MaixCAM2 экраном к оператору. Камеры стоят на полках на винт 1/4″,
    лидар — на 4 × M3 с тыла площадки.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold, OpType

import scanner_case as sc
from scanner_case import box, loft, prism_xz, prism_yz


@dataclass
class Rig:
    # ---- байонет (под гнездо в дне корпуса) ----
    neck_clear: float = 0.4         # зазор шейки в отверстии Ø 30
    lug_w: float = 9.0              # ширина лепестка (паз 10)
    lug_t: float = 2.0
    lug_ramp: float = 0.3           # набегающий край лепестка выше — заходит на кольцо
    lug_clamp: float = 0.15         # натяг под лепестком в конце поворота
    lug_flat: float = 2.8           # плоская часть низа лепестка, дальше фаска 45°
    flat_w: float = 26.0            # лыски на шейке: язык кронштейна не проворачивается

    # ---- кронштейн ----
    tongue_t: float = 5.0
    arm_w: float = 70.0
    tongue_back: float = 32.0       # язык заходит за ось байонета назад
    flange_gap: float = 1.5         # от правого борта корпуса до фланца
    flange_t: float = 8.0           # под термовставки M4 длиной 8
    flange_z0: float = -46.0
    flange_z1: float = 14.0
    rib_t: float = 6.0
    insert_d: float = 5.6           # отверстие под термовставку M4
    bolt_y: float = 27.0
    bolt_z: tuple = (-38.0, 6.0)

    # ---- площадка датчиков ----
    plate_t: float = 4.0
    rim_d: float = 6.0              # бортик по контуру спереди — жёсткость
    rim_w: float = 3.0
    plate_w_low: float = 132.0      # ширина внизу (под D455)
    plate_w_high: float = 92.0      # ширина у лидара
    plate_z0: float = -46.0
    plate_neck_z: float = 66.0      # где площадка начинает сужаться
    shelf_t: float = 3.5
    tripod_d: float = 6.6           # паз под винт 1/4″-20
    cable_slot: tuple = (6.0, 22.0)

    # ---- датчики: размеры Y × Z × X и центры по Z ----
    d405: tuple = (42.0, 42.0, 23.0)
    d405_z: float = 0.0
    d455: tuple = (124.0, 29.0, 26.0)
    d455_z: float = 46.0
    lidar: tuple = (75.0, 75.0, 65.0)
    lidar_z: float = 113.0
    lidar_optical: float = 44.5     # оптический центр L2 от основания
    lidar_pcd: float = 51.0         # окружность отверстий M3 — сверить по чертежу L2
    lidar_hole: float = 3.4
    lidar_cbore: float = 6.5
    lidar_center_hole: float = 26.0
    maix: tuple = (66.5, 50.0, 21.2)
    # поля зрения по вертикали, градусы (полные)
    d455_vfov: float = 65.0
    d405_vfov: float = 58.0
    thermal_vfov: float = 42.0

    # ---- рукоять ----
    grip_len: float = 110.0
    grip_tilt: float = 12.0         # наклон назад, как у пистолетной рукояти
    grip_a: float = 16.0            # полуширина поперёк
    grip_b: float = 22.0            # полуглубина вперёд-назад
    groove: float = 2.2             # выемки под пальцы спереди
    palm: float = 1.8               # выпуклость под ладонь сзади
    pommel: float = 4.0             # расширение внизу
    head_r: float = 28.0            # опорный диск под языком
    head_h: float = 4.0
    flare_h: float = 13.0           # переход от рукояти к диску (не круче 45°)
    channel_d: float = 6.0          # канал под провод кнопки
    button: float = 12.4            # гнездо под тактовую кнопку 12 × 12; 0 — без кнопки
    button_depth: float = 7.0
    insert_14_d: float = 7.9        # термовставка 1/4″-20 в торце
    insert_14_len: float = 11.0

    ring_n: int = 120


def teardrop(cy, cz, rr, n=24):
    """Круг с острым верхом под 45° в плоскости (y, z): остриё в +y."""
    t = np.linspace(math.radians(45), math.radians(315), n)
    pts = [(cy + rr * math.cos(a), cz + rr * math.sin(a)) for a in t]
    return pts + [(cy + rr * math.sqrt(2), cz)]


# ============================================================ общие размеры


def frame(cp: sc.Params, r: Rig):
    """Ключевые координаты сборки."""
    z_floor = cp.z_bot                                   # низ дна корпуса
    z_ring_top = cp.z_bot + cp.wall + cp.bay_ring_t      # верх кольца байонета
    x_case = cp.outer_axes()[0]                          # правый борт корпуса
    x_f0 = x_case + r.flange_gap
    x_f1 = x_f0 + r.flange_t
    x_p0 = x_f1                                          # тыл площадки
    x_p1 = x_p0 + r.plate_t                              # лицо площадки, основание датчиков
    return dict(z_floor=z_floor, z_ring_top=z_ring_top, z_tongue0=z_floor - r.tongue_t,
                x_case=x_case, x_f0=x_f0, x_f1=x_f1, x_p0=x_p0, x_p1=x_p1)


# ============================================================ рукоять


def handle(cp: sc.Params, r: Rig, locked=True):
    f = frame(cp, r)
    z_head_top = f["z_tongue0"]                          # верх опорного диска
    z_head0 = z_head_top - r.head_h
    z_grip_top = z_head0 - r.flare_h
    z_grip0 = z_grip_top - r.grip_len
    tilt = math.tan(math.radians(r.grip_tilt))
    th = np.linspace(0, 2 * math.pi, r.ring_n, endpoint=False)
    c, s = np.cos(th), np.sin(th)

    def grip_ring(z):
        h = z - z_grip0
        rad = 1.0 / np.sqrt((c / r.grip_b) ** 2 + (s / r.grip_a) ** 2)
        # выемки под пальцы спереди (+X): три, сверху вниз
        g = 0.0
        for hc in (r.grip_len - 24, r.grip_len - 48, r.grip_len - 72):
            d = abs(h - hc)
            if d < 10:
                g += 0.5 * (1 + math.cos(math.pi * d / 10))
        rad = rad - r.groove * g * np.clip(c, 0, None) ** 2
        # ладонь сзади
        rad = rad + r.palm * math.sin(math.pi * min(max(h / r.grip_len, 0), 1)) * np.clip(-c, 0, None) ** 2
        # расширение внизу
        if h < 12:
            rad = rad + r.pommel * (1 - h / 12) ** 1.5
        xc = -(z_grip_top - z) * tilt
        return np.stack([xc + rad * c, rad * s], 1)

    def circle_ring(rr):
        return np.stack([rr * c, rr * s], 1)

    zs, rings = [], []
    for z in np.concatenate([np.linspace(z_grip0, z_grip0 + 12, 13), np.linspace(z_grip0 + 13, z_grip_top, 60)]):
        zs.append(z)
        rings.append(grip_ring(z))
    g_top = grip_ring(z_grip_top)
    for k in range(1, 14):
        t = k / 13
        zs.append(z_grip_top + t * r.flare_h)
        rings.append((1 - t) * g_top + t * circle_ring(r.head_r))
    zs.append(z_head_top)
    rings.append(circle_ring(r.head_r))
    body = loft(rings, np.array(zs))

    # шейка: лыски там, где язык кронштейна, выше — круглая, сверху лепестки
    rn = cp.bay_r - r.neck_clear
    neck_d = (CrossSection.circle(rn, 96) ^ CrossSection.square([r.flat_w, 2 * rn + 2]).translate([-r.flat_w / 2, -rn - 1]))
    lug_top = f["z_ring_top"] + r.lug_t
    # шейка с лысками по всей высоте: лыски смотрят по ±X, лепестки — на 45°, им не мешают
    neck = neck_d.extrude(lug_top - z_head_top + 0.5).translate([0, 0, z_head_top - 0.5])
    lugs = []
    r_out = cp.bay_notch_r - 0.6
    for k in range(4):
        ang = 45 + 90 * k
        w = r.lug_w / 2
        zt = f["z_ring_top"]
        r_fl = rn + r.lug_flat
        pts = []
        # низ лепестка: набегающий край (против часовой, +t) выше, задний — с натягом
        for tt, zb in ((-w, zt - r.lug_clamp), (w, zt + r.lug_ramp)):
            pts += [(rn - 0.5, tt, zb), (r_fl, tt, zb), (r_out, tt, zb + (r_out - r_fl)),
                    (rn - 0.5, tt, lug_top), (r_out, tt, lug_top)]
        lug = Manifold.hull_points(np.array(pts)).rotate([0, 0, ang])
        lugs.append(lug)
    head = neck + Manifold.batch_boolean(lugs, OpType.Add)
    h = body + head

    def capsule(p0, p1, d):
        a = Manifold.sphere(d / 2, 32).translate(list(p0))
        b = Manifold.sphere(d / 2, 32).translate(list(p1))
        return Manifold.batch_hull([a, b])

    def axis_x(z):
        return -(z_grip_top - z) * tilt

    # гнездо под тактовую кнопку 12 × 12 под указательным пальцем (крыша гнезда под 45°)
    # и канал под провод: от гнезда вверх под 45° к оси, по оси вверх, выход назад над рукой
    if r.button > 0:
        zb = z_grip0 + r.grip_len - 24
        xs = axis_x(zb) + r.grip_b - r.groove - r.button_depth
        hb = r.button / 2
        poly = [(-hb, -hb), (hb, -hb), (hb, hb), (0, hb + hb), (-hb, hb)]
        pocket = prism_yz([(y, zb + z) for y, z in poly], xs, xs + 40)
        d = r.channel_d
        a_pt = (xs + 1.0, 0.0, zb)
        dz = (xs + 1.0) - axis_x(zb)
        b_pt = (axis_x(zb + dz), 0.0, zb + dz)
        c_pt = (axis_x(z_grip_top - 6), 0.0, z_grip_top - 6)
        e_pt = (c_pt[0] - (r.grip_b + 10), 0.0, c_pt[2] + (r.grip_b + 10))
        h -= pocket + capsule(a_pt, b_pt, d) + capsule(b_pt, c_pt, d) + capsule(c_pt, e_pt, d)

    # вставка 1/4″-20 в торце
    xb = -(z_grip_top - z_grip0) * tilt
    h -= Manifold.cylinder(r.insert_14_len, r.insert_14_d / 2, r.insert_14_d / 2, 48).translate([xb, 0, z_grip0 - 0.01])

    if not locked:
        h = h.rotate([0, 0, -45])
    return h


# ============================================================ кронштейн


def arm(cp: sc.Params, r: Rig):
    f = frame(cp, r)
    zt0, zt1 = f["z_tongue0"], f["z_floor"]
    x_back = -r.tongue_back
    w = r.arm_w / 2
    tongue = box(x_back, f["x_f1"], -w, w, zt0, zt1)
    flange = box(f["x_f0"], f["x_f1"], -w, w, r.flange_z0, r.flange_z1)
    # косынка во всю ширину под языком (при печати на боку нет висящих рёбер), с окном
    x_r0 = r.head_r + 3.0
    rib_poly = [(x_r0, zt0 + 0.01), (f["x_f0"] + 0.01, zt0 + 0.01), (f["x_f0"] + 0.01, r.flange_z0)]
    gusset = prism_xz(rib_poly, -w, w)
    k = r.rib_t
    win = [(x_r0 + 2.4 * k, zt0 - k), (f["x_f0"] - k, zt0 - k), (f["x_f0"] - k, r.flange_z0 + 3.4 * k)]
    gusset -= prism_xz(win, -w - 1, w + 1)
    a = Manifold.batch_boolean([tongue, flange, gusset], OpType.Add)
    # отверстие под шейку рукояти — с лысками; сверху (при печати на боку — +Y) острый верх
    rn = cp.bay_r - r.neck_clear + 0.2
    hole = CrossSection.circle(rn, 96) ^ CrossSection.square([r.flat_w + 0.4, 2 * rn + 2]).translate(
        [-(r.flat_w + 0.4) / 2, -rn - 1])
    k45 = rn / math.sqrt(2)
    roof = CrossSection([np.array([(-k45, k45), (k45, k45), (0.0, rn * math.sqrt(2))])])
    hole = hole + (roof ^ CrossSection.square([r.flat_w + 0.4, 40]).translate([-(r.flat_w + 0.4) / 2, 0]))
    a -= hole.extrude(r.tongue_t + 2).translate([0, 0, zt0 - 1])
    # термовставки M4 во фланце (оси по X); при печати на боку — капли острым верхом (+Y)
    for yb in (-r.bolt_y, r.bolt_y):
        for zb in r.bolt_z:
            a -= prism_yz(teardrop(yb, zb, r.insert_d / 2), f["x_f0"] - 1, f["x_f1"] + 1)
    return a


# ============================================================ площадка датчиков


def _outline(r: Rig):
    """Контур площадки в плоскости (y, z)."""
    wl, wh = r.plate_w_low / 2, r.plate_w_high / 2
    z0, zn = r.plate_z0, r.plate_neck_z
    z_top = shelf_tops(r)["maix"] + 4.0                  # бортик-упор за MaixCAM2
    zt = zn + (wl - wh)                                  # сужение под 45°
    pts = [(-wl, z0), (wl, z0), (wl, zn), (wh, zt), (wh, z_top), (-wh, z_top), (-wh, zt), (-wl, zn)]
    cs = CrossSection([np.array(pts)])
    return cs.offset(-8, JoinType.Round, 2.0, 32).offset(8, JoinType.Round, 2.0, 32)


def shelf_tops(r: Rig):
    return {"d405": r.d405_z - r.d405[1] / 2,
            "d455": r.d455_z - r.d455[1] / 2,
            "maix": r.lidar_z + r.lidar[1] / 2 + 5.5}


def _yz_prism(cs: CrossSection, x0, x1):
    return cs.extrude(x1 - x0).rotate([90, 0, 90]).translate([x0, 0, 0])


def plate(cp: sc.Params, r: Rig):
    f = frame(cp, r)
    x0, x1 = f["x_p0"], f["x_p1"]
    out = _outline(r)
    p = _yz_prism(out, x0, x1)
    p += _yz_prism(out - out.offset(-r.rim_w, JoinType.Round, 2.0, 32), x1 - 0.01, x1 + r.rim_d)
    tops = shelf_tops(r)

    def shelf(zt, half_w, depth, gussets, x_from=x1):
        s = box(x_from - 0.01, x1 + depth, -half_w, half_w, zt - r.shelf_t, zt)
        for yg in gussets:
            g = [(x1 - 0.01, zt - r.shelf_t + 0.01), (x1 + depth * 0.8, zt - r.shelf_t + 0.01),
                 (x1 - 0.01, zt - r.shelf_t - depth * 0.8)]
            s += prism_xz(g, yg - 1.5, yg + 1.5)
        # паз под 1/4″: вдоль X, острый край к лицу площадки (при печати — вверх)
        xc = x1 + depth / 2
        rr = r.tripod_d / 2
        pts = [(xc + dx + rr * math.cos(t), rr * math.sin(t))
               for dx in (-3.0, 3.0) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
        pts.append((xc + 3.0 + rr * math.sqrt(2), 0.0))          # острый торец: печать без нависания
        slot = Manifold.hull_points(np.array([(x_, y_, zz) for x_, y_ in pts for zz in (zt - r.shelf_t - 1, zt + 1)]))
        return s - slot

    p += shelf(tops["d405"], r.d405[0] / 2 + 3, r.d405[2] + 1, (-r.d405[0] / 2 + 4, r.d405[0] / 2 - 4))
    p += shelf(tops["d455"], r.plate_w_low / 2 - 0.5, r.d455[2] + 1, (-60, -30, 30, 60))
    yg = r.lidar[0] / 2 + 3.5                               # косынки полки MaixCAM2 — по бокам от лидара
    p += shelf(tops["maix"], yg + 2.5, r.rim_d + r.maix[2] + 2.5, (-yg, yg))

    # лидар: 4 × M3 по окружности, зенковки с тыла, центральное окно
    holes = []
    for k in range(4):
        a = math.radians(45 + 90 * k)
        yy, zz = 0.5 * r.lidar_pcd * math.cos(a), r.lidar_z + 0.5 * r.lidar_pcd * math.sin(a)
        holes.append(Manifold.cylinder(r.plate_t + r.rim_d + 2, r.lidar_hole / 2, r.lidar_hole / 2, 24)
                     .rotate([0, 90, 0]).translate([x0 - 1, yy, zz]))
        # потай 90° с тыла (стенки 45° — печать тылом на стол без нависаний)
        cs_h = (r.lidar_cbore - r.lidar_hole) / 2
        holes.append(Manifold.cylinder(cs_h + 0.6, r.lidar_cbore / 2 + 0.6, r.lidar_hole / 2, 24)
                     .rotate([0, 90, 0]).translate([x0 - 0.6, yy, zz]))
    holes.append(Manifold.cylinder(r.plate_t + 2, r.lidar_center_hole / 2, r.lidar_center_hole / 2, 64)
                 .rotate([0, 90, 0]).translate([x0 - 1, 0, r.lidar_z]))
    # болты к фланцу кронштейна: M4, зенковка спереди
    for yb in (-r.bolt_y, r.bolt_y):
        for zb in r.bolt_z:
            holes.append(Manifold.cylinder(r.plate_t + 2, 2.25, 2.25, 24).rotate([0, 90, 0]).translate([x0 - 1, yb, zb]))
            holes.append(Manifold.cylinder(r.rim_d + 2, 4.3, 4.3, 32).rotate([0, 90, 0]).translate([x1 - 1.5, yb, zb]))
    # пазы под кабели
    sw, sh = r.cable_slot
    for yy, zz in ((-32, r.lidar_z - 10), (32, r.lidar_z - 10), (-52, 8.0), (52, 8.0)):
        sl = CrossSection.square([sw, sh]).translate([yy - sw / 2, zz - sh / 2]).offset(-2.9, JoinType.Round, 2.0, 16) \
            .offset(2.9, JoinType.Round, 2.0, 16)
        holes.append(_yz_prism(sl, x0 - 1, x1 + 1))
    p -= Manifold.batch_boolean(holes, OpType.Add)
    return p


# ============================================================ датчики (условно, для просмотра и проверок)


def sensors(cp: sc.Params, r: Rig):
    f = frame(cp, r)
    x1 = f["x_p1"]
    tops = shelf_tops(r)
    out = {}
    w, hh, d = r.d405
    out["d405"] = box(x1 + 0.5, x1 + 0.5 + d, -w / 2, w / 2, tops["d405"], tops["d405"] + hh)
    w, hh, d = r.d455
    out["d455"] = box(x1 + 0.5, x1 + 0.5 + d, -w / 2, w / 2, tops["d455"], tops["d455"] + hh)
    w, hh, d = r.lidar
    base = box(x1, x1 + 24, -w / 2, w / 2, r.lidar_z - hh / 2, r.lidar_z + hh / 2)
    dome = Manifold.cylinder(d - 24 - 12, 35, 33, 96).rotate([0, 90, 0]).translate([x1 + 24, 0, r.lidar_z]) + \
        Manifold.sphere(33, 64).scale([12 / 33, 1, 1]).translate([x1 + d - 12, 0, r.lidar_z])
    out["lidar"] = base + dome
    w, hh, d = r.maix
    xm = x1 + r.rim_d + 0.5                              # перед бортиком площадки
    cam = box(xm, xm + d, -w / 2, w / 2, tops["maix"], tops["maix"] + hh)
    cam += Manifold.cylinder(7, 7.5, 7.5, 48).rotate([0, 90, 0]).translate([xm + d, -12, tops["maix"] + hh / 2 + 6])
    cam += box(xm + d, xm + d + 10, 6, 26, tops["maix"] + 8, tops["maix"] + 30)
    out["maixcam"] = cam
    return out


SENSOR_MASS = {"lidar": 230, "d455": 116, "d405": 58, "maixcam": 55}


def build_rig(cp: sc.Params | None = None, r: Rig | None = None):
    cp = cp or sc.Params()
    r = r or Rig()
    return {"handle": handle(cp, r), "arm": arm(cp, r), "plate": plate(cp, r)}


def fov_report(cp: sc.Params, r: Rig):
    """Запас по высоте между верхней границей поля камеры и низом корпуса лидара (на его переднем крае)."""
    f = frame(cp, r)
    x1 = f["x_p1"]
    lidar_front = x1 + r.lidar[2]
    lidar_bottom = r.lidar_z - r.lidar[1] / 2
    out = {}
    for name, zc, depth, vfov in (("D455", r.d455_z, r.d455[2], r.d455_vfov),
                                  ("D405", r.d405_z, r.d405[2], r.d405_vfov)):
        x_cam = x1 + 0.5 + depth
        z_ray = zc + math.tan(math.radians(vfov / 2)) * (lidar_front - x_cam)
        out[name + "_до_лидара_мм"] = round(lidar_bottom - z_ray, 1)
    out["оптический_центр_лидара_X"] = round(x1 + r.lidar_optical, 1)
    out["самая_выступающая_камера_X"] = round(x1 + r.rim_d + 0.5 + r.maix[2] + 10, 1)
    return out
