"""
Корпус ручного сканера под Orange Pi 6 Plus — параметрическая модель.

Форма: асимметричная «галька». В плане суперэллипс: сзади, где разъёмы, он почти
прямоугольный (n = 12), спереди, где выдув, круглее (n = 7) и даёт камеру перед
рёбрами радиатора. К дну и крышке план ещё скругляется, бока сходят на нет по
дуге с фаской 45°, поэтому обе детали печатаются без поддержек.

Корпус без крепежа:
  * плата лежит углами на четырёх площадках с бортиками, сверху её прижимают
    четыре стойки крышки;
  * крышка садится на поясок и держится на четырёх защёлках;
  * кнопка питания — гибкий язычок в левом борту с толкателем изнутри, без
    пружин;
  * в дне — байонетное гнездо под будущую рукоять (4 положения через 90°).

Система координат: Z = 0 — нижняя плоскость платы. X — вдоль ребра с разъёмами
(115 мм). Разъёмы смотрят в −Y (назад), выдув радиатора — в +Y (вперёд).
Если смотреть на разъёмы, левый борт — это −X: там кнопка питания.

Все размеры в миллиметрах.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import os

import numpy as np
from manifold3d import CrossSection, FillRule, JoinType, Manifold, Mesh, OpType

SQ2 = math.sqrt(2.0)


@dataclass
class Params:
    # ---- Плата: замеры владельца ----
    board_x: float = 115.0          # длина, вдоль ребра с разъёмами
    board_y: float = 100.0          # ширина
    pcb_t: float = 1.6
    top_h: float = 33.0             # выше верха платы: радиатор с кулером
    bot_h: float = 14.0             # ниже платы: NVMe
    btn_from_corner: float = 12.0   # центр PWR от угла со стороны выдува, левое ребро

    # ---- Допущения: сверить с платой ----
    btn_z: float = 1.8              # центр толкателя PWR над верхом платы
    btn_protrude: float = 0.8       # насколько толкатель выступает за край платы
    hole_inset: float = 3.5         # центр крепёжного отверстия от краёв платы
    conn_h: float = 17.5            # самый высокий разъём над верхом платы
    conn_overhang: float = 1.5      # выступ разъёмов за край платы
    conn_corner: float = 5.0        # от угла платы до начала окна разъёмов

    # ---- Оболочка ----
    wall: float = 2.0
    side_gap: float = 4.0           # от бокового края платы до стенки
    corner_gap: float = 1.0         # минимальный зазор у углов платы
    r_corner: float = 14.0          # радиус скругления углов полости в плане (снаружи + стенка)
    n_rear: float = 40.0            # «ядро» плана: n ≥ 30 — прямоугольник (скруглённый план)
    n_front: float = 40.0
    front_extra: float = 2.5        # спереди полость глубже: камера перед рёбрами радиатора
    bot_gap: float = 4.0            # под зоной NVMe (там ходят лепестки байонета)
    top_gap: float = 4.0            # над кулером: камера забора воздуха
    bot_chamfer: float = 1.5        # профиль снизу: фаска 45° ...
    bot_radius: float = 5.0        # ... переходит в дугу до вертикали
    top_chamfer: float = 5.0         # фаска по верхней кромке крышки, как у образца
    top_radius: float = 3.0
    # нос: дуга в профиль (горизонтальный цилиндр вдоль X); 0 — без носа
    nose_r: float = 0.0
    # стык: утопленный пояс между крышкой и низом (верх нижней детали, другим цветом),
    # спереди в нём щели-жалюзи, как у образца
    ring_h: float = 7.1             # высота пояса
    ring_d: float = 1.0             # утопление пояса
    ring_inner: float = 0.6         # спереди стенка пояса утолщена внутрь на столько
    part_gap: float = 1.0           # разъём корпуса над самым высоким разъёмом

    # ---- Стык крышки ----
    fit: float = 0.15               # зазор пояска к стенке
    band_h: float = 4.0
    band_t: float = 1.4
    snap_y: tuple = (-28.0, 2.0)    # защёлки на обоих бортах, по Y
    snap_w: float = 9.0
    snap_len: float = 12.0
    snap_t: float = 1.4
    hook_d: float = 0.8
    hook_flat: float = 1.0
    slit: float = 0.8

    # ---- Крепление платы ----
    post_r: float = 2.5             # стойки крышки, прижим по контактным площадкам отверстий
    clamp_interf: float = 0.1
    pad: float = 7.0                # площадка под углом платы
    fence_h: float = 1.2            # бортик вокруг угла платы
    fence_t: float = 1.2
    board_fit: float = 0.2

    # ---- Кнопка питания ----
    tongue_back: float = 14.0       # от толкателя до корня язычка (к разъёмам)
    tongue_front: float = 6.0       # от толкателя до свободного конца
    tongue_h: float = 11.0
    tongue_slit: float = 1.2         # по вертикали; поперёк щели ≈ 0,85
    pocket_d: float = 0.6           # углубление площадки снаружи
    tongue_thin: float = 0.3        # подрезка язычка изнутри
    pre_gap: float = 0.5            # зазор толкателя до кнопки в покое
    nub_w: float = 6.0
    nub_h: float = 3.0

    # ---- Вентиляция: круглая решётка сверху, перфорация спереди и в дне ----
    grille_r: float = 46.0          # радиус круглой решётки в крышке
    hole_d: float = 2.0             # отверстия решётки
    hole_pitch: float = 3.0         # шаг колец и отверстий в кольце
    louver_y0: float = 46.0         # щели в поясе — спереди и на передних углах, от этой Y
    louver_h: float = 3.0           # высота щели по вертикали (поперёк наклона ≈ 2,1)
    louver_len: float = 11.0
    louver_bar: float = 2.6         # перемычка между щелями
    louver_ring_z: float = 16.5     # центр щелей в поясе (на середине стенки пояса)
    # надпись на стороне выдува — сквозные буквы, они же вентиляция (два яруса, как у образца)
    vent_text: str = "WELL"
    text_font: str = "fonts/cinzel-latin-900-normal.woff"
    text_cap: float = 11.0          # высота заглавных букв
    text_track: float = 2.0         # добавочный межбуквенный интервал
    text_grow: float = 0.15         # прорези шире контура буквы на столько (тонкие штрихи)
    text_z_lid: float = 28.6        # центр строки на крышке
    text_z_base: float = 6.0        # и на нижней детали
    floor_hole_d: float = 2.2

    # ---- Байонет под рукоять ----
    bay_r: float = 15.0
    bay_notch_r: float = 20.0
    bay_notch_w: float = 10.0
    bay_ring_r: float = 25.0
    bay_ring_t: float = 1.0

    # ---- Сетка ----
    ring_n: int = 288

    # ---------------- производные ----------------
    @property
    def bx2(self):
        return self.board_x / 2

    @property
    def by2(self):
        return self.board_y / 2

    @property
    def z_bot(self):
        return -(self.bot_h + self.bot_gap + self.wall)

    @property
    def z_top(self):
        return self.pcb_t + self.top_h + self.top_gap + self.wall

    @property
    def z_part(self):
        return self.pcb_t + self.conn_h + self.part_gap

    @property
    def btn_y(self):
        return self.by2 - self.btn_from_corner

    @property
    def btn_zc(self):
        return self.pcb_t + self.btn_z

    def outer_axes(self):
        """Наружные полуоси в самом широком сечении: A, B спереди, B сзади."""
        pl = plan(self)
        r0 = pl.r0
        return pl.a + r0, pl.bf + r0, pl.br + r0


# ============================================================ профиль


def _transition(h, chamfer, radius):
    """Отступ внутрь на высоте h от плоской грани: фаска 45°, затем дуга до вертикали."""
    hc = chamfer / SQ2
    d_max = (1 - 1 / SQ2) * radius + hc
    if h <= hc:
        return d_max - h
    hv = hc + radius / SQ2
    if h >= hv:
        return 0.0
    dz = hv - h
    return radius - math.sqrt(radius * radius - dz * dz)


def outer_delta(p: Params, z):
    """Отступ наружной поверхности по бокам и сзади."""
    return max(_transition(z - p.z_bot, p.bot_chamfer, p.bot_radius),
               _transition(p.z_top - z, p.top_chamfer, p.top_radius))


def transition_heights(p: Params):
    """Высоты, где боковина становится вертикальной (снизу и сверху)."""
    zb = p.z_bot + p.bot_chamfer / SQ2 + p.bot_radius / SQ2
    zt = p.z_top - p.top_chamfer / SQ2 - p.top_radius / SQ2
    return zb, zt


def _profile_z(p: Params, z0, z1, dense=1.0):
    zb, zt = transition_heights(p)
    zs = [z0]
    for a, b, n in ((z0, zb, 34), (zb, zt, 2), (zt, z1, 34)):
        a, b = max(a, z0), min(b, z1)
        if b > a:
            k = max(2, int(n * dense))
            zs.extend(np.linspace(a, b, k)[1:])
    zs = np.unique(np.round(np.array(zs), 6))
    return zs


def inner_delta_fn(p: Params, delta=None):
    """Отступ внутренней поверхности (от наружного максимума) — эквидистанта профиля."""
    delta = delta or outer_delta
    zs = np.linspace(p.z_bot, p.z_top, 1200)
    pts = [(-300.0, p.z_bot)] + [(-delta(p, z), z) for z in zs] + [(-300.0, p.z_top)]
    cs = CrossSection([np.array(pts)]).offset(-p.wall, JoinType.Round, 2.0, 96)
    polys = cs.to_polygons()
    edges = []
    for poly in polys:
        poly = np.asarray(poly)
        edges.append(np.stack([poly, np.roll(poly, -1, axis=0)], 1))
    edges = np.concatenate(edges)

    def f(z):
        z1, z2 = edges[:, 0, 1], edges[:, 1, 1]
        m = (np.minimum(z1, z2) <= z) & (np.maximum(z1, z2) >= z) & (z1 != z2)
        e = edges[m]
        t = (z - e[:, 0, 1]) / (e[:, 1, 1] - e[:, 0, 1])
        u = e[:, 0, 0] + t * (e[:, 1, 0] - e[:, 0, 0])
        return -float(u.max())

    return f


# ============================================================ сечения и сетки


def superellipse_pts(a, bf, br, nf, nr, m=4000):
    """Контур против часовой: спереди суперэллипс nf, сзади nr (n ≥ 30 — прямоугольник)."""
    out = []
    for b, n, front in ((bf, nf, True), (br, nr, False)):
        sgn = 1.0 if front else -1.0
        if n >= 30:
            half = np.array([[a, 0.0], [a, b], [-a, b]]) * [1.0, sgn]
            if not front:
                half = np.array([[-a, 0.0], [-a, -b], [a, -b]])
            out.append(half)
            continue
        th = np.linspace(0.0, math.pi, m // 2, endpoint=False) + (0.0 if front else math.pi)
        c, s = np.cos(th), np.sin(th)
        x = a * np.sign(c) * np.abs(c) ** (2.0 / n)
        y = b * np.sign(s) * np.abs(s) ** (2.0 / n)
        out.append(np.stack([x, y], 1))
    return np.concatenate(out)


class Plan:
    """План корпуса: выпуклое «ядро» (суперэллипс) + скругление.

    Любое сечение — эквидистанта ядра на расстоянии r. Точки колец берутся по
    направлению нормали φ: точка с нормалью φ = опорная точка ядра + r·n(φ).
    Так соседние кольца сопоставлены точно, а отступы внутрь до r0 не дают петель.
    """

    def __init__(self, p: "Params"):
        self.r0 = p.r_corner + p.wall
        self.a = p.bx2 + p.side_gap - p.r_corner
        reach = p.r_corner - p.corner_gap
        corner = np.array([p.bx2, p.by2])

        def fits(b, n):
            pts = superellipse_pts(self.a, b, b, n, n, 6000)
            pts = pts[pts[:, 1] >= 0]
            # сгущаем контур, чтобы расстояние до отрезков считалось честно
            q = np.vstack([pts, pts[:1]])
            seg = np.linspace(0, 1, 12)[:, None, None]
            dense = (q[:-1][None] * (1 - seg) + q[1:][None] * seg).reshape(-1, 2)
            return np.min(np.hypot(*(dense - corner).T)) <= reach

        def solve(n):
            lo, hi = 1.0, 200.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if fits(mid, n) else (mid, hi)
            return hi

        self.br = solve(p.n_rear)
        self.bf = solve(p.n_front) + p.front_extra
        core = superellipse_pts(self.a, self.bf, self.br, p.n_front, p.n_rear, 6000)
        phi = np.linspace(0.0, 2 * math.pi, 16384, endpoint=False)
        nrm = np.stack([np.cos(phi), np.sin(phi)], 1)
        idx = np.argmax(core @ nrm.T, axis=0)
        sup = core[idx]
        pmax = sup + self.r0 * nrm
        self._dense = (phi, sup, nrm)

        def pick(n_pts):
            out = []
            for lo, hi in ((0.0, math.pi), (math.pi, 2 * math.pi)):
                m = (phi >= lo) & (phi < hi)
                ph, pp = phi[m], pmax[m]
                ds = np.hypot(*np.diff(pp, axis=0).T)
                w = np.concatenate([[0.0], np.cumsum(ds / ds.sum() + 0.8 * np.diff(ph) / (hi - lo))])
                w /= w[-1]
                out.append(np.interp(np.linspace(0, 1, n_pts // 2, endpoint=False), w, ph))
            ph = np.concatenate(out)
            nn = np.stack([np.cos(ph), np.sin(ph)], 1)
            return core[np.argmax(core @ nn.T, axis=0)], nn

        self.c_k, self.n_k = pick(p.ring_n)
        self.c_cs, self.n_cs = pick(720)

    @staticmethod
    def _pts(base, nrm, r0, inset):
        r = r0 - inset
        if r < -1e-6:
            raise ValueError(f"отступ {inset:.2f} больше радиуса плана {r0:.2f}")
        return base + max(r, 0.0) * nrm

    def ring(self, inset):
        return self._pts(self.c_k, self.n_k, self.r0, inset)

    def poly(self, inset):
        return self._pts(self.c_cs, self.n_cs, self.r0, inset)


_PLAN_CACHE = {}


def plan(p: "Params") -> Plan:
    key = (p.board_x, p.board_y, p.side_gap, p.corner_gap, p.r_corner, p.n_front, p.n_rear, p.wall, p.ring_n,
           p.front_extra)
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = Plan(p)
    return _PLAN_CACHE[key]


def section_cs(p: Params, inset):
    return CrossSection([plan(p).poly(inset)])


def _area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def ring_at(p: Params, inset):
    return plan(p).ring(inset)


def manifold_from(verts, tris):
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.uint32)
    m = Manifold(Mesh(vert_properties=verts, tri_verts=tris))
    if m.status().name != "NoError":
        raise RuntimeError(f"bad mesh: {m.status()}")
    return m


def loft(rings, zs):
    k, n = len(zs), len(rings[0])
    v = np.zeros((k * n + 2, 3))
    for i, (r, z) in enumerate(zip(rings, zs)):
        v[i * n:(i + 1) * n, :2] = r
        v[i * n:(i + 1) * n, 2] = z
    cb, ct = k * n, k * n + 1
    v[cb] = (0, 0, zs[0])
    v[ct] = (0, 0, zs[-1])
    j = np.arange(n)
    j1 = (j + 1) % n
    tris = []
    for i in range(k - 1):
        a, b, c, d = i * n + j, i * n + j1, (i + 1) * n + j1, (i + 1) * n + j
        tris += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    tris.append(np.stack([np.full(n, cb), j1, j], 1))
    tris.append(np.stack([np.full(n, ct), (k - 1) * n + j, (k - 1) * n + j1], 1))
    return manifold_from(v, np.concatenate(tris))


def nose_ring(p: Params, z, plan_inset, nose_inset):
    """Кольцо сечения на высоте z: план со сдвигом plan_inset, спереди срезан носом."""
    r = plan(p).ring(plan_inset).copy()
    r[:, 1] = np.minimum(r[:, 1], nose_y(p, z, nose_inset))
    return r


def outer_body(p: Params, inset=0.0):
    zs = _nose_z(p, _profile_z(p, p.z_bot + inset, p.z_top - inset))
    return loft([nose_ring(p, z, outer_delta(p, z) + inset, inset) for z in zs], zs)


def cavity_body(p: Params):
    din = inner_delta_fn(p)
    zs = _nose_z(p, _profile_z(p, p.z_bot + p.wall, p.z_top - p.wall))
    return loft([nose_ring(p, z, din(z), p.wall) for z in zs], zs)


def _nose_z(p: Params, zs, step=1.0):
    """Досыпает высоты, чтобы дуга носа была гладкой."""
    z0, z1 = float(zs[0]), float(zs[-1])
    extra = np.arange(z0, z1, step)
    return np.unique(np.round(np.concatenate([zs, extra, [z1]]), 6))


def nose_axis(p: Params):
    """Ось цилиндра носа (y, z): касается самой передней точки корпуса."""
    _, bf0, _ = p.outer_axes()
    zc = 0.5 * (p.z_bot + p.z_top)
    return bf0 - p.nose_r, zc


def nose_y(p: Params, z, inset=0.0):
    """Y поверхности носа на высоте z (без носа — бесконечность)."""
    if p.nose_r <= 0:
        return 1e9
    yc, zc = nose_axis(p)
    r = p.nose_r - inset
    return yc + math.sqrt(max(r * r - (z - zc) ** 2, 0.0))


def section_prism(p: Params, inset, z0, z1):
    """Вертикальная призма по самому широкому сечению, сдвинутому внутрь на inset."""
    return section_cs(p, inset).extrude(z1 - z0).translate([0, 0, z0])


def nose_prism(p: Params, inset, z0, z1, step=1.0):
    """Как section_prism, но спереди срезано носом (сдвинутым на тот же inset)."""
    zs = np.unique(np.concatenate([np.arange(z0, z1, step), [z1]]))
    return loft([nose_ring(p, z, inset, inset) for z in zs], zs)


def nose_frustum(p: Params, inset0, z0, inset1, z1):
    return loft([nose_ring(p, z0, inset0, inset0), nose_ring(p, z1, inset1, inset1)], [z0, z1])


def _line_hits(poly, axis, value):
    q = np.vstack([poly, poly[:1]])
    a, b = q[:-1], q[1:]
    o = 1 - axis
    m = (np.minimum(a[:, axis], b[:, axis]) <= value) & (np.maximum(a[:, axis], b[:, axis]) >= value) & \
        (a[:, axis] != b[:, axis])
    t = (value - a[m, axis]) / (b[m, axis] - a[m, axis])
    return a[m, o] + t * (b[m, o] - a[m, o])


def wall_x(p: Params, y, inset_from_outer):
    """|X| стенки при данном y для сечения со сдвигом inset (правый борт; левый симметричен)."""
    return float(_line_hits(plan(p).poly(inset_from_outer), 1, y).max())


def points_in_poly(pts, poly):
    pts = np.asarray(pts, float)
    x, y = pts[:, 0:1], pts[:, 1:2]
    q = np.vstack([poly, poly[:1]])
    x0, y0, x1, y1 = q[:-1, 0], q[:-1, 1], q[1:, 0], q[1:, 1]
    cond = (y0 <= y) != (y1 <= y)
    xi = x0 + (y - y0) * (x1 - x0) / np.where(y1 != y0, y1 - y0, 1e-12)
    return (np.sum(cond & (x < xi), axis=1) % 2) == 1


def dist_to_poly(pts, poly):
    pts = np.asarray(pts, float)
    q = np.vstack([poly, poly[:1]])
    a, b = q[:-1], q[1:]
    ab = b - a
    ap = pts[:, None, :] - a[None]
    t = np.clip(np.sum(ap * ab[None], 2) / np.maximum(np.sum(ab * ab, 1), 1e-12)[None], 0, 1)
    d = ap - t[..., None] * ab[None]
    return np.sqrt(np.min(np.sum(d * d, 2), axis=1))


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def _ccw(poly):
    poly = np.asarray(poly, float)
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if area > 0 else poly[::-1]


def prism_xz(poly, y0, y1):
    """Многоугольник в плоскости (x, z), вытянутый по Y от y0 до y1."""
    return CrossSection([_ccw(poly)]).extrude(y1 - y0).rotate([90, 0, 0]).translate([0, y1, 0])


def prism_yz(poly, x0, x1):
    """Многоугольник в плоскости (y, z), вытянутый по X от x0 до x1."""
    return CrossSection([_ccw(poly)]).extrude(x1 - x0).rotate([90, 0, 90]).translate([x0, 0, 0])


def polygon_prisms(polys, axis, a0, a1):
    """Пачка призм из выпуклых многоугольников (против часовой) одним мешем.

    axis='z': многоугольник в (x, y), призма по Z от a0 до a1.
    """
    verts, tris, o = [], [], 0
    for q in polys:
        q = _ccw(q)
        n = len(q)
        for w in (a0, a1):
            for (u, v) in q:
                verts.append((u, v, w) if axis == "z" else (u, w, v))
        for k in range(1, n - 1):
            tris.append((o, o + k + 1, o + k))
            tris.append((o + n, o + n + k, o + n + k + 1))
        for k in range(n):
            k1 = (k + 1) % n
            tris += [(o + k, o + k1, o + n + k1), (o + k, o + n + k1, o + n + k)]
        o += 2 * n
    tris = np.array(tris)
    if axis == "y":            # (u, v, w) -> (x, w, z) — зеркальное отображение
        tris = tris[:, ::-1]
    return manifold_from(np.array(verts), tris)


class _FlatPen:
    """Перо fontTools, которое раскладывает кривые глифа в ломаные."""

    def __new__(cls, glyphset, steps=10):
        from fontTools.pens.basePen import BasePen

        class Pen(BasePen):
            def __init__(self):
                super().__init__(glyphset)
                self.contours, self.cur = [], []

            def _moveTo(self, pt):
                self.cur = [pt]

            def _lineTo(self, pt):
                self.cur.append(pt)

            def _curveToOne(self, p1, p2, p3):
                p0 = self.cur[-1]
                for t in np.linspace(0, 1, steps + 1)[1:]:
                    a = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
                    self.cur.append(tuple(a[0] * np.array(p0) + a[1] * np.array(p1)
                                          + a[2] * np.array(p2) + a[3] * np.array(p3)))

            def _qCurveToOne(self, p1, p2):
                p0 = self.cur[-1]
                for t in np.linspace(0, 1, steps + 1)[1:]:
                    self.cur.append(tuple((1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1)
                                          + t * t * np.array(p2)))

            def _closePath(self):
                if len(self.cur) > 2:
                    self.contours.append(np.array(self.cur, float))
                self.cur = []

            _endPath = _closePath

        return Pen()


def text_contours(p: Params):
    """Контуры надписи в мм: x — вдоль строки (центр в 0), y — вверх от низа заглавных."""
    from fontTools.ttLib import TTFont
    path = p.text_font if os.path.isabs(p.text_font) else os.path.join(os.path.dirname(__file__), p.text_font)
    font = TTFont(path)
    gs, cmap = font.getGlyphSet(), font.getBestCmap()
    k = p.text_cap / font["OS/2"].sCapHeight
    out, x = [], 0.0
    for ch in p.vent_text:
        g = gs[cmap[ord(ch)]]
        pen = _FlatPen(gs)
        g.draw(pen)
        out += [c * k + [x, 0.0] for c in pen.contours]
        x += g.width * k + p.text_track
    width = x - p.text_track
    return [c - [width / 2, 0.0] for c in out], width


def text_cut(p: Params, z_center, y_face):
    """Сквозные буквы в лицевой стенке (+Y). Читаются слева направо, если смотреть спереди."""
    contours, _ = text_contours(p)
    polys = [np.stack([-c[:, 0], z_center - p.text_cap / 2 + c[:, 1]], 1) for c in contours]
    cs = CrossSection(polys, FillRule.NonZero)
    if p.text_grow > 0:
        cs = cs.offset(p.text_grow, JoinType.Round, 2.0, 16)
    y0, y1 = y_face - p.wall - 0.8, y_face + 1.0
    return cs.extrude(y1 - y0).rotate([90, 0, 0]).translate([0, y1, 0])


def build(p: Params | None = None):
    p = p or Params()
    a0, bf0, br0 = p.outer_axes()
    zp = p.z_part
    zb_v, zt_v = transition_heights(p)
    big = 400.0
    W = p.wall

    def sp(inset, z0, z1):
        """Сечение со сдвигом inset внутрь, с учётом носа."""
        return nose_prism(p, inset, z0, z1)

    outer = outer_body(p)
    cavity = cavity_body(p)
    shell = outer - cavity

    bottom = shell ^ box(-big, big, -big, big, p.z_bot - 1, zp)
    lid = shell ^ box(-big, big, -big, big, zp, p.z_top + 1)

    # --- пояс на стыке: верх нижней детали утоплен на ring_d (печатается другим цветом),
    #     нижняя кромка крышки — с фаской 0,5 мм, крышка «парит» над поясом
    zr = zp - p.ring_h
    bottom -= sp(-1.0, zr, zp + 1.0) - sp(p.ring_d, zr - 1.0, zp + 2.0)
    lid -= sp(-1.0, zp - 1.0, zp + 0.5) - sp(0.5, zp - 2.0, zp + 1.0)
    # спереди, где в поясе щели, его стенка утолщена внутрь; снизу упор под 45°
    w_in = W + p.ring_inner
    front_zone = box(-big, big, p.louver_y0 - 2.0, big, -big, big)
    thick = (sp(W - 0.05, zr, zp) - sp(w_in, zr - 1.0, zp + 1.0)) + \
        (sp(W - 0.05, zr - (p.ring_inner + 0.05) * 1.1, zr + 0.01)
         - nose_frustum(p, W - 0.05, zr - (p.ring_inner + 0.05) * 1.1 - 0.001, w_in, zr + 0.011))
    bottom += thick ^ front_zone

    # --- окно разъёмов сзади: открыто вверх, сверху его закрывает крышка
    hw = p.bx2 - p.conn_corner
    zn0 = p.pcb_t - 0.6
    r = 2.5
    notch_poly = ([(-hw, zp + 2), (-hw, zn0 + r)]
                  + [(-hw + r + r * math.cos(t), zn0 + r + r * math.sin(t)) for t in np.linspace(math.pi, 1.5 * math.pi, 10)]
                  + [(hw - r + r * math.cos(t), zn0 + r + r * math.sin(t)) for t in np.linspace(1.5 * math.pi, 2 * math.pi, 10)]
                  + [(hw, zp + 2)])
    bottom -= prism_xz(notch_poly, -(br0 + 5), -(p.by2 - 2))

    # --- поясок крышки и кольцо жёсткости (сверху скошено под 45° — печать вверх дном)
    band = (sp(W + p.fit, zp - p.band_h, zp + 0.5)
            - sp(W + p.fit + p.band_t, zp - p.band_h - 1, zp + 1))
    inner_edge = W + p.fit + p.band_t
    # кольцо жёсткости: сверху скошено под 45° (крышка печатается вверх дном)
    ch = (inner_edge - (W - 0.05)) * 1.15
    conn_ring = (sp(W - 0.6, zp, zp + 1.0 + ch) - sp(inner_edge, zp - 1, zp + 1.0)
                 - nose_frustum(p, inner_edge, zp + 1.0 - 0.001, W - 0.05, zp + 1.0 + ch + 0.001))
    conn_ring = conn_ring ^ outer
    band -= box(-(hw + 1.5), hw + 1.5, -big, -(p.by2 - 6), zp - p.band_h - 1, zp + 0.6)
    band -= box(-big, big, p.louver_y0 - 3.0, big, zp - p.band_h - 1, zp + 0.6)   # спереди щели пояса

    # --- защёлки: язычки из пояска вниз, крючок наружу; в стенке низа — гнездо
    arms, recesses, slits = [], [], []
    for sx in (-1, 1):
        for y0 in p.snap_y:
            xw = wall_x(p, y0, W)                      # внутренняя грань стенки
            xo = xw - p.fit                            # наружная грань язычка
            xi = xo - p.snap_t
            yl, yh = y0 - p.snap_w / 2, y0 + p.snap_w / 2
            zl = zp - p.snap_len
            arm = box(xi, xo + 0.01, yl, yh, zl, zp + 0.5)
            zb = zl + 0.3
            hd, hf = p.hook_d, p.hook_flat
            hook_poly = [(0.0, zb), (hd, zb + hd), (hd, zb + hd + hf), (0.0, zb + 2 * hd + hf)]
            hook = prism_xz([(xo - 0.01 + n, z) for n, z in hook_poly], yl, yh)
            zc = zb + hd + hf + 0.25
            n1 = hd - p.fit
            rd = n1 + 0.2
            rec_poly = [(-0.5, zb - 0.5), (rd, zb - 0.5), (rd, zc - 0.2), (-0.5, zc + n1 + 0.5)]
            rec = prism_xz([(xw + n, z) for n, z in rec_poly], yl - 0.5, yh + 0.5)
            sl = box(xi - 0.5, xo + 0.5, yl - p.slit, yl, zl - 1, zp) + \
                box(xi - 0.5, xo + 0.5, yh, yh + p.slit, zl - 1, zp)
            if sx < 0:
                arm, hook, rec, sl = (m.mirror([1, 0, 0]) for m in (arm, hook, rec, sl))
            arms.append(arm + hook)
            recesses.append(rec)
            slits.append(sl)
    band = band - Manifold.batch_boolean(slits, OpType.Add)
    lid = Manifold.batch_boolean([lid, band, conn_ring] + arms, OpType.Add)
    bottom -= Manifold.batch_boolean(recesses, OpType.Add)

    # --- угловые площадки с бортиками (низ) и прижимные стойки (крышка)
    supports, posts = [], []
    zf = p.z_bot + W - 0.5
    for sx in (-1, 1):
        for sy in (-1, 1):
            bx, by = p.bx2, p.by2               # строим для угла (+, +), потом отражаем
            hx, hy = bx - p.hole_inset, by - p.hole_inset
            o = p.fence_t + p.board_fit
            low = box(bx - 5.5, bx + o, by - 5.5, by + o, zf, -3.5)
            neck = box(bx - 5.5, bx + o, by - 5.5, by + o, -3.6, -3.5)
            top = box(bx - p.pad, bx + o, by - p.pad, by + o, -2.0, 0.0)
            col = low + Manifold.batch_hull([neck, top])
            fx = box(bx + p.board_fit, bx + o, by - p.pad, by + o, -0.5, p.fence_h)
            fy = box(bx - p.pad, bx + o, by + p.board_fit, by + o, -0.5, p.fence_h)
            web = box(-1.0, 1.0, -1.0, 20.0, zf, -1.0).rotate([0, 0, -45]).translate([bx - 2, by - 2, 0])
            sup = col + fx + fy + web
            post = Manifold.cylinder(p.z_top - (p.pcb_t - p.clamp_interf), p.post_r, p.post_r, 32).translate(
                [hx, hy, p.pcb_t - p.clamp_interf])
            fin = box(-0.8, 0.8, 0.0, 25.0, zp + 1.0, p.z_top).rotate([0, 0, -45]).translate([hx, hy, 0])
            pst = post + fin
            if sx < 0:
                sup, pst = sup.mirror([1, 0, 0]), pst.mirror([1, 0, 0])
            if sy < 0:
                sup, pst = sup.mirror([0, 1, 0]), pst.mirror([0, 1, 0])
            supports.append(sup)
            posts.append(pst)
    bottom += Manifold.batch_boolean(supports, OpType.Add) ^ outer
    lid += Manifold.batch_boolean(posts, OpType.Add) ^ outer

    # --- кнопка питания: язычок в левом борту (−X), корень к разъёмам
    yb, zc = p.btn_y, p.btn_zc
    y_root, y_free = yb - p.tongue_back, yb + p.tongue_front
    z0t, z1t = zc - p.tongue_h / 2, zc + p.tongue_h / 2
    s = p.tongue_slit
    xa, xb = -(a0 + 5), -(p.bx2 + 1.6)          # насквозь через стенку

    # стенка в зоне язычка: от наружной грани до внутренней
    ys = np.linspace(y_root - 2, y_free + 2, 12)
    x_out = max(wall_x(p, y, 0.0) for y in ys) + 0.6
    x_in = min(wall_x(p, y, W) for y in ys) - 0.6

    def hslit(z_mid, ya, yb_):
        # горизонтальная щель идёт сквозь стенку под 45° (вверх внутрь):
        # и потолок щели, и низ язычка печатаются без мостов и поддержек
        xm = -(x_out + x_in) / 2
        poly = []
        for xx in (-x_out, -x_in):
            zc_ = z_mid + (xx - xm)
            poly.append((xx, zc_ - s / 2))
        for xx in (-x_in, -x_out):
            zc_ = z_mid + (xx - xm)
            poly.append((xx, zc_ + s / 2))
        return prism_xz(poly, ya, yb_)

    top_slit = hslit(z1t + s / 2, y_root, y_free + s)
    bot_slit = hslit(z0t - s / 2, y_root, y_free + s)
    dz = (x_out - x_in) / 2 + s
    end_slit = prism_yz([(y_free, z0t - dz), (y_free + s, z0t - dz), (y_free + s, z1t + dz),
                         (y_free + s / 2, z1t + dz + s / 2), (y_free, z1t + dz)], -x_out, -x_in)
    reg = CrossSection.square([p.tongue_back + p.tongue_front + 5.0, p.tongue_h + 2 * s + 5.0]) \
        .translate([y_root - 2.5, z0t - s - 2.5])
    reg = reg.offset(-3.0, JoinType.Round, 2.0, 32).offset(3.0, JoinType.Round, 2.0, 32)
    pocket_box = reg.extrude(a0 + 5).rotate([90, 0, 90]).translate([-(a0 + 5), 0, 0])
    pocket = (sp(-2.0, z0t - 6, z1t + 6) - sp(p.pocket_d, z0t - 7, z1t + 7)) ^ pocket_box
    # верхний край углубления — фаской 45°, чтобы не было нависающей ступеньки
    z_top_r = z0t - s - 2.5 + p.tongue_h + 2 * s + 5.0
    x_o = max(wall_x(p, y, 0.0) for y in ys)
    slant = box(-big, big, -big, big, -big, 0).rotate([0, 45, 0]).translate([-x_o, 0, z_top_r])
    pocket = pocket ^ slant
    # подрезка язычка изнутри
    skin = sp(W - p.tongue_thin, z0t, z1t) - sp(W + 1.0, z0t - 1, z1t + 1)
    thin = skin ^ box(-big, -30.0, y_root + 0.5, y_free, z0t, z1t)
    # тактильная точка на площадке
    x_pocket = -wall_x(p, yb, p.pocket_d)
    dot = Manifold.sphere(1.6, 32).scale([0.28, 1, 1]).translate([x_pocket + 0.05, yb, zc])
    # толкатель от язычка к кнопке, низ под 45°
    x_wall_in = -wall_x(p, yb, W - p.tongue_thin)
    x_tip = -(p.bx2 + p.btn_protrude + p.pre_gap)
    nub_len = x_tip - x_wall_in
    hw_n, hh_n = p.nub_w / 2, p.nub_h / 2
    z_nub_low = max(zc - hh_n - nub_len - 0.6, z0t + 0.3)
    nub_pts = []
    for yy in (yb - hw_n, yb + hw_n):
        nub_pts += [(x_wall_in - 0.6, yy, zc + hh_n), (x_wall_in - 0.6, yy, z_nub_low),
                    (x_tip, yy, zc + hh_n), (x_tip, yy, zc - hh_n)]
    nub = Manifold.hull_points(np.array(nub_pts))
    slits_btn = Manifold.batch_boolean([top_slit, bot_slit, end_slit], OpType.Add)
    bottom = bottom - pocket - slits_btn - thin + nub + dot

    # --- вентиляция
    def circle(u, v, r, n=14):
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.stack([u + r * np.cos(a), v + r * np.sin(a)], 1)

    def drop2(u, v, r, n=5):
        # круг с остриями сверху и снизу под 45°: печатается без мостов
        # и в нижней части (дном на стол), и в крышке (верхом на стол)
        right = [(u + r * math.cos(t), v + r * math.sin(t)) for t in np.linspace(-0.25 * math.pi, 0.25 * math.pi, n)]
        left = [(u + r * math.cos(t), v + r * math.sin(t)) for t in np.linspace(0.75 * math.pi, 1.25 * math.pi, n)]
        return np.array(right + [(u, v + r * SQ2)] + left + [(u, v - r * SQ2)])

    def rings_of_holes(center, r_max, pitch, keep):
        out = [center] if keep(*center) else []
        k = 1
        while k * pitch <= r_max:
            rr = k * pitch
            n = max(6, int(round(2 * math.pi * rr / pitch)))
            off = 0.5 * (k % 2) * 2 * math.pi / n
            for t in off + np.arange(n) * 2 * math.pi / n:
                u, v = center[0] + rr * math.cos(t), center[1] + rr * math.sin(t)
                if keep(u, v):
                    out.append((u, v))
            k += 1
        return out

    din = inner_delta_fn(p)

    # сверху — большая круглая решётка: забор воздуха кулером
    ceil_poly = plan(p).poly(din(p.z_top - W - 0.02) + 1.5)
    yc_top = 0.5 * (ceil_poly[:, 1].max() + ceil_poly[:, 1].min())
    top_pts = rings_of_holes((0.0, yc_top), p.grille_r, p.hole_pitch, lambda u, v: True)
    lid -= polygon_prisms([circle(u, v, p.hole_d / 2) for u, v in top_pts], "z",
                          p.z_top - W - 1.0, p.z_top + 1.0)
    n_top = len(top_pts)

    # спереди — щели-жалюзи: ряд в поясе на стыке (спереди и на углах) и два ряда на лицевой стенке.
    # Щель идёт сквозь стенку под 45° вверх внутрь: потолок щели печатается без мостов,
    # а сверху внутрь не заглянуть.
    pitch = p.louver_len + p.louver_bar
    hh = p.louver_h / 2
    louvers = []

    def slot_hull(path_pts, normals, i_mid, half_t, zc):
        # path_pts — точки середины стенки (инсет i_mid) вдоль щели; n — наружу
        pts = []
        for q, n in zip(path_pts, normals):
            for dn in (-half_t - 0.6, half_t + 0.6):         # + наружу, − внутрь
                z0 = zc - dn                                   # вверх внутрь под 45°
                pts += [(q[0] + n[0] * dn, q[1] + n[1] * dn, z0 - hh),
                        (q[0] + n[0] * dn, q[1] + n[1] * dn, z0 + hh)]
        return Manifold.hull_points(np.array(pts))

    # путь по середине стенки пояса: спереди от центра к углу и вниз по боку до louver_y0
    i_mid = 0.5 * (p.ring_d + w_in)
    ring_pts = plan(p).poly(i_mid)
    half = ring_pts[:len(ring_pts) // 2]                 # передняя половина, от +X против часовой
    path = half[(half[:, 0] >= -1e-6)][::-1]             # от верха (x≈0) к +X
    path = np.vstack([[0.0, path[0, 1]], path]) if path[0, 0] > 1e-6 else path
    path = path[path[:, 1] >= p.louver_y0]
    seg = np.diff(path, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    s_max = cum[-1] - 1.5
    n_ring = 0
    k = 0
    while True:
        s0 = pitch / 2 + k * pitch - p.louver_len / 2
        s1 = s0 + p.louver_len
        if s1 > s_max:
            break
        ss = np.linspace(s0, s1, 9)
        q = np.stack([np.interp(ss, cum, path[:, 0]), np.interp(ss, cum, path[:, 1])], 1)
        t = np.gradient(q, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        nrm = np.stack([t[:, 1], -t[:, 0]], 1)            # наружу (обход против часовой)
        for sxm in (1, -1):
            qq, nn = q * [sxm, 1], nrm * [sxm, 1]
            louvers.append(slot_hull(qq, nn, i_mid, 0.5 * (w_in - p.ring_d), p.louver_ring_z))
            n_ring += 1
        k += 1

    bottom -= Manifold.batch_boolean(louvers, OpType.Add)

    # надпись на стороне выдува: сквозные буквы в крышке и в нижней детали
    y_face = plan(p).poly(0.0)[:, 1].max()
    lid_text = text_cut(p, p.text_z_lid, y_face)
    base_text = text_cut(p, p.text_z_base, y_face)
    text_area = lid_text.volume() / (p.wall + 1.8)
    lid -= lid_text
    bottom -= base_text

    # дно — продув NVMe кольцами вокруг байонета
    floor_poly = plan(p).poly(din(p.z_bot + W + 0.02) + 2.0)

    def floor_keep(u, v):
        if math.hypot(u, v) < p.bay_ring_r + 3.0:
            return False
        if abs(u) > p.bx2 - 9.5 and abs(v) > p.by2 - 9.5:
            return False
        q = np.array([[u, v]])
        return bool(points_in_poly(q, floor_poly)[0]) and dist_to_poly(q, floor_poly)[0] > p.floor_hole_d
    fl_pts = rings_of_holes((0.0, 0.0), 70.0, p.hole_pitch * 1.25, floor_keep)
    bottom -= polygon_prisms([circle(u, v, p.floor_hole_d / 2) for u, v in fl_pts], "z",
                             p.z_bot - 1.0, p.z_bot + W + 0.5)
    stats = {"отверстий_сверху": n_top, "щелей_в_поясе": n_ring,
             "площадь_одной_надписи_мм2": round(text_area, 0), "отверстий_в_дне": len(fl_pts)}

    # --- байонет под рукоять
    zf_in = p.z_bot + W
    ring = Manifold.cylinder(p.bay_ring_t + 0.5, p.bay_ring_r, p.bay_ring_r, 96).translate([0, 0, zf_in - 0.5])
    hole2d = CrossSection.circle(p.bay_r, 96)
    for k in range(4):
        hole2d = hole2d + CrossSection.square([p.bay_notch_r + 1.0, p.bay_notch_w]) \
            .translate([0, -p.bay_notch_w / 2]).rotate(90 * k)
    hole = hole2d.extrude(W + p.bay_ring_t + 3).translate([0, 0, p.z_bot - 1])
    stops, bumps = [], []
    lug_half = math.degrees(math.asin((p.bay_notch_w / 2 - 0.5) / p.bay_r))
    for k in range(4):
        ang = 90 * k + 45 + lug_half + 4.0
        stops.append(box(p.bay_r + 0.8, p.bay_ring_r - 0.5, -1.2, 1.2, zf_in, zf_in + p.bay_ring_t + 2.2)
                     .rotate([0, 0, ang]))
        ang_b = 90 * k + 45 - lug_half - 3.0
        bumps.append(Manifold.sphere(0.9, 24).scale([1, 1, 0.7])
                     .translate([(p.bay_r + p.bay_notch_r) / 2, 0, zf_in + p.bay_ring_t]).rotate([0, 0, ang_b]))
    bottom = bottom + ring + Manifold.batch_boolean(stops + bumps, OpType.Add)
    bottom -= hole
    bottom -= Manifold.cylinder(1.0, p.bay_r + 0.8, p.bay_r - 0.2, 96).translate([0, 0, p.z_bot - 0.01])

    # пояс — отдельная деталь: надевается на нижнюю перед крышкой
    ring = bottom ^ box(-big, big, -big, big, zr, zp + 1.0)
    bottom = bottom ^ box(-big, big, -big, big, p.z_bot - 1.0, zr)

    build.stats = stats
    return {"bottom": bottom, "ring": ring, "lid": lid}


# ============================================================ макет платы


def board_dummy(p: Params | None = None):
    """Условная плата для просмотра и проверки зазоров (не для печати)."""
    p = p or Params()
    bx, by = p.bx2, p.by2
    parts = {}
    pcb = box(-bx, bx, -by, by, 0, p.pcb_t)
    for sx in (-1, 1):
        for sy in (-1, 1):
            pcb -= Manifold.cylinder(4, 1.35, 1.35, 24).translate([sx * (bx - p.hole_inset), sy * (by - p.hole_inset), -1])
    parts["pcb"] = pcb
    # радиатор: рёбра вдоль Y, выдув вперёд; со стороны кнопки полоса ~14 мм свободна
    hs_x0, hs_x1 = -bx + 14.0, bx - 4.0
    hs_y0, hs_y1 = -by + 22.0, by - 2.0
    base = box(hs_x0, hs_x1, hs_y0, hs_y1, p.pcb_t, p.pcb_t + 6.0)
    fins = [box(x, x + 0.8, hs_y0, hs_y1, p.pcb_t + 6.0, p.pcb_t + p.top_h - 4.0)
            for x in np.arange(hs_x0 + 1.0, hs_x1 - 1.0, 2.6)]
    shroud = box(hs_x0, hs_x1, hs_y0, hs_y1 - 30.0, p.pcb_t + p.top_h - 4.0, p.pcb_t + p.top_h)
    parts["heatsink"] = Manifold.batch_boolean([base] + fins, OpType.Add)
    parts["fan"] = shroud - Manifold.cylinder(10, 21, 21, 64).translate(
        [(hs_x0 + hs_x1) / 2, (hs_y0 + hs_y1 - 30.0) / 2, p.pcb_t + p.top_h - 6.0]) + \
        Manifold.cylinder(3.0, 19, 19, 64).translate([(hs_x0 + hs_x1) / 2, (hs_y0 + hs_y1 - 30.0) / 2,
                                                     p.pcb_t + p.top_h - 4.0])
    # разъёмы сзади (условная раскладка по фото)
    conns = []
    x = -bx + 8.0
    for w, h in ((16.0, 17.5), (16.0, 17.5), (15.0, 6.5), (15.0, 6.5), (14.5, 16.0), (9.0, 3.5), (9.0, 3.5)):
        conns.append(box(x, x + w, -by - p.conn_overhang, -by + 14.0, p.pcb_t, p.pcb_t + h))
        x += w + 2.2
    parts["connectors"] = Manifold.batch_boolean(conns, OpType.Add)
    parts["nvme"] = box(-11, 11, -40, 40, -p.bot_h, 0) + box(-30, 30, -45, 45, -3.0, 0)
    parts["button"] = box(-bx - p.btn_protrude, -bx + 4.0, p.btn_y - 1.8, p.btn_y + 1.8,
                          p.pcb_t, p.pcb_t + 2 * p.btn_z)
    return parts


def keepout(p: Params | None = None):
    """Зоны, куда корпус заходить не должен (консервативно, по замерам)."""
    p = p or Params()
    bx, by = p.bx2, p.by2
    corner = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = box(bx - p.pad - 0.2, bx + 5, by - p.pad - 0.2, by + 5, -50, 80)
            if sx < 0:
                c = c.mirror([1, 0, 0])
            if sy < 0:
                c = c.mirror([0, 1, 0])
            corner.append(c)
    corner = Manifold.batch_boolean(corner, OpType.Add)
    hw = bx - p.conn_corner
    zones = {
        "плата": box(-bx, bx, -by, by, 0, p.pcb_t),
        "низ: выводы": box(-bx, bx, -by, by, -3.5, 0) - corner,
        "низ: NVMe": box(-bx + 6, bx - 6, -by + 6, by - 6, -p.bot_h, -3.5),
        "верх: разъёмы и детали": box(-bx, bx, -by, by, p.pcb_t, p.pcb_t + p.conn_h) - corner,
        "разъёмы за краем": box(-hw, hw, -by - p.conn_overhang, -by, p.pcb_t, p.pcb_t + p.conn_h),
        "верх: радиатор": box(-bx + 3, bx - 3, -by + 3, by - 3, p.pcb_t + p.conn_h, p.pcb_t + p.top_h) - corner,
        "кнопка PWR": box(-bx - p.btn_protrude, -bx, p.btn_y - 1.8, p.btn_y + 1.8, p.pcb_t, p.pcb_t + 2 * p.btn_z),
    }
    return zones
