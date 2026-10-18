"""
Настольный органайзер «вчера · сегодня · завтра».

Два рифлёных стакана под ручки по краям, между ними блок: сверху открытый карман
под скрепки и стикеры, спереди два выдвижных ящика. На стаканах и на лицевой
стороне блока утоплены таблички с надписями: слева «вчера», в центре «сегодня»,
справа «завтра». Шрифт Montserrat SemiBold (SIL OFL), строчные.

Система координат: Z = 0 — стол, X — вдоль органайзера (левый стакан в −X),
лицевая сторона смотрит в −Y. Корпус и ящики построены в положении для печати.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from manifold3d import CrossSection, FillRule, JoinType, Manifold

SEG = 240        # сегменты окружностей отверстий


@dataclass
class Params:
    # корпус
    height: float = 100.0
    cup_r: float = 36.0            # радиус стакана по вершинам рёбер (Ø72)
    cup_pitch: float = 168.0       # между осями стаканов: габарит 240 мм под стол P1S 256 мм
    rib_n: int = 60                # рёбер по кругу: шаг 3,8 мм
    rib_h: float = 1.2             # высота ребра над впадиной
    rib_shape: float = 0.5         # профиль ребра |cos|^k: 1 — синус, меньше — круглее вершина, острее впадина
    cup_wall: float = 2.8          # стенка стакана от впадины ребра до отверстия
    cup_floor: float = 2.4
    rim_chamfer: float = 0.8       # фаска по кромке отверстий сверху
    block_front: float = -22.0     # лицевая плоскость блока (Y): стаканы выступают вперёд на 14 мм
    # карман сверху
    pocket_w: float = 88.0
    pocket_wall: float = 3.2       # стенки кармана спереди и сзади
    pocket_r: float = 12.0
    pocket_floor_z: float = 64.0
    # ниша ящиков
    bay_w: float = 84.0
    bay_floor: float = 2.0
    slot_h: float = 27.5
    shelf: float = 2.4             # полка между ящиками
    bay_back_wall: float = 2.4
    # ящик
    gap_side: float = 0.35
    gap_top: float = 1.0           # над ящиком: запас на провис моста 84 мм
    gap_back: float = 0.5
    drawer_wall: float = 1.6
    drawer_front: float = 2.4
    drawer_floor: float = 1.2
    drawer_inner_r: float = 3.0
    notch_top: float = 30.0        # выемка под палец: ширина сверху
    notch_bottom: float = 18.0     # и у дна
    notch_depth: float = 8.0
    notch_r: float = 3.0
    # надписи
    font: str = "fonts/Montserrat-SemiBold.ttf"
    word_left: str = "вчера"
    word_center: str = "сегодня"
    word_right: str = "завтра"
    x_height: float = 6.0          # высота строчных: «завтра» 42 мм — табличка не доходит до стыка с блоком
    track: float = 0.35            # разрядка, мм
    text_depth: float = 0.8
    text_z: float = 82.0           # середина строчных по высоте
    panel_pad: float = 4.5         # поле таблички слева и справа от слова
    panel_h: float = 17.0
    panel_shift: float = -1.0      # табличка ниже середины строчных: у всех слов есть хвосты р и д
    panel_r: float = 3.0
    panel_below_rib: float = 0.2   # табличка стакана глубже впадин рёбер
    panel_depth_center: float = 0.6

    # производные размеры
    @property
    def rib_valley(self) -> float:
        return self.cup_r - self.rib_h

    @property
    def bore_r(self) -> float:
        return self.rib_valley - self.cup_wall

    @property
    def block_back(self) -> float:
        return self.rib_valley

    @property
    def panel_r_cup(self) -> float:
        return self.rib_valley - self.panel_below_rib

    def slot_z(self, i: int) -> tuple[float, float]:
        """Ниша i-го ящика по высоте: 0 — нижний, 1 — верхний."""
        z0 = self.bay_floor + i * (self.slot_h + self.shelf)
        return z0, z0 + self.slot_h

    @property
    def bay_y(self) -> tuple[float, float]:
        return self.block_front, self.block_back - self.bay_back_wall

    @property
    def drawer_size(self) -> tuple[float, float, float]:
        y0, y1 = self.bay_y
        return (self.bay_w - 2 * self.gap_side, y1 - y0 - self.gap_back, self.slot_h - self.gap_top)


# ---------- 2D ----------

def rib_profile(p: Params, cx: float) -> np.ndarray:
    """Контур рифлёного стакана: рёбра со скруглённой вершиной и узкой впадиной."""
    k = 16
    n = p.rib_n * k
    th = np.arange(n) * 2 * math.pi / n
    s = (th * p.rib_n / (2 * math.pi) + 0.5) % 1.0 - 0.5          # фаза в пределах шага, вершина при 0
    r = p.rib_valley + p.rib_h * np.abs(np.cos(math.pi * s)) ** p.rib_shape
    return np.stack([cx + r * np.cos(th), r * np.sin(th)], 1)


def rounded_rect(x0, y0, x1, y1, r, seg=48) -> CrossSection:
    return (CrossSection.square([x1 - x0 - 2 * r, y1 - y0 - 2 * r]).translate([x0 + r, y0 + r])
            .offset(r, JoinType.Round, 2.0, seg))


def outline(p: Params) -> CrossSection:
    h = p.cup_pitch / 2
    cups = CrossSection([rib_profile(p, -h)]) + CrossSection([rib_profile(p, h)])
    block = CrossSection.square([p.cup_pitch, p.block_back - p.block_front]).translate([-h, p.block_front])
    return cups + block


# ---------- надписи ----------

class _FlatPen:
    """Перо fontTools, которое раскладывает кривые глифа в ломаные."""

    def __new__(cls, glyphset, steps=12):
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


def word_contours(p: Params, word: str):
    """Контуры слова в мм: u — вдоль строки (центр в 0), v — вверх от середины строчных."""
    from fontTools.ttLib import TTFont
    path = p.font if os.path.isabs(p.font) else os.path.join(os.path.dirname(__file__), p.font)
    font = TTFont(path)
    gs, cmap = font.getGlyphSet(), font.getBestCmap()
    k = p.x_height / font["OS/2"].sxHeight
    out, x = [], 0.0
    for ch in word:
        g = gs[cmap[ord(ch)]]
        pen = _FlatPen(gs)
        g.draw(pen)
        out += [c * k + [x, -p.x_height / 2] for c in pen.contours]
        x += g.width * k + p.track
    width = x - p.track
    return [c - [width / 2, 0.0] for c in out], width


def label_cutter(p: Params, word: str, panel_depth: float):
    """Табличка со словом в плоских координатах (u, v, w): u вдоль строки, v вверх, w наружу от дна таблички.

    Верхняя кромка таблички скошена под 45°: при печати снизу вверх она не нависает.
    """
    contours, width = word_contours(p, word)
    pw = width + 2 * p.panel_pad
    v0, v1 = p.panel_shift - p.panel_h / 2, p.panel_shift + p.panel_h / 2
    out = panel_depth + 1.0
    eps = 0.01
    base = rounded_rect(-pw / 2, v0, pw / 2, v1, p.panel_r).extrude(eps)
    top = rounded_rect(-pw / 2, v0, pw / 2, v1 + out, p.panel_r).extrude(eps).translate([0, 0, out])
    panel = Manifold.batch_hull([base, top])
    text = CrossSection(contours, FillRule.NonZero).extrude(p.text_depth + 0.5).translate([0, 0, -p.text_depth])
    return panel + text, pw


def wrap_to_cup(m: Manifold, p: Params, cx: float) -> Manifold:
    """(u, v, w) → стакан с осью в (cx, 0): u — дуга по дну таблички, середина смотрит в −Y."""
    r0 = p.panel_r_cup

    def f(pts):
        th = -math.pi / 2 + pts[:, 0] / r0
        r = r0 + pts[:, 2]
        return np.stack([cx + r * np.cos(th), r * np.sin(th), p.text_z + pts[:, 1]], 1)

    return m.refine_to_length(0.8).warp_batch(f)


def to_front(m: Manifold, p: Params) -> Manifold:
    """(u, v, w) → лицевая плоскость блока: u → X, v → Z, w → −Y."""
    return m.warp_batch(lambda pts: np.stack([pts[:, 0], p.block_front - pts[:, 2], p.text_z + pts[:, 1]], 1))


# ---------- детали ----------

def chamfered_prism(cs: CrossSection, z0: float, z1: float, chamfer: float) -> Manifold:
    """Призма выпуклого сечения от z0 до z1 + 1 с фаской по верхней кромке z1 (сечение выпуклое — годится hull)."""
    body = cs.extrude(z1 - chamfer - z0).translate([0, 0, z0])
    lip = Manifold.batch_hull([
        cs.extrude(0.01).translate([0, 0, z1 - chamfer]),
        cs.offset(chamfer + 1.0, JoinType.Round, 2.0, 48).extrude(0.01).translate([0, 0, z1 + 1.0]),
    ])
    return body + lip


def body(p: Params) -> Manifold:
    h = p.cup_pitch / 2
    m = outline(p).extrude(p.height)
    # стаканы
    for cx in (-h, h):
        bore = CrossSection.circle(p.bore_r, SEG).translate([cx, 0])
        m -= chamfered_prism(bore, p.cup_floor, p.height, p.rim_chamfer)
    # карман
    py0, py1 = p.block_front + p.pocket_wall, p.block_back - p.pocket_wall
    pocket = rounded_rect(-p.pocket_w / 2, py0, p.pocket_w / 2, py1, p.pocket_r)
    m -= chamfered_prism(pocket, p.pocket_floor_z, p.height, p.rim_chamfer)
    # ниши ящиков
    y0, y1 = p.bay_y
    for i in (0, 1):
        z0, z1 = p.slot_z(i)
        m -= Manifold.cube([p.bay_w, y1 - y0 + 1.0, z1 - z0]).translate([-p.bay_w / 2, y0 - 1.0, z0])
    # таблички с надписями
    left, _ = label_cutter(p, p.word_left, p.rib_h + p.panel_below_rib)
    right, _ = label_cutter(p, p.word_right, p.rib_h + p.panel_below_rib)
    center, _ = label_cutter(p, p.word_center, p.panel_depth_center)
    m -= wrap_to_cup(left, p, -h)
    m -= wrap_to_cup(right, p, h)
    m -= to_front(center.translate([0, 0, -p.panel_depth_center]), p)
    return m


def drawer(p: Params) -> Manifold:
    """Ящик в положении для печати: лицо в −Y, центр по X в 0, дно на z = 0, лицо на y = 0."""
    w, d, hgt = p.drawer_size
    m = Manifold.cube([w, d, hgt]).translate([-w / 2, 0, 0])
    iw = w / 2 - p.drawer_wall
    cav = rounded_rect(-iw, p.drawer_front, iw, d - p.drawer_wall, p.drawer_inner_r)
    m -= cav.extrude(hgt).translate([0, 0, p.drawer_floor])
    # выемка под палец в лицевой стенке: трапеция со скруглённым дном
    a, b, dep = p.notch_top / 2, p.notch_bottom / 2, p.notch_depth
    notch = CrossSection([np.array([[-b, hgt - dep], [b, hgt - dep], [a, hgt + 5], [-a, hgt + 5]])])
    notch = notch.offset(-p.notch_r, JoinType.Round, 2.0, 48).offset(p.notch_r, JoinType.Round, 2.0, 48)
    # сечение в плоскости (x, z) → вдоль Y через лицевую стенку
    cut = notch.extrude(p.drawer_front + 2.0).rotate([90, 0, 0]).translate([0, p.drawer_front + 1.0, 0])
    return m - cut


def drawer_in_body(p: Params, i: int, pull: float = 0.0) -> Manifold:
    """Ящик i на своём месте в корпусе; pull — насколько выдвинут."""
    z0, _ = p.slot_z(i)
    return drawer(p).translate([0, p.block_front - pull, z0])


def build(p: Params | None = None) -> dict[str, Manifold]:
    p = p or Params()
    return {"body": body(p), "drawer": drawer(p)}
