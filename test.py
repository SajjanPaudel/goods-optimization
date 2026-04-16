"""
Goods–truck allocation: standardized pandas DataFrames only (no parsing, no classes).
"""

import copy
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# --- Standard column schemas (see goods_sample.csv, trucks_sample.csv) ---
# Goods:  id, name, weight_kg, l, b, h, max_stack, adr, adr_class
# Trucks: id, name, l, b, h, max_weight_kg, max_volume_m3, adr_suitable, adr_classes_allowed
#
# ADR: goods with adr=True may only be loaded on trucks with adr_suitable=True and
# adr_class in adr_classes_allowed (comma-separated, or "*" for any class).

_DATA_DIR = Path(__file__).resolve().parent

# Minimum free-rectangle edge (m); keep consistent so merge does not drop strips
# that stacking still appends, which used to yield [] tops and full-surface resets.
_MIN_FREE_DIM = 0.01


def _normalize_adr_class(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return s


def _read_goods_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": int})
    df["adr_class"] = df["adr_class"].map(_normalize_adr_class)
    return df


def _read_trucks_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str, "name": str})
    df["adr_classes_allowed"] = df["adr_classes_allowed"].fillna("").astype(str)
    return df


GOODS_SAMPLE: pd.DataFrame = _read_goods_csv(_DATA_DIR / "goods_sample.csv")
TRUCKS_SAMPLE: pd.DataFrame = _read_trucks_csv(_DATA_DIR / "trucks_sample.csv")


def _item_volume(row: pd.Series) -> float:
    return float(row["l"]) * float(row["b"]) * float(row["h"])


def _item_footprint(row: pd.Series) -> float:
    return float(row["l"]) * float(row["b"])


def _as_bool(val: Any, default: bool = False) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("", "nan", "none"):
        return default
    return s in ("1", "true", "yes", "y", "t")


def adr_good_allowed_on_truck(item: Dict[str, Any], truck: pd.Series) -> bool:
    """
    Non-ADR goods may use any truck. ADR goods require adr_suitable truck and,
    if adr_classes_allowed is not '*', a matching adr_class (comma-separated list).
    Missing adr columns default to non-ADR / not suitable.
    """
    if not _as_bool(item.get("adr"), False):
        return True
    if not _as_bool(truck.get("adr_suitable"), False):
        return False
    gclass = str(item.get("adr_class", "")).strip()
    if not gclass:
        # Flagged ADR but no class: allow any ADR-suitable truck
        return True
    allowed_raw = truck.get("adr_classes_allowed", "*")
    if allowed_raw is None or (isinstance(allowed_raw, float) and pd.isna(allowed_raw)):
        allowed_raw = "*"
    allowed_str = str(allowed_raw).strip()
    if allowed_str == "" or allowed_str == "*":
        return True
    allowed_set = {x.strip() for x in allowed_str.split(",") if x.strip()}
    return gclass in allowed_set


def build_items_for_loading(goods_df: pd.DataFrame, truck_width: float) -> List[Dict[str, Any]]:
    """One dict per row with idx = position in current frame (0..n-1)."""
    df = goods_df.reset_index(drop=True)
    items: List[Dict[str, Any]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        items.append(
            {
                "idx": idx,
                "id": int(row["id"]),
                "name": str(row["name"]),
                "weight": float(row["weight_kg"]),
                "l": float(row["l"]),
                "b": float(row["b"]),
                "h": float(row["h"]),
                "max_stack": int(row["max_stack"]),
                "volume": _item_volume(row),
                "footprint": _item_footprint(row),
                "adr": _as_bool(row.get("adr", False), False),
                "adr_class": str(row.get("adr_class", "") or "").strip(),
            }
        )
    # Geometry-first ordering to maximize space utilization; ignore weight priority.
    # Defer floor-hog items that consume most of truck width so narrow items can
    # seed back-right lanes first, preserving maneuvering space for later fits.
    for item in items:
        min_cross_width = min(item["l"], item["b"])
        item["defer_wide_floor_hog"] = (
            min_cross_width >= (0.75 * truck_width) and item["footprint"] >= 3.5
        )
    items.sort(
        key=lambda x: (
            -x["volume"]
            -x["footprint"], # Larger base first
            -x["h"],         # Tallest first to define stack heights
        )
    )
    return items


def can_load_by_capacity(
    item: Dict[str, Any],
    total_weight: float,
    total_volume: float,
    max_weight: float,
    max_volume: float,
) -> bool:
    if total_weight + item["weight"] > max_weight:
        return False
    if total_volume + item["volume"] > max_volume:
        return False
    return True


def merge_free_rectangles(
    rects: List[Tuple[float, float, float, float]],
    eps: float = 1e-9,
) -> List[Tuple[float, float, float, float]]:
    """
    Merge adjacent axis-aligned free rectangles.
    Rect format: (x, y, l, b). Merges when they share a full edge:
    - same y and b, touching in x-direction
    - same x and l, touching in y-direction
    """
    out = [r for r in rects if r[2] > _MIN_FREE_DIM and r[3] > _MIN_FREE_DIM]
    changed = True
    while changed:
        changed = False
        n = len(out)
        for i in range(n):
            if changed:
                break
            x1, y1, l1, b1 = out[i]
            for j in range(i + 1, n):
                x2, y2, l2, b2 = out[j]

                # Horizontal merge (along x): same row/height in y,b.
                if abs(y1 - y2) <= eps and abs(b1 - b2) <= eps:
                    if abs((x1 + l1) - x2) <= eps:
                        out[i] = (x1, y1, l1 + l2, b1)
                        out.pop(j)
                        changed = True
                        break
                    if abs((x2 + l2) - x1) <= eps:
                        out[i] = (x2, y1, l1 + l2, b1)
                        out.pop(j)
                        changed = True
                        break

                # Vertical merge (along y): same column/length in x,l.
                if abs(x1 - x2) <= eps and abs(l1 - l2) <= eps:
                    if abs((y1 + b1) - y2) <= eps:
                        out[i] = (x1, y1, l1, b1 + b2)
                        out.pop(j)
                        changed = True
                        break
                    if abs((y2 + b2) - y1) <= eps:
                        out[i] = (x1, y2, l1, b1 + b2)
                        out.pop(j)
                        changed = True
                        break
    return out


def _rect_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    eps: float = 1e-9,
) -> Optional[Tuple[float, float, float, float]]:
    ax, ay, al, ab = a
    bx, by, bl, bb = b
    ox1 = max(ax, bx)
    oy1 = max(ay, by)
    ox2 = min(ax + al, bx + bl)
    oy2 = min(ay + ab, by + bb)
    if (ox2 - ox1) <= eps or (oy2 - oy1) <= eps:
        return None
    return (ox1, oy1, ox2 - ox1, oy2 - oy1)


def _subtract_rect(
    rect: Tuple[float, float, float, float],
    cut: Tuple[float, float, float, float],
    eps: float = 1e-9,
) -> List[Tuple[float, float, float, float]]:
    overlap = _rect_overlap(rect, cut, eps=eps)
    if overlap is None:
        return [rect]
    rx, ry, rl, rb = rect
    ox, oy, ol, ob = overlap
    rx2, ry2 = rx + rl, ry + rb
    ox2, oy2 = ox + ol, oy + ob
    out: List[Tuple[float, float, float, float]] = []

    # Left and right strips.
    if (ox - rx) > eps:
        out.append((rx, ry, ox - rx, rb))
    if (rx2 - ox2) > eps:
        out.append((ox2, ry, rx2 - ox2, rb))
    # Bottom and top strips in the middle x-range.
    if (oy - ry) > eps:
        out.append((ox, ry, ol, oy - ry))
    if (ry2 - oy2) > eps:
        out.append((ox, oy2, ol, ry2 - oy2))
    return out


def _footprint_fully_supported(
    x: float,
    y: float,
    l: float,
    b: float,
    support_rects: List[Tuple[float, float, float, float]],
    eps: float = 1e-9,
) -> bool:
    target = (x, y, l, b)
    clipped: List[Tuple[float, float, float, float]] = []
    for rect in support_rects:
        overlap = _rect_overlap(rect, target, eps=eps)
        if overlap is not None:
            clipped.append(overlap)
    if not clipped:
        return False

    xs = sorted({x, x + l, *[r[0] for r in clipped], *[r[0] + r[2] for r in clipped]})
    ys = sorted({y, y + b, *[r[1] for r in clipped], *[r[1] + r[3] for r in clipped]})
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if (xs[i + 1] - xs[i]) <= eps or (ys[j + 1] - ys[j]) <= eps:
                continue
            cx = (xs[i] + xs[i + 1]) * 0.5
            cy = (ys[j] + ys[j + 1]) * 0.5
            covered = any(
                (rx - eps) <= cx <= (rx + rl + eps) and (ry - eps) <= cy <= (ry + rb + eps)
                for rx, ry, rl, rb in clipped
            )
            if not covered:
                return False
    return True


def _reserve_base_top_area(
    base: Dict[str, Any],
    world_rect: Tuple[float, float, float, float],
) -> None:
    top_free_rects: List[Tuple[float, float, float, float]] = base.setdefault(
        "top_free_rects", [(0.0, 0.0, base["l"], base["b"])]
    )
    updated: List[Tuple[float, float, float, float]] = []
    for fr in top_free_rects:
        world_fr = (base["x"] + fr[0], base["y"] + fr[1], fr[2], fr[3])
        overlap = _rect_overlap(world_fr, world_rect)
        if overlap is None:
            updated.append(fr)
            continue
        local_cut = (
            overlap[0] - base["x"],
            overlap[1] - base["y"],
            overlap[2],
            overlap[3],
        )
        updated.extend(_subtract_rect(fr, local_cut))
    base["top_free_rects"] = merge_free_rectangles(updated)


def _preferred_orientations(item: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Orientation order that prioritizes using truck width (y) more than length (x):
    prefer smaller footprint side along x and larger side along y.
    """
    a, b = float(item["l"]), float(item["b"])
    primary = (min(a, b), max(a, b))
    secondary = (max(a, b), min(a, b))
    if abs(primary[0] - secondary[0]) <= 1e-9 and abs(primary[1] - secondary[1]) <= 1e-9:
        return [primary]
    return [primary, secondary]


def _guillotine_side_front(
    rx: float, ry: float, rl: float, rb: float, il: float, ib: float
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """
    After placing footprint (il, ib) at (rx, ry) inside (rl, rb), return side (+y strip)
    and front (+x strip) free rectangles. Matches a row-first Tetris habit: grow along
    width at the same depth before growing toward the cab (+x).
    """
    if (rl - il) > (rb - ib):
        side = (rx, ry + ib, il, rb - ib)
        front = (rx + il, ry, rl - il, rb)
    else:
        side = (rx, ry + ib, rl, rb - ib)
        front = (rx + il, ry, rl - il, ib)
    return side, front


def _layer_score(p: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Lower is better for level-wise packing: minimize forward extent (x+l), stay at the
    back (small x), then shallow in y, then lower shelves first (small z).
    """
    return (
        round(float(p["x"]) + float(p["l"]), 4),
        round(float(p["x"]), 4),
        round(float(p["y"]), 4),
        round(float(p["z"]), 4),
    )


def _stack_top_waste_frac(il: float, ib: float, rl: float, rb: float, eps: float = 1e-12) -> float:
    """Fraction of free top rectangle still empty after placing (il, ib); lower = better shelf use."""
    denom = rl * rb + eps
    return max(0.0, (rl * rb - il * ib) / denom)


def try_place_on_floor(
    item: Dict[str, Any],
    free_rects: List[Tuple[float, float, float, float]],
    truck_height: float,
    max_front_x: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    best: Optional[Tuple[Tuple[float, float, float, float, float], int, float, float, float, float]] = None

    for ridx, (rx, ry, rl, rb) in enumerate(free_rects):
        for il, ib in _preferred_orientations(item):
            if il > rl + 1e-9 or ib > rb + 1e-9 or item["h"] > truck_height:
                continue
            
            # Constraint: Stay within the current "load zone" if provided
            if max_front_x is not None and (rx + il) > (max_front_x + 1e-9):
                continue

            # Tetris / back-first: fill from the back wall (small x), sweep y, keep cab
            # end (+x) free as long as possible; prefer tight forward edge (rx+il).
            score = (
                round(rx, 4),
                round(ry, 4),
                round(rx + il, 4),
                (rl * rb) - (il * ib),
                -ib,
            )
            cand = (score, ridx, rx, ry, il, ib)
            if best is None or cand[0] < best[0]:
                best = cand

    if best is not None:
        _, ridx, rx, ry, il, ib = best
        rl, rb = free_rects[ridx][2], free_rects[ridx][3]
        del free_rects[ridx]

        side, front = _guillotine_side_front(rx, ry, rl, rb, il, ib)

        for rect in (side, front):
            if rect[2] > _MIN_FREE_DIM and rect[3] > _MIN_FREE_DIM:
                free_rects.append(rect)
                
        free_rects[:] = merge_free_rectangles(free_rects)
        return {
            "idx": item["idx"],
            "ID": item["id"],
            "name": item["name"],
            "weight": item["weight"],
            "l": il,
            "b": ib,
            "h": item["h"],
            "x": rx,
            "y": ry,
            "z": 0.0,
            "level": 0,
            "top_free_rects": [(0.0, 0.0, il, ib)],
        }
    return None

def try_stack(
    item: Dict[str, Any],
    placements: List[Dict[str, Any]],
    truck_height: float,
    base_allowed: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Optional[Dict[str, Any]]:
    eps = 1e-9
    sorted_bases = sorted(
        placements,
        key=lambda p: (round(float(p["x"]), 4), round(float(p["y"]), 4), -float(p["z"])),
    )

    best_global: Optional[Tuple[Tuple, Dict[str, Any], Tuple]] = None
    # Tuple: ((waste_frac, _layer_score...), base, (ridx, rx, ry, il, ib, new_level, new_z))

    for base in sorted_bases:
        if base_allowed is not None and not base_allowed(base):
            continue
        new_level = int(base.get("level", 0)) + 1
        if new_level >= int(base.get("stack_limit", 2)):
            continue

        top_src = base.get("top_free_rects")
        if top_src is None:
            top_free_rects = [(0.0, 0.0, float(base["l"]), float(base["b"]))]
        else:
            top_free_rects = [tuple(r) for r in top_src]
        new_z = float(base["z"]) + float(base["h"])
        if new_z + float(item["h"]) > truck_height + eps:
            continue

        best_top = None
        for ridx, (rx, ry, rl, rb) in enumerate(top_free_rects):
            for il, ib in _preferred_orientations(item):
                if il > rl + eps or ib > rb + eps:
                    continue

                wx = float(base["x"]) + rx
                wy = float(base["y"]) + ry
                wf = _stack_top_waste_frac(il, ib, rl, rb, eps=eps)
                score = (
                    round(wf, 8),
                    round(wx + il, 4),
                    round(wx, 4),
                    round(wy, 4),
                    -ib,
                )
                cand = (score, ridx, rx, ry, il, ib)
                if best_top is None or cand[0] < best_top[0]:
                    best_top = cand

        if best_top is None:
            continue

        _, ridx, rx, ry, il, ib = best_top
        wx = float(base["x"]) + rx
        wy = float(base["y"]) + ry
        rl, rb = top_free_rects[ridx][2], top_free_rects[ridx][3]
        cand_box = {
            "x": wx,
            "y": wy,
            "z": new_z,
            "l": il,
            "b": ib,
            "h": float(item["h"]),
        }
        wf_g = _stack_top_waste_frac(il, ib, rl, rb, eps=eps)
        glob_tie = (round(wf_g, 8), _layer_score(cand_box))
        if best_global is None or glob_tie < best_global[0]:
            best_global = (glob_tie, base, (ridx, rx, ry, il, ib, new_level, new_z))

    if best_global is None:
        return None

    _, base, pack = best_global
    ridx, rx, ry, il, ib, new_level, new_z = pack

    top_src = base.get("top_free_rects")
    if top_src is None:
        top_free_rects = [(0.0, 0.0, float(base["l"]), float(base["b"]))]
    else:
        top_free_rects = [tuple(r) for r in top_src]

    rl, rb = top_free_rects[ridx][2], top_free_rects[ridx][3]
    del top_free_rects[ridx]

    side, front = _guillotine_side_front(rx, ry, rl, rb, il, ib)
    for rect in (side, front):
        if rect[2] > _MIN_FREE_DIM and rect[3] > _MIN_FREE_DIM:
            top_free_rects.append(rect)

    base["top_free_rects"] = merge_free_rectangles(top_free_rects)
    return {
        "idx": item["idx"],
        "ID": item["id"],
        "name": item["name"],
        "weight": item["weight"],
        "l": il,
        "b": ib,
        "h": item["h"],
        "x": base["x"] + rx,
        "y": base["y"] + ry,
        "z": new_z,
        "level": new_level,
        "top_free_rects": [(0.0, 0.0, il, ib)],
    }


def _merge_touching_world_rects(
    rects: List[Tuple[float, float, float, float]], eps: float = 1e-6
) -> List[Tuple[float, float, float, float]]:
    """Merge axis-aligned world (x, y, l, b) rectangles that share a full edge."""
    out = [tuple(r) for r in rects]
    changed = True
    while changed:
        changed = False
        n = len(out)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, l1, b1 = out[i]
                x2, y2, l2, b2 = out[j]
                if abs(y1 - y2) <= eps and abs(b1 - b2) <= eps and abs((x1 + l1) - x2) <= eps:
                    out[i] = (x1, y1, l1 + l2, b1)
                    out.pop(j)
                    changed = True
                    break
                if abs(y1 - y2) <= eps and abs(b1 - b2) <= eps and abs((x2 + l2) - x1) <= eps:
                    out[i] = (x2, y1, l1 + l2, b1)
                    out.pop(j)
                    changed = True
                    break
                if abs(x1 - x2) <= eps and abs(l1 - l2) <= eps and abs((y1 + b1) - y2) <= eps:
                    out[i] = (x1, y1, l1, b1 + b2)
                    out.pop(j)
                    changed = True
                    break
                if abs(x1 - x2) <= eps and abs(l1 - l2) <= eps and abs((y2 + b2) - y1) <= eps:
                    out[i] = (x1, y2, l1, b1 + b2)
                    out.pop(j)
                    changed = True
                    break
            if changed:
                break
    return out


def _axis_samples(lo: float, hi: float, maxn: int = 11) -> List[float]:
    """Sample positions along [lo, hi] inclusive (e.g. sliding a box inside a free rectangle)."""
    if hi <= lo + 1e-9:
        return [lo]
    n = min(maxn, max(2, int((hi - lo) / 0.15) + 1))
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def try_stack_on_merged_rear_floor_shelf(
    item: Dict[str, Any],
    placements: List[Dict[str, Any]],
    truck_height: float,
    t_length: float,
    t_width: float,
    eps: float = 1e-4,
) -> Optional[Dict[str, Any]]:
    """
    Stack on the union of coplanar top faces of the rear floor cluster (min x on floor),
    after subtracting whatever already sits on that shelf. Lets wide items use combined
    space over e.g. 1032 + 1099 at the same z-height, which single-base try_stack cannot.
    """
    floor = [p for p in placements if int(p.get("level", 0)) == 0]
    if not floor:
        return None
    min_xf = min(float(p["x"]) for p in floor)
    back_floor = [p for p in floor if float(p["x"]) <= min_xf + eps]
    if not back_floor:
        return None

    by_ztop: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for p in back_floor:
        zt = round(float(p["z"]) + float(p["h"]), 6)
        by_ztop[zt].append(p)

    best: Optional[Tuple[Tuple[float, float, float, float], Dict[str, Any]]] = None

    for _zk, parts in by_ztop.items():
        Zt = float(parts[0]["z"]) + float(parts[0]["h"])
        if Zt + float(item["h"]) > truck_height + eps:
            continue

        world = [(float(p["x"]), float(p["y"]), float(p["l"]), float(p["b"])) for p in parts]
        merged = _merge_touching_world_rects(world, eps=eps)
        free_list: List[Tuple[float, float, float, float]] = list(merged)

        for q in placements:
            if abs(float(q["z"]) - Zt) > eps:
                continue
            qx, qy, ql, qb = float(q["x"]), float(q["y"]), float(q["l"]), float(q["b"])
            nxt: List[Tuple[float, float, float, float]] = []
            for fr in free_list:
                nxt.extend(_subtract_rect(fr, (qx, qy, ql, qb), eps=eps))
            free_list = merge_free_rectangles(
                [r for r in nxt if r[2] > _MIN_FREE_DIM and r[3] > _MIN_FREE_DIM]
            )

        for R in free_list:
            rx, ry, rl, rb = R
            for il, ib in _preferred_orientations(item):
                if il > rl + eps or ib > rb + eps:
                    continue
                x_hi = rx + max(0.0, rl - il)
                y_hi = ry + max(0.0, rb - ib)
                for wx in _axis_samples(rx, x_hi, maxn=11):
                    for wy in _axis_samples(ry, y_hi, maxn=11):
                        if wx < -eps or wy < -eps:
                            continue
                        if wx + il > t_length + eps or wy + ib > t_width + eps:
                            continue
                        supports_full: List[Tuple[float, float, float, float]] = []
                        for f in parts:
                            ztf = float(f["z"]) + float(f["h"])
                            if abs(ztf - Zt) > eps:
                                continue
                            fx, fy, fl, fb = float(f["x"]), float(f["y"]), float(f["l"]), float(f["b"])
                            if _rect_overlap((wx, wy, il, ib), (fx, fy, fl, fb), eps=eps) is None:
                                continue
                            supports_full.append((fx, fy, fl, fb))
                        if not supports_full or not _footprint_fully_supported(
                            wx, wy, il, ib, supports_full, eps=eps
                        ):
                            continue
                        new_level = 1 + max(int(p.get("level", 0)) for p in parts)
                        if new_level >= int(item.get("max_stack", 10)):
                            continue
                        cand = {
                            "idx": item["idx"],
                            "ID": item["id"],
                            "name": item["name"],
                            "weight": item["weight"],
                            "l": il,
                            "b": ib,
                            "h": item["h"],
                            "x": wx,
                            "y": wy,
                            "z": Zt,
                            "level": new_level,
                            "top_free_rects": [(0.0, 0.0, il, ib)],
                        }
                        sc = _layer_score(cand)
                        wf_m = _stack_top_waste_frac(il, ib, rl, rb, eps=eps)
                        tie = (
                            round(wf_m, 8),
                            sc,
                            round(wx + il, 4),
                            round(wx, 4),
                            round(wy, 4),
                            -ib,
                        )
                        if best is None or tie < best[0]:
                            best = (tie, cand)

    return None if best is None else best[1]


def _floor_plan_max_x(placements: List[Dict[str, Any]], eps: float = 1e-9) -> float:
    """Max (x+l) over level-0 floor pieces (plan length consumed on the deck)."""
    floor = [p for p in placements if int(p.get("level", 0)) == 0]
    if not floor:
        return 0.0
    return max(float(p["x"]) + float(p["l"]) for p in floor)


def _rebuild_top_free_rects_from_geometry(
    placements: List[Dict[str, Any]], eps: float = 1e-4
) -> None:
    """Recompute each box's top free rectangles from 3D overlaps (ignores stale guillotine state)."""
    for base in placements:
        ztop = float(base["z"]) + float(base["h"])
        bx, by = float(base["x"]), float(base["y"])
        bl, bb = float(base["l"]), float(base["b"])
        rects: List[Tuple[float, float, float, float]] = [(0.0, 0.0, bl, bb)]
        for q in placements:
            if q is base:
                continue
            if abs(float(q["z"]) - ztop) > eps:
                continue
            qx, qy, ql, qb = float(q["x"]), float(q["y"]), float(q["l"]), float(q["b"])
            if not (qx < bx + bl - eps and qx + ql > bx + eps and qy < by + bb - eps and qy + qb > by + eps):
                continue
            ox1 = max(bx, qx)
            oy1 = max(by, qy)
            ox2 = min(bx + bl, qx + ql)
            oy2 = min(by + bb, qy + qb)
            local = (ox1 - bx, oy1 - by, ox2 - ox1, oy2 - oy1)
            nxt: List[Tuple[float, float, float, float]] = []
            for fr in rects:
                nxt.extend(_subtract_rect(fr, local, eps=eps))
            rects = merge_free_rectangles(
                [r for r in nxt if r[2] > _MIN_FREE_DIM and r[3] > _MIN_FREE_DIM]
            )
        base["top_free_rects"] = rects


def _has_immediate_stack_child(
    p: Dict[str, Any], placements: List[Dict[str, Any]], eps: float = 1e-4
) -> bool:
    """True if another box rests on p's top face (cannot relocate p without moving that child)."""
    zt = float(p["z"]) + float(p["h"])
    px, py, pl, pb = float(p["x"]), float(p["y"]), float(p["l"]), float(p["b"])
    for q in placements:
        if q is p:
            continue
        if abs(float(q["z"]) - zt) > eps:
            continue
        qx, qy, ql, qb = float(q["x"]), float(q["y"]), float(q["l"]), float(q["b"])
        if qx < px + pl - eps and qx + ql > px + eps and qy < py + pb - eps and qy + qb > py + eps:
            return True
    return False


def _placement_to_item_dict(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "idx": int(p["idx"]),
        "id": int(p["ID"]),
        "name": str(p["name"]),
        "weight": float(p["weight"]),
        "l": float(p["l"]),
        "b": float(p["b"]),
        "h": float(p["h"]),
        "max_stack": int(p.get("stack_limit", 10)),
        "volume": float(p["l"]) * float(p["b"]) * float(p["h"]),
        "footprint": float(p["l"]) * float(p["b"]),
        "adr": False,
        "adr_class": "",
    }


def refine_upper_levels_toward_back(
    placements: List[Dict[str, Any]],
    t_length: float,
    t_width: float,
    t_height: float,
) -> None:
    """
    Repeated passes by level (1, 2, …): try to move a box that has nothing on top onto
    unused top space on bases that are either on the floor or still within the level-0
    floor plan length (rear / above-deck), strictly improving (x+l, x, y, z) score.
    """
    max_passes = 48
    for _ in range(max_passes):
        moved = False
        _rebuild_top_free_rects_from_geometry(placements)
        back_len = _floor_plan_max_x(placements)
        if back_len <= 1e-12:
            break

        def base_in_back_zone(b: Dict[str, Any]) -> bool:
            if float(b["z"]) < 1e-6:
                return True
            return float(b["x"]) + float(b["l"]) <= back_len + 1e-6

        max_lev = max(int(p.get("level", 0)) for p in placements)
        for L in range(1, max_lev + 1):
            cands = [
                p
                for p in placements
                if int(p.get("level", 0)) == L and not _has_immediate_stack_child(p, placements)
            ]
            cands.sort(
                key=lambda p: (
                    -(float(p["x"]) + float(p["l"])),
                    -float(p["x"]),
                    int(p["idx"]),
                )
            )
            for P in list(cands):
                if P not in placements:
                    continue
                old_score = _layer_score(P)
                placements.remove(P)
                _rebuild_top_free_rects_from_geometry(placements)
                item = _placement_to_item_dict(P)
                probe_m = copy.deepcopy(placements)
                cand_m = try_stack_on_merged_rear_floor_shelf(
                    item, probe_m, t_height, t_length, t_width
                )
                probe_s = copy.deepcopy(placements)
                cand_s = try_stack(
                    item,
                    probe_s,
                    t_height,
                    base_allowed=base_in_back_zone,
                )
                cand_opts: List[Tuple[Tuple[float, float, float, float], Dict[str, Any], List]] = []
                if cand_m is not None and _layer_score(cand_m) < old_score:
                    cand_opts.append((_layer_score(cand_m), cand_m, probe_m))
                if cand_s is not None and _layer_score(cand_s) < old_score:
                    cand_opts.append((_layer_score(cand_s), cand_s, probe_s))
                cand_entry = min(cand_opts, key=lambda t: t[0]) if cand_opts else None
                if cand_entry is None:
                    placements.append(P)
                    _rebuild_top_free_rects_from_geometry(placements)
                    continue
                _sc, cand, probe = cand_entry
                if (
                    float(cand["x"]) < -1e-9
                    or float(cand["y"]) < -1e-9
                    or float(cand["z"]) < -1e-9
                    or float(cand["x"]) + float(cand["l"]) > t_length + 1e-6
                    or float(cand["y"]) + float(cand["b"]) > t_width + 1e-6
                    or float(cand["z"]) + float(cand["h"]) > t_height + 1e-6
                ):
                    placements.append(P)
                    _rebuild_top_free_rects_from_geometry(placements)
                    continue
                cand["stack_limit"] = int(P.get("stack_limit", item["max_stack"]))
                probe.append(cand)
                placements[:] = probe
                _rebuild_top_free_rects_from_geometry(placements)
                moved = True
        if not moved:
            break


def build_truck_plan(
    truck: pd.Series,
    truck_dims: Tuple[float, float, float],
    max_weight: float,
    max_volume: float,
    placements: List[Dict[str, Any]],
    unplaced: List[Dict[str, Any]],
    total_weight: float,
    total_volume: float,
) -> Dict[str, Any]:
    t_length, t_width, t_height = truck_dims
    placements.sort(key=lambda p: (p["x"], p["y"], p["z"]))
    placements_out = [{k: v for k, v in p.items() if k != "top_free_rects"} for p in placements]
    adr_truck = _as_bool(truck.get("adr_suitable"), False)
    adr_cls_raw = truck.get("adr_classes_allowed", "")
    if adr_cls_raw is None or (isinstance(adr_cls_raw, float) and pd.isna(adr_cls_raw)):
        adr_cls_disp = "*" if adr_truck else "—"
    else:
        adr_cls_disp = str(adr_cls_raw).strip() or ("*" if adr_truck else "—")

    return {
        "truck": truck["name"],
        "truck_dims": {"l": t_length, "b": t_width, "h": t_height},
        "adr_suitable": adr_truck,
        "adr_classes_allowed": adr_cls_disp,
        "placed_count": len(placements),
        "unplaced_count": len(unplaced),
        "weight_used": round(total_weight, 2),
        "weight_util_pct": round((total_weight / max_weight) * 100, 2) if max_weight > 0 else 0,
        "volume_used": round(total_volume, 2),
        "volume_util_pct": round((total_volume / max_volume) * 100, 2) if max_volume > 0 else 0,
        "placements": placements_out,
        "unplaced": [
            {
                "ID": u["id"],
                "name": u["name"],
                "weight": u["weight"],
                "adr": u.get("adr", False),
                "adr_class": u.get("adr_class", ""),
            }
            for u in unplaced
        ],
    }


def truck_dims_from_row(truck: pd.Series) -> Tuple[float, float, float]:
    return float(truck["l"]), float(truck["b"]), float(truck["h"])


def _overlap_area_xy(
    ax: float, ay: float, al: float, ab: float,
    bx: float, by: float, bl: float, bb: float,
    eps: float = 1e-9,
) -> float:
    ox1 = max(ax, bx)
    oy1 = max(ay, by)
    ox2 = min(ax + al, bx + bl)
    oy2 = min(ay + ab, by + bb)
    if (ox2 - ox1) <= eps or (oy2 - oy1) <= eps:
        return 0.0
    return float((ox2 - ox1) * (oy2 - oy1))


def _multiple_items_share_same_bottom_z(
    placements: List[Dict[str, Any]], z_decimals: int = 4
) -> bool:
    """
    True when two or more boxes share the same bottom z (same horizontal layer).
    Then a second pass can try to shorten truck-length use without changing stack heights.
    """
    counts: Dict[float, int] = defaultdict(int)
    for p in placements:
        z = round(float(p["z"]), z_decimals)
        counts[z] += 1
    return any(c > 1 for c in counts.values())


def _direct_supporters_sorted(
    placements: List[Dict[str, Any]], i: int, z_eps: float
) -> List[int]:
    """Indices whose top face touches the bottom of box i (best overlap first)."""
    p = placements[i]
    zi = float(p["z"])
    if zi < z_eps:
        return []
    scored: List[Tuple[float, int]] = []
    for j, q in enumerate(placements):
        if j == i:
            continue
        top_z = float(q["z"]) + float(q["h"])
        if abs(top_z - zi) > z_eps:
            continue
        a = _overlap_area_xy(
            float(p["x"]), float(p["y"]), float(p["l"]), float(p["b"]),
            float(q["x"]), float(q["y"]), float(q["l"]), float(q["b"]),
        )
        if a > 1e-12:
            scored.append((-a, j))
    scored.sort()
    return [j for _, j in scored]


def _floor_root_index(placements: List[Dict[str, Any]], i: int, z_eps: float) -> int:
    """Walk primary support chain to the floor piece that anchors this stack in x."""
    seen: set[int] = set()
    cur = i
    while True:
        if cur in seen:
            return i
        seen.add(cur)
        p = placements[cur]
        if float(p["z"]) < z_eps:
            return cur
        supp = _direct_supporters_sorted(placements, cur, z_eps)
        if not supp:
            return cur
        cur = supp[0]


def _groups_by_floor_root(
    placements: List[Dict[str, Any]], z_eps: float
) -> List[List[int]]:
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(placements)):
        buckets[_floor_root_index(placements, i, z_eps)].append(i)
    return list(buckets.values())


def _feasible_group_shift_x(
    placements: List[Dict[str, Any]],
    group: set[int],
    dx: float,
    t_length: float,
    t_width: float,
    t_height: float,
    z_eps: float,
) -> bool:
    """True if shifting all boxes in `group` by dx in x keeps bounds, support, and no 3D overlaps."""
    n = len(placements)

    def x_at(idx: int) -> float:
        return float(placements[idx]["x"]) + (dx if idx in group else 0.0)

    for i in group:
        p = placements[i]
        x = x_at(i)
        y, z = float(p["y"]), float(p["z"])
        l, b, h = float(p["l"]), float(p["b"]), float(p["h"])
        if x < -1e-9 or x + l > t_length + 1e-9:
            return False
        if y < -1e-9 or y + b > t_width + 1e-9:
            return False
        if z < -1e-9 or z + h > t_height + 1e-9:
            return False

    for ia in range(n):
        for ib in range(ia + 1, n):
            pa, pb = placements[ia], placements[ib]
            xa, ya, za = x_at(ia), float(pa["y"]), float(pa["z"])
            xb, yb, zb = x_at(ib), float(pb["y"]), float(pb["z"])
            la, ba, ha = float(pa["l"]), float(pa["b"]), float(pa["h"])
            lb, bb, hb = float(pb["l"]), float(pb["b"]), float(pb["h"])
            if xa < xb + lb and xb < xa + la and ya < yb + bb and yb < ya + ba and za < zb + hb and zb < za + ha:
                return False

    for i in group:
        p = placements[i]
        zi = float(p["z"])
        if zi < z_eps:
            continue
        xi = x_at(i)
        yi = float(p["y"])
        li, bi = float(p["l"]), float(p["b"])
        supporters: List[Tuple[float, float, float, float]] = []
        for j, q in enumerate(placements):
            if j == i:
                continue
            top_z = float(q["z"]) + float(q["h"])
            if abs(top_z - zi) > z_eps:
                continue
            xj = x_at(j)
            yj = float(q["y"])
            lj, bj = float(q["l"]), float(q["b"])
            if _overlap_area_xy(xi, yi, li, bi, xj, yj, lj, bj) <= 1e-12:
                continue
            supporters.append((xj, yj, lj, bj))
        if not supporters:
            return False
        if not _footprint_fully_supported(xi, yi, li, bi, supporters):
            return False

    return True


def _max_left_shift_for_group(
    placements: List[Dict[str, Any]],
    group: List[int],
    t_length: float,
    t_width: float,
    t_height: float,
    z_eps: float,
) -> float:
    """Most negative dx (shift toward back) that stays feasible; 0 if none."""
    G = set(group)
    if not G:
        return 0.0
    lo_bound = -min(float(placements[i]["x"]) for i in G)
    if lo_bound >= -1e-12:
        return 0.0

    def ok(d: float) -> bool:
        return _feasible_group_shift_x(placements, G, d, t_length, t_width, t_height, z_eps)

    if not ok(0.0):
        return 0.0
    if ok(lo_bound):
        return lo_bound
    lo, hi = lo_bound, 0.0
    for _ in range(56):
        mid = (lo + hi) * 0.5
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def compact_placements_minimize_truck_length_x(
    placements: List[Dict[str, Any]],
    t_length: float,
    t_width: float,
    t_height: float,
) -> float:
    """
    Second pass: slide whole support-linked groups toward x=0 to reduce max(x+l).
    Returns approximate length (m) saved along the truck.
    """
    if len(placements) < 2:
        return 0.0
    z_eps = 1e-4
    before = max(float(p["x"]) + float(p["l"]) for p in placements)
    for _ in range(64):
        groups = _groups_by_floor_root(placements, z_eps)
        ordered = sorted(
            groups,
            key=lambda G: max(float(placements[i]["x"]) + float(placements[i]["l"]) for i in G),
            reverse=True,
        )
        moved_any = False
        for G in ordered:
            dx = _max_left_shift_for_group(placements, G, t_length, t_width, t_height, z_eps)
            if dx < -1e-7:
                for i in G:
                    placements[i]["x"] = float(placements[i]["x"]) + dx
                moved_any = True
        if not moved_any:
            break
    after = max(float(p["x"]) + float(p["l"]) for p in placements)
    return max(0.0, before - after)


def optimize_load(goods_df: pd.DataFrame, truck: pd.Series) -> Dict[str, Any]:
    t_length, t_width, t_height = truck_dims_from_row(truck)
    max_weight = float(truck["max_weight_kg"])
    geom_vol = t_length * t_width * t_height
    max_volume = float(truck["max_volume_m3"]) if pd.notna(truck["max_volume_m3"]) else geom_vol

    items = build_items_for_loading(goods_df, t_width)
    placements: List[Dict[str, Any]] = []
    free_rects: List[Tuple[float, float, float, float]] = [(0.0, 0.0, t_length, t_width)]
    total_weight = 0.0
    total_volume = 0.0
    unplaced: List[Dict[str, Any]] = []

    pending: List[Dict[str, Any]] = []
    for item in items:
        if not adr_good_allowed_on_truck(item, truck):
            unplaced.append(item)
        else:
            pending.append(item)

    # Highest volume first; then larger footprint / height so big bases seed the rear.
    pack_order = sorted(
        pending,
        key=lambda x: (-float(x["volume"]), -float(x["footprint"]), -float(x["h"]), int(x["idx"])),
    )

    for item in pack_order:
        if not can_load_by_capacity(item, total_weight, total_volume, max_weight, max_volume):
            unplaced.append(item)
            continue

        floor_level_items = [p for p in placements if int(p.get("level", 0)) == 0]
        floor_frontier_x: Optional[float] = (
            max(float(p["x"]) + float(p["l"]) for p in floor_level_items)
            if floor_level_items
            else None
        )
        # Remaining breadth beside the *back* floor cluster only (min x on floor). If we
        # used global max(y+b), forward floor rows (same plan, larger x) would hide the
        # wall strip beside 1032 and force narrow kollis to stack instead.
        min_x_floor = min(float(p["x"]) for p in floor_level_items) if floor_level_items else 0.0
        back_floor = [
            p
            for p in floor_level_items
            if float(p["x"]) <= min_x_floor + 1e-6
        ]
        max_y_back = (
            max(float(p["y"]) + float(p["b"]) for p in back_floor) if back_floor else 0.0
        )
        strip_y = float(t_width) - max_y_back
        min_footprint_edge = min(float(item["l"]), float(item["b"]))
        # Narrow goods can use remaining breadth beside the rear row (e.g. kolli beside 1032).
        prefer_strip_floor_first = (
            floor_frontier_x is not None
            and strip_y > _MIN_FREE_DIM + 1e-9
            and min_footprint_edge <= strip_y + 1e-9
        )

        placed: Optional[Dict[str, Any]] = None

        if prefer_strip_floor_first:
            placed = try_place_on_floor(
                item,
                free_rects,
                t_height,
                max_front_x=floor_frontier_x,
            )

        # Stack on combined rear-floor shelf (coplanar tops, e.g. 1032 + 1099), then single-base.
        if placed is None and placements:
            placed = try_stack_on_merged_rear_floor_shelf(
                item, placements, t_height, t_length, t_width
            )
        if placed is None and placements:
            placed = try_stack(item, placements, t_height)

        # Floor within current max floor length (wide goods after failed stack).
        if placed is None and floor_frontier_x is not None:
            placed = try_place_on_floor(
                item,
                free_rects,
                t_height,
                max_front_x=floor_frontier_x,
            )

        # Extend floor along +x when stack and in-depth floor both failed.
        if placed is None:
            placed = try_place_on_floor(item, free_rects, t_height, max_front_x=None)

        if placed is None:
            unplaced.append(item)
            continue

        placed["stack_limit"] = item["max_stack"]
        placements.append(placed)
        _rebuild_top_free_rects_from_geometry(placements)
        total_weight += item["weight"]
        total_volume += item["volume"]

    # Pull upper levels onto rear / within floor-plan tops where geometry allows (strict
    # improvement in forward extent), without moving level-0.
    if placements:
        refine_upper_levels_toward_back(placements, t_length, t_width, t_height)

    # Pass 2: when several boxes share the same bottom z (a wide shelf / floor layer),
    # compact whole support columns along x to shorten how far the load reaches (+x).
    if placements and _multiple_items_share_same_bottom_z(placements):
        compact_placements_minimize_truck_length_x(placements, t_length, t_width, t_height)

    return build_truck_plan(
        truck=truck,
        truck_dims=(t_length, t_width, t_height),
        max_weight=max_weight,
        max_volume=max_volume,
        placements=placements,
        unplaced=unplaced,
        total_weight=total_weight,
        total_volume=total_volume,
    )


def allocate_with_subset(
    goods_df: pd.DataFrame, truck_subset: pd.DataFrame
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    remaining = goods_df.copy().reset_index(drop=True)
    plans_local: List[Dict[str, Any]] = []
    ordered = truck_subset.sort_values(
        by=["adr_suitable", "max_weight_kg", "max_volume_m3", "l"],
        ascending=[False, False, False, False],
    )
    for _, truck in ordered.iterrows():
        if remaining.empty:
            break
        plan = optimize_load(remaining, truck)
        plans_local.append(plan)
        loaded_idxs = {p["idx"] for p in plan["placements"]}
        remaining = remaining.drop(index=sorted(loaded_idxs)).reset_index(drop=True)
    return plans_local, remaining


def utilization_score(candidate: Tuple[List[Dict[str, Any]], pd.DataFrame]) -> float:
    plans_candidate = candidate[0]
    if not plans_candidate:
        return 0.0
    return sum(p["weight_util_pct"] + p["volume_util_pct"] for p in plans_candidate)


def select_best_plans(
    goods_df: pd.DataFrame, trucks_df: pd.DataFrame
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    n_trucks = len(trucks_df)
    for k in range(1, n_trucks + 1):
        feasible: List[Tuple[List[Dict[str, Any]], pd.DataFrame]] = []
        for idxs in combinations(range(n_trucks), k):
            subset = trucks_df.iloc[list(idxs)]
            plans_cand, remaining_cand = allocate_with_subset(goods_df, subset)
            if remaining_cand.empty:
                feasible.append((plans_cand, remaining_cand))
        if feasible:
            return max(feasible, key=utilization_score)
    return allocate_with_subset(goods_df, trucks_df)


def print_plan(plan: Dict[str, Any]) -> None:
    print(f"\nTruck: {plan['truck']}")
    dims = plan["truck_dims"]
    adr_ok = plan.get("adr_suitable", False)
    adr_allow = plan.get("adr_classes_allowed", "*")
    print(f"  ADR: suitable={adr_ok}, classes_allowed={adr_allow}")
    print(f"  Truck dims (l x b x h): {dims['l']} x {dims['b']} x {dims['h']}")
    print(
        f"  Loaded: {plan['placed_count']} goods | "
        f"Weight: {plan['weight_used']} kg ({plan['weight_util_pct']}%) | "
        f"Volume: {plan['volume_used']} m3 ({plan['volume_util_pct']}%)"
    )
    print("  Placements:")
    for p in plan["placements"]:
        print(
            f"    - ID {p['ID']} ({p['name']}), "
            f"pos=({p['x']:.2f},{p['y']:.2f},{p['z']:.2f}), "
            f"size=({p['l']:.2f}x{p['b']:.2f}x{p['h']:.2f}), "
            f"w={p['weight']} kg, level={p['level']}"
        )
    if plan["unplaced"]:
        # print("  Not loaded in this truck:")
        for u in plan["unplaced"]:
            adr_note = ""
            if u.get("adr"):
                adr_note = f", ADR class {u.get('adr_class', '') or '?'}"
            # print(f"    - ID {u['ID']} ({u['name']}), w={u['weight']} kg{adr_note}")


def print_allocation_summary(plans: List[Dict[str, Any]], remaining: pd.DataFrame) -> None:
    for plan in plans:
        print_plan(plan)
    if not remaining.empty:
        print("\nStill unassigned after all trucks:")
        for _, row in remaining.iterrows():
            extra = ""
            if _as_bool(row.get("adr"), False):
                extra = f" [ADR class {row.get('adr_class', '') or '?'}]"
            print(f"  - ID {row['id']} ({row['name']}){extra} w={row['weight_kg']} kg, l={row['l']} m, b={row['b']} m, h={row['h']} m")
    else:
        print(f"\nAll goods assigned with {len(plans)} truck(s).")


def run_allocation(
    goods_df: pd.DataFrame,
    trucks_df: pd.DataFrame,
    *,
    show_3d_preview: bool = False,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    plans, remaining = select_best_plans(goods_df, trucks_df)
    print_allocation_summary(plans, remaining)

    if show_3d_preview:
        try:
            import preview_3d

            preview_3d.show_load_preview(goods_df, plans, remaining)
        except ImportError:
            print(
                "3D preview skipped: install matplotlib (pip install matplotlib) "
                "and run again."
            )

    return plans, remaining


if __name__ == "__main__":
    run_allocation(
        GOODS_SAMPLE.copy(),
        TRUCKS_SAMPLE.copy(),
        show_3d_preview=True,
    )
