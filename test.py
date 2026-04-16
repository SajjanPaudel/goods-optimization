"""
Goods–truck allocation: standardized pandas DataFrames only (no parsing, no classes).
"""

import copy
from collections import defaultdict
from itertools import combinations, permutations
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
# Items with volume below this cap (and below a fraction of truck box volume) are biased
# toward +x ("front" / cab-ward in this coordinate system) on floor and when stacking.
_SMALL_ITEM_VOL_ABS_M3 = 0.35
_SMALL_ITEM_VOL_FRAC_OF_CARGO = 0.0025


def _volume_small_for_front(item: Dict[str, Any], cargo_geom_vol: Optional[float]) -> bool:
    if cargo_geom_vol is None or cargo_geom_vol <= 0:
        return float(item.get("volume", 0.0)) <= _SMALL_ITEM_VOL_ABS_M3
    cap = max(
        _SMALL_ITEM_VOL_ABS_M3,
        _SMALL_ITEM_VOL_FRAC_OF_CARGO * float(cargo_geom_vol),
    )
    return float(item.get("volume", 0.0)) <= cap


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
                "stackable": _as_bool(row.get("stackable", True), True),
                "max_stack": int(row["max_stack"]),
                "volume": _item_volume(row),
                "footprint": _item_footprint(row),
                "adr": _as_bool(row.get("adr", False), False),
                "adr_class": str(row.get("adr_class", "") or "").strip(),
            }
        )
    # Prioritize floor surface area coverage first (footprint), not weight.
    # Defer floor-hog items that consume most of truck width so narrow items can
    # seed back-right lanes first, preserving maneuvering space for later fits.
    for item in items:
        min_cross_width = min(item["l"], item["b"])
        item["defer_wide_floor_hog"] = (
            min_cross_width >= (0.75 * truck_width) and item["footprint"] >= 3.5
        )
    items.sort(
        key=lambda x: (
            -x["footprint"],  # Larger base first (surface-area priority)
            -x["volume"],
            x["h"],  # Shorter on ties: spread on deck before growing height
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


def _stack_elevation_key(
    new_level: int, new_z: float, waste_frac: float, layer_sc: Tuple[float, float, float, float]
) -> Tuple[int, float, Tuple[float, float, float, float], float]:
    """Lower is better: fewer stack layers, lower z, plan score, then top-surface waste."""
    return (new_level, round(new_z, 6), layer_sc, round(waste_frac, 8))


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


def _layer_score_for_item(
    item: Dict[str, Any],
    p: Dict[str, Any],
    cargo_geom_vol: Optional[float],
) -> Tuple[float, float, float, float]:
    """Like _layer_score, but small-volume items prefer larger +x (front of load)."""
    xl = round(float(p["x"]) + float(p["l"]), 4)
    x0 = round(float(p["x"]), 4)
    y = round(float(p["y"]), 4)
    z = round(float(p["z"]), 4)
    if _volume_small_for_front(item, cargo_geom_vol):
        return (-xl, -x0, y, z)
    return (xl, x0, y, z)


def _stack_top_waste_frac(il: float, ib: float, rl: float, rb: float, eps: float = 1e-12) -> float:
    """Fraction of free top rectangle still empty after placing (il, ib); lower = better shelf use."""
    denom = rl * rb + eps
    return max(0.0, (rl * rb - il * ib) / denom)


def _candidate_longitudinal_balance_score(
    candidate: Dict[str, Any], placements: List[Dict[str, Any]], truck_length: float
) -> float:
    """
    Lower is better: keep the longitudinal center of mass near the truck midpoint
    to reduce front/rear axle concentration.
    """
    total_weight = float(candidate["weight"])
    weighted_x = float(candidate["weight"]) * (
        float(candidate["x"]) + 0.5 * float(candidate["l"])
    )
    for p in placements:
        w = float(p["weight"])
        total_weight += w
        weighted_x += w * (float(p["x"]) + 0.5 * float(p["l"]))
    if total_weight <= 1e-12:
        return 0.0
    com_x = weighted_x / total_weight
    return abs(com_x - 0.5 * truck_length)


def _placement_overlaps_any_base(
    placement: Dict[str, Any], support_bases: List[Dict[str, Any]], eps: float = 1e-9
) -> bool:
    px, py = float(placement["x"]), float(placement["y"])
    pl, pb = float(placement["l"]), float(placement["b"])
    for base in support_bases:
        bx, by = float(base["x"]), float(base["y"])
        bl, bb = float(base["l"]), float(base["b"])
        if _rect_overlap((px, py, pl, pb), (bx, by, bl, bb), eps=eps) is not None:
            return True
    return False


def _direct_children_of_base(
    placements: List[Dict[str, Any]], base: Dict[str, Any], eps: float = 1e-9
) -> List[Dict[str, Any]]:
    children: List[Dict[str, Any]] = []
    base_top_z = float(base["z"]) + float(base["h"])
    bx, by = float(base["x"]), float(base["y"])
    bl, bb = float(base["l"]), float(base["b"])
    for placement in placements:
        if placement is base:
            continue
        if abs(float(placement["z"]) - base_top_z) > eps:
            continue
        px, py = float(placement["x"]), float(placement["y"])
        pl, pb = float(placement["l"]), float(placement["b"])
        if _rect_overlap((px, py, pl, pb), (bx, by, bl, bb), eps=eps) is not None:
            children.append(placement)
    return children


def _direct_support_bases(
    placements: List[Dict[str, Any]], placement: Dict[str, Any], eps: float = 1e-9
) -> List[Dict[str, Any]]:
    supporters: List[Dict[str, Any]] = []
    pz = float(placement["z"])
    if pz < eps:
        return supporters
    px, py = float(placement["x"]), float(placement["y"])
    pl, pb = float(placement["l"]), float(placement["b"])
    for base in placements:
        if base is placement:
            continue
        base_top_z = float(base["z"]) + float(base["h"])
        if abs(base_top_z - pz) > eps:
            continue
        bx, by = float(base["x"]), float(base["y"])
        bl, bb = float(base["l"]), float(base["b"])
        if _rect_overlap((px, py, pl, pb), (bx, by, bl, bb), eps=eps) is not None:
            supporters.append(base)
    return supporters


def _ancestor_bases_closure(
    placements: List[Dict[str, Any]], bases: List[Dict[str, Any]], eps: float = 1e-9
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
    stack = list(bases)
    while stack:
        base = stack.pop()
        base_key = id(base)
        if base_key in seen:
            continue
        seen.add(base_key)
        out.append(base)
        stack.extend(_direct_support_bases(placements, base, eps=eps))
    return out


def _descendant_weight_sum(
    placements: List[Dict[str, Any]], base: Dict[str, Any], eps: float = 1e-9
) -> float:
    """
    Sum the weight of all transitive descendants stacked above `base`.
    Boxes spanning multiple supports count toward every base they rest on.
    """
    total = 0.0
    seen: set[int] = set()
    stack = _direct_children_of_base(placements, base, eps=eps)
    while stack:
        node = stack.pop()
        node_key = id(node)
        if node_key in seen:
            continue
        seen.add(node_key)
        total += float(node["weight"])
        stack.extend(_direct_children_of_base(placements, node, eps=eps))
    return total


def _weight_supported_by_bases(
    candidate: Dict[str, Any],
    support_bases: List[Dict[str, Any]],
    placements: List[Dict[str, Any]],
    eps: float = 1e-9,
) -> bool:
    """
    Reject stacks when either:
    1) the new item is heavier than any direct supporting base, or
    2) the total transitive load above any supporting base would exceed that base's
       own weight, or
    3) the total transitive load above the full supporting base set would exceed the
       combined weight of those base items.
    """
    item_weight = float(candidate["weight"])
    for base in support_bases:
        if item_weight > float(base["weight"]) + eps:
            return False

    augmented = list(placements)
    augmented.append(candidate)

    all_bases_to_check = _ancestor_bases_closure(augmented, support_bases, eps=eps)

    for base in all_bases_to_check:
        if _descendant_weight_sum(augmented, base, eps=eps) > float(base["weight"]) + eps:
            return False

    base_weight_sum = sum(float(base["weight"]) for base in support_bases)
    combined_supported_sum = 0.0
    seen_nodes: set[int] = set()
    for base in support_bases:
        stack = _direct_children_of_base(augmented, base, eps=eps)
        while stack:
            node = stack.pop()
            node_key = id(node)
            if node_key in seen_nodes:
                continue
            seen_nodes.add(node_key)
            combined_supported_sum += float(node["weight"])
            stack.extend(_direct_children_of_base(augmented, node, eps=eps))
    return combined_supported_sum <= (base_weight_sum + eps)


def try_place_on_floor(
    item: Dict[str, Any],
    placements: List[Dict[str, Any]],
    free_rects: List[Tuple[float, float, float, float]],
    truck_height: float,
    truck_length: float,
    max_front_x: Optional[float] = None,
    cargo_geom_vol: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    best: Optional[Tuple[Tuple[float, float, float, float, float], int, float, float, float, float]] = None
    front_small = _volume_small_for_front(item, cargo_geom_vol)

    for ridx, (rx, ry, rl, rb) in enumerate(free_rects):
        for il, ib in _preferred_orientations(item):
            if il > rl + 1e-9 or ib > rb + 1e-9 or item["h"] > truck_height:
                continue

            # Constraint: Stay within the current "load zone" if provided
            if max_front_x is not None and (rx + il) > (max_front_x + 1e-9):
                continue

            cand_box = {
                "x": rx,
                "y": ry,
                "z": 0.0,
                "l": il,
                "b": ib,
                "h": float(item["h"]),
                "weight": float(item["weight"]),
            }
            balance_sc = round(
                _candidate_longitudinal_balance_score(cand_box, placements, truck_length), 4
            )

            # Large items: back-first (small x). Small-by-volume: prefer +x (front / cab-ward).
            if front_small:
                score = (
                    balance_sc,
                    -round(rx + il, 4),
                    -round(rx, 4),
                    round(ry, 4),
                    (rl * rb) - (il * ib),
                    -ib,
                )
            else:
                score = (
                    balance_sc,
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
    truck_length: float,
    base_allowed: Optional[Callable[[Dict[str, Any]], bool]] = None,
    cargo_geom_vol: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    eps = 1e-9
    sorted_bases = sorted(
        placements,
        key=lambda p: (round(float(p["x"]), 4), round(float(p["y"]), 4), float(p["z"])),
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
                cand_box = {
                    "x": wx,
                    "y": wy,
                    "z": new_z,
                    "l": il,
                    "b": ib,
                    "h": float(item["h"]),
                    "weight": float(item["weight"]),
                }
                if not _weight_supported_by_bases(cand_box, [base], placements, eps=eps):
                    continue
                balance_sc = round(
                    _candidate_longitudinal_balance_score(cand_box, placements, truck_length), 4
                )
                wf = _stack_top_waste_frac(il, ib, rl, rb, eps=eps)
                if _volume_small_for_front(item, cargo_geom_vol):
                    score = (
                        balance_sc,
                        round(wf, 8),
                        -round(wx + il, 4),
                        -round(wx, 4),
                        round(wy, 4),
                        -ib,
                    )
                else:
                    score = (
                        balance_sc,
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
        glob_tie = _stack_elevation_key(
            new_level, new_z, wf_g, _layer_score_for_item(item, cand_box, cargo_geom_vol)
        )
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


def _is_lonely_coplanar_top(
    base: Dict[str, Any], placements: List[Dict[str, Any]], eps: float = 1e-4
) -> bool:
    """True if no other placement shares this base's top z (single-base shelf only)."""
    zt = float(base["z"]) + float(base["h"])
    n = sum(1 for p in placements if abs(float(p["z"]) + float(p["h"]) - zt) <= eps)
    return n == 1


def try_stack_on_merged_coplanar_tops(
    item: Dict[str, Any],
    placements: List[Dict[str, Any]],
    truck_height: float,
    t_length: float,
    t_width: float,
    eps: float = 1e-4,
    cargo_geom_vol: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Stack only on shelves formed by >=2 placements sharing the same top z (coplanar tops),
    after subtracting whatever already sits on that plane. Wide items can span multiple
    bases at the same elevation; lone tops are left to try_stack.
    """
    if len(placements) < 2:
        return None

    by_ztop: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for p in placements:
        zt = round(float(p["z"]) + float(p["h"]), 6)
        by_ztop[zt].append(p)

    best: Optional[Tuple[Tuple[float, float, float, float], Dict[str, Any]]] = None

    for _zk, parts in by_ztop.items():
        if len(parts) < 2:
            continue
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
                        support_bases = [
                            f
                            for f in parts
                            if any(
                                abs(float(f["x"]) - sx) <= eps
                                and abs(float(f["y"]) - sy) <= eps
                                and abs(float(f["l"]) - sl) <= eps
                                and abs(float(f["b"]) - sb) <= eps
                                for sx, sy, sl, sb in supports_full
                            )
                        ]
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
                        if not _weight_supported_by_bases(
                            cand, support_bases, placements, eps=eps
                        ):
                            continue
                        balance_sc = round(
                            _candidate_longitudinal_balance_score(cand, placements, t_length), 4
                        )
                        sc = _layer_score_for_item(item, cand, cargo_geom_vol)
                        wf_m = _stack_top_waste_frac(il, ib, rl, rb, eps=eps)
                        if _volume_small_for_front(item, cargo_geom_vol):
                            x_tail = (-round(wx + il, 4), -round(wx, 4))
                        else:
                            x_tail = (round(wx + il, 4), round(wx, 4))
                        tie = (
                            new_level,
                            round(Zt, 6),
                            balance_sc,
                            sc,
                            round(wf_m, 8),
                            x_tail[0],
                            x_tail[1],
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
        "stackable": _as_bool(p.get("stackable", True), True),
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
    cargo_geom_vol: Optional[float] = None,
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
                item = _placement_to_item_dict(P)
                old_score = _layer_score_for_item(item, P, cargo_geom_vol)
                placements.remove(P)
                _rebuild_top_free_rects_from_geometry(placements)
                probe_s1 = copy.deepcopy(placements)
                cand_s1 = try_stack(
                    item,
                    probe_s1,
                    t_height,
                    t_length,
                    base_allowed=lambda b: base_in_back_zone(b)
                    and _is_lonely_coplanar_top(b, probe_s1),
                    cargo_geom_vol=cargo_geom_vol,
                )
                probe_s2 = copy.deepcopy(placements)
                cand_s2 = try_stack(
                    item,
                    probe_s2,
                    t_height,
                    t_length,
                    base_allowed=base_in_back_zone,
                    cargo_geom_vol=cargo_geom_vol,
                )
                probe_m = copy.deepcopy(placements)
                cand_m = try_stack_on_merged_coplanar_tops(
                    item,
                    probe_m,
                    t_height,
                    t_length,
                    t_width,
                    cargo_geom_vol=cargo_geom_vol,
                )
                cand_opts: List[Tuple[Tuple[float, float, float, float], Dict[str, Any], List]] = []
                if cand_s1 is not None and _layer_score_for_item(item, cand_s1, cargo_geom_vol) < old_score:
                    cand_opts.append(
                        (_layer_score_for_item(item, cand_s1, cargo_geom_vol), cand_s1, probe_s1)
                    )
                if cand_s2 is not None and _layer_score_for_item(item, cand_s2, cargo_geom_vol) < old_score:
                    cand_opts.append(
                        (_layer_score_for_item(item, cand_s2, cargo_geom_vol), cand_s2, probe_s2)
                    )
                if cand_m is not None and _layer_score_for_item(item, cand_m, cargo_geom_vol) < old_score:
                    cand_opts.append(
                        (_layer_score_for_item(item, cand_m, cargo_geom_vol), cand_m, probe_m)
                    )
                cand_entry = (
                    min(
                        cand_opts,
                        key=lambda t: (
                            int(t[1].get("level", 0)),
                            round(float(t[1]["z"]), 6),
                            t[0],
                        ),
                    )
                    if cand_opts
                    else None
                )
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

    # Primary pass: floor surface area first across all eligible goods.
    pack_order = sorted(
        pending,
        key=lambda x: (
            -float(x["footprint"]),
            -float(x["volume"]),
            float(x["h"]),
            int(x["idx"]),
        ),
    )

    def _try_place_item(item: Dict[str, Any], *, prefer_stack_first: bool) -> Optional[Dict[str, Any]]:
        floor_level_items = [p for p in placements if int(p.get("level", 0)) == 0]
        floor_frontier_x: Optional[float] = (
            max(float(p["x"]) + float(p["l"]) for p in floor_level_items)
            if floor_level_items
            else None
        )

        placed_local: Optional[Dict[str, Any]] = None
        if prefer_stack_first and placements:
            placed_local = try_stack_on_merged_coplanar_tops(
                item,
                placements,
                t_height,
                t_length,
                t_width,
                cargo_geom_vol=geom_vol,
            )
            if placed_local is None:
                placed_local = try_stack(
                    item,
                    placements,
                    t_height,
                    t_length,
                    base_allowed=lambda b, pl=placements: _is_lonely_coplanar_top(b, pl),
                    cargo_geom_vol=geom_vol,
                )
            if placed_local is None:
                placed_local = try_stack(
                    item, placements, t_height, t_length, cargo_geom_vol=geom_vol
                )

        if placed_local is None and floor_frontier_x is not None:
            placed_local = try_place_on_floor(
                item,
                placements,
                free_rects,
                t_height,
                t_length,
                max_front_x=floor_frontier_x,
                cargo_geom_vol=geom_vol,
            )
        if placed_local is None:
            placed_local = try_place_on_floor(
                item,
                placements,
                free_rects,
                t_height,
                t_length,
                max_front_x=None,
                cargo_geom_vol=geom_vol,
            )

        if placed_local is None and (not prefer_stack_first) and placements:
            placed_local = try_stack_on_merged_coplanar_tops(
                item,
                placements,
                t_height,
                t_length,
                t_width,
                cargo_geom_vol=geom_vol,
            )
            if placed_local is None:
                placed_local = try_stack(
                    item,
                    placements,
                    t_height,
                    t_length,
                    base_allowed=lambda b, pl=placements: _is_lonely_coplanar_top(b, pl),
                    cargo_geom_vol=geom_vol,
                )
            if placed_local is None:
                placed_local = try_stack(
                    item, placements, t_height, t_length, cargo_geom_vol=geom_vol
                )
        return placed_local

    for item in pack_order:
        if not can_load_by_capacity(item, total_weight, total_volume, max_weight, max_volume):
            unplaced.append(item)
            continue

        placed: Optional[Dict[str, Any]] = _try_place_item(
            item,
            prefer_stack_first=False,
        )

        if placed is None:
            unplaced.append(item)
            continue

        placed["stackable"] = _as_bool(item.get("stackable", True), True)
        placed["stack_limit"] = int(item["max_stack"]) if placed["stackable"] else 1
        placements.append(placed)
        _rebuild_top_free_rects_from_geometry(placements)
        total_weight += item["weight"]
        total_volume += item["volume"]

    # Secondary pass for non-stackable leftovers: attempt top insertion first.
    if unplaced:
        still_unplaced: List[Dict[str, Any]] = []
        for item in unplaced:
            if _as_bool(item.get("stackable", True), True):
                still_unplaced.append(item)
                continue
            if not can_load_by_capacity(item, total_weight, total_volume, max_weight, max_volume):
                still_unplaced.append(item)
                continue
            placed = _try_place_item(item, prefer_stack_first=True)
            if placed is None:
                still_unplaced.append(item)
                continue
            placed["stackable"] = _as_bool(item.get("stackable", True), True)
            placed["stack_limit"] = int(item["max_stack"]) if placed["stackable"] else 1
            placements.append(placed)
            _rebuild_top_free_rects_from_geometry(placements)
            total_weight += item["weight"]
            total_volume += item["volume"]
        unplaced = still_unplaced

    # Pull upper levels onto rear / within floor-plan tops where geometry allows (strict
    # improvement in forward extent), without moving level-0.
    if placements:
        refine_upper_levels_toward_back(
            placements, t_length, t_width, t_height, cargo_geom_vol=geom_vol
        )

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


def allocate_with_truck_order(
    goods_df: pd.DataFrame, trucks_in_order: List[pd.Series]
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Greedy load: fill each truck in sequence from the same remaining pool."""
    remaining = goods_df.copy().reset_index(drop=True)
    plans_local: List[Dict[str, Any]] = []
    for truck in trucks_in_order:
        if remaining.empty:
            break
        plan = optimize_load(remaining, truck)
        plans_local.append(plan)
        loaded_idxs = {p["idx"] for p in plan["placements"]}
        remaining = remaining.drop(index=sorted(loaded_idxs)).reset_index(drop=True)
    return plans_local, remaining


def allocate_with_subset(
    goods_df: pd.DataFrame, truck_subset: pd.DataFrame
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Try every order of trucks in the subset, then keep the best outcome.
    Ranking: minimize leftover goods count, then maximize utilization_score.
    (A single fixed 'big truck first' order can strand awkward pieces on the second truck.)
    """
    n = len(truck_subset)
    if n == 0:
        return [], goods_df.copy().reset_index(drop=True)

    best_plans: Optional[List[Dict[str, Any]]] = None
    best_remaining: Optional[pd.DataFrame] = None
    best_rank: Optional[Tuple[int, float]] = None

    for perm in permutations(range(n)):
        trucks_in_order = [truck_subset.iloc[i] for i in perm]
        plans_local, remaining = allocate_with_truck_order(goods_df, trucks_in_order)
        n_unplaced = len(remaining)
        util = utilization_score((plans_local, remaining))
        rank = (n_unplaced, -util)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_plans = plans_local
            best_remaining = remaining

    assert best_plans is not None and best_remaining is not None
    return best_plans, best_remaining


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
