"""
Goods–truck allocation: standardized pandas DataFrames only (no parsing, no classes).
"""

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Standard column schemas (see goods_sample.csv, trucks_sample.csv) ---
# Goods:  id, name, weight_kg, l, b, h, max_stack, adr, adr_class
# Trucks: id, name, l, b, h, max_weight_kg, max_volume_m3, adr_suitable, adr_classes_allowed
#
# ADR: goods with adr=True may only be loaded on trucks with adr_suitable=True and
# adr_class in adr_classes_allowed (comma-separated, or "*" for any class).

_DATA_DIR = Path(__file__).resolve().parent


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


def build_items_for_loading(goods_df: pd.DataFrame) -> List[Dict[str, Any]]:
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
    items.sort(key=lambda x: (x["weight"], x["volume"], x["footprint"]), reverse=True)
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
    out = [r for r in rects if r[2] > 0.05 and r[3] > 0.05]
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


def try_place_on_floor(
    item: Dict[str, Any],
    free_rects: List[Tuple[float, float, float, float]],
    truck_height: float,
) -> Optional[Dict[str, Any]]:
    for ridx, (rx, ry, rl, rb) in enumerate(list(free_rects)):
        for il, ib in [(item["l"], item["b"]), (item["b"], item["l"])]:
            if il <= rl + 1e-9 and ib <= rb + 1e-9 and item["h"] <= truck_height:
                del free_rects[ridx]
                right = (rx + il, ry, max(0.0, rl - il), ib)
                top = (rx, ry + ib, rl, max(0.0, rb - ib))
                for rect in (right, top):
                    if rect[2] > 0.05 and rect[3] > 0.05:
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
                    # Free top-area rectangles (local to this placement top surface).
                    "top_free_rects": [(0.0, 0.0, il, ib)],
                }
    return None


def try_stack(
    item: Dict[str, Any],
    placements: List[Dict[str, Any]],
    truck_height: float,
) -> Optional[Dict[str, Any]]:
    for base in sorted(placements, key=lambda p: (p["weight"], p["l"] * p["b"]), reverse=True):
        new_level = int(base.get("level", 0)) + 1
        # Stack limit is interpreted as max allowed levels count per column:
        # max_stack=2 allows level 0 (base) and level 1 (one tier above).
        if new_level >= int(base.get("stack_limit", 2)):
            continue
        top_free_rects: List[Tuple[float, float, float, float]] = base.setdefault(
            "top_free_rects", [(0.0, 0.0, base["l"], base["b"])]
        )
        new_z = base["z"] + base["h"]
        if new_z + item["h"] > truck_height + 1e-9:
            continue

        for ridx, (rx, ry, rl, rb) in enumerate(list(top_free_rects)):
            for il, ib in [(item["l"], item["b"]), (item["b"], item["l"])]:
                if il <= rl + 1e-9 and ib <= rb + 1e-9:
                    del top_free_rects[ridx]
                    right = (rx + il, ry, max(0.0, rl - il), ib)
                    top = (rx, ry + ib, rl, max(0.0, rb - ib))
                    for rect in (right, top):
                        if rect[2] > 0.05 and rect[3] > 0.05:
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
    return None


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


def optimize_load(goods_df: pd.DataFrame, truck: pd.Series) -> Dict[str, Any]:
    t_length, t_width, t_height = truck_dims_from_row(truck)
    max_weight = float(truck["max_weight_kg"])
    geom_vol = t_length * t_width * t_height
    max_volume = float(truck["max_volume_m3"]) if pd.notna(truck["max_volume_m3"]) else geom_vol

    items = build_items_for_loading(goods_df)
    placements: List[Dict[str, Any]] = []
    free_rects: List[Tuple[float, float, float, float]] = [(0.0, 0.0, t_length, t_width)]
    total_weight = 0.0
    total_volume = 0.0
    unplaced: List[Dict[str, Any]] = []

    for item in items:
        if not adr_good_allowed_on_truck(item, truck):
            unplaced.append(item)
            continue
        if not can_load_by_capacity(item, total_weight, total_volume, max_weight, max_volume):
            unplaced.append(item)
            continue
        placed = try_place_on_floor(item, free_rects, t_height)
        if placed is None:
            placed = try_stack(item, placements, t_height)
        if placed is None:
            unplaced.append(item)
            continue
        placed["stack_limit"] = item["max_stack"]
        placements.append(placed)
        total_weight += item["weight"]
        total_volume += item["volume"]

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
