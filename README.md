# Goods Optimization Script Walkthrough

`test.py` assigns goods to trucks using a **heuristic 3D loading** strategy. Input is **standardized pandas DataFrames** only: there is **no flexible parsing** and **no classes**—logic is split into plain functions that operate on `pd.DataFrame` / `pd.Series` and small dicts used only inside the packing loop.

The script:

- Reads dimensions and capacities from **fixed column names**
- Enforces **ADR (dangerous goods) rules**: ADR goods may only be loaded on **ADR-suitable** trucks that allow the good’s **ADR class**
- Packs goods per truck (floor rectangles first, then stacking with top-surface free-space tracking)
- Tries truck subsets to **minimize the number of trucks**, with a utilization tie-breaker
- Prints a human-readable loading plan and opens a 3D preview

---

## Dependencies

Install [pandas](https://pandas.pydata.org/) and at least one preview backend:

```bash
pip install pandas matplotlib
# optional for interactive hover preview:
pip install plotly
```

---

## 1) Standard column schemas

All upstream data must conform to these columns (types should be numeric where noted).

### Goods (`goods_df`)

| Column      | Meaning                          |
|------------|-----------------------------------|
| `id`       | Good identifier                   |
| `name`     | Description label                 |
| `weight_kg`| Mass in kg                        |
| `l`, `b`, `h` | Length, width, height (m)     |
| `max_stack`| Max stack height count per column |
| `adr` | `True` if the good is dangerous goods (ADR); must use an ADR-suitable truck |
| `adr_class` | ADR class code as string (e.g. `"3"`, `"4.1"`). Ignored when `adr` is false; if `adr` is true but empty, any ADR-suitable truck is allowed |

Volume for capacity checks is `l * b * h`. Sorting priority for loading uses `weight_kg`, then volume, then footprint `l * b`.

### Trucks (`trucks_df`)

| Column           | Meaning                                |
|-----------------|-----------------------------------------|
| `id`            | Truck identifier                        |
| `name`          | Display name (also used in printout)    |
| `l`, `b`, `h`   | Cargo space length, width, height (m)   |
| `max_weight_kg` | Payload limit (kg)                      |
| `max_volume_m3` | Volume limit (m³)                     |
| `adr_suitable` | `True` if the vehicle may carry ADR goods (equipment, certification, etc.) |
| `adr_classes_allowed` | Comma-separated class codes this truck may carry (e.g. `"3,4.1,8"`). Use `"*"` (or leave empty when combined with `adr_suitable=True` in code: empty is treated as **any** class). Non-ADR trucks typically use `False` for `adr_suitable` and an empty or unused `adr_classes_allowed`. |

Truck interior size comes directly from `l`, `b`, `h`. There is no inference from alternate field names.

### ADR matching (optimization constraint)

- Goods with `adr=False` may use **any** truck.
- Goods with `adr=True` may only be assigned to trucks with `adr_suitable=True`, and their `adr_class` must appear in the truck’s allowed list **unless** the list is empty/`"*"`, meaning any class is allowed on that ADR truck.
- Inside `optimize_load`, items that fail the ADR check are **not** placed on that truck (they remain for later trucks in `allocate_with_subset`).
- Truck order within a subset prefers **`adr_suitable=True` first** so ADR cargo is more likely to find a legal truck before non-ADR-only vehicles consume capacity.

---

## 2) Sample data

Default samples are loaded from CSV next to `test.py`:

- `goods_sample.csv` — goods columns as in the table above (`adr` as `true` / `false`; empty `adr_class` when not ADR).
- `trucks_sample.csv` — truck columns (`adr_classes_allowed` e.g. `*`).

`GOODS_SAMPLE` and `TRUCKS_SAMPLE` are these tables read at import time. The entry point passes copies into `run_allocation`:

```python
if __name__ == "__main__":
    run_allocation(GOODS_SAMPLE.copy(), TRUCKS_SAMPLE.copy())
```

To use your own data, edit the CSV files or build DataFrames with the same columns and call `run_allocation(goods_df, trucks_df)`.

---

## 3) Imports and building load items

### Imports

```python
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
```

- `combinations`: enumerates truck subsets of size `k` (1 truck, then 2, …).
- `pandas`: tabular input and row operations when removing loaded goods.

### `build_items_for_loading(goods_df)`

- Resets the index so row positions are `0 .. n-1`.
- Builds one **dict per row** with `idx` = that position (used when dropping loaded rows after each truck).
- Adds `volume` and `footprint` for sorting and capacity.
- **Sorts** items by `(weight, volume, footprint)` descending so heavier, bulkier goods are tried first.

There is no `Good` class; the packing loop uses these dicts only.

---

## 4) Truck geometry and `optimize_load`

### `truck_dims_from_row(truck)`

Returns `(l, b, h)` from the truck `pd.Series`—straight column reads, no defaults or aliases.

### `optimize_load(goods_df, truck)`

Loads as many goods as possible into **one** truck:

1. Reads `t_length`, `t_width`, `t_height` from the truck row.
2. Sets `max_weight` from `max_weight_kg`. Sets `max_volume` from `max_volume_m3` when that value is not NaN; otherwise uses geometric volume `l * b * h`.
3. Calls `build_items_for_loading(goods_df)` to get sorted item dicts.
4. Maintains `free_rects`: one initial rectangle `(0, 0, t_length, t_width)` for the floor.
5. For each item: **`adr_good_allowed_on_truck`** (skip ADR-incompatible pairs) → `can_load_by_capacity` → `try_place_on_floor` → else `try_stack`; on failure, item goes to `unplaced`.
6. Returns a **plan dict** via `build_truck_plan` (placements sorted by `(x, y, z)`).

### `try_place_on_floor`

- Scans free rectangles; tries orientations `(l, b)` and `(b, l)`.
- On fit: guillotine split into `right` and `top` leftover rectangles.
- Runs `merge_free_rectangles(...)` to combine adjacent free slots (e.g. `0.7x2.0` + `0.7x0.48` -> `0.7x2.48`) so fragmentation is reduced.

### `try_stack`

- Considers existing placements as bases (heavier / larger footprint first).
- Respects `stack_limit` and checks total height vs truck `h`.
- Tracks **free rectangles on each base top surface** (`top_free_rects`) so multiple items can be stacked side-by-side on the same base **without overlap**.
- Uses the same guillotine split idea on the top surface after each stacked placement.
- Also merges adjacent top-surface rectangles with `merge_free_rectangles(...)` so larger stacked items can use combined contiguous space.

---

## 5) Multi-truck allocation

### `allocate_with_subset(goods_df, truck_subset)`

1. Copies `goods_df` to `remaining` with a fresh index.
2. Sorts `truck_subset` with **`adr_suitable` descending** (ADR-capable trucks first), then by `max_weight_kg`, `max_volume_m3`, and `l` descending.
3. For each truck: `optimize_load(remaining, truck)`, then drops rows whose `idx` appears in `plan["placements"]`.
4. Returns `(plans, remaining)` where `remaining` is a DataFrame of goods not yet assigned.

### `select_best_plans(goods_df, trucks_df)`

1. For `k = 1, 2, …, n_trucks`, tries every subset of `k` trucks via `combinations`.
2. Keeps subsets where `remaining` is **empty** (all goods assigned).
3. Among those, picks the subset with highest **utilization score**: sum of `weight_util_pct + volume_util_pct` over plans.
4. If no subset can load everything, falls back to **all trucks** in order (`allocate_with_subset` on the full `trucks_df`).

### `run_allocation(goods_df, trucks_df, show_3d_preview=False)`

- Calls `select_best_plans`, then `print_allocation_summary`.
- If `show_3d_preview=True`, calls `preview_3d.show_load_preview(...)`.
- Returns `(plans, remaining)` so other scripts can reuse the allocation result.

---

## 6) Entry point and how to run

```bash
python test.py
```

You get per-truck dimensions, utilization, placement lines, and either a “still unassigned” list (from the remaining DataFrame) or confirmation that all goods were assigned.

By default (`if __name__ == "__main__"`), `test.py` also opens the 3D preview.

You can still run preview directly:

```bash
python preview_3d.py
python preview_3d.py --save loads_3d.html
python preview_3d.py --save loads_3d.png
```

Current preview behavior:
- **One truck per figure/page** (easier to inspect than subplot grids).
- Plotly mode supports **hover** (item ID, name, dimensions, position, weight, level).
- Truck cargo hull is drawn from truck `l x b x h` and shown as a semi-transparent box.

---

## 7) Notes and limitations

- **Heuristic only**—not guaranteed globally optimal packing.
- Floor and top surfaces use guillotine splitting plus rectangle merging; this reduces, but does not eliminate, fragmentation artifacts.
- Stacking checks footprint and height vs truck; it does **not** model axle load, CoG, fragility, or orientations beyond floor rotation.
- **Weight** and **volume** are hard limits per truck.
- **ADR** here is a **structural filter** (right truck type / class whitelist). It does **not** replace legal ADR processes (documentation, driver qualification, segregation tables, tunnel codes, etc.).
