"""
3D preview of load plans produced by test.py.

Interactive (default when showing): **Plotly** — hover shows ID, name, dimensions, weight, level.
  pip install plotly

Static PNG: **matplotlib**
  pip install matplotlib

Run from project root:
  python preview_3d.py
  python preview_3d.py --save loads_3d.png
  python preview_3d.py --save loads_3d.html   # interactive file, hover works in browser
"""

from __future__ import annotations

import argparse
import html
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from test import _as_bool

# Plotly unit-cube topology (vertices 0..1 on each axis); scaled per box.
_MESH_I = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
_MESH_J = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
_MESH_K = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
_UX = [0, 1, 1, 0, 0, 1, 1, 0]
_UY = [0, 0, 1, 1, 0, 0, 1, 1]
_UZ = [0, 0, 0, 0, 1, 1, 1, 1]


def _mesh3d_box_vertices(x0: float, y0: float, z0: float, l: float, b: float, h: float) -> Tuple[List[float], List[float], List[float]]:
    x = [x0 + ux * l for ux in _UX]
    y = [y0 + uy * b for uy in _UY]
    z = [z0 + uz * h for uz in _UZ]
    return x, y, z


def _placement_hover_lines(p: Dict[str, Any], adr_by_id: Dict[int, bool]) -> str:
    pid = int(p["ID"])
    name = html.escape(str(p["name"]))
    adr = adr_by_id.get(pid, False)
    adr_line = "<br><b>ADR</b> yes" if adr else ""
    return (
        f"<b>ID {pid}</b><br>{name}{adr_line}<br>"
        f"<b>Dimensions (l×b×h)</b><br>"
        f"{p['l']:.3f} × {p['b']:.3f} × {p['h']:.3f} m<br>"
        # f"<b>Position (x,y,z)</b><br>"
        # f"{p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f} m<br>"
        f"<b>Weight</b> {p['weight']:.1f} kg<br>"
        # f"<b>Stack level</b> {p['level']}"
        "<extra></extra>"
    )


def _plotly_color_for_placement(
    i: int, p: Dict[str, Any], adr_by_id: Dict[int, bool], cmap_name: str = "tab10"
) -> Tuple[str, float]:
    """Returns (plotly color string, opacity)."""
    # is_adr = adr_by_id.get(int(p["ID"]), False)
    # if is_adr:
    #     return "rgb(220,90,80)", 1
    cmap = plt.get_cmap(cmap_name)
    base = cmap(i % cmap.N)
    r, g, b = int(base[0] * 255), int(base[1] * 255), int(base[2] * 255)
    return f"rgb({r},{g},{b})", 1


def _plotly_truck_wireframe_trace(l: float, b: float, h: float) -> Any:
    import plotly.graph_objects as go

    c = np.array(
        [
            [0, 0, 0],
            [l, 0, 0],
            [l, b, 0],
            [0, b, 0],
            [0, 0, h],
            [l, 0, h],
            [l, b, h],
            [0, b, h],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    for i, j in edges:
        xs.extend([c[i][0], c[j][0], None])
        ys.extend([c[i][1], c[j][1], None])
        zs.extend([c[i][2], c[j][2], None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="rgba(40,50,70,0.95)", width=4),
        hoverinfo="skip",
        showlegend=False,
    )


def _plotly_truck_cargo_hull_trace(l: float, b: float, h: float, truck_name: str) -> Any:
    """Semi-transparent cargo volume at origin, sized exactly to truck l × b × h (m)."""
    import plotly.graph_objects as go

    vx, vy, vz = _mesh3d_box_vertices(0.0, 0.0, 0.0, l, b, h)
    vol = l * b * h
    safe_name = html.escape(str(truck_name))
    hover = (
        f"<b>Truck cargo space</b><br>{safe_name}<br>"
        f"<b>l × b × h</b> {l:.3f} × {b:.3f} × {h:.3f} m<br>"
        f"<b>Geometric volume</b> {vol:.2f} m³"
        "<extra></extra>"
    )
    return go.Mesh3d(
        x=vx,
        y=vy,
        # z=vz,
        i=_MESH_I,
        j=_MESH_J,
        k=_MESH_K,
        color="rgb(160, 175, 205)",
        opacity=0.5,
        # hovertemplate=hover,
        name="Cargo space",
        showlegend=False,
        lighting=dict(ambient=0.55, diffuse=0.5),
        flatshading=True,
    )


def build_plotly_figure(
    plans: List[Dict[str, Any]],
    adr_by_id: Dict[int, bool],
) -> Any:
    """Build interactive Plotly figure with hover on each good."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(plans)
    cols = min(2, max(n, 1))
    rows = (n + cols - 1) // cols
    titles_list = []
    for p in plans:
        d = p["truck_dims"]
        tl, tb, th = d["l"], d["b"], d["h"]
        truck_volume = float(tl) * float(tb) * float(th)
        placed_volume = sum(
            float(item["l"]) * float(item["b"]) * float(item["h"])
            for item in p["placements"]
        )
        volume_util_pct = (placed_volume / truck_volume * 100.0) if truck_volume > 0 else 0.0
        titles_list.append(
            f"{p['truck']} {tl:g}m × {tb:g}m × {th:g}m <br>"
            f"{p['placed_count']} goods, {p['weight_util_pct']:.0f}% wt, "
            f"{placed_volume:.1f}/{truck_volume:.1f} m³ ({volume_util_pct:.0f}% vol)"
        )
    while len(titles_list) < rows * cols:
        titles_list.append("")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=tuple(titles_list),
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
    )
    # Keep per-subplot titles visually separated from the main figure title.
    fig.update_annotations(font=dict(size=20), yshift=-5)

    for idx, plan in enumerate(plans):
        r = idx // cols + 1
        c = idx % cols + 1
        dims = plan["truck_dims"]
        tl, tb, th = dims["l"], dims["b"], dims["h"]
        fig.add_trace(
            _plotly_truck_cargo_hull_trace(tl, tb, th, plan["truck"]),
            row=r,
            col=c,
        )
        fig.add_trace(_plotly_truck_wireframe_trace(tl, tb, th), row=r, col=c)

        for i, p in enumerate(plan["placements"]):
            vx, vy, vz = _mesh3d_box_vertices(p["x"], p["y"], p["z"], p["l"], p["b"], p["h"])
            color, opacity = _plotly_color_for_placement(i, p, adr_by_id)
            fig.add_trace(
                go.Mesh3d(
                    x=vx,
                    y=vy,
                    z=vz,
                    i=_MESH_I,
                    j=_MESH_J,
                    k=_MESH_K,
                    color=color,
                    opacity=opacity,
                    hovertemplate=_placement_hover_lines(p, adr_by_id),
                    name=f"ID {p['ID']}",
                    showlegend=False,
                    lighting=dict(ambient=0.45, diffuse=0.85),
                    flatshading=True,
                ),
                row=r,
                col=c,
            )

    fig.update_layout(
        title=dict(
            text=(
                "Load preview — shaded box = truck cargo (l×b×h from truck row)"
                "<br><sup>Hover cargo or goods</sup>"
            ),
            x=0.5,
            xanchor="center",
            y=0.985,
            yanchor="top",
        ),
        margin=dict(l=0, r=0, t=130, b=0),
        height=1080 * rows,
    )

    # Equal-ish axes per 3D subplot (first is "scene", then "scene2", "scene3", …)
    scene_layout: Dict[str, Any] = {}
    zoom = 5
    for idx, plan in enumerate(plans):
        dims = plan["truck_dims"]
        tl, tb, th = dims["l"], dims["b"], dims["h"]
        sid = "scene" if idx == 0 else f"scene{idx + 1}"
        scene_layout[sid] = dict(
            xaxis=dict(title="l (m)", range=[0, tl], backgroundcolor="rgb(248,248,248)"),
            yaxis=dict(title="b (m)", range=[0, tb], backgroundcolor="rgb(248,248,248)"),
            zaxis=dict(title="h (m)", range=[0, th], backgroundcolor="rgb(248,248,248)"),
            aspectmode="manual",
            aspectratio=dict(x=float(tl), y=float(tb), z=float(th)),
            camera=dict(eye=dict(x=1.35*zoom, y=-1.55*zoom, z=0.85*zoom)),
        )
    fig.update_layout(**scene_layout)

    return fig


def _show_plotly_interactive(fig: Any) -> None:
    fig.show()


def _save_plotly_html(fig: Any, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Saved interactive HTML (open in browser for hover): {path}")


def _box_faces(
    x: float, y: float, z: float, l: float, b: float, h: float
) -> List[List[Tuple[float, float, float]]]:
    pts = np.array(
        [
            [x, y, z],
            [x + l, y, z],
            [x + l, y + b, z],
            [x, y + b, z],
            [x, y, z + h],
            [x + l, y, z + h],
            [x + l, y + b, z + h],
            [x, y + b, z + h],
        ]
    )
    return [
        [pts[j] for j in [0, 1, 2, 3]],
        [pts[j] for j in [4, 5, 6, 7]],
        [pts[j] for j in [0, 1, 5, 4]],
        [pts[j] for j in [2, 3, 7, 6]],
        [pts[j] for j in [1, 2, 6, 5]],
        [pts[j] for j in [0, 3, 7, 4]],
    ]


def _wireframe_edges(
    x: float, y: float, z: float, l: float, b: float, h: float
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    c = np.array(
        [
            [x, y, z],
            [x + l, y, z],
            [x + l, y + b, z],
            [x, y + b, z],
            [x, y, z + h],
            [x + l, y, z + h],
            [x + l, y + b, z + h],
            [x, y + b, z + h],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    lines = []
    for i, j in edges:
        p0, p1 = c[i], c[j]
        lines.append(([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]]))
    return lines


def plot_truck_cargo_box(ax: Any, dims: Dict[str, float]) -> None:
    """Filled semi-transparent cargo volume from truck dimensions (l, b, h in m)."""
    l, b, h = float(dims["l"]), float(dims["b"]), float(dims["h"])
    faces = _box_faces(0.0, 0.0, 0.0, l, b, h)
    poly = Poly3DCollection(
        faces,
        facecolors=(0.72, 0.76, 0.86, 0.2),
        edgecolors=(0.35, 0.4, 0.5, 0.95),
        linewidths=1.0,
    )
    ax.add_collection3d(poly)


def plot_truck_wireframe(
    ax: Any, dims: Dict[str, float], color: str = "#2c3e50", linewidth: float = 1.5
) -> None:
    l, b, h = dims["l"], dims["b"], dims["h"]
    for xs, ys, zs in _wireframe_edges(0.0, 0.0, 0.0, l, b, h):
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, linestyle="-", alpha=0.9)


def plot_placement_boxes(
    ax: Any,
    placements: List[Dict[str, Any]],
    adr_by_id: Dict[int, bool],
    cmap_name: str = "tab10",
) -> None:
    cmap = plt.get_cmap(cmap_name)
    for i, p in enumerate(placements):
        is_adr = adr_by_id.get(int(p["ID"]), False)
        base = cmap(i % cmap.N)
        if is_adr:
            face = (0.95, 0.4, 0.35, 0.55)
            edge_color = (0.85, 0.15, 0.12, 1.0)
        else:
            face = (*base[:3], 0.38)
            edge_color = (*base[:3], 0.9)
        faces = _box_faces(p["x"], p["y"], p["z"], p["l"], p["b"], p["h"])
        poly = Poly3DCollection(
            faces,
            facecolors=face,
            edgecolors=edge_color,
            linewidths=0.6,
        )
        ax.add_collection3d(poly)


def _set_physical_aspect_3d(ax: Any, l: float, b: float, h: float) -> None:
    ax.set_xlim(0, l)
    ax.set_ylim(0, b)
    ax.set_zlim(0, h)
    try:
        ax.set_box_aspect((float(l), float(b), float(h)))
    except AttributeError:
        pass


def plot_plan_3d(ax: Any, plan: Dict[str, Any], adr_by_id: Dict[int, bool]) -> None:
    dims = plan["truck_dims"]
    l, b, h = float(dims["l"]), float(dims["b"]), float(dims["h"])
    plot_truck_cargo_box(ax, dims)
    plot_truck_wireframe(ax, dims)
    plot_placement_boxes(ax, plan["placements"], adr_by_id)

    ax.set_xlabel("l (m)")
    ax.set_ylabel("b (m)")
    ax.set_zlabel("h (m)")
    ax.set_title(
        f"{plan['truck']}\n"
        f"Cargo {l:g}×{b:g}×{h:g} m · {plan['placed_count']} placed | "
        f"{plan['weight_util_pct']:.1f}% wt | {plan['volume_util_pct']:.1f}% vol"
    )
    _set_physical_aspect_3d(ax, l, b, h)
    ax.view_init(elev=22, azim=-60)


def build_adr_lookup(goods_df: pd.DataFrame) -> Dict[int, bool]:
    out: Dict[int, bool] = {}
    for _, row in goods_df.iterrows():
        out[int(row["id"])] = _as_bool(row.get("adr"), False)
    return out


def show_load_preview(
    goods_df: pd.DataFrame,
    plans: List[Dict[str, Any]],
    remaining: Optional[pd.DataFrame] = None,
    *,
    save_path: Optional[str] = None,
    dpi: int = 120,
    use_matplotlib: bool = False,
) -> None:
    """
    Interactive view: Plotly (hover = ID, name, l×b×h, position, weight, level).
    PNG export: matplotlib. HTML export: Plotly (interactive in browser).

    Set use_matplotlib=True to force the static matplotlib window (no hover).
    """
    adr_by_id = build_adr_lookup(goods_df)
    save_lower = (save_path or "").lower()

    if save_path and save_lower.endswith(".html"):
        try:
            pfig = build_plotly_figure(plans, adr_by_id)
            _save_plotly_html(pfig, save_path)
        except ImportError:
            print("Saving .html requires plotly: pip install plotly")
        if remaining is not None and not remaining.empty:
            print(f"Note: {len(remaining)} good(s) still unassigned (not shown in 3D).")
        return

    if save_path:
        n = len(plans)
        cols = min(2, max(n, 1))
        rows = (n + cols - 1) // cols
        fig = plt.figure(figsize=(5.5 * cols, 4.8 * rows))
        for i, plan in enumerate(plans):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            plot_plan_3d(ax, plan, adr_by_id)
        fig.suptitle(
            "Load preview — truck cargo from l×b×h (shaded + outline); goods solid (red = ADR)",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
        if remaining is not None and not remaining.empty:
            print(f"Note: {len(remaining)} good(s) still unassigned (not shown in 3D).")
        return

    if not use_matplotlib:
        try:
            pfig = build_plotly_figure(plans, adr_by_id)
            print(
                "Opening interactive 3D view (Plotly). Hover a box for item details. "
                "Install plotly if this fails: pip install plotly"
            )
            _show_plotly_interactive(pfig)
            if remaining is not None and not remaining.empty:
                print(f"Note: {len(remaining)} good(s) still unassigned (not shown in 3D).")
            return
        except ImportError:
            print("Plotly not installed — using matplotlib (no hover). pip install plotly for hover.")

    n = len(plans)
    cols = min(2, max(n, 1))
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5.5 * cols, 4.8 * rows))
    for i, plan in enumerate(plans):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        plot_plan_3d(ax, plan, adr_by_id)
    fig.suptitle(
        "Load preview — truck cargo from l×b×h (shaded + outline); goods solid (red = ADR)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    plt.show()

    if remaining is not None and not remaining.empty:
        print(f"Note: {len(remaining)} good(s) still unassigned (not shown in 3D).")


def main() -> None:
    parser = argparse.ArgumentParser(description="3D preview of truck loads.")
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save figure to PNG (no display). Example: --save loads.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Raster resolution when using --save (default 120).",
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Use matplotlib window instead of Plotly (no hover).",
    )
    args = parser.parse_args()

    from test import GOODS_SAMPLE, TRUCKS_SAMPLE, select_best_plans

    goods_df = GOODS_SAMPLE.copy()
    trucks_df = TRUCKS_SAMPLE.copy()
    plans, remaining = select_best_plans(goods_df, trucks_df)
    show_load_preview(
        goods_df,
        plans,
        remaining,
        save_path=args.save,
        dpi=args.dpi,
        use_matplotlib=args.matplotlib,
    )


if __name__ == "__main__":
    main()
