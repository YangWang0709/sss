#!/usr/bin/env python3
"""Offline USD/USDA top-down footprint preview for human rollout review.

This script intentionally does not launch Isaac Sim.  It parses the staged
USDA scene text, extracts Xform transforms, Cube footprints, and referenced
mesh placements, then renders a world-coordinate 2D map.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/stage4a_usd_topdown_preview"
DEFAULT_ROLLOUT_MANIFEST = WORKSPACE / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime/short_rollout_manifest.jsonl"


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class PrimNode:
    name: str
    prim_type: str
    path: str
    parent: "PrimNode | None" = None
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    orient: tuple[float, float, float, float] | None = None
    reference: str | None = None
    has_cube_child: bool = False
    children: list["PrimNode"] = field(default_factory=list)
    line_no: int = 0


def _parse_tuple(text: str, count: int) -> tuple[float, ...] | None:
    values = [float(v) for v in re.findall(FLOAT_RE, text)]
    if len(values) < count:
        return None
    return tuple(values[:count])


def _quat_yaw(q: tuple[float, float, float, float] | None) -> float:
    if q is None:
        return 0.0
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y


def _prim_decl(line: str) -> tuple[str, str] | None:
    match = re.match(r'\s*(def|over|class)\s+(?:(\w+)\s+)?"([^"]+)"', line)
    if not match:
        return None
    prim_type = match.group(2) or match.group(1)
    return prim_type, match.group(3)


def parse_usda_xforms(path: Path) -> list[PrimNode]:
    """Parse enough USDA syntax for the staged scene's Xform footprints."""

    roots: list[PrimNode] = []
    stack: list[PrimNode] = []
    pending: PrimNode | None = None
    for line_no, raw in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        decl = _prim_decl(raw)
        if decl:
            prim_type, name = decl
            parent = stack[-1] if stack else None
            prim_path = f"{parent.path}/{name}" if parent else f"/{name}"
            pending = PrimNode(name=name, prim_type=prim_type, path=prim_path, parent=parent, line_no=line_no)
            if prim_type == "Cube" and parent is not None:
                parent.has_cube_child = True

        ref_match = re.search(r"prepend references\s*=\s*@([^@]+)@", raw)
        if ref_match:
            target = pending if pending is not None else (stack[-1] if stack else None)
            if target is not None:
                target.reference = ref_match.group(1)

        translate_match = re.search(r"xformOp:translate\s*=\s*\(([^)]*)\)", raw)
        if translate_match and stack:
            parsed = _parse_tuple(translate_match.group(1), 3)
            if parsed:
                stack[-1].translate = parsed  # type: ignore[assignment]

        scale_match = re.search(r"xformOp:scale\s*=\s*\(([^)]*)\)", raw)
        if scale_match and stack:
            parsed = _parse_tuple(scale_match.group(1), 3)
            if parsed:
                stack[-1].scale = parsed  # type: ignore[assignment]

        orient_match = re.search(r"xformOp:orient\s*=\s*\(([^)]*)\)", raw)
        if orient_match and stack:
            parsed = _parse_tuple(orient_match.group(1), 4)
            if parsed:
                stack[-1].orient = parsed  # type: ignore[assignment]

        open_count = raw.count("{")
        close_count = raw.count("}")
        if open_count and pending is not None:
            if pending.parent is None:
                roots.append(pending)
            else:
                pending.parent.children.append(pending)
            stack.append(pending)
            pending = None
            open_count -= 1

        for _ in range(close_count):
            if stack:
                stack.pop()

    nodes: list[PrimNode] = []

    def visit(node: PrimNode) -> None:
        nodes.append(node)
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return nodes


def world_transform(node: PrimNode) -> dict[str, Any]:
    chain: list[PrimNode] = []
    cur: PrimNode | None = node
    while cur is not None:
        chain.append(cur)
        cur = cur.parent
    chain.reverse()
    x = y = z = 0.0
    sx = sy = sz = 1.0
    yaw = 0.0
    for item in chain:
        tx, ty, tz = item.translate
        rx, ry = _rotate_xy(tx * sx, ty * sy, yaw)
        x += rx
        y += ry
        z += tz * sz
        yaw += _quat_yaw(item.orient)
        lsx, lsy, lsz = item.scale
        sx *= lsx
        sy *= lsy
        sz *= lsz
    return {"center": (x, y, z), "scale": (abs(sx), abs(sy), abs(sz)), "yaw": yaw}


def proxy_dims_for_reference(name: str, reference: str | None, scale: tuple[float, float, float]) -> tuple[float, float, float]:
    label = f"{name} {reference or ''}".lower()
    sxy = max(0.20, 0.5 * (abs(scale[0]) + abs(scale[1])))
    sz = max(0.20, abs(scale[2]))
    if "bed" in label:
        base = (2.2, 1.35, 0.9)
    elif "sofa" in label or "recliner" in label:
        base = (2.1, 0.9, 0.9)
    elif "desk" in label or "tablework" in label:
        base = (1.6, 0.8, 0.9)
    elif "dining_table" in label or "tableb" in label or "tablea" in label:
        base = (1.6, 1.0, 0.8)
    elif "chair" in label or "armchair" in label:
        base = (0.75, 0.75, 0.9)
    elif "cabinet" in label or "rack" in label or "shelf" in label or "book" in label:
        base = (1.35, 0.55, 1.4)
    elif "fridge" in label:
        base = (0.85, 0.85, 1.8)
    elif "bathtub" in label:
        base = (1.4, 0.8, 0.8)
    elif "toilet" in label or "washbasin" in label or "sink" in label:
        base = (0.7, 0.7, 0.9)
    elif "monitor" in label or "pc" in label or "printer" in label:
        base = (0.55, 0.35, 0.45)
    elif "plant" in label or "lamp" in label or "trash" in label or "extinguisher" in label:
        base = (0.45, 0.45, 1.0)
    elif "barrel" in label:
        base = (0.65, 0.65, 1.0)
    elif "crate" in label or "cardbox" in label:
        base = (0.65, 0.65, 0.65)
    elif "pallet" in label:
        base = (1.1, 0.9, 0.2)
    elif "pushcart" in label:
        base = (1.1, 0.7, 1.0)
    else:
        base = (0.85, 0.85, 0.85)
    return base[0] * sxy, base[1] * sxy, base[2] * sz


def oriented_rect(center: tuple[float, float], size: tuple[float, float], yaw: float) -> list[tuple[float, float]]:
    cx, cy = center
    hx, hy = 0.5 * size[0], 0.5 * size[1]
    pts = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [(cx + _rotate_xy(px, py, yaw)[0], cy + _rotate_xy(px, py, yaw)[1]) for px, py in pts]


def classify_node(node: PrimNode, center: tuple[float, float, float], scale: tuple[float, float, float]) -> str | None:
    path = node.path
    name = node.name.lower()
    if not path.startswith("/World/Building"):
        return None
    if "/ShowcaseAssets" in path or "/A1" in path or "sunlight" in name:
        return None
    if center[2] > 3.2 and "stair" not in name:
        return None
    if "roof" in name:
        return None
    if (
        "foundation" in name
        or "entrance_apron" in name
        or "slab_floor" in name
        or "infill_floor" in name
        or name.endswith("_floor")
    ):
        return "floor"
    if (
        "wall" in name
        or "exterior" in name
        or "stair_core" in name
        or "elevator_main_back" in name
        or "elevator_main_left" in name
        or "elevator_main_right" in name
    ):
        return "wall"
    if node.reference:
        return "furniture"
    if "door" in path.lower() and node.has_cube_child:
        return "door"
    if node.has_cube_child and max(scale[0], scale[1]) > 0.05 and scale[2] > 0.05:
        return "structural_obstacle"
    return None


def extract_footprints(nodes: list[PrimNode]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        if node.prim_type != "Xform":
            continue
        wt = world_transform(node)
        center = wt["center"]
        scale = wt["scale"]
        category = classify_node(node, center, scale)
        if category is None:
            continue
        if node.reference:
            sx, sy, sz = proxy_dims_for_reference(node.name, node.reference, scale)
        else:
            sx, sy, sz = scale
        if sx <= 0.03 or sy <= 0.03:
            continue
        polygon = oriented_rect((center[0], center[1]), (sx, sy), wt["yaw"])
        out.append(
            {
                "name": node.name,
                "path": node.path,
                "category": category,
                "center_xyz": [float(center[0]), float(center[1]), float(center[2])],
                "size_xyz": [float(sx), float(sy), float(sz)],
                "yaw_rad": float(wt["yaw"]),
                "reference": node.reference,
                "polygon_xy": [[float(x), float(y)] for x, y in polygon],
                "source_line": node.line_no,
                "source": "cube_scale" if not node.reference else "reference_proxy_footprint",
            }
        )
    return out


def render_map(footprints: list[dict[str, Any]], output_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon
    from matplotlib.ticker import MultipleLocator

    xs = [p[0] for fp in footprints for p in fp["polygon_xy"]]
    ys = [p[1] for fp in footprints for p in fp["polygon_xy"]]
    if not xs or not ys:
        raise RuntimeError("No footprints extracted from USD.")
    xmin, xmax = min(xs) - 1.0, max(xs) + 1.0
    ymin, ymax = min(ys) - 1.0, max(ys) + 1.0
    width = max(8.0, min(16.0, (xmax - xmin) * 0.38))
    height = max(8.0, min(22.0, (ymax - ymin) * 0.38))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    styles = {
        "floor": {"face": "#f1efe3", "edge": "#c8c3aa", "alpha": 0.92, "lw": 0.35, "z": 1},
        "wall": {"face": "#1f2937", "edge": "#0b1220", "alpha": 0.96, "lw": 0.25, "z": 6},
        "furniture": {"face": "#d97706", "edge": "#7c2d12", "alpha": 0.72, "lw": 0.35, "z": 4},
        "door": {"face": "#8b1e1e", "edge": "#450a0a", "alpha": 0.70, "lw": 0.25, "z": 5},
        "structural_obstacle": {"face": "#6b7280", "edge": "#27272a", "alpha": 0.76, "lw": 0.25, "z": 3},
    }
    for category in ("floor", "structural_obstacle", "furniture", "door", "wall"):
        for fp in footprints:
            if fp["category"] != category:
                continue
            style = styles[category]
            ax.add_patch(
                Polygon(
                    fp["polygon_xy"],
                    closed=True,
                    facecolor=style["face"],
                    edgecolor=style["edge"],
                    alpha=style["alpha"],
                    linewidth=style["lw"],
                    zorder=style["z"],
                )
            )

    label_candidates = [fp for fp in footprints if fp["category"] in {"wall", "furniture", "door"}]
    for fp in label_candidates:
        sx, sy, _ = fp["size_xyz"]
        if max(sx, sy) < 1.25:
            continue
        x, y, _ = fp["center_xyz"]
        label = fp["name"].replace("floor_0_", "").replace("_", " ")
        ax.text(x, y, label[:22], ha="center", va="center", fontsize=4.2, color="#111827", zorder=8)

    ax.axhline(0.0, color="#64748b", linewidth=0.7, alpha=0.55, zorder=2)
    ax.axvline(0.0, color="#64748b", linewidth=0.7, alpha=0.55, zorder=2)
    ax.scatter([0.0], [0.0], marker="+", s=70, c="#dc2626", linewidths=1.4, zorder=9)
    ax.text(0.15, 0.2, "world origin (0,0)", fontsize=6.2, color="#b91c1c", zorder=9)

    scalebar_x = xmin + 1.0
    scalebar_y = ymin + 0.8
    ax.plot([scalebar_x, scalebar_x + 5.0], [scalebar_y, scalebar_y], color="#111827", linewidth=2.0, zorder=10)
    ax.text(scalebar_x + 2.5, scalebar_y + 0.28, "5 m", ha="center", fontsize=7.0, color="#111827")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.yaxis.set_major_locator(MultipleLocator(2.0))
    ax.grid(True, color="#d1d5db", linewidth=0.35, alpha=0.85)
    ax.legend(
        handles=[
            Patch(facecolor=styles["floor"]["face"], edgecolor=styles["floor"]["edge"], label="floor / traversable slab"),
            Patch(facecolor=styles["wall"]["face"], edgecolor=styles["wall"]["edge"], label="wall"),
            Patch(facecolor=styles["furniture"]["face"], edgecolor=styles["furniture"]["edge"], label="furniture / obstacle proxy"),
            Patch(facecolor=styles["door"]["face"], edgecolor=styles["door"]["edge"], label="door panels"),
            Patch(facecolor=styles["structural_obstacle"]["face"], edgecolor=styles["structural_obstacle"]["edge"], label="structural obstacle"),
        ],
        loc="upper right",
        fontsize=7.0,
        frameon=True,
    )
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def point_in_poly(x: float, y: float, poly: list[list[float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def load_rollout_alignment_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not manifest_path.is_file():
        return records
    for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        pose_path = Path(item["pose"])
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        src = [float(v) for v in pose["position"]]
        dst = [float(v) for v in item["action_world_xyz"]]
        records.append(
            {
                "start_variant_id": int(item["start_variant_id"]),
                "step_id": int(item["step_id"]),
                "capture_index": int(item["capture_index"]),
                "source_pose_xyz": src,
                "source_yaw_rad": float(pose.get("yaw_rad", pose.get("yaw", 0.0))),
                "action_world_xyz": dst,
                "action_yaw_rad": float(item["action_yaw"]),
                "projected_source_xy": [src[0], src[1]],
                "projected_action_xy": [dst[0], dst[1]],
                "source_pose_file": str(pose_path),
                "rgb": item.get("rgb"),
                "depth": item.get("depth"),
                "observed_ratio_before": float(item.get("observed_ratio_before", 0.0)),
                "observed_ratio_after_current_capture": float(item.get("observed_ratio_after_current_capture", 0.0)),
            }
        )
    return records


def analyze_alignment(footprints: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    floors = [fp for fp in footprints if fp["category"] == "floor"]
    walls = [fp for fp in footprints if fp["category"] == "wall"]
    blocking = [fp for fp in footprints if fp["category"] in {"wall", "structural_obstacle"}]
    xs = [p[0] for fp in footprints for p in fp["polygon_xy"]]
    ys = [p[1] for fp in footprints for p in fp["polygon_xy"]]
    bounds = {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]} if xs and ys else {"x": [0.0, 0.0], "y": [0.0, 0.0]}

    def classify_point(x: float, y: float) -> dict[str, bool]:
        return {
            "inside_map_bounds": bounds["x"][0] <= x <= bounds["x"][1] and bounds["y"][0] <= y <= bounds["y"][1],
            "inside_floor_footprint": any(point_in_poly(x, y, fp["polygon_xy"]) for fp in floors),
            "inside_wall_footprint": any(point_in_poly(x, y, fp["polygon_xy"]) for fp in walls),
            "inside_blocking_footprint": any(point_in_poly(x, y, fp["polygon_xy"]) for fp in blocking),
        }

    point_checks = []
    for row in records:
        for kind, xyz_key in (("source_pose", "source_pose_xyz"), ("action_target", "action_world_xyz")):
            x, y, z = row[xyz_key]
            check = classify_point(x, y)
            check.update(
                {
                    "kind": kind,
                    "start_variant_id": row["start_variant_id"],
                    "step_id": row["step_id"],
                    "xyz": [x, y, z],
                    "projected_xy": [x, y],
                    "xy_roundtrip_error_m": 0.0,
                }
            )
            point_checks.append(check)

    def count_false(key: str) -> int:
        return sum(1 for item in point_checks if not item[key])

    def count_true(key: str) -> int:
        return sum(1 for item in point_checks if item[key])

    z_values = [float(v) for row in records for v in (row["source_pose_xyz"][2], row["action_world_xyz"][2])]
    return {
        "coordinate_contract": "2D projection is exactly (world_x, world_y) from 3D Isaac/USD coordinates; z is ignored only for top-down display.",
        "rollout_record_count": len(records),
        "projected_point_count": len(point_checks),
        "map_bounds_xy_m": bounds,
        "outside_map_bounds_count": count_false("inside_map_bounds"),
        "outside_floor_footprint_count": count_false("inside_floor_footprint"),
        "inside_wall_footprint_count": count_true("inside_wall_footprint"),
        "inside_blocking_footprint_count": count_true("inside_blocking_footprint"),
        "max_xy_roundtrip_error_m": max((item["xy_roundtrip_error_m"] for item in point_checks), default=0.0),
        "z_min_m": min(z_values) if z_values else None,
        "z_max_m": max(z_values) if z_values else None,
        "z_unique_m": sorted({round(z, 4) for z in z_values}),
        "point_checks": point_checks,
    }


def render_alignment_overlay(
    footprints: list[dict[str, Any]],
    records: list[dict[str, Any]],
    output_png: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon
    from matplotlib.ticker import MultipleLocator

    xs = [p[0] for fp in footprints for p in fp["polygon_xy"]]
    ys = [p[1] for fp in footprints for p in fp["polygon_xy"]]
    xmin, xmax = min(xs) - 1.0, max(xs) + 1.0
    ymin, ymax = min(ys) - 1.0, max(ys) + 1.0
    fig, ax = plt.subplots(figsize=(11.0, 17.0), constrained_layout=True)
    styles = {
        "floor": {"face": "#f1efe3", "edge": "#c8c3aa", "alpha": 0.92, "lw": 0.35, "z": 1},
        "wall": {"face": "#1f2937", "edge": "#0b1220", "alpha": 0.96, "lw": 0.25, "z": 6},
        "furniture": {"face": "#d97706", "edge": "#7c2d12", "alpha": 0.42, "lw": 0.25, "z": 4},
        "door": {"face": "#8b1e1e", "edge": "#450a0a", "alpha": 0.55, "lw": 0.25, "z": 5},
        "structural_obstacle": {"face": "#6b7280", "edge": "#27272a", "alpha": 0.62, "lw": 0.25, "z": 3},
    }
    for category in ("floor", "structural_obstacle", "furniture", "door", "wall"):
        for fp in footprints:
            if fp["category"] != category:
                continue
            style = styles[category]
            ax.add_patch(
                Polygon(
                    fp["polygon_xy"],
                    closed=True,
                    facecolor=style["face"],
                    edgecolor=style["edge"],
                    alpha=style["alpha"],
                    linewidth=style["lw"],
                    zorder=style["z"],
                )
            )

    colors = ["#2563eb", "#0891b2", "#16a34a", "#9333ea", "#db2777", "#ea580c", "#4f46e5", "#0f766e", "#65a30d", "#be123c"]
    by_start: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        by_start.setdefault(row["start_variant_id"], []).append(row)
    for start_id, rows in sorted(by_start.items()):
        rows = sorted(rows, key=lambda r: r["step_id"])
        color = colors[start_id % len(colors)]
        src_x = [r["source_pose_xyz"][0] for r in rows]
        src_y = [r["source_pose_xyz"][1] for r in rows]
        dst_x = [r["action_world_xyz"][0] for r in rows]
        dst_y = [r["action_world_xyz"][1] for r in rows]
        ax.plot(src_x, src_y, color=color, linewidth=1.25, alpha=0.78, zorder=10)
        ax.scatter(src_x, src_y, s=24, color=color, edgecolor="white", linewidth=0.5, zorder=11)
        ax.scatter(dst_x, dst_y, s=32, marker="x", color="#ec4899", linewidth=1.4, zorder=12)
        if rows:
            ax.text(src_x[0] + 0.08, src_y[0] + 0.08, f"S{start_id}", fontsize=7.0, color=color, weight="bold", zorder=13)
        for row in rows:
            sx, sy, _ = row["source_pose_xyz"]
            tx, ty, _ = row["action_world_xyz"]
            ax.annotate(
                "",
                xy=(tx, ty),
                xytext=(sx, sy),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8, "alpha": 0.7},
                zorder=10,
            )
            ax.text(tx + 0.04, ty + 0.04, f"{start_id}:{row['step_id']}", fontsize=4.8, color="#831843", zorder=13)

    ax.axhline(0.0, color="#64748b", linewidth=0.7, alpha=0.55, zorder=2)
    ax.axvline(0.0, color="#64748b", linewidth=0.7, alpha=0.55, zorder=2)
    ax.scatter([0.0], [0.0], marker="+", s=70, c="#dc2626", linewidths=1.4, zorder=14)
    ax.text(0.15, 0.2, "world origin (0,0)", fontsize=6.2, color="#b91c1c", zorder=14)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.yaxis.set_major_locator(MultipleLocator(2.0))
    ax.grid(True, color="#d1d5db", linewidth=0.35, alpha=0.85)
    ax.legend(
        handles=[
            Patch(facecolor=styles["wall"]["face"], edgecolor=styles["wall"]["edge"], label="USD walls"),
            Patch(facecolor=styles["furniture"]["face"], edgecolor=styles["furniture"]["edge"], label="USD furniture/obstacle proxy"),
            Line2D([0], [0], color="#2563eb", marker="o", label="3D source camera pose projected to 2D"),
            Line2D([0], [0], color="#ec4899", marker="x", linestyle="None", label="3D action target projected to 2D"),
        ],
        loc="upper right",
        fontsize=7.0,
        frameon=True,
    )
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def write_alignment_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "start_variant_id",
                "step_id",
                "capture_index",
                "source_x",
                "source_y",
                "source_z",
                "action_x",
                "action_y",
                "action_z",
                "source_yaw_rad",
                "action_yaw_rad",
                "source_pose_file",
            ],
        )
        writer.writeheader()
        for row in records:
            sx, sy, sz = row["source_pose_xyz"]
            ax, ay, az = row["action_world_xyz"]
            writer.writerow(
                {
                    "start_variant_id": row["start_variant_id"],
                    "step_id": row["step_id"],
                    "capture_index": row["capture_index"],
                    "source_x": sx,
                    "source_y": sy,
                    "source_z": sz,
                    "action_x": ax,
                    "action_y": ay,
                    "action_z": az,
                    "source_yaw_rad": row["source_yaw_rad"],
                    "action_yaw_rad": row["action_yaw_rad"],
                    "source_pose_file": row["source_pose_file"],
                }
            )


def write_html(path: Path, png_name: str, summary: dict[str, Any]) -> None:
    rows = "\n".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
        if not isinstance(v, (dict, list))
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>USD Topdown Preview</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #f8fafc; color: #111827; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 24px; margin: 0 0 10px; }}
    p {{ max-width: 900px; line-height: 1.5; }}
    img {{ width: 100%; height: auto; border: 1px solid #cbd5e1; background: white; }}
    table {{ border-collapse: collapse; margin: 18px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 7px 10px; text-align: left; }}
    th {{ background: #e5e7eb; }}
  </style>
</head>
<body>
  <main>
    <h1>USD Topdown Preview</h1>
    <p>Offline USDA footprint compression. World x/y coordinates are preserved in meters; this preview did not launch Isaac, training, rollout, map_predict, checkpoints, or RL.</p>
    <table>{rows}</table>
    <img src="{html.escape(png_name)}" alt="USD topdown 2D map" />
  </main>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rollout_manifest", type=Path, default=DEFAULT_ROLLOUT_MANIFEST)
    parser.add_argument("--with_rollout_overlay", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = parse_usda_xforms(args.usd)
    footprints = extract_footprints(nodes)
    counts: dict[str, int] = {}
    for fp in footprints:
        counts[fp["category"]] = counts.get(fp["category"], 0) + 1

    png_path = output_dir / "home_like_scene_v1_usd_topdown_floor0_preview.png"
    json_path = output_dir / "home_like_scene_v1_usd_topdown_footprints.json"
    summary_path = output_dir / "home_like_scene_v1_usd_topdown_summary.json"
    html_path = output_dir / "home_like_scene_v1_usd_topdown_preview.html"
    overlay_png_path = output_dir / "home_like_scene_v1_usd_topdown_stage4a714_alignment_overlay.png"
    alignment_json_path = output_dir / "home_like_scene_v1_usd_topdown_stage4a714_alignment_audit.json"
    alignment_csv_path = output_dir / "home_like_scene_v1_usd_topdown_stage4a714_alignment_points.csv"

    render_map(footprints, png_path, "home_like_scene_v1 USD top-down footprint preview (world x/y meters)")
    summary = {
        "stage": "USD topdown preview",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "usd_path": str(args.usd),
        "output_dir": str(output_dir),
        "total_xform_nodes_parsed": len([n for n in nodes if n.prim_type == "Xform"]),
        "total_footprints_drawn": len(footprints),
        "footprint_counts": counts,
        "main_png": str(png_path),
        "main_html": str(html_path),
        "coordinate_contract": "2D x/y axes are Isaac/USD world x/y meters; no axis remapping or normalization.",
        "runtime_scope": "offline_parse_only_no_isaac_no_rollout_no_training_no_checkpoint_no_rl",
        "reference_mesh_note": "Referenced furniture assets use heuristic footprint proxies from their USD placement and asset name because pxr/USD SDK is unavailable in this environment.",
    }
    json_path.write_text(json.dumps({"summary": summary, "footprints": footprints}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_html(html_path, png_path.name, summary)
    if args.with_rollout_overlay:
        records = load_rollout_alignment_records(args.rollout_manifest)
        alignment = analyze_alignment(footprints, records)
        render_alignment_overlay(
            footprints,
            records,
            overlay_png_path,
            "Stage 4A-7.14 3D world poses projected onto USD 2D top-down map",
        )
        write_alignment_csv(alignment_csv_path, records)
        alignment_json_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "stage": "USD topdown 3D-to-2D alignment audit",
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "usd_path": str(args.usd),
                        "rollout_manifest": str(args.rollout_manifest),
                        "overlay_png": str(overlay_png_path),
                        "alignment_csv": str(alignment_csv_path),
                        "alignment_json": str(alignment_json_path),
                        "runtime_scope": "offline_parse_only_no_isaac_no_rollout_no_training_no_checkpoint_no_rl",
                    },
                    "alignment": alignment,
                    "records": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        summary["alignment_overlay_png"] = str(overlay_png_path)
        summary["alignment_audit_json"] = str(alignment_json_path)
        summary["alignment_points_csv"] = str(alignment_csv_path)
        summary["alignment_rollout_record_count"] = len(records)
        summary["alignment_max_xy_roundtrip_error_m"] = alignment["max_xy_roundtrip_error_m"]
        summary["alignment_inside_wall_footprint_count"] = alignment["inside_wall_footprint_count"]
        summary["alignment_inside_blocking_footprint_count"] = alignment["inside_blocking_footprint_count"]
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
