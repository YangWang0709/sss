#!/usr/bin/env python3
"""Phase 8 primary-scene mapping smoke.

Bounded smoke only: opens the repaired primary USD, reuses the Phase 7 robot
proxy and USD-geometry depth/pointcloud proxy observation style, updates a
small BEV partial occupancy map over a short action sequence, and writes small
CSV/JSON/PNG/NPZ review artifacts. It does not train, run RL, checkpoint, run
map_predict, or run a primary long rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import traceback

UNKNOWN = 0
FREE = 1
OCCUPIED = 2
ROBOT = 3


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def vec3_tuple(v):
    return (float(v[0]), float(v[1]), float(v[2]))


def bbox_to_record(prim, bbox_cache):
    try:
        bound = bbox_cache.ComputeWorldBound(prim)
        rng = bound.ComputeAlignedBox()
        if rng.IsEmpty():
            return None
        mn = vec3_tuple(rng.GetMin())
        mx = vec3_tuple(rng.GetMax())
        if not all(finite(x) for x in (*mn, *mx)):
            return None
        size = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
        center = ((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0)
        if any(s < 0 for s in size):
            return None
        return {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName() or "",
            "min": mn,
            "max": mx,
            "size": size,
            "center": center,
        }
    except Exception:
        return None


def collect_geometry(stage):
    from pxr import UsdGeom  # type: ignore

    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    records = []
    type_counts = {}
    for prim in stage.Traverse():
        t = prim.GetTypeName() or ""
        type_counts[t] = type_counts.get(t, 0) + 1
        if t in {"Cube", "Mesh", "Cylinder", "Sphere", "Plane"}:
            rec = bbox_to_record(prim, bbox_cache)
            if rec:
                records.append(rec)
    return records, type_counts


def is_floor_like(rec):
    path = rec["path"].lower()
    sx, sy, sz = rec["size"]
    return (
        "floor" in path
        or "slab" in path
        or "foundation" in path
        or "ceiling" in path
        or (sx > 8.0 and sy > 8.0 and sz < 0.5)
    )


def is_obstacle_like(rec):
    sx, sy, sz = rec["size"]
    zmin, zmax = rec["min"][2], rec["max"][2]
    if is_floor_like(rec):
        return False
    if zmax < 0.15 or zmin > 2.2:
        return False
    if sx <= 0.02 or sy <= 0.02 or sz <= 0.02:
        return False
    # Skip huge enclosing artifacts; they are scene extents, not local obstacles.
    if sx > 24.0 or sy > 45.0:
        return False
    return True


def collision_xy(x, y, obstacles, radius=0.35):
    for rec in obstacles:
        mn, mx = rec["min"], rec["max"]
        if (mn[0] - radius) <= x <= (mx[0] + radius) and (mn[1] - radius) <= y <= (mx[1] + radius):
            return rec["path"]
    return ""


def clearance_xy(x, y, obstacles):
    best = float("inf")
    for rec in obstacles:
        mn, mx = rec["min"], rec["max"]
        dx = max(mn[0] - x, 0.0, x - mx[0])
        dy = max(mn[1] - y, 0.0, y - mx[1])
        best = min(best, math.hypot(dx, dy))
    return best if math.isfinite(best) else 999.0


def visible_records(pose, records, max_range=12.0, fov_deg=360.0):
    px, py, pz, yaw = pose
    out = []
    half = math.radians(fov_deg) / 2.0
    for rec in records:
        cx, cy, cz = rec["center"]
        dz = cz - pz
        dx, dy = cx - px, cy - py
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= 0.1 or dist > max_range:
            continue
        bearing = math.atan2(dy, dx)
        delta = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
        if abs(delta) <= half:
            out.append((dist, rec))
    out.sort(key=lambda item: item[0])
    return out


def choose_start(records, obstacles):
    floor_records = [r for r in records if is_floor_like(r) and r["size"][0] * r["size"][1] > 20.0]
    if floor_records:
        floor = max(floor_records, key=lambda r: r["size"][0] * r["size"][1])
        xmin, xmax = floor["min"][0] + 0.8, floor["max"][0] - 0.8
        ymin, ymax = floor["min"][1] + 0.8, floor["max"][1] - 0.8
        strategy = "largest_floor_visibility_grid"
    else:
        xs = [v for rec in records for v in (rec["min"][0], rec["max"][0]) if finite(v)]
        ys = [v for rec in records for v in (rec["min"][1], rec["max"][1]) if finite(v)]
        if not xs or not ys:
            return (0.0, 0.0, 0.35), {"strategy": "fallback_origin"}
        xmin, xmax = max(min(xs), -12.0), min(max(xs), 12.0)
        ymin, ymax = max(min(ys), -12.0), min(max(ys), 12.0)
        strategy = "clipped_visibility_grid"

    cx0, cy0 = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    candidates = []
    step = 0.75
    x = math.ceil(xmin / step) * step
    while x <= xmax:
        y = math.ceil(ymin / step) * step
        while y <= ymax:
            if not collision_xy(x, y, obstacles):
                clearance = clearance_xy(x, y, obstacles)
                visible = visible_records((x, y, 1.2, 0.0), records, max_range=12.0, fov_deg=360.0)
                visible_count = len([item for item in visible if not is_floor_like(item[1])])
                if visible_count >= 3:
                    center_penalty = 0.015 * math.hypot(x - cx0, y - cy0)
                    score = min(clearance, 3.0) + 0.08 * visible_count - center_penalty
                    candidates.append((score, visible_count, clearance, x, y))
            y += step
        x += step
    if not candidates:
        return (cx0, cy0, 0.35), {"strategy": f"{strategy}_fallback_center", "bounds": [xmin, xmax, ymin, ymax]}
    candidates.sort(reverse=True)
    _, visible_count, clearance, x, y = candidates[0]
    return (float(x), float(y), 0.35), {
        "strategy": strategy,
        "clearance": clearance,
        "visible_count": visible_count,
        "bounds": [xmin, xmax, ymin, ymax],
        "candidate_count": len(candidates),
    }


def define_robot_proxy(stage, pose):
    from pxr import Gf, UsdGeom  # type: ignore

    robot_xform = UsdGeom.Xform.Define(stage, "/World/Phase8RobotProxy")
    robot_xform.ClearXformOpOrder()
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(pose[0], pose[1], pose[2]))
    body = UsdGeom.Cube.Define(stage, "/World/Phase8RobotProxy/Body")
    body.CreateSizeAttr(1.0)
    body.ClearXformOpOrder()
    body.AddScaleOp().Set(Gf.Vec3f(0.35, 0.25, 0.18))
    return robot_xform


def set_robot_pose(robot_xform, pose):
    from pxr import Gf  # type: ignore

    ops = robot_xform.GetOrderedXformOps()
    if ops:
        ops[0].Set(Gf.Vec3d(pose[0], pose[1], pose[2]))


def run_control_path(start_pose, obstacles, action_count):
    pose = start_pose
    yaw = 0.0
    actions = [
        (1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0), (-1.0, 0.0),
        (0.0, 1.0), (1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (1.0, 0.0),
        (0.0, 1.0), (-1.0, 0.0),
    ]
    alternatives = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.7, 0.7), (-0.7, 0.7)]
    rows = []
    current = (pose[0], pose[1])
    for step_id in range(action_count):
        dx, dy = actions[step_id % len(actions)]
        selected = None
        collision_path = ""
        for adx, ady in [(dx, dy)] + alternatives:
            nx, ny = current[0] + adx, current[1] + ady
            hit = collision_xy(nx, ny, obstacles)
            if not hit:
                selected = (nx, ny, adx, ady)
                break
            collision_path = hit
        if selected is None:
            rows.append({
                "step_id": step_id,
                "pose_x": current[0],
                "pose_y": current[1],
                "pose_z": pose[2],
                "yaw": yaw,
                "action_dx": 0.0,
                "action_dy": 0.0,
                "moved": False,
                "collision_risk": True,
                "collision_prim": collision_path,
                "stuck": True,
            })
            continue
        nx, ny, adx, ady = selected
        yaw = math.atan2(ady, adx) if abs(adx) + abs(ady) > 1e-6 else yaw
        current = (nx, ny)
        rows.append({
            "step_id": step_id,
            "pose_x": nx,
            "pose_y": ny,
            "pose_z": pose[2],
            "yaw": yaw,
            "action_dx": adx,
            "action_dy": ady,
            "moved": True,
            "collision_risk": False,
            "collision_prim": "",
            "stuck": False,
        })
    return rows


def point_samples_for_visible(visible, limit=2200):
    points = []
    point_sources = []
    for dist, rec in visible:
        mn, mx = rec["min"], rec["max"]
        cx, cy, cz = rec["center"]
        corners = [
            (mn[0], mn[1], mn[2]), (mn[0], mx[1], mn[2]), (mx[0], mn[1], mn[2]), (mx[0], mx[1], mn[2]),
            (mn[0], mn[1], mx[2]), (mn[0], mx[1], mx[2]), (mx[0], mn[1], mx[2]), (mx[0], mx[1], mx[2]),
            (cx, cy, cz),
        ]
        for pt in corners:
            points.append(pt)
            point_sources.append(rec)
            if len(points) >= limit:
                return points, point_sources
    return points, point_sources


def make_map_bounds(start_pose, records, margin=12.0):
    sx, sy = start_pose[0], start_pose[1]
    nearby = []
    for rec in records:
        cx, cy, _ = rec["center"]
        if abs(cx - sx) <= margin and abs(cy - sy) <= margin:
            nearby.append(rec)
    if nearby:
        xs = [sx] + [v for rec in nearby for v in (rec["min"][0], rec["max"][0])]
        ys = [sy] + [v for rec in nearby for v in (rec["min"][1], rec["max"][1])]
        xmin, xmax = max(min(xs) - 1.0, sx - margin), min(max(xs) + 1.0, sx + margin)
        ymin, ymax = max(min(ys) - 1.0, sy - margin), min(max(ys) + 1.0, sy + margin)
    else:
        xmin, xmax = sx - margin, sx + margin
        ymin, ymax = sy - margin, sy + margin
    # Keep a stable minimum window so unknown cells remain visible.
    if xmax - xmin < 16.0:
        cx = (xmin + xmax) / 2.0
        xmin, xmax = cx - 8.0, cx + 8.0
    if ymax - ymin < 16.0:
        cy = (ymin + ymax) / 2.0
        ymin, ymax = cy - 8.0, cy + 8.0
    return xmin, xmax, ymin, ymax


class BevMap:
    def __init__(self, xmin, xmax, ymin, ymax, resolution):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.resolution = float(resolution)
        self.width = max(8, int(math.ceil((self.xmax - self.xmin) / self.resolution)))
        self.height = max(8, int(math.ceil((self.ymax - self.ymin) / self.resolution)))
        import numpy as np

        self.occ = np.zeros((self.height, self.width), dtype=np.uint8)
        self.observed = np.zeros((self.height, self.width), dtype=np.uint16)
        self.robot_trace = []

    def world_to_grid(self, x, y):
        ix = int(math.floor((float(x) - self.xmin) / self.resolution))
        iy = int(math.floor((float(y) - self.ymin) / self.resolution))
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return ix, iy
        return None

    def mark_cell(self, ix, iy, value):
        if 0 <= ix < self.width and 0 <= iy < self.height:
            if self.occ[iy, ix] == UNKNOWN:
                self.occ[iy, ix] = value
            elif value == OCCUPIED:
                self.occ[iy, ix] = OCCUPIED
            elif self.occ[iy, ix] != OCCUPIED:
                self.occ[iy, ix] = FREE
            self.observed[iy, ix] = min(int(self.observed[iy, ix]) + 1, 65535)

    def mark_world(self, x, y, value):
        cell = self.world_to_grid(x, y)
        if cell:
            self.mark_cell(cell[0], cell[1], value)

    def mark_robot_disk(self, x, y, radius=0.45):
        self.robot_trace.append((float(x), float(y)))
        cells = max(1, int(math.ceil(radius / self.resolution)))
        c = self.world_to_grid(x, y)
        if not c:
            return
        cx, cy = c
        for dy in range(-cells, cells + 1):
            for dx in range(-cells, cells + 1):
                wx = self.xmin + (cx + dx + 0.5) * self.resolution
                wy = self.ymin + (cy + dy + 0.5) * self.resolution
                if math.hypot(wx - x, wy - y) <= radius:
                    self.mark_cell(cx + dx, cy + dy, FREE)

    def mark_ray(self, x0, y0, x1, y1, terminal_value):
        c0 = self.world_to_grid(x0, y0)
        c1 = self.world_to_grid(x1, y1)
        if not c0 and not c1:
            return 0
        dist = max(math.hypot(x1 - x0, y1 - y0), self.resolution)
        steps = max(2, int(math.ceil(dist / (self.resolution * 0.5))))
        changed_before = int((self.occ != UNKNOWN).sum())
        for i in range(max(1, steps - 1)):
            t = i / float(steps)
            self.mark_world(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, FREE)
        self.mark_world(x1, y1, terminal_value)
        changed_after = int((self.occ != UNKNOWN).sum())
        return max(0, changed_after - changed_before)

    def stats(self):
        import numpy as np

        occupied = int((self.occ == OCCUPIED).sum())
        free = int((self.occ == FREE).sum())
        unknown = int((self.occ == UNKNOWN).sum())
        known = occupied + free
        total = int(self.occ.size)
        return {
            "occupied_cells": occupied,
            "free_cells": free,
            "unknown_cells": unknown,
            "known_cells": known,
            "known_ratio": known / max(total, 1),
            "observed_count_sum": int(np.asarray(self.observed, dtype=np.uint64).sum()),
            "map_min_x": self.xmin,
            "map_max_x": self.xmax,
            "map_min_y": self.ymin,
            "map_max_y": self.ymax,
        }

    def snapshot_arrays(self, robot_pose=None):
        import numpy as np

        robot_layer = np.zeros_like(self.occ, dtype=np.uint8)
        if robot_pose is not None:
            c = self.world_to_grid(robot_pose[0], robot_pose[1])
            if c:
                robot_layer[c[1], c[0]] = 1
        return {
            "occupancy": self.occ.copy(),
            "observed_count": self.observed.copy(),
            "robot_pose_layer": robot_layer,
            "resolution": np.asarray([self.resolution], dtype=np.float32),
            "bounds": np.asarray([self.xmin, self.xmax, self.ymin, self.ymax], dtype=np.float32),
        }


def update_map_from_observation(bev, pose, records, obstacles, max_range=12.0):
    px, py, pz, yaw = pose
    visible = visible_records(pose, records, max_range=max_range, fov_deg=360.0)
    # Avoid far scene extents dominating point statistics; keep local geometry.
    visible = [(d, r) for d, r in visible if d <= max_range and (is_obstacle_like(r) or is_floor_like(r))]
    points, point_sources = point_samples_for_visible(visible)
    finite_points = [(p, rec) for p, rec in zip(points, point_sources) if all(finite(v) for v in p)]
    changed_before = bev.stats()["known_cells"]
    bev.mark_robot_disk(px, py, radius=0.45)
    for (x, y, z), rec in finite_points:
        terminal = OCCUPIED if is_obstacle_like(rec) and z > 0.05 else FREE
        bev.mark_ray(px, py, float(x), float(y), terminal)
    stats = bev.stats()
    distances = [math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) for (x, y, z), _ in finite_points]
    stats.update({
        "point_count": len(points),
        "finite_point_ratio": len(finite_points) / max(len(points), 1),
        "depth_min": min(distances) if distances else None,
        "depth_max": max(distances) if distances else None,
        "depth_mean": statistics.fmean(distances) if distances else None,
        "new_known_cells": max(0, stats["known_cells"] - changed_before),
        "visible_record_count": len(visible),
    })
    return stats


def save_npz(path, arrays):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_pngs(plot_dir, step_rows, bev):
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap

        steps = [int(r["step_id"]) for r in step_rows]
        known = [float(r["known_ratio"]) for r in step_rows]
        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        ax.plot(steps, known, marker="o", linewidth=2)
        ax.set_xlabel("step")
        ax.set_ylabel("known_ratio")
        ax.set_title("Known ratio by mapping step")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = plot_dir / "known_ratio_curve.png"
        fig.savefig(out)
        plt.close(fig)
        saved.append(str(out))

        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        ax.plot(steps, [int(r["occupied_cells"]) for r in step_rows], label="occupied", linewidth=2)
        ax.plot(steps, [int(r["free_cells"]) for r in step_rows], label="free", linewidth=2)
        ax.plot(steps, [int(r["unknown_cells"]) for r in step_rows], label="unknown", linewidth=2)
        ax.set_xlabel("step")
        ax.set_ylabel("cells")
        ax.set_title("BEV cell counts by step")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = plot_dir / "occupied_free_unknown_by_step.png"
        fig.savefig(out)
        plt.close(fig)
        saved.append(str(out))

        display = bev.occ.copy()
        for x, y in bev.robot_trace:
            c = bev.world_to_grid(x, y)
            if c:
                display[c[1], c[0]] = ROBOT
        cmap = ListedColormap(["#6f6f6f", "#f7f7f2", "#151515", "#d62728"])
        fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
        ax.imshow(display, origin="lower", cmap=cmap, vmin=0, vmax=3,
                  extent=[bev.xmin, bev.xmax, bev.ymin, bev.ymax])
        if bev.robot_trace:
            xs, ys = zip(*bev.robot_trace)
            ax.plot(xs, ys, color="#1f77b4", linewidth=1.5, marker="o", markersize=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Final BEV partial map")
        fig.tight_layout()
        out = plot_dir / "bev_map_final.png"
        fig.savefig(out)
        plt.close(fig)
        saved.append(str(out))

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        if bev.robot_trace:
            xs, ys = zip(*bev.robot_trace)
            ax.plot(xs, ys, marker="o", linewidth=2)
            ax.scatter([xs[0]], [ys[0]], color="green", label="start")
            ax.scatter([xs[-1]], [ys[-1]], color="red", label="end")
        ax.set_xlim(bev.xmin, bev.xmax)
        ax.set_ylim(bev.ymin, bev.ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Robot XY trace")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = plot_dir / "robot_xy_trace.png"
        fig.savefig(out)
        plt.close(fig)
        saved.append(str(out))
    except Exception as exc:
        (plot_dir / "plot_error.txt").write_text(repr(exc), encoding="utf-8")
    return saved


def write_ascii_map(path, bev):
    chars = {UNKNOWN: "?", FREE: ".", OCCUPIED: "#"}
    lines = []
    occ = bev.occ
    trace_cells = {bev.world_to_grid(x, y) for x, y in bev.robot_trace}
    trace_cells.discard(None)
    # Downsample if the grid is large.
    step_y = max(1, occ.shape[0] // 60)
    step_x = max(1, occ.shape[1] // 100)
    for iy in range(occ.shape[0] - 1, -1, -step_y):
        row = []
        for ix in range(0, occ.shape[1], step_x):
            row.append("R" if (ix, iy) in trace_cells else chars.get(int(occ[iy, ix]), "?"))
        lines.append("".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()

    scene_path = Path(args.scene).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    maps_dir = out_dir / "maps"
    plots_dir = out_dir / "plots"
    summary_dir = out_dir / "summary"
    for d in [maps_dir, plots_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary = {
        "scene_path": str(scene_path),
        "scene_exists": scene_path.exists(),
        "scene_size_bytes": scene_path.stat().st_size if scene_path.exists() else None,
        "simulation_app_started": False,
        "stage_opened": False,
        "stage_available": False,
        "prim_count": 0,
        "type_counts": {},
        "geometry_record_count": 0,
        "obstacle_record_count": 0,
        "robot_spawned": False,
        "robot_pose_readable": False,
        "robot_initial_pose": None,
        "step_count": 0,
        "valid_observation_steps": 0,
        "final_known_ratio": 0.0,
        "final_occupied_cells": 0,
        "final_free_cells": 0,
        "final_unknown_cells": 0,
        "total_new_known_cells": 0,
        "known_ratio_monotonic_non_decreasing": False,
        "map_snapshots_saved": 0,
        "plots_saved": 0,
        "plot_paths": [],
        "collision_count": 0,
        "stuck_count": 0,
        "training_started": False,
        "RL_started": False,
        "checkpoint_created": False,
        "primary_long_rollout_started": False,
        "pi_finetune_started": False,
        "map_predict_started": False,
        "phase8_passed": False,
        "allow_phase9_candidate_gain_smoke": False,
        "error": None,
        "traceback": None,
    }

    app = None
    step_rows = []
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True})
        summary["simulation_app_started"] = True

        import omni.usd
        ctx = omni.usd.get_context()
        summary["stage_opened"] = bool(ctx.open_stage(str(scene_path)))
        for _ in range(max(args.updates, 1)):
            app.update()
        stage = ctx.get_stage()
        summary["stage_available"] = bool(stage)
        if not stage:
            raise RuntimeError("stage unavailable after open")

        records, type_counts = collect_geometry(stage)
        obstacles = [rec for rec in records if is_obstacle_like(rec)]
        summary["type_counts"] = type_counts
        summary["prim_count"] = sum(type_counts.values())
        summary["geometry_record_count"] = len(records)
        summary["obstacle_record_count"] = len(obstacles)

        start_pose, start_meta = choose_start(records, obstacles)
        summary["robot_initial_pose"] = {"x": start_pose[0], "y": start_pose[1], "z": start_pose[2], **start_meta}
        robot = define_robot_proxy(stage, start_pose)
        summary["robot_spawned"] = True
        summary["robot_pose_readable"] = True
        for _ in range(3):
            app.update()

        xmin, xmax, ymin, ymax = make_map_bounds(start_pose, records, margin=12.0)
        bev = BevMap(xmin, xmax, ymin, ymax, args.resolution)
        controls = run_control_path(start_pose, obstacles, max(args.steps, 8))
        previous_known_ratio = -1.0
        monotonic = True
        snapshot_count = 0
        for row in controls[: max(args.steps, 8)]:
            pose = (float(row["pose_x"]), float(row["pose_y"]), float(row["pose_z"]) + 0.85, float(row["yaw"]))
            set_robot_pose(robot, (pose[0], pose[1], start_pose[2]))
            app.update()
            mstats = update_map_from_observation(bev, pose, records, obstacles, max_range=12.0)
            failure_reason = ""
            if row["collision_risk"]:
                failure_reason = "collision_risk"
            elif row["stuck"]:
                failure_reason = "stuck"
            elif mstats["point_count"] <= 0 or mstats["finite_point_ratio"] <= 0.0:
                failure_reason = "empty_observation"
            if mstats["known_ratio"] + 1e-9 < previous_known_ratio:
                monotonic = False
            previous_known_ratio = max(previous_known_ratio, mstats["known_ratio"])
            out_row = {
                "step_id": int(row["step_id"]),
                "pose_x": round(pose[0], 6),
                "pose_y": round(pose[1], 6),
                "pose_z": round(pose[2], 6),
                "yaw": round(pose[3], 6),
                "point_count": int(mstats["point_count"]),
                "finite_point_ratio": round(float(mstats["finite_point_ratio"]), 6),
                "occupied_cells": int(mstats["occupied_cells"]),
                "free_cells": int(mstats["free_cells"]),
                "unknown_cells": int(mstats["unknown_cells"]),
                "known_cells": int(mstats["known_cells"]),
                "known_ratio": round(float(mstats["known_ratio"]), 8),
                "new_known_cells": int(mstats["new_known_cells"]),
                "observed_count_sum": int(mstats["observed_count_sum"]),
                "map_min_x": round(float(mstats["map_min_x"]), 6),
                "map_max_x": round(float(mstats["map_max_x"]), 6),
                "map_min_y": round(float(mstats["map_min_y"]), 6),
                "map_max_y": round(float(mstats["map_max_y"]), 6),
                "failure_reason": failure_reason,
            }
            step_rows.append(out_row)
            save_npz(maps_dir / f"map_step_{int(row['step_id']):03d}.npz", bev.snapshot_arrays(robot_pose=pose))
            snapshot_count += 1

        with (summary_dir / "mapping_steps.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "step_id", "pose_x", "pose_y", "pose_z", "yaw", "point_count", "finite_point_ratio",
                "occupied_cells", "free_cells", "unknown_cells", "known_cells", "known_ratio",
                "new_known_cells", "observed_count_sum", "map_min_x", "map_max_x", "map_min_y",
                "map_max_y", "failure_reason",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(step_rows)
        write_ascii_map(summary_dir / "bev_map_final_ascii.txt", bev)
        plot_paths = write_pngs(plots_dir, step_rows, bev)

        final = step_rows[-1] if step_rows else {}
        valid_obs = sum(1 for r in step_rows if int(r["point_count"]) > 0 and float(r["finite_point_ratio"]) > 0.0)
        summary.update({
            "step_count": len(step_rows),
            "valid_observation_steps": valid_obs,
            "final_known_ratio": float(final.get("known_ratio", 0.0) or 0.0),
            "final_occupied_cells": int(final.get("occupied_cells", 0) or 0),
            "final_free_cells": int(final.get("free_cells", 0) or 0),
            "final_unknown_cells": int(final.get("unknown_cells", 0) or 0),
            "total_new_known_cells": int(sum(int(r["new_known_cells"]) for r in step_rows)),
            "known_ratio_monotonic_non_decreasing": monotonic,
            "map_snapshots_saved": snapshot_count,
            "plots_saved": len(plot_paths),
            "plot_paths": plot_paths,
            "collision_count": sum(1 for r in controls if r["collision_risk"]),
            "stuck_count": sum(1 for r in controls if r["stuck"]),
            "map_min_x": bev.xmin,
            "map_max_x": bev.xmax,
            "map_min_y": bev.ymin,
            "map_max_y": bev.ymax,
            "map_resolution": bev.resolution,
        })
        first_positive = any(int(r["new_known_cells"]) > 0 for r in step_rows[: min(4, len(step_rows))])
        summary["phase8_passed"] = bool(
            summary["simulation_app_started"]
            and summary["stage_available"]
            and summary["robot_spawned"]
            and summary["robot_pose_readable"]
            and summary["step_count"] >= 8
            and summary["valid_observation_steps"] >= math.ceil(0.8 * summary["step_count"])
            and summary["known_ratio_monotonic_non_decreasing"]
            and first_positive
            and summary["final_occupied_cells"] > 0
            and summary["final_free_cells"] > 0
            and summary["final_unknown_cells"] > 0
            and summary["collision_count"] == 0
            and summary["stuck_count"] == 0
            and summary["map_snapshots_saved"] >= summary["step_count"]
            and summary["plots_saved"] >= 4
        )
        summary["allow_phase9_candidate_gain_smoke"] = summary["phase8_passed"]
    except Exception as exc:
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
    finally:
        (summary_dir / "mapping_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        if app is not None:
            try:
                app.close()
            except Exception:
                pass

    return 0 if summary.get("phase8_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
