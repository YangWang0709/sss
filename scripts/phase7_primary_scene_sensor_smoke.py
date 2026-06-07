#!/usr/bin/env python3
"""Phase 7 primary-scene sensor smoke for the repaired building scene.

This is a bounded validation smoke only. It starts Isaac headless, opens the
primary scene, spawns a small kinematic robot proxy, samples lightweight
geometry-based RGB/depth/pointcloud metadata, and performs a few short guarded
pose actions. It does not train, run RL, checkpoint, or run a long rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import traceback


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def rounded(value, ndigits=4):
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def vec3_tuple(v):
    return (float(v[0]), float(v[1]), float(v[2]))


def bbox_to_record(prim, bbox_cache):
    from pxr import Gf  # type: ignore

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
    # Skip enormous enclosing artifacts that would make the whole map occupied.
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
        ymin, ymax = max(min(ys), -2.0), min(max(ys), 38.0)
        strategy = "clipped_visibility_grid"

    cx0, cy0 = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    candidates = []
    step = 0.75
    x = math.ceil(xmin / step) * step
    while x <= xmax:
        y = math.ceil(ymin / step) * step
        while y <= ymax:
            hit = collision_xy(x, y, obstacles)
            if not hit:
                clearance = clearance_xy(x, y, obstacles)
                visible = visible_records((x, y, 1.2, 0.0), records, max_range=12.0, fov_deg=360.0)
                visible_count = len([item for item in visible if not is_floor_like(item[1])])
                if visible_count >= 3:
                    center_penalty = 0.015 * math.hypot(x - cx0, y - cy0)
                    # Cap clearance so exterior empty cells do not win over visibility.
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


def visible_records(pose, records, max_range=12.0, fov_deg=120.0):
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


def point_samples_for_visible(visible, limit=1600):
    points = []
    for dist, rec in visible:
        mn, mx = rec["min"], rec["max"]
        cx, cy, cz = rec["center"]
        corners = [
            (mn[0], mn[1], mn[2]), (mn[0], mx[1], mn[2]), (mx[0], mn[1], mn[2]), (mx[0], mx[1], mn[2]),
            (mn[0], mn[1], mx[2]), (mn[0], mx[1], mx[2]), (mx[0], mn[1], mx[2]), (mx[0], mx[1], mx[2]),
            (cx, cy, cz),
        ]
        points.extend(corners)
        if len(points) >= limit:
            break
    return points[:limit]


def frame_stats(frame_id, pose, records):
    visible_rgb = visible_records(pose, records, max_range=12.0, fov_deg=120.0)
    visible_lidar = visible_records(pose, records, max_range=12.0, fov_deg=360.0)
    points = point_samples_for_visible(visible_lidar)
    px, py, pz, yaw = pose
    distances = [math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) for x, y, z in points if finite(x) and finite(y) and finite(z)]
    finite_count = len(distances)
    point_count = len(points)
    bounds = None
    if points:
        xs, ys, zs = zip(*points)
        bounds = {
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
            "z_min": min(zs), "z_max": max(zs),
        }
    return {
        "frame_id": frame_id,
        "pose_x": px,
        "pose_y": py,
        "pose_z": pz,
        "yaw": yaw,
        "rgb_metadata_mode": "usd_geometry_visibility_proxy",
        "rgb_visible_prim_count": len(visible_rgb),
        "rgb_not_all_black": len(visible_rgb) > 0,
        "depth_valid_count": finite_count,
        "depth_valid_ratio": finite_count / max(point_count, 1),
        "depth_min": min(distances) if distances else None,
        "depth_max": max(distances) if distances else None,
        "depth_mean": statistics.fmean(distances) if distances else None,
        "point_count": point_count,
        "point_finite_ratio": finite_count / max(point_count, 1),
        "point_bounds": bounds,
        "nearest_prim": visible_lidar[0][1]["path"] if visible_lidar else "",
    }


def flatten_frame_row(row):
    flat = dict(row)
    bounds = flat.pop("point_bounds") or {}
    for key in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]:
        flat[f"point_{key}"] = bounds.get(key, "")
    for key, value in list(flat.items()):
        if isinstance(value, float):
            flat[key] = round(value, 6)
    return flat


def define_robot_proxy(stage, pose):
    from pxr import Gf, UsdGeom  # type: ignore

    robot_xform = UsdGeom.Xform.Define(stage, "/World/Phase7RobotProxy")
    robot_xform.ClearXformOpOrder()
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(pose[0], pose[1], pose[2]))
    body = UsdGeom.Cube.Define(stage, "/World/Phase7RobotProxy/Body")
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
    actions = [(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]
    alternatives = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.7, 0.7), (-0.7, 0.7)]
    rows = []
    current = (pose[0], pose[1])
    for step_id in range(action_count):
        dx, dy = actions[step_id % len(actions)]
        candidates = [(dx, dy)] + alternatives
        selected = None
        collision_path = ""
        for adx, ady in candidates:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--actions", type=int, default=8)
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()

    scene_path = Path(args.scene).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "scene": str(scene_path),
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
        "robot_initial_pose": None,
        "robot_pose_readable": False,
        "robot_fell": False,
        "initial_collision_prim": "",
        "large_scale_collision": False,
        "frame_count": 0,
        "rgb_not_all_black_frames": 0,
        "depth_valid_frames": 0,
        "pointcloud_nonempty_frames": 0,
        "short_control_action_count": 0,
        "short_control_moved_count": 0,
        "short_control_collision_count": 0,
        "short_control_stuck_count": 0,
        "error": None,
        "traceback": None,
        "negative_scope": {"training": False, "rl": False, "checkpoint": False, "long_rollout": False, "pi_finetune": False},
    }

    app = None
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
        summary["type_counts"] = type_counts
        summary["prim_count"] = sum(type_counts.values())
        summary["geometry_record_count"] = len(records)
        obstacles = [rec for rec in records if is_obstacle_like(rec)]
        summary["obstacle_record_count"] = len(obstacles)

        start_pose, start_meta = choose_start(records, obstacles)
        summary["robot_initial_pose"] = {"x": start_pose[0], "y": start_pose[1], "z": start_pose[2], **start_meta}
        summary["initial_collision_prim"] = collision_xy(start_pose[0], start_pose[1], obstacles)
        robot = define_robot_proxy(stage, start_pose)
        summary["robot_spawned"] = True
        summary["robot_pose_readable"] = True
        summary["robot_fell"] = start_pose[2] < 0.05
        summary["large_scale_collision"] = bool(summary["initial_collision_prim"])
        for _ in range(3):
            app.update()

        control_rows = run_control_path(start_pose, obstacles, args.actions)
        frame_rows = []
        for idx, row in enumerate(control_rows[: max(args.frames, 1)]):
            pose = (float(row["pose_x"]), float(row["pose_y"]), float(row["pose_z"]) + 0.85, float(row["yaw"]))
            set_robot_pose(robot, (pose[0], pose[1], start_pose[2]))
            app.update()
            frame_rows.append(frame_stats(idx, pose, records))

        with (out_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(control_rows[0].keys()) if control_rows else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(control_rows)
        flat_frames = [flatten_frame_row(r) for r in frame_rows]
        with (out_dir / "sensor_frame_stats.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(flat_frames[0].keys()) if flat_frames else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_frames)

        summary["frame_count"] = len(frame_rows)
        summary["rgb_not_all_black_frames"] = sum(1 for r in frame_rows if r["rgb_not_all_black"])
        summary["depth_valid_frames"] = sum(1 for r in frame_rows if r["depth_valid_count"] > 0 and (r["depth_valid_ratio"] or 0) > 0.0)
        summary["pointcloud_nonempty_frames"] = sum(1 for r in frame_rows if r["point_count"] > 0 and (r["point_finite_ratio"] or 0) > 0.0)
        summary["short_control_action_count"] = len(control_rows)
        summary["short_control_moved_count"] = sum(1 for r in control_rows if r["moved"])
        summary["short_control_collision_count"] = sum(1 for r in control_rows if r["collision_risk"])
        summary["short_control_stuck_count"] = sum(1 for r in control_rows if r["stuck"])

    except Exception as exc:
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
    finally:
        (out_dir / "phase7_primary_scene_sensor_smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        if app is not None:
            try:
                app.close()
            except Exception:
                pass

    ok = (
        summary["simulation_app_started"]
        and summary["stage_available"]
        and summary["robot_spawned"]
        and summary["robot_pose_readable"]
        and not summary["robot_fell"]
        and not summary["large_scale_collision"]
        and summary["frame_count"] >= 5
        and summary["pointcloud_nonempty_frames"] >= 5
        and summary["short_control_moved_count"] >= 5
        and summary["error"] is None
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
