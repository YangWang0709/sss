#!/usr/bin/env python3
"""Open a USD stage in Isaac headless and report only metadata.

This script does not simulate, train, rollout, checkpoint, or run RL. It only
starts SimulationApp, opens one USD, advances a few app updates, and records
stage metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    usd_path = Path(args.usd).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "usd": str(usd_path),
        "exists": usd_path.exists(),
        "size_bytes": usd_path.stat().st_size if usd_path.exists() else None,
        "sha256": hashlib.sha256(usd_path.read_bytes()).hexdigest() if usd_path.exists() else None,
        "simulation_app_started": False,
        "open_stage_result": None,
        "stage_available": False,
        "prim_count": 0,
        "root_prims": [],
        "type_counts": {},
        "meters_per_unit": None,
        "up_axis": None,
        "error": None,
        "negative_scope": {"training": False, "rl": False, "checkpoint": False, "rollout": False, "pi_finetune": False},
    }

    def alarm_handler(_signum, _frame):
        raise TimeoutError(f"probe timeout after {args.timeout}s")

    app = None
    try:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(max(args.timeout, 1))
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
        result["simulation_app_started"] = True

        import omni.usd
        ctx = omni.usd.get_context()
        result["open_stage_result"] = bool(ctx.open_stage(str(usd_path)))
        for _ in range(max(args.updates, 1)):
            app.update()
        stage = ctx.get_stage()
        result["stage_available"] = bool(stage)
        if stage:
            try:
                from pxr import UsdGeom  # type: ignore
                result["meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(stage)
                result["up_axis"] = str(UsdGeom.GetStageUpAxis(stage))
            except Exception as exc:
                result["usdgeom_metadata_error"] = repr(exc)
            result["root_prims"] = [str(c.GetPath()) for c in stage.GetPseudoRoot().GetChildren()]
            for prim in stage.Traverse():
                result["prim_count"] += 1
                type_name = prim.GetTypeName() or ""
                result["type_counts"][type_name] = result["type_counts"].get(type_name, 0) + 1
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        # Write the result before app.close(); some Kit builds terminate during close.
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        if app is not None:
            try:
                app.close()
            except Exception:
                pass

    ok = result["simulation_app_started"] and result["stage_available"] and result["prim_count"] > 0 and result["error"] is None
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
