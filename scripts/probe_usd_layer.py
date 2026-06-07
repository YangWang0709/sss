#!/usr/bin/env python3
"""Probe a USD layer without starting Isaac SimulationApp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def safe_string(value, limit=800):
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    usd_path = Path(args.usd).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "usd": str(usd_path),
        "exists": usd_path.exists(),
        "size_bytes": usd_path.stat().st_size if usd_path.exists() else None,
        "sha256": hashlib.sha256(usd_path.read_bytes()).hexdigest() if usd_path.exists() else None,
        "python": sys.executable,
        "pxr_import_ok": False,
        "sdf_layer_open": False,
        "stage_open": False,
        "default_prim": None,
        "root_prims": [],
        "prim_count": 0,
        "type_counts": {},
        "root_layer_sublayers": [],
        "references": [],
        "payloads": [],
        "asset_paths": [],
        "materials": [],
        "physics_schema_prims": [],
        "errors": [],
        "negative_scope": {"isaac_simulation_app": False, "training": False, "rl": False, "checkpoint": False},
    }

    try:
        from pxr import Sdf, Usd  # type: ignore
        report["pxr_import_ok"] = True
    except Exception:
        report["errors"].append({"phase": "pxr_import", "traceback": traceback.format_exc()})
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    try:
        layer = Sdf.Layer.FindOrOpen(str(usd_path))
        report["sdf_layer_open"] = bool(layer)
        if layer:
            report["root_layer_identifier"] = layer.identifier
            report["root_layer_real_path"] = layer.realPath
            report["root_layer_sublayers"] = list(layer.subLayerPaths)
    except Exception:
        report["errors"].append({"phase": "sdf_layer_find_or_open", "traceback": traceback.format_exc()})

    try:
        stage = Usd.Stage.Open(str(usd_path))
        report["stage_open"] = bool(stage)
        if stage:
            default = stage.GetDefaultPrim()
            report["default_prim"] = str(default.GetPath()) if default and default.IsValid() else None
            report["root_prims"] = [str(c.GetPath()) for c in stage.GetPseudoRoot().GetChildren()]
            root_layer = stage.GetRootLayer()
            report["root_layer_sublayers"] = list(root_layer.subLayerPaths)
            for prim in stage.Traverse():
                report["prim_count"] += 1
                type_name = prim.GetTypeName() or ""
                report["type_counts"][type_name] = report["type_counts"].get(type_name, 0) + 1
                path = str(prim.GetPath())
                if type_name == "Material":
                    report["materials"].append(path)
                applied = [str(s) for s in prim.GetAppliedSchemas()]
                if any("Physics" in s or "Physx" in s for s in applied):
                    report["physics_schema_prims"].append({"prim": path, "schemas": applied})
                refs = prim.GetMetadata("references")
                if refs:
                    report["references"].append({"prim": path, "value": safe_string(refs)})
                payload = prim.GetMetadata("payload")
                if payload:
                    report["payloads"].append({"prim": path, "value": safe_string(payload)})
                try:
                    for spec in prim.GetPrimStack():
                        for key in spec.ListInfoKeys():
                            val = spec.GetInfo(key)
                            sval = str(val)
                            if any(token in sval.lower() for token in [".usd", ".usda", ".usdc", ".png", ".jpg", ".jpeg", ".mdl", "@"]):
                                report["asset_paths"].append({"prim": path, "field": str(key), "value": safe_string(val)})
                except Exception as exc:
                    report["errors"].append({"phase": "prim_stack_scan", "prim": path, "error": repr(exc)})
    except Exception:
        report["errors"].append({"phase": "stage_open_or_traverse", "traceback": traceback.format_exc()})

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["stage_open"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
