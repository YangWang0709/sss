#!/usr/bin/env python3
"""Attempt clean USDA and flattened USDA export from an input USD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    usd_path = Path(args.usd).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "building_scene_clean.usda"
    flat_path = out_dir / "building_scene_flattened.usda"
    report_path = out_dir / "repair_usd_export_clean_report.json"

    report = {
        "usd": str(usd_path),
        "exists": usd_path.exists(),
        "size_bytes": usd_path.stat().st_size if usd_path.exists() else None,
        "sha256": hashlib.sha256(usd_path.read_bytes()).hexdigest() if usd_path.exists() else None,
        "python": sys.executable,
        "pxr_import_ok": False,
        "sdf_layer_open": False,
        "clean_export_path": str(clean_path),
        "clean_export_success": False,
        "clean_stage_open": False,
        "flattened_export_path": str(flat_path),
        "flattened_export_success": False,
        "errors": [],
        "negative_scope": {"isaac_simulation_app": False, "training": False, "rl": False, "checkpoint": False},
    }

    try:
        from pxr import Sdf, Usd  # type: ignore
        report["pxr_import_ok"] = True
    except Exception:
        report["errors"].append({"phase": "pxr_import", "traceback": traceback.format_exc()})
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    try:
        layer = Sdf.Layer.FindOrOpen(str(usd_path))
        report["sdf_layer_open"] = bool(layer)
        if layer:
            layer.Export(str(clean_path))
            report["clean_export_success"] = clean_path.exists() and clean_path.stat().st_size > 0
    except Exception:
        report["errors"].append({"phase": "clean_export", "traceback": traceback.format_exc()})

    try:
        if report["clean_export_success"]:
            stage = Usd.Stage.Open(str(clean_path))
            report["clean_stage_open"] = bool(stage)
            if stage:
                flat_layer = stage.Flatten()
                flat_layer.Export(str(flat_path))
                report["flattened_export_success"] = flat_path.exists() and flat_path.stat().st_size > 0
    except Exception:
        report["errors"].append({"phase": "flatten_export", "traceback": traceback.format_exc()})

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["clean_export_success"] or report["flattened_export_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
