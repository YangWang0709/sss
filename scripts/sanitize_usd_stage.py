#!/usr/bin/env python3
"""Attempt lightweight sanitization of an exported USDA stage.

This is intentionally conservative. It operates only on a copy/export and never
modifies the original USD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import traceback


MATERIAL_PATTERNS = ["material:binding", "rel material", "def Material", "def Shader", "outputs:surface", "inputs:diffuse"]
PHYSICS_PATTERNS = ["Physics", "Physx", "physics", "physx", "apiSchemas"]


def filter_text(text: str, patterns: list[str]) -> str:
    lines = []
    for line in text.splitlines():
        if any(p in line for p in patterns):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    usd_path = Path(args.usd).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "usd": str(usd_path),
        "exists": usd_path.exists(),
        "outputs": {},
        "errors": [],
        "negative_scope": {"training": False, "rl": False, "checkpoint": False, "original_modified": False},
    }

    try:
        text = usd_path.read_text(encoding="utf-8")
    except Exception:
        report["errors"].append({"phase": "read_text", "traceback": traceback.format_exc()})
        (out_dir / "sanitize_usd_stage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    variants = {
        "building_scene_no_materials.usda": filter_text(text, MATERIAL_PATTERNS),
        "building_scene_no_physics.usda": filter_text(text, PHYSICS_PATTERNS),
        "building_scene_mesh_only.usda": filter_text(text, MATERIAL_PATTERNS + PHYSICS_PATTERNS),
    }
    for name, value in variants.items():
        path = out_dir / name
        path.write_text(value, encoding="utf-8")
        report["outputs"][name] = {"path": str(path), "size_bytes": path.stat().st_size}

    (out_dir / "sanitize_usd_stage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
