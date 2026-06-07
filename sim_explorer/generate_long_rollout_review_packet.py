#!/usr/bin/env python3
"""Generate LR-7 human review packet for the bounded long expert rollout."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_stage4a714_2d_review_packet import DEFAULT_SCENE_METADATA, DEFAULT_USD, generate_packet


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "outputs/autonomous_long_rollout_bridge"
RUNTIME = ROOT / "outputs/stage4a_long_bounded_expert_rollout_runtime"
OUT = BRIDGE / "stage_lr7_long_rollout_review_packet"
DEFAULT_MANIFEST = RUNTIME / "long_rollout_manifest.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            lines.extend([f"## {key}", "", "```json", json.dumps(value, indent=2, sort_keys=True), "```", ""])
        else:
            lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def rel(path: str | Path, root: Path = OUT) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return Path(p.relative_to(root) if p.is_relative_to(root) else Path("..") / Path(p.name)).as_posix()


def copy_if_present(src: Path, dst: Path) -> str | None:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_file() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)
    return str(dst)


def write_how_to_save(path: Path, main_html: Path, action_story_html: Path, mp4_path: str | None) -> None:
    lines = [
        "# LR-7 Bounded Long Rollout Human Review Instructions",
        "",
        f"Open `{main_html.name}` first. Use the Start and Step buttons to inspect all 150 decisions.",
        "",
        "For each step:",
        "",
        "1. Check the 2D map: blue is historical swept/observed area, cyan is newly observed area, blue dots are source camera poses, pink crosses are selected action targets, and the green arrow is the current move.",
        "2. Check the RGB camera image for the same step.",
        "3. Set `human_review_status`, `human_review_reason`, optional `human_comment`, and leave `promote_candidate_yes_no` empty unless the step is approved and should be considered later.",
        "4. Click `Export review JSON`, then `Download review JSON`; use `Download review CSV` if spreadsheet review is easier.",
        "",
        f"Use `{action_story_html.name}` for a scrollable start/step story view.",
    ]
    if mp4_path:
        lines.append(f"Use `{Path(mp4_path).name}` as the flythrough MP4 reference.")
    lines.extend(
        [
            "",
            "Human approval does not automatically promote labels. A future import/promotion stage must make a separate decision.",
            "",
            "Negative scope: no label promotion, no training, no checkpoint, no Isaac startup, no map_predict, no rollout, no RL/GDPO/PPO during review packet generation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_action_story(path: Path, records: list[dict[str, Any]], mp4_rel: str | None) -> None:
    starts = sorted({int(row["start_variant_id"]) for row in records})
    cards = []
    for row in sorted(records, key=lambda item: (int(item["start_variant_id"]), int(item["step_id"]))):
        start = int(row["start_variant_id"])
        step = int(row["step_id"])
        map_img = esc(row.get("map_image", ""))
        rgb_img = esc(row.get("relative_rgb", ""))
        source = ", ".join(f"{float(v):.3f}" for v in row.get("source_pose_xyz", [0, 0, 0]))
        action = ", ".join(f"{float(v):.3f}" for v in row.get("action_world_xyz", [0, 0, 0]))
        cards.append(
            f"""
      <article class="card" id="start-{start:03d}-step-{step:03d}" data-start="{start}" data-step="{step}">
        <header>
          <strong>start {start:03d} / step {step:03d}</strong>
          <span>{esc(row.get("action_distance_flag", ""))} / {float(row.get("source_to_action_distance_m", 0.0)):.3f} m</span>
        </header>
        <div class="media">
          <img src="{map_img}" alt="2D map start {start:03d} step {step:03d}" />
          <img src="{rgb_img}" alt="RGB camera start {start:03d} step {step:03d}" />
        </div>
        <dl>
          <dt>source xyz</dt><dd>{esc(source)}</dd>
          <dt>action xyz</dt><dd>{esc(action)}</dd>
          <dt>observed before</dt><dd>{float(row.get("observed_ratio_before", 0.0)):.4f}</dd>
          <dt>observed after</dt><dd>{float(row.get("observed_ratio_after_current_capture", 0.0)):.4f}</dd>
          <dt>new xy cells</dt><dd>{int(row.get("newly_observed_xy_cells", 0))}</dd>
        </dl>
      </article>"""
        )
    start_links = "\n".join(f'<a href="#start-{start:03d}-step-000">start {start:03d}</a>' for start in starts)
    video = f'<video controls src="{esc(mp4_rel)}"></video>' if mp4_rel else '<p class="muted">No MP4 was copied into this packet.</p>'
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LR-7 bounded long rollout action story</title>
  <style>
    body {{ margin:0; font-family: Inter, Segoe UI, Arial, sans-serif; background:#f5f7fa; color:#17202c; }}
    header.top {{ position:sticky; top:0; z-index:2; background:#17202c; color:white; padding:14px 18px; box-shadow:0 3px 12px rgba(0,0,0,.18); }}
    h1 {{ font-size:20px; margin:0 0 8px; }}
    nav {{ display:flex; flex-wrap:wrap; gap:6px; }}
    nav a {{ color:#e9f3ff; border:1px solid rgba(255,255,255,.32); text-decoration:none; padding:4px 8px; border-radius:6px; font-size:13px; }}
    main {{ max-width:1280px; margin:0 auto; padding:18px; }}
    .video {{ background:white; border:1px solid #d8dee8; border-radius:8px; padding:12px; margin-bottom:16px; }}
    video {{ width:100%; max-height:520px; background:#0b0f14; }}
    .card {{ background:white; border:1px solid #d8dee8; border-radius:8px; padding:12px; margin-bottom:14px; }}
    .card header {{ display:flex; justify-content:space-between; gap:12px; font-size:15px; border-bottom:1px solid #edf0f5; padding-bottom:8px; margin-bottom:10px; }}
    .media {{ display:grid; grid-template-columns:minmax(0,1.25fr) minmax(260px,.75fr); gap:12px; align-items:start; }}
    img {{ width:100%; height:auto; border:1px solid #d8dee8; border-radius:6px; background:#fff; }}
    dl {{ display:grid; grid-template-columns:150px minmax(0,1fr); gap:4px 10px; margin:10px 0 0; font-size:13px; }}
    dt {{ color:#5b6778; }}
    dd {{ margin:0; overflow-wrap:anywhere; }}
    .muted {{ color:#5b6778; }}
    @media (max-width: 820px) {{ .media {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header class="top">
    <h1>LR-7 bounded long rollout action story</h1>
    <nav>{start_links}</nav>
  </header>
  <main>
    <section class="video">{video}</section>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def generate(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_summary = generate_packet(
        argparse.Namespace(
            usd=args.usd,
            rollout_dir=args.rollout_dir,
            manifest=args.manifest,
            scene_metadata=args.scene_metadata,
            output_dir=args.output_dir,
            voxel_size=args.voxel_size,
        )
    )

    main_html = args.output_dir / "stage4a714_2d_rollout_review_index.html"
    main_alias = args.output_dir / "long_rollout_2d_review_index.html"
    if main_html.is_file():
        shutil.copy2(main_html, main_alias)

    records_path = args.output_dir / "stage4a714_2d_rollout_review_records.json"
    records = load_json(records_path)
    if not isinstance(records, list):
        records = []

    mp4_src = args.mp4
    if mp4_src is None:
        candidates = sorted(args.rollout_dir.glob("*.mp4"))
        mp4_src = candidates[0] if candidates else None
    mp4_dst = copy_if_present(mp4_src, args.output_dir / "long_rollout_flythrough.mp4") if mp4_src else None

    action_story = args.output_dir / "long_rollout_action_story_index.html"
    write_action_story(action_story, records, Path(mp4_dst).name if mp4_dst else None)

    instructions = args.output_dir / "stage_lr7_long_rollout_human_review_how_to_save.md"
    write_how_to_save(instructions, main_alias, action_story, mp4_dst)

    html_audit = {
        "stage": "LR-7 long rollout HTML/MP4 review audit",
        "generated_at_utc": now(),
        "main_html": str(main_alias),
        "legacy_html": str(main_html),
        "action_story_html": str(action_story),
        "mp4": mp4_dst,
        "review_records": str(records_path),
        "review_rows": len(records),
        "map_png_count": len(list((args.output_dir / "maps").glob("start_*_step_*_2d_review.png"))),
        "overview_png_count": len(list((args.output_dir / "maps").glob("start_*_overview_2d_review.png"))),
        "export_controls": ["Export review JSON", "Copy review JSON to clipboard", "Download review JSON", "Download review CSV"],
        "decision_controls": ["human_review_status", "human_review_reason", "promote_candidate_yes_no", "human_comment"],
        "all_passed": main_alias.is_file() and action_story.is_file() and bool(mp4_dst) and len(records) == 150,
    }
    summary = {
        "stage": "LR-7 bounded long rollout human review packet",
        "generated_at_utc": now(),
        "runtime_dir": str(args.rollout_dir),
        "output_dir": str(args.output_dir),
        "main_html": str(main_alias),
        "action_story_html": str(action_story),
        "mp4": mp4_dst,
        "instructions": str(instructions),
        "review_rows": len(records),
        "expected_review_rows": 150,
        "source_packet_summary": packet_summary,
        "html_audit": html_audit,
        "negative_scope": {
            "isaac_startup": False,
            "map_predict": False,
            "runtime_rollout": False,
            "training": False,
            "checkpoint": False,
            "label_promotion": False,
            "RL_GDPO_PPO": False,
        },
    }
    summary["all_passed"] = html_audit["all_passed"]
    summary["blocked"] = not summary["all_passed"]
    summary["main_blocker"] = "" if summary["all_passed"] else "lr7_review_packet_incomplete"

    write_json(args.output_dir / "long_rollout_html_review_audit.json", html_audit)
    write_md(args.output_dir / "long_rollout_html_review_audit.md", "LR-7 Long Rollout HTML Review Audit", html_audit)
    write_json(args.output_dir / "long_rollout_review_packet_summary.json", summary)
    write_md(args.output_dir / "long_rollout_review_packet_summary.md", "LR-7 Long Rollout Review Packet Summary", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--rollout_dir", type=Path, default=RUNTIME)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene_metadata", type=Path, default=DEFAULT_SCENE_METADATA)
    parser.add_argument("--output_dir", type=Path, default=OUT)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--mp4", type=Path, default=None)
    args = parser.parse_args()
    summary = generate(args)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
