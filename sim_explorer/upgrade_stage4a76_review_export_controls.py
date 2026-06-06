from __future__ import annotations

import csv
import hashlib
import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a76_stage4a72_manual_topdown_review_packet"
MAIN_HTML = OUT / "stage4a72_action_story_review_index.html"
SAMPLE_INDEX = OUT / "stage4a72_action_story_sample_index.json"

STATUS_OPTIONS = [
    "unreviewed",
    "approve",
    "reject",
    "unsure",
    "needs_closer_inspection",
]
REASON_OPTIONS = [
    "local_jitter_acceptable",
    "local_jitter_unacceptable",
    "unsafe_outside_stuck_revisit",
    "poor_uncertainty_choice",
    "poor_path_choice",
    "visual_mismatch",
    "good_exploration_choice",
    "other",
]
PROMOTE_OPTIONS = ["", "yes", "no"]
EXPORT_FIELDS = [
    "review_id",
    "start_id",
    "step_id",
    "sample_id",
    "human_review_status",
    "human_review_reason",
    "human_comment",
    "promote_candidate_yes_no",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def option_tags(options, default="") -> str:
    tags = []
    for value in options:
        selected = " selected" if value == default else ""
        label = "(empty)" if value == "" else value
        tags.append(f'<option value="{html.escape(value)}"{selected}>{html.escape(label)}</option>')
    return "\n".join(tags)


def rel(path: Path) -> str:
    return path.relative_to(OUT).as_posix()


def load_rows():
    sample_rows = read_json(SAMPLE_INDEX)
    rows = []
    for item in sample_rows:
        start_id = int(item["start_id"])
        step_id = int(item["step_id"])
        sample_id = item["sample_id"]
        sample_dir = OUT / "samples" / f"start_{start_id:03d}" / f"step_{step_id:03d}"
        review = read_json(sample_dir / "review_row.json")
        rows.append(
            {
                "review_id": review.get("review_id", sample_id),
                "start_id": start_id,
                "step_id": step_id,
                "sample_id": sample_id,
                "action_storyboard": item["action_storyboard"],
                "action_floorplan": item["action_floorplan"],
                "camera_transition": item["camera_transition"],
                "sample_review_card": item["sample_review_card"],
                "current_x": item.get("current_x"),
                "current_y": item.get("current_y"),
                "target_x": item.get("target_x"),
                "target_y": item.get("target_y"),
                "action_distance_m": item.get("action_distance_m"),
                "has_next_camera_after": item.get("has_next_camera_after"),
                "auto_quality_verdict": review.get("auto_quality_verdict", ""),
                "quality_warning": review.get("quality_warning", ""),
                "quality_blocker": review.get("quality_blocker", ""),
                "selected_confidence": review.get("selected_confidence"),
                "selected_entropy": review.get("selected_entropy"),
                "selected_margin": review.get("selected_margin"),
                "uncertainty_composite": review.get("uncertainty_composite"),
                "primary_vs_lambda48_distance": review.get("primary_vs_lambda48_distance"),
                "primary_vs_measured_distance": review.get("primary_vs_measured_distance"),
                "default_human_review_status": "unreviewed",
                "default_human_review_reason": "",
                "default_human_comment": "",
                "default_promote_candidate_yes_no": "",
            }
        )
    rows.sort(key=lambda r: (r["start_id"], r["step_id"]))
    return rows


def render_article(row: dict) -> str:
    status_options = option_tags(STATUS_OPTIONS, "unreviewed")
    reason_options = '<option value="" selected>(empty)</option>\n' + option_tags(REASON_OPTIONS)
    promote_options = option_tags(PROMOTE_OPTIONS, "")
    warning_class = "block" if row["quality_blocker"] else ("warn" if row["quality_warning"] else "pass")
    after = "yes" if row["has_next_camera_after"] else "no - final sampled step"
    return f"""
<article class="action {warning_class}" id="{html.escape(row['sample_id'])}" data-review-card="true" data-review-id="{html.escape(row['review_id'])}" data-start-id="{row['start_id']}" data-step-id="{row['step_id']}" data-sample-id="{html.escape(row['sample_id'])}">
  <div class="card-head">
    <h3>{html.escape(row['sample_id'])}</h3>
    <span class="badge">{html.escape(row['auto_quality_verdict'] or 'unknown')}</span>
  </div>
  <div class="meta">
    <div><strong>movement</strong><br>({row['current_x']:.2f},{row['current_y']:.2f}) -> ({row['target_x']:.2f},{row['target_y']:.2f})</div>
    <div><strong>distance</strong><br>{row['action_distance_m']:.2f} m</div>
    <div><strong>after camera</strong><br>{html.escape(after)}</div>
    <div><strong>confidence / entropy / margin</strong><br>{row['selected_confidence']} / {row['selected_entropy']} / {row['selected_margin']}</div>
    <div><strong>uncertainty composite</strong><br>{row['uncertainty_composite']}</div>
    <div><strong>vs lambda48 shadow</strong><br>{row['primary_vs_lambda48_distance']}</div>
    <div><strong>vs measured shadow</strong><br>{row['primary_vs_measured_distance']}</div>
    <div><strong>warning / blocker</strong><br>{html.escape(row['quality_warning'] or 'none')} / {html.escape(row['quality_blocker'] or 'none')}</div>
  </div>
  <section class="review-controls" aria-label="manual review controls">
    <label>human_review_status
      <select name="human_review_status" data-field="human_review_status">
        {status_options}
      </select>
    </label>
    <label>human_review_reason
      <select name="human_review_reason" data-field="human_review_reason">
        {reason_options}
      </select>
    </label>
    <label>promote_candidate_yes_no
      <select name="promote_candidate_yes_no" data-field="promote_candidate_yes_no">
        {promote_options}
      </select>
    </label>
    <label class="comment-label">human_comment
      <textarea name="human_comment" data-field="human_comment" rows="3" placeholder="Optional note for Stage 4A-7.7 import"></textarea>
    </label>
    <div class="row-validation" data-row-validation></div>
  </section>
  <img class="story" src="{html.escape(row['action_storyboard'])}" loading="lazy" alt="action review storyboard for {html.escape(row['sample_id'])}">
  <p class="links">
    <a href="{html.escape(row['action_floorplan'])}">action floorplan</a>
    <a href="{html.escape(row['camera_transition'])}">camera transition</a>
    <a href="{html.escape(row['sample_review_card'])}">original card/form</a>
  </p>
</article>
"""


def render_html(rows: list[dict]) -> str:
    rows_json = json.dumps(
        [
            {
                "review_id": row["review_id"],
                "start_id": row["start_id"],
                "step_id": row["step_id"],
                "sample_id": row["sample_id"],
            }
            for row in rows
        ],
        sort_keys=True,
    )
    starts = {}
    for row in rows:
        starts.setdefault(row["start_id"], []).append(row)

    start_sections = []
    for start_id in sorted(starts):
        start_sections.append(
            f"""
<section class="start" data-start-id="{start_id}">
  <h2>Start {start_id:03d}</h2>
  <p class="start-images">
    <img src="samples/start_{start_id:03d}/start_path_floorplan.png" loading="lazy" alt="start path floorplan">
    <img src="samples/start_{start_id:03d}/start_action_sequence_floorplan.png" loading="lazy" alt="start action sequence floorplan">
  </p>
  {''.join(render_article(row) for row in starts[start_id])}
</section>
"""
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 4A-7.6c Manual Review Export Controls</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#f8fafc;color:#0f172a}}
header{{position:sticky;top:0;background:#0f172a;color:white;padding:14px 22px;z-index:10}}
main{{max-width:1320px;margin:0 auto;padding:20px}}
.notice,.instruction-panel{{background:#fff7ed;border:1px solid #fdba74;padding:12px;margin:14px 0}}
.instruction-panel{{background:#ecfeff;border-color:#67e8f9}}
.toolbar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:#e2e8f0;padding:12px;margin:12px 0;border:1px solid #cbd5e1}}
.toolbar input{{padding:8px;border:1px solid #94a3b8;border-radius:4px}}
button{{padding:8px 10px;border:1px solid #475569;background:#fff;border-radius:4px;cursor:pointer}}
button.primary{{background:#0f172a;color:white}}
button:disabled{{opacity:.55;cursor:not-allowed}}
.summary-box{{white-space:pre-wrap;background:#f1f5f9;border:1px solid #cbd5e1;padding:10px;margin:10px 0;display:none}}
.start{{margin:22px 0;padding-top:6px;border-top:3px solid #cbd5e1}}
.start-images img{{max-width:460px;border:1px solid #cbd5e1;background:white;margin-right:8px}}
.action{{background:white;border:1px solid #cbd5e1;border-radius:8px;margin:16px 0;padding:14px;box-shadow:0 1px 2px #0001}}
.card-head{{display:flex;justify-content:space-between;gap:12px;align-items:center}}
.card-head h3{{margin:4px 0}}
.badge{{font-size:12px;background:#e2e8f0;padding:4px 8px;border-radius:4px}}
.story{{width:100%;max-width:1240px;border:1px solid #cbd5e1;background:white}}
.meta{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:10px 0}}
.meta div{{background:#f1f5f9;padding:8px;border-radius:6px;font-size:13px}}
.review-controls{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:12px 0;padding:12px;background:#f8fafc;border:1px solid #cbd5e1}}
.review-controls label{{display:flex;flex-direction:column;gap:4px;font-size:13px;font-weight:600}}
.review-controls select,.review-controls textarea{{font:inherit;padding:8px;border:1px solid #94a3b8;border-radius:4px;background:white}}
.comment-label{{grid-column:1 / -1}}
.row-validation{{grid-column:1 / -1;color:#b91c1c;font-weight:600;min-height:18px}}
.links a{{margin-right:12px}}
.warn{{border-left:6px solid #f59e0b}}.pass{{border-left:6px solid #16a34a}}.block{{border-left:6px solid #dc2626}}
code{{background:#e2e8f0;padding:1px 4px;border-radius:4px}}
@media (max-width: 860px){{.meta,.review-controls{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header><strong>Stage 4A-7.6c Manual Review Export Controls</strong> | Stage 4A-7.2 candidate review only</header>
<main>
<section class="instruction-panel">
  <h1>Manual Review Workflow</h1>
  <p>Review each action storyboard, then set <code>human_review_status</code>, optional <code>human_review_reason</code>, optional <code>promote_candidate_yes_no</code>, and <code>human_comment</code>. Use the export buttons to save JSON/CSV for future Stage 4A-7.7 import.</p>
  <p><strong>Warning:</strong> Human approval does not automatically promote labels. Future Stage 4A-7.7 must import this review and make a separate promotion decision.</p>
  <p>Validation rule: <code>promote_candidate_yes_no=yes</code> is only allowed when <code>human_review_status=approve</code>. Reject, unsure, and needs_closer_inspection cannot be exported with promote=yes.</p>
</section>
<section class="toolbar" aria-label="review export toolbar">
  <label>reviewer_name <input id="reviewer-name" type="text" placeholder="optional"></label>
  <button id="mark-all-unreviewed" type="button">Mark all unreviewed</button>
  <button id="export-review-json" class="primary" type="button">Export review JSON</button>
  <button id="copy-review-json" type="button">Copy review JSON to clipboard</button>
  <button id="download-review-csv" type="button">Download review CSV</button>
  <button id="show-review-summary" type="button">Show review completion summary</button>
  <a href="stage4a72_human_review_how_to_save.md">offline save instructions</a>
</section>
<div id="review-summary" class="summary-box" aria-live="polite"></div>
<div class="notice"><strong>Safety:</strong> offline review/export only. No Isaac startup, no capture, no map_predict, no rollout, no BC training, no checkpoint, no label promotion, no RL/GDPO/PPO. Primary action source remains <code>stage4a613_uncertainty_bonus_executed_primary</code> / <code>uncertainty_bonus_composite_beta8</code>; lambda48 remains shadow/baseline only.</div>
<p><a href="stage4a72_manual_review_template.csv">original review CSV template</a> | <a href="stage4a72_action_story_sample_index.csv">action story sample index CSV</a> | <a href="stage4a72_topdown_review_index.html">original 7.6 index</a></p>
{''.join(start_sections)}
</main>
<script>
const REVIEW_SCHEMA_VERSION = "stage4a76c_manual_review_export_v1";
const REVIEW_BASE_ROWS = {rows_json};
const EXPORT_FIELDS = {json.dumps(EXPORT_FIELDS)};

function cards() {{
  return Array.from(document.querySelectorAll('[data-review-card="true"]'));
}}

function getField(card, field) {{
  return card.querySelector(`[data-field="${{field}}"]`);
}}

function rowFromCard(card) {{
  return {{
    review_id: card.dataset.reviewId,
    start_id: Number(card.dataset.startId),
    step_id: Number(card.dataset.stepId),
    sample_id: card.dataset.sampleId,
    human_review_status: getField(card, 'human_review_status').value || 'unreviewed',
    human_review_reason: getField(card, 'human_review_reason').value || '',
    human_comment: getField(card, 'human_comment').value || '',
    promote_candidate_yes_no: getField(card, 'promote_candidate_yes_no').value || ''
  }};
}}

function validateRow(row) {{
  if (row.promote_candidate_yes_no === 'yes' && row.human_review_status !== 'approve') {{
    return 'promote_candidate_yes_no=yes is only allowed when human_review_status=approve';
  }}
  return '';
}}

function validateAllRows() {{
  let errors = [];
  cards().forEach(card => {{
    const row = rowFromCard(card);
    const msg = validateRow(row);
    const target = card.querySelector('[data-row-validation]');
    target.textContent = msg;
    if (msg) errors.push(`${{row.review_id}}: ${{msg}}`);
  }});
  return errors;
}}

function buildExportPayload() {{
  const rows = cards().map(rowFromCard);
  return {{
    review_schema_version: REVIEW_SCHEMA_VERSION,
    source_stage: "Stage 4A-7.2",
    packet_stage: "Stage 4A-7.6",
    exported_at_local_browser_time: new Date().toISOString(),
    reviewer_name: document.getElementById('reviewer-name').value || '',
    rows
  }};
}}

function csvEscape(value) {{
  const text = String(value ?? '');
  if (/[",\\n\\r]/.test(text)) return '"' + text.replaceAll('"', '""') + '"';
  return text;
}}

function buildCsv(rows) {{
  const header = EXPORT_FIELDS.join(',');
  const lines = rows.map(row => EXPORT_FIELDS.map(field => csvEscape(row[field])).join(','));
  return [header, ...lines].join('\\n') + '\\n';
}}

function downloadText(filename, text, mimeType) {{
  const blob = new Blob([text], {{type: mimeType}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

function requireValidRows() {{
  const errors = validateAllRows();
  if (errors.length) {{
    alert('Review validation failed:\\n' + errors.join('\\n'));
    return false;
  }}
  return true;
}}

function exportJson() {{
  if (!requireValidRows()) return;
  const payload = buildExportPayload();
  downloadText('stage4a72_manual_review_export.json', JSON.stringify(payload, null, 2), 'application/json');
}}

async function copyJson() {{
  if (!requireValidRows()) return;
  const payload = buildExportPayload();
  const text = JSON.stringify(payload, null, 2);
  try {{
    await navigator.clipboard.writeText(text);
    alert('Review JSON copied to clipboard.');
  }} catch (err) {{
    const box = document.getElementById('review-summary');
    box.style.display = 'block';
    box.textContent = 'Clipboard copy failed. Select and copy this JSON manually:\\n\\n' + text;
  }}
}}

function downloadCsv() {{
  if (!requireValidRows()) return;
  const payload = buildExportPayload();
  downloadText('stage4a72_manual_review_export.csv', buildCsv(payload.rows), 'text/csv');
}}

function markAllUnreviewed() {{
  cards().forEach(card => {{
    getField(card, 'human_review_status').value = 'unreviewed';
    getField(card, 'human_review_reason').value = '';
    getField(card, 'human_comment').value = '';
    getField(card, 'promote_candidate_yes_no').value = '';
    card.querySelector('[data-row-validation]').textContent = '';
  }});
  showSummary();
}}

function showSummary() {{
  const rows = cards().map(rowFromCard);
  const counts = {{
    total: rows.length,
    unreviewed: 0,
    approve: 0,
    reject: 0,
    unsure: 0,
    needs_closer_inspection: 0,
    promote_yes: 0,
    invalid: 0
  }};
  rows.forEach(row => {{
    counts[row.human_review_status] = (counts[row.human_review_status] || 0) + 1;
    if (row.promote_candidate_yes_no === 'yes') counts.promote_yes += 1;
    if (validateRow(row)) counts.invalid += 1;
  }});
  const box = document.getElementById('review-summary');
  box.style.display = 'block';
  box.textContent = [
    `total rows: ${{counts.total}}`,
    `unreviewed: ${{counts.unreviewed}}`,
    `approve: ${{counts.approve}}`,
    `reject: ${{counts.reject}}`,
    `unsure: ${{counts.unsure}}`,
    `needs_closer_inspection: ${{counts.needs_closer_inspection}}`,
    `promote_candidate_yes_no=yes: ${{counts.promote_yes}}`,
    `invalid rows: ${{counts.invalid}}`
  ].join('\\n');
  validateAllRows();
}}

function enforcePromoteRule(card) {{
  const status = getField(card, 'human_review_status');
  const promote = getField(card, 'promote_candidate_yes_no');
  if (promote.value === 'yes' && status.value !== 'approve') {{
    promote.value = '';
    card.querySelector('[data-row-validation]').textContent = 'promote_candidate_yes_no=yes requires approve; value reset to empty.';
  }} else {{
    card.querySelector('[data-row-validation]').textContent = validateRow(rowFromCard(card));
  }}
}}

document.getElementById('mark-all-unreviewed').addEventListener('click', markAllUnreviewed);
document.getElementById('export-review-json').addEventListener('click', exportJson);
document.getElementById('copy-review-json').addEventListener('click', copyJson);
document.getElementById('download-review-csv').addEventListener('click', downloadCsv);
document.getElementById('show-review-summary').addEventListener('click', showSummary);
cards().forEach(card => {{
  ['human_review_status', 'promote_candidate_yes_no'].forEach(field => {{
    getField(card, field).addEventListener('change', () => enforcePromoteRule(card));
  }});
}});
showSummary();
</script>
</body>
</html>
"""


def write_instruction_file(rows: list[dict]) -> None:
    text = f"""# Stage 4A-7.2 Human Review: How To Save

Open:

`{MAIN_HTML}`

Review each of the {len(rows)} action cards.

For every row:

1. Set `human_review_status` to `approve`, `reject`, `unsure`, or `needs_closer_inspection`.
2. Optionally set `human_review_reason`.
3. Optionally set `promote_candidate_yes_no`.
4. Add `human_comment` if helpful.

Important: `promote_candidate_yes_no=yes` is only valid when `human_review_status=approve`.

Use one of the export controls at the top of the page:

- `Export review JSON` downloads `stage4a72_manual_review_export.json`.
- `Copy review JSON to clipboard` copies the same JSON payload.
- `Download review CSV` downloads `stage4a72_manual_review_export.csv`.
- `Show review completion summary` displays counts and invalid rows.
- `Mark all unreviewed` resets all human fields to default.

Human approval does not automatically promote labels. Future Stage 4A-7.7 must import the exported JSON or CSV and make a separate promotion decision.

No promotion, training, checkpoint, Isaac runtime, map_predict, rollout, or RL/GDPO/PPO is performed by this review page.
"""
    (OUT / "stage4a72_human_review_how_to_save.md").write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def refresh_manifest() -> None:
    artifacts = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": rel(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(OUT / "artifact_manifest.json", {"artifact_count": len(artifacts), "artifacts": artifacts})
    lines = ["# Artifact Manifest", "", f"artifact_count: {len(artifacts)}", ""]
    for item in artifacts:
        lines.append(f"- `{item['path']}` ({item['size_bytes']} bytes)")
    (OUT / "artifact_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 rows, got {len(rows)}")
    MAIN_HTML.write_text(render_html(rows), encoding="utf-8")
    write_instruction_file(rows)
    report = {
        "completed": True,
        "blocked": False,
        "stage": "Stage 4A-7.6c",
        "main_html": str(MAIN_HTML),
        "instruction_file": str(OUT / "stage4a72_human_review_how_to_save.md"),
        "review_cards": len(rows),
        "default_human_review_status": "unreviewed",
        "default_human_review_reason": "",
        "default_human_comment": "",
        "default_promote_candidate_yes_no": "",
        "export_json_filename": "stage4a72_manual_review_export.json",
        "export_csv_filename": "stage4a72_manual_review_export.csv",
        "label_promotion": False,
        "training": False,
        "checkpoint": False,
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "rl_gdpo_ppo": False,
        "lambda48_role": "shadow/baseline only",
        "expert_action_index_primary_created_from_stage4a72": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "stage4a76c_review_export_controls_report.json", report)
    md = ["# Stage 4A-7.6c Review Export Controls Report", ""]
    for key, value in report.items():
        md.append(f"- {key}: `{value}`" if isinstance(value, str) else f"- {key}: {value}")
    (OUT / "stage4a76c_review_export_controls_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    refresh_manifest()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
