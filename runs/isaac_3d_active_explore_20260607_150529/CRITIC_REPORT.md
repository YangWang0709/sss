# Critic Report

- phase: Phase 5 fallback long rollout
- result: fallback pipeline passed; waiting for human review
- scope_check: passed
- drift_to_training_or_RL: false
- PI_workspace_used: true
- original_USD_deleted: false
- primary_usd_used: false
- fallback_scene: scenes/minimal_indoor_smoke.usda

## Conclusion

- The fallback pipeline passed through Phase 5.
- The fallback rollout packet is suitable for schema and pipeline sanity review.
- This data must not be treated as final primary-scene training data.
- The original /home/ubuntu22/pi/building_scene.usd is still preserved, but remains blocked because Isaac/Omniverse crashes while opening it.
- The original building_scene.usd still needs repair, conversion, or replacement before final primary-scene rollout collection.
- Current state is waiting for human review.

## Phase 5 Metrics

- start_count: 10
- total_action_count: 143
- total_step_rows: 149
- candidate_rows: 3576
- average_final_known_ratio: 0.8947
- starts_with_failures: 6
- primary_usd_used: false

## Negative Scope

- training: false
- RL: false
- checkpoint: false
- PI fine-tuning: false
- openpi fine-tuning: false
- primary-scene label promotion: false

## Required Human Review

- Inspect trajectory continuity per start.
- Inspect candidate validity and selected viewpoint behavior.
- Confirm known_ratio growth.
- Review 6 failure starts and decide whether failures are acceptable.
- Check for repeated same-point selection, wall collision, unreachable targets, or local spinning.
- Compare candidate score against actual coverage gain.

## Gate

- proceed_to_training: false
- proceed_to_RL: false
- proceed_to_PI_finetuning: false
- proceed_to_primary_scene_collection: false until building_scene.usd is repaired/replaced and a primary-scene rollout packet is collected.
