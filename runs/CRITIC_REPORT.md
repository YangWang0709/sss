# Critic Report

- phase: Phase 6 primary USD repair decision
- result: replacement_primary_scene_created
- scope_check: passed
- original_USD_deleted: false
- original_USD_overwritten: false
- training: false
- RL: false
- checkpoint: false
- PI_finetuning: false

## Conclusion

The original /home/ubuntu22/pi/building_scene.usd remains preserved but is not usable in Isaac/Kit: the open_stage probe still aborts with exit code 134 inside the USD crate read path. The environment also lacks standalone USD repair tooling, so a safe clean/flatten/sanitize conversion could not be produced.

A replacement temporary primary scene was created at /home/ubuntu22/pi/scenes/primary_indoor_scene.usda and passed Isaac headless open_stage with prim_count=52. This is now the recommended scene for the next primary-scene sensor smoke gate.

## Gates

- proceed_to_primary_scene_sensor_smoke: true
- proceed_to_primary_mapping_smoke: false until sensor smoke passes
- proceed_to_primary_candidate_gain_smoke: false until mapping smoke passes
- proceed_to_primary_closed_loop_smoke: false until candidate gain smoke passes
- proceed_to_primary_long_rollout: false until all smoke gates pass
- proceed_to_training: false
- proceed_to_RL: false
- proceed_to_PI_finetuning: false

## Risks

- The replacement scene is procedural and simpler than a production building asset, although it is more complex than the fallback smoke scene.
- The original USDC still needs manual rebuild, external USD tooling, or replacement by a production-ready primary USD if fidelity becomes important.
- Fallback Phase 5 data remains schema/pipeline sanity data only and is not training data.
