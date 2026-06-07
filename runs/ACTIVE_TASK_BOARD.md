# Active Task Board

- PI_WORKSPACE: /home/ubuntu22/pi
- current_phase: Phase 8 primary-scene mapping smoke passed
- current_goal: Prepare Phase 9 candidate viewpoint + information gain smoke
- latest_phase8_run_dir: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019
- latest_primary_scene: /home/ubuntu22/pi/scenes/primary_building_scene_repaired/home_like_scene_v1.usd
- scene_bundle_size: 490M
- latest_decision: proceed_to_phase_9_candidate_viewpoint_information_gain_smoke

| Agent | Status | Task | Notes |
| --- | --- | --- | --- |
| Main Coordinator | done | Validate Phase 8 bounded mapping smoke | No training/RL/PI fine-tuning/checkpoint/long rollout. |
| Git Safety | done | Confirm primary scene bundle stays local-only | Scene bundle ignored and not tracked. |
| Isaac Stage Probe | done | Headless open repaired primary scene | open_stage_result=true, prim_count=1324. |
| Mapping Smoke | done | Update BEV partial map from 10 short poses | final_known_ratio=0.48057726, occupied/free/unknown all non-zero. |
| Review Visualization | done | Generate curves and final BEV/trace plots | 4 PNG plots plus ASCII map fallback saved locally. |
| Phase 9 Candidate Gain Smoke | ready | Validate candidate viewpoint + information gain next | Bounded smoke only. |
| Primary Long Rollout | blocked | Wait for remaining smoke gates | Not started. |
| Training/RL/PI | forbidden | No training, RL, checkpoint, or PI/openpi/VLM fine-tuning | Remains false. |
