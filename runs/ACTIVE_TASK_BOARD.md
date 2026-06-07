# Active Task Board

- PI_WORKSPACE: /home/ubuntu22/pi
- current_phase: Phase 6 primary USD repair decision complete
- current_goal: Use a stable primary indoor scene for primary-scene smoke gates before any rollout/training
- latest_repair_run_dir: /home/ubuntu22/pi/runs/usd_repair_building_scene_20260607_162058
- latest_primary_scene: /home/ubuntu22/pi/scenes/primary_indoor_scene.usda
- latest_decision: replacement_primary_scene_created

| Agent | Status | Task | Notes |
| --- | --- | --- | --- |
| Main Coordinator | done | Preserve original USD and run repair decision | No training/RL/PI fine-tuning. |
| USD Repair | done | Diagnose building_scene.usd core dump | Original still aborts with exit code 134. |
| USD Tooling | blocked | usdchecker/usdcat/Python pxr conversion | Standalone tools/pxr unavailable in env_isaaclab. |
| Replacement Scene | done | Create stable temporary primary indoor USDA | /home/ubuntu22/pi/scenes/primary_indoor_scene.usda opens headless. |
| Isaac Probe | done | Headless open replacement primary scene | prim_count=52, cube_count=37. |
| Primary Smoke Pipeline | ready | Run sensor smoke next | Do not run long rollout until smoke gates pass. |
| Training/RL/PI | forbidden | No training, RL, checkpoint, or PI/openpi/VLM fine-tuning | Remains false. |
