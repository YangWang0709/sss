# Active Task Board

- PI_WORKSPACE: /home/ubuntu22/pi
- current_phase: Phase 1 robot pose smoke failed
- current_goal: Isaac Sim 3D active exploration long rollout data collection closed loop
- latest_run_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529
- latest_commit: 361ba9f phase 0: clean USD blocker reports

| Agent | Status | Task | Notes |
| --- | --- | --- | --- |
| Main Coordinator | done | Confirm PI workspace and phase order | Do not jump to training/RL. |
| SSH Experiment | done | Audit Ubuntu, conda, GPU, git, workspace | See ENVIRONMENT_AUDIT.md. |
| IsaacSim Scene and Sensor | running | RGB-D camera smoke passed; robot pose and LiDAR smoke next | Primary USD still crashes; fallback scene is usable. |
| Mapping and Viewpoint | pending | Partial map and candidate gain | Wait for scene and sensor smoke. |
| Rollout Data | pending | Long rollout schema and collection | Wait for Phase 1-4. |
| WebGPT Liaison | ready | Prepare web brief | WEBGPT_BRIEF.md updated for blocker. |
| GitHub Sync | running | Push Phase 0 audit to YangWang0709/sss | Exclude large data. |
| Critic and Context Manager | done | Review Phase 0 scope and blocker | Minimal fallback USD is allowed next fix. |
