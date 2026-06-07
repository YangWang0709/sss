# Active Task Board

- PI_WORKSPACE: /home/ubuntu22/pi
- current_phase: Phase 0 audit complete; USD load blocker active
- current_goal: Isaac Sim 3D active exploration long rollout data collection closed loop
- latest_run_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529
- latest_commit: fac1622 phase 0: document USD load blocker

| Agent | Status | Task | Notes |
| --- | --- | --- | --- |
| Main Coordinator | done | Confirm PI workspace and phase order | Do not jump to training/RL. |
| SSH Experiment | done | Audit Ubuntu, conda, GPU, git, workspace | See ENVIRONMENT_AUDIT.md. |
| IsaacSim Scene & Sensor | blocked | Validate scene load before Phase 1 | Primary USD crashes on open_stage. |
| Mapping & Viewpoint | pending | Partial map and candidate gain | Wait for scene/sensor smoke. |
| Rollout Data | pending | Long rollout schema and collection | Wait for Phase 1-4. |
| WebGPT Liaison | ready | Prepare web brief | WEBGPT_BRIEF.md updated for blocker. |
| GitHub Sync | running | Push Phase 0 audit to YangWang0709/sss | Exclude large data. |
| Critic & Context Manager | done | Review Phase 0 scope and blocker | Minimal fallback USD is allowed next fix. |
