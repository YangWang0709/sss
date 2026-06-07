# Active Task Board

- PI_WORKSPACE: /home/ubuntu22/pi
- current_phase: Phase 0 complete; Phase 1 pending
- current_goal: Isaac Sim 3D active exploration long rollout data collection closed loop
- latest_run_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529
- latest_commit: be5924e phase 0: audit pi workspace and env_isaaclab

| Agent | Status | Task | Notes |
| --- | --- | --- | --- |
| Main Coordinator | done | Confirm PI workspace and phase order | Do not jump to training/RL. |
| SSH Experiment | done | Audit Ubuntu, conda, GPU, git, workspace | See ENVIRONMENT_AUDIT.md. |
| IsaacSim Scene & Sensor | pending | Build/run Phase 1 USD sensor smoke test | Isaac headless is available; smoke script still needed. |
| Mapping & Viewpoint | pending | Partial map and candidate gain | Wait for Phase 1 observations. |
| Rollout Data | pending | Long rollout schema and collection | Wait for Phase 1-4. |
| WebGPT Liaison | pending | Prepare web brief | WEBGPT_BRIEF.md updated for Phase 0. |
| GitHub Sync | running | Commit Phase 0 reports to YangWang0709/sss | Exclude large data. |
| Critic & Context Manager | done | Review Phase 0 scope | Phase 0 stays on 3D active exploration only. |
