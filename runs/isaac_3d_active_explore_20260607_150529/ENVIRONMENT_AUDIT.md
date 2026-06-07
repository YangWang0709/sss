# Phase 0 Environment Audit

- phase: Phase 0 environment and project audit
- generated_at: 2026-06-07T15:07:21+08:00
- host: ubuntu22
- PI_WORKSPACE: /home/ubuntu22/pi
- run_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529
- conda_env: env_isaaclab
- env_activation_command: `source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab`
- git_root: /home/ubuntu22/pi
- remote_origin: git@github.com:YangWang0709/sss.git

## Summary

| Check | Result |
| --- | --- |
| PI_WORKSPACE exists | yes |
| env_isaaclab activation command found | yes |
| GPU command nvidia-smi ran | yes |
| USD asset count | 1 |
| Primary USD | /home/ubuntu22/pi/building_scene.usd |
| USD opened through pxr.Usd | false |
| Python/source file count | 0 |
| isaacsim module found | true |
| omni module found | false |
| pxr module found | false |
| Isaac headless SimulationApp started | true |
| Files over 50MB | 0 |
| Phase 0 pass | false |

## Current Project Capability

- Existing USD scene: /home/ubuntu22/pi/building_scene.usd
- Existing robot/sensor/mapping/rollout code: not found in PI_WORKSPACE yet
- Current shortest path: keep the USD scene, add a minimal Isaac USD/sensor smoke-test script in Phase 1, then add mapping, candidate viewpoints, and closed-loop rollout phase by phase.

## Key Logs

- initial audit log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/logs/phase0_command_outputs.log
- corrected probe log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/logs/phase0_command_outputs_fix.log

## Constraints Confirmed

- No PI/openpi/VLM training was run.
- No RL was run.
- No checkpoint was created.
- Original USD scene was not deleted.
- Large generated data is excluded by .gitignore patterns.
