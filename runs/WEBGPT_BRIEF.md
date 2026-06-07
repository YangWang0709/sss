# WebGPT Brief

## Current Phase

Phase 6 primary USD repair decision complete.

## Decision

- conclusion: replacement_primary_scene_created
- original primary USD: /home/ubuntu22/pi/building_scene.usd
- original sha256: 11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b
- original status: preserved, not deleted, not overwritten
- original Isaac open_stage: still aborts with exit code 134
- final usable temporary primary scene: /home/ubuntu22/pi/scenes/primary_indoor_scene.usda
- temporary primary scene headless open: passed
- ready_for_primary_scene_sensor_smoke: true
- ready_for_primary_long_rollout: false
- training: false
- RL: false
- checkpoint: false
- PI/openpi/VLM fine-tuning: false

## Why Replacement Was Needed

- usdchecker/usdcat/usdzip/usdedit are not available in PATH.
- env_isaaclab does not expose standalone pxr for non-Isaac USD API repair.
- original building_scene.usd still aborts inside USD crate / SdfLayer loading when opened by Isaac/Kit.
- clean/flattened/sanitized variants could not be produced safely from the original USDC.

## Replacement Scene

- path: /home/ubuntu22/pi/scenes/primary_indoor_scene.usda
- sha256: a52f6fedb5393c8fd6001d5ec71643228d47b48bf77883aca9b0a1e15728453d
- prim_count: 52
- cube_count: 37
- up_axis: Z
- meters_per_unit: 1.0
- structure: multi-room indoor layout with two corridor axes, door gaps, walls, floor, ceiling, and obstacle/furniture proxy cubes.

## Reports

- decision: /home/ubuntu22/pi/runs/USD_REPAIR_DECISION.md
- repair report: /home/ubuntu22/pi/runs/usd_repair_building_scene_20260607_162058/USD_REPAIR_REPORT.md
- commands: /home/ubuntu22/pi/runs/usd_repair_building_scene_20260607_162058/USD_REPAIR_COMMANDS.sh
- original probe: /home/ubuntu22/pi/runs/usd_repair_building_scene_20260607_162058/probes/isaac_open_original.json
- replacement probe: /home/ubuntu22/pi/runs/usd_repair_building_scene_20260607_162058/probes/isaac_open_primary_indoor_scene.json

## Next Step

Run primary-scene sensor smoke on /home/ubuntu22/pi/scenes/primary_indoor_scene.usda. Do not run primary long rollout until sensor smoke, mapping smoke, candidate gain smoke, and closed-loop smoke pass.
