#!/usr/bin/env bash
# Collect the group2 actor on real MPM; trailing CLI arguments override these defaults.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec .venv/bin/python -m sand_pinn_lab.collect \
  --checkpoint logs/rsl_rl/deeprobotics_m20_walk_on_sand_proxy/2026-11-07_21-23-23_198a52_02_group_02/model_12500.pt \
  --command 0.2 0.0 0.0 \
  --episodes 20 \
  --episode-seconds 10 \
  "$@"
