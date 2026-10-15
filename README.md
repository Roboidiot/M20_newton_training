# M20_newton_training

An installable downstream Isaac Lab task package generated with a standard uv `src` layout.

The registered tasks are:

- `M20NewtonTraining-M20-M20-Flat`
- `M20NewtonTraining-M20-M20-Rough`
- `M20NewtonTraining-M20-M20-Sand`
- `M20NewtonTraining-M20-M20-Stand`

The M20 wheeled robot model lives in `deep_robotics_model/M20/` (USD, URDF, and MJCF), referenced by the
`DEEPROBOTICS_M20_CFG` articulation in `src/M20_newton_training/assets/deeprobotics.py`.

`M20NewtonTraining-M20-M20-Sand` puts each environment on a granular bed advanced by Newton's implicit MPM
solver, coupled to the rigid robot solve. It therefore requires the Newton backend: the `newton_sand` physics
preset is its default, and the PhysX presets leave the sand frozen.

`M20NewtonTraining-M20-M20-Stand` trains upright locomotion with the robot's base +X axis pointing up.
It uses only random rough roads, with a fixed height-noise sampling range of 1--16 cm and no terrain
difficulty curriculum. Road layouts remain randomized within that range. The command slots
`[vx, vy, wz]` map to body `[linear Y, linear Z, angular X]`, with ranges `[-1, 1] m/s`, `[-0.5, 0.5] m/s`, and
`[-1, 1] rad/s`. Sampling reserves 20% for zero commands, 2% each for single-axis translation, and 20% for pure
turning. The policy is blind to terrain heights; the critic receives the height scan. Height rewards target
0.8028 m above the local ground. The default PPO budget is 20,000 iterations.

Stand's `joint_mirror` term penalizes left/right front-leg asymmetry with weight `-1.0`, including
during motion and turning. It uses `(hipx_left + hipx_right)^2 + (hipy_left - hipy_right)^2 +
(knee_left - knee_right)^2`, without a gait-level or gravity gate. Rear legs and wheel joints are
excluded from this term. Symmetry does not fix the front legs to their default joint angles.

The shared `joint_pos_rel_without_wheel` observation now masks wheel angles in the articulation's native
joint order before selecting/reordering observation joints. This preserves all leg-angle observations;
the actor's wheel-angle slots are 12--15. Observation and action dimensions are unchanged. Checkpoints
trained before this fix used an incorrect actor mask on Newton (slots 3, 7, 11, 15), so resuming them
changes their observation inputs and requires adaptation and validation. For MuJoCo replay, use the
matching `joint_position_zero_indices` in `mojuco_sim/stand_config.json`; its default retains the old mask
for the existing exported policy. Policies trained with the fix require `[12, 13, 14, 15]`.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project was generated from an Isaac Lab source checkout. Its `pyproject.toml` uses editable relative paths to that
checkout, so regenerate the project or update `[tool.uv.sources]` if either directory moves.

Optional backends are available through extras. Pass the extra to each `uv run` command that needs it:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx
```

The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes. You can also select `ovrtx` independently when using
Newton physics with the OVRTX renderer. Commit `pyproject.toml` and `uv.lock` so collaborators use the same environment.

## Run the generated tasks

Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run python scripts/list_envs.py --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `list_envs.py`.

### Stand keyboard playback

Stand's play mode replaces the random command sampler with keyboard input from a Newton window:

```bash
uv run isaaclab play --rl_library rsl_rl --task M20NewtonTraining-M20-M20-Stand \
  --checkpoint latest physics=newton_mjwarp --visualizer newton --num_envs 1 --real-time
```

Click the Newton window to focus it. Hold `I/K` for positive/negative `vx` (body Y, 0.5 m/s),
`J/L` for positive/negative `vy` (body Z, 0.25 m/s), and `U/O` for positive/negative `wz`
(body X, 0.5 rad/s). Releasing a key stops that command axis; holding Backspace forces all commands
to zero. These speeds are half of each configured command range's maximum magnitude. Space pauses
the viewer, and WASD/arrow keys retain their camera controls. Closing the window exits playback.
For automatic random commands during play, add `--train_env_cfg` to bypass the play-mode switch.
The training command generator and policy input dimensions are unchanged.

## Project structure

Task families live in `src/M20_newton_training/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/M20`. Add another robot by
creating a sibling of `config/M20`; add another task by creating a sibling task-family directory.

## Development

Run the generated registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and generates a git-ignored `pyrightconfig.json`. The generated
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
