# M20_newton_training

An installable downstream Isaac Lab task package generated with a standard uv `src` layout.

The registered tasks are:

- `M20NewtonTraining-M20-M20-Flat`
- `M20NewtonTraining-M20-M20-Walk`
- `M20NewtonTraining-M20-M20-WalkOnSand`
- `M20NewtonTraining-M20-M20-WalkOnSandProxy`
- `M20NewtonTraining-M20-M20-Rough`
- `M20NewtonTraining-M20-M20-Sand`
- `M20NewtonTraining-M20-M20-Stand`

The M20 wheeled robot model lives in `deep_robotics_model/M20/` (USD, URDF, and MJCF), referenced by the
`DEEPROBOTICS_M20_CFG` articulation in `src/M20_newton_training/assets/deeprobotics.py`.

`M20NewtonTraining-M20-M20-Walk` uses Flat's plane, 57-dimensional actor observation, 60-dimensional critic
observation, 16 actions, command distribution, and PPO networks. It replaces the turn-only air-time reward
with touchdown air-time credit (weight +10, reference 0.2 s, capped at 0.5 s). Commands above 0.1 m/s laterally
double this term, commands above 0.1 rad/s in yaw triple it, and combined side/turn commands quadruple it;
zero commands disable the term. Shorter-than-reference air times give negative credit, amplified by the same
factors. Actual wheel-speed mean square (wheel speeds in rad/s) is penalized at -0.2, multiplied by five for any turning
command. Other reward weights match Flat. Walk sets a task-local reward gait level of 1 and disables the
terrain/gait curricula, so the inherited gait-scaled rewards stay active without modifying sibling tasks.
These rewards encourage stepping without imposing a clock or removing wheel actions; gait quality requires
training and playback validation.

```bash
uv run isaaclab train --rl_library rsl_rl --task M20NewtonTraining-M20-M20-Walk \
  --num_envs 2048 --logger tensorboard physics=newton_mjwarp
uv run tensorboard --logdir logs/rsl_rl/deeprobotics_m20_walk --port 6006
```

`M20NewtonTraining-M20-M20-WalkOnSand` inherits Walk's complete observation/action interface and PPO
configuration (57 actor inputs, 60 critic inputs, 16 actions, 10,000 iterations), on Sand's per-environment
MPM beds and `newton_sand` physics. It uses the current Sand reset position (about z=0.68 m with the 20 cm
bed), centered XY and roll/pitch in +/-0.1 rad. The six contact-dependent rewards (undesired contacts,
contact force, air time, turn sliding, turn gait status and symmetry) are disabled along with ContactSensor;
their existing functions remain unchanged. Relative diagonal lift (+2) and completed-stroke credit (+0.5)
reuse Sand's kinematic rewards, with 7 cm lift and 6 cm horizontal travel references. Their scores double
for side commands, triple for turn commands, and quadruple for combined commands, using 0.1 thresholds.
They do not measure actual sand clearance, air time or contact. Other non-contact rewards, wheel-speed
penalty and command sampling remain identical to Walk. No contact proxy or new network input is added.
Play uses Walk's existing command behavior; it does not inherit Sand's keyboard override.

```bash
uv run isaaclab train --rl_library rsl_rl --task M20NewtonTraining-M20-M20-WalkOnSand \
  --num_envs 4 --logger tensorboard physics=newton_sand
# To fine-tune an existing Walk checkpoint, append:
# --checkpoint /absolute/path/to/walk/model_XXXX.pt --no-resume_optimizer_state
uv run tensorboard --logdir logs/rsl_rl/deeprobotics_m20_walk_on_sand --port 6006
```

`M20NewtonTraining-M20-M20-WalkOnSandProxy` replaces the particle bed with calibrated external forces
on the four M20 wheel feet. It keeps Walk's 57/60 observation dimensions, 16 actions and PPO networks;
no shin, thigh or belly forces are applied. The deployed parameters are in
`src/M20_newton_training/sand_proxy/calibrated/m20_foot_sand.json`. The original MPM tasks remain available.
This task uses plain `newton_mjwarp`, with a virtual sand surface at z=0.30 m and a non-colliding surface
visual. There are no particles to render. Commands are limited to +/-0.8 m/s forward,
+/-0.2 m/s sideways and +/-0.6 rad/s turning. Foot forces update at 2 kHz; the policy remains at 50 Hz.
The task owns physics decimation so Newton cannot fold these state-dependent external forces into one
policy-step update.

WalkOnSandProxy keeps speed tracking, air-time (+5), actual diagonal lift progress (+10),
diagonal contact symmetry (+3), body stability, leg torque/acceleration, action rate and wheel-speed costs.
HipX default-posture weight is -1.5. Its two new symmetry penalties are:

- `hipx_symmetry` (-0.5): mean squared FL+FR and HL+HR HipX angle sums in radians. The ordered left/right
  axes require opposite angles. For abs(vy)>0.05 m/s or abs(wz)>0.1 rad/s the cost is multiplied by 0.25.
- `diagonal_lift_consistency` (-3): compare completed swings, not simultaneous stance/swing heights.
  Record each foot's peak lift from its support position at swing start. Let A=(FL,HR), B=(FR,HL),
  within=0.5*((H_FL-H_HR)^2+(H_FR-H_HL)^2), between=(mean(A)-mean(B))^2.
  Raw cost is min((within+between)/0.25^2,4). It is settled once after both groups have a fresh completed
  swing, then both samples are consumed. A swing must begin in a valid diagonal support pattern,
  last at least 0.1 s (including the sampled landing interval), and contain no all-foot flight.
  Unmatched samples expire after 2 s. Initial drops are ignored; partial resets clear only that world.
  For equal within-pair peaks of 25 cm and 5 cm, the raw cost is 0.64 and the weighted 50 Hz event is -0.0384.

The simplified task disables `contact_forces`, HipY/Knee default-posture penalties,
`joint_acc_wheel_l2`, and `action_smooth_l2`. `feet_slide_ang_z_cmd`, `joint_mirror`, `stand_still`,
`undesired_contacts`, `sand_diagonal_steps` and `sand_step_completion` remain disabled.
Disabling force-excess rewards does not disable the physical sand force or contact timers.
Air-time tracking and both rotation gait rewards consider a foot supported only when its applied
upward world-Z force exceeds 5 N; horizontal drag and downward pull-out forces do not count as support.
Air/contact timers and touchdown detection
sample the last applied force once per 20 ms policy step (50 Hz), matching newton-terrain's sampling;
intermediate physics-rate force crossings are not counted. Each foot can produce at most one touchdown
per policy step. The first landing after reset is ignored, and each environment's timers clear on reset.
Physical proxy forces still update at 2 kHz. The air-time score uses a 0.1 s threshold and 1.0 s cap;
touchdowns after less than 0.1 s receive a negative score while stepping is enabled.
The full command norm sqrt(vx^2+vy^2+wz^2) must be >=0.1 for `feet_air_time`,
`rotation_gait_status`, `rotation_gait_symmetry`, and `diagonal_lift_consistency` to score.
Below 0.1 all four terms are zero, including their penalties; the three stateful gait terms clear
that world's support/progress/pending/window history. Exactly 0.1 remains enabled, including pure yaw.
HipX posture/mirroring and body stability remain active at low commands; physical contact timers
continue running. The separate `command_threshold=0.1` still selects side/turn amplification
(x2/x3, or x4 together). All movement directions are eligible above the norm threshold.
`rotation_gait_status` rewards only new lift progress of both unloaded diagonal feet, measured from
individual last-support capsule-bottom heights (25 cm target, weaker foot sets the score). The opposite
diagonal must support the robot; hovering and revisiting a credited height add no score.
`rotation_gait_symmetry` uses a 1 s window of true opposing diagonal states and scores
2*sqrt(p_A*p_B). Fixed support patterns, front-only support and all-four synchronous contacts score zero.
Each environment clears gait history on reset or when the command norm is below 0.1.
Changing direction or reversing yaw while remaining above that threshold does not erase support
anchors, already credited lift progress, or the symmetry window. The 25 cm target is relative
to each foot's last support position, not an absolute height above the virtual sand surface.
`joint_mirror` and `stand_still` are disabled in WalkOnSandProxy.
Non-foot undesired contacts remain disabled because the proxy models only feet. No ContactSensor or
force observations are added. WalkOnSand's MPM reward configuration is unchanged.
The robot starts at z=0.88 m, maintaining its 0.58 m initial base clearance above the reference surface.
The optional proxy-depth termination is currently disabled. The fitted depth-domain marker remains
0.28 m; deeper inputs retain the model's existing depth clamping.

```bash
# Start small; scale num_envs after inspecting the policy and measured force-domain coverage.
.venv/bin/isaaclab train --rl_library rsl_rl --task M20NewtonTraining-M20-M20-WalkOnSandProxy \
  --num_envs 64 --logger tensorboard --viz none physics=newton_mjwarp
.venv/bin/tensorboard --logdir logs/rsl_rl/deeprobotics_m20_walk_on_sand_proxy --port 6006

# Keyboard play: I/K forward/back, J/L sideways, U/O turn; release to stop.
.venv/bin/isaaclab play --rl_library rsl_rl --task M20NewtonTraining-M20-M20-WalkOnSandProxy \
  --num_envs 1 --checkpoint latest --viz newton physics=newton_mjwarp

# Nominal support check, without PPO or MPM.
.venv/bin/python -m sand_proxy_lab.validation.check_runtime --num-envs 4 --steps 500
```

The current schema-2 foot model combines an intruder-calibrated elastoplastic backbone with a
sliding response fitted to a whole-robot MPM trace. Its parameters are unchanged by the directory
reorganization. Collection, fitting, shadow validation, plotting and deployment tools are maintained
in [sand_proxy_lab](sand_proxy_lab/README.md). The Chinese [complete workflow guide](sand_proxy_lab/docs/标定与训练完整流程.md)
explains the model, data, equations and commands from fresh calibration through training.

Active data is under `output/sand_proxy/terrains/<地形名>/`; archived models, obsolete tools and unused trials are sealed
under [the legacy archive](archive/sand_proxy_legacy_20261023/README.md). The independent shadow tools
use either a Walk checkpoint (`python -m sand_proxy_lab.shadow`) or IsaacLab RandomAgent
(`python -m sand_proxy_lab.shadow.random_agent`); only MPM forces act on the robot in those comparisons.
Current-model provenance and dataset roles are indexed in `sand_proxy_lab/model_registry.json`.

`M20NewtonTraining-M20-M20-Sand` puts each environment on a granular bed advanced by Newton's implicit MPM
solver, coupled to the rigid robot solve. It therefore requires the Newton backend: the `newton_sand` physics
preset is its default, and the PhysX presets leave the sand frozen.

Sand playback uses keyboard commands with `--visualizer newton_gl`. Click the Newton window, then hold
I/K for +/- body X velocity (0.5 m/s), J/L for +/- body Y velocity (0.3 m/s), and U/O for +/- body Z angular
velocity (0.5 rad/s). Releasing keys stops the corresponding command; Backspace zeros all commands. WASD
and arrows remain available for camera movement. Key speeds are configured in Sand's `play_mode()` through
`keyboard_speeds`. Training continues to sample random commands.

Sand stepping is shaped without changing the 57-dimensional actor observation or 16-dimensional action:
diagonal wheel-center lift coordination (+2.0), completed lift/advance/lower strokes (+0.5, event credit per
second), low-leg horizontal motion (-0.2), and normalized actual wheel speed (-0.5). Lift targets 7 cm of
relative leg motion; a stroke must advance at least 6 cm for full credit. These are kinematic proxies, not
measurements of sand-surface clearance or contact. There is no imposed clock or added phase input. Sand
disables `gait_level`, uses ungated torque/action penalties, and reduces the default-pose penalties tenfold.
Its inherited raw joint-angle mirror penalty is replaced by foot-space diagonal coordination. Rough and
Stand retain their existing reward definitions.

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
uv sync --locked
```

This project was generated from an Isaac Lab source checkout. Its `pyproject.toml` uses editable relative paths to that
checkout, so regenerate the project or update `[tool.uv.sources]` if either directory moves.

After updating that Isaac Lab checkout, also align this project's `[tool.uv].override-dependencies` with the
checkout's overrides, then run `uv lock` and `uv sync --locked`. uv reads overrides from the root project;
overrides declared by the editable `isaaclab-dev` dependency are not inherited. Resolution includes optional
backends even when they are not installed, so an outdated root override can also block Newton-only training.

The local environment retains PyTorch CUDA 12.8 on Linux x86_64 and Windows (CUDA 13.0 on Linux aarch64).
NumPy is constrained to `<2.4` for Numba 0.64 and cmeel-boost 1.89 compatibility, and `ipywidgets` is declared explicitly to satisfy
the installed Open3D wheel. Keep these local adjustments when refreshing configuration from a generated project.

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

Click the Newton window to focus it. Hold `J/L` for positive/negative `vx` (body Y, 0.75 m/s),
`I/K` for positive/negative `vy` (body Z, 0.375 m/s), and `U/O` for positive/negative `wz`
(body X, 0.75 rad/s). Releasing a key stops that command axis; holding Backspace forces all commands
to zero. These speeds are 75% of each configured command range's maximum magnitude. Space pauses
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

The checked-in Pyright settings use Python 3.12 and this project's `.venv`, and check `src`, `scripts`,
`tests`, and `mojuco_sim` in `basic` mode. To repeat the command-line check from this directory:

```bash
npx --yes pyright@1.1.414
```

The local `typings/mujoco/__init__.pyi` describes the public binary bindings used by the CPU replay tool;
the installed MuJoCo wheel does not ship declarations for those exports. It has no runtime effect.
Isaac Lab configuration copies use `isaaclab.utils.replace`/`clone` so their types remain visible to Pylance.
Callable event classes and physics presets retain their runtime behavior, with casts at the upstream boundaries
whose annotations do not describe these supported inputs.

When opening the parent workspace instead of this package directory, its `pyrightconfig.json` should inherit
`./M20_newton_training/pyrightconfig.json`. Both workspace levels must use this package's `.venv/bin/python`.
After changing interpreters or upgrading Isaac Lab, run **Python: Restart Language Server** in VS Code.

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
