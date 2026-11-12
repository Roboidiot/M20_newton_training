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

按多组奖励权重依次训练：编辑 `scripts/train_batches.json`，执行
`.venv/bin/python scripts/train_batch.py --config scripts/train_batches.json`。
支持先预览/检查配置、各组独立训练或接续上一组 checkpoint；详见 [批量训练说明](scripts/批量训练说明.md)。

The M20 wheeled robot model lives in `deep_robotics_model/M20/` (USD, URDF, and MJCF), referenced by the
`DEEPROBOTICS_M20_CFG` articulation in `src/M20_newton_training/assets/deeprobotics.py`.

`M20NewtonTraining-M20-M20-Walk` uses Flat's plane, 57-dimensional actor observation, 60-dimensional critic
observation, 16 actions and PPO networks. Training commands are forward-only: vx=0.25..0.4 m/s, vy=wz=0,
without stationary samples. It replaces the turn-only air-time reward
with touchdown air-time credit (weight +50, reference 0.5 s, no duration cap). Commands above 0.1 m/s laterally
double this term, commands above 0.1 rad/s in yaw triple it, and combined side/turn commands quadruple it;
zero commands disable the term. Shorter-than-reference air times give negative credit, amplified by the same
factors. Actual wheel-speed mean square (wheel speeds in rad/s) is penalized at -0.2, multiplied by five for any turning
command; the current batch overrides this weight to -2.
`rotation_gait_status` uses `ProxyRotationGaitStatus(reward_mode="rough", min_air_time=0)`:
while one diagonal swings and the other supports, mean swing-foot height must exceed mean support-foot
height by 20 cm. Every qualifying control step scores one, including a held pose (source weight +5).
This retains exact opposing contact patterns and the all-motion command gate, without an air-time lower bound.
`rotation_gait_symmetry` balances the two opposing diagonal states in a 4 s sliding window (current source weight +15),
using `2*sqrt(p_A*p_B)`. It imposes no fixed cycle, order or minimum duration on phase occupancy.
Both gait terms activate under straight commands with command norm >=0.1.
The shared gait formulas read measured rigid `ContactSensor` forces directly, using upward Fz>5 N as support.
Contact timers/events sample once per 50 Hz control step, shared across terms; no proxy force is computed or applied.
Walk also enables `base_height_l2` with weight -10 and a 0.58 m world-Z target on the flat ground,
without height-scanner adjustment. Its `flat_orientation_l2` weight is -10.
Walk disables `ang_vel_xy_l2`, `lin_vel_z_l2`, `joint_mirror`,
`joint_torques_l2`, `hipx_joint_pos_penalty`, `hipy_joint_pos_penalty`, `knee_joint_pos_penalty`,
`action_rate_l2` and the `contact_forces` reward. The contact sensor remains active for gait judgment.
Other reward weights match Flat. Walk sets a task-local reward gait level of 1 and disables the
terrain/gait curricula, so the inherited gait-scaled rewards stay active without modifying sibling tasks.
These rewards encourage stepping without imposing a clock or removing wheel actions; gait quality requires
training and playback validation.

```bash
uv run isaaclab train --rl_library rsl_rl --task M20NewtonTraining-M20-M20-Walk \
  --num_envs 2048 --logger tensorboard physics=newton_mjwarp
uv run tensorboard --logdir logs/rsl_rl/deeprobotics_m20_walk --port 6006
```

`M20NewtonTraining-M20-M20-WalkOnSand` keeps Walk's policy interfaces and PPO networks
(57 actor inputs, 60 critic inputs, 16 actions), on Sand's per-environment MPM beds with
`physics=newton_sand`. It now shares all reward definitions, weights, command sampling and robot
initialization with WalkOnSandProxy through `config/M20/walk_sand_common_cfg.py`. Change shared
training settings there; the original Walk, Rough and Sand tasks remain independent.

For MPM training/play, select `physics=newton_sand`; `physics=newton_mjwarp` selects
rigid-only physics and does not advance the particles. The MPM foot-impulse observer
also requires `env.sim.physics.use_cuda_graph=False` (already the task default).
WalkOnSandProxy instead uses `physics=newton_mjwarp` with the neural force action.

`mdp/mpm_foot_contact.py` passively collects every MPM collider impulse belonging to each ordered
wheel foot and environment. It sums the full 20 ms control window before converting to force:
`F_world = sum(J_world) / 0.02`. Force conversion, support/air-time timers, touchdown events and
rewards all update at 50 Hz. Repeated reward reads reuse the same sample. Support is upward
world-Z force >5 N; horizontal drag and downward forces do not count. Other links and tray
colliders are excluded from foot rewards. The real MPM solver still handles their physical contact.
The observer adds zero action dimensions and never applies extra proxy forces.

MPM physics remains 200 Hz (`dt=0.005`, decimation 4), with the rigid entry taking two substeps.
This task uses eager stepping (`use_cuda_graph=False`) so the passive hook reads every actual MPM
substep, including sparse-grid rebuilds. Neural proxy forces update at 200 Hz; the preserved analytical model uses 2 kHz with its original timestep.
The MPM walking beds are 2.4 m square and 0.20 m thick above the rigid floor. The robot starts centered at z=0.73 m,
with roll/pitch reset ranges +/-0.02 rad. Current commands enable only forward/backward motion,
`abs(vx)=0.15..0.5 m/s`, `vy=wz=0`; play keeps the shared Walk keyboard behavior.

```bash
uv run isaaclab train --rl_library rsl_rl --task M20NewtonTraining-M20-M20-WalkOnSand \
  --num_envs 4 --logger tensorboard physics=newton_sand
# To fine-tune an existing Walk checkpoint, append:
# --checkpoint /absolute/path/to/walk/model_XXXX.pt --no-resume_optimizer_state
uv run tensorboard --logdir logs/rsl_rl/deeprobotics_m20_walk_on_sand --port 6006
```

`M20NewtonTraining-M20-M20-WalkOnSandProxy` now defaults to the frozen neural wheel-wrench model
copied from `sand_pinn_lab/model`, with fixed reference/local sand surfaces at environment origin +0.20 m.
Runtime weights live in `src/M20_newton_training/sand_proxy/neural/best.pt`; the preserved analytical model
lives in `sand_proxy/analytical/`. See [model layout, feature/torque conventions and backend switching](src/M20_newton_training/sand_proxy/README.md).
Actor/critic/action dimensions remain unchanged. The following analytical calibration notes remain available
for the explicitly selected analytical backend.

The proxy task replaces the particle bed with predicted external forces
on the four M20 wheel feet. It keeps Walk's 57/60 observation dimensions, 16 actions and PPO networks;
no shin, thigh or belly forces are applied. The deployed parameters are in
`src/M20_newton_training/sand_proxy/analytical/calibrated/m20_foot_sand.json`. The original MPM tasks remain available.
This task uses plain `newton_mjwarp`, with a virtual sand surface at z=0.20 m and a non-colliding surface
visual. There are no particles to render. Each environment independently resamples every 10 s,
choosing uniformly among axes with a nonzero configured range. A range of `(0, 0)` disables
that axis; all-zero ranges produce stationary commands. Enabled axes share `min_abs_command=0.15`,
including the boundary. The extraction/stepping stage currently enables forward motion only:
`vx` is in `[0.15,0.25]` m/s, with `vy=wz=0`. To restore backward/lateral/turn sampling, give those axes nonzero ranges
that reach the minimum magnitude. `SingleAxisVelocityCommand` in `mdp/single_axis_velocity.py`
bypasses the shared sampler's <=0.2 m/s zeroing rule. Heading/standing overrides are disabled.
Neural wrenches/history update at 200 Hz (0.005 s), analytical forces use 2 kHz when selected with the original timestep; the policy
and all rewards remain at 50 Hz. The task owns physics decimation so Newton cannot fold these
state-dependent forces into one policy-step update.

Both sand-walking tasks restore body-frame exponential tracking:
`r_lin = exp(-((vx_cmd-vx)^2 + (vy_cmd-vy)^2) / 0.25)`, with `std=0.5` and weight +5.
Zero command components remain in the error, so unwanted lateral motion is penalized during straight commands.
The angular slot also binds `track_ang_vel_z_exp` with `std=0.5`, but remains disabled by the shared
zero-weight settings. If enabled, its raw score is `exp(-(wz_cmd-wz)^2 / 0.25)`.
Raw scores lie in [0,1] and peak when actual velocity equals the command; RewardManager applies weight and dt.
The previous linear ratio functions remain available in `mdp/rewards.py` for experiments.
Walk/Rough retain their existing exponential tracking configurations. Batch Proxy runs inherit these shared defaults.
Air-time, crawl lift, forward touchdown and crawl phase weights are
+5/+200/+5/+2.5 respectively. HipX posture/mirroring weights are -0.25/-0.5 and the straight-command
wheel-speed cost is -10.0. `feet_lift_consistency` is disabled; its implementation remains available. The `rotation_gait_status`/`rotation_gait_symmetry`
slot names are retained, but now bind the existing single-swing crawl helpers.
`flat_orientation_l2` uses weight -20.0 and `forward_pitch_orientation_l2`: with body-X
command >0.1 m/s, target pitch is -15 degrees (nose up), with zero roll. Other commands keep
the level target. Adjust `flat_orientation_l2.params["forward_pitch_deg"]` in
`walk_sand_common_cfg.py` to tune the angle. This changes the training objective; existing
checkpoint playback does not acquire the new posture without further training. `base_height_l2` uses `sand_base_height_l2`, weight -10,
and a root target 0.32 m above the nominal sand surface (world Z=0.62 m for origin Z=0).

`com_forward_l2` adds a whole-robot center-of-mass bias penalty, weight -1.0, to both sand-walking tasks.
It uses current link masses and world CoM positions, relative to the geometric mean of the supporting foot
positions (upward force >5 N; at least two supports). With the normalized planar command rotated by chassis
yaw into world XY, let d be the projection of CoM-minus-support-center along that direction. The raw cost is
`(max(d - 0.03, 0) / 0.10)**2`; RewardManager applies the negative weight and 0.02 s timestep.
The 3 cm deadband allows ordinary weight shifts. Backward and lateral commands use their own direction;
idle/pure-yaw commands and planar command magnitudes below 0.1 disable this term. It does not penalize
world-space progress, uses neither the base origin nor force-weighted pressure center as CoM, and is not a
support-polygon/ZMP stability guarantee. Parameters live in `SandWalkRewardsCfg.com_forward_l2` and can
also be changed with the batch trainer. Walk/Rough, observations/actions and sand-force physics are unchanged.

Proxy support means world-Z `(applied proxy force + measured rigid floor reaction) >5 N`.
The `foot_ground_contacts` sensor filters the four wheel bodies against `/World/ground` shapes,
excluding self contacts. Air/contact timers and touchdown events sample once per 20 ms policy step,
after physics, using the last applied proxy force and the current floor reaction. The floor reaction
is not applied again as an external wrench. The first landing after reset is ignored. Air-time credit on
subsequent touchdowns is `min(unloaded_time,1.0)-0.1`; short air times receive a negative score.
Air time itself does not check whether the foot has left the sand surface.

`rotation_gait_status` now uses `ProxyRotationGaitStatus`: FL+HR swing while FR+HL support, or vice versa.
Both swing feet must satisfy `min_air_time` (default 0.5 s; batch overrides remain per run).
Each foot's normalized lift is `clip((z_foot_world-z_foot_world_last_support)/0.20,0,1)`;
the pair earns only new progress of the smaller value, at most one raw unit per paired swing.
As of 2026-11-07, both lift and forward-step rewards use the calibrated foot-bottom application point
in world coordinates; neither subtracts chassis nor HipX positions. Each foot's supported world Z is
updated until unloading and then frozen as its reference. The 20 cm target is a change in world height;
a foot still below the sand surface can earn progress. Initial
unobserved support, hovering, and revisiting an already credited height earn zero.
Chassis or HipX motion with a fixed world-space foot earns no lift. Upward whole-robot translation can
contribute if the feet satisfy the opposing-diagonal contact gate.
Current gait terms are:

- `rotation_gait_status` uses `ProxyRotationGaitStatus` with the world-Z 20 cm target and
  a continuous-unloading gate for both diagonal swing feet. There is no per-foot rotation discount.
- `feet_air_time` uses the stateless `proxy_feet_air_time_command_scaled`: on touchdown, score
  `min(T,1)-threshold`, retaining the short-air penalty and command multipliers, without foot balancing.
- `rotation_gait_symmetry` uses `ProxyRotationGaitSymmetry`: within a 4 s window, the two opposing
  diagonal states have occupancies p_A/p_B; raw score is `2*sqrt(p_A*p_B)`.
  Three-support/one-swing, same-side/front-pair support, all-support and full flight receive no phase credit.
- `feet_forward_step` uses `ProxyFootForwardStepReward`: compare the foot's world-XY positions
  at consecutive touchdowns, project onto the current planar command rotated to world XY using chassis yaw, and score
  `clip(world_advance/target_stride,0,1)`. Foot motion carried by the chassis counts; chassis motion with
  a fixed world-space foot does not. The current target is 12 cm and unloading must last
  0.1 s. The original `repeat_decay=1` remains; chassis-progress and world-placement caps are removed.

The inverse discount is `1/(1+progress_i-min(progress))`; it decreases repeated-foot gain but never
forces that gain to zero. All four terms retain their current weights and run at 50 Hz.
Existing body/action constraints, observations, network interfaces and batch checkpoint configuration
are unaffected. Forward stride still requires a planar command; the other three terms use full-command
norm >=0.1. Contact still uses upward force >5 N.

The experimental finite-lead helpers remain available in `mdp/crawl_rewards.py`, with opt-in tests,
for possible later comparison; they are not bound to these four task slots. Their experimental formulas
remain documented in the implementation docstrings.
`feet_sand_clearance` was outside this four-term rollback and retains its separately configured helper;
it is currently disabled in the batch JSON. Per-foot lift and forward diagnostics retain only
the original count/progress logs.

Foot-force diagnostics log only `Metrics/foot_forces/{fl,fr,hl,hr}_fz_mean`, in newtons.
They use 50 Hz world-Z force samples: the last applied proxy force of each control interval,
or MPM whole-window impulse / 0.02 s. Proxy force metrics exclude the rigid floor reaction,
even though Proxy contact timing includes it.
Signed means include airborne zeros and negative pull-out forces. At reset, each scalar
averages the resetting worlds' own episode time means. TensorBoard further aggregates
these scalars per PPO iteration; this is not a raw single-environment 50 Hz force trace.
These metrics do not affect rewards or observations.

Both sand-walking tasks disable `base_height_low` and `front_wheels_low`, and set
`base_height_reset_penalty` and `front_wheels_reset_penalty` to zero.
The continuous `base_height_l2` cost and `front_wheels_height` reward remain enabled.

Body stability, action rate (-0.01) and wheel-speed costs are enabled; leg torque/acceleration are disabled.
`contact_forces`, HipY/Knee posture costs, wheel acceleration, second-order action smoothing,
sliding, `joint_mirror`, `stand_still`, non-foot undesired contacts, `sand_diagonal_steps` and
`sand_step_completion` remain disabled. Proxy adds a filtered floor ContactSensor for gait judgment;
no force observations are added. MPM retains its passive impulse observer.
Walk and Rough configurations are unchanged. Both sand-walking tasks start at z=0.73 m,
0.53 m above the virtual surface. The optional proxy-depth termination remains disabled;
the preserved analytical model retains its fitted-depth clamping marker of 0.28 m; the neural model has no calibrated depth limit.
Existing checkpoints keep the same 57/60 observations, 16 actions and PPO network architecture.
New weights affect policy learning only after training continues.

```bash
# Start small; scale num_envs after inspecting the policy and measured force-domain coverage.
.venv/bin/isaaclab train --rl_library rsl_rl --task M20NewtonTraining-M20-M20-WalkOnSandProxy \
  --num_envs 64 --logger tensorboard --viz none physics=newton_mjwarp
.venv/bin/tensorboard --logdir logs/rsl_rl/deeprobotics_m20_walk_on_sand_proxy --port 6006

# Keyboard play: I/K forward/back, J/L sideways, U/O turn; release to stop.
# Current forward-only ranges allow I up to 0.25 m/s; other directions produce zero. Release always stops.
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
