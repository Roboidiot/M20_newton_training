# Stand 的 MuJoCo sim-to-sim 验证

此目录独立使用 `mujoco`、`numpy` 和 `torch`，不启动 Isaac Lab，也不需要 GPU。
使用项目现有的 Python 环境即可。原始 MJCF 和训练代码不会被修改。

## 启动

在项目根目录运行：

```bash
# 带窗口：零 command，默认运行 20 秒
uv run python mojuco_sim/sim2sim_stand.py

# 指定导出的 TorchScript 策略
uv run python mojuco_sim/sim2sim_stand.py \
  --policy logs/rsl_rl/deeprobotics_m20_stand/2026-10-13_19-03-03/exported/policy.pt

# 给定固定 command：[vx, vy, wz] 分别对应机身 [Y 平移, Z 平移, X 转向]
uv run python mojuco_sim/sim2sim_stand.py --command 0.3 0 0 --duration 30

# 无窗口测试并保存误差统计
uv run python mojuco_sim/sim2sim_stand.py --headless --duration 20 --summary /tmp/stand_sim2sim.json

# 键盘交互，直到窗口关闭
uv run python mojuco_sim/sim2sim_stand.py --keyboard --duration 0
```

不传 `--policy` 时，脚本选择 `logs/rsl_rl/deeprobotics_m20_stand/*/exported/policy.pt` 中
最近修改的导出策略。这里需要 **TorchScript `policy.pt`**，不是训练 checkpoint `model_*.pt`。
若调整奖励后又续训，需要先在 Isaac Lab 中 play 新 checkpoint，生成对应的导出策略。

键盘输入采用 MuJoCo 原生 viewer 的按键回调：

| 按键 | 功能 |
| --- | --- |
| I / K | vx 增加 / 减少 0.1 m/s |
| J / L | vy 增加 / 减少 0.1 m/s |
| U / O | wz 增加 / 减少 0.1 rad/s |
| Backspace | 所有 command 归零 |
| R | 重置机器人与统计，并将 command 归零 |
| Space | 暂停 / 继续物理仿真 |

这是**按键增加、保持指令值**的控制方式，松开按键不会自动归零。
command 会限幅到训练范围 `[±1.0 m/s, ±0.5 m/s, ±1.0 rad/s]`。
`--command` 和键盘输入都没有随机重采样。多个方向可以组合。

## 配置与接口

`stand_config.json` 保存此项目的 Stand 接口和标称控制参数：

- 机身初始高度 0.8028 m，MuJoCo 的 wxyz 四元数 `[0.7071, 0, -0.7071, 0]`。
- 使用 MJCF 的原始平地，关闭机器人内部自碰撞，保留机器人与地面的碰撞。
- 物理频率 400 Hz，策略频率 50 Hz；每个策略周期推进 8 个物理步。
- Actor 观测依次为：角速度×0.25、投影重力、command、相对关节位置、关节速度×0.05、上一动作。
  总计 57 维；没有机身线速度和高度扫描。
- 策略输出为 12 个腿部位置动作和 4 个轮速动作，总计 16 维。
- 根据名称映射 MuJoCo 的 `qpos`、`qvel` 和 actuator，处理其腿/轮交错的原始排列。
- 腿部 PD：Kp=80、Kd=2，位置动作尺度 hipx=0.125、hipy/knee=0.25。
- 轮部速度反馈：Kd=0.6，轮速动作尺度=5，armature=0.00243216。
- 腿/轮力矩分别限幅到 ±76.4 / ±21.6 N·m，策略动作与处理后的目标也按训练配置截断。

### 已训练策略的关节位置观测兼容性

修复前的训练函数 `joint_pos_rel_without_wheel()` 在重排关节位置后，直接使用 Newton 原始轮关节索引
`[3, 7, 11, 15]` 置零。此配置的 `joint_position_zero_indices` 默认仍使用这些索引，
以兼容本目录实测的旧 checkpoint。它们并不等于重排后的最后四个轮关节槽位。

训练函数现已修复：先在原始关节顺序中将轮关节置零，再重排观测。
**验证修复后训练/续训的策略时，必须将配置中的 `joint_position_zero_indices` 改为
`[12, 13, 14, 15]`。** 验证旧策略时保留旧索引；脚本无法从 `policy.pt` 自动识别其训练时的置零逻辑。
可分别保存两份 JSON，通过 `--config` 选择；复制后的 JSON 请仍放在本目录，以保持模型相对路径有效。

若修改过训练时的初始姿态、动作尺度、观测顺序或 PD 参数，应依据该次训练的 `params/env.yaml`
和运行时关节顺序更新此 JSON。奖励权重不参与 MuJoCo 策略推理。

## 验证范围

这是平地和固定标称参数下的基础验证，不复现崎岖地形、训练中的观测噪声、参数随机化或执行器延迟。
MuJoCo 的碰撞模型/接触求解与 Newton 的训练模型也存在差异；脚本可运行不等于完整的迁移验证通过。
可以先测试零 command，再测试单轴平移、转向与组合指令。

结束时打印平均线速度/角速度误差、机身高度、投影重力、超出 Stand 姿态范围的控制步数和峰值力矩。
默认不在 20 秒或跌倒时随机重置机器人：`--duration` 指定整个验证时长，`--stop-on-fall` 可在
触发相同的姿态失稳条件时提前停止。`--summary` 保存的是最近一次 R 重置后的统计。

诊断时可先用纯 PD 保持默认姿态：

```bash
uv run python mojuco_sim/sim2sim_stand.py --pd-only --headless --duration 5
```

纯 PD 模式不执行策略，因此不跟踪非零 command。

## 本次实测

使用 `2026-10-13_19-03-03/exported/policy.pt`，无窗口、平地、固定标称参数：

| command `[vx, vy, wz]` | 时长 | 平均平移误差（m/s） | 平均转向误差（rad/s） | 超出姿态范围的步数 |
| --- | --- | --- | --- | --- |
| `[0, 0, 0]` | 20 s | 0.0472 | 0.0515 | 0 |
| `[0.3, 0, 0]` | 10 s | 0.3022 | 0.0587 | 0 |
| `[0, 0.3, 0]` | 10 s | 0.0314 | 0.0168 | 0 |
| `[0, 0, 0.3]` | 10 s | 0.0560 | 0.1225 | 0 |

此策略能保持竖立，并跟踪机身 Z 方向的平移；机身 Y 方向的平移跟踪较差，仍需后续排查和训练。
窗口交互没有在本次无窗口测试中验证。

脚本的观测、动作、关节/力矩映射和键盘回调测试可单独运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/test_sim2sim_stand.py -q
```
