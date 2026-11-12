# 通用 19 维轮模型迁移到 IsaacLab

此目录已包含迁移文件，可整体复制到目标项目；原训练文件保留在 Train 中。

## 需要复制的文件

推理只需要两个文件：

1. `wheel_model_runtime.py`：独立 Torch 推理、历史缓存、输入构造与力矩平移工具。
2. `best.pt`：来自 `shared_wheel_19d_random_v1` 的共享模型，包含权重、训练缩放参数、输入/输出定义、对称规则和历史步长。

将它们放到 IsaacLab 项目能导入的位置，例如：

```text
your_project/
  model/
    __init__.py             # 已提供包导出，可直接 from model import ...
    wheel_model_runtime.py
    best.pt
    README.md
    reference/              # 原采集几何、物理参数、元数据及部署核对资料
```

使用 IsaacLab 环境已有的 PyTorch 即可；无需复制训练脚本、数据目录、DeepXDE 或 NumPy 推理器。
`scalers.json` 无需单独复制，缩放参数已经包含在 best.pt 中。
`reference/` 已附带原采集目录的 `geometry.json`、`physics.json`、`metadata.json` 和
`body_properties.npz`，以及 `deployment_checks.json` 和原局部沙面估计 `surface.py`。
这些文件作为几何、土壤、采样时序及参考质心的核对资料，不由推理模块读取。
`surface.py` 是可选的原采集算法参考，单独使用它时需要 NumPy 与 Torch。

## 模型在强化学习中的位置

模型是环境动力学的一部分：读取刚体/轮状态 → 预测沙力与力矩 → 写入物理引擎 → 推进仿真。
策略网络仍输出关节控制动作，接收观测/奖励；本模型固定权重，不加入 PPO 等策略优化器。
模型输出不是关节电机力矩，不应直接传入关节 effort target 接口。

## 19 维输入与 6 维输出

每个物理步构造 `x[num_envs,4,19]`，轮序必须是 FL、FR、HL、HR。
输入均为原始物理单位，尚未标准化、尚未进行对称变换。

| 索引 | 含义 | 单位/约定 |
|---|---|---|
| 0 | 初始参考沙面减胶囊最低点高度 | m，允许负值 |
| 1 | 局部沙面减胶囊最低点高度 | m，无效时填 0 |
| 2 | 局部沙面有效标记 | 0/1 |
| 3–5 | 轮胶囊中心线速度 | 世界系，m/s |
| 6–8 | 轮刚体角速度 | 世界系，rad/s |
| 9–11 | 胶囊轴在世界系的有向单位向量 | 与采集几何轴符号一致 |
| 12 | 轮关节相对小腿的实际自转速度 | rad/s，保留正负号 |
| 13–15 | 同一轮上一物理步中心线速度 | 世界系，m/s |
| 16–18 | 同一轮上一物理步刚体角速度 | 世界系，rad/s |

输出 `y[num_envs,4,6]`，依次是世界系 Fx/Fy/Fz（N）、Tx/Ty/Tz（N·m）。
力矩绕该轮 MPM 代理输入质心；不自动平移到 IsaacLab 当前刚体质心。

`WheelModelRuntime` 内部执行：目标轮 → FL 标准表达 → 训练统计量标准化 →
网络 → 反标准化 → 恢复目标轮世界系。调用方不要再做一次符号变换或归一化。

## 直接使用

```python
from pathlib import Path
import torch
import model as wheel_model_package
from model import (
    WheelModelRuntime,
    WheelMotionHistory,
    build_current_features,
    shift_torque_reference,
)

# 环境初始化时只加载一次，device 与仿真张量一致。
model_path = Path(wheel_model_package.__file__).resolve().parent / "best.pt"
sand_model = WheelModelRuntime(model_path, device=self.device)
sand_model.check_physics_dt(self.physics_dt)  # 本模型要求 0.005 s
history = WheelMotionHistory(self.num_envs, device=self.device)

# 在合适的每物理步/耦合步钩子中运行。下面的状态由你的环境提供：
# reference_height_w: [E,1]，每个环境的初始参考沙面世界高度
# local_height_w/local_valid/bottom_z_w/joint_velocity: [E,4]
# center_velocity_w/angular_velocity_w/axis_w: [E,4,3]
current13 = build_current_features(
    reference_height_w, local_height_w, local_valid, bottom_z_w,
    center_velocity_w, angular_velocity_w, axis_w, joint_velocity,
)
x, history_ready = history.append(current13)
y = sand_model(x)                           # [E,4,6]，保持在同一设备

# 新环境第一次尚无上一帧速度：本示例这一物理步不施加代理轮沙力。
# 这是显式的部署过渡策略，不表示训练数据的第一帧真实力为零。
y = torch.where(history_ready[:, None, None], y, torch.zeros_like(y))
force_w, torque_at_proxy_com_w = y[..., :3], y[..., 3:]

# _reset_idx(env_ids) 中，与机器人和沙面状态一起重置该环境的历史。
history.reset(env_ids)
```

上述代码分属初始化、物理步和 reset 三个位置，不能原样顺序写在同一个物理步内。
环境字段和状态读取函数需要按你的项目实现；这里没有自动读取任意 IsaacLab 场景的适配器。
如果已自行维护 x19，可直接调用 `sand_model(x19)`，无需使用历史类或输入构造工具。
单轮可调用 `sand_model.predict_wheel(x_fr, "FR")`，输入形状 `[...,19]`。

## 从机器人状态取得训练一致的输入

按名称查询轮刚体 `fl_wheel/fr_wheel/hl_wheel/hr_wheel` 及轮关节
`fl_wheel_joint/fr_wheel_joint/hl_wheel_joint/hr_wheel_joint`，明确按此顺序组成索引。
不要使用另一个项目的硬编码 DOF 列或求解器全局索引；关节速度应来自实际 joint velocity，
不是目标自转速度，也不是世界角速度模长。

原采集胶囊半径约 0.09 m、半长约 0.027 m、局部中心为零，局部轴约 `[0,-1,0]`。
准确值以 reference/geometry.json 为准。设胶囊中心世界位置 c、轴单位向量 a、半长 L、半径 r：

```python
bottom_z_w = center_w[..., 2] - half_length * axis_w[..., 2].abs() - radius
center_velocity_w = com_velocity_w + torch.linalg.cross(
    angular_velocity_w, center_w - rigid_com_w, dim=-1
)
```

不能用 `reference_height - center_z` 代替训练下陷量，也不能直接用轮 COM 线速度
代替胶囊中心线速度（除非两个点确实重合）。局部中心与轴应由刚体 link 位姿旋转到世界系；
IsaacLab 的 link 四元数和旧采集器的 xyzw 四元数不可混用旋转函数。
多环境的参考沙面高度必须包含各自环境在世界中的高度偏移，不能只对机器人位置减 origin。

IsaacLab 2.3.1 文档区分 link 状态与 COM 状态，并给出了对应的世界系速度和位姿属性。
按项目版本核对 `body_link_pos_w/body_link_quat_w/body_com_pos_w/body_com_lin_vel_w/body_com_ang_vel_w`。
参考：[官方资产数据接口](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/api/lab/isaaclab.assets.html)。

## 写入物理引擎

若模型参考点 P 与当前仿真施力参考点 C 不同，先在同一世界系中平移力矩：

```python
torque_at_sim_com_w = shift_torque_reference(
    force_w, torque_at_proxy_com_w, proxy_input_com_w, sim_body_com_w
)
# 即 tau_C = tau_P + (P - C) × F
```

以下为支持该 API 的 IsaacLab 2.3.x Articulation/PhysX 路径示意。明确把施力位置设为 C，
并传入已经换算到 C 的力矩，避免依赖默认施力点：

```python
robot.set_external_force_and_torque(
    forces=force_w.contiguous(),             # [E,4,3]
    torques=torque_at_sim_com_w.contiguous(),
    positions=sim_body_com_w.contiguous(),   # 世界系施力位置 C
    body_ids=wheel_body_ids,                 # FL/FR/HL/HR 顺序
    is_global=True,
)
# 在仿真步之前由 robot/scene.write_data_to_sim() 将缓冲写入引擎。
# 如果原训练循环已经执行该操作，不要再重复推进/更新一次仿真。
```

另一种等价方式是在 P 施加 F 并传入 tau_P，此时不要再重复平移力矩。
这些接口会写外力缓冲，具体版本可能使用不同的外力接口；Newton/MPM 自定义耦合路径
应使用其当前的逐刚体施力/耦合接口，不能把上述 PhysX 示例不加核对地复制进去。
官方参考：[外力接口](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/api/lab/isaaclab.assets.html)、
[2.3.0 Articulation 源码](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/assets/articulation/articulation.py)。

## 调用时序与模型覆盖范围

- 本模型的一步历史是 0.005 s。策略若每 0.02 s 更新，应在其间每个 0.005 s
  物理/耦合步更新模型及历史，不是只在策略动作更新时运行一次。
- DirectRLEnv 的 `_apply_action()` 是一种每物理步接入位置，
  `_pre_physics_step()` 是每策略步位置，见[官方 Direct 环境说明](https://isaac-sim.github.io/IsaacLab/main/source/how-to/create_direct_rl_env.html)。
  原数据来自刚体推进后、MPM 求解前的采样状态，标签为该耦合区间平均沙力。
  若保留原 Newton staggered 流程，最接近原训练语义的是在原 MPM 输入/求解钩子接入。
  改成普通刚体步前读状态、对下一步施力会改变采样相位，应做闭环对照，不能视为已验证等价。
- 完整 19 维输入需要局部沙面高度及有效性。原采集器使用邻域粒子柱顶的环带中位数估计，
  见 `reference/surface.py`。只保留一个平面而完全删除 MPM 后，没有粒子状态可供同样估计；
  需提供地形/沙面演化近似并重新验证。模型本身不预测沙面，也不能长期随意用 0 填两个深度和有效标记。
- 如果保留真实 MPM 做对照，仅记录模型预测；替代时要去掉同一轮原有的 MPM 沙力，防止叠加两份。
  只学习了四个轮的模型，不能同时替代小腿、机身等其他部位的 MPM 力。
- 训练所用土壤参数、轮几何不在 19 维条件输入中；更换这些参数不是直接修改一个运行时开关。
  固定世界 X/Y 对称规则只在当前采集朝向下验证过，随机 yaw 的训练需要参考系统一并验证。
- 力和力矩允许负值，保持现有符号；不能无依据将 Fz 截断为非负。

## 已完成的验证

独立 Torch 模块已用实际共享权重和随机/行走策略保存数据与原 SharedWheelPredictor 对照。
两者因 float32 与原 NumPy float64 中间缩放而有微小浮点差异，详见部署检查记录。
另外检查了并行环境局部 reset、第一帧历史标记、六维输出符号、缩放和力矩参考点平移。
当前机器是 CPU PyTorch，尚未执行 GPU 吞吐测试或 IsaacLab 真实闭环接入。
