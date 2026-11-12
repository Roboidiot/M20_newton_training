# M20 沙地代理模型

WalkOnSandProxy 的训练和 play 默认使用 `neural/best.pt`。
原解析模型完整保存在 `analytical/`，可以显式切回；原标定、拟合和影子实验仍使用解析模型。
两套运行时代码都不导入 `sand_pinn_lab` 或 `sand_proxy_lab`。

```text
sand_proxy/
├── neural/
│   ├── best.pt                       # 从 sand_pinn_lab/model 原样复制的权重
│   ├── wheel_model_runtime.py        # 原样复制的19维共享模型、归一化与对称变换
│   ├── foot_model.py                 # 固定沙面、逐环境速度历史、力/力矩输出适配
│   ├── deployment.json              # 沙面/几何/参考质心约定、来源哈希
│   └── reference/                   # 原采集几何、刚体属性、元数据、物理与检查记录
├── analytical/
│   ├── foot_model.py                 # 原弹塑性、脚印历史与滑移混合实现
│   ├── shadow_response.py            # 原滑移响应函数
│   └── calibrated/m20_foot_sand.json  # 原解析系数，内容保留
├── foot_model.py                     # 原解析API的兼容导出，供离线标定工具使用
└── shadow_response.py                # 原解析滑移API的兼容导出
```

## 固定沙面与深度

共享高度定义在`tasks/m20/sand_walk_settings.py`的`SAND_WALK_SURFACE_HEIGHT`，
由`walk_sand_common_cfg.py`绑定到奖励、加载高度和代理模型。每个环境沙面为 `env_origin_z + 0.20 m`，不随脚印、支撑、粒子或时间变化。
原来的足底几何和深度定义保持不变：

```text
c_world = wheel_link_position + R(link_quaternion) · capsule_local_center
a_world = R(link_quaternion) · capsule_local_axis
z_bottom = c_world.z − radius − half_length × abs(a_world.z)
depth = env_origin_z + 0.20 − z_bottom
```

胶囊半径约0.09 m、半长约0.027 m。深度允许负值，不用轮中心高度代替足底。
网络的“初始参考下陷”和“局部下陷”两个输入都使用这个深度，`surface_valid=1`，
表示采用始终有效的固定平面；不把局部深度误填成零，也不估计沙面演化。
`depth<=0` 时实际沙力和沙力矩均置零。网络没有旧模型的脚印坑或弹塑性内部状态。

当前机器人初始高度为环境原点上方0.73m（沙面0.20m + 初始净空0.53m），比此前降低0.10m。
机身高度重置及其重置惩罚已禁用。连续机身高度奖励的目标净空为0.40m，对应环境原点上方0.60m。
MPM WalkOnSand的粒子层也从刚性底板0m生成到0.20m；原Sand任务和PINN采集仍保留0.30m沙床。

这按用户要求简化了训练时的局部沙面输入；原网络训练数据使用粒子估计沙面，两者不是完全相同的输入分布。

## 输入、输出和施力

按FL、FR、HL、HR顺序，每轮输入19维：

| 索引 | 输入 |
| --- | --- |
| 0、1 | 固定沙面减足底高度，两个深度相同 |
| 2 | 沙面有效标记，固定为1 |
| 3–5 | 胶囊中心的世界系线速度 |
| 6–8 | 轮刚体世界系角速度 |
| 9–11 | 胶囊有向轴在世界系的向量 |
| 12 | 实际轮关节自转速度，按关节名称映射 |
| 13–15 | 上一物理步胶囊中心线速度 |
| 16–18 | 上一物理步轮刚体角速度 |

中心速度通过 `v_COM + omega × (center − COM)` 计算，不使用足底速度、速度目标或偏航局部坐标系速度。
`wheel_model_runtime.py`内部完成对称变换、标准化、网络和输出还原；外部不再次处理。
网络为冻结推理，参数不加入Actor/Critic优化器，策略观测57/60维和16维动作不变。

网络输出世界系 `[Fx,Fy,Fz,Tx,Ty,Tz]`。力矩参考点使用原采集刚体属性中的对应轮质心，
按当前link位姿转换成世界点P，再移到当前仿真轮质心C：

```text
tau_C = tau_P + (P − C) × F
```

通过 `instantaneous_wrench_composer` 在C施加F和tau_C，不能再在足底施加同一份F。
足底位置继续供迈步高度/落足统计使用，不是NN的额外施力点。
保留网络输出的力和力矩符号，不将负Fz截断；旧解析阻尼限幅、bearing/drag缩放及footprint_keep
只作用于解析后端，不对神经网络预测再套一遍解析公式。

## 时间与重置

- 原网络一步历史间隔是 **0.005 s**，因此神经网络任务使用 `sim.dt=0.005`、`decimation=4`。
- 外力/历史更新200 Hz；策略、价值网络、奖励、接触计时和落足事件保持50 Hz。
- 加载时及推理时校验历史时间步，错误步长会明确报错。
- 每次环境重置后，第一次5 ms预测因缺少历史而施加零力和零力矩；随后正常推理。
- 局部重置只清除被重置环境的速度历史、预测和接触缓存，不影响其他环境。
- 继续由`FootSandProxyEnv`逐物理步调用动作项，避免Newton把外力更新折叠成一次策略步。

当前接入在普通刚体物理步之前根据当前状态预测下一步外力，
与原采集“刚体推进后、MPM求解前”的耦合相位有差别；CPU接口/数值检查不能替代GPU闭环测试。
网络原世界X/Y对称规则保持不变，既有任务初始化也保留，任意随机yaw的泛化仍需实际验证。
刚性地板接触继续由MJWarp求解器处理。Proxy任务额外配置`foot_ground_contacts`传感器，
只读取四个轮端与`/World/ground`碰撞形状之间的实际反力（包含摩擦），不把自身碰撞当成地面支撑。
每个20ms控制周期在物理推进完成后，将最后一次施加的代理力与当前地板反力相加：
`Fz_support = Fz_proxy + Fz_ground`，仅当`Fz_support > 5 N`时判为支撑，其他情况累计卸载时间。
腾空、抬升、前向迈步和相位奖励共用这一50Hz接触状态；同一步重复读取不会重复计时。
地板反力仅供判定，不会再次作为外力施加。`force_w`、外力历史和`fz_mean`日志仍只记录代理力。
Actor/Critic观测、动作和网络结构保持不变；MPM任务仍使用自己的冲量采样器。

Walk和共享沙地任务的抬升、相位奖励当前都使用对角分组：FL+HR与FR+HL交替。
`rotation_gait_status`调用`ProxyRotationGaitStatus`：一组两脚均卸载、另一组两脚均支撑；
沙地任务中，两条摆动腿都达到`min_air_time`后，以两腿较小的归一化抬升进度计分，只奖励新的最大进度。
沙地抬升参考仍为各足最近支撑时的世界Z，目标20cm，无需抬出沙面。
Walk单独使用`reward_mode="rough"`、`min_air_time=0`：摆动组平均高度比支撑组高出20cm时，
每个控制步持续得1分；仍要求对角双脚卸载、另一对角双脚支撑。
Walk的`feet_air_time.max_air_time=None`，取消计分时间上限，0.5s计分基准和短腾空负分保留。
沙地任务的进度模式、腾空时间门槛及1s计分上限保持不变。
`rotation_gait_symmetry`调用`ProxyRotationGaitSymmetry`，4s滑动窗口中两种对角状态的占比分别为pA/pB，
原始分数为`2*sqrt(pA*pB)`，最高1；不设固定周期，不要求相位占用满足抬升的最短卸载时间。
原三腿支撑爬行函数仍保留，可切回。现有权重和批量各组的时间门槛保持不变。

## 单环境训练与受力曲线验证

在项目根目录执行（使用GPU，运行真实PPO训练）：

```bash
uv run python scripts/proxy_training_trace.py --iterations 100
```

默认从`height0.62deg15/model_11000.pt`加载策略和价值网络，不恢复优化器；单环境额外训练100轮，
使用当前任务源码默认参数。可用`--checkpoint PATH`更换起点，或`--from-scratch`从零训练。
脚本仅在子进程用原生`--external_callback`安装观测钩子，不修改训练函数、奖励或网络。
每次50Hz接触判定发布后、环境重置前采集，重复奖励读取不重复记录。

输出到`output/proxy_training_trace/<时间戳>/`：

- `trace.npz`：时间、FL/FR/HL/HR足端有符号深度、网络输入深度、代理/刚性/合力的世界XYZ分量、
  支撑/落足、卸载时间、重置标记、回合步数和速度指令。
- `signals.json`：采样周期、沙面、5N阈值、模型与奖励权重；`command.json`保存完整启动命令。
- `training.log`：原生训练输出，策略checkpoint与TensorBoard仍写入原生`logs/rsl_rl/`目录。
- `foot_depth_forces.png`/`.pdf`：四足深度与Fz曲线；`foot_depth_forces_first_8s.png`为前8秒细节。
- `summary.json`：深度/受力范围、重置次数，以及刚性地板反力补充支撑判定的样本数。

`depth_m`在物理推进后按同一胶囊足底公式计算；`input_depth_m`是最后一个5ms物理步之前，
实际输入网络的深度，两者可能相差一步。深度正值表示下陷，当前刚性底板约对应20cm下陷。
合力仅用于判定；曲线中的刚性力为判定合力减代理力，与实际参与判定的分量严格一致。
中途异常也会尽量保留已经采集的数据。已有数据可重新绘图：

```bash
uv run python scripts/proxy_training_trace.py --plot-only output/proxy_training_trace/<时间戳>
```

## 切回解析后端

Python配置（创建仿真之前）：

```python
from M20_newton_training.sand_proxy.analytical.foot_model import DEFAULT_MODEL_PATH

cfg.actions.sand_proxy.model_kind = "analytical"
cfg.actions.sand_proxy.model_path = str(DEFAULT_MODEL_PATH)
cfg.sim.dt = 0.0005
cfg.decimation = 40
cfg.sim.render_interval = 40
```

或者在原Proxy训练命令后追加：

```bash
env.actions.sand_proxy.model_kind=analytical \
env.actions.sand_proxy.model_path=src/M20_newton_training/sand_proxy/analytical/calibrated/m20_foot_sand.json \
env.sim.dt=0.0005 env.decimation=40 env.sim.render_interval=40
```

两套模型的奖励/网络结构共享，切换后端不自动改变奖励权重。
旧`python -m sand_proxy_lab.validation.check_runtime`显式验证解析后端；
旧refit/deploy只发布解析JSON到`analytical/calibrated/`，不会覆盖神经网络权重。
可选深度终止项默认仍禁用；NN没有声明标定深度上限，若启用需显式传`limit_m`。

## 替换神经网络权重

运行只读取`neural/`中的副本，不读取实验目录。重新训练后需复制新的权重和兼容运行时，
核对输入列、轮序、几何、质心参考和历史步长，并更新部署记录。
本次权重及独立推理源码与`sand_pinn_lab/model`逐字节一致，来源哈希见`deployment.json`。
