# M20 沙地 PINN 数据采集

独立采集工具：不导入 `sand_proxy_lab`，不读取代理模型、拟合系数或策略 checkpoint。
复用项目的 M20 机器人和 `WalkOnSand` 物理场景，采集逻辑、随机策略和记录器均在本目录。

## 目录

```text
sand_pinn_lab/
  collect/
    __main__.py           # 命令行入口
    config.py             # 采集参数、环境组装
    physics.py            # 统一物理参数接口和校验
    configs/default.json  # 完整默认物理参数
    configs/example.json  # 可覆盖默认值的示例
    collector.py          # 回合采集、重置与运行元数据
    record.py             # MPM逐步记录及沙力/力矩归约
    geometry.py           # USD几何与坐标变换
    surface.py            # 沙面估计
    policy.py             # 慢变化随机动作
    storage.py            # 分块保存
    validate.py           # 数据完整性检查
    tests/                # 配置、数值与存储验证
  data/                   # 采集输出
```

## 物理参数接口

以后更换物理参数，使用本目录下的JSON文件或命令行 `--set`，无需修改训练任务或shadow源文件。
优先级：**`configs/default.json` → `--physics-config`文件 → `--set`**。
未指定参数沿用默认值，未知字段、错误类型和不合法的时间比例会报错。
参数在仿真创建前应用；修改文件后需重新启动一次采集。

| 分组 | 常用字段 | 含义/单位 |
| --- | --- | --- |
| `material` | `density`, `friction`, `damping` | 密度kg/m³、沙内部摩擦系数、弹性阻尼松弛时间s |
| `material` | `young_modulus`, `poisson_ratio`, `viscosity` | 杨氏模量Pa、泊松比、塑性黏度Pa·s |
| `material` | `yield_pressure`, `yield_stress`, `tensile_yield_ratio` | 压力/偏应力屈服参数Pa、拉压比 |
| `material` | `hardening`, `dilatancy`, `hardening_rate`, `softening_rate` | 硬化、剪胀及演化参数 |
| `mpm` | `max_iterations`, `tolerance`, `solver`, `warmstart_mode` | 求解迭代、容差、算法及热启动 |
| `mpm` | `voxel_size`, `grid_type`, `grid_padding` | 网格尺寸m、网格类型、填充层数 |
| `mpm` | `collider_basis`, `velocity_basis`, `strain_basis`, `collider_velocity_mode` | 接触/速度/应变离散及碰撞体速度计算方式 |
| `mpm` | `transfer_scheme`, `integration_scheme`, `critical_fraction`, `air_drag` | 传递、积分、临界体积分数及背景阻力 |
| `particles` | `particles_per_cell`, `jitter_fraction` | 每轴粒子密度倍率、位置扰动宽度比例 |
| `coupling` | `mode`, `mass_scale`, `iterations`, `robot_substeps` | lagged/staggered、代理质量倍率、外层迭代、刚体子步 |
| `time` | `physics_dt_s`, `control_dt_s` | 物理与控制时间步s，后者必须是前者的整数倍 |

`material.friction`是沙内部摩擦，不是机器人形状的接触摩擦。
`coupling.mass_scale`只改变MPM代理质量/惯量，不改变真实机器人质量。
沙床几何和碰撞形状继续继承M20场景，本接口不更改沙床长宽和托盘尺寸。
改变 `mpm.voxel_size` 会同步更新粒子生成的网格尺寸；扰动宽度按粒子间距重算，沙厚向下取整到完整粒子层。
控制步数、采样时间、力=冲量/dt及数据帧数均使用实际配置。粒子生成参数只影响仿真，不导出粒子数据。
配置和采集错误会在启动时校验，后端的算法组合约束仍由Newton检查。

例如新建 `my_sand.json`（可以只写修改项）：

```json
{
  "material": {"density": 1600.0, "friction": 0.75},
  "mpm": {"max_iterations": 500, "tolerance": 1e-6, "collider_basis": "S2"},
  "coupling": {"mode": "lagged"}
}
```

```bash
.venv/bin/python -m sand_pinn_lab.collect --physics-config my_sand.json
```

也可以直接使用附带的示例文件：

```bash
.venv/bin/python -m sand_pinn_lab.collect --physics-config sand_pinn_lab/collect/configs/example.json
```

单次实验用命令行覆盖；加 `--print-config` 可在无GPU时核对合并结果，不启动仿真：

```bash
.venv/bin/python -m sand_pinn_lab.collect \
  --set material.friction=0.8 \
  --set mpm.max_iterations=500 \
  --set mpm.tolerance=1e-6 \
  --set coupling.mode=lagged

.venv/bin/python -m sand_pinn_lab.collect --set time.physics_dt_s=0.0025 --print-config
```

Python接口（直接修改后也会在应用时重新校验）：

```python
from sand_pinn_lab.collect import load_physics_config, apply_physics_config

physics = load_physics_config("my_sand.json", overrides=["mpm.tolerance=1e-6"])
physics.material["density"] = 1800.0
# cfg是新创建的M20环境配置，cfg.sim.physics已选择newton_sand。
apply_physics_config(cfg, physics)
```

每次采集将合并结果写入运行目录的 `physics.json` 和 `metadata.json.physics`，
`environment.json`记录应用后的完整环境配置。后续修改默认文件不会改写已有数据的配置记录。

## 运行

在项目根目录（包含 `.venv`、`src` 和本目录）执行：

```bash
.venv/bin/python -m sand_pinn_lab.collect --episodes 20 --episode-seconds 10
```

带画面：

```bash
.venv/bin/python -m sand_pinn_lab.collect --episodes 20 --episode-seconds 10 --viz newton
```

先做两轮短检查，覆盖重置：

```bash
.venv/bin/python -m sand_pinn_lab.collect --episodes 2 --episode-seconds 1 --warmup 0.1
```

需要现有 IsaacLab/Newton 环境和可访问的 CUDA GPU。上述 10 秒是**仿真时间**。
每次运行创建新的 `sand_pinn_lab/data/random_<时间>/`；`--output-dir` 可指定一个不存在的目录。
Ctrl+C/SIGTERM 在控制步边界停止并保存剩余数据。强制杀死进程时，已完成的原子写入块仍保留。
数据不自动追加到旧运行；划分数据集时应按完整回合划分。

## 默认物理配置

| 项目 | 值 |
| --- | --- |
| 并行环境 | 1 |
| MPM / 外层物理时间步 | 0.005 s（200 Hz） |
| 控制周期 | 0.02 s（50 Hz），decimation=4 |
| 刚体内部子步 | 2（每个 0.005 s 步内） |
| MPM 最大迭代 / 容差 | 250 / 1e-5 |
| 网格 / 速度 / 应变 / 接触基 | sparse、2 cm / Q1 / P0 / pic27 |
| 粒子网格 | 每轴 1 粒子，cell_center，jitter=0.3×间距 |
| 耦合 | staggered，外层 iterations=1 |
| 沙床 | 名义 2.4×2.4×0.30 m，密度1700 kg/m³，内部摩擦0.7，阻尼0.005 s |
| 物理来源 | 与 shadow 相同的 `WalkOnSand` 场景；实际完整配置见 environment.json |

关闭外层 CUDA graph replay，确保每个 MPM 步执行采集钩子。
每轮先把机器人放在高处，让沙沉降0.3秒（不计入采样），再放到测得沙面以上0.58米。
每轮完整记录10秒（2000帧），默认20轮共40000帧。
每轮前0.5秒使用零动作，属于10秒采样的一部分。
到时重置机器人、粒子位置/速度、MPM内部状态和耦合反馈，再沉降。
初始沙床重复使用同一粒子排列；每轮随机动作种子为 `seed+episode_id`。
跌倒、越出沙床均不提前重置，保留相关标志和非足端受力；非有限数据导致失败并保留已采集块。

## 随机策略

16维动作目标独立采样 `U(-1.5,1.5)`；
每0.5–1.0秒改变一次目标，用0.2秒 smoothstep 从旧动作过渡到新动作。
按当前任务动作缩放，对应髋外展约±0.1875 rad、髋屈伸和膝约±0.375 rad的默认姿态偏移，轮速约±7.5 rad/s。
实际处理后的目标也逐帧保存；实际运动仍受执行器和关节限制影响。
这是一种关节激励策略，不保证站立或行走。

```bash
.venv/bin/python -m sand_pinn_lab.collect --action-limit 2.5 --hold-seconds 1 2 --transition-seconds 0.3
```

## 文件和数据约定

```text
data/random_<时间>/
  metadata.json                 # 参数、版本、时间对齐、回合状态、身体/关节索引
  environment.json              # 完整环境配置
  physics.json                  # 本次运行合并后的材料/MPM/耦合/时间参数
  geometry.json                 # 从实际USD读取的四个轮足胶囊参数、USD层哈希
  body_properties.npz           # 真实机器人各刚体质量、惯量、局部质心
  episode_000/
    episode.json
    schema.json                # 每帧字段形状与dtype
    chunks.json                # 可直接按顺序加载的分块清单
    chunk_00000.npz            # 默认每1秒保存一个200Hz数据块
  episode_001/ ...
```

NPZ 无 pickle/object 数组。长度单位m、时间s、速度m/s、角速度rad/s、力N、力矩N·m。
所有位姿、速度和力都在世界坐标系；四元数 **xyzw**。
`pose=[x,y,z,qx,qy,qz,qw]`，`com_twist=[vx,vy,vz,wx,wy,wz]`，线速度位于质心。
身体/关节顺序见 `metadata.json.mapping`；四轮与四小腿顺序为 FL、FR、HL、HR。
小腿对应 `*_knee` 刚体；其余身体的独立受力全部保留在 `body_*` 数组中。

主要字段（N=块内帧数，B=机器人刚体数，J=16）：

| 字段 | 形状 / 含义 |
| --- | --- |
| `episode_time_start_s`, `episode_time_s`, `dt_s` | [N]，MPM求解区间起止和时长 |
| `time_s`, `simulation_time_s` | [N]，拼接采样时间 / 包含沉降的仿真时间 |
| `episode_id`, `frame_index`, `control_step`, `warmup` | 回合、帧、控制步、预热标志 |
| `body_pose_w`, `body_com_position_w_m`, `body_com_twist_w` | [N,B,7/3/6]，真实机器人求解器输出 |
| `body_link_velocity_w_m_s` | [N,B,3]，刚体坐标系原点的线速度 |
| `base_pose_w`, `base_com_twist_w` | [N,7/6]，机身位姿和质心速度 |
| `joint_position_rad`, `joint_velocity_rad_s` | [N,J]，关节角和角速度 |
| `joint_effort_input_Nm` | [N,J]，刚体求解器control.joint_f输入，不含约束反力 |
| `action`, `random_target`, `random_segment` | 应用动作、随机目标、目标编号 |
| `joint_position_target_rad`, `wheel_velocity_target_rad_s` | [N,12/4]，动作项转换后的目标（执行器延迟前） |
| `wheel_pose_w`, `wheel_center_w_m`, `wheel_center_velocity_w_m_s` | 轮足位姿、胶囊中心和中心速度 |
| `wheel_angular_velocity_w_rad_s`, `wheel_axis_w` | 轮足角速度和胶囊轴方向 |
| `wheel_bottom_w_m`, `wheel_bottom_velocity_w_m_s` | 倾斜胶囊的真实最低支撑点和该点刚体速度 |
| `wheel_sinkage_reference_m` | [N,4]，沉降后初始沙面减最低点高度，负值表示在沙面上方 |
| `wheel_surface_height_w_m`, `wheel_surface_valid` | [N,4]，当前局部沙面估计及有效性 |
| `wheel_sinkage_local_m` | [N,4]，当前局部沙面减最低点高度；仅valid=true时使用 |
| `wheel_column_top_w_m`, `wheel_column_top_valid` | 轮中心所在竖直柱的粒子顶部高度；可能落在轮足挖出的坑内 |
| `wheel_surface_patch_w_m`, `wheel_surface_patch_valid` | [N,4,13,13]，周围4cm网格的柱顶和掩码 |
| `wheel_surface_patch_center_xy_m` | [N,4,2]，patch中心网格坐标；轴顺序为x,y |
| `body_mpm_force_w_N`, `body_mpm_torque_com_w_Nm` | [N,B,3]，各刚体沙力和绕各自代理质心的力矩 |
| `wheel_mpm_*`, `shin_mpm_*` | [N,4,3]，轮足、小腿的沙力/质心力矩 |
| `other_mpm_force_w_N`, `other_mpm_torque_base_com_w_Nm` | 除轮足/小腿之外的合力、移到机身代理质心后的合力矩 |
| `total_mpm_force_w_N`, `total_mpm_torque_base_com_w_Nm` | 全机器人沙力及绕机身代理质心的总力矩 |
| `mpm_proxy_body_*` | MPM实际读取的代理位姿、质心和质心速度 |
| `wheel_contact`, `base_up_z`, `base_clearance_m`, `base_outside_tray` | 接触阈值5N、姿态、离沙面高度、越界标记 |

**时序：** 每条记录在MPM入口读取已经完成本次刚体更新的状态，随后运行MPM，读取整个物理时间步的冲量。
因此机器人状态对应区间末端，而局部沙面来自区间起始的沙粒；力是这段区间的平均值。
`staggered`代理位姿对应步末，`lagged`对应步初；实际代理输入另存于`mpm_proxy_body_*`。
若耦合迭代大于1，仅保存最后一轮MPM求解结果，不把重复求解当成时间推进。
这些是耦合求解器实际使用的输入/输出，不能把所有字段当成同一时刻的连续体真值。
沙力仅来自MPM：不包含重力、驱动力矩或底部刚性平面的接触力。
全身合力矩先做力矩平移再求和，不能直接相加各身体自己的质心力矩。

**局部沙面：** 先按4cm柱计算 `max(particle_z + particle_radius)`，再取轮中心周围
半径0.15–0.24m环带内有效柱顶的中位数（至少3柱）。这用于避开轮足覆盖区，属于估计量；
局部坡面、飞散沙粒、空洞和沙床边缘会影响结果。无效值填0，必须结合掩码使用。
初始参考沙面使用中央未扰动区域柱顶中位数，算法与shadow一致。
胶囊轴恰好水平时最低点不唯一，选中心下方；倾斜时会切换到更低的胶囊端部。

**输出范围：** 只保存机器人状态、动作、各部位沙力/力矩、沙面高度与局部高度图等汇总量。
不保存沙粒位置、速度、应力、速度梯度、逐粒子属性或每轮初末粒子快照。
采集器在内存中读取粒子位置和半径来计算沙面与下陷深度，MPM仿真仍正常推进。
运行元数据使用 `schema_version=2`，并标记 `particle_state_saved=false`。

## 检查与加载

```bash
.venv/bin/python -m sand_pinn_lab.collect.validate sand_pinn_lab/data/random_<时间>
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sand_pinn_lab/collect/tests -q
```

```python
import json
from pathlib import Path
import numpy as np

episode = Path("sand_pinn_lab/data/random_<时间>/episode_000")
manifest = json.loads((episode / "chunks.json").read_text())
for chunk in manifest["chunks"]:
    with np.load(episode / chunk["file"], allow_pickle=False) as data:
        t = data["episode_time_s"]
        force = data["wheel_mpm_force_w_N"]  # [N, 4, 3]
        depth = data["wheel_sinkage_reference_m"]  # [N, 4]
```
