# M20 足端沙地代理模型工作区

这里维护当前用于 `WalkOnSandProxy` 的**足端弹塑性基底＋MPM 滑移修正模型**。不导入旧 `SandProxy`、分段受力场或九系数主受力多项式。

- **完整说明**：[标定与训练完整流程](docs/标定与训练完整流程.md)：结构、公式、目录、数据、从零标定、影子验证、发布和训练命令。
- **训练运行时**：[`src/M20_newton_training/sand_proxy/`](../src/M20_newton_training/sand_proxy/)。训练只依赖该包及任务接入代码，不依赖这里的标定工具。
- **当前参数**：[`m20_foot_sand.json`](../src/M20_newton_training/sand_proxy/calibrated/m20_foot_sand.json)，schema 2；本次整理没有改变参数文件内容。
- **现用数据/结果**：`output/sand_proxy/terrains/<地形名>/`，包括 179 组足端探针、整机 MPM 拟合轨迹及独立验证；默认不提交 Git。
- **旧版本封存**：[归档索引](../archive/sand_proxy_legacy_20261023/README.md)，含完整迁移清单及 SHA256。

所有命令在项目根目录执行：

```bash
# 只查看现有探针数据是否齐全，不运行物理
.venv/bin/python -m sand_proxy_lab.calibration status

# 指定新地形：所有阶段都使用同一个名称
.venv/bin/python -m sand_proxy_lab.calibration plan --terrain sand_soft_v1
.venv/bin/python -m sand_proxy_lab.calibration collect --terrain sand_soft_v1
.venv/bin/python -m sand_proxy_lab.calibration fit-backbone --terrain sand_soft_v1

# 新流程入口帮助
.venv/bin/python -m sand_proxy_lab.calibration --help
.venv/bin/python -m sand_proxy_lab.shadow --help
.venv/bin/python -m sand_proxy_lab.shadow.random_agent --help
.venv/bin/python -m sand_proxy_lab.shadow.refit --help

# 使用已部署的新代理训练；无需重新采集和拟合
.venv/bin/isaaclab train --rl_library rsl_rl \
  --task M20NewtonTraining-M20-M20-WalkOnSandProxy \
  --num_envs 64 --logger tensorboard --viz none physics=newton_mjwarp
```

当前模型继续用于训练。随机动作影子采集已停止，保留到磁盘的 7 秒记录和完整第一回合分析；本次整理不恢复采集、不启动长时间训练、不重新部署候选参数。
