# 足端受力影子实验

本包只记录真实 MPM 受力并调用当前代理预测，不将代理力施加给机器人。

完整的物理对齐、坐标系、模型公式、数据结构及操作步骤见 [标定与训练完整流程](../docs/标定与训练完整流程.md)。

| 入口 | 用途 |
|---|---|
| `python -m sand_proxy_lab.shadow` | Walk checkpoint 的单回合 MPM 对照 |
| `python -m sand_proxy_lab.shadow.random_agent` | IsaacLab RandomAgent，多回合及摔倒重置 |
| `python -m sand_proxy_lab.shadow.refit` | 单连续回合的滑移响应标定 |
| `python -m sand_proxy_lab.shadow.episode` | 无仿真提取回合并生成统计和诊断图 |
| `python -m sand_proxy_lab.shadow.diagnostics` | 重画已提取回合的原始样本误差诊断 |
| `python -m sand_proxy_lab.shadow.compare_versions` | 同轨迹回放基底与修正模型 |

在项目根目录使用 `.venv/bin/python`，为入口追加 `--help` 查看参数。默认新数据写入 `output/sand_proxy/terrains/<地形名>/shadow/`，拟合结果写入 `output/sand_proxy/terrains/<地形名>/fit/contact/`。训练任务不会导入实验工具。

现有模型拟合使用 `shadow/walk_forward_001`；`shadow/refit_validation_offset065` 和随机首回合用于独立验证。`random_30s_001` 实际已保存 7 秒后停止，完整首回合 6 秒，不能把目录名称作为完成时长。
