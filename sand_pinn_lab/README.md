# Sand PINN Lab

独立的M20沙地实验工具。

- [数据采集与物理参数接口](collect/README.md)
- 默认物理参数：[collect/configs/default.json](collect/configs/default.json)
- 数据目录：`data/`

```bash
.venv/bin/python -m sand_pinn_lab.collect --episodes 20 --episode-seconds 10
```

使用group2的`model_12500.pt`在MPM中采集同格式行走数据：

```bash
bash sand_pinn_lab/experiment/collect_group02_mpm.sh
```

详见[已训练策略采集](collect/README.md#已训练策略在mpm中的行走采集)。

原命令 `python -m sand_pinn_lab` 仍可使用，转发到采集入口。
