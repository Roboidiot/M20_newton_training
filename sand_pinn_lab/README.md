# Sand PINN Lab

独立的M20沙地实验工具。

- [数据采集与物理参数接口](collect/README.md)
- 默认物理参数：[collect/configs/default.json](collect/configs/default.json)
- 数据目录：`data/`

```bash
.venv/bin/python -m sand_pinn_lab.collect --episodes 20 --episode-seconds 10
```

原命令 `python -m sand_pinn_lab` 仍可使用，转发到采集入口。
