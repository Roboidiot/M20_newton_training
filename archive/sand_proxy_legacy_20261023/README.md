# 沙地代理旧版本封存：2026-10-23

这里存放不再参与当前训练、标定和验证入口的历史内容。当前流程见 [`sand_proxy_lab/`](../../sand_proxy_lab/README.md)。压缩包在迁移时已逐文件核验 SHA256，数据未因整理而丢弃。

| 文件 | 内容 |
|---|---|
| `legacy_code_and_models.tar.gz` | 原 `sand_calibration` 全目录：旧 v1、三轴分段受力场、v3 主受力多项式、旧模型/说明；重构前影子工具与运行时快照、旧专用测试及原基底备份 |
| `legacy_data_and_trials.tar.gz` | 早期标定批次、未使用的 179 组小腿数据、旧拟合模型和图表、失败修正候选、重复种子回放及 smoke 试采 |
| `legacy_ad_hoc_analysis.tar.gz` | 原来临时放在数据目录里的两个专用绘图脚本；通用版本现已并入 `sand_proxy_lab/shadow/` |
| `migration_manifest.json` | 原路径 → 现路径或 `压缩包::成员路径`，含原文件 SHA256 与归档/现用角色 |
| `SHA256SUMS` | 三个封存包的整体 SHA256 |

当前模型需要的 179 组足端原始数据、原始整机拟合轨迹、独立验证及随机第一回合，保留在 `output/sand_proxy/terrains/<地形名>/`。弹塑性基底属于当前模型的必要组成，没有作为废弃实现移走。

原始数据、旧报告和部署 JSON 内的历史路径没有批量改写，避免来源哈希失效。使用迁移清单查找现位置。归档程序不会被当前流程导入，也没有旧命令兼容入口。

检查封存包完整性：

```bash
cd archive/sand_proxy_legacy_20261023
sha256sum -c SHA256SUMS
tar -tzf legacy_code_and_models.tar.gz
```

如需查看旧程序，请解包到单独的历史目录，不覆盖当前工作区：

```bash
mkdir -p /tmp/m20_sand_proxy_history
tar -xzf legacy_code_and_models.tar.gz -C /tmp/m20_sand_proxy_history
```

封存包和现用 `output/` 均默认被 Git 忽略；本索引与校验/迁移清单可随核心代码保存。封存包保持原样，后续模型发布备份使用 `output/sand_proxy/terrains/<地形名>/deployment_backups/`。
