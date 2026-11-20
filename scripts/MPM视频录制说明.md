# MPM沙地表面视频

在项目根目录运行：

```bash
uv run python scripts/record_mpm_surface.py \
  --checkpoint logs/rsl_rl/deeprobotics_m20_walk_on_sand_proxy/2026-11-19_18-27-07_72e5ee_05_proxy_05_combined/model_8600.pt \
  --seconds 20 \
  --output-dir output/mpm_surface_video/model_8600_20s
```

默认录制一个环境、固定前进指令0.2m/s、1280×720、50fps。
`--seconds 20`对应1000个控制步，成片20秒，仿真运行耗时可以更长。
`--speed`可调整前进指令，`--width`和`--height`可调整分辨率。

只显示MPM粒子、不重建沙面网格：在录制命令中加入`--sand-display particles`。
此模式设置Newton可视化的`show_particles=True`，不创建`ParticleSurface`和
`ParticleSurfaceRenderer`，因此`--surface-voxel`等重建参数不参与显示。
视频前缀变为`mpm_sand_particles`。默认`--sand-display surface`保持原来的连续沙面效果。
粒子模式仍需渲染粒子球来生成视频，省去的是沙面重建；MPM物理参数保持不变。

提高到1080p、100fps，并细化沙面：

```bash
uv run python scripts/record_mpm_surface.py \
  --checkpoint logs/rsl_rl/deeprobotics_m20_walk_on_sand_proxy/2026-11-19_18-27-07_72e5ee_05_proxy_05_combined/model_8600.pt \
  --seconds 20 --width 1920 --height 1080 --fps 100 \
  --surface-voxel 0.015 --crf 18 \
  --output-dir output/mpm_surface_video/model_8600_1080p100
```

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--width / --height` | 1280 / 720 | 最终视频像素数；必须都是偶数。可设1920/1080、2560/1440、3840/2160。 |
| `--fps` | 50 | 输出帧率1～200。50和100fps与200Hz物理步整除，采样间隔均匀。 |
| `--supersample` | 1 | 设2后按宽、高各2倍渲染，再用Lanczos缩小，改善边缘锯齿；渲染像素数为4倍。 |
| `--crf` | 18 | H.264质量，越小压缩损失越少、文件通常越大；可用16～18。 |
| `--preset` | medium | 编码速度/压缩效率，可选fast、medium、slow。 |
| `--surface-voxel` | 0.02m | 沙面重建网格精度，0.015m更细。 |
| `--surface-kernel` | 0.045m | 重建核半径，控制平滑程度；先保持默认，减小过多可能产生孔洞。 |
| `--surface-max-cells` | 按体素自动估算 | 网格容量，默认以0.02m时750000格为基准，按体素尺寸的反立方缩放。 |
| `--eye X Y Z` | 2.8 -3.2 2.5 | 世界坐标下的相机位置。 |
| `--lookat X Y Z` | 0 0 0.25 | 相机注视点。 |
| `--focal-length` | 12 | 相机焦距，数值增大可缩窄视野、放大主体，注意机器人走出画面。 |
| `--renderer` | gl | GL已经验证；rtx为可选路径追踪后端。 |
| `--sand-display` | surface | particles只显示原生粒子，跳过沙面网格重建。 |
| `--rtx-quality` | 32 | RTX质量目标，对GL无效。 |

更注重边缘画质时，可在上面的命令后加`--supersample 2 --crf 16`。
1080p加2倍超采样实际渲染3840×2160，耗时和显存需求会增加；先录制1秒预览。
网格体素从0.02m减到0.01m时，同体积格子数约为8倍。它提高表面几何精度，
不会改变粒子数量、物理精度，也不会自动产生沙粒纹理。当前显示的是沙色连续表面。

## 帧率与物理时间

控制频率仍为50Hz，物理步长仍为0.005s（200Hz），没有改变训练或仿真步长。
新版录制器在Newton已有的内部物理步之后采集实际状态：

- 20秒、50fps：1000帧，每隔4个物理步采样。
- 20秒、100fps：2000帧，每隔2个物理步采样。
- 20秒、60fps：1200帧，在200Hz网格上按15/20ms交替间隔取样，有至多一个物理步的时间量化误差。

提高帧率不会加速播放，也没有插帧；要均匀采样，优先25、50、100、200fps。
时长先对齐20ms控制步，并要求该时长对应整数个输出帧。
每帧立即送入FFmpeg编码，不缓存整段原始视频到内存；增加帧率仍会增加渲染与重建耗时。

脚本依赖当前IsaacLab 3.0 Newton的内部`_run_solver_substeps`钩子，
因为一次`sim.step()`会执行多个物理步。钩子仅存在于本次录制子进程，
不改库文件、控制降采样或受力；要求`use_cuda_graph=False`（newton_sand当前默认值）。

## 可选RTX渲染

`--renderer rtx --rtx-quality 64`使用Newton RTX和studio照明环境。
当前虚拟环境没有安装`ovrtx`，所以脚本会提前提示；本机尚未验证RTX成片，
不要把GL的成功验证等同于RTX已经可用。先配置匹配本地Newton版本的RTX依赖，
再录制1秒测试，确认动态网格、光照和显存开销后再长录。

脚本调用原生`isaaclab play`，使用`WalkOnSand`和`physics=newton_sand`。
默认使用Newton GL和EGL进行无窗口录制；子进程清除`DISPLAY`以避免X11与EGL上下文混用。
原生play模式的键盘指令仅在本次录制进程中替换为固定前进指令。

沙面由当前MPM粒子位置经过`newton.geometry.ParticleSurface`重建，
每个录制帧通过`ParticleSurfaceRenderer`更新不透明、粗糙的沙色网格，隐藏粒子球。
表面重建体素0.02m、核半径0.045m；这些参数只影响显示，不修改MPM物理参数。
相机固定观察完整沙地，采用任务原有回合终止和重置行为。
训练配置、策略网络和checkpoint参数均不修改。

输出目录包含：

- `mpm_sand_surface_model_8600_0000.mp4`：视频；同目录再次录制会增加末尾编号。
- `recording.json`：checkpoint、前进速度、时长、步数与分辨率。
- `command.json`：实际调用的原生play参数。
- `recording.log`：仿真和录制日志。
- `surface_statistics.json`：粒子数、重建帧数与最大三角形数量。
- `encoding_statistics.json`：实际视频路径、帧数、帧率、时长和物理频率。

检查视频：

```bash
ffprobe -v error -show_entries format=duration \
  -show_entries stream=width,height,nb_frames,r_frame_rate \
  -of json output/mpm_surface_video/model_8600_20s/mpm_sand_surface_model_8600_0000.mp4
```
