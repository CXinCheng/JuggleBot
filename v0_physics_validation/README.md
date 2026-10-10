# v0 — Physics Validation：固定杯子接自由落体小球

Robot Juggling 项目的第一版物理原型：小球在重力作用下落入一个固定的开口杯中，程序判定是否接住。
没有机械臂、没有强化学习。此文件夹作为 v0 存档保留，后续版本请在新文件夹中开发，以便写 report 时重现数据。

## 文件

| 文件 | 内容 |
|---|---|
| `mjmodel.xml` | MJCF 场景（地面、灯光、相机、杯子、小球、接触参数）。**场景参数的唯一来源** |
| `physics_validation.py` | 试验逻辑：重置、随机初始偏移、仿真循环、成功判定、命令行参数 |

脚本从编译后的模型中读取球半径、球初始位置和杯子内腔尺寸，因此只改 `mjmodel.xml` 即可，成功判定会自动适配。
要求：geom 名称 `ball`、`cup_base`、`cup_wall_px/nx/py/ny` 和关节 `ball_free` 保持不变；杯子保持轴对齐（不旋转）。

## 运行（在仓库根目录）

```bash
uv sync                                                                    # 安装依赖

uv run python v0_physics_validation/physics_validation.py                  # 可视化单次试验
uv run python v0_physics_validation/physics_validation.py --headless --trials 10 --seed 42
uv run python v0_physics_validation/physics_validation.py --scene path/to/variant.xml
```

macOS 上可视化需要用 `mjpython` 替代 `python`。

## 用 viewer 查看 / 编辑场景

```bash
cd v0_physics_validation
uv run python -m mujoco.viewer --mjcf=mjmodel.xml
```

- **必须在本文件夹内启动**：viewer 的 File → **Save xml** 会把 `mjmodel.xml` 写到*启动时的当前目录*，在这里启动才会直接覆盖本文件。
- 手动修改 `mjmodel.xml` 后，点击 Simulation 面板中的 **Reload** 即可看到效果。
- Save xml 会根据当前模型重新生成文件：XML 中的注释基本会丢失，等于 MuJoCo 默认值的属性会被省略（如 `gravity="0 0 -9.81"`），但模型是等价的。
  已验证：原始文件与重新保存的文件在 `--seed 42` 下结果完全一致。
- 建议保存前先 `git commit`，方便用 `git diff` 查看 viewer 改了什么。

## 参数

**场景（`mjmodel.xml`）**

| 参数 | 值 |
|---|---|
| 重力 / 时间步长 / 积分器 | 9.81 m/s² / 1 ms / implicitfast |
| 小球 | 半径 0.035 m，质量 0.057 kg，初始高度 0.75 m |
| 杯子内腔 | 0.16 × 0.16 m，壁厚 0.01 m，底厚 0.01 m，壁高 0.12 m（杯口 z = 0.13 m） |
| 接触 | `solref="0.003 0.5"`，`solimp="0.95 0.99 0.001"` |
| 小球摩擦 | `friction="0.8 0.005 0.0005"`，`condim="6"`（滑动 + 扭转 + 滚动） |

**试验（`physics_validation.py` 顶部常量）**

| 参数 | 值 |
|---|---|
| 初始水平随机偏移 | ±0.03 m（均匀分布，x/y 独立） |
| 成功判定 | 球心在杯内（考虑半径，容差 2 mm），速度 < 0.05 m/s，持续 ≥ 0.5 s |
| 超时 | 5 s 仿真时间 |

## 参考结果

环境：Windows 11，Python 3.12.5，MuJoCo 3.15.0，NumPy 2.5.3。

| 命令 | 结果 |
|---|---|
| `--headless --seed 42` | SUCCESS；t = 0.366 s 入杯，t = 1.092 s 判定稳定；最终位置 (+0.0164, −0.0037, +0.0450) m，速度 0 |
| `--headless --trials 10 --seed 42` | 10/10 |
| `--headless --trials 50 --seed 7` | 50/50 |

补充的物理检查（通过临时脚本/场景完成，未存档）：

- 球落在杯外 0.3 m 处 → 正确判定为 FAILED。
- 在杯内以 2 m/s 和 5 m/s 侧向撞墙 → 不穿墙，最终稳定在杯内。
- 从 1.5 m 落下 → 接住，冲击时最大穿透约 4.6 mm。
- 接触参数对比：`solref="0.005 1"` 冲击穿透约 5 mm 且几乎不反弹；`0.003 0.5` 穿透约 2 mm，有几厘米的小反弹，约 1 s 内稳定，故选用后者。
