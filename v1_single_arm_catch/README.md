# v1 — Stage 1 Single-Arm Catching：UR5e 用杯子动态接球

Robot Juggling 项目的 Stage 1：一只 UR5e 机械臂在末端装一个杯子，主动移动去接住一个被抛出的、运动中的小球。
没有强化学习，是一套经典的"预测 → 选点 → IK/规划 → 执行验证"流水线。v0（`v0_physics_validation/`）保持不变，作为物理参数基线。

## 文件

| 文件 | 内容 |
|---|---|
| `scene.xml` | 世界：地面、灯光、相机、发射器标记（纯视觉）、小球、`ready` 关键帧。`<include>` 了 `ur5e_cup.xml` |
| `ur5e_cup.xml` | Menagerie UR5e + 杯形末端执行器（改动见文件头注释） |
| `../assets/universal_robots_ur5e/` | （仓库根目录，各阶段共用）Menagerie 原始 mesh、`LICENSE`（BSD-3），以及未修改的原版 `ur5e.xml` 供对照 |
| `prediction.py` | 1. 弹道预测：已知重力下对观测位置做最小二乘拟合 |
| `kinematics.py` | 杯口位置 + 杯轴方向的 5 维 DLS IK、FK、碰撞检查（在独立的 scratch `MjData` 上） |
| `planning.py` | 2. 接球点选择 + 3. 轨迹生成（quintic / 笛卡尔直线段） |
| `catch_sim.py` | 抛球、观测、重规划、跟踪控制、4. 接球判定、命令行、统计 |

模型来源：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `universal_robots_ur5e`，commit `0059d43`。

## 运行（在仓库根目录）

```bash
uv run python v1_single_arm_catch/catch_sim.py                                     # 可视化单次（黄点=预测轨迹，绿点=接球点）
uv run python v1_single_arm_catch/catch_sim.py --trials 5 --seed 0                 # 可视化多次
uv run python v1_single_arm_catch/catch_sim.py --headless --trials 50 --seed 0     # 成功率统计
uv run python v1_single_arm_catch/catch_sim.py --headless --trials 50 --seed 0 --strategy static     # baseline
uv run python v1_single_arm_catch/catch_sim.py --headless --trials 50 --seed 0 --obs-noise 0.005     # 5 mm 观测噪声
```

macOS 上可视化需要用 `mjpython` 替代 `python`。查看/编辑场景：`cd v1_single_arm_catch && uv run python -m mujoco.viewer --mjcf=scene.xml`。

## 流水线

**0. 抛球（规划器不可见）**：发射器在 (1.8, 0, 0.65) m 附近随机扰动；在工作空间内随机取一个"穿越点"
x∈[0.35, 0.65]、y∈[−0.3, 0.3]、z∈[0.45, 0.70] m 和飞行时间 T∈[0.8, 1.1] s，反解初速度
`v0 = (p_target − p0 − ½gT²)/T`。球速在接球时约 4.5–6 m/s，几乎是竖直下落（与竖直方向夹角约 15–25°）。

**1. 轨迹预测**（`prediction.py`）：MuJoCo 默认无空气阻力，球的轨迹是抛物线 `p(t) = p0 + v0·t + ½g·t²`。
重力已知，p0、v0 是线性未知量 → 每轴独立做最小二乘。以 100 Hz 采样球心位置（可加高斯噪声 `--obs-noise`），
前 50 ms 只观测；之后每 50 ms 用全部观测重新拟合一次。

**2. 接球点选择**（`planning.py: CatchPlanner.plan`）：在预测轨迹上每 10 ms 取一个候选接球时刻 t_c，依次过滤：
- 下降段，杯口高度 z∈[0.40, 0.85] m（更低时手腕会碰地，因为手腕吊在杯子下方约 0.2 m）；
- 杯轴 = 来球方向的反方向 −v̂，倾角限制在离竖直 ≤ 40°；
- IK 有解（位置误差 < 10 µm）、关节限位内、无自碰撞 / 地面碰撞；
- 时间可行：从当前期望状态出发、满足关节速度（π rad/s）和加速度（15 / 25 rad/s²）约束的最短 quintic，
  到达预接球位后还要能静止等待 ≥ 0.1 s（重规划时 ≥ 0.02 s）。

在可行候选中选加权关节位移最小者（倾角为次要代价）。cushion 策略下若没有可行的缓冲接法，自动退化为 static 接法。

**3. IK + 运动规划**：
- IK（`kinematics.py`）：杯口位置 3 维 + 杯轴对齐 2 维（绕杯轴的转动自由），阻尼最小二乘；剩下 1 维冗余用
  精确零空间投影拉向 `ready` 姿态。
- 轨迹由三段组成：
  1. **Approach**：关节空间 quintic，从当前期望状态（位置/速度/加速度连续，重规划时无跳变）到预接球位 q_pre，然后静止等待；
  2. **Match**：笛卡尔直线，杯子从静止沿来球方向加速，t_c 时刻杯速 u = min(0.4·|v_ball|, 0.7 m/s)，杯口与球心重合；
  3. **Cushion**：继续沿来球方向 10 cm 减速到 0。
  2、3 段按 1 ms 逐点 IK 离散成关节轨迹。`--strategy static` 跳过 2、3 段，杯子在接球点静止等待。
- 控制：Menagerie 的位置伺服 `τ = kp(ctrl − q) − kd·q̇`。令 `ctrl = q_d + (kd/kp)·q̇_d + τ_ff/kp`，
  其中 `τ_ff = M(q)·q̈_d`，等价于带速度和惯性前馈的 PD 跟踪；重力由 `gravcomp` 补偿。
- 重规划：拟合结果使 t_c 时刻的预测点移动 > 5 mm 时，从当前期望状态重新做选点和规划；Match 段开始前 30 ms 锁定。

**4. 接球判定**：在杯子坐标系下判断球是否在内腔内：径向上整个球在内壁以内，轴向上球心不高于杯沿
（杯深 0.06 m 小于球径，和手掌托球一样，球会露出杯口一截；容差 2 mm），球相对杯子的速度 < 0.05 m/s，
且机械臂已停止后持续 ≥ 0.5 s → `SUCCESS`。其他结果：`NO_FEASIBLE_TARGET`、`MISSED`（从未入杯）、
`BOUNCED_OUT`（入杯后弹出）、`NOT_SETTLED`（超时仍未稳定）。

## 参数

**场景**

| 参数 | 值 |
|---|---|
| 时间步长 / 积分器 / 接触参数 | 1 ms / implicitfast / 同 v0（`solref="0.003 0.5"`, `solimp="0.95 0.99"`） |
| 小球 | 同 v0：半径 0.035 m，质量 0.057 kg，`condim=6` |
| 杯子 | 空心圆柱，大小参照灵巧手团成杯状（为后续换成机械手做准备）：内径 0.10 m，内深 0.06 m，壁厚 5 mm，杯底厚 1 cm，杯口距法兰 0.07 m，质量 0.2 kg。装在法兰上（原 `attachment_site` 位姿）。杯壁由 24 块 box 拼成 |
| UR5e 执行器 | Menagerie 原值：kp = 2000 / 500，kd = 400 / 100，力矩上限 150 / 28 N·m（大关节 / 腕关节） |
| `ready` 姿态 | 杯口在 (0.45, 0, 0.50) m，开口朝上 |

**规划 / 试验**（`planning.py: PlannerConfig`、`catch_sim.py` 顶部常量）

| 参数 | 值 |
|---|---|
| 观测 | 100 Hz，首次规划前观测 50 ms，每 50 ms 重新拟合 |
| 候选接球点 | 步长 10 ms，z∈[0.40, 0.85] m，倾角 ≤ 40° |
| 关节限制（用于计时） | 速度 π rad/s；加速度 15（大关节）/ 25（腕关节）rad/s² |
| 缓冲 | α = 0.4，杯速上限 0.7 m/s，Match 段 5 cm，Cushion 段 10 cm |
| 成功判定 | 相对速度 < 0.05 m/s，持续 0.5 s；超时 = t_c + 2 s |

杯速上限 0.7 m/s 的原因：在这些姿态下，杯子沿来球方向每 1 m/s 需要 2–4 rad/s 的关节速度，
更高的杯速会超过 UR5e 的 π rad/s 关节速度限制。

## 参考结果

环境：Windows 11，Python 3.12.5，MuJoCo 3.15.0，NumPy 2.5.3。所有行都是 `--headless --trials 50 --seed 0`，
抛球和观测噪声用独立的随机数流，因此**各行用的是同一组 50 次抛球**。

| 配置 | 成功率 | 杯口跟踪误差 @t_c（均值） | 入杯相对速度（均值 / 最大） | 入杯到稳定（均值 / 最大） | 退化为 static |
|---|---|---|---|---|---|
| cushion | 50/50 | 0.22 mm | 4.43 / 5.11 m/s | 0.32 / 0.45 s | 0 |
| static | 50/50 | 0.03 mm | 5.13 / 5.81 m/s | 0.39 / 0.46 s | — |
| cushion，噪声 5 mm | 50/50 | 0.19 mm | 4.52 / 5.73 m/s | 0.32 / 0.46 s | 7 |
| static，噪声 5 mm | 50/50 | 0.03 mm | 5.13 / 5.82 m/s | 0.40 / 0.59 s | — |
| cushion，噪声 10 mm | 50/50 | 0.15 mm | 4.69 / 5.51 m/s | 0.34 / 0.52 s | 21 |
| static，噪声 10 mm | 50/50 | 0.04 mm | 5.14 / 5.83 m/s | 0.40 / 0.66 s | — |

- 无噪声时 t_c 时刻的预测误差为 0.00 mm：解析弹道与 MuJoCo 积分一致。噪声 5 / 10 mm 时均值约 3–4 / 5.5 mm，最大约 9 / 13 mm。
- 峰值执行器力矩约为上限的 15–25%，没有饱和。
- 单次规划（选点 + IK + 离散化）约 50 ms。

补充检查（临时脚本，未存档）：
- IK：在接球区域随机取 200 个目标位置和来球方向，200/200 求解成功且无碰撞，最大位置误差 10 µm，每次约 0.6 ms。
- Quintic 边界条件：随机的 (位置, 速度, 加速度) 首末状态都能精确满足。
- 离屏渲染确认：杯子朝来球方向倾斜，球落入杯中并留在杯内。

**结论与局限**：在当前设定下 static 和 cushion 都能 100% 接住。杯子缩小到手掌大小（内径 0.10 m，内深 0.06 m，球径 0.07 m）后结果仍是如此，
说明 v0 的接触参数阻尼很大，球一碰到杯底几乎不反弹，单靠静止等待就能接住。缓冲的作用体现在冲击上：入杯相对速度降低约 14%，稳定时间缩短约 18%。
由于关节速度限制，杯速只能到球速的约 15%，缓冲效果有限。有噪声时，后期重规划常常没有时间做完整的缓冲动作，
会更多地退化为 static。想让两种策略拉开差距，可以降低接触阻尼（如 `solref` 阻尼比调小，让球真的会弹），或者提高抛球速度。
