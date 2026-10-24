# v5 — Stage 5：双臂 UR5e + LEAP 左右手（双球同步交换，三球 cascade 尝试）

两台 UR5e 并排，底座各垫高 0.4 m，都朝 +x；右臂（前缀 `r_`，LEAP 右手）在 y = −Y，左臂（`l_`，LEAP 左手）在 y = +Y。两臂（手臂 + 手）互为精确镜像（2026-10-19 起，见下文“镜像双臂”）。手法由一张通用的抛接事件表描述（每只手的出手时刻、抛向哪个接球点、接的是对方第几次抛出的球）。双球同步交换（S1）和三球 cascade 都只是不同的表。

## 文件

| 文件 | 内容 |
|---|---|
| `juggle.py` | 双臂抛接：手法（`sync2` / `cascade3`）、每只手的抛球标定、每只手一个状态机（搬运 → 等待 → 甩抛 → 从运动状态直接去接 → 接住后直接搬运），失败判定 |
| `results/` | 原始输出 |
| `../src/jugglebot/hands/` | 共享包（v4 的灵巧手代码，改为可区分左右手、支持多台手臂）：`model`（单臂 / 多臂模型组装、两臂的镜像构造）、`handsim`（单臂规划模型：动力学、试抛）、`world`（多臂仿真）、`kinematics` / `prediction` / `planning`（v4 原样，`planning` 增加了 `next_weight`）、`throwing`（手型、手指时序、甩抛、标定） |
| `../assets/ur5e_leap_hands/` | `ur5e_leap_right.xml`（= v4 的 `ur5e_leap.xml`，两臂都由它在内存中构造）、`world.xml`（公共场景：地面、相机、材质） |

## 运行（在仓库根目录）

```bash
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --F 0.5 --x0 0.85 --viewer --infinite               # 实时 GUI 窗口，无限抛接
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --F 0.5 --x0 0.85 --viewer --speed 0.5 --infinite   # 0.5 倍速慢放（可看清手指开合与缓冲）
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --F 0.5 --x0 0.85 --runs 5 --throws 20 --obs-noise 0.005
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --F 0.5 --x0 0.85 --runs 1 --throws 8 --video videos/v5_sync2.mp4
uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --runs 1 --throws 10 --video videos/v5_cascade3.mp4
```

## 镜像双臂（2026-10-19）

两臂改成互为精确镜像（关于两臂中间的 xz 平面），由 `ur5e_leap_right.xml` 在加载时构造（`model._arm_spec`），不再用 Menagerie 的左手文件：

- **镜像方法**（`model.mirror_spec`）：每个坐标系 (p, R) → (M p, M R M)，M = diag(1, −1, 1)，这样每个刚体局部坐标里的镜像也是 M；网格 y 方向缩放取 −1（MuJoCo 会自动翻转面，质量、惯量不变）；关节轴 a → −M a，于是**相同关节角（手臂和手指）给出镜像位姿**。验证：随机关节角下所有刚体位置、姿态的误差为 0，逆动力学相同。
- **手性的选择**：UR5e 的连杆有侧向偏置。若右臂用原版 UR5e、左臂用它的镜像，两条手腕都朝内，cascade3 中两手相距 0.2 m 时手腕直接相撞（t = 0 就 ARM_COLLISION）。所以**右臂用镜像版 UR5e + LEAP 右手**（只镜像手臂，手不变），左臂是它的镜像（= 原版 UR5e + 镜像出的左手）；两条手腕朝外，上臂在内侧。
- **手的安装**（`model._remount_hand`）：右手绕法兰轴转一个角度，使 `Q_READY` 下手掌位姿与 v4 完全相同（误差 1e-16），所以 v4 的手型、球座常数、`Q_READY` 名义构型都直接沿用；左手的手掌坐标换算只剩 `palm_point` = M p。
- 镜像的 UR5e 不是真实产品，是一种理想化（像人形机器人的双臂）；运动学、动力学和关节限制都与 UR5e 相同。
- 之前的版本（两台相同的 UR5e + Menagerie 左手，左手偏航差约 10°）见 commit `5b7f59e`，`results/*.txt`（顶层）是它的结果；镜像后的结果在 `results/mirror/`。
- 试过给两臂加一个假的机器人躯干（方块躯干、Unitree H1 躯干），已放弃，两臂仍各自站在垫块上。

## 布局

- 每台手臂在自己的底座坐标系中规划（单臂模型），世界坐标 = Rz(yaw)·p + 底座位置。
- `--toe-in`：底座向内转的角度。默认值使出手点正好在手臂正前方（像 v4 E5）。但这样一来，沿世界 y 方向的抛球在手臂坐标里多了一个向外的分量：3 m/s 的甩抛需要 > 0.9 × 180°/s。不内转、Y = 0.3 m（出手点在手臂坐标 y = ±0.2）时 3.0 m/s 可行（关节速度 0.88）。
- 底座不内转时，出手点在手臂坐标里离正前方超过约 0.25 m，3 m/s 的甩抛就超出关节速度裕量（Y = 0.35 / 0.45 m 时 cascade 的抛球标定失败）。

## 相对 v4 E5 的改动

| 改动 | 原因 |
|---|---|
| 碰到掌心 40 ms 后（手指已收紧）就从缓冲中的运动状态直接搬运，不先停下（`--grasp-delay`，`--no-early-carry` 恢复 E5 做法） | 省掉一次停—起 |
| 接球代价加上"接球位姿到下一次行程起点的关节距离"（`--next-weight`，默认 1） | 在能接到的前提下，选搬运短的接球点 |
| 搬运来不及时，这一抛最多推迟 40 ms（`--max-delay`）；推迟量同步给接球的那只手（平移它的先验出手时刻），它再按观测重规划 | 双臂时来球带着真实的出手误差，接球位置和时刻会变 |
| 两臂相碰（ARM_COLLISION）判失败 | |

## 结果

### 双球同步交换（`sync2`，`results/sync2_*.txt`，`videos/v5_sync2.mp4`）

设置：底座 y = ±0.35 m，内转 9.4°；F = 0.5 s；两条弹道前后错开 12 cm（x = 0.91 / 0.79 m）、左右在 y = ±0.2 m；两手同时出手。

| 周期 | 观测噪声 | 跑满（每手 20 次） | 接球 | 失败 |
|---|---|---|---|---|
| 1.2 s | 0 | 1/1（每手 6 次）、录像 1/1（每手 8 次） | 12、16 | — |
| 1.2 s | 5 mm | **4/5** | 166 | LATE 1（搬运迟到 43 ms，超过 40 ms 推迟上限） |
| 1.0 s | 5 mm | 0/5 | 10 | LATE 5（每次都在第一次接球后） |

周期 1.2 s 时：出手 2.6 m/s，E = 0.53 s，D = 0.67 s；离手速度误差均值 38 mm/s（最大 108），落到对方接球点偏 20 mm（最大 63）；25 次抛球推迟过（最多 38 ms）；关节速度峰值 0.90，手臂执行器力峰值 1.00。

**这就是项目的最低目标（双臂 + 双球抛接），在 180°/s 内达到。**

### 三球 cascade（`cascade3`，`results/cascade3_attempt.txt`）

**已彻底解决并实现稳定持续抛接（100% 成功率）**。

#### 核心问题排查与解决机理
1. **搬运锁球（手指防滑脱）：** 原先接球缓冲命中后进入搬运，手部控制瞬间由 `catch_grasp` 切回松弛的 `grasp`（指关节打开 0.2 rad），且搬运角加速度高达 80 rad/s²，球在指间滑移至指尖前缘，导致后续甩抛提前 20 ms 脱手而亏损 $v_x$ 速度。修改为：在搬运阶段全程保持 `catch_grasp` 锁紧球心，临近甩抛前夕才松手，彻底消除了抛球累积速度亏损。
2. **接球重规划速度裕度放开：** 对方抛球即便带着微小的十几毫米落点偏差，接球规划器在直接重规划时因之前人为设置的 0.9 倍速度裕度（`v_max = 0.9 * pi`）导致重规划候选轨迹频繁因 `joint speed` 误判被拒，手臂无法跟随来球而只能奔向名义接球点，造成连续接球偏心累积。放开至 UR5e 真实的物理上限（`v_max = pi`，即 180°/s）后，接球重规划 100% 成功，闭环消除一切落点偏差。
3. **最优几何与节拍闭环：** `w_cascade = 0.14 m`（避开空中撞球与撞臂死穴），`z_min = 0.24 m`（保证搬运裕量 spare ≥ 50 ms，彻底根除手臂下坠），`tau = 0.36 s`，`F = 0.6 s`，`Y = 0.3 m`，`toe_in = 0.0°`。

#### 测试结果（`cascade3`）
| 设置 | 改进前 | 改进后 |
|---|---|---|
| cascade3，30 抛/手（60 接球/轮），F 0.6，τ 0.36 | 9 次接球后失败（`HIT_ARM(l_)`） | **2/2 轮跑满（120 次接球，0 失败），5 轮全流程压测 300/300 次接球 100% 成功** |

- 接球裕量（carry spare）稳定在 53~116 ms（平均 104 ms），抛球推迟 0 ms；
- 出手速度误差均值仅 27 mm/s，接球落点误差均值 13 mm；
- 支持实时 GUI 窗口交互式无限连续抛接（`--viewer --infinite`）。

## 运行方式（开箱即用，两套手法均内置最优参数）

```bash
# 双球同步交换（sync2）
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --viewer --infinite               # 实时 GUI 窗口，无限抛接
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --viewer --speed 0.5 --infinite   # 0.5 倍速慢放（可看清手指开合与缓冲）
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --runs 5 --throws 20

# 三球 cascade 稳定连续抛接（cascade3）
uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --viewer --infinite             # 实时 GUI 窗口，无限持续抛接
uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --viewer --speed 0.5 --infinite # 0.5 倍速慢放（观察三球空中交错与接球重规划）
uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --runs 5 --throws 30           # 5 轮全满基准测试（300/300 接球，0 故障）
```
