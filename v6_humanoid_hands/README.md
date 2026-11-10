# v6 — G1 上半身与全尺寸双 LEAP Hand

固定底座的 G1 上半身搭载官方左右 LEAP Hand 模型，完成三球级联抛接。每只手为四指、16 关节，保留原始尺寸、质量、惯量和关节范围；球的直径为 85 mm，质量为 120 g。

控制器采用椭圆内舀（scoop）、接球下沉缓冲和接触反馈持球。球由重力和接触碰撞驱动，不使用手球抓握约束。

## 运行

在仓库根目录执行：

```bash
uv sync
uv run python v6_humanoid_hands/juggle.py --pattern cascade3 --viewer --infinite
uv run python v6_humanoid_hands/juggle.py --pattern cascade3 --throws 30
uv run python v6_humanoid_hands/juggle.py --pattern cascade3 --throws 30 --video videos/v6_cascade3.mp4
```

`--viewer --infinite` 持续交互运行；`--throws 30` 验证 30 次完整飞行；`--video` 录制 [MP4 视频](../videos/v6_cascade3.mp4)，分辨率为 1280×720、60 fps。添加 `--speed 0.5` 可半速查看或录制。录制会覆盖同名视频。

一次完整飞行包括真实离手和确认接稳。为接回最后一球，释放总数可能超过 `--throws`。控制器包含前向物理试算，计算耗时可能明显长于仿真时长。

## 验证

```bash
uv run python -u v6_humanoid_hands/verify_g1_hands.py --log v6_humanoid_hands/results/assembly.txt
uv run python -u -m v6_humanoid_hands.test_cradle_contact -v --log v6_humanoid_hands/results/validation.txt
uv run python -m unittest v6_humanoid_hands.test_evaluation -v
```

三个命令分别检查模型装配与 FK/IK、接触与连续抛接、评估工具与日志功能。`--log` 保存并覆盖相应日志；省略时仅运行检查。验证失败返回非零退出码。

| 检查项 | 默认配置参考结果 |
| --- | --- |
| 装配、原始物理参数与 FK/IK | 通过 |
| 接触与抛接回归 | 6/6 通过 |
| 评估工具与日志测试 | 13/13 通过 |
| 三球 30/50 次完整飞行 | 全部接回，零掉球、零球球碰撞 |

连续抛接测试逐物理步检查球高、碰撞和接触动力学，并验证接球支撑、缓冲减速及手臂姿态。具体断言见 [test_cradle_contact.py](test_cradle_contact.py)。

## 局部噪声实验

`--obs-noise` 为部分接球位置预测添加高斯扰动，单位为米；`--seed` 指定随机种子，默认值为 0。例如，标准差为 5 mm 的试验：

```bash
uv run python v6_humanoid_hands/juggle.py --pattern cascade3 --throws 30 --obs-noise 0.005 --seed 1
```

该扰动只作用于部分接球位置预测，球速和前向试算初态仍使用仿真真值。实验不包含完整的传感器采样、延迟、丢帧或执行器误差模型。

运行五档噪声、每档两个种子的十轮实验：

```bash
uv run python -m v6_humanoid_hands.evaluate_noise --noise-mm 0 1 3 5 10 --seeds 0 1 --throws 30 --output v6_humanoid_hands/results/noise_sensitivity.jsonl
```

输出 JSONL 必须不存在；重新生成同名文件前需备份或删除旧文件。默认串行运行，`--jobs 2` 启用两个进程。`--timeout` 限制每轮仿真时长，不限制实际计算时间。

实验完成后，从同一 JSONL 生成摘要和 CSV：

```bash
uv run python -m v6_humanoid_hands.summarize_noise --input v6_humanoid_hands/results/noise_sensitivity.jsonl --output v6_humanoid_hands/results/noise_sensitivity.txt --csv-output v6_humanoid_hands/results/noise_trials.csv --execution-output v6_humanoid_hands/results/execution_demand.csv --overwrite
```

汇总读取已有 JSONL，不运行仿真；也可直接用于仓库中的归档数据。`--overwrite` 覆盖指定的 TXT/CSV，两个 CSV 输出参数均可省略。

### 实测结果

以下实验每档使用两个种子，每轮请求完成 30 次飞行，运行参数与上述命令一致，使用默认单进程配置（`--jobs 1`）。

| 位置扰动标准差 | 种子 0 完成数 | 种子 1 完成数 | 成功轮数 | 停止原因 |
| --- | --- | --- | --- | --- |
| 0 mm | 30 | 30 | 2/2 | 均完成 |
| 1 mm | 4 | 4 | 0/2 | 两轮均接球规划拒绝 |
| 3 mm | 4 | 21 | 0/2 | 两轮均接球规划拒绝 |
| 5 mm | 7 | 13 | 0/2 | 两轮均接球规划拒绝 |
| 10 mm | 1 | 25 | 0/2 | 种子 0 规划拒绝，种子 1 掉球 |

七轮带噪试验因 `NO_FEASIBLE_RECEPTION` 停止。10 mm、种子 1 在仿真 9.071 s 时球高降至约 0.398 m，触发低于 0.40 m 的掉球判据。全部十轮均无球球碰撞。

每档仅两个种子，结果不足以估计总体成功概率，完成次数也不随噪声幅度单调变化。零噪声下两个种子得到相同轨迹。

## 结果文件

上述命令在 `v6_humanoid_hands/results/` 下生成以下文件：

| 文件 | 内容 |
| --- | --- |
| `assembly.txt` | 模型装配、原始物理参数、FK/IK 与初始接触检查 |
| `validation.txt` | 无噪声接触与 30/50 次抛接回归日志 |
| `noise_sensitivity.jsonl` | 十轮噪声实验的完整记录，支持校验和重新生成摘要及 CSV |
| `noise_sensitivity.txt` | 噪声实验配置、逐轮摘要、停止原因及源码与初值指纹 |
| `noise_trials.csv` | 十轮试验的配置、完成数、成功标记、掉球/碰撞计数与停止事件 |
| `execution_demand.csv` | 无噪声 30 次基线的 49 个关节执行统计 |

噪声摘要、逐轮 CSV 和基线执行统计均从 `noise_sensitivity.jsonl` 生成。汇总工具校验轮次、配置和统计一致性；物理复现需重新运行实验，并核对环境版本、实验条件和初值指纹。

按上述命令生成的 `execution_demand.csv` 对应 **0 mm、种子 0、30 次完整飞行**，包含峰值关节速度、峰值/RMS 执行器力矩及接近配置力限的比例。力矩按执行器力与关节传动比换算，不包含被动力和约束力；接近力限定义为距任一配置边界不超过力范围的 1%。该文件不包含 50 次运行的逐关节数据。

`results/` 保存正式验证日志、实验 JSONL 及其派生 TXT/CSV，不收录标定候选和调试数据。

## 模型与控制

| 参数 | 默认值 |
| --- | --- |
| 机器人 | 固定底座 G1 上身，每臂 7 自由度，双 LEAP Hand |
| 球 | 直径 85 mm，质量 120 g |
| 物理步长与求解器 | 1 ms，`implicitfast`，椭圆摩擦锥，`impratio=100` |
| 交替出手间隔 `tau` | 0.32 s |
| 分离至掌心接触的目标时间 `flight` | 0.53 s |
| LEAP 手指力矩范围 | ±0.5 N·m |
| 手部重力补偿 | `gravcomp=1` |

手臂从外侧接球点回收，经内舀加速后向内侧抛出；接球时向下缓冲，手指在触掌后根据相对位置、几何间隙和接触力调整闭合。确认接稳需要持续的掌指支撑和相对速度衰减。运行及试算均检查碰撞、关节范围和姿态限制。

实现位于 [src/jugglebot/humanoid](../src/jugglebot/humanoid/)：`model.py` 负责装配，`kinematics.py` 负责运动学，`cascade.py` 负责抛接规划与控制，`throwing.py` 负责手指控制。模型资源见 [Unitree G1](../assets/unitree_g1/) 和 [LEAP Hand 来源信息](../assets/leap_hand/UPSTREAM.json)。

### 重新标定（可选）

正常运行使用 `src/jugglebot/humanoid/leap_cascade_seed_plans.json`，无需重新标定。修改模型或控制参数后，可将标定结果导出至指定路径：

```bash
uv run python v6_humanoid_hands/calibrate.py --balls 3 --throws 50 --timeout 35 --output <输出文件路径>
```

将 `<输出文件路径>` 替换为实际 JSON 文件路径。脚本在指定条件下的物理验证通过后导出优化器初值；导出文件不会自动用于正常运行。采用新初值时，需审查内容、替换源码中的 `leap_cascade_seed_plans.json`，再运行完整回归。

`--initial-plans <文件>` 可指定标定的起始初值；`forecast` 与 `execution` 分别保存预测动作和当前动作的优化初值。

抛球初值含四项运动参数和可选的 `finger_release_lead_s`。后者调整手指张开时机，正值表示提前，范围为 ±25 ms，默认 0。更改模型、节拍或初值后应重新运行完整回归。

## 实验范围

正式验收条件为无噪声三球级联，初态为两球在手、一球在空中。初始化空中球的接球不计入完成飞行数。入口支持 `cascade3` 和三个球，底座固定，不包含移动或平衡控制。

与 v5 相比，v6 使用 G1 上身、官方左右手及仿真位置和速度真值；v5 使用镜像 UR5e 双臂、镜像左手及位置采样估计。球规格、手部执行器和接触参数已对齐，节拍、工作空间、控制器与验收条件分别标定。v5 按每手次数计数，v6 按全局完整飞行计数。

`--tau` 和 `--flight` 可覆盖默认节拍，但已保存的验收结果仅适用于默认参数。手指增益、力矩限制、接触柔顺参数、重力补偿及安装支架尚未经过硬件辨识；关节速度、力矩—转速曲线和热限制尚未对齐。仿真抛接通过不构成真实机器人部署验证。
