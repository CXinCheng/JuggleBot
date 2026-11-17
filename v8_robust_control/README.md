# v8 — 感知扰动与控制评估

为 G1 上身与双 LEAP Hand 三球抛接仿真提供球观测、状态估计、感知扰动、残差训练和完整任务评估。物理模型复用 [v6](../v6_humanoid_hands/README.md)，完整任务的三球开局与收尾复用 [v7](../v7_three_ball_hold/README.md)。抛接通过手与球的物理接触实现。

## 运行

在仓库根目录执行：

```bash
uv sync
uv run python -m v8_robust_control.juggle --throws 1 --observation-mode truth
uv run python -m v8_robust_control.juggle --throws 1 --observation-mode sampled
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--throws` | 1 | 目标完整飞行数 |
| `--timeout` | 自动 | 仿真时间上限（秒） |
| `--observation-mode` | `truth` | `truth` 读取位置与速度真值；`sampled` 从位置采样估计状态 |
| `--observation-period` | 1/60 s | 采样模式的位置测量周期 |
| `--observation-max-age` | 0.05 s | 从采样时刻起的测量有效期，须不小于采样周期 |
| `--noise-mm` | 0 | 各坐标轴的位置高斯噪声标准差（mm） |
| `--delay-ms` | 0 | 固定测量传输延迟（ms） |
| `--dropout` | 0 | 整帧丢失概率，范围为 0–1 |
| `--seed` | 0 | 感知扰动随机种子 |
| `--output` | 无 | 实验 JSON 的保存路径，须为新文件 |
| `--video` | 无 | 保存正常速度 MP4，路径须为新文件 |

`juggle` 从两球在手、一球在空中开始，不包含三球在手的开局和收尾。一次完整飞行从球离手开始，到确认接稳结束。初始空中球的接球不计入，释放总数可能超过目标飞行数。任务失败返回非零退出码。

### 完整三球任务

`complete_task` 从右手两球、左手一球开始，完成指定次数的抛接，再以右手两球、左手一球结束。`--throws` 须为不小于 4 的偶数，默认 30；`--timeout` 默认 40 s，`--output` 必填且须为新文件。

生成参考结果的命令见[完整任务参考结果](#完整任务参考结果)。

支持上述 `--observation-mode`、感知扰动、种子和视频参数；采样周期固定为 1/60 s，测量有效期为 0.05 s。扰动仅用于 `sampled` 模式。`--seed-plans` 可指定完整任务的规划初值，默认使用 [complete_seed_plans.json](complete_seed_plans.json)。

## 观测接口

[observation.py](../src/jugglebot/robustness/observation.py) 定义球观测和状态估计的数据结构。位置、速度和加速度使用世界坐标系，单位分别为 m、m/s 和 m/s²；时间戳使用仿真时间，单位为秒。

| 字段 | 含义 |
| --- | --- |
| `ball_id` | 球的模型编号 |
| `sample_time_s` | 最新测量的采样时刻 |
| `received_time_s` | 测量可供控制器使用的时刻 |
| `state_time_s` | 状态估计对应的控制时刻 |
| `position_m` | 测量位置或控制时刻的估计位置 |
| `velocity_m_s` | 测量速度或估计速度；不可用时为 `None` |
| `acceleration_m_s2` | 估计加速度；不可用时为 `None` |
| `valid` | 测量或状态估计是否有效 |

经典控制入口（`juggle`、`complete_task`）的采样模式根据连续三次球位置测量估计速度和加速度，并预测控制时刻的球状态。位置测量依次经过噪声、丢帧和延迟队列后交给估计器。接球规划使用有效且未过期的估计结果。

噪声按球和坐标轴独立采样；丢帧作用于一次采样中的全部球。固定延迟通过 FIFO 队列实现，接收时间为测量实际交付的物理步时刻。固定种子可重放噪声和丢帧序列。状态估计采用未滤波的三点位置差分。

## 控制与适用范围

[ObservedCascade](../src/jugglebot/robustness/controller.py) 根据球状态预测接球位置，并规划抛接动作：

| 环节 | 处理方式 |
| --- | --- |
| 抛球 | 复用已执行的运动命令，在限定范围内调整手指释放时机 |
| 接球 | 规划失败时重新预测，并调整接球轨迹与回腕姿态 |
| 空手回收 | 接球延迟时，提前启动下一次接球的手掌回收 |

动作执行前检查接球稳定性、下一次抛球的准备状态、碰撞和关节姿态。两种观测模式使用相同的控制逻辑。

上述经典控制器将采样估计用于实时接球及完整任务的收尾预测；持球控制、接触判定、安全检查和规划试算使用仿真真值。残差环境的位置估计和策略输入见[残差控制](#残差控制)。

[CompleteCascade](complete_task.py) 复用三球开局与收尾控制，并接入观测和完整任务验收。

## 验证

```bash
uv run python -m unittest discover -s v8_robust_control -t . -v
uv run python -m unittest v6_humanoid_hands.test_evaluation -v
```

测试覆盖时间戳、状态估计、无效数据处理、规划试算隔离、动作规划与实验记录导出。

运行零扰动基线：

<details>
<summary>生成零扰动基线</summary>

```bash
uv run python -m v8_robust_control.verify_baselines --observation-mode truth --output-dir v8_robust_control/results
uv run python -m v8_robust_control.verify_baselines --observation-mode sampled --output-dir v8_robust_control/results
```

</details>

每种模式连续运行至 50 次完整飞行，并保存 30 次和 50 次检查点。验收失败时保存失败结果并返回非零退出码。重新运行前需更换输出目录，或移走已有的同名文件。

### 零扰动参考结果

采用默认物理参数，无观测噪声、传输延迟或丢帧；采样周期为 1/60 s，测量有效期为 0.05 s。

| 模式及记录 | 完成飞行 | 释放总数 | 掉球 | 球球碰撞 | 仿真时长 |
| --- | --- | --- | --- | --- | --- |
| [truth，30 次](results/truth_baseline30.json) | 30/30 | 32 | 0 | 0 | 10.210 s |
| [sampled，30 次](results/sampled_baseline30.json) | 30/30 | 32 | 0 | 0 | 10.236 s |
| [truth，50 次](results/truth_baseline50.json) | 50/50 | 52 | 0 | 0 | 16.620 s |
| [sampled，50 次](results/sampled_baseline50.json) | 50/50 | 52 | 0 | 0 | 16.671 s |

各结果文件记录验收指标、运行配置，以及源码和规划初值的 SHA-256 校验值。

### 完整任务参考结果

<details>
<summary>生成完整任务参考结果</summary>

```bash
uv run python -m v8_robust_control.complete_task --throws 30 --observation-mode truth --output v8_robust_control/results/complete_truth30.json
uv run python -m v8_robust_control.complete_task --throws 30 --observation-mode sampled --output v8_robust_control/results/complete_sampled30.json
uv run python -m v8_robust_control.complete_task --throws 50 --observation-mode truth --output v8_robust_control/results/complete_truth50.json
uv run python -m v8_robust_control.complete_task --throws 50 --observation-mode sampled --output v8_robust_control/results/complete_sampled50.json
```

</details>

各零扰动任务独立运行，结束时三球均连续接稳 0.8 s。

| 模式及记录 | 完成飞行 | 掉球 | 禁止的球球接触 | 仿真时长 |
| --- | --- | --- | --- | --- |
| [truth，30 次](results/complete_truth30.json) | 30/30 | 0 | 0 | 11.147 s |
| [sampled，30 次](results/complete_sampled30.json) | 30/30 | 0 | 0 | 11.145 s |
| [truth，50 次](results/complete_truth50.json) | 50/50 | 0 | 0 | 17.562 s |
| [sampled，50 次](results/complete_sampled50.json) | 50/50 | 0 | 0 | 17.562 s |

### 完整任务验收

| 检查 | 条件 |
| --- | --- |
| 开局 | 三球接触对应手部并连续稳定至少 0.05 s，才启动首次抛球 |
| 完成次数 | 恰好释放指定次数，每个释放编号均已确认接回，结束时无空中球 |
| 最终持球 | 三球连续 0.8 s 接触手部，相对手掌速度均小于 0.15 m/s，球心距对应掌心小于 0.16 m，球高大于 0.40 m |
| 收尾接近 | 在提前到达时间之外保留至少 25 ms 接近余量，当前要求预测剩余飞行时间大于 125 ms |
| 球球接触 | 仅首次确认释放前允许开局右手的球 0/2 接触，且保留球仍接触右手、两球均在承托范围内；该接触单独记录。此后的球球接触判为失败 |
| 姿态 | 默认支架高度下，球高低于 1.30 m、掌心高度不超过 1.15 m、手肘至少低于肩部 0.08 m、手肘外展小于 0.10 m、腕部俯仰绝对值小于 38°、关节越限量不超过 0.004 rad；收尾不放宽限制 |
| 其他接触 | 臂部与躯干等非手指内部接触的穿透深度不超过 4 mm；球触地面、支架、手臂，以及两臂相撞均判为失败 |
| 规划失败 | 同一次抛球最多进行 8 次规划尝试，不可行时以 `NO_FEASIBLE_THROW` 停止；接球规划不可行时以 `NO_FEASIBLE_RECEPTION` 停止 |

JSON 保存配置、逐球承托证据、姿态指标及源码和标定文件的 SHA-256 校验值。开局允许的球对接触单独保存为 `supported_pair_contact_steps` 和 `supported_pair_force_max_n`。

## 感知扰动评估

单次扰动实验：

```bash
uv run python -m v8_robust_control.juggle --observation-mode sampled --throws 30 --noise-mm 3 --delay-ms 20 --dropout 0.1 --seed 0
```

默认在终端显示结果；添加 `--output <新文件路径>` 可保存单次实验 JSON。

评估位置噪声、延迟、丢帧及组合条件：

<details>
<summary>生成级联扰动评估</summary>

```bash
uv run python -m v8_robust_control.evaluate_perception --seeds 0 1 --throws 30 --jobs 3 --output v8_robust_control/results/perception_trials.jsonl --video-cases baseline noise_3mm combined
uv run python -m v8_robust_control.summarize_perception --input v8_robust_control/results/perception_trials.jsonl --output v8_robust_control/results/perception_sensitivity.md --csv-output v8_robust_control/results/perception_trials.csv --overwrite
```

</details>

| 扰动 | 测试范围 |
| --- | --- |
| 位置噪声标准差 | 0、1、3、5、10 mm |
| 固定传输延迟 | 0、10、20、40 ms |
| 整帧丢失概率 | 0、5%、15%、30% |
| 组合条件 | 3 mm 噪声、20 ms 延迟、10% 丢帧 |

单项测试每次只改变一种扰动。各条件使用相同的控制器、初始物理状态、采样周期和测量有效期，并运行 `--seeds` 指定的全部种子。扰动范围用于仿真敏感性分析，未经真实传感器标定。

JSONL 保存配置、逐次结果、感知序列校验值和估计误差。汇总程序检查试验完整性、计数和配置一致性后生成报告与 CSV。中断实验保留已完成的记录，汇总程序要求输入包含全部试验。

成功要求完成全部目标飞行，并通过接稳、碰撞和姿态检查。未找到可行接球方案时，以 `NO_FEASIBLE_RECEPTION` 终止实验。规划失败、实际掉球和实际碰撞分别统计。

评估结果：[退化报告](results/perception_sensitivity.md)、[逐次指标](results/perception_trials.csv)、[原始记录](results/perception_trials.jsonl)。

完整任务使用 `--task complete`，结果以 `complete_` 前缀区分：

<details>
<summary>生成完整任务扰动评估</summary>

```bash
uv run python -m v8_robust_control.evaluate_perception --task complete --seeds 0 1 --throws 30 --jobs 3 --output v8_robust_control/results/complete_perception_trials.jsonl --video-cases baseline noise_3mm combined
uv run python -m v8_robust_control.summarize_perception --input v8_robust_control/results/complete_perception_trials.jsonl --output v8_robust_control/results/complete_perception_sensitivity.md --csv-output v8_robust_control/results/complete_perception_trials.csv --overwrite
```

</details>

完整任务结果：[退化报告](results/complete_perception_sensitivity.md)、[逐次指标](results/complete_perception_trials.csv)、[原始记录](results/complete_perception_trials.jsonl)。`--jobs` 默认为 1。每组两个种子的结果不作为总体成功概率的精确估计。

## 视频

演示：[感知扰动对照视频](../videos/v8_perception_before_rl.mp4)。画面标注配置、种子、完成次数和停止原因；对应试验信息见 [video_comparison.json](results/video_comparison.json)。

`--video-cases` 为选定条件的第一个种子录制原始视频，文件名为 `videos/v8_before_rl_<配置>_seed<种子>.mp4`。单次实验可通过 `--video` 指定视频路径。

原始视频以 60 fps 记录实际执行过程，不包含规划试算。接球规划失败时，试验立即终止，因此视频可能短于一秒，球尚未落地。

| 原始视频（种子 0） | 仿真停止时刻 | 完成飞行 | 结果 |
| --- | --- | --- | --- |
| [零扰动](../videos/v8_before_rl_baseline_seed0.mp4) | 10.236 s | 30/30 | 完成 |
| [3 mm 位置噪声](../videos/v8_before_rl_noise_3mm_seed0.mp4) | 0.279 s | 0/30 | 接球规划失败 |
| [组合扰动](../videos/v8_before_rl_combined_seed0.mp4) | 0.287 s | 0/30 | 接球规划失败 |

视频的帧数、播放速率和试验结果保存在原始评估 JSONL 中。

生成带配置和结果标注的并排视频：

<details>
<summary>生成级联对照视频与元数据</summary>

```bash
uv run python -m v8_robust_control.compare_videos --input v8_robust_control/results/perception_trials.jsonl --cases baseline noise_3mm combined --seed 0 --output videos/v8_perception_before_rl.mp4 --metadata-output v8_robust_control/results/video_comparison.json
```

</details>

并排视频按正常速度播放，时长与最长片段一致。提前终止的片段定格并显示 `STOPPED - final frame held`，仿真停止时刻保持不变。`--hold-seconds` 可额外设置结尾停留时间，默认为 `0` 秒。

`--cases` 指定单个配置时可生成单独的展示片段。`--output` 指定视频路径，`--metadata-output` 指定结果目录中的 JSON 路径；两者均须为新文件。

完整任务的原始视频命名为 `videos/v8_complete_before_rl_<配置>_seed<种子>.mp4`。生成对应的并排视频：

<details>
<summary>生成完整任务对照视频与元数据</summary>

```bash
uv run python -m v8_robust_control.compare_videos --input v8_robust_control/results/complete_perception_trials.jsonl --cases baseline noise_3mm combined --seed 0 --output videos/v8_complete_perception_before_rl.mp4 --metadata-output v8_robust_control/results/complete_video_comparison.json
```

</details>

完整任务演示：[零扰动](../videos/v8_complete_before_rl_baseline_seed0.mp4)、[感知扰动对照](../videos/v8_complete_perception_before_rl.mp4)、[视频信息](results/complete_video_comparison.json)。

### PPO 残差策略演示

以下视频加载训练记录选定的 PPO 检查点，执行无扰动任务。控制方式为名义控制器加 PPO 残差，正常速度播放；成功演示不代表扰动条件下的性能提升。

| 任务 | 视频 | 元数据 |
| --- | --- | --- |
| 完整单球抛接 | [单球](../videos/v8_ppo_single_seed0.mp4) | [single_seed0.json](results/policy_videos/single_seed0.json) |
| 双球交换 | [双球](../videos/v8_ppo_two_seed0.mp4) | [two_seed0.json](results/policy_videos/two_seed0.json) |
| 三球 30 抛及收尾 | [三球](../videos/v8_ppo_complete30_seed0.mp4) | [complete30_seed0.json](results/policy_videos/complete30_seed0.json) |

元数据保存训练与试验种子、选定预算、模型和视频校验值、任务配置及物理验收结果。录制完整回合，包括初始持球和末端接稳；失败回合保留终止结果，命令返回非零退出码。

录制命令见[录制残差策略视频](#录制残差策略视频)。

## 规划初值

完整任务默认加载 [complete_seed_plans.json](complete_seed_plans.json)。初值包含手部运动参数，供规划器试算与修正，不用于回放或覆盖球状态。复现现有基线无需重新标定。

<details>
<summary>可选：修改模型后重新标定</summary>

修改几何、接触参数或节拍后，可重新标定并验证候选初值：

```bash
uv run python -m v8_robust_control.calibrate_complete --throws 50 --output v8_robust_control/results/complete_seed_candidate.json --report v8_robust_control/results/complete_calibration.jsonl
uv run python -m v8_robust_control.complete_task --throws 50 --observation-mode sampled --seed-plans v8_robust_control/results/complete_seed_candidate.json --output v8_robust_control/results/complete_candidate_sampled50.json
```

标定程序仅在完整任务通过后导出初值，JSONL 保留各轮结果；`--input <初值文件>` 可指定起点。

</details>

## 残差控制

残差环境在名义控制器上施加有界电机修正，分别覆盖来球接稳、完整单球抛接及多球任务。

### 来球接稳

使用相同的 G1、双 LEAP Hand 和球模型，在经典反馈控制器上评估有界残差。任务从下降球开始，以左手连续接稳结束。

| 项目 | 默认设置 |
| --- | --- |
| 来球 | 落点各轴 ±50 mm；水平速度各轴 ±0.12 m/s；初始高度为承托点上方 0.20 ±0.015 m，向下速度 0.30 ±0.10 m/s |
| 控制与测量 | 策略 50 Hz，伺服与物理步进 1 kHz，位置测量 60 Hz；测量有效期 50 ms |
| 状态估计 | 自由飞行用最多八帧位置拟合速度；接触后使用位置测量、接触状态和掌心速度进行 Kalman 估计；不读取球真值速度 |
| 名义控制器 | 根据有效估计预测落点，生成限速、限加速度的五次轨迹；掌心接触后下沉缓冲并闭合手指 |
| 残差 | 手球受力接触后可放松手指，掌心承托后可额外闭合，协同幅度为 ±0.15 rad；预定闭指时段结束、掌心仍受力且球估计有效、不确定度不超过 15 mm 时，开放掌心 XYZ 每轴 ±15 mm；位移速度不超过 0.15 m/s，手指也限制变化率 |
| 成功 | 连续 0.2 s 存在受力手球接触；相对速度小于 0.15 m/s，掌心切向距离小于 40 mm，法向距离为 10–60 mm |
| 截止时间 | 0.8 s，未接稳判为任务失败 |

球状态仅在重置时初始化；运行中只写执行器目标，不使用抓握约束、球体外力或球状态回写。独立验收器在每个物理步检查接稳、碰撞、球高和姿态。

<details>
<summary>观测与控制接口</summary>

[ResidualCatchEnv](residual_env.py) 提供 Gymnasium 接口。环境版本为 6，73 维观测包括估计的球相对位置与速度、有效性和年龄、关节状态、掌心速度、接触、时间、已应用动作、名义参考位置误差与速度、估计不确定度，以及额外闭指和掌心位移的权限。测量过期时保留已有参考，不回退到球真值。

扰动仅作用于球位置测量链路；关节、接触信号和物理参数尚未随机化。

接触阶段继续处理实际收到的位置测量。接触转换改变运动模型和速度先验；延迟测量在采样时刻更新，再预测到当前控制时刻。接触信号本身不会生成位置测量，过期规则在两个阶段均适用。估计误差真值仅用于评估记录。

自由飞行的不确定度由位置拟合残差、采样间隔和预测年龄计算。接触滤波沿用飞行阶段估计的测量方差，并保留 3 mm 的传感器噪声下限；不读取实际扰动等级或球速度真值。

飞行拦截使用名义控制器，残差策略用于接触后的缓冲与接稳。手球受力接触期间暂停弹道重规划。掌心接触切换时，名义参考扣除已施加的残差，保证重建后的掌心目标连续，不重复叠加位移。所有来球回合均完整模拟并计入评估。

物理步缓存复用用于减少重复计算；伺服、积分和逐步安全检查频率保持不变。[吞吐量对照](results/reception_refinement/benchmark.json) 保存相同场景下的配置、结果与耗时。

</details>

<details>
<summary>奖励定义</summary>

默认配置下，来球接稳、完整单球和多球环境使用相同的终止奖励与动作成本；多球环境额外记录接回进度。

| 项目 | 单步奖励 |
| --- | --- |
| 任务成功 | 一次性 +10 |
| 普通失败 | 一次性 −10，包括掉球、未释放、错过窗口及超时 |
| 安全失败 | 一次性 −20，包括禁止接触、姿态越限、球高达到上限、外力及非有限状态 |
| 请求幅度 | `−0.05 × Δt × mean(u²)` |
| 相邻请求变化 | `−0.02 × Δt × mean(((u − previous_u) / 0.2)²)` |
| 接回进度，仅多球 | 每个确认接回的新释放编号 +`1/N`，回合累计不超过 +1；`N` 为目标抛数 |
| 距离引导、存活、持球、滞空及重复触碰 | 0 |

`Δt` 为实际执行时长，通常为 0.02 s。`u` 在来球接稳环境中是权限处理并限速后的归一化位移和手指动作；在完整单球环境中是限幅、权限处理后的请求；在多球环境中是同样处理后的增益请求。多球零请求对应 XY 增益 0.5、手指增益 1。上述成本不直接计算关节运动、力矩或能耗。

球高、禁止接触、姿态限制和接稳判据由独立验收器检查。0.8 s 是观测中可见的任务截止时间，以 `terminated=True` 返回；它不是训练采样预算造成的截断。

</details>

### 参考结果

每个配置使用测试种子 140000–140019。策略候选要求：每个训练种子均保持零扰动成功数、扰动配对汇总超过经典控制器，且各训练种子的安全失败数不增加。

| 策略 | 零扰动接稳（各 20 次） | 扰动接稳（各 100 次） | 安全失败（各 120 次） | 筛选 |
| --- | --- | --- | --- | --- |
| 经典控制器 | 19 | 67 | 15 | 对照 |
| 固定残差 | 19 | 72 | 15 | 通过 |
| PPO，训练种子 0/1/2 | 19、19、19 | 72、72、71 | 15、15、15 | 通过 |
| SAC，训练种子 0/1/2 | 19、19、17 | 73、72、50 | 15、15、20 | 未通过 |

固定残差达到 PPO 的接稳水平，动态策略的额外收益尚未确认。该表覆盖来球接稳子任务；主动释放和完整三球验收见下文。有限预算结果不表示算法已收敛。

结果：[策略对照报告](results/reception_comparison/report.md)、[PPO 评估报告](results/reception_refinement/report.md)、[PPO 学习曲线](results/reception_refinement/learning_curve.png)、[SAC 学习曲线](results/reception_comparison/sac_learning_curve.png)、[吞吐量对照](results/reception_refinement/benchmark.json)。

固定残差从零请求、三个 PPO 各自的初始输出和接触阶段请求均值中选择，保留名义反馈、接触权限、限幅及变化率限制。参数仅由验证集确定；该对照复用 PPO 训练资产，不单独比较参数搜索成本。

反馈诊断包含球测量完全丢失、遮盖策略直接球状态字段、冻结策略初始输出。后两者仍保留名义参考和经典反馈；现有小样本未显示实时策略反馈的额外收益。

<details>
<summary>训练、独立评估与报告生成命令</summary>

PPO 与 SAC 各使用三个训练种子，每种子 4,096 个控制步，按验证集选择 2,048 或 4,096 步检查点。训练和验证包含零扰动、1/3/10 mm 噪声、40 ms 延迟、30% 丢帧及组合条件；采样概率保存在训练记录中。

验证种子为 90000–90009，独立测试为 140000–140019，反馈诊断为 150000–150004。检查点优先保持零扰动成功数和安全失败数，再比较完成次数与残差；同分选择较早预算。各检查点沿同一次训练推进，评估隔离训练随机状态，并检查模型保存、加载的一致性。

名义回归与吞吐量：

```bash
uv run python -m v8_robust_control.benchmark_reception --episodes 8 --output v8_robust_control/results/reception_refinement/benchmark.json
uv run python -m v8_robust_control.evaluate_residual --seed-start 1000 --episodes 20 --cases baseline --output v8_robust_control/results/reception_refinement/nominal.json
```

PPO 训练与归档：

```bash
uv run python -m v8_robust_control.train_residual --algorithms ppo --steps 4096 --checkpoints 2048 4096 --seeds 0 1 2 --validation-seed-start 90000 --eval-episodes 10 --jobs 3 --output-dir build/v8_reception_refinement
uv run --with matplotlib python -m v8_robust_control.summarize_residual --input-dir build/v8_reception_refinement --output v8_robust_control/results/reception_refinement/training.json --report build/v8_reception_refinement/validation.md --curve-output v8_robust_control/results/reception_refinement/learning_curve.png
```

PPO 独立测试、反馈诊断与报告：

```bash
uv run python -m v8_robust_control.evaluate_residual --input-dir build/v8_reception_refinement --seed-start 140000 --episodes 20 --cases baseline noise_3mm combined noise_10mm delay_40ms dropout_30pct --jobs 3 --output v8_robust_control/results/reception_refinement/test.json
uv run python -m v8_robust_control.audit_residual_feedback --input-dir build/v8_reception_refinement --seed-start 150000 --episodes 5 --output v8_robust_control/results/reception_refinement/feedback.json --checkpoints ppo_seed0_lr0.0003_shape0.zip ppo_seed1_lr0.0003_shape0.zip ppo_seed2_lr0.0003_shape0.zip
uv run python -m v8_robust_control.summarize_residual --input v8_robust_control/results/reception_refinement/test.json --training v8_robust_control/results/reception_refinement/training.json --nominal v8_robust_control/results/reception_refinement/nominal.json --feedback v8_robust_control/results/reception_refinement/feedback.json --report v8_robust_control/results/reception_refinement/report.md
```

相同预算的 SAC 训练与测试：

```bash
uv run python -m v8_robust_control.train_residual --algorithms sac --steps 4096 --checkpoints 2048 4096 --seeds 0 1 2 --validation-seed-start 90000 --eval-episodes 10 --jobs 3 --output-dir build/v8_reception_sac
uv run --with matplotlib python -m v8_robust_control.summarize_residual --input-dir build/v8_reception_sac --output v8_robust_control/results/reception_comparison/sac_training.json --report build/v8_reception_sac/validation.md --curve-output v8_robust_control/results/reception_comparison/sac_learning_curve.png
uv run python -m v8_robust_control.evaluate_residual --input-dir build/v8_reception_sac --seed-start 140000 --episodes 20 --cases baseline noise_3mm combined noise_10mm delay_40ms dropout_30pct --jobs 3 --output v8_robust_control/results/reception_comparison/sac_test.json
```

固定残差的验证集选择与独立测试：

```bash
uv run python -m v8_robust_control.compare_residual select-fixed --input-dir build/v8_reception_refinement --jobs 3 --output v8_robust_control/results/reception_comparison/fixed_selection.json
uv run python -m v8_robust_control.compare_residual test-fixed --selection v8_robust_control/results/reception_comparison/fixed_selection.json --reference v8_robust_control/results/reception_refinement/test.json --output v8_robust_control/results/reception_comparison/fixed_test.json
```

从存档生成策略对照报告：

```bash
uv run python -m v8_robust_control.compare_residual report --ppo-training v8_robust_control/results/reception_refinement/training.json --ppo-test v8_robust_control/results/reception_refinement/test.json --sac-training v8_robust_control/results/reception_comparison/sac_training.json --sac-test v8_robust_control/results/reception_comparison/sac_test.json --fixed-selection v8_robust_control/results/reception_comparison/fixed_selection.json --fixed-test v8_robust_control/results/reception_comparison/fixed_test.json --output v8_robust_control/results/reception_comparison/report.md
```

</details>

### 完整单球名义参考

`single_ball` 使用真实单球模型，从右手持球开始，完成一次物理释放、飞行和左手接回。验收要求开局连续持稳至少 0.05 s、仅一个释放编号、对应编号接回，并在左手连续接稳至少 0.2 s。运行中逐步检查球高、禁止接触、姿态及外力；接回后不再安排下一次抛球。

```bash
uv run python -m v8_robust_control.single_ball --observation-mode truth --output v8_robust_control/results/single_ball_truth.json
uv run python -m v8_robust_control.single_ball --observation-mode sampled --output v8_robust_control/results/single_ball_sampled.json
```

两种模式均使用零感知扰动；`--timeout` 为仿真时间上限，默认 5 s。结果保存初始持稳、释放与接回事件、末端支撑证据，以及源码和规划输入的校验值。

两份参考结果均完成一次释放与接回，开局连续持稳 0.180 s、末端连续接稳 0.200 s，总仿真时长 1.095 s。

该入口用于名义控制器验证；带残差动作的 Gymnasium 接口见下节。

### 完整单球残差环境

[TransferEnv](transfer_env.py) 提供完整单球的 Gymnasium `reset/step` 接口。名义电机参考由经典控制器离线生成；训练时连续执行物理步进，不进行在线规划试算。参考中的球状态仅用于零残差一致性检查，不进入策略观测或控制。

球状态由带时间戳的位置测量估计。[接触估计器](transfer_observation.py)补偿支撑手的运动，脱手后按重力预测；手部运动不会生成球位置测量或刷新其有效期。位置、速度估计误差按控制采样记录，仅用于评估。

| 接口 | 定义 |
| --- | --- |
| 动作 | 两手各两个 XY 请求及一个手指协同，共 6 维；默认每轴请求不超过 5 mm，关节附加目标不超过 0.02 rad、变化率不超过 0.15 rad/s，手指协同不超过 0.1 rad、变化率不超过 0.5 rad/s |
| 观测 | 191 维，包含相对双掌的估计球状态、测量有效性与年龄、关节状态、掌心运动、接触载荷、时间、已施加修正和权限；预留三个球观测槽，单球任务的其余槽无效 |
| 权限 | XY 需要有效且不确定度合格的位置估计；源手在释放后停止修正，接球手保留接触前通道；手指使用实测接触权限，额外闭合需要掌心载荷 |
| 物理初态变化 | `seat_jitter_m` 只在重置时改变右手持球位置；执行过程中不改写球状态 |
| 验收 | 开局连续持稳 0.05 s、实际释放、至少 0.3 s 无手部接触的飞行、0.4–0.75 s 内触及目标掌心、末端连续接稳 0.2 s；释放须在名义时刻 ±50 ms 内，确认释放后不得再次接触源手 |

固定物理初态下，感知噪声不会改变零残差电机重放的轨迹。因此物理初态变化与感知扰动分别定义，并在各控制器之间保持一致，不能将此基线解释为原在线规划器的噪声容忍度。

```bash
uv run python -m v8_robust_control.transfer_env prepare --output-dir build/v8_transfer_reference
uv run python -m v8_robust_control.transfer_env check --reference-dir build/v8_transfer_reference --episodes 5 --output v8_robust_control/results/full_transfer/zero_residual.json
uv run python -m v8_robust_control.train_transfer run --reference-dir build/v8_transfer_reference --output-dir build/v8_transfer_ppo --checkpoints 8192 16384 32768 --seed-start 380000 --episodes 5 --jobs 3 --output v8_robust_control/results/full_transfer/training.json
uv run python -m v8_robust_control.train_transfer report --input v8_robust_control/results/full_transfer/training.json --output v8_robust_control/results/full_transfer/report.md
```

单球反馈收益试验使用三个 PPO 训练种子，按独立的验证场景选择检查点。对照包含零残差、13 组固定请求中的最优者、策略在验证集上的逐时刻平均请求，以及首次有效观测后冻结的输出。该试验要求各训练种子保持理想初态、仅感知噪声及安全指标，并严格提高扰动成功数；未通过时不运行其独立测试。完整单球与多球任务分别验收。

`report` 命令汇总训练与对照指标。完整单球的验收不代替双球、三球及开局收尾的残差策略验收。

<details>
<summary>可选：通过单球反馈收益筛选后的独立测试</summary>

仅在 `full_transfer/training.json` 的 `summary.all_candidates_passed` 为 `true` 时运行以下两条命令。为 `false` 时跳过独立测试及合并报告；单球验证报告、策略录制和多球训练仍可按各自前置条件执行。

```bash
uv run python -m v8_robust_control.train_transfer test --training v8_robust_control/results/full_transfer/training.json --input-dir build/v8_transfer_ppo --seed-start 390000 --episodes 20 --jobs 3 --output v8_robust_control/results/full_transfer/test.json
uv run python -m v8_robust_control.train_transfer report --input v8_robust_control/results/full_transfer/training.json --test v8_robust_control/results/full_transfer/test.json --output v8_robust_control/results/full_transfer/independent_report.md
```

</details>

### 多球残差环境与训练

[CascadeEnv](cascade_env.py) 提供双球交换和包含三球开局、收尾的完整任务。名义电机轨迹离线生成，运行时通过位置估计、真实接触和解析收尾计算反馈，不进行前向物理试算。

| 接口 | 定义 |
| --- | --- |
| 动作 | 6 维增益请求；前四维映射为两手 XY 反馈增益 0–1，零请求对应 0.5；后两维映射为手指反馈增益 0–2，零请求对应 1 |
| 电机修正 | XY 跟随估计状态相对名义轨迹的误差，每轴不超过 5 mm；附加关节目标不超过 0.02 rad、变化率不超过 0.15 rad/s；手指修正不超过 0.55 rad、变化率不超过 5 rad/s |
| 观测 | 191 维，包含估计球状态、名义参考误差、测量有效性与年龄、关节和掌心状态、接触载荷、节拍与动作权限 |
| 时序 | 策略 50 Hz、物理步进 1 kHz、位置测量 60 Hz、测量有效期 50 ms |
| 任务 | 双球各抛接一次，结束连续持稳 0.2 s；三球按指定次数抛接，结束时三球连续持稳 0.8 s；沿用上文的物理、安全和姿态条件 |

零误差时，任意有界反馈增益均保持名义轨迹。无扰动完成数用于兼容性验收，学习收益由扰动条件及对照试验判断。运行中只改变电机目标，球状态仅在重置时设置。策略使用位置测量推导的状态与参考误差，真实球速度用于独立物理验收和误差统计。

名义参考包含实际执行的电机命令、位置估计、接触几何及闭环状态轨迹。包裹在单球真实接触掌心且估计有效时启用；收尾以掌指接触和估计相对速度确认承托，并至少使用三帧离手后的位置测量拟合重力飞行模型。

<details>
<summary>生成参考与零残差结果</summary>

```bash
uv run python -m v8_robust_control.cascade_env prepare --task two --output-dir build/v8_two_ball_reference
uv run python -m v8_robust_control.cascade_env prepare --task complete --throws 30 --output-dir build/v8_complete30_reference
uv run python -m v8_robust_control.cascade_env check --reference-dir build/v8_two_ball_reference --output v8_robust_control/results/full_transfer/two_ball_zero.json
uv run python -m v8_robust_control.cascade_env check --reference-dir build/v8_complete30_reference --output v8_robust_control/results/full_transfer/complete30_zero.json
```

</details>

PPO 使用三个训练种子。双球从单球编码器初始化，重置参考误差通道和动作输出层；三球保留双球策略，并初始化新增球观测通道。默认训练条件为无扰动、3 mm 位置噪声、2 mm 初始球位偏差及二者组合。初始偏差按手采样，两水平轴分别均匀分布于 ±2 mm，同手的球共享位移；验证覆盖全部八类场景。检查点先按无扰动完整任务选择，再比较扰动完成数与安全失败数。

八类场景包括：无扰动、仅 3 mm 噪声、2/5 mm 初态偏差、两种初态偏差分别叠加 3 mm 噪声、2 mm 初态偏差叠加 20 ms 延迟，以及 2 mm 初态偏差叠加 3 mm 噪声、20 ms 延迟和 10% 丢帧。

奖励沿用上述终止奖励与请求成本，并加入接回进度项。重复接触或持球不会重复计分。

<details>
<summary>训练双球与三球策略</summary>

```bash
uv run python -m v8_robust_control.train_cascade run --reference-dir build/v8_two_ball_reference --output-dir build/v8_two_gain_ppo --warm-start-study v8_robust_control/results/full_transfer/training.json --warm-start-dir build/v8_transfer_ppo --checkpoints 1024 4096 8192 --seed-start 380000 --episodes 3 --jobs 3 --output v8_robust_control/results/cascade_training/two.json
uv run python -m v8_robust_control.train_cascade report --input v8_robust_control/results/cascade_training/two.json --output v8_robust_control/results/cascade_training/two.md
uv run python -m v8_robust_control.train_cascade run --reference-dir build/v8_complete30_reference --output-dir build/v8_complete30_gain_ppo --warm-start-study v8_robust_control/results/cascade_training/two.json --warm-start-dir build/v8_two_gain_ppo --checkpoints 1024 4096 8192 --seed-start 380000 --episodes 3 --jobs 3 --output v8_robust_control/results/cascade_training/complete30.json
uv run python -m v8_robust_control.train_cascade report --input v8_robust_control/results/cascade_training/complete30.json --output v8_robust_control/results/cascade_training/complete30.md
```

</details>

验证结果：[双球交换](results/cascade_training/two.md)、[三球完整任务](results/cascade_training/complete30.md)。

[独立评估](evaluate_cascade.py) 先保存候选检查点、对照与测试种子协议，再执行固定测试集。无扰动完整任务、扰动非退化和严格反馈收益分别报告；未通过鲁棒性比较的候选也保留实际失败记录。独立种子不用于训练或重新选择检查点。

<details>
<summary>独立评估与报告</summary>

```bash
uv run python -m v8_robust_control.evaluate_cascade run --training v8_robust_control/results/cascade_training/complete30.json --input-dir build/v8_complete30_gain_ppo --seed-start 390000 --episodes 5 --jobs 3 --protocol-output v8_robust_control/results/cascade_training/complete30_protocol.json --output v8_robust_control/results/cascade_training/complete30_test.json
uv run python -m v8_robust_control.evaluate_cascade report --training v8_robust_control/results/cascade_training/complete30.json --input v8_robust_control/results/cascade_training/complete30_test.json --output v8_robust_control/results/cascade_training/complete30_test.md
```

</details>

评估退出码按无扰动完整任务是否全部完成判定。扰动非退化要求感知噪声及物理初态扰动完成数不降低，且安全失败数不增加；严格反馈收益还要求物理初态扰动完成数超过零残差、时序增益和自身冻结增益对照。

非退化是相对比较，不表示扰动任务通过。零完成数即使与对照持平，也不构成鲁棒性提升的证据。

独立测试使用种子 390000–390004，每类场景五次。无扰动重复相同物理初态，用于检查任务兼容性与重置一致性。

| 控制器 | 无扰动（各 5 次） | 仅 3 mm 噪声（各 5 次） | 初态扰动及组合（各 30 次） | 安全失败（各 40 次） |
| --- | --- | --- | --- | --- |
| 零请求增益 | 5 | 0 | 0 | 29 |
| 时序增益 | 5 | 0 | 0 | 31 |
| PPO，训练种子 0/1/2 | 5、5、5 | 0、0、0 | 0、0、0 | 33、33、32 |

三个 PPO 策略均通过无扰动完整任务验收；扰动任务未完成，安全失败数高于零请求对照，未通过扰动非退化和严格反馈收益验收。

结果：[独立评估报告](results/cascade_training/complete30_test.md)、[原始记录](results/cascade_training/complete30_test.json)、[冻结协议](results/cascade_training/complete30_protocol.json)。

### 录制残差策略视频

<details>
<summary>录制选定策略</summary>

单球录制需要完整单球训练的 `training.json` 和检查点；双球、三球分别需要对应的多球训练记录和检查点。

```bash
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/full_transfer/training.json --input-dir build/v8_transfer_ppo --training-seed 0 --seed 0 --case clean --video videos/v8_ppo_single_seed0.mp4 --output v8_robust_control/results/policy_videos/single_seed0.json
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/cascade_training/two.json --input-dir build/v8_two_gain_ppo --training-seed 0 --seed 0 --case clean --video videos/v8_ppo_two_seed0.mp4 --output v8_robust_control/results/policy_videos/two_seed0.json
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/cascade_training/complete30.json --input-dir build/v8_complete30_gain_ppo --training-seed 0 --seed 0 --case clean --video videos/v8_ppo_complete30_seed0.mp4 --output v8_robust_control/results/policy_videos/complete30_seed0.json
```

`--training-seed` 选择训练种子，`--seed` 设置回合随机种子，`--case` 选择与训练评估相同的场景。视频和元数据路径须为新文件。

</details>

## 结果复现

`results/` 保存当前参考实验的原始记录、汇总指标、报告和视频元数据。按对应章节的生成命令依次运行，可重新生成以下结果：

| 结果文件 | 生成入口 |
| --- | --- |
| `truth_baseline30/50.json`、`sampled_baseline30/50.json` | `verify_baselines`，分别运行 truth、sampled |
| `complete_truth30/50.json`、`complete_sampled30/50.json` | `complete_task`，分别运行四种组合 |
| `perception_trials.jsonl`、`perception_trials.csv`、`perception_sensitivity.md` | `evaluate_perception` → `summarize_perception` |
| `complete_perception_trials.jsonl`、`complete_perception_trials.csv`、`complete_perception_sensitivity.md` | 同上，评估指定 `--task complete` |
| `video_comparison.json`、`complete_video_comparison.json` | 相应评估录制原始视频 → `compare_videos` |
| [reception_refinement/](results/reception_refinement/) | 名义回归、吞吐量、PPO 训练归档、独立测试、反馈诊断、报告 |
| [reception_comparison/](results/reception_comparison/) | SAC 训练归档与测试、固定残差选择与测试、对照报告 |
| `single_ball_truth.json`、`single_ball_sampled.json` | `single_ball`，分别运行 truth、sampled |
| [full_transfer/](results/full_transfer/) | 单球及多球零残差检查、单球 PPO 反馈收益试验 |
| [policy_videos/](results/policy_videos/) | `record_policy`，加载选定 PPO 检查点并录制物理回合 |
| [cascade_training/](results/cascade_training/) | 多球 PPO 训练验证、冻结评估协议、独立测试及报告 |

完整抛接结果按名义参考、单球训练、双球训练、三球训练、独立评估、策略录制的顺序生成。

除显式支持 `--overwrite` 的汇总命令外，输出路径须为新路径。重新生成结果时还需检查以下输出位置：

- `build/v8_reception_refinement/`、`build/v8_reception_sac/`：保留且与源码匹配时，可跳过训练，从归档命令开始。归档命令的同名 `validation.md` 也须移走或改用新报告路径；重新训练须使用新目录，并同步后续输入路径。
- `build/v8_transfer_reference/`、`build/v8_transfer_ppo/`：分别保存完整单球的名义电机参考、训练模型及检查点；准备或训练时使用新目录。
- `build/v8_two_ball_reference/`、`build/v8_complete30_reference/`：双球及三球任务的名义参考；`build/v8_two_gain_ppo/`、`build/v8_complete30_gain_ppo/`：对应的 PPO 检查点。重新准备或训练时使用新目录，并同步后续输入路径。
- `videos/`：录制与合成命令均拒绝覆盖同名 MP4。重跑前移走对应视频，或通过评估器的 `--video-dir` 和合成器的 `--output` 指定新位置。

报告可以从原始记录重新生成。来球接稳训练目录可通过 `summarize_residual` 导出训练归档；完整抛接训练的原始记录缺失时，需要重新运行相应训练与评估。视频元数据通过重新录制生成，已有的同名 MP4 须先移走或更换输出路径。单元测试不依赖 `results/`。

复现采用相同任务、配置、种子和验收流程，不保证文件逐字节一致。耗时、并行记录顺序和模型文件校验值可能变化；依赖、硬件及数值计算差异也可能影响训练和接触结果，应以重新运行的验收指标为准。模型兼容性由环境版本、源码和检查点校验值检查，修改观测或控制代码后需重新训练。
