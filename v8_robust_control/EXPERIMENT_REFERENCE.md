# v8 实验与接口参考

本页保存各实验入口、接口说明及历史结果。首次运行见 [v8 README](README.md) 和 [Quickstart](QUICKSTART_RL.md)。各节的结果对应各自注明的任务、控制器版本和评估配置。

为 G1 上身与双 LEAP Hand 三球抛接仿真提供球观测、状态估计、感知扰动、残差训练和完整任务评估。物理模型复用 [v6](../v6_humanoid_hands/README.md)，完整任务的三球开局与收尾复用 [v7](../v7_three_ball_hold/README.md)。抛接通过手与球的物理接触实现。

报告对应的配置与结果见[证据索引](../report/evidence/README.md)。部分历史诊断依赖本地生成的中间数据或原交接包。

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

三点差分会放大位置噪声；相应扰动结果反映观测、估计与规划整条链路的敏感性，不能单独归因于控制器。事件级接口默认使用三点差分，也支持下述滤波模式；电机残差环境使用带重力先验及接触处理的 [SupportPositionTracker](transfer_observation.py)，各入口的结果应分别比较。

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

测试覆盖时间戳、状态估计、无效数据处理、接触权限转换、规划试算隔离、动作规划与实验记录导出。

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

以下视频加载训练记录选定的 PPO 检查点。控制方式为名义控制器加 PPO 残差，正常速度播放；未标注扰动的案例采用无扰动配置。单次成功演示不代表总体性能提升。

| 任务 | 视频 | 元数据 |
| --- | --- | --- |
| 完整单球抛接 | [单球](../videos/v8_ppo_single_seed0.mp4) | [single_seed0.json](results/policy_videos/single_seed0.json) |
| 双球交换 | [双球](../videos/v8_ppo_two_seed0.mp4) | [two_seed0.json](results/policy_videos/two_seed0.json) |
| 三球 30 抛及收尾 | [三球](../videos/v8_ppo_complete30_seed0.mp4) | [complete30_seed0.json](results/policy_videos/complete30_seed0.json) |
| 三球 30 抛及收尾，`guarded_feedback`，3 mm 噪声 | [感知噪声](../videos/v8_ppo_feedback_noise3_seed0.mp4) | [feedback_noise3_seed0.json](results/policy_videos/feedback_noise3_seed0.json) |

`guarded_feedback` 视频的附加关节和手指修正均为零，实际物理状态与名义参考一致。该回合展示带噪测量下保持名义轨迹，不作为 PPO 改善任务性能的证据。

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
| 动作 | 默认 `reference` 控制器使用 6 维增益请求；前四维映射为两手 XY 反馈增益 0–1，零请求对应 0.5；后两维映射为手指反馈增益 0–2，零请求对应 1 |
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

### 感知扰动下的反馈控制

[feedback_env.py](feedback_env.py) 提供可选的反馈控制器，沿用多球任务、观测和电机修正上限。训练和评估记录保存控制器名称，检查点按对应环境加载。

| `--controller` | 请求与控制方式 |
| --- | --- |
| `reference`（默认） | 上述增益映射；在策略时刻按接触权限屏蔽请求 |
| `latched_feedback` | 保持相同映射，保存策略的原始请求；在每个伺服步检查实际控制权限 |
| `guarded_feedback` | 保存原始请求；非正请求关闭修正，正请求 `u` 映射为 XY 增益 `u²`、手指增益 `2u²`；根据测量不确定度检查偏差证据 |

`reference` 的零请求对应非零增益，因此接触权限在两次策略更新之间改变时，屏蔽后的请求可能启用默认反馈。`latched_feedback` 保留策略发出的关闭请求，实际电机修正仍受有效估计、接触和释放权限约束。

`guarded_feedback` 要求连续三个不同采样时刻的测量，在同一主轴方向上超过位置估计标准差的三倍，才启用对应偏差修正。初始持球阶段仅在真实承托接触和有效位置估计下允许手指居中修正。收尾同样检查连续三帧显著偏差；证据不足时继续执行名义收尾轨迹，测量无效或过期时清除证据。

这些控制器只修改电机目标。物理验收、终止奖励、请求幅度与变化率成本保持不变；请求成本按策略时刻具有权限的原始请求计算。不同增益映射的收益不能仅按回合奖励比较，评估使用完成数和物理安全指标。

`train_cascade run --controller motor_residual` 提供有界关节残差接口，每臂控制肩部侧摆、肘部和腕部俯仰。关节偏置上限为 0.02 rad，变化率上限为 0.15 rad/s，在 1 kHz 伺服步上执行；名义手指命令保持不变。策略使用球位置估计，以及相对名义轨迹的关节、手指和掌心误差。

该接口沿用离线参考的时序，不提供抛球等待时间、接球路径参数或接触状态机转换的动作。

该接口根据启动前 0.22 s 的关节位置与速度偏差确定修正权限，随后保持该权限。关节编码器采用无噪声仿真读数；仅球感知受扰动时，修正保持为零。此权限机制用于初态偏差试验，不覆盖运行中新增的物理扰动。

添加 `--motor-curriculum` 可按确认接球次数训练短回合，并记录课程进度。仅少于完整任务次数的目标可提前结束并获得前缀奖励；达到完整次数时仍须完成末端接稳，不叠加前缀奖励。独立评估执行完整次数、开局与收尾验收。该接口尚无通过完整扰动任务验收的策略。

电机接口的冻结输出对照在所有球观测有效、启动观察完成且修正权限开放后取一次策略输出；此前保持零请求。冻结输出不随后续观测改变。

PPO 在电机权限关闭时输出零请求，并屏蔽这些动作的策略概率与熵梯度；价值网络仍学习对应状态的回报。

电机策略的 gSDE 初始归一化动作标准差上界约为 0.04，与环境的动作和变化率边界分别约束。

诊断电机请求、控制权限与释放误差：

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --controllers motor_residual --cases clean perception_extreme seat_2mm --requests zero positive negative --seeds 410200 410201 --jobs 2 --output build/v8_motor_action_audit.json
```

诊断保存每次释放确认时的物理状态、相同仿真时刻的名义状态、位置与速度估计误差和终止球状态。物理真值只用于诊断，不加入策略观测。`positive` 和 `negative` 分别为全维度 +1 和 −1 请求；关节残差的关闭请求为 `zero`。

使用相同初态偏差对照原规划控制器：

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --native-timeout 2.2 --cases seat_2mm --seeds 410200 --jobs 1 --output build/v8_native_action_audit.json
```

`--native-timeout` 指定仿真时间上限，并改为运行原规划控制器。短时诊断用于定位回放与规划控制的差异，超时不计为完整任务成功。

<details>
<summary>反馈诊断与策略训练</summary>

先生成上述单球、双球和三球名义参考及单球训练结果，再运行：

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --seeds 410000 410001 --jobs 3 --output v8_robust_control/results/feedback_training/diagnosis.json
uv run python -m v8_robust_control.train_cascade run --controller guarded_feedback --reference-dir build/v8_two_ball_reference --output-dir build/v8_feedback_two_final_ppo --warm-start-study v8_robust_control/results/full_transfer/training.json --warm-start-dir build/v8_transfer_ppo --checkpoints 4096 --seed-start 410100 --episodes 3 --jobs 3 --output v8_robust_control/results/feedback_training/two.json
uv run python -m v8_robust_control.train_cascade report --input v8_robust_control/results/feedback_training/two.json --output v8_robust_control/results/feedback_training/two.md
uv run python -m v8_robust_control.train_cascade run --controller guarded_feedback --reference-dir build/v8_complete30_reference --output-dir build/v8_feedback_complete30_final_ppo --warm-start-study v8_robust_control/results/feedback_training/two.json --warm-start-dir build/v8_feedback_two_final_ppo --checkpoints 4096 --seed-start 410100 --episodes 3 --jobs 3 --output v8_robust_control/results/feedback_training/complete30.json
uv run python -m v8_robust_control.train_cascade report --input v8_robust_control/results/feedback_training/complete30.json --output v8_robust_control/results/feedback_training/complete30.md
```

</details>

诊断使用相同初态和扰动序列，比较默认请求及关闭请求在实际接触转换中的电机修正。训练使用三个 PPO 种子，各训练 4096 步；双球迁移单球编码器并重置动作层，三球迁移同控制器的双球策略。诊断种子、验证种子与独立测试种子分开。

记录：[接触转换诊断](results/feedback_training/diagnosis.json)、[双球验证](results/feedback_training/two.md)、[三球验证](results/feedback_training/complete30.md)。

独立评估分为两套条件：`reference` 包含物理初态偏差及组合扰动；`perception` 固定物理初态，仅改变位置噪声（1、3、5、10 mm）、延迟（20、40 ms）、丢帧（10%、30%）及组合条件。每套均包含零请求、时序增益、三个 PPO 策略及各自的冻结输出对照。

<details>
<summary>独立评估反馈控制器</summary>

```bash
uv run python -m v8_robust_control.evaluate_cascade run --training v8_robust_control/results/cascade_training/complete30.json --input-dir build/v8_complete30_gain_ppo --suite perception --seed-start 510000 --episodes 5 --jobs 3 --protocol-output v8_robust_control/results/feedback_training/reference_perception_protocol.json --output v8_robust_control/results/feedback_training/reference_perception_test.json
uv run python -m v8_robust_control.evaluate_cascade report --training v8_robust_control/results/cascade_training/complete30.json --input v8_robust_control/results/feedback_training/reference_perception_test.json --output v8_robust_control/results/feedback_training/reference_perception_test.md
uv run python -m v8_robust_control.evaluate_cascade run --training v8_robust_control/results/feedback_training/complete30.json --input-dir build/v8_feedback_complete30_final_ppo --suite reference --seed-start 500000 --episodes 5 --jobs 3 --protocol-output v8_robust_control/results/feedback_training/reference_protocol.json --output v8_robust_control/results/feedback_training/reference_test.json
uv run python -m v8_robust_control.evaluate_cascade report --training v8_robust_control/results/feedback_training/complete30.json --input v8_robust_control/results/feedback_training/reference_test.json --output v8_robust_control/results/feedback_training/reference_test.md
uv run python -m v8_robust_control.evaluate_cascade run --training v8_robust_control/results/feedback_training/complete30.json --input-dir build/v8_feedback_complete30_final_ppo --suite perception --seed-start 510000 --episodes 5 --jobs 8 --protocol-output v8_robust_control/results/feedback_training/perception_protocol.json --output v8_robust_control/results/feedback_training/perception_test.json
uv run python -m v8_robust_control.evaluate_cascade report --training v8_robust_control/results/feedback_training/complete30.json --input v8_robust_control/results/feedback_training/perception_test.json --output v8_robust_control/results/feedback_training/perception_test.md
```

</details>

感知测试使用相同种子比较 `reference` 与 `guarded_feedback`。两种控制器的零请求对应不同增益，跨控制器结果反映整体控制配置的变化；PPO 收益由同一控制器内的零请求、时序和冻结对照判断。

固定物理初态下，抑制缺乏偏差证据的测量修正可以保持名义轨迹；这不代表能纠正真实飞行偏差。感知扰动下成功不代表物理初态偏差下成功；PPO 与简单控制持平时，也不构成学习收益的证据。

感知独立评估使用种子 510000–510004，每类条件五次。各控制方案无扰动均完成 5/5；下表的扰动完成数包含九类感知扰动，安全失败数包含全部十类条件。

| 控制配置 | 感知扰动完成 | 安全失败 |
| --- | --- | --- |
| `reference`，零请求 | 0/45 | 41/50 |
| `guarded_feedback`，零请求 | 45/45 | 0/50 |
| `guarded_feedback`，时序增益 | 40/45 | 1/50 |
| `guarded_feedback`，PPO 种子 0 | 41/45 | 2/50 |
| `guarded_feedback`，PPO 种子 1 | 40/45 | 1/50 |
| `guarded_feedback`，PPO 种子 2 | 41/45 | 0/50 |

PPO 的感知测试失败集中在 5 mm 和 10 mm 噪声条件，三个策略均未通过相对零请求的扰动非退化验收，不作为鲁棒改进策略推荐。物理初态测试使用种子 500000–500004，全部八种控制方案的初态偏差及组合条件均为 0/30。上述结果不表明已解决真实初态变化或取得 RL 额外收益。

独立报告：[reference 感知测试](results/feedback_training/reference_perception_test.md)、[guarded_feedback 感知测试](results/feedback_training/perception_test.md)、[物理初态及组合测试](results/feedback_training/reference_test.md)。每类五次的结果用于固定测试集比较，不作为总体成功概率的精确估计。

### 任务参数接口

[TaskSpaceCascade](task_control.py) 在原控制器的抛球、接球调用点接收有界参数，保留接触状态机和手指反馈，省去在线物理试算。回调输入包含带时间戳的球状态估计、关节状态、掌心运动及掌指法向力，不包含球真值速度。底层经典接触控制仍使用仿真接触几何和球状态。

| 动作 | 限制 |
| --- | --- |
| 抛球命令残差 | XYZ 速度分别为 ±0.75、±0.75、±1.25 m/s；掌面法向偏置 ±0.25 |
| 手指释放提前量残差 | ±15 ms，绝对提前量不超过模型原有限制 |
| 抛球等待 | 每次 5–25 ms，按物理步对齐；每次抛球累计不超过 35 ms，保留重试次数上限 |
| 接球目标残差 | XYZ 各 ±60 mm，法向偏置各 ±0.25，并进一步受原接球参数范围约束 |
| 接球倾斜幅度倍率 | 1–1.5，沿用模型原有限制 |

`decode_action` 将六维 `[-1, 1]` 动作转换为对应阶段的参数。收尾肘部反馈修正外展趋势；腕部与支架的距离反馈沿掌心位置和法向任务的零空间抑制接近速度，整体缩放修正量以满足原力矩上限。修正受执行器余量约束，不能替代轨迹可行性检查。球高、关节姿态、4 mm 穿透上限及最终持球仍按原标准验收。

等待分量先按物理步量化，不足 5 ms 的量化值按零处理，避免小幅动作被强制转换为一次等待。

起抛沿用完整任务控制器的接触条件：球已确认接稳、保持手球接触，球相对掌心的位置与速度满足持球要求。训练课程另以掌心距回收目标不超过 30 mm、速度不超过 0.30 m/s 统计回收完成；该指标不额外阻塞正常起抛。接触时掌心下沉速度超过 1.8 m/s 时，参考制动段缩短至不超过 45 ms，保留边界状态、整段回收的结束时间及姿态；实际减速由受限关节力矩和手球接触产生。

固定名义命令保存在 [event_seed_plans.json](event_seed_plans.json)，来自原完整任务控制器通过验收后的实际执行记录，并经过事件入口独立重放。它与 `complete_seed_plans.json` 的用途不同，后者为在线优化器提供初值。固定命令的离线标定收益不计作 RL 收益。

运行零参数残差：

```bash
uv run python -m v8_robust_control.task_control --throws 30 --output build/v8_task_zero30.json
```

检查初始持球位置偏差：

```bash
uv run python -m v8_robust_control.task_control --seat-jitter-mm 2 --seed 410202 --output build/v8_task_seat2mm_seed410202.json
```

`--seat-jitter-mm` 在重置时对初始持球位置施加均匀分布的水平偏差，同手两球使用相同偏移；仿真开始后不改写球状态。添加 `--disable-stand-guard` 可对照腕部距离反馈。该接口尚未通过 2 mm 初态偏差下的多种子完整任务验收。

`--recovery-gate` 额外要求起抛前达到上述世界坐标回收条件，供控制时序对照；`--disable-adaptive-braking` 保留原制动参考。两者均不改变物理验收标准。

机器人、轨迹或接触参数变更后，可重新导出候选固定命令并独立验收：

```bash
uv run python -m v8_robust_control.calibrate_complete --throws 50 --output build/v8_native_guesses.json --report build/v8_native_calibration.jsonl --executed-output build/v8_event_candidate.json
uv run python -m v8_robust_control.task_control --throws 50 --estimator filtered --seed-plans build/v8_event_candidate.json --timeout 20 --output build/v8_event_candidate50.json
```

第一条命令保存在线规划初值和实际执行命令，第二条关闭在线物理试算后验证候选命令。候选通过后再更新固定名义命令及对应参考；仅通过原控制器验收不代表事件入口通过。

检验记录中的规划决定能否由该接口执行：

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --native-timeout 12 --cases seat_2mm --seeds 410200 --jobs 1 --output build/v8_task_oracle.json
uv run python -m v8_robust_control.task_control --oracle-input build/v8_task_oracle.json --case seat_2mm --seed 410200 --output build/v8_task_oracle_replay.json
```

`build/v8_complete30_reference` 的生成命令见多球训练部分。规划记录包含接受、拒绝和重试决定；重放校验任务源码、MuJoCo 版本和初态。添加 `--disable-ending-guard` 可单独比较专用收尾保护的作用。未完成任务时保存实际结果并返回非零退出码；记录提前结束时，以 `ORACLE_EXHAUSTED` 标记。

记录中的决定使用同一回合的信息，只用于动作表达能力检查，不作为因果策略或 RL 收益证据。该接口尚未完成多种子扰动验收。

### 事件驱动参数训练

[TaskResidualEnv](task_env.py) 将任务参数接口接入 Gymnasium，[train_task.py](train_task.py) 提供 PPO、SAC 小试与对照评估。策略在实际抛球、接球调用点决策，训练预算按决策次数计数；底层伺服与物理步进为 1 kHz，位置测量为 60 Hz。该入口使用独立的观测与动作标识，检查点不与 191 维反馈增益策略通用。

此处零残差指关闭在线物理试算后的名义控制器，与保留在线规划的 `complete_task` 入口不同。前者用于同一环境和计算路径下的残差对照，后者用于评估原完整控制器。二者的规划成本及附加保护不同，零残差下的退化不能直接归因于原控制器，也不能单独证明引入 RL 的必要性。

事件级观测可选择 `difference`（默认三点差分）或 `filtered`（接触跟踪与弹道拟合）。滤波模式在接触阶段复用掌心本体感觉和位置测量；离手后，仅用最后一次手球接触之后采样的位置，按配置重力拟合位置和速度。至少三帧有效飞行测量后才提供飞行估计；过期或无效测量不由本体感觉补造。拟合使用原采样时间，并补偿采样至控制时刻的时差及半隐式积分约定，不读取球速度真值或使用标称释放速度代替测量。

比较两种估计方式下的零残差完整任务：

```bash
uv run python -m v8_robust_control.train_task observation-baseline --estimators difference filtered --cases clean sensor_only_1mm sensor_only_3mm sensor_only_5mm seat_2mm combined --seeds 416100 416101 --jobs 3 --output build/v8_event_estimator_baseline30.json
uv run python -m v8_robust_control.train_task observation-baseline --estimators difference filtered --cases clean --seeds 0 --throws 50 --jobs 2 --output build/v8_event_estimator_baseline50.json
```

报告保存完整任务结果、估计模式、固定测试配置和源码校验值。位置与速度误差仅在暂停的决策回调处作只读评估，不输入控制器。该对照不训练策略，估计器改进与 RL 收益分别评估。

无扰动下，差分与滤波模式均通过 30 次和 50 次完整任务验收；50 次对照使用种子 0，零球球碰撞、无抓握约束，最终三球连续持稳 0.8 s。扰动性能须使用相同固定命令分别评估，较低的估计误差不能替代完整任务与安全验收。

30 次测试使用种子 416100、416101。下表列出每回合已确认接回次数；所有扰动条件下两种模式的完整任务成功数均为 0/2，不能据此宣称抗扰验收通过。

| 条件 | 差分模式 | 滤波模式 |
| --- | --- | --- |
| 无扰动 | 30、30 | 30、30 |
| 位置噪声 1 mm | 0、1 | 4、4 |
| 位置噪声 3 mm | 0、0 | 2、2 |
| 位置噪声 5 mm | 0、0 | 2、1 |
| 初始持球偏差 2 mm | 0、1 | 0、1 |
| 组合扰动 | 0、0 | 0、1 |

生成滤波模式的训练参考：

```bash
uv run python -m v8_robust_control.train_task prepare --estimator filtered --output build/v8_task_filtered_reference.json
```

后续 `pilot`、`audit` 和 `evaluate` 使用该文件作为 `--reference` 时沿用其估计模式。参考文件、源码和检查点须匹配；三点差分与滤波模式的参考不能混用。独立运行任务也可通过 `task_control --estimator filtered` 选择滤波模式。此选项不改变经典 `juggle`、`complete_task` 或电机残差环境的估计链路。

观测包含球的采样估计及有效期、关节和掌心状态、掌指法向力、名义命令和事件上下文。参考偏差来自单独零扰动回合的采样记录，按事件匹配。全部球的采样偏差及当前侧手臂、手指、掌心的本体感觉用于判定残差权限：相关测量与参考在数值容差内一致时执行零残差，否则保留声明的动作范围。未匹配到参考的事件保留动作权限，探索幅度由策略分布控制。

奖励按确认接回的完整飞行计数，每次 +1；完整任务额外 +10，所有任务失败统一 -20。动作幅度与同手同阶段的动作变化分别按均方值扣分，系数均为 0.002；没有存活或空中停留奖励。球高、碰撞、姿态及末端接稳仍独立验收。任务截止时间记为失败，计算预算截断单独标记。

参数分为控制范围、训练设置和验收标准。部分参数由命令行指定，其余为对应模块中的常量。改变控制或观测定义后需重新生成兼容的参考及检查点；优化超参数的试验保留独立验证和测试。

| 类别 | 当前设置 | 定义位置 |
| --- | --- | --- |
| 抛球修正 | 速度三个分量上限为 ±0.75、±0.75、±1.25 m/s；掌面法向参数 ±0.25；手指提前量残差 ±15 ms | `task_control.THROW_RESIDUAL_LIMITS` |
| 接球修正 | 位置各轴 ±60 mm；两个法向参数各 ±0.25；倾斜倍率 1–1.5，另受接球参数边界限制 | `task_control.RECEPTION_RESIDUAL_LIMITS`、`decode_action` |
| 抛球等待 | 单次最多 25 ms、累计最多 35 ms，按物理步长离散，受最小重试间隔限制 | `task_control.MAX_DEFER_S`、`MAX_TOTAL_DEFER_S` |
| 扰动与任务 | 初始位置偏差、感知噪声、延迟、丢帧率、次数、截止时间和决策预算 | `TaskConfig` |
| PPO 更新 | 学习率 `1e-4`，每批采集 64 次决策，批大小 32，更新 5 轮，目标 KL 为 0.01，熵系数为 0 | `train_task.make_model` |
| 探索初始化 | 六维标准差边界为 `(0.20, 0.05, 0.05, 0.05, 0.05, 0.05)` | `train_task.INITIAL_ACTION_STD_BOUNDS` |

RL 的奖励项如下；`task_imitation` 和 `task_corrections` 使用动作标签及保持损失，不使用这些奖励进行优化。

| 项目 | 数值或规则 |
| --- | --- |
| 确认接回 | 每次 +1 |
| 完整任务成功 | 额外 +10 |
| 任务失败 | 一次 −20；计算预算截断不作为任务失败扣分 |
| 动作幅度、变化率 | 分别为均方值乘以 −0.002 |
| 课程前缀 | 达到当前真实回收目标额外 +2，不作为完整任务成功 |
| 可选势函数塑形 | `Φ(next) − Φ(current)`，`Φ ∈ [−0.25, 0]`，由 `--curriculum-shaping` 启用 |
| 存活、持球时间、空中时间 | 无奖励 |

RL 折扣因子为 1。表中为原始奖励，`train_task pilot` 的 `--reward-scale` 默认值为 `0.01`，仅用于训练回报缩放，评估仍报告原始单位。球高、碰撞、姿态和接稳判据属于独立验收约束，不通过修改奖励权重放宽。

课程前缀完成指当前回合达到指定的真实接稳与回收目标。例如 `catch_goal=3` 要求第 1–3 次飞行均已接稳并回收，额外奖励 +2；它不表示 30 次完整任务成功。这里的 1/3/6 次是同一三球场景的回收目标，不是球数。自动课程使用最近 32 回合的结果，成功率达到 75% 后在下一次重置时升级。默认阶段依次为：1 次回收、初态偏差 ±0.5/1/2 mm，再到 ±2 mm 下的 3/6/30 次回收。监督微调和闭环诊断使用固定的 3 次回收阶段，不自动升级，也不使用课程奖励训练模型。完整任务阶段不发前缀奖励，按完整任务成功判据验收。

生成名义采样参考，再运行短预算试验：

```bash
uv run python -m v8_robust_control.train_task prepare --output build/v8_task_event_reference.json
uv run python -m v8_robust_control.train_task pilot --reference build/v8_task_event_reference.json --steps 64 --train-seeds 0 --validation-seeds 610000 --test-seeds 630000 --train-cases seat_0p5mm seat_2mm --cases clean seat_2mm combined --jobs 1 --output-dir build/v8_task_event_pilot --output build/v8_task_event_pilot.json
```

报告比较零残差、简单反馈、由验证集拟合的固定及逐时刻请求、PPO/SAC；测试种子不参与拟合。`combined` 同时包含 2 mm 初态偏差、3 mm 位置噪声、20 ms 延迟和 10% 丢帧。参考、检查点和小试报告保存在 `build/`，输出路径须为新文件或新目录。

检查有界动作对早期失败的影响：

```bash
uv run python -m v8_robust_control.train_task audit --reference build/v8_task_event_reference.json --seeds 410200 410201 410202 --profiles zero lower_third_throw_medium lower_first_throw lower_second_throw --jobs 3 --output build/v8_task_action_audit.json
```

动作检查运行固定的 1.8 s 时间窗口，记录每个开发种子的实际失败与接回次数。`safe_window` 仅表示窗口内未触发物理或姿态失败，不表示完整任务成功。

在滤波参考上检查 2 mm 初态偏差的首抛修正：

```bash
uv run python -m v8_robust_control.train_task audit --reference build/v8_task_filtered_reference.json --seeds 418000 418001 418002 --profiles zero forward_first_throw_small backward_first_throw_small lateral_first_throw_positive_small lateral_first_throw_negative_small earlier_first_release_small --jobs 3 --output build/v8_filtered_action_diagnosis.json
```

速度探测分别为前后或左右 ±0.075 m/s，释放提前量探测为 +3 ms；其余命令保持名义值。报告按相同物理初态配对，保存决策时间、估计有效性与误差、动作权限、实际解码命令、接触事件和终止状态。`paired_probes` 比较实际非零动作次数、接回与回收次数变化及安全退化。仿真真值只作只读诊断，不输入策略或奖励。

检查后续抛球与首次接球对早期失败的影响：

```bash
uv run python -m v8_robust_control.train_task audit --reference build/v8_task_filtered_reference.json --seeds 418000 418001 418002 --profiles zero higher_first_reception lower_second_throw lateral_second_throw_positive lateral_second_throw_negative lower_third_throw_medium --jobs 3 --output build/v8_filtered_event_diagnosis.json
```

先检查动作是否在失败前获得执行机会，再判断接回及安全是否改善。有限动作未获益不能证明整个动作空间不可行；窗口内接回增加也不能替代完整任务验收。开发中查看过的种子不再用于独立测试。

使用接球与回收课程训练：

```bash
uv run python -m v8_robust_control.train_task pilot --reference build/v8_task_event_reference.json --curriculum --curriculum-shaping --development-only --steps 512 --train-seeds 0 1 2 --validation-seeds 410200 410201 410202 410203 410204 410205 --jobs 3 --output-dir build/v8_task_development_ppo --output build/v8_task_development_ppo.json
uv run python -m v8_robust_control.train_task pilot --algorithm sac --reference build/v8_task_event_reference.json --curriculum --curriculum-shaping --development-only --steps 512 --train-seeds 0 1 2 --validation-seeds 410200 410201 410202 410203 410204 410205 --jobs 3 --output-dir build/v8_task_development_sac --output build/v8_task_development_sac.json
```

`--development-only` 仅评估开发种子上的首次接稳与回收，不执行独立完整任务测试，报告的 `all_screened` 为 `null`。完整任务训练使用更长预算，并用独立种子进行对照评估：

```bash
uv run python -m v8_robust_control.train_task pilot --reference build/v8_task_event_reference.json --curriculum --curriculum-shaping --steps 4096 --train-seeds 0 1 2 --validation-seeds 890000 --test-seeds 910000 910001 --cases clean seat_2mm combined --jobs 3 --output-dir build/v8_task_confirm_ppo --output build/v8_task_confirm.json
```

训练和开发评估使用相同的 12 s 基础配置，课程在真实接稳与回收后结束。重新评估已有检查点：

```bash
uv run python -m v8_robust_control.train_task evaluate --training build/v8_task_development_ppo.json --development-only --seeds 410200 410201 410202 410203 410204 410205 --jobs 3 --output build/v8_task_evaluation_ppo.json
uv run python -m v8_robust_control.train_task evaluate --training build/v8_task_development_sac.json --development-only --seeds 410200 410201 410202 410203 410204 410205 --jobs 3 --output build/v8_task_evaluation_sac.json
uv run python -m v8_robust_control.train_task evaluate --training build/v8_task_development_ppo.json --training-seed 2 --seeds 910000 910001 --validation-seeds 890000 --cases clean seat_2mm combined --output build/v8_task_complete_evaluation.json
```

`evaluate` 只读取保存的策略，报告绑定原训练记录、模型及参考校验值。省略 `--development-only` 时执行完整 30 次任务，并比较零残差、简单反馈、固定及时序对照；后两者仅拟合验证回合中实际开放残差的决策。`--training-seed` 选择检查点，`--input-dir` 可指定模型目录。

课程先在 0.5、1、2 mm 初态偏差下完成一次接回与回收，再将连续目标增加为 3、6、30 次。短目标要求对应飞行已确认接稳，且手掌实际回到距抛球起点 30 mm 内、速度不超过 0.30 m/s；每个短回合额外奖励 +2，单独记入训练奖励。最近 32 回合至少 75% 达标后才推进课程。完整 30 次阶段不提前结束、不增加短回合奖励，仍须完成末端接稳。

`--curriculum-shaping` 添加采样误差的有界势函数引导：`F = Φ(next) − Φ(current)`，`Φ` 范围为 `[-0.25, 0]`，按 20 mm 位置和 0.5 m/s 速度偏差归一化。物理终止及课程目标终止时，后续势为零；该引导在相同初态下的累计量与终止时机无关。

独立评估始终执行完整 30 次任务，不使用课程截断或引导奖励。归一化动作首维的初始探索标准差上界为 0.20，其余维为 0.05，动作经压缩和原有限幅后执行。首维对应抛球前后速度或接球前后位置偏置。报告分别记录课程进度与完整任务结果。任务控制器通过原有执行器和力矩限制制动接近限位的肘部，实际掉球、碰撞及姿态越限仍按原规则验收。

有限时长任务使用 `gamma=1`，等待决策不会折减失败惩罚。`--reward-scale` 默认为 `0.01`，仅缩放优化器接收的训练回报，保留奖励比例；评估和事件账本使用原始单位。

`--algorithm` 默认为 `ppo`。SAC 使用相同观测、动作、课程和奖励，前 128 次决策使用从零均值初始化的 gSDE 探索；之后每 4 次决策进行一次回放更新，熵系数按原始单位的 `0.01` 同步缩放。报告保存各算法配置、交互次数、实际耗时和检查点校验值；短预算对照用于筛选配置，不代表算法的充分训练结果。

评估入口检查环境版本、观测和动作标识、运行源码及参考文件校验值。短预算用于接口与学习流程验证；局部接回次数不替代完整任务成功，多种子鲁棒性仍需单独验收。

### 观测与修正动作的可学习性

[audit_task_learning](audit_task_learning.py) 检查抛球前的采样观测能否预测有效的小幅修正。动作库包含零修正、前后速度 +0.1 m/s、竖直速度 +0.1 m/s、手指展开提前量 −3 ms 及两项速度修正的组合；仅作用于首次抛球，其余事件执行零残差。

```bash
uv run python -m v8_robust_control.audit_task_learning --reference build/v8_task_event_reference.json --jobs 3 --output-dir build/v8_task_learning_audit
```

默认使用 12 个训练、6 个验证及 6 个独立测试种子，初始持球水平偏差按每轴 `[-0.5, 0.5] mm` 均匀采样，同手两球使用相同偏移。每个初态分别执行全部候选动作，记录真实接稳、回收及安全失败；选择器只读取当前策略观测，不读取随机种子或仿真真值。分别拟合完整 306 维观测及 40 维局部特征，后者包含球相对掌心的位置与速度、掌心位置与速度偏差、掌指接触力。模型配置仅按验证集选择，测试集采集前保存并冻结选择器。

复用已有数据并补充示范时，旧训练、验证、测试数据均归入新的训练集，验证与测试使用新种子：

```bash
uv run python -m v8_robust_control.audit_task_learning --reference build/v8_task_event_reference.json --training-data build/v8_task_learning_audit --train-seeds 410300 410301 410302 410303 410304 410305 410306 410307 410308 410309 410310 410311 --validation-seeds 410330 410331 410332 410333 410334 410335 410336 410337 --test-seeds 410350 410351 410352 410353 410354 410355 410356 410357 410358 410359 410360 410361 --jobs 3 --output-dir build/v8_task_learning_local
```

输出目录保存实验协议、各数据集、冻结选择器及 `report.json`。报告比较零修正、按训练集及验证集分别校准的固定修正、完整及局部观测选择器、三个打乱训练标签的局部特征对照，并在物理仿真中重放测试动作。固定修正用于区分名义动作校准与状态反馈的收益。

`test_library_coverage` 表示动作库中至少有一项能完成首次接稳与回收的测试初态数，不作为可部署策略的成绩。初始持球偏差属于初态随机化，不等同于位置测量噪声、延迟或丢帧。本检查不代替完整三球任务或 RL 抗扰收益验收。

该前缀仍运行三球开局，接稳与回收前可能已发生后续抛球；后续命令保持名义值。失败既可能来自首抛，也可能来自其它球的运动。有限动作库的失败不能证明整个残差动作空间不可行，选择器未获益也不能据此判定 RL 无效。

### 多事件规划示范

[task_demonstrations](task_demonstrations.py) 在实时抛球、接球决策点记录采样观测，并在隔离的仿真副本中调用原规划算法。规划器与残差环境使用相同的底层保护；实时环境只执行经动作解码器校验的有界参数。规划器拒绝或决定超出动作范围时，记录原因并停止该诊断回合。

```bash
uv run python -m v8_robust_control.task_demonstrations --reference build/v8_task_event_reference.json --seeds 410400 410401 410402 --jobs 3 --output-dir build/v8_joint_demonstrations
```

默认比较零残差、首抛固定修正、仅首抛使用规划器、前四次释放范围内的抛球与接球均使用规划器。目标为首次真实接稳并回收；`--goal` 可增加回收目标，`--control-releases` 设置允许修正的事件范围。任务仍使用 30 次抛接的基础配置及原物理、安全标准，前缀达标不等于完整任务完成。

输出目录包含实验协议、逐回合记录和 `report.json`。示范保留决策前观测、请求、可表达的归一化动作及试算耗时。规划教师使用完整仿真状态进行预测，属于有额外信息与计算成本的参考；其成绩不计作学习策略或 RL 收益。

`--planner-budget-s` 和 `--trial-budget-s` 分别限制单次查询和教师回合的计算时间，默认为 120 s、300 s。超过预算标记为 `TEACHER_RESOURCE_LIMIT`，单独报告，不作为物理失败或动作不可行证据。

教师保留原规划器的重试参数缓存及已实际执行的抛球历史；这些信息只用于隔离副本中的优化，不写入实时球状态。每个完成的回合立即保存，汇总保持请求顺序。

每次查询的 `computation` 保存初始化、参考构造与物理试算耗时、试算步调用数及预算检查阶段。接球候选分别记录参数、捕获诊断、首次禁止接触和接触后的试算步数；另一只手构造的预测轨迹单独计数。尚未完成试算的候选没有捕获诊断，不能据此判为不可行。禁止接触后的步数用于定位计算开销，不代表接触得到豁免。

重放成功示范，并在新种子上评估只读取观测的示范选择器：

```bash
uv run python -m v8_robust_control.task_demonstrations --reference build/v8_task_event_reference.json --evaluate build/v8_joint_demonstrations --seeds 410420 410421 410422 410423 410424 410425 --jobs 3 --output-dir build/v8_joint_students
```

评估只使用已真实回收达标的联合示范，先保存并重新加载事件表，再执行测试。对照包括零残差、首抛固定修正、按事件取示范均值、仅首抛选择示范、各抛接事件按局部观测选择最近示范。后两者只读取当前观测，不调用规划器；训练数据、归一化及事件表均不使用测试种子。记录动作的原场景重放仅验证执行能力，不作为独立策略成绩。该小试不包含 PPO/SAC 训练。

### 连续回收与模仿初始化

采集连续三次真实接稳并回收的示范，允许修正前六次释放范围内的抛接事件：

```bash
uv run python -m v8_robust_control.task_demonstrations --reference build/v8_task_event_reference.json --seeds 410401 410402 410403 410404 410405 410406 --goal 3 --control-releases 6 --modes planner_joint --planner-budget-s 180 --trial-budget-s 600 --jobs 3 --output-dir build/v8_joint_recovery3
```

也可导出原规划器的候选轨迹，再通过相同动作接口验证：

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --native-timeout 2.2 --cases seat_2mm --seeds 410200 --jobs 1 --output build/v8_recorded_candidates.json
uv run python -m v8_robust_control.task_demonstrations --reference build/v8_task_event_reference.json --oracle-input build/v8_recorded_candidates.json --seeds 410200 --jitter-mm 2 --goal 3 --control-releases 6 --modes recorded_joint --jobs 1 --output-dir build/v8_joint_candidate3
```

完整任务参考的准备命令见多球训练部分。候选导出仅覆盖 2.2 s，是否成为示范由当前环境的三次接稳与回收结果决定，原记录的成功标记不参与判定。候选重放使用预先记录的决定，不作为因果策略成绩。

[task_imitation](task_imitation.py) 使用达标示范初始化 PPO 的动作网络。训练示范、闭环验证及测试使用不同初态种子，测试种子不参与训练、归一化或检查点选择：

```bash
uv run python -m v8_robust_control.task_imitation --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --reference build/v8_task_event_reference.json --validation-seeds 410450 410451 --test-seeds 410460 410461 410462 410463 410464 410465 --goal 3 --control-releases 6 --training-seeds 0 1 2 --epochs 2000 --jobs 3 --output-dir build/v8_task_imitation
```

仅使用实际达到回收目标的示范训练，损失按回合等权，输入统计仅由训练集计算；价值网络不使用动作标签训练。保存零初始化及固定训练进度的检查点，按闭环验证回合的安全失败数、目标完成数和已回收飞行数依次选择。检查点保存输入归一化和事件范围，范围外继续执行零残差，可由 PPO 加载并继续训练。

示范可以来自不同初态偏差幅度，记录逐回合保存训练条件；验证与测试的偏差幅度由 `--jitter-mm` 指定，默认为每轴 ±0.5 mm。

输出包含 `protocol.json`、`training.json`、PPO 检查点及 `report.json`。报告分别记录训练动作拟合、原示范重放、float32 动作重放、闭环训练回合及独立测试回合，并比较零残差、首抛固定校准、按事件固定校准和最近示范选择器。训练回放使用各示范的原始偏差幅度。动作误差较小不代表闭环回收成功；模仿初始化不代表已通过 RL 或完整三球任务验收。

默认完整训练预算的初始化文件为 `ppo_bc_seed*_epoch2000.zip`，其训练初态闭环结果位于 `fit_rollouts`。`selection.json` 和 `evaluated` 记录验证集选出的评估检查点及其测试结果，可能回退到零初始化；这些结果不等同于训练末期模型的独立测试成绩。

### 闭环纠正示范

[task_corrections](task_corrections.py) 从完整预算的模仿初始化出发，在新初态上执行前两次策略决策，随后由规划器接管。规划器在各决策时刻的独立物理副本上生成有界标签；记录分别保存策略动作、专家标签和实际执行动作。规划拒绝或计算超时终止该采集回合，不补造标签。

```bash
uv run python -m v8_robust_control.task_corrections --parent-dir build/v8_task_imitation --reference build/v8_task_event_reference.json --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --train-seeds 410500 410501 --validation-seeds 410550 410551 --test-seeds 410560 410561 410562 410563 410564 410565 --epochs 1000 --jobs 2 --output-dir build/v8_task_corrections
```

默认使用 `ppo_bc_seed0_epoch2000.zip`。纠正初态、验证及测试初态与初始化阶段隔离。训练合并原有达标示范和已接受的局部纠正标签，固定训练轮数，保留原模型的输入归一化、价值网络及动作边界。后续回合失败不等于已接受的局部动作无效，但这些标签不作为完整任务成功的证据。

输出包含采集记录、`corrected.zip`、拟合记录及 `report.json`。报告比较纠正前后策略、零残差、首抛固定校准和按事件固定校准；验证及测试均执行固定训练预算的模型。专家接管回合不计入策略成绩。这是监督微调试验，三次接稳回收的结果不代表完整三球任务或抗扰 RL 已通过验收。

可复用纠正标签并启用原示范回归约束，补充从开局由专家执行的训练回合：

```bash
uv run python -m v8_robust_control.task_corrections --parent-dir build/v8_task_imitation --reference build/v8_task_event_reference.json --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --correction-input build/v8_task_corrections --retain-anchors --learner-steps 0 --train-seeds 410700 410701 --validation-seeds 410750 410751 --test-seeds 410760 410761 410762 410763 410764 410765 --jobs 2 --output-dir build/v8_task_retention_coverage
```

`--learner-steps 0` 从初态起执行专家动作；默认值 `2` 先执行两次策略决策，再由专家接管。两类采集方式分别记录，均不计入策略评估成绩。

`--retain-anchors` 冻结网络主体、价值网络及输入统计，以带正则的最小二乘拟合输出层修正，并约束原示范观测上的输出变化；此模式不使用 `--epochs`。固定的四档更新幅度分别进行物理回放，只有保留原模型示范回收能力的候选才参与验证集选择，测试集在选择完成后评估一次。`selection.json` 保存各候选的回放、验证结果及选择依据；回退到零更新时，`corrected.zip` 的动作与原模型相同，不表示学到了新的能力。

已有纠正记录的训练初态可以复用；初始化记录和复用记录中的验证、测试初态不进入训练。这里的扰动为初始持球位置偏差，不包含感知噪声、延迟或丢帧。

若需调整隐藏层，可在同一批标签上进行局部策略微调，无需重新采集：

```bash
uv run python -m v8_robust_control.task_corrections --parent-dir build/v8_task_imitation --reference build/v8_task_event_reference.json --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --correction-input build/v8_task_retention_coverage --retain-anchors --partial-actor --epochs 6000 --validation-seeds 410850 410851 --test-seeds 410860 410861 410862 410863 410864 410865 --jobs 2 --output-dir build/v8_task_partial_actor
```

`--partial-actor` 训练策略最后一层隐藏层和输出层，其余参数保持不变。损失为纠正动作均方误差，加上权重为 100 的原模型动作保持项；学习率从 `3e-4` 余弦下降至 `1e-6`。保存初始、四分之一、二分之一及完整预算的检查点，仍按原示范物理回归和独立验证结果选择，测试不参与选择。该模式必须与 `--retain-anchors` 一起使用；仅复用标签时可省略 `--train-seeds`，新增采集时仍需指定新的训练初态。若选中 `epoch=0`，报告中的 `corrected` 测试项对应原模型，不作为训练后策略的成绩。

按事件固定校准使用与本次微调相同的原示范及局部纠正标签，校准表保存在 `fixed_event.json`。标签拟合与闭环回收分别报告；监督微调不计为 PPO 奖励训练。

局部微调支持以下命令行参数，实际值写入实验协议：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--actor-learning-rate` | `3e-4` | 初始学习率 |
| `--actor-final-learning-rate` | `1e-6` | 余弦学习率调度的终值 |
| `--anchor-weight` | `100` | 原模型动作保持项权重，不替代物理回归筛选 |
| `--epochs` | `1000` | 优化轮数；上述示例使用 `6000` |

### 闭环偏离诊断

在原示范初态上比较原模型与指定检查点，并执行单次动作替换、分量替换、前缀替换和修正幅度对照：

```bash
uv run python -m v8_robust_control.task_corrections diagnose --parent-dir build/v8_task_imitation --candidate-dir build/v8_task_partial_actor --candidate partial_epoch6000.zip --reference build/v8_task_event_reference.json --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --jobs 2 --output-dir build/v8_task_closed_loop_axes
```

记录包含首次有效命令差异、测量差异、原生物理事件和各对照的终止原因。事件标识或仿真时刻不一致后停止逐项配对，避免将不同阶段误作同一时刻的误差。物理事件仅用于诊断，不进入策略观测；全部对照经正常动作接口执行，不修改运行中的球状态或外力。诊断只使用训练初态，不作为策略收益或独立测试验收。

### 开局标定与分阶段权限

[calibrate_opening](calibrate_opening.py) 比较首抛参数的局部候选，并测量小速度扰动下的接稳、安全和空中球面净间隙。局部搜索保持原准备时长，首抛前向及竖直速度偏置各取 −10/0/+10 mm/s；验证和测试再分别施加各速度轴 ±1 mm/s 的探测扰动。

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --phase-policy-dir build/v8_task_partial_actor --parent-dir build/v8_task_imitation --local-velocity-grid --train-seeds 411200 411201 --validation-seeds 411220 411221 --test-seeds 411240 411241 411242 411243 --jobs 3 --output-dir build/v8_opening_local_calibration
```

初筛、验证和扰动测试以三次真实回收为目标，另检查无扰动完整 30 次任务。初筛保留无扰动三次回收成功的候选，再按验证集安全和完成情况选择，测试不参与选择。省略 `--local-velocity-grid` 时比较额外 0/100/200 ms 准备时间及固定首抛修正。输出包含逐试验记录、选择依据和 `report.json`；选出候选不等于通过独立验收。

名义标定沿用 `--reference` 的估计模式并直接执行固定命令，仅关闭零创新残差门控；预览与普通标定使用相同估计模式。已有策略保留原训练参考和门控：前两次释放和所有抛球事件不开放修正，仅在后续接球事件开放位置修正，默认每轴上限 2 mm。该限制通过 `--reception-cap-mm` 设置，始终受原 60 mm 边界约束。此对照评估的是旧策略经过新权限映射后的行为，不表示已在新接口上完成 RL 训练。

净间隙只统计已确认离手且未接触任一手掌的球，使用真实物理状态作只读诊断，不进入策略观测或奖励。标定试验不修改球的运行状态、外力、接触模型或安全阈值；其收益与 RL 收益分别报告。

联合标定前三次抛球和早期接球：

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --joint-opening --train-seeds 412000 412001 --validation-seeds 412020 412021 --test-seeds 412040 412041 412042 412043 --jobs 3 --output-dir build/v8_joint_opening_calibration
```

`--joint-opening` 比较前三抛的协调前后速度修正、释放提前量及早期接球位置偏移。所有修正均通过原动作接口执行：第三次释放后恢复原抛球命令，第六次释放后恢复原接球命令。三次回收用于初筛，六次回收用于验证和扰动测试，再检查无扰动完整 30 次任务。验证还包含无扰动六次回收，防止只保留开局表现而破坏后续交接。

余量记录包括空中球面净间隙、手掌及腕部到支架的距离、肘部姿态余量和机器人接触穿透。支架距离按 10 ms 采样，并在 50 mm 截断；接触穿透和原安全检查仍按物理步处理。余量只作诊断及同等完成表现下的排序，不作为额外奖励，也不替代完整任务验收。

<details>
<summary>第三抛高度与对应接球位置标定</summary>

保留前两次名义抛球，仅对第三抛施加 −0.05/−0.10/−0.15 m/s 的竖直速度修正；对应的接球高度偏移为 0/−5/+5 mm，从第四次释放后的接球事件开始应用，第六次释放后恢复原命令。首抛速度探测扰动仍按测试协议施加。

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --retained-height-grid --train-seeds 413000 413001 --validation-seeds 413020 413021 --test-seeds 413040 413041 413042 413043 --jobs 3 --output-dir build/v8_retained_height_calibration
```

本模式沿用三次回收初筛、六次回收验证和测试，以及无扰动完整 30 次检查；球高和碰撞阈值保持不变。标定与 RL 训练分别进行。

</details>

<details>
<summary>接球轨迹的可达性与避碰诊断</summary>

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --reception-planner project --third-launch-dvz -0.2 --train-seeds 414000 414001 --validation-seeds 414020 414021 --test-seeds 414040 414041 414042 414043 --jobs 3 --output-dir build/v8_reception_constraint_calibration
```

`--reception-planner audit` 只预览原接球轨迹；`project` 在预览不通过时，尝试原动作边界内的位置和掌面法向修正。默认 `off` 保持原控制。`--third-launch-dvz` 可另外比较第三抛的竖直速度修正，不改变前两抛。

预览在独立运动学数据中检查接近及未接住时的缓冲轨迹，包括腕角、肘部、躯干间距和机器人碰撞。其他关节保持当前状态，仅检查有限轨迹采样点，因此不能证明动态避碰或接球成功；实际接稳后切换的回收轨迹仍由物理回合验收。未找到可行修正时执行原轨迹，并记录 `fallback`，不将预览失败计为任务完成。

逐回合的 `reception_plans` 保存候选、约束余量和回退情况。标定仍要求保留无扰动完整任务能力，在独立扰动测试中增加完成次数且安全不退化。该诊断不训练策略，结果不表示 RL 收益。

</details>

<details>
<summary>漏接缓冲制动</summary>

`--uncaught-braking` 比较原控制与漏接后的关节制动。预计接球时刻之后，若手尚未接触球、掌心或腕部预计接近支架，则停止跟踪缓冲尾段，通过原执行器限幅内的力矩减速。接近阶段、实际手球接触和持球阶段不启用该制动；默认关闭。漏接仍须按掉球或超时等原规则判定。

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --uncaught-braking --third-launch-dvz -0.2 --train-seeds 415000 415001 --validation-seeds 415020 415021 --test-seeds 415040 415041 --jobs 3 --output-dir build/v8_uncaught_braking_calibration
```

测试按相同初态与命令扰动成对比较开关制动；指定 `--third-launch-dvz` 时另比较第三抛速度修正。所有组合均执行独立扰动测试和无扰动完整任务检查，包含初筛未通过的漏接案例。`report.json` 的 `summary.braking_pairs` 保存成对指标；逐回合记录制动次数及首次触发状态。避免碰撞、增加完成次数和取得 RL 收益是不同指标，不能相互替代。

</details>

### 录制残差策略视频

<details>
<summary>录制选定策略</summary>

单球录制需要完整单球训练的 `training.json` 和检查点；双球、三球分别需要对应的多球训练记录和检查点。

```bash
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/full_transfer/training.json --input-dir build/v8_transfer_ppo --training-seed 0 --seed 0 --case clean --video videos/v8_ppo_single_seed0.mp4 --output v8_robust_control/results/policy_videos/single_seed0.json
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/cascade_training/two.json --input-dir build/v8_two_gain_ppo --training-seed 0 --seed 0 --case clean --video videos/v8_ppo_two_seed0.mp4 --output v8_robust_control/results/policy_videos/two_seed0.json
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/cascade_training/complete30.json --input-dir build/v8_complete30_gain_ppo --training-seed 0 --seed 0 --case clean --video videos/v8_ppo_complete30_seed0.mp4 --output v8_robust_control/results/policy_videos/complete30_seed0.json
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/feedback_training/complete30.json --input-dir build/v8_feedback_complete30_final_ppo --training-seed 0 --seed 510000 --case sensor_only_3mm --video videos/v8_ppo_feedback_noise3_seed0.mp4 --output v8_robust_control/results/policy_videos/feedback_noise3_seed0.json
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
| [feedback_training/](results/feedback_training/) | 接触转换诊断、反馈控制器训练、感知及物理初态扰动的独立评估 |

完整抛接结果按名义参考、单球训练、双球训练、三球训练、独立评估、策略录制的顺序生成。

除显式支持 `--overwrite` 的汇总命令外，输出路径须为新路径。重新生成结果时还需检查以下输出位置：

- `build/v8_reception_refinement/`、`build/v8_reception_sac/`：保留且与源码匹配时，可跳过训练，从归档命令开始。归档命令的同名 `validation.md` 也须移走或改用新报告路径；重新训练须使用新目录，并同步后续输入路径。
- `build/v8_transfer_reference/`、`build/v8_transfer_ppo/`：分别保存完整单球的名义电机参考、训练模型及检查点；准备或训练时使用新目录。
- `build/v8_two_ball_reference/`、`build/v8_complete30_reference/`：双球及三球任务的名义参考；`build/v8_two_gain_ppo/`、`build/v8_complete30_gain_ppo/`：对应的 PPO 检查点。重新准备或训练时使用新目录，并同步后续输入路径。
- `build/v8_feedback_two_final_ppo/`、`build/v8_feedback_complete30_final_ppo/`：反馈控制器的双球、三球 PPO 检查点。
- `videos/`：录制与合成命令均拒绝覆盖同名 MP4。重跑前移走对应视频，或通过评估器的 `--video-dir` 和合成器的 `--output` 指定新位置。

报告可以从原始记录重新生成。来球接稳训练目录可通过 `summarize_residual` 导出训练归档；完整抛接训练的原始记录缺失时，需要重新运行相应训练与评估。视频元数据通过重新录制生成，已有的同名 MP4 须先移走或更换输出路径。单元测试不依赖 `results/`。

复现采用相同任务、配置、种子和验收流程，不保证文件逐字节一致。耗时、并行记录顺序和模型文件校验值可能变化；依赖、硬件及数值计算差异也可能影响训练和接触结果，应以重新运行的验收指标为准。模型兼容性由环境版本、源码和检查点校验值检查，修改观测或控制代码后需重新训练。
