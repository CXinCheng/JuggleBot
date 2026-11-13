# v8 — 感知扰动与控制评估

为 G1 上身与双 LEAP Hand 三球抛接仿真提供球观测、状态估计、感知扰动和基线评估。物理模型复用 [v6](../v6_humanoid_hands/README.md)，完整任务的三球开局与收尾复用 [v7](../v7_three_ball_hold/README.md)。抛接通过手与球的物理接触实现。

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

```bash
uv run python -m v8_robust_control.complete_task --throws 30 --observation-mode sampled --output v8_robust_control/results/complete_sampled30.json
```

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

采样模式根据连续三次球位置测量估计速度和加速度，并预测控制时刻的球状态。位置测量依次经过噪声、丢帧和延迟队列后交给估计器。接球规划使用有效且未过期的估计结果。

噪声按球和坐标轴独立采样；丢帧作用于一次采样中的全部球。固定延迟通过 FIFO 队列实现，接收时间为测量实际交付的物理步时刻。固定种子可重放噪声和丢帧序列。状态估计采用未滤波的三点位置差分。

## 控制与适用范围

[ObservedCascade](../src/jugglebot/robustness/controller.py) 根据球状态预测接球位置，并规划抛接动作：

| 环节 | 处理方式 |
| --- | --- |
| 抛球 | 复用已执行的运动命令，在限定范围内调整手指释放时机 |
| 接球 | 规划失败时重新预测，并调整接球轨迹与回腕姿态 |
| 空手回收 | 接球延迟时，提前启动下一次接球的手掌回收 |

动作执行前检查接球稳定性、下一次抛球的准备状态、碰撞和关节姿态。两种观测模式使用相同的控制逻辑。

采样估计用于实时接球及完整任务的收尾预测。持球控制、接触判定、安全检查和规划试算使用仿真真值。

[CompleteCascade](complete_task.py) 复用三球开局与收尾控制，并接入观测和完整任务验收。

## 验证

```bash
uv run python -m unittest discover -s v8_robust_control -t . -v
uv run python -m unittest v6_humanoid_hands.test_evaluation -v
```

测试覆盖时间戳、状态估计、无效数据处理、规划试算隔离、动作规划与实验记录导出。

运行零扰动基线：

```bash
uv run python -m v8_robust_control.verify_baselines --observation-mode truth --output-dir v8_robust_control/results
uv run python -m v8_robust_control.verify_baselines --observation-mode sampled --output-dir v8_robust_control/results
```

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

```bash
uv run python -m v8_robust_control.complete_task --throws 30 --observation-mode truth --output v8_robust_control/results/complete_truth30.json
uv run python -m v8_robust_control.complete_task --throws 50 --observation-mode truth --output v8_robust_control/results/complete_truth50.json
uv run python -m v8_robust_control.complete_task --throws 50 --observation-mode sampled --output v8_robust_control/results/complete_sampled50.json
```

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

```bash
uv run python -m v8_robust_control.evaluate_perception --seeds 0 1 --throws 30 --jobs 3 --output v8_robust_control/results/perception_trials.jsonl --video-cases baseline noise_3mm combined
uv run python -m v8_robust_control.summarize_perception --input v8_robust_control/results/perception_trials.jsonl --output v8_robust_control/results/perception_sensitivity.md --csv-output v8_robust_control/results/perception_trials.csv --overwrite
```

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

```bash
uv run python -m v8_robust_control.evaluate_perception --task complete --seeds 0 1 --throws 30 --jobs 3 --output v8_robust_control/results/complete_perception_trials.jsonl --video-cases baseline noise_3mm combined
uv run python -m v8_robust_control.summarize_perception --input v8_robust_control/results/complete_perception_trials.jsonl --output v8_robust_control/results/complete_perception_sensitivity.md --csv-output v8_robust_control/results/complete_perception_trials.csv --overwrite
```

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

```bash
uv run python -m v8_robust_control.compare_videos --input v8_robust_control/results/perception_trials.jsonl --cases baseline noise_3mm combined --seed 0 --output videos/v8_perception_before_rl.mp4 --metadata-output v8_robust_control/results/video_comparison.json
```

并排视频按正常速度播放，时长与最长片段一致。提前终止的片段定格并显示 `STOPPED - final frame held`，仿真停止时刻保持不变。`--hold-seconds` 可额外设置结尾停留时间，默认为 `0` 秒。

`--cases` 指定单个配置时可生成单独的展示片段。`--output` 指定视频路径，`--metadata-output` 指定结果目录中的 JSON 路径；两者均须为新文件。

完整任务的原始视频命名为 `videos/v8_complete_before_rl_<配置>_seed<种子>.mp4`。生成对应的并排视频：

```bash
uv run python -m v8_robust_control.compare_videos --input v8_robust_control/results/complete_perception_trials.jsonl --cases baseline noise_3mm combined --seed 0 --output videos/v8_complete_perception_before_rl.mp4 --metadata-output v8_robust_control/results/complete_video_comparison.json
```

完整任务演示：[零扰动](../videos/v8_complete_before_rl_baseline_seed0.mp4)、[感知扰动对照](../videos/v8_complete_perception_before_rl.mp4)、[视频信息](results/complete_video_comparison.json)。

本实现支持感知扰动评估。执行器延迟、物理参数随机化和 RL 不在当前实现范围内。

## 规划初值

完整任务默认加载 [complete_seed_plans.json](complete_seed_plans.json)。初值包含手部运动参数，供规划器试算与修正，不用于回放或覆盖球状态。复现现有基线无需重新标定。

修改几何、接触参数或节拍后，可按需重新标定并验证候选初值：

```bash
uv run python -m v8_robust_control.calibrate_complete --throws 50 --output v8_robust_control/results/complete_seed_candidate.json --report v8_robust_control/results/complete_calibration.jsonl
uv run python -m v8_robust_control.complete_task --throws 50 --observation-mode sampled --seed-plans v8_robust_control/results/complete_seed_candidate.json --output v8_robust_control/results/complete_candidate_sampled50.json
```

标定程序仅在完整任务通过后导出初值，JSONL 保留各轮结果；`--input <初值文件>` 可指定起点。
