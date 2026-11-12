# v7 — 感知扰动与控制评估

为 G1 上身与双 LEAP Hand 三球抛接仿真提供球观测、状态估计、感知扰动和基线评估。物理模型与规划初值复用 [v6](../v6_humanoid_hands/README.md)，抛接通过手与球的物理接触实现。

## 运行

在仓库根目录执行：

```bash
uv sync
uv run python -m v7_robust_control.juggle --throws 1 --observation-mode truth
uv run python -m v7_robust_control.juggle --throws 1 --observation-mode sampled
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

一次完整飞行从球离手开始，到确认接稳结束。初始空中球的接球不计入，释放总数可能超过目标飞行数。任务失败返回非零退出码。

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

采样估计用于实时接球预测。持球控制、接触判定、安全检查和规划试算使用仿真真值。

## 验证

```bash
uv run python -m unittest discover -s v7_robust_control -t . -v
uv run python -m unittest v6_humanoid_hands.test_evaluation -v
```

测试覆盖时间戳、状态估计、无效数据处理、规划试算隔离、动作规划与实验记录导出。

运行零扰动基线：

```bash
uv run python -m v7_robust_control.verify_baselines --observation-mode truth --output-dir v7_robust_control/results
uv run python -m v7_robust_control.verify_baselines --observation-mode sampled --output-dir v7_robust_control/results
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

结果文件包含完成次数、掉球与碰撞计数、运行配置，以及对应源码和规划初值的 SHA-256 校验值。感知扰动评估包含独立运行的零扰动对照。

## 感知扰动评估

单次扰动实验：

```bash
uv run python -m v7_robust_control.juggle --observation-mode sampled --throws 30 --noise-mm 3 --delay-ms 20 --dropout 0.1 --seed 0
```

结果显示在终端。需要单独留档时，可添加 `--output <新文件路径>`；批量评估统一保存为 JSONL。

评估位置噪声、延迟、丢帧及组合条件：

```bash
uv run python -m v7_robust_control.evaluate_perception --seeds 0 1 --throws 30 --jobs 3 --output v7_robust_control/results/perception_trials.jsonl --video-cases baseline noise_3mm combined
uv run python -m v7_robust_control.summarize_perception --input v7_robust_control/results/perception_trials.jsonl --output v7_robust_control/results/perception_sensitivity.md --csv-output v7_robust_control/results/perception_trials.csv --overwrite
```

| 扰动 | 测试范围 |
| --- | --- |
| 位置噪声标准差 | 0、1、3、5、10 mm |
| 固定传输延迟 | 0、10、20、40 ms |
| 整帧丢失概率 | 0、5%、15%、30% |
| 组合条件 | 3 mm 噪声、20 ms 延迟、10% 丢帧 |

单项测试每次只改变一种扰动。所有条件使用同一控制器、初始物理状态、采样周期和测量有效期；每种条件运行指定的全部种子。扰动范围用于仿真敏感性分析，未经真实传感器标定。

JSONL 保存实验配置、逐次结果、感知序列校验值和估计误差。汇总程序校验试验完整性、计数和配置一致性后生成报告与 CSV。估计误差仅用于评估，控制器不读取误差统计。中断实验保留已完成的试验，不生成完整报告。

成功要求完成全部目标飞行，并通过接稳、碰撞和姿态检查。规划器未找到可行接球方案时，以 `NO_FEASIBLE_RECEPTION` 终止实验。报告中的完成飞行数、规划失败、实际掉球和实际碰撞分别统计；零掉球不表示任务成功。

评估结果：[退化报告](results/perception_sensitivity.md)、[逐次指标](results/perception_trials.csv)、[原始记录](results/perception_trials.jsonl)。

## 视频

对照演示：[感知扰动对照视频](../videos/v7_perception_before_rl.mp4)。视频标注配置、种子、最终完成次数和停止原因；配置与原始数据关联信息见 [video_comparison.json](results/video_comparison.json)。

`--video-cases` 为选定条件的第一个种子录制原始视频，文件名为 `videos/v7_before_rl_<配置>_seed<种子>.mp4`。单次实验可通过 `--video` 指定视频路径。

原始视频以 60 fps 记录实际执行过程，时长由实验停止时刻决定。规划试算不录入视频；未找到可行接球方案时，实验在执行该方案前终止，因此这类片段可能短于一秒，球尚未落地。

| 原始视频（种子 0） | 仿真停止时刻 | 完成飞行 | 结果 |
| --- | --- | --- | --- |
| [零扰动](../videos/v7_before_rl_baseline_seed0.mp4) | 10.236 s | 30/30 | 完成 |
| [3 mm 位置噪声](../videos/v7_before_rl_noise_3mm_seed0.mp4) | 0.279 s | 0/30 | 接球规划失败 |
| [组合扰动](../videos/v7_before_rl_combined_seed0.mp4) | 0.287 s | 0/30 | 接球规划失败 |

视频的帧数、播放速率和试验结果保存在原始评估 JSONL 中。

生成带配置和结果标注的并排视频：

```bash
uv run python -m v7_robust_control.compare_videos --input v7_robust_control/results/perception_trials.jsonl --cases baseline noise_3mm combined --seed 0 --output videos/v7_perception_before_rl.mp4 --metadata-output v7_robust_control/results/video_comparison.json
```

各片段按正常速度播放，默认在最长片段结束时结束。较早停止的片段在其余片段播放期间定格，并显示 `STOPPED - final frame held`。需要结尾停留时可指定 `--hold-seconds`，默认值为 `0`。定格只延长展示时间，仿真已在标注时刻终止。

`--cases` 指定单个配置时可生成单独的展示片段。`--output` 指定视频路径，`--metadata-output` 指定结果目录中的 JSON 路径；两者均须为新文件。

上述结果每种配置包含两个随机种子。策略性能比较应使用未参与训练或调参的独立测试集，保持配置、种子和镜头一致。

本实现支持感知扰动评估。执行器延迟、物理参数随机化和 RL 不在当前实现范围内。
