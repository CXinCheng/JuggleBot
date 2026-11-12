# v7 — 球观测与状态估计

为 G1 上身与双 LEAP Hand 三球抛接仿真提供球观测、状态估计和基线评估。物理模型与规划初值复用 [v6](../v6_humanoid_hands/README.md)，抛接通过手与球的物理接触实现。

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
| `--observation-max-age` | 0.05 s | 测量有效期，须不小于采样周期 |
| `--output` | 无 | 实验 JSON 的保存路径，须为新文件 |

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

采样模式根据连续三次球位置测量估计速度和加速度，并预测控制时刻的球状态。接球规划使用有效且未过期的估计结果。

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

### 基线结果

采用默认物理参数，无观测噪声、传输延迟或丢帧；采样周期为 1/60 s，测量有效期为 0.05 s。

| 模式及记录 | 完成飞行 | 释放总数 | 掉球 | 球球碰撞 | 仿真时长 |
| --- | --- | --- | --- | --- | --- |
| [truth，30 次](results/truth_baseline30.json) | 30/30 | 32 | 0 | 0 | 10.210 s |
| [sampled，30 次](results/sampled_baseline30.json) | 30/30 | 32 | 0 | 0 | 10.236 s |
| [truth，50 次](results/truth_baseline50.json) | 50/50 | 52 | 0 | 0 | 16.620 s |
| [sampled，50 次](results/sampled_baseline50.json) | 50/50 | 52 | 0 | 0 | 16.671 s |

结果文件包含完成次数、掉球与碰撞计数、运行配置，以及源码和规划初值的 SHA-256 校验值。
