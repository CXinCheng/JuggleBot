# v8 感知扰动与控制评估

v8 在 MuJoCo 中运行 G1 上身与双 LEAP Hand 抛接任务，提供球状态估计、感知扰动、完整任务评估和残差策略训练。[Quickstart](QUICKSTART_RL.md) 提供英文运行说明和训练示例。

## 运行

在仓库根目录执行，完整任务的输出请使用新文件：

~~~bash
uv sync --locked
uv run --locked python -m v8_robust_control.juggle --throws 1 --observation-mode sampled
uv run --locked python -m v8_robust_control.complete_task --throws 30 --observation-mode sampled --output build/v8_sampled30.json
~~~

- `juggle` 从两球在手、一球在空中开始。一次完整飞行从确认离手到接稳，初始空中球的接稳不计入。
- `complete_task` 从三球持稳开始，完成指定次数的抛接及三球收尾。完整成功要求通过安全检查，并完成 0.8 秒收尾持稳。抛接次数须为不小于 4 的偶数。

在 `sampled` 模式中，用 `--noise-mm`、`--delay-ms` 和 `--dropout` 分别设置位置噪声标准差、测量延迟和整帧丢失概率。用 `--seed` 固定扰动序列。持球初态偏移属于独立的物理扰动，比较策略时应单独记录。

## 测试

~~~bash
uv run --locked python -m unittest discover -s v8_robust_control -t . -v
~~~

## 结果与复现

- [报告证据索引](../report/evidence/README.md) 提供冻结配置、逐种子结果和重试对照复现命令。
- [实验与接口参考](EXPERIMENT_REFERENCE.md) 收录控制接口、历史评估、训练命令和视频。重新评估时请把新产物写入 `build/`。

各实验的感知估计器、扰动类型和任务阶段可能不同。比较结果时使用相同的任务、种子、预算和安全判据。规划器使用在线物理试算，报告的仿真结果尚未经硬件验证。
