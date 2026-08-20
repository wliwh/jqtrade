# 点时信号

`definitions/` 中的每个模块只负责一个已冻结定义的因果信号：保存逐日连续原始量、触发状态和数据质量，不读取 `ground_truth/` 或 `artifacts/ground_truth/`。

[`events.py`](events.py) 是阶段 C 的统一事件层。`build_signal_events()` 接收至少包含 `date/signal_id/direction/raw_value/triggered/universe_size/valid_count/version` 的逐日表，返回：

- 保留原始字段的 daily 表，并增加稳定 `episode_id`、`episode_day`、onset、continuation、exit 和 capped confirmation；
- 一行一个连续段的 episode 表，记录起止、活跃日数、退出日、确认日及 `closed/active`、`confirmed/pending` 状态。

不足 `N` 日的短段只在看到退出日时确认；未退出的样本尾部不回填。逐日事件对任意截止日保持截断不变，episode 尾部汇总则如实反映该次运行的当前状态。

新增信号前先按 [`../AGENTS.md`](../AGENTS.md) 在 `docs/signals/` 写定义，并通过截断不变性测试。

现役 Top1 定义为 [`four_industry_top1.py`](definitions/four_industry_top1.py) 和 [`single_industry_top1.py`](definitions/single_industry_top1.py)；口径只在对应的 [`docs/signals/`](../docs/signals/README.md) 规格中维护。
