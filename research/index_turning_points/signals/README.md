# 点时信号

`definitions/` 每个模块只实现一条已冻结的因果信号：保存逐日原始量、触发状态和数据质量，不读取 ground truth。规格统一在 [`docs/signals/`](../docs/signals/README.md) 维护。

[`events.py`](events.py) 是统一事件层。`build_signal_events()` 接收含 `date/signal_id/direction/raw_value/triggered/universe_size/valid_count/version` 的逐日表，返回带事件字段的 daily 表和一行一个连续段的 episode 表。

短段只在看到退出日后确认；样本尾部不回填，逐日事件必须满足截断不变性。新增信号流程见 [`AGENTS.md`](../AGENTS.md)。
