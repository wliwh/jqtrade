# 两套离线评测

阶段 C 已实现 [`region_matching.py`](region_matching.py) 的一对一区域匹配器；阶段 D 已补齐完整切片、OHLC 结果、统计推断、两份报告和版本化 manifest。

区域匹配器：

- 分 signal series、事件流和指数独立匹配，顶部只对顶部、底部只对底部；
- 固定优先级为核心峰瓣、完整包络、包络外 5/10/20 日窗口，再按锚点距离和稳定 ID 决胜；
- 明细保留 `matched/duplicate_alarm/false_alarm/missed_region`，不会静默丢弃额外 episode 或未命中区域；
- `strict` 只认核心峰瓣，`loose` 认核心或包络，`window` 认全部冻结窗口主匹配；
- lead/lag 使用严格点时信号的 JQ 交易日历。输入信号必须与至少一个指数日历完整对齐，单个 TDX 指数的缺日或伪交易日不会改变距离。

完整评测分为：

1. 顶部/底部区域定位，区分预测、确认、严格峰瓣与宽松包络；
2. 信号后 5/10/20 日价格结果。

阶段 D 现役模块：

- [`region_metrics.py`](region_metrics.py)：预测/确认、单峰/多峰、指数/跨指数和 strict/loose/window 完整切片；
- [`post_event.py`](post_event.py)：从原始 OHLC 计算 terminal/max-up/max-down，以覆盖期内非事件日为基线，输出 Newey–West 标准误、95% 区间和局部/全局 BH-FDR；
- [`reports.py`](reports.py)：分别生成区域定位与信号后结果 Markdown，不产生综合总分；
- [`../pipelines/evaluate_signal.py`](../pipelines/evaluate_signal.py)：校验 signal/ground-truth/TDX 哈希并写入不可覆盖的评测 bundle。

推断至少需要 20 个完整事件和 30 个完整基线日；不足时保留描述统计，显著性与 FDR 留空。跨指数区域汇总的单位是指数×区域/episode 对，不假设指数彼此独立。误报的最近区域关联只用于报表切片，不改变其误报状态。

两份 Markdown 使用固定章节、固定字段顺序和固定主体表格。针对某个 `signal_id/direction/version/event_kind` 测试组的数据缺口、窗口不完整、重复报警、口径敏感、样本门槛或 FDR 注意点，只能追加在报告末尾的“分组发现与注意事项”，不得插入主体或临时改变表格结构。

评测器可以读取未来定义的答案，但输入信号必须严格点时。两套结果不合并为总分，详细协议见 [`../docs/top_bottom_region_evaluation_plan.md`](../docs/top_bottom_region_evaluation_plan.md)。
