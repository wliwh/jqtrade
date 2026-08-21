# 两套离线评测

评测分为区域定位和信号后 5/10/20 日 OHLC，两者不合成总分。完整冻结协议见 [`top_bottom_region_evaluation_plan.md`](../docs/top_bottom_region_evaluation_plan.md)。

## 模块

- [`region_matching.py`](region_matching.py)、[`region_metrics.py`](region_metrics.py)：按信号、事件流和指数完成一对一区域匹配及切片；
- [`post_event.py`](post_event.py)：计算 terminal/max-up/max-down、Newey–West 区间和 BH-FDR；
- [`reports.py`](reports.py)：生成两份固定格式报告；
- [`evaluate_signal.py`](../pipelines/evaluate_signal.py)：校验 signal、ground-truth、TDX 哈希并写入不可覆盖的版本目录。

## 不变量

- 顶部只匹配顶部、底部只匹配底部；优先级为核心峰瓣、包络、冻结窗口、锚点距离。
- 明细保留命中、重复报警、误报和漏检；`strict/loose/window`、预测/确认及指数切片分别报告。
- 至少 20 个完整事件和 30 个基线日才推断；跨指数汇总不视为独立样本。
- 报告主体结构固定，缺口、样本门槛和 FDR 注意点只追加到末尾。
- 评测可读事后答案，但输入信号必须严格点时。
