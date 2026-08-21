# 多周期均线宽度 V1

状态：阶段 E 冻结定义；定义先于首次评测落盘，评测后未修改。用于检验全A短、中、长周期宽度处于极端区间后发生同步方向变化，是否能提示顶部或底部。

## 输入与点时边界

- 输入快照：`all_a_p1_inputs_v2_20120101_20260814` 的 `daily_market_features.csv`；日期 `t` 的全A成分、前复权收盘价和均线宽度均在当日收盘后可得。
- 宽度分别为 MA20、MA60、MA120 有效股票中，`close > MA + 1e-12 + 1e-12 × abs(MA)` 的比例；三个周期保留各自有效分母。
- 研究起点固定为快照起点，实际首个交易日为 `2012-01-04`。5 日变化使用当前行与前第 5 个交易日之差，前 5 日只保留连续量，不触发信号。
- 不读取顶底标签、指数未来价格或全样本结果来生成阈值；日期 `t` 只依赖不晚于 `t` 的输入行。

## 冻结公式

```text
composite_t = mean(breadth_ma20_t, breadth_ma60_t, breadth_ma120_t)
change_5d_t = composite_t - composite_(t-5)

top_trigger_t    = composite_t >= 0.70 and change_5d_t <= -0.05
bottom_trigger_t = composite_t <= 0.30 and change_5d_t >=  0.05
```

- `0.70/0.30` 是对称极端区间，`±0.05` 要求 5 个交易日内至少变化 5 个百分点；V1 不使用滚动分位数，不让未来样本改变历史阈值。
- 顶部与底部是两条独立序列，`signal_id` 分别为 `multi_period_ma_breadth_top` 和 `multi_period_ma_breadth_bottom`。
- `raw_value` 为 `composite_t`；输出同时保留三个周期宽度、计数、有效分母、5 日变化与变化是否可得。
- `universe_size` 使用点时全A成分数，`valid_count` 取三个周期有效分母的最小值，作为保守的数据覆盖诊断；它不参与触发。

## Episode 与评测

连续触发日使用统一 [`events.py`](../../signals/events.py) 合并为 episode；onset 是首个触发日，capped confirmation 固定为第 2 个活跃日，单日短段在看到退出日时确认，样本尾部不回填。

顶部/底部分开进行区域定位和事件后 5/10/20 日 OHLC 评测；预测/确认、区域/OHLC 也分别报告，不生成综合总分。阈值、方向、变化窗口和确认规则在首次阶段 D 评测后保持冻结；任何改动必须升级版本。

## 结果入口

正式产物见 [`signal bundle`](../../artifacts/signals/multi_period_ma_breadth_v1_20120104_20260814/) 和 [`stage_d_v1`](../../artifacts/evaluations/multi_period_ma_breadth_v1_20120104_20260814__stage_d_v1/)；跨信号结论见 [`signal_backlog.md`](../signal_backlog.md#首次冻结评测总览)。
