# MA20/60/120 宽度周期拆分 V1

状态：阶段 E 冻结定义；定义先于首次评测落盘，评测后未修改。用于在完全相同的阈值和事件协议下，比较 MA20、MA60、MA120 单周期宽度的独立效力；不改写多周期等权组合 V1。

## 输入与点时边界

- 输入使用 `all_a_p1_inputs_v2_20120101_20260814` 的 `daily_market_features.csv`，实际覆盖 2012-01-04 至 2026-08-14。
- 每个周期的宽度、站上计数和有效分母均直接来自已验收快照；日期 `t` 的信号只读取不晚于 `t` 的行。
- 三个周期从同一日期开始，前 5 个交易日因变化量不可得而保持不触发，不删除。
- 阈值完全继承已冻结的多周期等权组合 V1，不根据任一周期的标签、收益或触发频率分别调整。

## 冻结公式

对 `N ∈ {20, 60, 120}`：

```text
change_N_5d_t = breadth_maN_t - breadth_maN_(t-5)

top_N_t    = breadth_maN_t >= 0.70 and change_N_5d_t <= -0.05
bottom_N_t = breadth_maN_t <= 0.30 and change_N_5d_t >=  0.05
```

- 每个周期输出顶部、底部两条独立序列，共六条；`raw_value` 为该周期宽度。
- `universe_size` 使用点时全A成分数，`valid_count` 使用该周期自身有效分母。
- 输出保留 `ma_window`、宽度、5 日变化、站上计数、有效分母及变化可用状态。
- 信号内样本量预检只用于确认能进入冻结推断：MA20 顶/底 83/83 段，MA60 为 41/77 段，MA120 为 36/55 段；预检没有读取顶底区域或指数未来收益。

## Episode 与比较

六条序列统一使用 [`events.py`](../../signals/events.py)：连续触发日合并为 episode，onset 为首个活跃日，capped confirmation 固定为第 2 个活跃日，单日短段在退出日确认，样本尾部不回填。

区域定位与 5/10/20 日 OHLC 评测使用同一个 Stage D bundle，使六条序列共享一个全局 FDR 家族。比较时分别报告顶部/底部、预测/确认、strict/loose/window 和各指数，不通过调阈值、改起点或删年份提高某个周期成绩。

## 结果入口

正式产物见 [`signal bundle`](../../artifacts/signals/ma_period_breadth_decomposition_v1_20120104_20260814/) 和 [`stage_d_v1`](../../artifacts/evaluations/ma_period_breadth_decomposition_v1_20120104_20260814__stage_d_v1/)；跨信号结论见 [`signal_backlog.md`](../signal_backlog.md#首次冻结评测总览)。
