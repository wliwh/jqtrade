# 全A新高—新低广度 V1

状态：阶段 E 冻结定义；定义先于首次评测落盘，评测后未修改。用于检验全A新高—新低净广度处于明显单边状态后发生短期反转，是否能够提示顶部或底部。

## 输入与点时边界

- 输入使用已验收 `all_a_p1_inputs_v2_20120101_20260814` 的 60/120/250 日新高、新低计数、比例和各自有效分母；实际覆盖 2012-01-04 至 2026-08-14。
- 日期 `t` 的股票池、前复权日内最高/最低价及历史极值只读取不晚于 `t` 的数据；新高、新低均以包含当日的对应回看窗口判断，信号在当日收盘后可得。
- 三个周期保留各自有效分母；因历史不足、停牌、ST 或价格缺失而不可比较的股票不进入对应分母，不将缺失视为未创新高或未创新低。
- 5 日变化使用当前行与前第 5 个交易日之差。前 5 个交易日保留连续量但不触发，不删除。

## 冻结公式

```text
net_w_t       = new_high_ratio_w_t - new_low_ratio_w_t
composite_t   = mean(net_60_t, net_120_t, net_250_t)
change_5d_t   = composite_t - composite_(t-5)

top_trigger_t    = composite_t >= 0.05 and change_5d_t <= -0.03
bottom_trigger_t = composite_t <= -0.05 and change_5d_t >= 0.03
```

- 初始方向固定为顶部和底部两条独立序列。顶部要求净新高仍占至少 5%，但 5 日内退潮至少 3 个百分点；底部使用完全对称的净新低与修复规则。
- `raw_value` 为三周期净广度等权均值；`universe_size` 使用点时全A成分数，`valid_count` 使用三个周期有效分母的最小值。
- 输出保留三个周期的新高/新低计数、比例、净值、有效分母、组合值、5 日变化和变化可用状态，便于审计。
- 固定比例阈值避免使用全样本分位数生成点时信号。首次评测前仅查看无标签连续量分布和事件样本量：冻结规则约产生顶部/底部各 50 个 episode，足以进入统一阶段 D 评测；该预检未读取区域标签或未来收益。

## Episode 与评测

两条序列统一使用 [`events.py`](../../signals/events.py)：连续触发日合并为 episode，onset 为首个活跃日，capped confirmation 固定为第 2 个活跃日，单日短段在看到退出日时确认，样本尾部不回填。

首次评测固定使用现役 `top_bottom_regions_v2`、onset/确认两种事件日、strict/loose/window 与预测/确认切片，以及 5/10/20 日 OHLC 结果。顶部和底部、区域定位和 OHLC 结果分别报告，不合并总分；局部与全局 FDR 使用完整评测家族。

## 结果入口

正式产物见 [`signal bundle`](../../artifacts/signals/new_high_low_breadth_v1_20120104_20260814/) 和 [`stage_d_v1`](../../artifacts/evaluations/new_high_low_breadth_v1_20120104_20260814__stage_d_v1/)；分项定义见 [`new_high_low_period_decomposition.md`](new_high_low_period_decomposition.md)，跨信号结论见 [`signal_backlog.md`](../signal_backlog.md#首次冻结评测总览)。
