# 全A涨跌停广度 V1

状态：阶段 E 冻结定义并完成首次评测；本定义先于首次评测落盘，评测后未修改。用于检验涨停相对占优但开始退潮能否提示顶部，以及跌停相对占优但开始修复能否提示底部。

## 输入与点时边界

- 输入只使用已验收 `all_a_p1_inputs_v2_20120101_20260814` 中全A点时成分的 JQ 实际 `high_limit/low_limit` 统计，覆盖源数据为 2012-01-04 至 2026-08-14。
- `hit` 表示前复权日内最高/最低价触及同一价格尺度的实际涨跌停价；`close` 表示收盘封在实际涨跌停价。股票池、ST、停牌和价格有效性均使用日期 `t` 当时可知状态，信号在当日收盘后可得。
- `hit_net = limit_up_hit_ratio - limit_down_hit_ratio`，直接使用已由计数和 `valid_count_limit` 验证的 `limit_hit_net_ratio`；`close_net` 同理使用 `limit_close_net_ratio`。
- 缺失日定义为 `valid_count_limit == 0` 且相应计数为零、比例为空。缺失行保留，不作为零值、不进入历史分位的有效分母、不触发；但它仍占最近 250 个交易日窗口中的一个日期位置。输出显式标记 `quality_available = false`。

## 冻结历史分位

每个原始量独立使用严格排除当日的最近最多 250 个交易日，分位分母只包含该窗口内的有效历史日；有效历史少于 120 日时不生成分位：

```text
rank250_t(x) = (
    count(prior x < x_t) + 0.5 * count(prior x == x_t)
) / valid_prior_count
```

- 相等按通过计数一致性校验后保留的输入浮点值精确比较；不以计数重算覆盖输入比例。分位边界允许为 0 和 1。
- 当前行只在计算完当日分位后进入交易日窗口，因此不含任何当日或未来信息；当前值缺失时仍推进窗口，但不进入有效分母。
- 两个分项都可用时，`limit_score = mean(rank250(hit_net), rank250(close_net))`。
- 120 日预热行只参与历史分布，不进入信号 bundle 或评测基线。首个可用 `limit_score` 日是比较覆盖期起点；真实 V2 输入预期为 2012-07-05。

## 冻结触发

```text
change_5d_t = limit_score_t - limit_score_(t-5)

top_trigger_t =
    limit_score_t >= 0.75 and change_5d_t <= -0.10

bottom_trigger_t =
    limit_score_t <= 0.25 and change_5d_t >= 0.10
```

- 5 日变化按输出中的第 5 个交易日位置计算。首 5 个 score 日保留但 `change_available = false`，不触发；任一端 score 缺失时变化也缺失，不插值。
- 顶部表示涨停相对占优仍在自身历史上四分位、但 5 日内至少退潮 10 个分位点；底部是下四分位内至少修复 10 个分位点的对称规则。
- `raw_value` 为 `limit_score`，`valid_count` 为 `valid_count_limit`。输出同时保留触及/封板的上下限数量、比例、净数量、净比例、总强度、两个历史分位和历史样本数，供审计但不派生额外信号。

## Episode 与评测

顶部和底部是两条独立序列，统一使用 [`events.py`](../../signals/events.py)：连续触发日合并为 episode，onset 为首个活跃日，capped confirmation 固定为第 2 个活跃日；单日短段只在观察到退出日时确认，样本尾部不回填。

首次评测固定使用现役 `top_bottom_regions_v2`、onset/确认两种事件日、strict/loose/window 与预测/确认切片，以及 5/10/20 日 OHLC 结果。顶部和底部、区域定位和 OHLC 结果分别报告，并纳入局部与全局 FDR 完整家族；不根据首次成绩修改本定义。

## 结果入口

正式产物见 [`signal bundle`](../../artifacts/signals/limit_up_down_breadth_v1_20120705_20260814/) 和 [`stage_d_v1`](../../artifacts/evaluations/limit_up_down_breadth_v1_20120705_20260814__stage_d_v1/)；跨信号结论见 [`signal_backlog.md`](../signal_backlog.md#首次冻结评测总览)。
