# 全A换手热度 V1

状态：阶段 E 冻结定义并完成首次评测；定义、方向和阈值先于首次评测落盘，评测后未修改。用于检验全A换手热度仍处历史高位但短期明显退潮时，是否能够提示顶部。只生成顶部序列，不生成对称底部信号。

## 输入与点时边界

- 输入使用已验收 `all_a_p1_inputs_v2_20120101_20260814`。三个主分量为横截面换手率 P50、流通市值加权换手率均值和换手率不低于 10% 的股票占比；JQ `turnover_ratio` 为百分数单位，全部字段在日期 `t` 收盘后可得。
- 每个主分量独立使用此前最多 250 个交易日的历史窗口，严格排除当日；窗口内只让有效历史值进入排序分母，至少需要 120 个有效历史日。缺失日仍占交易日窗口位置，不向更早日期扩窗补足。
- 历史排序先从输入首日 2012-01-04 开始建立；daily bundle 从三个排序分量首次同时可用的 2012-07-05 开始，避免把预热期当作评测基线。
- 5 日变化按 bundle 时间轴的前第 5 个交易日计算。首 5 个 score 日保留，但 `change_available=False`、不可触发；首个具备变化条件的日期为 2012-07-12。

## 冻结公式

对分量 `x`，令 `H_t(x)` 为日期 `t` 之前最多 250 个交易日中 `x` 有效的历史值，`n_t` 为其有效数量：

```text
rank250_t(x) =
  (count(H_t(x) < x_t) + 0.5 * count(H_t(x) == x_t)) / n_t

要求：x_t 有效，n_t >= 120；当日 x_t 不进入 H_t(x)。

turnover_score_t = mean(
    rank250_t(turnover_ratio_pct_p50),
    rank250_t(turnover_ratio_pct_cap_weighted_mean),
    rank250_t(turnover_ge_10pct_ratio)
)

change_5d_t = turnover_score_t - turnover_score_(t-5)

top_trigger_t =
    turnover_score_t >= 0.75 and change_5d_t <= -0.10
```

- midrank 对并列值计半权重；排序只依赖当日之前的数据，不使用全样本分位数。
- `raw_value` 固定为三个历史分位的等权平均 `turnover_score`。
- `valid_count` 为 `turnover_valid_count` 与 `turnover_cap_weight_valid_count` 的较小值；`universe_size` 为点时全A成分数。
- 只要任一主分量当日缺失、历史有效数不足，或 5 日前组合分数缺失，就保留该日期但显式置 `quality_available=False` 或 `change_available=False`，并且不触发。不插值，也不把缺失当作零。

## 审计字段

daily 输出除三个主分量、各自 `rank250` 和历史有效数外，还保留：

- 普通换手率均值和 P25/P50/P75/P90/P95；
- 流通市值加权均值及其有效分母；
- 换手率不低于 5%/10%/20% 的股票计数、比例和统一换手有效分母；
- `turnover_score`、5 日变化、质量与变化可用状态。

普通均值、其他分位数以及 5%/20% 极端占比只用于数据审计，不进入组合，也不生成额外信号，避免扩大检验家族。

## Episode 与评测

序列统一使用 [`events.py`](../../signals/events.py)：连续触发日合并为 episode，onset 为首个活跃日，capped confirmation 固定为第 2 个活跃日；单日短段只在看到退出日时确认，未退出的样本尾部保持 pending，不回填确认日。

首次评测固定使用现役 `top_bottom_regions_v2`、onset/确认两种事件日、strict/loose/window 与预测/确认切片，以及 5/10/20 日 OHLC 结果。区域定位与 OHLC 结果分开报告，并纳入完整的局部和全局 FDR 家族；不根据首次成绩调整本页定义。

## 结果入口

正式产物见 [`signal bundle`](../../artifacts/signals/turnover_heat_v1_20120705_20260814/) 和 [`stage_d_v1`](../../artifacts/evaluations/turnover_heat_v1_20120705_20260814__stage_d_v1/)；跨信号结论见 [`signal_backlog.md`](../signal_backlog.md#首次冻结评测总览)。
