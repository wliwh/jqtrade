# 顶底候选信号结论

截至 2026-08-14，区域 V2、阶段 C/D、P1 单信号和三类 ML 实验均已完成回顾性评测。没有任何信号可以独立、稳定地判断精确顶部或底部；现有能力应按“顶部风险、底部修复、底部确认、波动状态、候选排序”使用。

统一约束见 [`top_bottom_region_evaluation_plan.md`](top_bottom_region_evaluation_plan.md)：日期 `t` 的信号只能使用当时可得数据，标签和窗口必须预先冻结，区域定位与信号后 OHLC 分开报告。

## 单信号角色

| 信号 | 当前用途 | 关键限制 | 规格 |
| --- | --- | --- | --- |
| 四行业 Top1 | 高频市场状态基线 | window 召回 94.1%，精确率仅 8.7% | [`four_industry_top1.md`](signals/four_industry_top1.md) |
| 单行业 Top1 | 部分行业的趋势延续研究线索 | 通用顶部假设被反驳 | [`single_industry_top1.md`](signals/single_industry_top1.md) |
| MA 宽度 | MA20 底部定位、MA60 底部确认 | 通常滞后；顶部较弱 | [`组合`](signals/multi_period_ma_breadth.md)、[`分项`](signals/ma_period_breadth_decomposition.md) |
| 宽度—指数背离 | 顶部风险预警 | window 精确率 14.1%，误报多，无 FDR 发现 | [`breadth_price_divergence.md`](signals/breadth_price_divergence.md) |
| 新高—新低广度 | 60 日底部早期修复；120/250 日表示严重度 | 触发后仍可能下探；顶部方向淘汰 | [`组合`](signals/new_high_low_breadth.md)、[`分项`](signals/new_high_low_period_decomposition.md) |
| 涨跌停广度 | “先冲高、后走弱”的顶部风险辅助 | 不是立即顶部；底部方向淘汰 | [`limit_up_down_breadth.md`](signals/limit_up_down_breadth.md) |
| 换手热度 | 波动放大和风险辅助 | 上下振幅都可能扩大，方向性不足 | [`turnover_heat.md`](signals/turnover_heat.md) |

最值得继续验证的结构是：

```text
底部：60日新高—新低修复 → MA20定位 → MA60确认
顶部：宽度—指数背离 → 涨跌停/换手风险分级
```

这两条仍是待预冻结验证的新组合，不能把单项历史最佳结果直接拼成已验证信号。

## ML 实验

| 实验 | 输出与可用角色 | 冻结结论 |
| --- | --- | --- |
| 未来进入 ML V1/V2/V3 | 未来 5/10/20 日进入 strict 峰瓣的概率；适合连续风险/修复评分 | V3 修复长期连续报警，但精确率和 OHLC 证据不足；见 [`ml_training_v1_memo.md`](ml_training_v1_memo.md) |
| 当日 membership ML V1/V2 | 收盘后“当日属于 strict 顶/底峰瓣”的概率；适合状态量和排序 | 顶部校准改善、底部混合；0.50 门槛使报警过稀，不作独立报警器；见 [`V1`](ml_today_probability_v1_results.md)、[`V2`](ml_today_probability_v2_results.md) |
| MA20 候选 episode ML V1 | MA20 onset 命中 strict 或锚点前后 5 日的条件概率；适合候选排序 | 顶部只保留 2 个成功报警、原命中保留率 25%；底部精确率 29.4%→30.0%，不替代 MA20；见 [`结果`](ma20_episode_ml_v1_results.md) |

## 后续顺序

1. 预冻结评测底部“修复→定位→确认”状态机。
2. 复核宽度—指数背离的跨年份顶部风险稳定性。
3. 只把涨跌停和换手用于风险分级，不再扩展对称底部规则。
4. ML 优先作为连续评分或候选排序，等待新增日期积累前瞻证据。

以上均为离线研究结论，不是交易回测或投资建议。
