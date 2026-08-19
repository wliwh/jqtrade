# P1 顶底候选信号

状态：当前只保留第一批信号。顶底区域 V1 已完成；统一事件流和两套评测完成后再逐个实现或重新评价，初始方向只是假设，不代表信号有效或可交易。

区域、事件流和评测输出的统一定义见 [`top_bottom_region_evaluation_plan.md`](top_bottom_region_evaluation_plan.md)；本页只维护候选顺序和各信号的最小输入。

## 共用约束

1. 日期 `t` 的信号只使用当时可得的股票池、行业、价格和状态，不使用未来成分或事后修订值。
2. 先保存连续原始量，再冻结方向、标准化、阈值和 episode 规则；不按单一指数或事后成绩寻优。
3. 每个信号单独评价顶部和底部，不合成总分或组合信号。
4. 区域定位与 5/10/20 日结果分别报告，预测与确认分别报告。

P1 共用 JQ 点时全A数据层。股票池、ST、停牌、缺价、申万行业、均线分母和质量字段以 [`all_a_breadth_v1` 导出口径](../datas/all_a_breadth_v1_20120101_20260814/jq_breadth_export.md)为准；本地通达信文件只有指数 OHLC，不能代替全A横截面。

## 候选清单

| 信号 | 初始方向 | 连续原始量 | 初始触发思路 | 状态 |
| --- | --- | --- | --- | --- |
| 四行业 Top1 | 顶部 | 银行、煤炭、钢铁、有色宽度、名次和并列数 | 任一目标行业成为全市场 MA20 宽度 Top1 | V1 已归档，待新版区域评测 |
| 多周期均线宽度 | 顶部、底部 | `close > MA20/60/120` 的全A比例及变化 | 极高/极低水平和变化率分开研究 | 下一候选 |
| 宽度—指数背离 | 顶部优先 | 指数与 MA20 宽度距阶段高点的差 | 指数近高位而宽度回落 | 待实现 |
| 新高—新低广度 | 顶部、底部 | 60/120/250 日新高、新低比例和净值 | 新高萎缩；新低扩散与衰竭分开 | 待实现 |
| 涨跌停广度 | 顶部、底部 | 实际涨跌停数量、比例、净差和累计值 | 涨停拥挤；跌停扩散与衰竭分开 | 待实现 |
| 换手热度 | 顶部优先 | 全A换手中位数、加权均值和极端占比 | 只用历史数据的滚动分位数 | 待实现 |

### 特征约定

```text
breadth_ma_n(t) = count(close(t) > MA_n(t)) / valid_count_n(t)

index_drawdown_n   = close / rolling_max(close, n) - 1
breadth_drawdown_n = breadth_ma20 / rolling_max(breadth_ma20, n) - 1
divergence_n       = breadth_drawdown_n - index_drawdown_n

high_low_net_n = new_high_ratio_n - new_low_ratio_n
```

- 均线宽度保存原值、5/20 日变化和仅用历史数据计算的滚动分位数；阈值在分布审计后一次性登记。
- 背离优先检查 60/120 日连续特征，不先拼复杂布尔规则。
- 新高/新低需要足够历史；底部的“极端扩散”和“随后衰竭”是两个假设。
- 涨跌停使用当日实际 `high_limit/low_limit`，不能统一回推 5%、10% 或 20%。
- 换手优先用横截面中位数和极端占比，避免少数大市值股票支配。

四行业旧结果位于 [`../artifacts/archive/four_industry_width_v1/`](../artifacts/archive/four_industry_width_v1/README.md)。旧实验使用单点顶底和极值前窗口，只能追溯，不能替代新版区域结论。

## 统一输出与顺序

每个信号至少输出：

| 字段 | 含义 |
| --- | --- |
| `date` / `signal_id` / `direction` | 点时日期、稳定名称、顶部/底部方向 |
| `raw_value` / `triggered` | 连续原始量和冻结规则后的触发状态 |
| `episode_id` / `episode_stage` | 连续区间及 onset、continuation、capped confirmation、exit |
| `universe_size` / `valid_count` / `version` | 数据质量和口径版本 |

实施顺序：区域标签与评测框架 → 多周期均线宽度 → 宽度—指数背离 → 新高—新低广度 → 涨跌停广度 → 换手热度。四行业 Top1 只在框架稳定后作为历史基线重跑。
