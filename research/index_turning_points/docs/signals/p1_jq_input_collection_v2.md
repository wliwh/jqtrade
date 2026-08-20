# P1 JQ 共用输入采集 V2

状态：真实 JQ V2 快照已接收并通过[本地数据验收](p1_jq_input_v2_acceptance.md)。[`export_all_a_p1_inputs.py`](../../adapters/jq/export_all_a_p1_inputs.py) 已通过本地测试；[`V1`](../../data/inputs/all_a_p1_inputs/all_a_p1_inputs_v1_20120101_20260814/) 因 MA 浮点相等判定不稳定，只保留追溯。

## 冻结口径

每日股票池为 `000985.XSHG` 历史成分；基础分母排除无有效收盘价、停牌状态未知/停牌、ST 状态未知/ST。所有字段在日期 `t` 收盘后可得，不生成信号阈值、episode 或标签。

| 特征 | 口径 |
| --- | --- |
| MA 宽度 | 前复权 `close - MA20/60/120 > 1e-12 + 1e-12 × abs(MA)`；相等不算站上 |
| 新高新低 | 前复权日内 high/low 达到含当日的 60/120/250 日极值 |
| 涨跌停 | JQ 实际 `high_limit/low_limit`，分别统计盘中触及和收盘封板 |
| 换手 | JQ 百分数单位；均值、P25/50/75/90/95、流通市值加权均值、≥5/10/20% 占比 |
| 行业 | 点时申万一级 MA20/60/120 宽度、Top1 并列及四行业映射 |

V2 仅修订 MA 比较容差，并把绝对/相对容差写入 manifest；同一输出区间不得因额外预热历史改变。各特征使用独立有效分母，只导出聚合表。

## 输出与运行

默认输出 `all_a_p1_inputs_v2_20120101_20260814.zip`：

```text
manifest.json
data/daily_market_features.csv
data/industry_breadth.csv
```

脚本按年处理，预热 250 个交易日；`get_price` 每批 500 只，`get_valuation` 每次不超过 4500 行。

1. 复制脚本到 JQ，先用约一个月区间冒烟；正式改变日期或口径必须升级 `DATA_VERSION`。
2. 检查价格、ST、行业、估值、ZIP，以及 manifest 中 V2 和两个 `ma_comparison_*_tolerance`。
3. 恢复完整区间运行；下载后校验行数、字段、SHA-256、比例范围和 MA 预热不变性。
4. 原样放入 [`data/inputs/all_a_p1_inputs/all_a_p1_inputs_v2_20120101_20260814/`](../../data/inputs/all_a_p1_inputs/all_a_p1_inputs_v2_20120101_20260814/)，不得覆盖 V1；当前快照已按此路径验收。

JQ 兼容限制见 [`../jq_research_compatibility.md`](../jq_research_compatibility.md)。
