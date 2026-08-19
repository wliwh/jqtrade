# 文档检索索引

先按问题定位目录，再对最小相关范围全文搜索；不要一次性加载整个 `docs/` 或整份 JQ API。

## 检索方法

```bash
rg -l -i '关键词' <目标目录>
rg -n -i -C 3 '关键词' <目标文件或目录>
```

## 按问题定位

| 你要解决的问题 | 先读 | 再搜索 |
| --- | --- | --- |
| JQ 策略语义、定时调度、下单、账户、回测或模拟盘 | [`reference/joinquant/README.md`](reference/joinquant/README.md) | `reference/joinquant/official/strategy-api.md` |
| JQ 股票、ETF、指数、期货、宏观、因子或技术指标数据 | [`reference/joinquant/README.md`](reference/joinquant/README.md) | `reference/joinquant/official/` 对应主题文件 |
| `create_backtest()` / `get_backtest()` 与本项目参数回测衔接 | [`../backtest_executor/README.md`](../backtest_executor/README.md) | `reference/joinquant/official/strategy-api.md`、`backtest_executor/` |
| 当前研究自动化的目标、阶段门和已知风险 | [`planning/strategy_research_automation_pipeline_design.md`](planning/strategy_research_automation_pipeline_design.md) | `backtest_executor/` |
| 现役 ETF 策略、旧策略或策略辅助脚本 | [`../strategies/etf_rotation/README.md`](../strategies/etf_rotation/README.md) | `strategies/etf_rotation/` |
| ETF 资产池、聚类、PCA 和研究产物 | [`../research/pools/README.md`](../research/pools/README.md) | `research/pools/src/`、`research/pools/docs/` |
| 市场宽度与 micro 研究 | [`../research/micro/README.md`](../research/micro/README.md) | `research/micro/src/`、`research/micro/artifacts/` |
| 指数顶底区域、点时信号和两套离线评测 | [`../research/index_turning_points/README.md`](../research/index_turning_points/README.md)、[实施计划](../research/index_turning_points/docs/top_bottom_region_evaluation_plan.md) | `research/index_turning_points/docs/`、`research/index_turning_points/datas/` |
| JQ 历史数据导出 | [`../tools/jq_data_export/README.md`](../tools/jq_data_export/README.md) | `tools/jq_data_export/src/` |
| 策略比较、历史方案或外部项目资料 | [`reports/forum/articles/`](reports/forum/articles/)、[`archive/`](archive/) | 只作追溯，不作当前事实源 |

## 检索顺序

1. 读根 README、本索引和目标模块 README。
2. 在最小范围搜索函数名、字段名或关键词，只打开命中上下文。
3. JQ 平台语义优先查 `official/` 快照，必要时核对官网和实际环境。
4. `archive/` 和历史文章只能解释来历，不能覆盖现行设计、配置或实验事实。

## 文档归位规则

- 项目入口、运行方法和稳定边界：各目录 `README.md`。
- 现行设计与计划：`docs/planning/`。
- 外部平台参考：`docs/reference/`；JQ 首选 `docs/reference/joinquant/official/`。
- 研究文章及其图片：`docs/reports/`。
- 已失效或仅供追溯的材料：`docs/archive/` 或相应模块的 `archive/`。
