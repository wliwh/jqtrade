# 文档检索索引

这里是人和 Codex 类工具的统一文档导航入口。先按问题定位目录，再对目标文件使用全文搜索；不要把整个 `docs/` 或整份 JQ API 一次性读入上下文。

## 检索方法

```bash
# 先按主题限定目录，再带上下文输出命中位置
rg -n -i -C 3 '关键词' <目标目录>

# 仅列出可能相关的文件
rg -l -i '关键词' <目标目录>
```

例如：

```bash
# JQ 调度、下单、数据与回测 API
rg -n -i -C 3 'run_daily|order_target|get_price|create_backtest' \
  docs/reference/joinquant/official/strategy-api.md

# ETF 策略实现与参数
rg -n -i -C 3 '关键词' strategies/etf_rotation backtest_executor/config

# 当前自动化研究计划
rg -n -i -C 3 '关键词' docs/planning backtest_executor
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
| 指数顶部/底部定义及预警信号关系 | [`../research/index_turning_points/README.md`](../research/index_turning_points/README.md) | `research/index_turning_points/` |
| JQ 历史数据导出 | [`../tools/jq_data_export/README.md`](../tools/jq_data_export/README.md) | `tools/jq_data_export/src/` |
| 策略比较、历史方案或外部项目资料 | [`reports/forum/articles/`](reports/forum/articles/)、[`archive/`](archive/) | 只作追溯，不作当前事实源 |

## 给 Codex 的检索顺序

1. 读取根 `README.md` 和本索引，确定领域与当前入口。
2. 读取该领域的 `README.md`，确认代码、文档和产物边界。
3. 用 `rg -n -i -C 3` 在最小相关范围搜索问题中的函数名、字段名或关键词。
4. 只打开命中片段及其直接上下文；遇到 JQ 平台语义时优先取 `official/` 快照，必要时再核对官网和实际 JQ 环境。
5. 历史文档和 `archive/` 仅用于解释来历，不能覆盖现行设计、配置或实验事实。

## 文档归位规则

- 项目入口、运行方法和稳定边界：各目录 `README.md`。
- 现行设计与计划：`docs/planning/`。
- 外部平台参考：`docs/reference/`；JQ 首选 `docs/reference/joinquant/official/`。
- 研究文章及其图片：`docs/reports/`。
- 已失效或仅供追溯的材料：`docs/archive/` 或相应模块的 `archive/`。
