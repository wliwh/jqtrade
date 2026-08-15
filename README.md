# jqtrade

`jqtrade` 是一组围绕聚宽（JQ）的 ETF 轮动策略、资产池研究和参数回测工具。策略文件主要在 JQ 回测或研究环境中运行；本地代码负责参数生成、结果分析、数据导出和研究辅助，不构成统一的本地交易应用。

## 当前重点

下一阶段以 [策略研究自动化流水线](docs/planning/strategy_research_automation_pipeline_design.md) 为唯一现行计划，入口见 [backtest_executor 文档](backtest_executor/docs/README.md)。目标是串联候选发现、源码审计、参数化、JQ 冒烟回测、自适应搜索和研究报告生成。

## 目录

| 目录 | 用途 | 状态 |
| --- | --- | --- |
| [`strategies/etf_rotation/`](strategies/etf_rotation/README.md) | 4 个现役 JQ ETF 策略、辅助工具和旧策略归档 | 活跃 |
| [`backtest_executor/`](backtest_executor/README.md) | YAML 驱动的 JQ 参数回测与结果分析 | 活跃 |
| [`research/pools/`](research/pools/README.md) | ETF 池过滤、聚类、族谱和 PCA 研究 | 研究中 |
| [`research/micro/`](research/micro/README.md) | 市场宽度和一致性研究 | 探索中 |
| [`tools/jq_data_export/`](tools/jq_data_export/README.md) | JQ 数据导出和批次续传工具 | 辅助工具 |
| [`docs/`](docs/README.md) | 计划、JQ 参考、研究报告与历史归档 | 文档 |
| `tests/` | 本地可执行的回测框架与导出工具测试 | 测试 |

## 现役 ETF 策略

`strategies/etf_rotation/` 根目录只保留一个经典基线和按 Git 新增时间确定的三份最新策略：

| 文件 | 定位 |
| --- | --- |
| [`ETF_wy03.py`](strategies/etf_rotation/ETF_wy03.py) | 经典 4ETF 动量轮动基线：黄金、纳指、创业板、上证180 |
| [`ETF_7star_opt.py`](strategies/etf_rotation/ETF_7star_opt.py) | 固定池、集中参数化的七星优化版 |
| [`ETF_7star_opt_dynamic.py`](strategies/etf_rotation/ETF_7star_opt_dynamic.py) | 动态 ETF 池七星优化版，已有 YAML 和回测结果 |
| [`ETF_7star_175.py`](strategies/etf_rotation/ETF_7star_175.py) | 2026-06-21 引入的 QX4.0.1 / 七星175策略 |

“最新”只表示仓库演进顺序，不代表收益或稳健性排名。旧策略没有删除，统一位于 [`strategies/etf_rotation/archive/`](strategies/etf_rotation/archive/README.md)。

## 快速验证

本地测试不执行真实 JQ 回测：

```bash
pytest -q
```

真实策略运行和参数回测需要 JQ 研究环境提供 `jqdata`、`create_backtest()` 和 `get_backtest()`；具体用法见 [backtest_executor/README.md](backtest_executor/README.md)。

## 文档规则

- 当前计划只写在 `docs/planning/`，历史计划移入对应 `archive/`。
- JQ API 参考以 `docs/reference/joinquant/` 为入口；论坛文章与配图保留在 `docs/reports/forum/`。
- 归档代码默认只用于追溯和复现，不作为新开发入口。
- 策略表现必须以明确区间、成本口径和实际回测 ID 为依据，不能由文件名或历史文章推断。
