# 聚宽官方 API 离线参考

本目录是编写、审阅和排查 JQ 目标程序时的**首选查询入口**。凡程序将运行在 JQ 投资研究、回测或模拟盘，必须先遵守 [`jq_research_compatibility.md`](jq_research_compatibility.md)。`official/` 保存了 2026-08-15 从聚宽官网公开 API 正文接口取得的完整快照：可读、可全文检索的 Markdown 位于该目录根部；同名的官方 HTML 原始片段位于 `official/source/`，用于核验转换内容。

## 查询顺序

1. 先查 [`jq_research_compatibility.md`](jq_research_compatibility.md)，确认 Python、pandas、导入、日期、批次、输出和平台冒烟约束。
2. 策略运行语义、调度、数据获取、下单、账户对象、回测与模拟盘行为：查 [`official/strategy-api.md`](official/strategy-api.md)。
3. 具体数据、指标、因子或优化器：查下表对应模块。
4. 当前项目封装的 `create_backtest()` / `get_backtest()` 用法：查 [`official/strategy-api.md`](official/strategy-api.md) 后，再查 [`../../../backtest_executor/README.md`](../../../backtest_executor/README.md)。
5. 如果本地快照与当前 JQ 页面、实际 JQ 运行环境冲突，以官网和实测结果为准；平台 API 会更新，不能把本地快照当作永久事实。

本目录不再保留与官方快照重复的旧 API 摘录；所有 JQ 平台语义以 `official/` 与官网为准。

## 官方模块快照

| 主题 | 离线文档 | 官方页面 |
| --- | --- | --- |
| 策略引擎与策略 API | [`strategy-api.md`](official/strategy-api.md) | [API新](https://www.joinquant.com/help/api/help?name=api) |
| JQData / `jqdatasdk` | [`jqdata-api.md`](official/jqdata-api.md) | [JQData](https://www.joinquant.com/help/api/help?name=JQData) |
| 股票数据 | [`stock-data-api.md`](official/stock-data-api.md) | [Stock](https://www.joinquant.com/help/api/help?name=Stock) |
| 场内基金数据 | [`fund-data-api.md`](official/fund-data-api.md) | [fund](https://www.joinquant.com/help/api/help?name=fund) |
| 场外基金数据 | [`otc-fund-api.md`](official/otc-fund-api.md) | [OTCfund](https://www.joinquant.com/help/api/help?name=OTCfund) |
| 指数与期货 | [`index-data-api.md`](official/index-data-api.md)、[`futures-data-api.md`](official/futures-data-api.md) | [index](https://www.joinquant.com/help/api/help?name=index)、[Future](https://www.joinquant.com/help/api/help?name=Future) |
| 宏观与板块 | [`macro-data-api.md`](official/macro-data-api.md)、[`plate-data-api.md`](official/plate-data-api.md) | [macroData](https://www.joinquant.com/help/api/help?name=macroData)、[plateData](https://www.joinquant.com/help/api/help?name=plateData) |
| 技术分析 | [`technical-analysis-api.md`](official/technical-analysis-api.md) | [technicalanalysis](https://www.joinquant.com/help/api/help?name=technicalanalysis) |
| 因子与 Alpha | [`factor-api.md`](official/factor-api.md)、[`factor-values-api.md`](official/factor-values-api.md)、[`alpha101.md`](official/alpha101.md)、[`alpha191.md`](official/alpha191.md) | [factor](https://www.joinquant.com/help/api/help?name=factor) |
| 投资组合优化 | [`portfolio-optimizer-api.md`](official/portfolio-optimizer-api.md) | [optimizer](https://www.joinquant.com/help/api/help?name=optimizer) |
| 常见问题 | [`faq.md`](official/faq.md) | [faq](https://www.joinquant.com/help/api/help?name=faq) |

## 完整性与更新

本次快照覆盖官网 `getContent?name=` 接口公开返回的 16 个策略与数据相关模块：`api`、`Stock`、`fund`、`index`、`Future`、`macroData`、`plateData`、`technicalanalysis`、`factor`、`factor_values`、`optimizer`、`JQData`、`Alpha101`、`Alpha191`、`faq`、`OTCfund`。

每一模块均保存为：

- `official/<module>.md`：由官方 HTML 正文转换的检索版；
- `official/source/<module>.html`：接口返回的原始 HTML 正文。

更新时只使用聚宽官方页面 `https://www.joinquant.com/help/api/help?name=<module>` 及其 `getContent?name=<module>` 正文接口；先覆盖原始 HTML，再重新生成同名 Markdown，并更新本说明中的获取日期与范围。
