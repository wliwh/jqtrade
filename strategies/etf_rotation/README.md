# ETF 策略

本目录根部只放现役 JQ 策略。历史变体已归档，研究辅助脚本与策略代码分开存放。

## 现役策略

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| [`ETF_wy03.py`](ETF_wy03.py) | 经典基线 | 25 日对数价格线性回归，按 `annualized_return × R²` 选择黄金、纳指、创业板、上证180中的一只 |
| [`ETF_7star_opt.py`](ETF_7star_opt.py) | 最新策略 | 固定 ETF 池、集中 `EXECUTION_` 参数、13:10/13:11 分离卖买 |
| [`ETF_7star_opt_dynamic.py`](ETF_7star_opt_dynamic.py) | 最新策略 | 全市场动态池、流动性和指数去重过滤，可由现役 YAML 驱动寻优 |
| [`ETF_7star_175.py`](ETF_7star_175.py) | 最新策略 | QX4.0.1，国内/海外双风险状态、商品与多市场 ETF 池、盘中保护 |

最新三份按文件首次进入 Git 的时间确定：`ETF_7star_opt.py`（2026-03-25）、`ETF_7star_opt_dynamic.py`（2026-04-11）、`ETF_7star_175.py`（2026-06-21）。这不是绩效排名。

## 配套目录

- [`tools/`](tools/README.md)：本地动量指标实验和 JQ 动态池验证脚本。
- [`archive/`](archive/README.md)：2026-08-15 整理前的其余策略与策略研究笔记。
- [`../../backtest_executor/config/`](../../backtest_executor/config/)：现役七星参数优化配置。
- [`../../backtest_executor/results/`](../../backtest_executor/results/)：已保存的 JQ 回测映射和分析结果。

这些策略依赖 JQ API，不能在普通本地 Python 中直接执行。复制到 JQ 前应保留来源说明，并核对回测区间、交易成本、未来数据设置和 ETF 上市时间。
