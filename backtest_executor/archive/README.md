# 旧回测系统说明

这里说明早于当前 `backtest_executor` 工作流的旧版回测程序。

## 目录内容

- `backtest_executor/archive/backtest_jq/`: 早期 JQ Notebook 风格的参数研究脚本。主要依赖硬编码 `Rules` / `Baseline`，通过简单替换 `EXECUTION_` 参数行批量提交聚宽回测。
- `backtest_executor/archive/backtest_engine/`: 更早的回测管理器原型、ETF 池分析工具、
  策略比较文章和部分结果快照。

## 当前入口

新的参数优化工作优先使用顶层 `backtest_executor` 包：

```python
from backtest_executor import nb_run, nb_analyze
```

旧代码主要用于参考、结果复现和迁移旧分析逻辑，不作为当前主执行路径。

## 已被当前框架覆盖的功能

当前 `backtest_executor` 已经覆盖了旧系统中最核心的参数优化主流程：

- 参数组合生成：旧版 `generate_stage_1` / `generate_stage_2` 的开关测试和网格组合，已由 `optimize.py` 的 `grid`、`random`、`list`、`sensitivity` 多种轮次配置替代。
- 参数注入：旧版逐行替换 `EXECUTION_` 变量，已由 `executor.py` 的 AST 参数提取和头部注入替代，能更好处理多行参数和默认值归一化。
- 重复任务规避：旧版按名称或任务 hash 保存回测 ID，当前框架通过 `mapper.json` 记录完整参数上下文、回测设置和策略逻辑 hash。
- 参数结果对比：旧版 `analyze_simple.py` 的参数指标表，已迁移为 `backtest_executor/analyzer.py`，并改为从 YAML 参数定义读取列含义。

## 当前框架尚未覆盖的旧功能

这些功能仍主要存在于旧系统里，后续如果要增强 `backtest_executor`，优先考虑从这里迁移。

### 1. 完整回测分析视图

相关文件：

- `backtest_executor/archive/backtest_jq/analyze_backtest.py`
- `backtest_executor/archive/backtest_engine/backtest_analyse.py`

当前 `backtest_executor/analyzer.py` 偏向“参数组合结果表”：读取 `mapper.json`，展示收益、年化、回撤、Calmar、Sharpe、胜率、换手和年度收益列。旧版 `BacktestAnalyzer` 还包含更完整的回测分析视图：

- `compare_results()`: 按 registry 批量拉取多个回测的完整风险指标，并转置成便于横向比较的表格。
- `plot_curves()`: 绘制多个策略/参数回测的收益曲线，支持对数坐标、起止日期过滤和净值归一化。
- `show_monthly_returns()`: 展示月度收益和年度收益，支持单策略模式、横向对比模式、表格和图形输出。
- `_plot_monthly_heatmap()`: 月度收益热力图，比当前年度收益列更适合观察策略季节性、连续亏损区间和年份内收益分布。

如果迁移，建议做成 `backtest_executor/analyzer.py` 的扩展入口，例如 `nb_report()` 或 `BacktestReport`，输入仍优先使用当前 `mapper.json`，不要恢复旧版 registry 格式。

### 2. ETF 池相对表现与持仓归因

相关文件：

- `backtest_executor/archive/backtest_jq/analyze_backtest.py`
- `backtest_executor/archive/backtest_engine/evaluate_pool.py`

旧版 `PoolEvaluator` 是当前框架没有的能力。它不只是比较参数优劣，而是回答“策略为什么赚钱或亏钱”：

- `evaluate_rolling_returns()`: 比较策略在滚动窗口内的收益分位，判断策略相对 ETF 池是持续跑赢还是只在少数阶段贡献收益。
- `plot_rolling_returns()`: 将策略滚动收益与池内高/中/低分位收益画在一起，直观看出策略选择能力。
- `evaluate_holding_attribution()`: 按每段持仓周期计算持仓标的收益、池内排名、相对池均值和相对分位表现。
- `evaluate_switching_effect()`: 分析每次换仓后新标的相对旧标的的超额收益，区分含换仓日和 T+1 后效果。
- `plot_relative_strength()`: 绘制策略相对 ETF 池等权指数的强弱曲线，用于判断策略 Alpha 是否稳定。

这部分对 ETF 轮动策略很关键。当前 `backtest_executor` 只记录回测级指标，无法解释持仓选择、换仓质量和相对池子的 Alpha 来源。

### 3. 本地缓存与非阻塞任务管理

相关文件：

- `backtest_executor/archive/backtest_engine/Test_ETF.py`

旧版 `Test_ETF.py` 有一套本地缓存和活跃任务记录：

- `ActiveTaskManager`: 将运行中的回测任务落到 `active_tasks.json`，支持下次继续检查。
- `run_strategy_backtest(..., block=False)`: 支持提交后立即返回，后续再轮询任务状态。
- `save_cache()`: 将风险指标、日度结果和月度指标保存到本地 JSON，减少重复访问 JQ API。

当前 `BacktestExecutorV3` 是串行阻塞轮询模型，适合 JQ 不支持并发的约束，但对长任务恢复、断线继续分析、本地离线分析支持较弱。若 JQ 环境不稳定，可以考虑迁移“任务缓存和结果缓存”，但不建议恢复旧版并发提交逻辑。

### 4. 多策略 registry 视角

相关文件：

- `backtest_executor/archive/backtest_engine/Simple_Backtest_Manager.py`
- `backtest_executor/archive/backtest_engine/backtest_analyse.py`

旧版 `SimpleBacktestManager` 面向多个策略文件，例如 `wy03`、`long`、`yj15`，通过 registry 统一记录策略名、测试名、参数、区间和状态。当前 `backtest_executor` 更偏向“单策略一个 YAML 配置，一个策略级 mapper”。

如果需要跨策略横向比较，可以从旧版 registry 思路迁移一个轻量索引层，但要避免破坏当前策略级 `mapper.json` 的清晰边界。

## 迁移优先级建议

1. 先迁移完整分析视图：收益曲线、月度/年度收益表、月度热力图。这部分与当前 `mapper.json` 兼容度最高。
2. 再迁移 `PoolEvaluator`：ETF 池滚动收益、持仓归因、换仓效果、相对强弱。这部分依赖 `get_positions()` 和 `get_price()`，需要在 JQ 环境中验证。
3. 最后考虑本地缓存和非阻塞任务管理。只有当回测耗时、JQ API 不稳定或需要离线复盘时再做。
