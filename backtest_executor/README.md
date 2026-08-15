# backtest_executor

`backtest_executor` 是面向 JQ 研究环境的 YAML 驱动参数回测工具，负责参数组合生成、策略源码注入、串行提交、状态轮询、去重记录和结果表分析。

## 当前入口

- `optimize.py`：Grid、Random、List、Sensitivity 四种参数组合生成和回测编排。
- `executor.py`：提取 `EXECUTION_` 参数、生成逻辑 hash、注入参数并维护 `mapper.json`。
- `analyzer.py`：读取 mapper 与 YAML，输出参数和风险指标对比表。
- `config/`：只保留现役七星策略配置。
- `results/`：已有回测事实与分析结果，不在目录整理时改写。
- [`docs/README.md`](docs/README.md)：下一阶段计划和研究文章入口。
- [`archive/README.md`](archive/README.md)：旧系统、旧配置和旧设计。

## JQ Notebook 用法

在仓库根目录可被 import 的前提下：

```python
from backtest_executor import nb_run, nb_analyze

nb_run(
    "backtest_executor/config/etf_7star_opt_dynamic.yaml",
    "round1a_switches",
)

df = nb_analyze(
    "backtest_executor/results/ETF_7star_opt_dynamic/mapper.json",
    "backtest_executor/config/etf_7star_opt_dynamic.yaml",
)
```

JQ 必须提供真实可访问的锚点策略 ID。现有执行器仍使用 `executor.py` 的模块级默认 ID；YAML 中的 `strategy.base_id` 尚未传入执行器。更多已知风险见 [自动化流水线设计的“当前工程基线”](../docs/planning/strategy_research_automation_pipeline_design.md#2-当前工程基线)。

## 本地验证

本地环境可以验证离线逻辑，但不能创建真实 JQ 回测：

```bash
pytest -q tests/test_backtest_executor.py tests/test_get_param_id.py
```

完整工程测试使用 `pytest -q`。如果 `jqdata` 不可用，真实回测入口应明确失败。
