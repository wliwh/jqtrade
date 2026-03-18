# 多参数回测比较方案

## 目标

针对同一策略、不同参数组合的回测结果，输出一张综合表格，同时展示参数配置和性能指标，方便直接排序定位最优参数。

---

## 相关程序概览

| 程序 | 功能 |
|------|------|
| `generate_params.py` | 参数生成：定义规则（Rules）和默认值（Baseline），生成开关项和范围项的组合 |
| `backtest_manager.py` | 回测管理：读取策略文件替换参数 → 批量提交回测 → 持久化 name→id 映射 |
| `analyze_simple.py` | 轻量级分析：实现 `compare_params()` 核心逻辑，输出格式化表格（终端/Jupyter/Markdown） |
| `analyze_backtest.py` | 深度分析：提供 `PoolEvaluator`（持仓归因、相对强弱、换仓效果）和 `BacktestAnalyzer`（对比分析、月度收益） |
| `optimize.py` | 主控程序：串联参数生成、批量回测、结果分析为一键流程，支持 Stage1/Stage2/多轮优化模式 |
| `jq_strategy_research.py` | **单文件版**：整合了 generate_params/backtest_manager/analyze_simple 的全部功能，最简单的参数寻优-回测-输出框架 |

---

## 表格结构

每一行是一个参数组合，列分为两部分：**参数列 + 指标列**。

### 示意（横向展开）

```
参数ID               | S    | ls  | ma  | v    | r    | st  | ar  | dl  || Ann.Ret  MaxDD   Calmar  Sharpe  WinRate  Trades  Turnover
S060_ls95_ma_dl      | 6.0  | ✓   | ✓   | -    | -    | -   | -   | ✓   || 18.5%  -12.3%   1.50   1.23   65.0%    28      12.5%
S060_ls95_ma_v06_dl  | 6.0  | ✓   | ✓   | 0.6  | -    | -   | -   | ✓   || 17.2%  -11.8%   1.46   1.18   63.0%    24      11.2%
S060_ls95_ma_v06_r05 | 6.0  | ✓   | ✓   | 0.6  | 0.5  | -   | -   | ✓   || 16.8%  -10.5%   1.60   1.31   61.0%    20      10.8%
S080_ls95_ma_dl      | 8.0  | ✓   | ✓   | -    | -    | -   | -   | ✓   || 15.0%  -14.0%   1.07   1.05   60.0%    32      14.2%
```

---

## 参数列设计

| 参数 | 关闭时显示 | 开启时显示 |
|------|-----------|-----------|
| `S` (评分上限) | 数值（始终显示，如 `6.0`） | — |
| `ls`, `ma`, `st`, `ar`, `dl`（开关） | `-` | `✓` |
| `v` (成交量阈值) | `-` | 数值，如 `0.6` |
| `r` (R²阈值) | `-` | 数值，如 `0.5` |

---

## 指标列设计（共 10 列）

### 核心指标（来自 `get_risk()`）

| 列名 | 指标 | 说明 |
|------|------|------|
| `Return` | 总收益率 | 策略累计收益 |
| `Ann.Ret` | 年化收益率 | 主要收益指标 |
| `MaxDD` | 最大回撤 | 主要风险指标 |
| `Calmar` | 年化收益 / 最大回撤 | **主排序列**，综合风险收益 |
| `Sharpe` | 夏普比率 | 经典风险调整收益 |
| `Volatility` | 波动率 | 策略稳定性 |
| `WinRate` | 胜率 | 交易质量 |
| `Trades` | 总交易次数 | 过滤频率异常的组合 |
| `AvgHoldDays` | 平均持仓天数 | 过滤器严格程度的体现 |
| `Turnover` | 换手率 | 交易成本指标 |

### 池归因指标（来自 `PoolEvaluator`，暂未集成）

| 列名 | 来源方法 | 说明 |
|------|---------|------|
| `HitRate` | `evaluate_holding_attribution` | 持有期间命中最优标的的天数占比 |
| `BeatAvg` | `evaluate_holding_attribution` | 跑赢池子均值的天数占比 |

> **提示：** 以上两项指标暂未在 `compare_params` 中自动获取，如需使用，可对候选参数单独调用 `PoolEvaluator.evaluate_holding_attribution()`

---

## 接口设计

```python
compare_params(
    tasks,               # list of (name, param_dict, backtest_id)
    sort_by='Calmar',    # 排序列
    ascending=False,     # 降序
    yearly=False,        # 是否显示年收益列
)
```

### 输入格式

`tasks` 接受 `[(参数ID, 参数字典, 回测ID)]` 三元组列表，例如：

```python
tasks = [
    ('S060_ls95_ma_dl', {'S': (0.0, 6.0), 'ls': (True, 0.95), ...}, 'abc123'),
    ('S060_ls95_ma_v06_dl', {'S': (0.0, 6.0), 'ls': (True, 0.95), 'v': (True, 5, 0.6), ...}, 'def456'),
]
```

参数字典直接来自 `generate_stage_1/2` 的输出，不需要反解析名称字符串。

### 排序与过滤

- 默认按 `Calmar` 降序排列
- 可按任意列排序
- 可通过 `yearly=True` 显示年收益列 (Y2018, Y2019, ...)

> **注意：** `HitRate`/`BeatAvg` 功能暂未实现（如需使用，可对候选参数单独调用 `PoolEvaluator.evaluate_holding_attribution()`）

---

## 实施建议

1. **核心功能已实现：** `analyze_simple.py` 模块已实现参数对比表格功能，支持：
   - 自动解析参数字典为参数列
   - 获取 `get_risk()` 的 10 项核心指标
   - 按任意列排序（默认 Calmar 降序）
   - 可选显示年收益列
   - 三种输出格式：终端打印 / Jupyter 显示 / Markdown 表格

2. **扩展建议：** 如需 `HitRate`/`BeatAvg` 指标，可对候选参数单独初始化 `PoolEvaluator` 调用 `evaluate_holding_attribution()`
