# `backtest_executor` 回测框架简介

最近把仓库里的参数回测流程整理成了一套独立工具，它主要解决一个问题：当策略参数比较多、需要反复做多轮回测时，如何把“参数管理、回测提交、结果记录、结果对比”这几件事串起来，减少重复劳动。

## 这个工具做什么

`backtest_executor` 的思路很简单：

- 策略里把可调参数统一写成 `EXECUTION_` 开头的全局变量
- 用一个 YAML 文件定义参数空间和回测轮次
- 工具自动把参数注入策略代码并提交 JQ 回测
- 回测完成后，把参数和结果统一记录下来
- 最后再自动整理成对比表，方便筛选最优参数

## 适合什么场景

它比较适合下面这类策略研究：

- 同一套策略逻辑比较稳定，但参数很多
- 需要做开关测试、随机搜索、灵敏度分析
- 不只看收益，还要一起比较回撤、Sharpe、Calmar、换手、年度收益

像 ETF 轮动这类策略，通常就很适合这种工作流。

## 目录结构

```text
backtest_executor/
├── config/         # 每个策略对应一份 YAML 参数配置
├── executor.py     # 参数注入、回测提交、结果登记
├── optimize.py     # 参数组合生成与轮次调度
├── analyzer.py     # 结果汇总与对比分析
└── results/        # 自动生成的回测记录
```

## 使用方式

### 1. 策略中预留参数

比如：

```python
EXECUTION_SCORE_THRESHOLD = (0, 5)
EXECUTION_R2_PARAM = (True, 0.35)
EXECUTION_MA_PARAM = (True, 20, 60)
```

### 2. 用 YAML 定义参数空间

例如：

```yaml
strategy:
  file: "ETFs/ETF_opt_dynamic.py"

params:
  S:
    var: EXECUTION_SCORE_THRESHOLD
    default: [0, 5]
    values: [[0, 4], [0, 5], [0, 6]]

rounds:
  - name: "round1_switches"
    method: "grid"
    search: [S]
```

目前支持：

- `grid`
- `random`
- `list`
- `sensitivity`

### 3. 在 JQ Notebook 里运行

```python
from backtest_executor import nb_run

nb_run(
    'backtest_executor/config/etf_opt_dynamic.yaml',
    'round2_fine_tuning'
)
```

### 4. 分析结果

```python
from backtest_executor import nb_analyze

df = nb_analyze(
    'backtest_executor/results/ETF_opt_dynamic/mapper.json',
    'backtest_executor/config/etf_opt_dynamic.yaml',
    sort_by='Calmar',
    yearly=True,
    output='jupyter'
)
```

## 输出结果

分析模块会把回测结果整理成统一表格，常用指标包括：

- `Return`
- `Ann.Ret`
- `MaxDD`
- `Calmar`
- `Sharpe`
- `Volatility`
- `WinRate`
- `Trades`
- `AvgHoldDays`
- `Turnover`

如果需要，也可以附带年度收益列，便于观察不同年份下的稳定性。

## 这个工具的几个优点

- 参数、轮次、结果分离，回测流程更清晰
- 同样的参数组合不会重复提交，省时间
- 结果统一保存，方便后续复盘
- 很适合做多轮迭代，而不是一次性试几个参数

## 需要注意的地方

- 主要运行在 JoinQuant 研究环境
- 默认是串行回测，不做并发
- 如果策略核心逻辑变了，旧结果需要重新看待，不能直接混用

## 一句话总结

`backtest_executor` 做的事情，就是把“手工改参数反复回测”这件事，变成一套可以批量执行、自动记录、统一比较的流程。
