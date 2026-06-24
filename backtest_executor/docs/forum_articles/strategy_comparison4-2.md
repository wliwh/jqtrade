# ETF轮动策略研究(四之二)：把参数回测流程整理成一套小工具

前面几篇文章主要围绕 ETF 轮动策略本身展开：固定池怎么选、动态池是否有效、过滤条件和参数怎么影响结果。实际做下来有一个很明显的感受：策略逻辑只是一部分，**参数回测流程本身也需要工程化**。

原因很简单。ETF 轮动策略的参数通常不少，例如评分区间、成交量过滤、均线过滤、R2 过滤、短期动量、止损条件、池子刷新频率等。每次手动改代码、提交回测、等待结果、记录回测 ID、再整理成表格，操作并不复杂，但很容易重复、漏记，也很难保证多轮实验之间可比。

所以我把这套流程整理成了一个轻量工具：`backtest_executor`。

这篇文章只介绍这套新框架的设计思路和使用方式，不讨论某个具体策略参数是否最优。下一篇会用 `ETFs/ETF_7star_opt_dynamic` 作为实际案例，展示如何通过配置文件在 JQ 平台上完成多轮回测，并分析结果。

## 它解决什么问题

`backtest_executor` 想解决的是这样一条链路：

1. 在策略代码里预留可调参数。
2. 在配置文件里定义参数空间和回测区间。
3. 自动生成本轮需要测试的参数组合。
4. 自动把参数注入策略代码并提交回测。
5. 等待回测结束后记录参数、回测 ID 和主要指标。
6. 最后把所有结果整理成可排序的对比表。

换句话说，它不是一个新的策略，也不是一个新的回测引擎。它只是把“反复改参数跑回测”这件事，从手工流程变成相对标准化的流程。

## 基本思路

工具约定：策略里所有需要从外部调整的参数，都写成 `EXECUTION_` 开头的全局变量。

例如：

```python
EXECUTION_SCORE_THRESHOLD = (0, 5)
EXECUTION_R2_PARAM = (False, 0.4)
EXECUTION_MA_PARAM = (False, 20)
EXECUTION_VOLUME_PARAM = (False, 5, 1.0)
```

策略内部再把这些占位参数收进 `Config` 类，这样做的好处是：策略源码只需要保留一份，参数组合由 YAML 文件负责描述。每次回测前，工具根据 YAML 生成一个具体参数版本，提交到 JQ 平台运行。

## YAML 配置文件

一份配置文件主要分成四块：

```yaml
strategy:
  file: "ETFs/ETF_7star_opt_dynamic"

results:
  mapper_file: "mapper.json"

backtest:
  start_day: "2023-01-01"
  end_day: "2026-03-01"
  initial_cash: 100000

params:
  S:
    var: EXECUTION_SCORE_THRESHOLD
    default: [0, 5]
    values: [[0, 4], [0, 5], [0, 6]]

rounds:
  - name: "round2_fine_tuning"
    method: "random"
    count: 50
    search: [S]
```

`strategy` 指明要测试的策略文件；`backtest` 指明回测区间和初始资金；`params` 是全局参数库；`rounds` 是具体实验计划。

这个结构有一个实际好处：同一个策略可以分多轮实验推进。第一轮先测开关项，第二轮再调阈值，第三轮做敏感度分析。每一轮测什么、固定什么、搜索什么，都写在配置文件里。

## 支持的几种参数搜索方式

目前主要支持四种方式。

### 1. 网格搜索

`grid` 会对指定参数做笛卡尔积组合，适合参数数量不多、希望完整观察组合效果的场景。

```yaml
- name: "round1_switches"
  method: "grid"
  search: [ls, ma, st]
```

这类轮次适合做开关项测试，例如止损是否开启、均线过滤是否开启、短期动量是否开启。

### 2. 随机搜索

`random` 会从候选值中随机抽取若干组，适合参数空间较大、不想一次性跑完所有组合的场景。

```yaml
- name: "round2_random"
  method: "random"
  count: 20
  search: [S, v, r]
```

对 ETF 轮动策略来说，很多参数之间存在相互作用。随机搜索有时比机械地铺满全部组合更省时间。

### 3. 手动列表

`list` 用于指定几组人工挑选的组合，适合在前几轮筛选后做最终验证。

```yaml
- name: "round_final_check"
  method: "list"
  combinations:
    - {S: [0, 5], r: [true, 0.4]}
    - {S: [0, 6], r: [true, 0.5]}
```

### 4. 灵敏度分析

`sensitivity` 用于固定一组基准参数，然后逐个改变某个参数，观察结果是否稳定。

这一步很重要。很多参数组合看起来收益很高，但只要阈值略微变化，收益和回撤就明显恶化，这种组合通常不适合作为最终选择。

## 如何运行

在 Jupyter 或研究环境里，可以直接调用：

```python
from backtest_executor import nb_run

nb_run(
    "backtest_executor/config/etf_7star_opt_dynamic.yaml",
    "round2_fine_tuning"
)
```

工具会读取配置文件，生成参数组合，注入策略代码，提交 JQ 回测，并在本地保存记录。

## 结果如何记录

每个策略默认会生成一个 `mapper.json`，记录该策略所有已跑过的参数组合。保存目录由策略文件名决定，默认形如：

```text
backtest_executor/results/{策略文件名}/mapper.json
```

如果想区分不同实验，也可以在 YAML 里通过 `results.mapper_file` 修改文件名，例如：

```yaml
results:
  mapper_file: "mapper_2023_2026.json"
```

这个记录文件里面主要包含：

- 策略文件路径
- 策略逻辑 hash
- 回测 ID
- 参数完整取值
- 回测状态
- 收益、年化、回撤、Sharpe、Calmar 等指标

这里有一个比较关键的设计：工具会计算“策略逻辑 hash”。它会尽量忽略 `EXECUTION_` 参数初值、注释和空行，只关注策略核心逻辑。

这样做的目的，是避免把不同策略逻辑下的结果混在一起。如果策略核心逻辑改了，旧结果就不应该直接和新结果比较。

## 如何分析结果

回测跑完后，可以用：

```python
from backtest_executor import nb_analyze

df = nb_analyze(
    "backtest_executor/results/ETF_7star_opt_dynamic/mapper.json",
    "backtest_executor/config/etf_7star_opt_dynamic.yaml",
    sort_by="Calmar",
    yearly=True,
    output="jupyter"
)
```

输出结果会把参数和指标放到同一张表里，便于直接排序。常用指标包括：

- 总收益
- 年化收益
- 最大回撤
- Calmar
- Sharpe
- 波动率
- 胜率
- 交易次数
- 平均持仓天数
- 换手率
- 年度收益

我个人比较常用的排序方式是先看 `Calmar`，再看最大回撤和年度收益分布。单纯按年化收益排序，很容易选到过度激进的参数。

## 适合什么场景

这套工具比较适合：

- ETF 轮动策略参数较多，需要多轮测试。
- 同一策略逻辑相对稳定，只是在调整过滤条件和阈值。
- 希望记录每一轮回测参数，方便后续复盘。
- 不只看收益，还要一起比较回撤、Calmar、Sharpe、年度稳定性。

不太适合：

- 每次都在大改策略逻辑。
- 只临时跑一两个参数，不需要记录实验过程。

## 小结

`backtest_executor` 的定位很朴素：它不是为了替代策略研究，而是为了减少参数研究中的重复劳动。

对 ETF 轮动这类参数较多、需要反复验证的策略来说，把参数空间、回测轮次、结果记录和指标对比统一起来，会让研究过程更可控。更重要的是，后面回头复盘时，能清楚知道每一组结果是怎么来的。

下一篇将以 `ETF_7star_opt_dynamic` 为例，展示这套框架在七星动态池策略上的一次实际参数优化过程。
