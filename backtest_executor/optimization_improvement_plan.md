# 策略参数优化工具设计方案

在 `backtest_executor/` 下构建一套独立的、YAML 驱动的策略参数优化系统，与旧系统 `backtest_jq/` 并行共存。

## 约束与前提

- 运行环境：JQ 研究环境（Notebook），直接调用 `create_backtest` / `get_backtest`
- JQ 平台不支持 `extras` 传参、不支持并发回测
- 需要支持多个不同参数空间的策略
- 每轮 10-50 组参数，单次回测 1-3 分钟，串行执行

## 系统结构

```text
backtest_executor/
├── config/
│   └── etf_gao.yaml          # 策略配置文件
├── results/                   # 自动生成
│   └── etf_gao/
│       ├── mapper.json        # 策略级全局 MD5→BacktestID 表（所有轮次共享）
│       └── round1_switches_report.csv  # 每轮独立报表
├── executor.py                # 代码注入 + 单任务执行（已完成）
├── analyzer.py                # 结果分析（从旧系统复制并解耦）
└── optimize.py                # 主控入口
```

## 1. YAML 配置格式

每个策略对应一个 YAML 文件。用户通过修改 `rounds` 来驱动多轮迭代寻优。

```yaml
strategy:
  file: "ETFs/ETF_gao_opt.py"
  base_id: "6e21bbaee8cc3f8423def84436a2bf49"

backtest:
  start_day: "2018-01-01"
  end_day: "2026-01-10"
  initial_cash: 100000

analysis:
  sort_by: "Calmar"

# 参数定义（全局）：每个参数声明 var（策略全名）、default（默认值）、values（候选值）
params:
  S:
    var: EXECUTION_SCORE_RANGE
    default: [0.0, 6.0]
    values: [[0.0, 4.0], [0.0, 5.0], [0.0, 6.0], [0.0, 7.0], [0.0, 8.0]]
  ls:
    var: EXECUTION_LOSE_PARAM
    default: [false, 0.95]
    values: [[false, 0.95], [true, 0.95]]
  v:
    var: EXECUTION_VOLUME_PARAM
    default: [false, 5, 0.6]
    values: [[false, 5, 0.6], [true, 5, 0.4], [true, 5, 0.6], [true, 5, 0.8]]
  ma:
    var: EXECUTION_MA_PARAM
    default: [false, 20]
    values: [[false, 20], [true, 20]]
  r:
    var: EXECUTION_R2_PARAM
    default: [false, 0.4]
    values: [[false, 0.4], [true, 0.4], [true, 0.5], [true, 0.6], [true, 0.7]]
  st:
    var: EXECUTION_SHORT_MOMENTUM_PARAM
    default: [false, 10, 0.0]
    values: [[false, 10, 0.0], [true, 10, 0.0]]
  ar:
    var: EXECUTION_ANNUAL_RETURN_PARAM
    default: [false, 1.0]
    values: [[false, 1.0], [true, 1.0]]
  dl:
    var: EXECUTION_DAY_LIMIT_PARAM
    default: [true, 0.95]
    values: [[true, 0.95], [false, 0.95]]

# 多轮搜索计划：YAML 定义所有轮次，每次运行指定一轮
rounds:
  - name: "round1_switches"
    search: [ls, ma, st, ar]     # 笛卡尔积搜索
    fixed:                        # 只写需要覆盖 default 的参数
      S: [0.0, 5.0]              # 其余未列出的参数自动用 default

  - name: "round2_core"
    search: [S, v]
    fixed:
      ls: [true, 0.95]           # ← 来自 round1 最优结果（用户回填）
      ma: [false, 20]
```

### 参数合并优先级

对于每一轮中不在 `search` 里的参数：`fixed 覆盖值 > params.default`

## 2. 主控逻辑 (`optimize.py`)

用户在 Notebook Cell 中调用 `run_round(config_path, round_index)` 执行单轮寻优。

```python
def run_round(config_path, round_index):
    # 1. 加载 YAML 配置
    cfg = yaml.safe_load(open(config_path))
    round_cfg = cfg['rounds'][round_index]

    # 2. 构建固定参数（default + fixed 覆盖）
    all_fixed = {}
    for key, param_def in cfg['params'].items():
        if key not in round_cfg['search']:
            all_fixed[key] = round_cfg.get('fixed', {}).get(key, param_def['default'])

    # 3. 笛卡尔积生成搜索组合
    search_values = [cfg['params'][k]['values'] for k in round_cfg['search']]
    combos = itertools.product(*search_values)

    # 4. 为每个组合构建完整参数 → 转换为 EXECUTION_ 全名
    for combo in combos:
        config = dict(all_fixed)
        for i, key in enumerate(round_cfg['search']):
            config[key] = combo[i]
        exec_params = {cfg['params'][k]['var']: v for k, v in config.items()}

    # 5. 串行执行回测（executor.py 的 BacktestExecutorV3）
    # 6. 输出报表（analyzer.py）
```

### 工作流

1. 编写 `config/strategy.yaml`，定义所有轮次框架
2. 在 Notebook 中运行 `run_round("config/strategy.yaml", 0)`
3. 查看报表，选择种子参数
4. 修改 YAML 中后续轮次的 `fixed` 字段
5. 运行 `run_round("config/strategy.yaml", 1)`
6. 重复直到满意

## 3. 代码注入 (`executor.py`，已完成)

- 使用 AST 解析源码，精确定位并删除所有 `EXECUTION_` 顶层赋值（支持多行）
- 在代码头部注入带标记的参数定义块
- `BacktestExecutorV3` 提供串行执行、状态轮询、即时存盘、断点续跑

### 跨轮次去重机制（方案 X）

`mapper.json` 为**策略级全局文件**，以参数字典的 **MD5 哈希**为主键。同一策略中，只要参数完全相同，无论在哪一轮，都不会重复提交回测。`round` 字段记录该次回测属于哪一轮，供分析时筛选使用。

```json
{
  "metadata": { "strategy": "ETF_gao", "last_updated": "2026-03-11" },
  "runs": {
    "a3f8c2...": {
      "id_name": "S60_ls95",
      "bt_id": "6e21bb...",
      "status": "done",
      "round": "round1_switches",
      "config": { "S": [0.0, 6.0], "ls": [true, 0.95] },
      "metrics": { "Calmar": 2.1, "MaxDD": -0.15 }
    }
  }
}
```

## 4. 结果分析 (`analyzer.py`)

从 `backtest_jq/analyze_simple.py` 复制并解耦，核心改动：

- 删除 `from generate_params import Rules, Baseline` 依赖
- 参数列名由调用方（`optimize.py`）从 YAML 读取后传入
- 参数显示逻辑：布尔开关显示 `✓/-`，其余显示数值

## 5. 统计与分析扩展（*暂不实现*）

- 网络容错与本地缓存：API 请求重试 + 指标本地缓存
- 多维度筛选：结果过滤（如 `--min-calmar 1.5`）、过拟合孤点标注

---

## 决策日志

| # | 决策 | 备选方案 | 选择原因 |
|---|------|---------|---------|
| 1 | 并行共存，不修改旧系统 | 替换 / 渐进增强 | 旧系统仍在使用，零风险 |
| 2 | 运行在 JQ 研究环境 | 本地运行 | 需要直接调用 JQ API |
| 3 | YAML 配置驱动 | JSON / Python 文件 | 可读性好，JQ 环境支持 |
| 4 | 单文件 Orchestrator | 模块拆分 | 10-50 组规模不需要过度工程化 |
| 5 | 所有轮次定义在 YAML，每次运行一轮 | 全自动多轮 | 用户需在轮次间审查并注入种子 |
| 6 | params 带 default 值 | 不带默认值 | 减少 fixed 中的冗余配置 |
| 7 | analyzer.py 复制并解耦 | import 旧系统 | 完全独立，无跨目录依赖 |
| 8 | AST 头部注入 | 行替换 / extras | extras 不可用，行替换不稳定 |
| 9 | 策略级全局 mapper（MD5 哈希去重） | 轮次独立 mapper | 彻底避免跨轮次重复提交同参数回测 |
