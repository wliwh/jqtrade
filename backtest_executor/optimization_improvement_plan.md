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
│   └── etf_gao.yaml          # 策略配置文件（已创建示例）
├── results/                   # 自动生成
│   └── etf_gao/
│       └── mapper.json        # 策略级全局 MD5→BacktestID 表（含逻辑哈希校验）
├── executor.py                # 执行引擎：代码注入 + 安全校验 (已完成，兼容 Py3.6)
├── analyzer.py                # 结果分析：报表生成 (待办)
└── optimize.py                # 主控调度：多模式参数生成 (已完成)
```

## 1. YAML 配置格式 (已扩展)

每个策略对应一个 YAML 文件。支持多种搜索方法：

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
  - name: "round1_grid"
    method: "grid"           # 网格搜索（笛卡尔积）
    search: [ls, ma]
    fixed:
      S: [0.0, 5.0]
  - name: "round2_random"
    method: "random"         # 随机搜索
    count: 20
    search: [S, v]
    fixed:
      ls: [true, 0.95]
      ma: [false, 20]
      st: [true, 10, 0.0]
  - name: "round3_sens"
    method: "sensitivity"    # 灵敏度分析（控制变量）
    search: [r]
    base: { S: [0, 5], ls: [true, 0.95] }
  - name: "round4_list"
    method: "list"           # 手动列表
    combinations:
      - { S: [0, 6], v: [true, 5, 0.4] }
```

## 2. 主控逻辑 (`optimize.py` 已完成)

支持多级参数合并：`params.default` < `round.fixed` < `generator.combo`。
自动生成简短的任务 ID 用于 JQ 后台识别（如 `S05_lsT.95`）。

## 3. 代码注入与安全校验 (`executor.py` 已完成)

- **Python 3.6 兼容性**：
  - 使用 `ast.dump(tree, include_attributes=False)` 替代 `ast.unparse` 计算逻辑哈希。
  - 通过节点 `lineno` 差值估算删除范围，支持多行参数定义剔除。
- **逻辑哈希校验 (Logical Hash)**：
  - 自动忽略注释、空行、缩进及 `EXECUTION_` 参数初值的变动。
  - 仅在策略核心逻辑改变时触发报警，确保历史回测结果的可比性。
- **自动指标抓取**：
  - 回测完成后自动获取 `annual_return`, `max_drawdown`, `sharpe`, `calmar` 并存入 `mapper.json`。

### 跨轮次去重机制（方案 X）

`mapper.json` 为**策略级全局文件**，以任务名称（包含参数缩写）为主键。通过 `metadata` 中的逻辑哈希确保策略身份唯一。

```json
{
  "metadata": {
    "strategy_path": "ETFs/ETF_gao_opt.py",
    "strategy_logic_hash": "d0cfdab3f0b5c82ca1daa9e07f04c2a6",
    "python_version": "3.6_compat",
    "last_updated": "2026-03-12 15:30:00"
  },
  "runs": {
    "round1_S05_lsT.95": {
      "bt_id": "6e21bb...",
      "status": "done",
      "params": {
        "EXECUTION_SCORE_RANGE": [0.0, 5.0],
        "EXECUTION_LOSE_PARAM": [true, 0.95]
      },
      "metrics": {
        "annual_return": 0.25,
        "max_drawdown": -0.12,
        "sharpe": 1.8,
        "calmar": 2.08
      },
      "timestamp": "2026-03-12 15:35:00"
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
