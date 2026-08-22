# 动量信号验证（P1）

本目录完成 ETF 轮动研究的第一步：先在不同指数截面分别验证动量信号是否能预测后续相对收益，再讨论 ETF 载体、资产池、TopN、缓冲、择时、止损和盘中执行。它不复用 `research/pools/src/ap_pools.py` 的过滤后评分，也不把策略回测收益当成因子有效性的证据。

该程序目标平台为 JQ 投资研究，必须同时遵守项目级 [`JQ 目标程序运行时兼容性`](../../docs/reference/joinquant/jq_research_compatibility.md)。脚本保持 Python 3.6 和旧 pandas 兼容，平台入口使用 `from jqdata import *`，并在导入边界恢复可能被 JQ 覆盖的 Python 内置名；本地测试只通过假数据和注入 API 验证确定性逻辑。

## 研究截面

三个截面独立计算 Rank IC、分组和协议门槛，不把行业数量优势与宽基/风格指数混在同一个排名中：

| 截面 | V2 定义 | JQ 数据入口 |
| --- | --- | --- |
| `broad` | 上证综指、深证成指、上证50、沪深300、中证500、中证1000、中证全指、创业板指、国证2000、科创50 | `get_price` 固定代码清单 |
| `industry_sw_l1` | `get_industries('sw_l1')` 返回的全部历史申万一级代码，包括分类调整前后代码；研究期内是否有行情由官方日表决定 | `finance.SW1_DAILY_PRICE`，逐行业查询以避开单次 5000 行限制 |
| `style` | 300成长/价值、300R成长/价值、中证红利、上证红利，以及国证大/中/小盘成长与价值 | `get_price` 固定代码清单 |

行业层不是“当前 31 个行业回填历史”。全部历史一级代码都会进入目录和覆盖审计，已终止或新设行业只在官方日表实际存在的日期参与横截面。风格代码由[中证指数产品资料](https://www.csindex.com.cn/)和[国证风格指数系列](https://www.cnindex.com.cn/zh_indices/cni/style/index.html?index_type=202)核对；实际可用期仍以 JQ `universe_coverage` 为准。

## 冻结协议

| 项目 | P1 口径 |
| --- | --- |
| 研究对象 | 宽基、完整历史申万一级行业、风格三个独立截面；不按结果换池 |
| 形成期 | 10、15、20、25、30、40、60、90 个交易日 |
| 预测期 | 1、3、5、10、20 个交易日 |
| 候选信号 | 区间收益、对数价格回归年化斜率、R²、年化斜率 × R² |
| 主检验单元 | `slope_x_r2 / L=25 / H=5`；其他网格只做稳健性诊断 |
| 主指标 | 每日横截面 Spearman Rank IC；Pearson IC 为辅 |
| 重叠样本 | 日样本使用 Newey–West（滞后 `H-1`），另报每 H 日一次的非重叠样本 |
| 横截面诊断 | 五分组单调性、Top1/3/5 相对当日等权池、R² 条件双排序 |
| 时间切片 | 开发期 2016—2021、验证期 2022—2023、锁定样本外 2024—2026-08-20 |
| 禁止混入 | 动态选池、R² 门槛、分数门槛、择时、止损、持仓延续、交易成本和盘中价格 |

日期 `t` 的信号只使用截至 `close[t]` 的数据，未来 H 日收益定义为 `close[t+H] / close[t] - 1`。`close[t]` 是因子研究的统一参考价，不代表能在该收盘价成交；交易可实现性留到执行层研究。

开发期可以用于提出解释，验证期决定该信号是否值得进入下一步。锁定样本外只用于确认，不应根据其结果回头更换池、窗口或公式。三个截面分别应用同一门槛：开发期和验证期主单元 Rank IC 均为正，验证期相邻窗口中位数、最高减最低分组收益、Top1 相对当日等权池收益也均为正。显著性、年度稳定性及多重检验修正同时报告，但不以单个最优参数点替代上述门槛。

## 在 JQ 研究环境运行

1. 新建研究 Notebook，把 [`p1_jq_signal_validation.py`](p1_jq_signal_validation.py) 全部复制到一个代码单元并执行。
2. 检查文件顶部的 `BROAD_INDEX_UNIVERSE`、`STYLE_INDEX_UNIVERSE`、`DEFAULT_UNIVERSES`、固定日期和参数，不要先看结果再改协议。
3. 运行：

```python
RESULTS = run_p1()
```

默认的 `verbose=True` 会打印阶段进度，例如 `[P1  40% | 00:03:12] broad protocol checks complete`。百分比按已完成的取数与评测阶段计算，时间为本次运行的实际累计耗时；申万一级行业逐代码查询仍会额外打印 `sw_l1 prices: 10/N`。如需安静运行，可使用 `run_p1(verbose=False)`。

关键结论会打印在 Notebook 中；完整结果保存在 `RESULTS`：

| 表名 | 内容 |
| --- | --- |
| `runtime_environment` | 平台 Python/pandas/numpy、JQ 导入状态、运行时间、源码路径和已加载函数逻辑 hash |
| `protocol`、`universe_coverage` | 各截面冻结口径、完整成员目录和实际覆盖期 |
| `ic_summary`、`ic_daily` | 全信号/窗口/期限的 IC 汇总；日明细只保留冻结主信号与主窗口 |
| `parameter_plateau` | 当前窗口及相邻窗口的 IC 中位数、离散度和平台分数 |
| `group_returns`、`group_diagnostics` | 主信号五分组收益、单调性和多空组差 |
| `topk_summary` | Top1/3/5 与同日指数池等权收益的比较 |
| `r2_double_sort`、`r2_quality_spreads` | 在高/低动量组内检验高 R² 是否提供增量信息 |
| `yearly_primary_ic` | 主信号逐年稳定性 |
| `protocol_checks` | 预先冻结的开发/验证门槛及只作描述的样本外结果 |

如需下载全部表格，可运行：

```python
FILES = export_results(RESULTS)
```

第一次平台运行先把日期和参数缩成约一个月的完整冒烟，保存 `runtime_environment`、三个截面的 `universe_coverage`、异常 traceback 和输出结构；随后重启内核、恢复冻结配置再跑正式区间。若某个固定指数代码没有研究期行情，或历史申万目录成员没有预期日表记录，覆盖表会明确显示；不要静默用表现好的指数替换。任何清单修订都要升级 universe version。

## 本地验证边界

本地测试使用三个合成截面验证窗口、无前视、IC、HAC、Python 3.6 禁用项和输出结构，并用假 API 验证完整历史申万目录的取数边界。它不能证明默认指数或 `SW1_DAILY_PRICE` 在 JQ 的真实覆盖，也不能证明信号有效。第一次 JQ 正式运行后应保存 Notebook 版本、源码 SHA-256、运行日期、完整 `RESULTS` CSV 和任何协议偏离。
