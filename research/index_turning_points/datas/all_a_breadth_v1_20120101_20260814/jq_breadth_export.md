# JQ 全A市场宽度导出口径

[`jq_export_breadth.py`](jq_export_breadth.py) 用于复制到聚宽投资研究环境运行。JQ 内完成点时股票池、价格有效性、均线、申万一级行业聚合和四行业 Top1 判定；本地只接收一个 ZIP，不接收逐股行情或逐股行业明细。本页、导出脚本、manifest 和两张数据表共同记录 `all_a_breadth_v1` 数据包，不应脱离版本目录单独解释。

平台 Python、pandas 和导入方式的已知限制见
[`../jq_research_compatibility.md`](../jq_research_compatibility.md)。

该导出服务于信号与指数顶部、底部关系研究，不生成交易动作、仓位或策略收益。

## 固定口径

| 项目 | 定义 |
| --- | --- |
| 默认区间 | `2012-01-01`—`2026-08-14`；JQ 实测 `000985.XSHG` 的历史成分从 2012 年起可用 |
| 股票池 | 每个交易日调用 `get_index_stocks('000985.XSHG', date=t)` 获取中证全指历史成分 |
| 可得时间 | 使用当日收盘价，记录在日期 `t`，实际在 `t` 日收盘后可得 |
| 价格 | 日频前复权收盘价；停牌日价格允许向前填充以维持均线，但停牌股票当日不进分母 |
| ST | `get_extras('is_st')`；ST、`*ST` 和退市整理期当日不进分母 |
| 均线 | `MA20/60/120`，必须分别具有完整的 20/60/120 个市场交易日价格窗口 |
| 行业 | 每日调用 `get_industry(..., date=t)` 的 `sw_l1`；不拿单一时点行业回填历史 |
| 行业排名 | MA20 宽度降序、`method='min'`；有效样本至少 5 只，并列第一全部保留 |
| 四行业 | 银行、有色金属、钢铁、煤炭；按名称去除末尾 `I`/`Ⅰ` 后精确匹配 |
| 历史采掘 | 不自动映射为煤炭；分类切换前煤炭缺席会反映在覆盖字段中 |
| 单位 | 所有宽度均为 `[0, 1]` 比例，不是 0—100 的百分数 |

`000985.XSHG` 已在真实 JQ 研究环境确认可查询，但 2012 年以前没有可用历史成分。脚本发现任一交易日成分为空会直接报错，不会退回当前股票列表、沪深300或其他指数。

## 分母和排除项

全市场某个 MA 窗口的有效股票必须同时满足：

1. 当日属于 `000985.XSHG` 的历史成分；
2. 当日收盘价有效；
3. 停牌状态已知且当日未停牌；
4. ST 状态已知且当日不是 ST、`*ST` 或退市整理期；
5. 具有该 MA 窗口所需的完整前复权收盘价历史。

行业宽度还要求当日存在申万一级行业。缺行业的股票仍可进入全市场宽度，但不能进入行业宽度。每日质量计数是独立诊断项，部分原因可能重叠，不能把各排除项简单相加还原分母。

```text
breadth_ma_n = above_count_ma_n / valid_count_ma_n
```

其中 `above_count_ma_n` 是当日 `close > MA_n` 的有效股票数，严格相等不算站上均线。

## ZIP 内容与详细程度

每次运行只写一个 ZIP：

```text
manifest.json
data/
├── daily_summary.csv
└── industry_breadth.csv
```

### `daily_summary.csv`

一行对应一个交易日，保留可以直接研究的信号量和复核分母所需的质量字段：

| 字段或字段组 | 含义 |
| --- | --- |
| `date` | 信号日期，格式 `YYYY-MM-DD`，收盘后可得 |
| `universe_size` | 当日中证全指历史成分数 |
| `close_missing_count` | 当日缺少有效收盘价的成分数 |
| `paused_count` / `paused_status_missing_count` | 当日停牌数 / 停牌状态缺失数 |
| `st_count` / `st_status_missing_count` | 当日 ST 类数 / ST 状态缺失数 |
| `base_valid_count` | 通过当日价格、停牌和 ST 检查的数量，尚未要求足够的 MA 历史 |
| `base_valid_missing_industry_count` | 基础有效但缺少当日申万一级行业的数量 |
| `insufficient_history_count_ma{n}` | 基础有效但不足以计算对应 MA 的数量 |
| `valid_count_ma{n}` | 对应 MA 宽度分母 |
| `above_count_ma{n}` | 当日收盘价高于对应 MA 的数量 |
| `breadth_ma{n}` | 全市场 MA20/60/120 宽度，范围 `[0, 1]` |
| `ranked_industry_count_ma20` | 进入 MA20 Top1 排名的行业数 |
| `top1_tie_count_ma20` | 并列第一的行业数 |
| `top1_industry_codes_ma20` / `top1_industry_names_ma20` | 所有 Top1 行业，使用 `|` 连接 |
| `target_{id}_*` | 四个目标行业各自的行业代码、实际名称、映射数、有效样本数、宽度、名次和 Top1 状态 |
| `four_industry_present_count` | 当日能映射到的四行业数量，可识别分类切换造成的缺席 |
| `four_industry_top1_triggered` | 四行业中是否至少一个为 Top1 |
| `four_industry_top1_ids` | 当日触发的目标行业稳定 ID，使用 `|` 连接 |

目标行业稳定 ID 为 `bank`、`nonferrous`、`steel`、`coal`。宽度和名次都保留，不能只使用最终布尔触发值。

### `industry_breadth.csv`

一行对应“交易日 × 当日观察到的申万一级行业”，用于复核 Top1 和以后研究一般行业宽度：

| 字段或字段组 | 含义 |
| --- | --- |
| `date` | 交易日 |
| `industry_code` / `industry_name` | JQ 当日返回的申万一级行业代码与名称 |
| `universe_count` | 当日股票池中归入该行业的数量 |
| `base_valid_count` | 通过当日基础有效性检查的行业股票数 |
| `valid_count_ma{n}` / `above_count_ma{n}` | MA20/60/120 的行业分母和站上均线数量 |
| `breadth_ma{n}` | 行业 MA20/60/120 宽度，范围 `[0, 1]` |
| `rank_eligible_ma20` | MA20 有效样本是否达到 5 只并可参与排名 |
| `rank_ma20` / `is_top1_ma20` | 未经四舍五入的宽度排名及 Top1 标记 |
| `target_id` / `is_target_industry` | 是否映射为四个目标行业之一 |

不导出证券代码、逐股收盘价、逐股 MA、逐股 ST 状态或逐股行业映射。这样显著缩小文件，同时保留研究信号、所有行业竞争关系、分母和数据质量。

### `manifest.json`

记录数据版本、实际起止日、可得时间、股票池和 API 口径、复权、均线窗口、排名规则、四行业映射、是否导出逐股数据，以及两张 CSV 的行数、字段、字节数、UTF-8 BOM 编码和 SHA-256。本地接收数据时应先校验 ZIP 和哈希，再进入关系研究。

## 在 JQ 投资研究中运行

1. 将 [`jq_export_breadth.py`](jq_export_breadth.py) 复制到 JQ 投资研究根目录。
2. 可先执行 `get_index_stocks('000985.XSHG', date='2012-01-04')`，复核平台返回非空历史成分。
3. 检查脚本顶部的日期、股票池、均线和最小行业样本数。改变任一口径时同步修改 `DATA_VERSION`，不要覆盖旧版本。
4. 初次建议先把日期改为一个月做冒烟；确认 ZIP 后再恢复完整区间运行。
5. 运行脚本。它按自然年处理以限制价格矩阵内存，但最终只生成一个 ZIP。
6. 下载 ZIP。相同文件名已存在时脚本会拒绝覆盖，需先下载并移走旧文件或修改版本名。

完整历史需要逐日查询历史成分和行业，运行时间主要取决于 JQ API；本地测试不能替代真实 JQ 环境的代码、历史覆盖和资源限制验证。
