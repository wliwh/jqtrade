# 市场宽度历史探索

状态：只保留作来源追溯和可视化参考，不是现役策略入口，也不是 `research/index_turning_points/` 的事实数据源。

该目录源于聚宽社区文章[《中证2000搅屎棍指数增强策略》](https://www.joinquant.com/post/56890)，围绕银行、有色金属、钢铁、煤炭的 MA20 行业宽度排名做过探索。代码和产物具有不同口径，使用前必须先核对，不能把 README 中的概念描述当成已验证结论。

## 现有文件

| 文件 | 作用 | 当前边界 |
| --- | --- | --- |
| [`src/jsg_2000.py`](src/jsg_2000.py) | 社区 JQ 策略副本 | 含下单逻辑，只作来源追溯 |
| [`src/market_breadth.py`](src/market_breadth.py) | 计算 MA20 行业宽度 | 探索脚本，不满足严格点时历史口径 |
| [`src/plt_indus_breadth.py`](src/plt_indus_breadth.py) | 国证2000与行业宽度可视化 | 可参考展示方式，不作为信号实现 |
| [`artifacts/data/industries_score.csv`](artifacts/data/industries_score.csv) | 历史宽度导出 | 研究产物，不是通用数据接口 |
| [`artifacts/reports/industries_breadth.html`](artifacts/reports/industries_breadth.html) | 历史交互图 | 研究产物 |

## 已核对的真实口径

### 社区策略副本

`src/jsg_2000.py` 的注释与实现并不完全一致：

- `get_market_breadth()` 实际调用 `get_index_stocks("000300.XSHG")`，使用的是沪深300股票池，不是中证全指或全A；
- 个股强势状态是收盘价高于 MA20，`g.num = 1` 表示只取行业宽度 Top1；
- 风险行业集合除银行、有色金属、煤炭、钢铁外还包含“采掘”；
- 文件含 ETF 买卖和空仓月份逻辑，不能接入只做关系研究的顶底项目。

### 历史宽度脚本

`src/market_breadth.py` 使用 `get_all_securities(date=start_date)` 固定整段窗口的股票集合，并使用 `get_industry(..., date=end_date)` 的单一时点行业归属回看窗口内所有日期。因此它适合近期展示，不满足长期研究所需的逐日历史成分和逐日行业归属要求。

### 历史可视化

`src/plt_indus_breadth.py` 对行业宽度使用 `rank(method="min", ascending=False)`。配置名 `Rank_Threshold = 2` 容易产生歧义，但实际筛选条件是 `rank < 2`，即只选择名次 1；并列第一会同时保留。主图默认使用国证2000代码 `399303`。

## 与当前顶底研究的关系

现行研究边界以 [`../index_turning_points/README.md`](../index_turning_points/README.md) 为准：先建立事后顶底区域，再用严格点时信号分别评价区域定位和信号后结果。四行业 MA20 Top1 的旧单点口径结果已经归档，只作方法追溯。

后续可借用本目录的行业宽度计算思路和图表表达，但必须重新建立点时数据层，不直接读取这里的 CSV 作为正式样本，也不复用任何交易或仓位逻辑。
