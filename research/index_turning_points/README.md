# 指数顶底区域与点时信号研究

状态：阶段 A/B 已完成，`top_bottom_regions_v2` 顶底区域和峰瓣产物已可用；下一步实现统一信号事件流与两套评测。

本项目研究信号能否提前预测或短期确认指数顶部、底部，不开发交易策略、仓位规则或综合择时分数。顶部和底部独立评价，一个信号可以只服务其中一类。

## 研究边界

| 层 | 可否使用未来数据 | 产物 |
| --- | --- | --- |
| 事后事实 | 可以 | 顶底区域、规范极值、W/M 峰瓣、未来 5/10/20 日结果 |
| 点时信号 | 不可以 | 日期 `t` 当时可得的原始值、触发状态和事件日期 |
| 离线评测 | 可以读取事后事实 | 区域定位成绩与信号后结果，不得反向修改信号 |

“评测不能使用未来数据”具体约束的是被评价的信号生成过程。评分器必须读取事后区域和未来结果作为标准答案，但这些数据不能进入信号特征、阈值选择或日期回填。

研究拆成两个交付任务：

1. 用完整指数行情生成顶底区域；保留中级 directional-change 点作为规范锚点，用多个核心峰瓣表达 W 底、M 顶和平台顶底。该任务的 V1 已实现。
2. 建立两套互不合并的评测：区域定位能力，以及信号后 5/10/20 日结果；区域定位再分为极值前预测和极值后短期确认。

详细定义、输出表和验收顺序见 [`docs/top_bottom_region_evaluation_plan.md`](docs/top_bottom_region_evaluation_plan.md)。第一批信号仅见 [`docs/signal_backlog.md`](docs/signal_backlog.md)。

## 当前基线

指数使用自身日 K OHLC，不以 ETF 替代。当前覆盖上证、沪深300、中证500、中证1000、国证2000、微盘股和全A，数据截止 `2026-08-14`；国证2000 `399303` 不是中证2000 `932000` 的等价替代。

现有单点标签以日 K 高低价做 directional change：顶部在后续下跌达到 `delta` 后确认，底部在后续上涨达到 `delta` 后确认。小、中、大基础尺度为 5%、10%、20%，再乘冻结的指数倍率：

| 指数 | 倍率 | 小 | 中级 | 大 |
| --- | ---: | ---: | ---: | ---: |
| 上证 | 0.8 | 4.0% | 8.0% | 16.0% |
| 沪深300 | 0.9 | 4.5% | 9.0% | 18.0% |
| 全A | 1.0 | 5.0% | 10.0% | 20.0% |
| 中证500 | 1.1 | 5.5% | 11.0% | 22.0% |
| 中证1000、国证2000 | 1.2 | 6.0% | 12.0% | 24.0% |
| 微盘股 | 1.3 | 6.5% | 13.0% | 26.0% |

单点标签是区域构造的中间层，不是最终评分对象。区域 V2 的价格带为 `min(中级阈值 × 0.20, 2%)`，同时使用锚点左右 20 个交易日上限、相邻锚点时间中点分区和 10 日峰瓣间隔。各指数实际价格带为 1.6%—2.0%；小级极值只记录为诊断字段，不作为峰瓣准入条件。上证 2021 年三个中级顶部锚点为 `2021-02-18`、`2021-06-02`、`2021-09-14`，确认分别滞后 12、39、88 个交易日；三个区域均已保留，峰瓣数分别为 1、3、2。

## 运行与文件

```bash
# 生成七指数清单、三档单点标签、顶底区域和未来结果
/home/hh01/anaconda3/envs/fin/bin/python -m research.index_turning_points.pipeline

# 生成离线交互图
/home/hh01/anaconda3/envs/fin/bin/python -m research.index_turning_points.visualize

# 当前本地回归测试
/home/hh01/anaconda3/bin/python -m pytest -q \
  tests/test_index_turning_point_*.py \
  tests/test_jq_export_breadth.py
```

| 路径 | 当前职责 |
| --- | --- |
| [`labels.py`](labels.py)、[`pipeline.py`](pipeline.py) | 单点标签、七指数解码和未来结果 |
| [`regions.py`](regions.py) | 冻结区域协议、顶底区域和核心峰瓣生成 |
| [`visualize.py`](visualize.py) | 七指数阶段背景、顶底区域与 W/M 峰瓣离线交互图 |
| [`datas/all_a_breadth_v1_20120101_20260814/`](datas/all_a_breadth_v1_20120101_20260814/jq_breadth_export.md) | 已接收的点时全A宽度数据、manifest、生成脚本和版本口径 |
| [`datas/jq_research_compatibility.md`](datas/jq_research_compatibility.md) | 跨数据包复用的 JQ 研究环境限制 |
| [`artifacts/turning_point_labels.csv`](artifacts/turning_point_labels.csv)、[`artifacts/forward_outcomes.csv`](artifacts/forward_outcomes.csv) | 现有单点标签和事后结果 |
| [`artifacts/regions/top_bottom_regions_v2/`](artifacts/regions/top_bottom_regions_v2/manifest.json) | 当前 975 个顶底区域、1494 个峰瓣及可复现 manifest |
| `artifacts/index_turning_points.html` | 按需载入七指数图表的离线检查页 |
| [`artifacts/archive/four_industry_width_v1/`](artifacts/archive/four_industry_width_v1/README.md) | 旧单点口径的四行业结果，只供追溯 |
| [`analyze_breadth.py`](analyze_breadth.py)、[`analyze_forward_returns.py`](analyze_forward_returns.py) | 四行业 V1 复现工具，不是新版评测入口 |

JQ 导出脚本需要复制到真实 JQ 研究环境运行；本地只能验证确定性逻辑和数据包，不代表 JQ 完整历史导出已经重新通过。

## 红线

- 区域参数、信号方向、阈值和评测窗口必须在查看候选信号成绩前冻结；
- 不用当前成分或当前行业分类回填历史，不把事后区域、确认日或未来收益当作实时信息；
- 不把“最后一个活跃日”事后回填为信号日期，不只保留成功案例；
- 两套评测、顶部与底部、预测与确认都分别报告，不合成掩盖差异的总分；
- 统一事件流和两套评测验收前，不继续扩展 P1 信号实现。
