# 全 A 顶底评分 ML V1/V2/V3 备忘

状态：V2 首次建模及 V3 结果知情改版均已完成回顾性 walk-forward 与统一区域定位/OHLC 评测；V3 修正了长时间连续报警，但区域定位结果有得有失，不构成前瞻证据。任何标签、特征、切分、模型参数、报警预算、评分公式或告警语义变更都必须升级版本，不覆盖既有产物。

## 目标与范围

- 主任务只训练全 A `000985.XSHG` 的 `top_score` 与 `bottom_score`，二者独立且不要求和为 100。
- 其他六个指数第一版只作迁移评测，不把同一日期复制成独立训练样本。
- 一行代表一个交易日；每行使用截至当日收盘可得的连续量及 1/5/10/20/60 等历史窗口，不使用序列神经网络。

## 标签与预测分

事后标签与点时预测严格分离：

- `truth_top_intensity` / `truth_bottom_intensity` 是 0～100 的事后强度标签。
- `target_*_within_5d/10d/20d` 表示从日期 `t` 到未来第 5/10/20 个指数交易日是否进入相应 strict 主瓣，尾部窗口不完整时为空。
- 最终 `pred_top_score` / `pred_bottom_score` 是模型点时输出，不要求在真实极值日恰好等于 100。

三个期限的概率必须满足 `p5 <= p10 <= p20`。V1/V2 展示分冻结为：

```text
pred_score = 100 × (0.5×p5 + 0.3×p10 + 0.2×p20)
```

V3 在查看 V2 结果后另立版本，将展示分改为 `100 × (0.7×p5 + 0.3×p10)`；`p20` 仍照常训练、导出和评测，只是不再进入综合展示分。该修改是结果知情的回顾性研究选择，不能算前瞻冻结。

每个 strict 主瓣独立取代表极值。顶部使用日内最高价、底部使用日内最低价：

```text
top_intensity(t) = 100 × clip(
    1 - ((H_lobe - high[t]) / H_lobe) / price_band_pct,
    0, 1
)

bottom_intensity(t) = 100 × clip(
    1 - ((low[t] - L_lobe) / L_lobe) / price_band_pct,
    0, 1
)
```

主瓣代表极值为 100；strict 主瓣之外，包括 M/W 主瓣之间的桥接区间，均为 0。`loose/window` 只用于评测。

## 最大容差点时状态

状态复用 Plotly 图使用的方向变化阈值和 OHLC 判断顺序。全 A 使用基础阈值 10% 及其固定倍率 1.0。训练字段 `index_phase_pti` 必须逐日在线生成且永不回填：

- 初始化为 `pending`。
- 已确认上行时：创/平运行最高价为 `up`；未创新高且回撤不足阈值为 `pending`；达到阈值则确认并切换为 `down`。
- 已确认下行时：创/平运行最低价为 `down`；未创新低且反弹不足阈值为 `pending`；达到阈值则确认并切换为 `up`。
- 同日既更新极值又跨越反向阈值时，沿用现有标签器的保守顺序：先更新极值，不假定日内高低发生顺序。

Plotly 的完整历史背景会在未来确认后把锚点至确认日回画为 `pending`，记为 `plot_phase_posthoc`，只能用于展示和事后诊断，不能作为训练特征。

## 第一版输入

- 全 A MA20/60/120 宽度及 1/5/10 日变化；
- 60/120/250 日新高—新低净广度；
- 涨跌停触及/封板净比例；
- 换手中位数、流通市值加权均值、极端占比；
- 各口径有效覆盖比例；
- 全 A 指数的多期限收益、均线距离、回撤/反弹、振幅和波动；
- 最大容差点时三状态及 one-hot 字段。

共同起点为 2012-07-05。第一版不加入行业维度，也不使用离散候选信号的 `triggered` 字段。

## 模型与验证

V1 已冻结并保留两个固定参数基线，不做测试段调参：

- `elastic_net`：中位数缺失填充（附缺失指示）→标准化→Elastic Net 逻辑回归；`l1_ratio=0.5`、`C=0.05`、`class_weight=balanced`、`max_iter=3000`。
- `shallow_gbdt`：中位数缺失填充（附缺失指示）→`GradientBoostingClassifier`；60 棵树、`learning_rate=0.03`、`max_depth=2`、`min_samples_leaf=20`、`subsample=0.8`，拟合时使用类别平衡权重。
- 随机种子统一为 `20260821`。

V1 结果查看后才在 `fin` 环境安装 XGBoost，所以不能原地替换 V1。V2 版本 `all_a_ml_walk_forward_v2` 冻结比较：

- `simple_rule`：顶部/底部各使用 MA20/MA60 宽度 10 日变化、60 日新高—新低 10 日变化、涨跌停触及净比例 5 日变化、换手中位数 10 日变化及全 A 相对 MA60 距离；符号按顶部退潮/高位与底部修复/低位预先指定。仅用训练窗中位数和 IQR 缩放、截断至 ±3 后等权平均，不用标签拟合权重。
- `elastic_net`：完全沿用 V1 参数，作为跨版本锚点。
- `shallow_xgboost`：`XGBClassifier`，100 棵树、`learning_rate=0.03`、`max_depth=2`、`min_child_weight=20`、`subsample=0.8`、`colsample_bytree=0.8`、`reg_alpha=0.05`、`reg_lambda=1.0`、CPU `hist`、单线程及类别平衡权重。

正式运行环境固定为 `/home/hh01/anaconda3/envs/fin/bin/python`（Python 3.9.0）；依赖版本见 [`../requirements-ml.txt`](../requirements-ml.txt)。训练 manifest 同时记录解释器绝对路径和 Pandas/scikit-learn/XGBoost 版本。

年度 expanding walk-forward 冻结为：

1. 首个验证年为 2018，首个测试年为 2019；以后逐年滚动，训练窗只扩张不滑动。
2. 对测试年 `Y`，训练集为 `Y-1` 年以前的全部日期，但去掉末尾 20 个交易日；验证集为 `Y-1` 年，同样去掉末尾 20 个交易日；测试集为 `Y` 年全部已具备特征的日期。
3. 两段 20 日边界隔离防止最长 20 日未来进入标签跨越 train→validation 或 validation→test。
4. 缺失处理、标准化和分类器只拟合训练段；sigmoid 校准与报警阈值只拟合验证段。验证标签只有单一类别时，校准器使用加一平滑后的验证期基准概率，并在审计表中明示。
5. 三期限各自拟合、校准，再按逐行 L2 单调投影得到 `p5 <= p10 <= p20`。

报警预算固定为每个验证年、每个模型、每个方向最多 6 个连续 episode。阈值遍历验证分数的唯一值，选择不超预算且 episode 数最多的阈值；episode 数相同时保留更高阈值。测试年沿用该阈值，测试实际报警数可以超过 6，不能回看测试结果重选。

输出 `oos_signal_daily.csv`、`oos_signal_episodes.csv`、`probability_metrics.csv`、`folds.csv`、`thresholds.csv`、`fit_audit.csv` 和 manifest。逐日文件只含 2019 年起各测试年的 OOS 预测；尾部未来窗口不完整的真实标签保持为空，但仍可生成当日点时预测。该 signal/episode schema 可直接交给 `evaluate_signal.py`，分别生成区域定位和信号后 OHLC bundle。

区域定位与信号后 5/10/20 日 OHLC 必须继续分开报告，不生成组合总分。截止 2026-08-14 的结果已被查看，本次只能称为回顾性时间外评测；真正前瞻证据从协议冻结后的新增日期开始积累。

## V2 首次冻结结果

正式产物为：

- [点时训练数据集](../artifacts/modeling/all_a_ml_dataset_v1_20120705_20260814/)；
- [V2 OOS 模型 bundle](../artifacts/modeling/all_a_ml_walk_forward_v2_20190102_20260814/)；
- [Stage-D 区域/OHLC 评测](../artifacts/evaluations/all_a_ml_walk_forward_v2_20190102_20260814__stage_d_v1/)。

2019-01-02 至 2026-08-14 共导出 8 个测试年度、11088 行模型/方向逐日记录和 203 个 episode。全 A onset `window` 口径下，浅层 XGBoost 顶部召回/精确率为 0.7143/0.3226，底部为 0.6000/0.2903；但 `strict` 精确率仅为 0.1290/0.0968。简单规则的多数概率 AUC 接近 0.5，适合作为低门槛对照，不能作为有效信号。

OHLC 共 756 个合格检验；全局 FDR 后有 4 项低于 0.05，但主要结果不符合“顶部后应转弱”的预期方向，不能据此宣布顶底预测成立，也不能回看结果修改本版本阈值或参数。

## V3 结果知情改版

V3 的标签、特征、年度切分、三个模型、概率校准和每年 6 个 episode 预算完全沿用 V2，只冻结两项修改：

1. 综合分从 `50%×p5 + 30%×p10 + 20%×p20` 改为 `70%×p5 + 30%×p10`，聚焦 V2 中区分能力更强的 5/10 日概率；20 日概率继续作为独立审计输出。
2. 每次原始分数连续越过阈值时，只把前 2 个交易日标为有效报警；其后保持抑制，直到原始分数先跌回阈值以下才允许重新触发。该状态机只依赖截至当日可见的原始阈值状态，不回填触发日。

正式产物为：

- [V3 OOS 模型 bundle](../artifacts/modeling/all_a_ml_walk_forward_v3_20190102_20260814/)；
- [V3 Stage-D 区域/OHLC 评测](../artifacts/evaluations/all_a_ml_walk_forward_v3_20190102_20260814__stage_d_v1/)。

模型概率没有变化：V2/V3 的 `probability_metrics.csv` SHA-256 同为 `6b9580afd3572cb67185140ad2c5409249cad3c5a9a1bfbb9c4fb02a4be686df`。变化只发生在三个期限如何合成分数，以及越阈值后如何输出有效报警。

V3 共导出相同的 8 个测试年度和 11088 行逐日记录、194 个 episode。两日状态机把 V3 原始阈值激活的 1951 天压缩为 304 天；相对 V2 的 1908 个激活日减少 84.1%，所有模型/方向的最长 episode 均为 2 天，消除了 V2 中最长 120～254 天的持续报警。

全 A onset 的主要区域指标如下。括号内依次为召回/episode 精确率：

| 模型/方向 | strict V2 → V3 | window V2 → V3 |
| --- | --- | --- |
| Elastic Net 顶部 | 0.2143/0.1000 → 0.2857/0.1667 | 0.3571/0.1667 → 0.4286/0.2500 |
| Elastic Net 底部 | 0.2000/0.1034 → 0.2000/0.1429 | 0.4000/0.2069 → 0.5333/0.3810 |
| 浅层 XGBoost 顶部 | 0.2857/0.1290 → 0.3571/0.1724 | 0.7143/0.3226 → 0.5000/0.2414 |
| 浅层 XGBoost 底部 | 0.2000/0.0968 → 0.2667/0.1333 | 0.6000/0.2903 → 0.6000/0.3000 |

结论必须拆开看：V3 明确修复了“高分后数月持续报警”的输出语义，Elastic Net 和 XGBoost 的 strict 定位也有所改善；但 XGBoost 顶部 window 指标明显下降，不能称为全面优于 V2。V3 的 756 个合格 OHLC 检验没有全局 FDR 发现；这消除了 V2 中 4 项方向不符合顶部预期的显著结果，但也意味着仍无稳健的信号后走势证据。V3 应作为更合理的短时告警候选继续前瞻积累，不应据本次回顾性结果继续调权重或阈值。

## 运行入口

输出目录必须为空，已有 bundle 不会被覆盖：

```bash
# 1. 生成点时特征 + 事后标签数据集
/home/hh01/anaconda3/envs/fin/bin/python -m research.index_turning_points.pipelines.build_ml_dataset \
  --output-dir research/index_turning_points/artifacts/modeling/<dataset_version>

# 2. 年度 walk-forward 拟合与 OOS 导出
/home/hh01/anaconda3/envs/fin/bin/python -m research.index_turning_points.pipelines.train_ml_walk_forward \
  --dataset-dir research/index_turning_points/artifacts/modeling/<dataset_version> \
  --output-dir research/index_turning_points/artifacts/modeling/<training_version>

# 3. 分别生成区域定位与信号后 OHLC 评测
/home/hh01/anaconda3/envs/fin/bin/python -m research.index_turning_points.pipelines.evaluate_signal \
  --signal-daily research/index_turning_points/artifacts/modeling/<training_version>/oos_signal_daily.csv \
  --signal-episodes research/index_turning_points/artifacts/modeling/<training_version>/oos_signal_episodes.csv \
  --ground-truth-dir research/index_turning_points/artifacts/ground_truth/index_ohlc_20260814 \
  --evaluation-version <evaluation_version> \
  --output-dir research/index_turning_points/artifacts/evaluations/<evaluation_version>
```
