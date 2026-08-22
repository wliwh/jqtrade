# 全 A 当日顶底概率 ML V1 结果备忘

状态：冻结规格、数据集、2019—2026 年度 expanding walk-forward 与统一 Stage-D 评测均已完成。结果已经查看，本版不得再回调标签、输入、模型、切分、校准或告警语义。

冻结口径见 [`ml_today_probability_v1_spec.md`](ml_today_probability_v1_spec.md)。本版只估计“当日是否属于 strict 顶部/底部峰瓣”的收盘后概率，不预测未来 5/10/20 日，也不把事后 intensity 当训练目标。

## 产物

- [`all_a_ml_today_dataset_v1_20120705_20260814`](../artifacts/modeling/all_a_ml_today_dataset_v1_20120705_20260814/)：3429 个交易日，3427 日具备全 A 价格与标签；strict 顶部/底部日分别为 310/158；
- [`all_a_ml_today_walk_forward_v1_20190102_20260814`](../artifacts/modeling/all_a_ml_today_walk_forward_v1_20190102_20260814/)：8 个测试年度、7392 行“日期×模型×方向”OOS 输出、157 个阈值 episode；
- [`Stage-D V1`](../artifacts/evaluations/all_a_ml_today_walk_forward_v1_20190102_20260814__stage_d_v1/)：统一区域定位与信号后 OHLC 评测。

数据集只导出冻结的 15 个模型输入、数据质量字段、二分类 membership 和辅助 intensity；没有未来期限目标。训练 bundle 中 `pred_score` 与 `raw_value` 均严格等于 `100 × pred_probability_today`。

## 概率结果

2019—2026 OOS 中，顶部 membership 为 163/1848 日，底部为 74/1848 日。下表是年度指标的中位数与范围；单一年份没有正例时 AUC/AP 留空，不纳入中位数。

| 模型/方向 | Brier 中位数 | AUC 中位数（范围） | AP 中位数（范围） |
| --- | ---: | ---: | ---: |
| Elastic Net 顶部 | 0.076 | 0.782（0.094～0.938） | 0.248（0.027～0.730） |
| 浅层 GBDT 顶部 | 0.079 | 0.810（0.683～0.980） | 0.279（0.108～0.862） |
| Elastic Net 底部 | 0.035 | 0.932（0.500～0.991） | 0.467（0.048～0.784） |
| 浅层 GBDT 底部 | 0.034 | 0.961（0.500～0.992） | 0.578（0.066～0.751） |

浅层 GBDT 的年度排序整体强于 Elastic Net，尤其避免了 Elastic Net 顶部在 2022 年 AUC 仅 0.094 的明显反向失效。但指标跨年波动很大：2023 年没有底部正例，导致用于 2024 测试折的验证校准退化为常数概率，两个底部模型在 2024 年 AUC 均为 0.5。当前结果支持“存在区分信息”，不支持“概率已经跨状态稳定校准”。完整逐年数据见模型 bundle 的 `probability_metrics.csv` 与 `fit_audit.csv`。

## 区域定位

全 A、onset、全部时点/区域形态的主要结果如下：

| 模型/方向 | strict 召回/episode 精确率 | window 召回/episode 精确率 | window 中位 lead/lag |
| --- | --- | --- | ---: |
| Elastic Net 顶部 | 0.357/0.135 | 0.500/0.189 | −4 日 |
| Elastic Net 底部 | 0.267/0.148 | 0.533/0.296 | −2.5 日 |
| 浅层 GBDT 顶部 | 0.571/0.182 | 0.786/0.250 | −2 日 |
| 浅层 GBDT 底部 | 0.400/0.122 | 0.600/0.184 | −1 日 |

这里的提前量来自验证期阈值派生的 episode，不改变“当日 membership 概率”的训练语义。误报仍多：浅层 GBDT 顶部/底部在全 A 各有 44/49 个 onset episode，strict 主匹配只有 8/6 个。2024 底部常数校准还会形成整段长报警，因此 episode 结果必须连同 `calibration_status` 审计，不能只看 window 召回。

## 信号后结果与结论

Stage-D 共 504 个合格 OHLC 推断，没有局部或全局 FDR 低于 0.05；全局最小 q 值约 0.820。该任务是区域当日识别，OHLC 本来就不是训练目标，本结果也不提供交易方向证据。

冻结结论：

1. 当日 strict membership 把分数语义简化成了明确的单一概率，工程目标已达到；
2. 15 个连续输入中存在可供浅层 GBDT 利用的顶底区分信息，顶部结果比 Elastic Net 稳定；
3. 稀少且不均匀的底部年份导致验证校准脆弱，当前概率不能视为跨年份稳定的绝对概率；
4. 阈值 episode 的精确率仍低，OHLC 无 FDR 发现，本版不足以升级为独立顶底预测器。

下一步若继续，应另立预冻结版本研究跨多年校准或只把概率作为连续状态量；不得围绕本次已看结果修改 V1。
