# MA20 候选 episode 命中概率 ML V1 冻结规格

状态：本规格在首次生成真实 OOS 结果前冻结。任何候选来源、命中窗口、特征、模型、切分、校准、阈值或报警语义变更必须升级版本，不覆盖本版产物。

## 任务与概率语义

本版不再用 ML 每日直接产生顶底报警。冻结的 [`MA20 宽度周期拆分 V1`](signals/ma_period_breadth_decomposition.md) 先产生顶部、底部候选 episode；ML 只在候选 onset 当日估计：

```text
p_episode_match(t)
    = P(MA20 候选 episode 命中同方向 operational 区域
        | t 日收盘时可得的数据，MA20_candidate(t)=True)
```

该概率是条件于 MA20 候选已经出现的 episode 命中概率，不是任意交易日属于顶部/底部的无条件概率。顶部和底部分开拟合。一个候选对应一行训练样本，最终保留的候选只在 onset 当日输出一个交易日的有效报警。

## Operational 命中标签

继续使用 `top_bottom_regions_v2` 的区域、核心峰瓣、方向和全 A 交易日历，但为本任务新增独立标签版本 `ma20_episode_operational_window_v1`。候选与同方向区域按以下边建立一对一匹配：

1. onset 位于任一 strict 核心峰瓣；或
2. onset 与区域锚点的交易日 lead/lag 位于 `[-5, +5]`。

边的冻结优先级为 strict 核心峰瓣、绝对锚点距离、候选日期、候选 ID、区域 ID；使用稳定贪心一对一分配。主匹配为 `target_operational_match=1`，重复候选和无边候选均为 0。预测侧和确认侧均为 5 个交易日，不再使用原评测 `window` 的 20 日外圈。

原 `strict/loose/window_20d` 一对一结果继续写入候选数据集作审计，不参与主标签生成。Operational 匹配必须在收窄后的可用边上重新执行，不得过滤旧匹配结果来回填标签。

标准答案可以使用未来完整历史；候选日期、特征和最终报警日期只能使用当日及之前数据。

## 候选输入

候选来源只包括全 A MA20 顶部/底部 reversal onset。点时特征复用 `all_a_ml_today_dataset_v1`，并增加同方向上一个 MA20 候选距今的交易日数，首个候选和超过 252 日均记为 252：

```text
breadth_ma20
breadth_ma20_change_5d
breadth_ma60
breadth_ma60_change_10d
new_high_low_net_ratio_60
new_high_low_net_ratio_60_change_5d
limit_hit_net_ratio
turnover_ratio_pct_p50
index_close_to_ma60
index_drawdown_60d
index_rebound_60d
index_volatility_20d
candidate_gap_trade_days
```

候选 episode 的持续天数、退出日、未来区域、未来收益、lead/lag、匹配状态、intensity 和标签均不得进入模型。

## 模型与时间切分

唯一模型 `l2_logistic` 冻结为：中位数缺失填充、标准化、`LogisticRegression(penalty="l2", C=0.1, solver="lbfgs", max_iter=2000, class_weight=None, random_state=20260821)`。训练段只有一个类别时使用加一平滑常数概率。不比较 GBDT/XGBoost。

首个历史 OOF 候选年为 2016，正式测试年为 2019—2026。每个年度 `Y` 的原始候选概率由所有早于 `Y` 的候选拟合，但去掉 `Y` 前最后 20 个全 A 交易日内的候选，隔离事后标签边界。每个历史候选的 OOF 概率只能由更早年份数据生成。

测试年 `Y` 的概率校准只使用 2016 至 `Y-1` 的历史 OOF 候选，并再次排除 `Y` 前最后 20 个交易日。至少需要 3 个 operational 正例和 10 个负例才拟合 sigmoid；不足时只做保持排序的训练正例率 logit 平移，并标记 `insufficient_episode_evidence`。

## Episode 过滤阈值

证据充足时，只在历史 OOF 候选上遍历唯一概率阈值。候选阈值必须同时满足：

- operational 正例召回不低于 0.60；
- 至少保留 6 个候选；
- 平均每个校准年度最多保留 6 个候选。

在可行阈值中最大化命中精确率的 95% Wilson 下界；依次用原始精确率、召回率和更高阈值打破平局。没有可行阈值时原样放行 MA20 候选。证据不足时同样不做 ML 过滤，阈值记为 0，避免把“不知道”解释成“不报警”。

`raw_triggered` 表示 MA20 候选 onset，`triggered` 表示通过 ML 过滤后的单日报警。ML 只能过滤 MA20 候选，不能发现 MA20 完全漏掉的区域。

## 输出、指标与版本

- dataset：`all_a_ma20_episode_dataset_v1`；
- training：`all_a_ma20_episode_match_walk_forward_v1`；
- operational label：`ma20_episode_operational_window_v1`。

候选概率报告逐年 Brier、log loss、AUC、AP、ECE 和固定十等分可靠性表；过滤结果继续进入统一 Stage-D，报告 strict/loose/旧 window 的区域召回、episode 精确率、误报和 lead/lag。另按本版 operational 标签直接报告候选保留数、命中数、精确率与召回率。

必须在相同 2019—2026 覆盖期比较：MA20 原始候选、MA20＋ML 过滤。结果是回顾性 OOS，不是协议冻结后的新增前瞻证据。
