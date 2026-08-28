# 研究产物

产物按事实类型和版本隔离：

- `ground_truth/<bundle>/`：事后标签、区域、峰瓣和未来结果；
- `signals/<signal_version>/`：严格点时的逐日信号与 episode；
- `evaluations/<evaluation_version>/`：区域定位明细/完整切片/报告、信号后 OHLC 明细/统计/报告及统一 manifest；
- `modeling/<dataset_version>/`：点时特征与事后训练目标组成的版本化 ML 数据集；
- `modeling/<training_version>/`：年度 walk-forward 的 OOS 逐日评分、episode、概率指标、切分/阈值/拟合审计表；
- `viewers/<viewer_version>/`：可删除并重建的人工审计 HTML。

现行人工审计入口为 [`top_bottom_regions_ma20_v1/index_turning_points_ma20.html`](viewers/top_bottom_regions_ma20_v1/index_turning_points_ma20.html)：上排显示所选指数及 `top_bottom_regions_v2`，下排显示同一条点时全 A MA20 宽度，默认标签为全 A。旧 `top_bottom_regions_v2` 与四行业 V1 HTML 保留为历史查看器。

[`bsearch_index_exploration_v1_6`](viewers/bsearch_index_exploration_v1_6_20110104_20260814/bsearch_index_exploration.html) 是唯一保留的搜索热度查看器。它展示 13 个有本地可比指数的原始关键词、11 个本地指数、OHLC K线、原始热度和实线点时 Z252；自动匹配的指数仍可手动改选，CSV 非交易日会被跳过。

峰值标注按关键词保存在浏览器 `localStorage`，也可通过带源数据哈希的 JSON 整包导入/导出。标注模式保留平移和缩放；页面不展示统计结果，也不写回输入或冻结标签。

现有 [`ground_truth/index_ohlc_20260814/`](ground_truth/index_ohlc_20260814/) 是目录重组前已经生成的 bundle。其 CSV、manifest、哈希和 manifest 内的旧逻辑文件路径均保持原样，代表生成时事实；新 bundle 由 `pipelines/build_ground_truth.py` 写入一个全新目录。

评测 bundle 由 `pipelines/evaluate_signal.py` 写入全新目录，拒绝覆盖非空目录，并要求 TDX OHLC 与 ground-truth manifest 的来源哈希一致。区域与信号后结果不合成总分。

ML 数据集由 `pipelines/build_ml_dataset.py` 生成。旧版未来进入概率由 `pipelines/train_ml_walk_forward.py` 训练，冻结口径见 [`../docs/ml_training_v1_memo.md`](../docs/ml_training_v1_memo.md)；当日 strict membership V1/V2 分别由 `pipelines/train_ml_today_walk_forward.py` 和 `pipelines/train_ml_today_calibrated_walk_forward.py` 训练，口径与结果见 [`V1 规格`](../docs/ml_today_probability_v1_spec.md)、[`V1 结果`](../docs/ml_today_probability_v1_results.md)、[`V2 规格`](../docs/ml_today_probability_v2_spec.md) 和 [`V2 结果`](../docs/ml_today_probability_v2_results.md)。MA20 候选 episode ML 由 `pipelines/build_ma20_episode_dataset.py` 和 `pipelines/train_ma20_episode_walk_forward.py` 生成，主标签收窄为 strict 或锚点前后 5 个交易日，见 [`冻结规格`](../docs/ma20_episode_ml_v1_spec.md) 与 [`结果`](../docs/ma20_episode_ml_v1_results.md)。所有入口都拒绝覆盖非空目录。

## 现役信号与评测

| 信号 | signal bundle | evaluation bundle |
| --- | --- | --- |
| 四行业 Top1 | [`four_industry_top1_v2_20211213_20260814`](signals/four_industry_top1_v2_20211213_20260814/) | [`stage_d_v2`](evaluations/four_industry_top1_v2_20211213_20260814__stage_d_v2/)¹ |
| 单行业 Top1 | [`single_industry_top1_v1_20170103_20260814`](signals/single_industry_top1_v1_20170103_20260814/) | [`stage_d_v1`](evaluations/single_industry_top1_v1_20170103_20260814__stage_d_v1/) |
| 多周期 MA 宽度 | [`multi_period_ma_breadth_v1_20120104_20260814`](signals/multi_period_ma_breadth_v1_20120104_20260814/) | [`stage_d_v1`](evaluations/multi_period_ma_breadth_v1_20120104_20260814__stage_d_v1/) |
| MA 周期拆分 | [`ma_period_breadth_decomposition_v1_20120104_20260814`](signals/ma_period_breadth_decomposition_v1_20120104_20260814/) | [`stage_d_v1`](evaluations/ma_period_breadth_decomposition_v1_20120104_20260814__stage_d_v1/) |
| 宽度—指数背离 | [`breadth_price_divergence_v1_20120104_20260814`](signals/breadth_price_divergence_v1_20120104_20260814/) | [`stage_d_v1`](evaluations/breadth_price_divergence_v1_20120104_20260814__stage_d_v1/) |
| 新高—新低广度 | [`new_high_low_breadth_v1_20120104_20260814`](signals/new_high_low_breadth_v1_20120104_20260814/) | [`stage_d_v1`](evaluations/new_high_low_breadth_v1_20120104_20260814__stage_d_v1/) |
| 新高—新低周期拆分 | [`new_high_low_period_decomposition_v1_20120104_20260814`](signals/new_high_low_period_decomposition_v1_20120104_20260814/) | [`stage_d_v1`](evaluations/new_high_low_period_decomposition_v1_20120104_20260814__stage_d_v1/) |
| 涨跌停广度 | [`limit_up_down_breadth_v1_20120705_20260814`](signals/limit_up_down_breadth_v1_20120705_20260814/) | [`stage_d_v1`](evaluations/limit_up_down_breadth_v1_20120705_20260814__stage_d_v1/) |
| 换手热度 | [`turnover_heat_v1_20120705_20260814`](signals/turnover_heat_v1_20120705_20260814/) | [`stage_d_v1`](evaluations/turnover_heat_v1_20120705_20260814__stage_d_v1/) |
| 全 A ML V2 OOS | [`all_a_ml_walk_forward_v2_20190102_20260814`](modeling/all_a_ml_walk_forward_v2_20190102_20260814/) | [`stage_d_v1`](evaluations/all_a_ml_walk_forward_v2_20190102_20260814__stage_d_v1/) |
| 全 A ML V3 OOS | [`all_a_ml_walk_forward_v3_20190102_20260814`](modeling/all_a_ml_walk_forward_v3_20190102_20260814/) | [`stage_d_v1`](evaluations/all_a_ml_walk_forward_v3_20190102_20260814__stage_d_v1/) |
| 全 A 当日顶底概率 ML V1 OOS | [`all_a_ml_today_walk_forward_v1_20190102_20260814`](modeling/all_a_ml_today_walk_forward_v1_20190102_20260814/) | [`stage_d_v1`](evaluations/all_a_ml_today_walk_forward_v1_20190102_20260814__stage_d_v1/) |
| 全 A 当日顶底概率 ML V2 OOS | [`all_a_ml_today_walk_forward_v2_20190102_20260814`](modeling/all_a_ml_today_walk_forward_v2_20190102_20260814/) | [`stage_d_v1`](evaluations/all_a_ml_today_walk_forward_v2_20190102_20260814__stage_d_v1/) |
| MA20 候选 episode ML V1 OOS | [`all_a_ma20_episode_match_walk_forward_v1_20190102_20260814`](modeling/all_a_ma20_episode_match_walk_forward_v1_20190102_20260814/) | [`stage_d_v1`](evaluations/all_a_ma20_episode_match_walk_forward_v1_20190102_20260814__stage_d_v1/) |

全 A ML V2/V3 的共同输入数据集为 [`all_a_ml_dataset_v1_20120705_20260814`](modeling/all_a_ml_dataset_v1_20120705_20260814/)。V3 是查看 V2 结果后的短周期评分和两日告警改版。模型 bundle 和评测 bundle 均记录 `fin` 解释器、输入文件及逻辑哈希；其 2019—2026 结果是回顾性 OOS，不是协议冻结后的新增前瞻样本。

全 A 当日顶底概率 ML V1/V2 的共同输入为 [`all_a_ml_today_dataset_v1_20120705_20260814`](modeling/all_a_ml_today_dataset_v1_20120705_20260814/)，只保留 15 个冻结连续输入、strict membership 二分类真值和辅助 intensity。V2 增加三年校准、固定概率箱、校准状态和滞回/冷却报警审计；其 2019—2026 结果同样是回顾性 OOS，且固定 0.50 进入门槛被评测证明过于保守。

MA20 候选 episode ML V1 的输入为 [`all_a_ma20_episode_dataset_v1_20120705_20260814`](modeling/all_a_ma20_episode_dataset_v1_20120705_20260814/)。它只在 MA20 onset 上输出条件命中概率；2019—2026 顶部仅保留 2 个报警，底部精确率几乎没有提高，因此目前仅作候选评分审计，不替代原 MA20 信号。

¹ `stage_d_v1` 是报告规则增强前的不可变中间版本；指标 CSV 口径未变，现行入口为 v2。
