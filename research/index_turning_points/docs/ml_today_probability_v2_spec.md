# 全 A 当日顶底概率 ML V2 冻结规格

状态：本规格在首次生成 V2 真实 OOS 结果前冻结。V2 只改进概率校准与报警决策层，复用 V1 数据集、15 个输入、strict membership 真值和两个模型；任何下述口径变更必须升级版本，不覆盖 V1/V2 产物。

## 不变部分

- 数据集：`all_a_ml_today_dataset_v1`；
- 任务：收盘后估计当日属于 strict 顶部/底部峰瓣的两个独立概率；
- 分数：`100 × pred_probability_today`；
- 输入、标签、辅助 intensity、Elastic Net、浅层 GBDT 及随机种子完全沿用 V1；
- 结果仍是回顾性年度 OOS，不是交易策略或新增前瞻证据。

V1 完整定义见 [`ml_today_probability_v1_spec.md`](ml_today_probability_v1_spec.md)。

## 三年校准窗

每个测试年 `Y` 冻结为：

```text
训练：所有年份 < Y-3
校准：Y-3、Y-2、Y-1 三个完整日历年
测试：Y
```

训练段末尾和校准段末尾各去掉 20 个交易日，继续隔离事后 strict 标签的边界依赖。模型、缺失处理和标准化只拟合训练段；三年校准段不参与模型拟合，只用于 sigmoid 概率校准与报警阈值选择。首个测试年仍为 2019，对应 2016—2018 校准窗。

有效 sigmoid 校准至少需要 5 个 strict 正例和 30 个负例。若三年窗不满足条件：

1. 不拟合双参数 sigmoid；
2. 保留模型原始排序，用单一 logit 截距平移，使校准后平均概率等于训练段加一平滑后的正例率；
3. 标记 `probability_status=insufficient_calibration_events`；
4. 该测试折只输出参考概率，不产生正式报警。

有效校准标记为 `probability_status=calibrated`。回退不是可靠绝对概率，不得与正常折混报。

## 概率审计

除 Brier、log loss、ROC AUC、average precision 和正例率外，V2 新增固定十等分概率箱：`[0,0.1)` 至 `[0.9,1.0]`。每个测试折、模型和方向记录：

- 样本数；
- 平均预测概率；
- 实际 membership 比例；
- 二者绝对差。

总体 ECE 为各非空概率箱绝对差按样本数加权的平均值。概率校准是否改善主要看 Brier、log loss、ECE 与可靠性表，不用 AUC 单独下结论。

## 报警状态机

阈值只在 `probability_status=calibrated` 时启用。

1. 三年校准窗的 episode 总预算为 `3 × 6 = 18`；沿用标签无关的唯一分数遍历，选择不超过预算且 episode 数最多的候选阈值。
2. 正式进入阈值为 `max(候选阈值, 0.50)`。
3. 未激活时，概率达到进入阈值才产生 onset。
4. 激活后使用固定退出阈值 `0.30`；概率仍不低于 0.30 时保持 continuation，即使已低于进入阈值。
5. 概率低于 0.30 的当日退出；随后 10 个交易日处于冷却期，不允许重新进入。
6. 冷却结束后再次达到进入阈值才可产生新 episode。

`raw_triggered` 只表示当日概率是否达到进入阈值；`triggered` 表示迟滞与冷却状态机的最终活跃状态。该策略预期以降低报警频率和重复报警为代价牺牲部分召回，不回看测试结果重选参数。

## 版本与入口

- training：`all_a_ml_today_walk_forward_v2`；
- dataset 继续使用 `all_a_ml_today_dataset_v1`，不重复生成。

```bash
/home/hh01/anaconda3/envs/fin/bin/python \
  -m research.index_turning_points.pipelines.train_ml_today_calibrated_walk_forward \
  --dataset-dir research/index_turning_points/artifacts/modeling/all_a_ml_today_dataset_v1_20120705_20260814 \
  --output-dir research/index_turning_points/artifacts/modeling/<training_version>
```

输出目录必须不存在或为空。训练 bundle 除既有 6 个 CSV 外新增 `calibration_reliability.csv`；manifest 记录校准窗、最低事件数、概率状态、迟滞阈值、冷却期、运行环境与全部逻辑哈希。
