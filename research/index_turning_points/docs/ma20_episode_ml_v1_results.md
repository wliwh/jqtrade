# MA20 候选 episode 命中概率 ML V1 结果

状态：冻结规格已按 `[-5,+5]` 交易日 operational 窗口完成 2019—2026 回顾性 walk-forward。结果不支持用本版 ML 全面替代原 MA20 信号；顶部过滤精度高但召回过低，底部过滤几乎没有提高精度。

## 实现口径

本版先由 `ma20_breadth_reversal_top/bottom` 产生候选 onset，再在候选当日收盘后输出：

```text
p_episode_match = P(该 MA20 候选命中同方向 operational 区域 | 候选已出现)
score = 100 × p_episode_match
```

主标签为候选 onset 位于 strict 核心峰瓣，或距同方向区域锚点不超过前后 5 个交易日；候选和区域一对一匹配。它不是任意交易日属于顶部/底部的概率。输入、模型、切分、校准和过滤规则见 [`ma20_episode_ml_v1_spec.md`](ma20_episode_ml_v1_spec.md)。

全样本候选数据覆盖 2012-07-05 至 2026-08-14，共 161 个候选：顶部 80 个、底部 81 个。收窄后的 operational 命中为顶部 19 个、底部 23 个；旧 ±20 日审计口径分别为 26 个、28 个，说明 5 日窗口已经实际改变标签。

## 2019—2026 候选概率

以下为逐年因果 OOS 概率合并后的诊断。每个年度候选很少，逐年 AUC 波动很大，因此以合并结果为主，不把年度中位 AUC 当作稳定证据。

| 方向 | 候选/命中 | 实际命中率 | 平均预测概率 | pooled AUC | AP | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 顶部 | 34 / 8 | 23.5% | 24.2% | 0.606 | 0.527 | 0.174 | 0.532 |
| 底部 | 34 / 10 | 29.4% | 20.2% | 0.542 | 0.466 | 0.208 | 0.612 |

顶部的跨年平均概率与实际频率接近，但排序能力仅为弱正向；底部平均概率低估实际命中率约 9.2 个百分点，排序能力接近随机。当前校准不能称为稳定解决：样本只有 68 个候选，单年通常只有 1～9 个候选，概率和阈值会受少数 episode 明显影响。

## Operational 过滤结果

这里的“命中保留率”是 ML 保留的 operational 命中数除以原 MA20 候选已经命中的数量；它衡量过滤损失，不等于全部顶底区域召回。

| 方向 | 原 MA20：命中/候选（精确率） | MA20＋ML：命中/报警（精确率） | 命中保留率 |
| --- | ---: | ---: | ---: |
| 顶部 | 8 / 34（23.5%） | 2 / 2（100.0%） | 25.0% |
| 底部 | 10 / 34（29.4%） | 6 / 20（30.0%） | 60.0% |
| 合计 | 18 / 68（26.5%） | 8 / 22（36.4%） | 44.4% |

合计精确率的提高主要来自顶部只留下两个成功案例。顶部获得“少而准”的回顾性结果，但漏掉 6/8 个原本能命中的 MA20 候选；底部减少 14 个候选后只少 4 个命中，精确率几乎不变。

## 统一 Stage-D 对照

以下均为相同 2019-01-02 至 2026-08-14 覆盖期、全 A、onset、全部区域形态。括号为区域召回 / episode 精确率；`window` 是统一 Stage-D 保留的旧 ±20 日审计口径，不是模型的 5 日主标签。

| 方向 | strict：原 MA20 → MA20＋ML | window：原 MA20 → MA20＋ML |
| --- | --- | --- |
| 顶部 | 0.571 / 0.235 → 0.143 / 1.000 | 0.929 / 0.382 → 0.143 / 1.000 |
| 底部 | 0.333 / 0.147 → 0.133 / 0.100 | 0.867 / 0.382 → 0.467 / 0.350 |

这进一步说明本版不是 MA20 的升级替代：顶部精确率换来了大幅召回损失；底部的 strict/window 精确率反而略降。Stage-D 的信号后 OHLC 表已生成，但本版主要目标是候选区域命中，且只有 2 个顶部报警、20 个底部报警，不据此追加收益结论或回看调参。

## 结论与使用边界

- 原 MA20 候选继续作为事实上的主信号和必须战胜的基线。
- `pred_probability_episode_match` 可以作为候选的辅助排序字段；不能宣称是稳定绝对概率，尤其不能把底部概率直接解释为可靠置信度。
- 冻结过滤器不应默认替换 MA20：顶部过稀，底部没有形成有意义的精度提升。
- 不在 V1 上根据本次结果修改特征或阈值。若继续研究，应新立版本，优先验证跨年概率尺度或只排序不硬过滤，并等待新增日期作真正前瞻检验。

## 产物与复现

- [候选 episode 数据集](../artifacts/modeling/all_a_ma20_episode_dataset_v1_20120705_20260814/)
- [OOS 概率与过滤 bundle](../artifacts/modeling/all_a_ma20_episode_match_walk_forward_v1_20190102_20260814/)
- [统一 Stage-D 评测](../artifacts/evaluations/all_a_ma20_episode_match_walk_forward_v1_20190102_20260814__stage_d_v1/)

```bash
/home/hh01/anaconda3/envs/fin/bin/python -m research.index_turning_points.pipelines.build_ma20_episode_dataset \
  --output-dir research/index_turning_points/artifacts/modeling/<dataset_version>

/home/hh01/anaconda3/envs/fin/bin/python -m research.index_turning_points.pipelines.train_ma20_episode_walk_forward \
  --dataset-dir research/index_turning_points/artifacts/modeling/<dataset_version> \
  --output-dir research/index_turning_points/artifacts/modeling/<training_version>
```

以上结果是截至 2026-08-14 的回顾性 OOS，不是交易回测、投资建议或协议冻结后的新增前瞻证据。
