# 全 A 未来进入概率 ML V1/V2/V3

本文件只记录“当日或未来 5/10/20 日进入 strict 峰瓣”的旧实验。当日 membership 与 MA20 候选任务分别见 [`ml_today_probability_v2_results.md`](ml_today_probability_v2_results.md) 和 [`ma20_episode_ml_v1_results.md`](ma20_episode_ml_v1_results.md)。截至 2026-08-14 的结果均为回顾性 OOS，任何规则变更必须升级版本。

## 概率语义

顶部和底部独立建模，一行一个全 A 交易日：

```text
p5/p10/p20 = P(从当日到未来 5/10/20 日进入同方向 strict 峰瓣)
V1/V2 score = 100 × (0.5×p5 + 0.3×p10 + 0.2×p20)
V3 score    = 100 × (0.7×p5 + 0.3×p10)
```

三个概率经单调投影满足 `p5 <= p10 <= p20`。事后 intensity、loose/window、未来收益和完整历史绘图 phase 不进入模型；点时 phase 只按当日及历史 OHLC 在线生成。

## 冻结设计

| 项目 | 口径 |
| --- | --- |
| 数据 | `all_a_ml_dataset_v1`，2012-07-05 起；市场宽度、新高新低、涨跌停、换手、覆盖率和全 A 价格状态连续量 |
| V1 模型 | Elastic Net 逻辑回归；深度 2、60 棵树的浅层 GBDT |
| V2 模型 | 等权简单规则、V1 Elastic Net、100 棵深度 2 的浅层 XGBoost |
| 切分 | 首个验证年 2018、测试年 2019；expanding walk-forward；训练和验证末尾各隔离 20 个交易日 |
| 校准 | 只用验证段 sigmoid；单类别验证段使用加一平滑常数概率 |
| 阈值 | 每验证年、模型、方向最多 6 个 episode；测试年原样沿用 |
| 环境 | `/home/hh01/anaconda3/envs/fin/bin/python`；版本和参数写入 manifest |

V3 查看 V2 结果后只改两项：展示分不再使用 `p20`，并把每次连续越阈值的有效报警限制为前 2 个交易日；概率模型和 `probability_metrics.csv` 与 V2 相同。

## 回顾性结果

V2 覆盖 2019—2026，共 203 个 episode。浅层 XGBoost 的全 A onset `window` 召回/精确率为：顶部 `0.714/0.323`、底部 `0.600/0.290`；strict 精确率仅 `0.129/0.097`。简单规则多数 AUC 接近 0.5。OHLC 的 4 项全局 FDR 发现方向不符合顶部假设，不能作为顶底证据。

V3 把原始阈值激活的 1951 天压缩为 304 天，所有 episode 最长 2 天。主要区域指标如下（召回/episode 精确率）：

| 模型/方向 | strict V2→V3 | window V2→V3 |
| --- | --- | --- |
| Elastic Net 顶部 | 0.214/0.100 → 0.286/0.167 | 0.357/0.167 → 0.429/0.250 |
| Elastic Net 底部 | 0.200/0.103 → 0.200/0.143 | 0.400/0.207 → 0.533/0.381 |
| XGBoost 顶部 | 0.286/0.129 → 0.357/0.172 | 0.714/0.323 → 0.500/0.241 |
| XGBoost 底部 | 0.200/0.097 → 0.267/0.133 | 0.600/0.290 → 0.600/0.300 |

V3 修复了长期持续报警，但不是全面优于 V2；其 756 个合格 OHLC 检验没有全局 FDR 发现。当前只能作为短时连续评分候选，不能升级为独立顶底预测器。

## 产物

- [训练数据集](../artifacts/modeling/all_a_ml_dataset_v1_20120705_20260814/)
- [V2 OOS](../artifacts/modeling/all_a_ml_walk_forward_v2_20190102_20260814/)与[评测](../artifacts/evaluations/all_a_ml_walk_forward_v2_20190102_20260814__stage_d_v1/)
- [V3 OOS](../artifacts/modeling/all_a_ml_walk_forward_v3_20190102_20260814/)与[评测](../artifacts/evaluations/all_a_ml_walk_forward_v3_20190102_20260814__stage_d_v1/)

运行入口见项目 [`README.md`](../README.md#常用命令)；输出目录必须为空，既有 bundle 不覆盖。
