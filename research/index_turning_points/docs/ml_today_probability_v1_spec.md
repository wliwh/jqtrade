# 全 A 当日顶底概率 ML V1 冻结规格

状态：本规格在首次生成真实 OOS 结果前冻结。任何标签、输入、模型、切分、校准、阈值或 episode 语义变更必须升级版本，不覆盖本版产物。

## 任务与输出

本版是收盘后当日识别（nowcast），不预测未来 5/10/20 日：

```text
top_probability_today(t)
    = P(t 属于 strict 顶部峰瓣 | 截至 t 日收盘的数据)

bottom_probability_today(t)
    = P(t 属于 strict 底部峰瓣 | 截至 t 日收盘的数据)

top_score(t)    = 100 × top_probability_today(t)
bottom_score(t) = 100 × bottom_probability_today(t)
```

顶部和底部独立建模，不要求概率之和为 1。分数就是单一事件概率的百分制表达，不再合成跨期限紧迫度指数。

## 真值

延续 `top_bottom_regions_v2` 的 strict 峰瓣边界，使用二分类 membership：

- `truth_top_in_strict_lobe`：日期位于任一 strict 顶部峰瓣时为 1，否则为 0；
- `truth_bottom_in_strict_lobe`：日期位于任一 strict 底部峰瓣时为 1，否则为 0。

`truth_top_intensity` / `truth_bottom_intensity` 保留为事后辅助评测：代表极值为 100，峰瓣内按价格距离连续衰减，峰瓣外为 0。强度不参与概率拟合，也不把只有代表最高/最低价的一天设为唯一正样本。

标准答案可以使用未来完整历史；模型日期 `t` 的输入仍只能使用截至 `t` 收盘已知的数据。最新未完成区域可能尚未成熟，因此本版只生成冻结标签快照上的回顾性 OOS，不声称在线标签实时可得。

## 输入

训练对象只包括全 A `000985.XSHG`，一行一个交易日，共享同一组 15 个输入，顶部和底部分别拟合：

```text
breadth_ma20
breadth_ma20_change_5d
breadth_ma60
breadth_ma60_change_10d
new_high_low_net_ratio_60
new_high_low_net_ratio_60_change_5d
limit_hit_net_ratio
limit_hit_net_ratio_change_5d
turnover_ratio_pct_p50
turnover_ratio_pct_p50_change_10d
index_close_to_ma60
index_drawdown_60d
index_rebound_60d
index_return_5d
index_volatility_20d
```

不加入行业维度、离散 `triggered`、未来收益、未来确认状态、事后绘图 phase、120/250 日重复口径或其余 V1 扩展特征。`index_price_available` 与 `target_available` 只作数据质量审计，不进入模型。

## 模型、切分与校准

冻结比较两个简单模型：

- `elastic_net`：沿用既有 Elastic Net 逻辑回归参数；
- `shallow_gbdt`：沿用既有深度 2、60 棵树的 scikit-learn GBDT 参数。

随机种子为 `20260821`。缺失填充、标准化和模型只拟合训练段；类别平衡方式沿用已有模型实现。验证段单独做 sigmoid 校准，单类别验证段使用加一平滑后的常数概率。

年度 expanding walk-forward 保持：首个验证年 2018、首个测试年 2019，训练窗逐年扩张。虽然本版没有未来期限标签，strict membership 仍是未来完整历史生成的事后答案，因此保守保留 20 个交易日的 train→validation 与 validation→test 边界隔离。

## 概率、告警与评测

逐日主输出为 `pred_probability_today`；`pred_score` 与 `raw_value` 均严格等于 `100 × pred_probability_today`。

概率之外另用验证年、每模型、每方向最多 6 个连续 episode 的预算选择阈值。测试年原样沿用阈值；本版不做 V3 的两日告警抑制，使连续高概率日期仍表达模型认为处于同一顶底区域。阈值只派生 `onset/continuation/exit`，不改变概率本身。

概率指标报告 Brier、log loss、ROC AUC、average precision 与正例率；区域定位继续按 strict/loose/window 一对一 episode 规则评测；信号后 OHLC 继续单列，不合成总分。高分日期的事后 intensity 和距离代表极值的天数只作辅助诊断。

## 版本与入口

- dataset：`all_a_ml_today_dataset_v1`；
- training：`all_a_ml_today_walk_forward_v1`。

```bash
/home/hh01/anaconda3/envs/fin/bin/python \
  -m research.index_turning_points.pipelines.build_ml_dataset \
  --target-mode today_strict_lobe_membership \
  --output-dir research/index_turning_points/artifacts/modeling/<dataset_version>

/home/hh01/anaconda3/envs/fin/bin/python \
  -m research.index_turning_points.pipelines.train_ml_today_walk_forward \
  --dataset-dir research/index_turning_points/artifacts/modeling/<dataset_version> \
  --output-dir research/index_turning_points/artifacts/modeling/<training_version>
```

两个入口都要求输出目录不存在或为空，不覆盖既有事实。截止 `2026-08-14` 的结果均为回顾性 OOS；真正前瞻证据只能从本规格冻结后的新增交易日积累。
