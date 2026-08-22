# 顶底区域定位评测

- 评测版本：`stage_d_v1`
- 区域标签：`top_bottom_regions_v2`
- 本报告只评价区域定位，不与信号后价格结果合成总分。

## 口径

- 顶部信号只与顶部区域匹配，底部信号只与底部区域匹配；每个 episode 和区域最多形成一个主匹配。
- `strict` 只认核心峰瓣，`loose` 认核心峰瓣或连续包络，`window` 再纳入冻结的 5/10/20 日窗口。
- `lead_lag_days <= 0` 为预测，正值为确认。预测/确认召回率只使用对应 20 日窗口完整的区域作分母。
- 单峰/多峰由标准答案的 `lobe_count` 决定。误报为便于切片，会关联覆盖期内最近的同向区域；该诊断关联不会改变误报状态或主匹配。
- `all_indices` 以“指数×区域”和“指数×episode”对汇总；指数间相关，不能把这些对当成独立统计样本。

## 跨指数总览

| signal_id | direction | version | event_kind | match_scope | region_count | matched_region_count | region_recall | episode_count | matched_episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ma20_episode_ml_l2_logistic | bottom | all_a_ma20_episode_match_walk_forward_v1 | capped_confirmation | loose | 109 | 13 | 0.119266 | 140 | 13 | 0.0928571 | 127 | 8 | 0 |
| ma20_episode_ml_l2_logistic | bottom | all_a_ma20_episode_match_walk_forward_v1 | capped_confirmation | strict | 109 | 7 | 0.0642202 | 140 | 7 | 0.05 | 133 | 8 | 0 |
| ma20_episode_ml_l2_logistic | bottom | all_a_ma20_episode_match_walk_forward_v1 | capped_confirmation | window | 109 | 52 | 0.477064 | 140 | 52 | 0.371429 | 88 | 8 | 4 |
| ma20_episode_ml_l2_logistic | bottom | all_a_ma20_episode_match_walk_forward_v1 | onset | loose | 109 | 11 | 0.100917 | 140 | 11 | 0.0785714 | 129 | 8 | 1 |
| ma20_episode_ml_l2_logistic | bottom | all_a_ma20_episode_match_walk_forward_v1 | onset | strict | 109 | 7 | 0.0642202 | 140 | 7 | 0.05 | 133 | 8 | -1 |
| ma20_episode_ml_l2_logistic | bottom | all_a_ma20_episode_match_walk_forward_v1 | onset | window | 109 | 52 | 0.477064 | 140 | 52 | 0.371429 | 88 | 8 | 3 |
| ma20_episode_ml_l2_logistic | top | all_a_ma20_episode_match_walk_forward_v1 | capped_confirmation | loose | 105 | 7 | 0.0666667 | 14 | 7 | 0.5 | 7 | 0 | 2 |
| ma20_episode_ml_l2_logistic | top | all_a_ma20_episode_match_walk_forward_v1 | capped_confirmation | strict | 105 | 7 | 0.0666667 | 14 | 7 | 0.5 | 7 | 0 | 2 |
| ma20_episode_ml_l2_logistic | top | all_a_ma20_episode_match_walk_forward_v1 | capped_confirmation | window | 105 | 14 | 0.133333 | 14 | 14 | 1 | 0 | 0 | 2 |
| ma20_episode_ml_l2_logistic | top | all_a_ma20_episode_match_walk_forward_v1 | onset | loose | 105 | 8 | 0.0761905 | 14 | 8 | 0.571429 | 6 | 0 | 1 |
| ma20_episode_ml_l2_logistic | top | all_a_ma20_episode_match_walk_forward_v1 | onset | strict | 105 | 8 | 0.0761905 | 14 | 8 | 0.571429 | 6 | 0 | 1 |
| ma20_episode_ml_l2_logistic | top | all_a_ma20_episode_match_walk_forward_v1 | onset | window | 105 | 14 | 0.133333 | 14 | 14 | 1 | 0 | 0 | 1 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 15 | 0.466667 | 20 | 0.35 | 13 | 1 | 3 |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 16 | 0.5 | 20 | 0.4 | 12 | 1 | 3.5 |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 14 | 0.5 | 20 | 0.35 | 13 | 1 | 3 |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 15 | 0.466667 | 20 | 0.35 | 13 | 1 | 5 |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 16 | 0.5 | 20 | 0.4 | 12 | 1 | 3.5 |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 15 | 0.533333 | 20 | 0.4 | 12 | 2 | 4.5 |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 18 | 0.388889 | 20 | 0.35 | 13 | 1 | 5 |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 15 | 0.466667 | 20 | 0.35 | 13 | 1 | 2 |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 16 | 0.5 | 20 | 0.4 | 12 | 1 | 2.5 |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 14 | 0.5 | 20 | 0.35 | 13 | 1 | 2 |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 15 | 0.466667 | 20 | 0.35 | 13 | 1 | 4 |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 16 | 0.5 | 20 | 0.4 | 12 | 1 | 2.5 |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 15 | 0.533333 | 20 | 0.4 | 12 | 2 | 3.5 |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 18 | 0.388889 | 20 | 0.35 | 13 | 1 | 4 |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 14 | 0.142857 | 2 | 1 | 0 | 0 | 2.5 |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 15 | 0.133333 | 2 | 1 | 0 | 0 | -8.5 |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 13 | 0.153846 | 2 | 1 | 0 | 0 | 0.5 |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 15 | 0.133333 | 2 | 1 | 0 | 0 | 3.5 |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 15 | 0.133333 | 2 | 1 | 0 | 0 | -0.5 |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 15 | 0.133333 | 2 | 1 | 0 | 0 | -8.5 |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 18 | 0.111111 | 2 | 1 | 0 | 0 | 3.5 |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 14 | 0.142857 | 2 | 1 | 0 | 0 | 1.5 |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 15 | 0.133333 | 2 | 1 | 0 | 0 | -9.5 |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 13 | 0.153846 | 2 | 1 | 0 | 0 | -0.5 |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 15 | 0.133333 | 2 | 1 | 0 | 0 | 2.5 |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 15 | 0.133333 | 2 | 1 | 0 | 0 | -1.5 |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 15 | 0.133333 | 2 | 1 | 0 | 0 | -9.5 |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 18 | 0.111111 | 2 | 1 | 0 | 0 | 2.5 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 16 |
| false_alarm | 160 |
| matched | 132 |
| missed_region | 296 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `ma20_episode_ml_l2_logistic/bottom/all_a_ma20_episode_match_walk_forward_v1/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 8 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 7/13/52。
- `ma20_episode_ml_l2_logistic/bottom/all_a_ma20_episode_match_walk_forward_v1/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 8 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 7/11/52。
- `ma20_episode_ml_l2_logistic/top/all_a_ma20_episode_match_walk_forward_v1/capped_confirmation`：区域命中对口径敏感：strict/loose/window 分别为 7/7/14。
- `ma20_episode_ml_l2_logistic/top/all_a_ma20_episode_match_walk_forward_v1/onset`：区域命中对口径敏感：strict/loose/window 分别为 8/8/14。
