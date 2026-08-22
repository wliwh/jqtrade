# 顶底区域定位评测

- 评测版本：`all_a_ml_today_walk_forward_v2_stage_d_v1`
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
| ml_today_calibrated_elastic_net | bottom | all_a_ml_today_walk_forward_v2 | capped_confirmation | loose | 109 | 1 | 0.00917431 | 21 | 1 | 0.047619 | 20 | 0 | -2 |
| ml_today_calibrated_elastic_net | bottom | all_a_ml_today_walk_forward_v2 | capped_confirmation | strict | 109 | 1 | 0.00917431 | 21 | 1 | 0.047619 | 20 | 0 | -2 |
| ml_today_calibrated_elastic_net | bottom | all_a_ml_today_walk_forward_v2 | capped_confirmation | window | 109 | 13 | 0.119266 | 21 | 13 | 0.619048 | 8 | 0 | -2 |
| ml_today_calibrated_elastic_net | bottom | all_a_ml_today_walk_forward_v2 | onset | loose | 109 | 0 | 0 | 21 | 0 | 0 | 21 | 0 |  |
| ml_today_calibrated_elastic_net | bottom | all_a_ml_today_walk_forward_v2 | onset | strict | 109 | 0 | 0 | 21 | 0 | 0 | 21 | 0 |  |
| ml_today_calibrated_elastic_net | bottom | all_a_ml_today_walk_forward_v2 | onset | window | 109 | 13 | 0.119266 | 21 | 13 | 0.619048 | 8 | 0 | -3 |
| ml_today_calibrated_elastic_net | top | all_a_ml_today_walk_forward_v2 | capped_confirmation | loose | 105 | 10 | 0.0952381 | 56 | 10 | 0.178571 | 46 | 5 | -2 |
| ml_today_calibrated_elastic_net | top | all_a_ml_today_walk_forward_v2 | capped_confirmation | strict | 105 | 10 | 0.0952381 | 56 | 10 | 0.178571 | 46 | 5 | -2 |
| ml_today_calibrated_elastic_net | top | all_a_ml_today_walk_forward_v2 | capped_confirmation | window | 105 | 25 | 0.238095 | 56 | 25 | 0.446429 | 31 | 5 | -2 |
| ml_today_calibrated_elastic_net | top | all_a_ml_today_walk_forward_v2 | onset | loose | 105 | 7 | 0.0666667 | 56 | 7 | 0.125 | 49 | 5 | -7 |
| ml_today_calibrated_elastic_net | top | all_a_ml_today_walk_forward_v2 | onset | strict | 105 | 7 | 0.0666667 | 56 | 7 | 0.125 | 49 | 5 | -7 |
| ml_today_calibrated_elastic_net | top | all_a_ml_today_walk_forward_v2 | onset | window | 105 | 25 | 0.238095 | 56 | 25 | 0.446429 | 31 | 5 | -3 |
| ml_today_calibrated_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v2 | capped_confirmation | loose | 109 | 0 | 0 | 35 | 0 | 0 | 35 | 0 |  |
| ml_today_calibrated_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v2 | capped_confirmation | strict | 109 | 0 | 0 | 35 | 0 | 0 | 35 | 0 |  |
| ml_today_calibrated_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v2 | capped_confirmation | window | 109 | 16 | 0.146789 | 35 | 16 | 0.457143 | 19 | 0 | -11 |
| ml_today_calibrated_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v2 | onset | loose | 109 | 0 | 0 | 35 | 0 | 0 | 35 | 0 |  |
| ml_today_calibrated_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v2 | onset | strict | 109 | 0 | 0 | 35 | 0 | 0 | 35 | 0 |  |
| ml_today_calibrated_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v2 | onset | window | 109 | 16 | 0.146789 | 35 | 16 | 0.457143 | 19 | 0 | -12 |
| ml_today_calibrated_shallow_gbdt | top | all_a_ml_today_walk_forward_v2 | capped_confirmation | loose | 105 | 0 | 0 | 7 | 0 | 0 | 7 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | all_a_ml_today_walk_forward_v2 | capped_confirmation | strict | 105 | 0 | 0 | 7 | 0 | 0 | 7 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | all_a_ml_today_walk_forward_v2 | capped_confirmation | window | 105 | 0 | 0 | 7 | 0 | 0 | 7 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | all_a_ml_today_walk_forward_v2 | onset | loose | 105 | 0 | 0 | 7 | 0 | 0 | 7 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | all_a_ml_today_walk_forward_v2 | onset | strict | 105 | 0 | 0 | 7 | 0 | 0 | 7 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | all_a_ml_today_walk_forward_v2 | onset | window | 105 | 0 | 0 | 7 | 0 | 0 | 7 | 0 |  |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 15 | 0.0666667 | 3 | 0.333333 | 2 | 0 | -2 |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 16 | 0.125 | 3 | 0.666667 | 1 | 0 | -6 |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 14 | 0.142857 | 3 | 0.666667 | 1 | 0 | -6.5 |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 15 | 0.133333 | 3 | 0.666667 | 1 | 0 | -3 |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 16 | 0.125 | 3 | 0.666667 | 1 | 0 | -6.5 |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 15 | 0.133333 | 3 | 0.666667 | 1 | 0 | -6 |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 18 | 0.111111 | 3 | 0.666667 | 1 | 0 | -3 |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 15 | 0.0666667 | 3 | 0.333333 | 2 | 0 | -3 |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 16 | 0.125 | 3 | 0.666667 | 1 | 0 | -7 |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 14 | 0.142857 | 3 | 0.666667 | 1 | 0 | -7.5 |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 15 | 0.133333 | 3 | 0.666667 | 1 | 0 | -4 |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 16 | 0.125 | 3 | 0.666667 | 1 | 0 | -7.5 |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 15 | 0.133333 | 3 | 0.666667 | 1 | 0 | -7 |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 18 | 0.111111 | 3 | 0.666667 | 1 | 0 | -4 |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 14 | 0.214286 | 8 | 0.375 | 5 | 1 | -6 |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 15 | 0.266667 | 8 | 0.5 | 4 | 1 | -1.5 |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 13 | 0.230769 | 8 | 0.375 | 5 | 1 | -7 |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 15 | 0.266667 | 8 | 0.5 | 4 | 0 | -3.5 |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 15 | 0.2 | 8 | 0.375 | 5 | 1 | -1 |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 15 | 0.266667 | 8 | 0.5 | 4 | 1 | -1.5 |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 18 | 0.222222 | 8 | 0.5 | 4 | 0 | -3.5 |
| ml_today_calibrated_elastic_net | top | onset | 全A | 14 | 0.214286 | 8 | 0.375 | 5 | 1 | -7 |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 15 | 0.266667 | 8 | 0.5 | 4 | 1 | -2.5 |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 13 | 0.230769 | 8 | 0.375 | 5 | 1 | -8 |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 15 | 0.266667 | 8 | 0.5 | 4 | 0 | -4.5 |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 15 | 0.2 | 8 | 0.375 | 5 | 1 | -2 |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 15 | 0.266667 | 8 | 0.5 | 4 | 1 | -2.5 |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 18 | 0.222222 | 8 | 0.5 | 4 | 0 | -4.5 |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 15 | 0.133333 | 5 | 0.4 | 3 | 0 | -13.5 |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 16 | 0.125 | 5 | 0.4 | 3 | 0 | -6.5 |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 14 | 0.142857 | 5 | 0.4 | 3 | 0 | -7 |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 15 | 0.2 | 5 | 0.6 | 2 | 0 | -11 |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 16 | 0.125 | 5 | 0.4 | 3 | 0 | -7 |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 15 | 0.133333 | 5 | 0.4 | 3 | 0 | -6.5 |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 18 | 0.166667 | 5 | 0.6 | 2 | 0 | -11 |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 15 | 0.133333 | 5 | 0.4 | 3 | 0 | -14.5 |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 16 | 0.125 | 5 | 0.4 | 3 | 0 | -7.5 |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 14 | 0.142857 | 5 | 0.4 | 3 | 0 | -8 |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 15 | 0.2 | 5 | 0.6 | 2 | 0 | -12 |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 16 | 0.125 | 5 | 0.4 | 3 | 0 | -8 |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 15 | 0.133333 | 5 | 0.4 | 3 | 0 | -7.5 |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 18 | 0.166667 | 5 | 0.6 | 2 | 0 | -12 |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 14 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 15 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 13 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 15 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 15 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 15 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 18 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 14 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 15 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 13 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 15 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 15 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 15 | 0 | 1 | 0 | 1 | 0 |  |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 18 | 0 | 1 | 0 | 1 | 0 |  |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 10 |
| false_alarm | 120 |
| matched | 108 |
| missed_region | 748 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `ml_today_calibrated_elastic_net/bottom/all_a_ml_today_walk_forward_v2/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 区域命中对口径敏感：strict/loose/window 分别为 1/1/13。
- `ml_today_calibrated_elastic_net/bottom/all_a_ml_today_walk_forward_v2/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 区域命中对口径敏感：strict/loose/window 分别为 0/0/13。
- `ml_today_calibrated_elastic_net/top/all_a_ml_today_walk_forward_v2/capped_confirmation`：存在 5 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 10/10/25。
- `ml_today_calibrated_elastic_net/top/all_a_ml_today_walk_forward_v2/onset`：存在 5 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 7/7/25。
- `ml_today_calibrated_shallow_gbdt/bottom/all_a_ml_today_walk_forward_v2/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 区域命中对口径敏感：strict/loose/window 分别为 0/0/16。
- `ml_today_calibrated_shallow_gbdt/bottom/all_a_ml_today_walk_forward_v2/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 区域命中对口径敏感：strict/loose/window 分别为 0/0/16。
