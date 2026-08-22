# 顶底区域定位评测

- 评测版本：`all_a_ml_walk_forward_v3_20190102_20260814__stage_d_v1`
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
| ml_elastic_net | bottom | all_a_ml_walk_forward_v3 | capped_confirmation | loose | 109 | 16 | 0.146789 | 147 | 16 | 0.108844 | 131 | 29 | 0 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v3 | capped_confirmation | strict | 109 | 12 | 0.110092 | 147 | 12 | 0.0816327 | 135 | 29 | 0 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v3 | capped_confirmation | window | 109 | 53 | 0.486239 | 147 | 53 | 0.360544 | 94 | 29 | -2 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v3 | onset | loose | 109 | 14 | 0.12844 | 147 | 14 | 0.0952381 | 133 | 26 | -1 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v3 | onset | strict | 109 | 13 | 0.119266 | 147 | 13 | 0.0884354 | 134 | 26 | -1 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v3 | onset | window | 109 | 53 | 0.486239 | 147 | 53 | 0.360544 | 94 | 26 | -3 |
| ml_elastic_net | top | all_a_ml_walk_forward_v3 | capped_confirmation | loose | 105 | 21 | 0.2 | 168 | 21 | 0.125 | 147 | 26 | 0 |
| ml_elastic_net | top | all_a_ml_walk_forward_v3 | capped_confirmation | strict | 105 | 21 | 0.2 | 168 | 21 | 0.125 | 147 | 26 | 0 |
| ml_elastic_net | top | all_a_ml_walk_forward_v3 | capped_confirmation | window | 105 | 44 | 0.419048 | 168 | 44 | 0.261905 | 124 | 26 | -2 |
| ml_elastic_net | top | all_a_ml_walk_forward_v3 | onset | loose | 105 | 21 | 0.2 | 168 | 21 | 0.125 | 147 | 29 | -1 |
| ml_elastic_net | top | all_a_ml_walk_forward_v3 | onset | strict | 105 | 21 | 0.2 | 168 | 21 | 0.125 | 147 | 29 | -1 |
| ml_elastic_net | top | all_a_ml_walk_forward_v3 | onset | window | 105 | 41 | 0.390476 | 168 | 41 | 0.244048 | 127 | 29 | -3 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v3 | capped_confirmation | loose | 109 | 32 | 0.293578 | 210 | 32 | 0.152381 | 178 | 61 | 0 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v3 | capped_confirmation | strict | 109 | 29 | 0.266055 | 210 | 29 | 0.138095 | 181 | 61 | 0 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v3 | capped_confirmation | window | 109 | 61 | 0.559633 | 210 | 61 | 0.290476 | 149 | 61 | -1 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v3 | onset | loose | 109 | 22 | 0.201835 | 210 | 22 | 0.104762 | 188 | 61 | 0 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v3 | onset | strict | 109 | 18 | 0.165138 | 210 | 18 | 0.0857143 | 192 | 61 | 0 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v3 | onset | window | 109 | 60 | 0.550459 | 210 | 60 | 0.285714 | 150 | 61 | -2 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v3 | capped_confirmation | loose | 105 | 37 | 0.352381 | 203 | 37 | 0.182266 | 166 | 22 | 0 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v3 | capped_confirmation | strict | 105 | 34 | 0.32381 | 203 | 34 | 0.167488 | 169 | 22 | 0 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v3 | capped_confirmation | window | 105 | 50 | 0.47619 | 203 | 50 | 0.246305 | 153 | 22 | -1 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v3 | onset | loose | 105 | 30 | 0.285714 | 203 | 30 | 0.147783 | 173 | 22 | -1.5 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v3 | onset | strict | 105 | 29 | 0.27619 | 203 | 29 | 0.142857 | 174 | 22 | -1 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v3 | onset | window | 105 | 50 | 0.47619 | 203 | 50 | 0.246305 | 153 | 22 | -2 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v3 | capped_confirmation | loose | 109 | 17 | 0.155963 | 273 | 17 | 0.0622711 | 256 | 63 | 0 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v3 | capped_confirmation | strict | 109 | 15 | 0.137615 | 273 | 15 | 0.0549451 | 258 | 63 | 0 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v3 | capped_confirmation | window | 109 | 60 | 0.550459 | 273 | 60 | 0.21978 | 213 | 63 | 3 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v3 | onset | loose | 109 | 13 | 0.119266 | 273 | 13 | 0.047619 | 260 | 63 | 0 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v3 | onset | strict | 109 | 12 | 0.110092 | 273 | 12 | 0.043956 | 261 | 63 | 0 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v3 | onset | window | 109 | 60 | 0.550459 | 273 | 60 | 0.21978 | 213 | 63 | 2 |
| ml_simple_rule | top | all_a_ml_walk_forward_v3 | capped_confirmation | loose | 105 | 18 | 0.171429 | 357 | 18 | 0.0504202 | 339 | 57 | 1 |
| ml_simple_rule | top | all_a_ml_walk_forward_v3 | capped_confirmation | strict | 105 | 16 | 0.152381 | 357 | 16 | 0.0448179 | 341 | 57 | 1.5 |
| ml_simple_rule | top | all_a_ml_walk_forward_v3 | capped_confirmation | window | 105 | 61 | 0.580952 | 357 | 61 | 0.170868 | 296 | 57 | 1 |
| ml_simple_rule | top | all_a_ml_walk_forward_v3 | onset | loose | 105 | 22 | 0.209524 | 357 | 22 | 0.0616246 | 335 | 51 | 1 |
| ml_simple_rule | top | all_a_ml_walk_forward_v3 | onset | strict | 105 | 21 | 0.2 | 357 | 21 | 0.0588235 | 336 | 51 | 1 |
| ml_simple_rule | top | all_a_ml_walk_forward_v3 | onset | window | 105 | 59 | 0.561905 | 357 | 59 | 0.165266 | 298 | 51 | 1 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_elastic_net | bottom | capped_confirmation | 全A | 15 | 0.533333 | 21 | 0.380952 | 13 | 3 | -4 |
| ml_elastic_net | bottom | capped_confirmation | 国证2000 | 16 | 0.4375 | 21 | 0.333333 | 14 | 5 | -3 |
| ml_elastic_net | bottom | capped_confirmation | 中证1000 | 14 | 0.5 | 21 | 0.333333 | 14 | 5 | -3 |
| ml_elastic_net | bottom | capped_confirmation | 沪深300 | 15 | 0.466667 | 21 | 0.333333 | 14 | 3 | 0 |
| ml_elastic_net | bottom | capped_confirmation | 中证500 | 16 | 0.5 | 21 | 0.380952 | 13 | 5 | -2.5 |
| ml_elastic_net | bottom | capped_confirmation | 微盘股 | 15 | 0.533333 | 21 | 0.380952 | 13 | 5 | -1 |
| ml_elastic_net | bottom | capped_confirmation | 上证指数 | 18 | 0.444444 | 21 | 0.380952 | 13 | 3 | -2 |
| ml_elastic_net | bottom | onset | 全A | 15 | 0.533333 | 21 | 0.380952 | 13 | 2 | -5 |
| ml_elastic_net | bottom | onset | 国证2000 | 16 | 0.4375 | 21 | 0.333333 | 14 | 5 | -4 |
| ml_elastic_net | bottom | onset | 中证1000 | 14 | 0.5 | 21 | 0.333333 | 14 | 5 | -4 |
| ml_elastic_net | bottom | onset | 沪深300 | 15 | 0.466667 | 21 | 0.333333 | 14 | 2 | -1 |
| ml_elastic_net | bottom | onset | 中证500 | 16 | 0.5 | 21 | 0.380952 | 13 | 5 | -3.5 |
| ml_elastic_net | bottom | onset | 微盘股 | 15 | 0.533333 | 21 | 0.380952 | 13 | 5 | -2 |
| ml_elastic_net | bottom | onset | 上证指数 | 18 | 0.444444 | 21 | 0.380952 | 13 | 2 | -3 |
| ml_elastic_net | top | capped_confirmation | 全A | 14 | 0.5 | 24 | 0.291667 | 17 | 4 | -2 |
| ml_elastic_net | top | capped_confirmation | 国证2000 | 15 | 0.466667 | 24 | 0.291667 | 17 | 5 | -2 |
| ml_elastic_net | top | capped_confirmation | 中证1000 | 13 | 0.461538 | 24 | 0.25 | 18 | 3 | -1.5 |
| ml_elastic_net | top | capped_confirmation | 沪深300 | 15 | 0.4 | 24 | 0.25 | 18 | 4 | -0.5 |
| ml_elastic_net | top | capped_confirmation | 中证500 | 15 | 0.333333 | 24 | 0.208333 | 19 | 3 | -2 |
| ml_elastic_net | top | capped_confirmation | 微盘股 | 15 | 0.466667 | 24 | 0.291667 | 17 | 4 | -2 |
| ml_elastic_net | top | capped_confirmation | 上证指数 | 18 | 0.333333 | 24 | 0.25 | 18 | 3 | -1.5 |
| ml_elastic_net | top | onset | 全A | 14 | 0.428571 | 24 | 0.25 | 18 | 5 | -2.5 |
| ml_elastic_net | top | onset | 国证2000 | 15 | 0.466667 | 24 | 0.291667 | 17 | 5 | -3 |
| ml_elastic_net | top | onset | 中证1000 | 13 | 0.384615 | 24 | 0.208333 | 19 | 4 | -2 |
| ml_elastic_net | top | onset | 沪深300 | 15 | 0.4 | 24 | 0.25 | 18 | 4 | -1.5 |
| ml_elastic_net | top | onset | 中证500 | 15 | 0.266667 | 24 | 0.166667 | 20 | 4 | -2 |
| ml_elastic_net | top | onset | 微盘股 | 15 | 0.466667 | 24 | 0.291667 | 17 | 4 | -3 |
| ml_elastic_net | top | onset | 上证指数 | 18 | 0.333333 | 24 | 0.25 | 18 | 3 | -2.5 |
| ml_shallow_xgboost | bottom | capped_confirmation | 全A | 15 | 0.6 | 30 | 0.3 | 21 | 9 | -1 |
| ml_shallow_xgboost | bottom | capped_confirmation | 国证2000 | 16 | 0.5 | 30 | 0.266667 | 22 | 8 | -0.5 |
| ml_shallow_xgboost | bottom | capped_confirmation | 中证1000 | 14 | 0.642857 | 30 | 0.3 | 21 | 8 | -1 |
| ml_shallow_xgboost | bottom | capped_confirmation | 沪深300 | 15 | 0.533333 | 30 | 0.266667 | 22 | 8 | 0 |
| ml_shallow_xgboost | bottom | capped_confirmation | 中证500 | 16 | 0.625 | 30 | 0.333333 | 20 | 11 | -2 |
| ml_shallow_xgboost | bottom | capped_confirmation | 微盘股 | 15 | 0.533333 | 30 | 0.266667 | 22 | 8 | -0.5 |
| ml_shallow_xgboost | bottom | capped_confirmation | 上证指数 | 18 | 0.5 | 30 | 0.3 | 21 | 9 | 0 |
| ml_shallow_xgboost | bottom | onset | 全A | 15 | 0.6 | 30 | 0.3 | 21 | 9 | -2 |
| ml_shallow_xgboost | bottom | onset | 国证2000 | 16 | 0.5 | 30 | 0.266667 | 22 | 8 | -1.5 |
| ml_shallow_xgboost | bottom | onset | 中证1000 | 14 | 0.642857 | 30 | 0.3 | 21 | 8 | -2 |
| ml_shallow_xgboost | bottom | onset | 沪深300 | 15 | 0.533333 | 30 | 0.266667 | 22 | 8 | -1 |
| ml_shallow_xgboost | bottom | onset | 中证500 | 16 | 0.5625 | 30 | 0.3 | 21 | 11 | -2 |
| ml_shallow_xgboost | bottom | onset | 微盘股 | 15 | 0.533333 | 30 | 0.266667 | 22 | 8 | -1.5 |
| ml_shallow_xgboost | bottom | onset | 上证指数 | 18 | 0.5 | 30 | 0.3 | 21 | 9 | -1 |
| ml_shallow_xgboost | top | capped_confirmation | 全A | 14 | 0.5 | 29 | 0.241379 | 22 | 3 | -1 |
| ml_shallow_xgboost | top | capped_confirmation | 国证2000 | 15 | 0.466667 | 29 | 0.241379 | 22 | 7 | 0 |
| ml_shallow_xgboost | top | capped_confirmation | 中证1000 | 13 | 0.384615 | 29 | 0.172414 | 24 | 2 | 0 |
| ml_shallow_xgboost | top | capped_confirmation | 沪深300 | 15 | 0.533333 | 29 | 0.275862 | 21 | 4 | -2.5 |
| ml_shallow_xgboost | top | capped_confirmation | 中证500 | 15 | 0.466667 | 29 | 0.241379 | 22 | 2 | 0 |
| ml_shallow_xgboost | top | capped_confirmation | 微盘股 | 15 | 0.466667 | 29 | 0.241379 | 22 | 2 | 0 |
| ml_shallow_xgboost | top | capped_confirmation | 上证指数 | 18 | 0.5 | 29 | 0.310345 | 20 | 2 | -6 |
| ml_shallow_xgboost | top | onset | 全A | 14 | 0.5 | 29 | 0.241379 | 22 | 3 | -2 |
| ml_shallow_xgboost | top | onset | 国证2000 | 15 | 0.466667 | 29 | 0.241379 | 22 | 7 | -1 |
| ml_shallow_xgboost | top | onset | 中证1000 | 13 | 0.384615 | 29 | 0.172414 | 24 | 2 | -1 |
| ml_shallow_xgboost | top | onset | 沪深300 | 15 | 0.533333 | 29 | 0.275862 | 21 | 4 | -3.5 |
| ml_shallow_xgboost | top | onset | 中证500 | 15 | 0.466667 | 29 | 0.241379 | 22 | 2 | -1 |
| ml_shallow_xgboost | top | onset | 微盘股 | 15 | 0.466667 | 29 | 0.241379 | 22 | 2 | -1 |
| ml_shallow_xgboost | top | onset | 上证指数 | 18 | 0.5 | 29 | 0.310345 | 20 | 2 | -7 |
| ml_simple_rule | bottom | capped_confirmation | 全A | 15 | 0.533333 | 39 | 0.205128 | 31 | 9 | 2 |
| ml_simple_rule | bottom | capped_confirmation | 国证2000 | 16 | 0.5625 | 39 | 0.230769 | 30 | 8 | 3 |
| ml_simple_rule | bottom | capped_confirmation | 中证1000 | 14 | 0.571429 | 39 | 0.205128 | 31 | 8 | 3 |
| ml_simple_rule | bottom | capped_confirmation | 沪深300 | 15 | 0.533333 | 39 | 0.205128 | 31 | 8 | 4 |
| ml_simple_rule | bottom | capped_confirmation | 中证500 | 16 | 0.5625 | 39 | 0.230769 | 30 | 10 | 3 |
| ml_simple_rule | bottom | capped_confirmation | 微盘股 | 15 | 0.533333 | 39 | 0.205128 | 31 | 9 | 2 |
| ml_simple_rule | bottom | capped_confirmation | 上证指数 | 18 | 0.555556 | 39 | 0.25641 | 29 | 11 | 4 |
| ml_simple_rule | bottom | onset | 全A | 15 | 0.533333 | 39 | 0.205128 | 31 | 9 | 1 |
| ml_simple_rule | bottom | onset | 国证2000 | 16 | 0.5625 | 39 | 0.230769 | 30 | 8 | 2 |
| ml_simple_rule | bottom | onset | 中证1000 | 14 | 0.571429 | 39 | 0.205128 | 31 | 8 | 2 |
| ml_simple_rule | bottom | onset | 沪深300 | 15 | 0.533333 | 39 | 0.205128 | 31 | 8 | 4 |
| ml_simple_rule | bottom | onset | 中证500 | 16 | 0.5625 | 39 | 0.230769 | 30 | 10 | 2 |
| ml_simple_rule | bottom | onset | 微盘股 | 15 | 0.533333 | 39 | 0.205128 | 31 | 9 | 1 |
| ml_simple_rule | bottom | onset | 上证指数 | 18 | 0.555556 | 39 | 0.25641 | 29 | 11 | 3 |
| ml_simple_rule | top | capped_confirmation | 全A | 14 | 0.571429 | 51 | 0.156863 | 43 | 6 | -2.5 |
| ml_simple_rule | top | capped_confirmation | 国证2000 | 15 | 0.466667 | 51 | 0.137255 | 44 | 9 | 5 |
| ml_simple_rule | top | capped_confirmation | 中证1000 | 13 | 0.538462 | 51 | 0.137255 | 44 | 4 | -8 |
| ml_simple_rule | top | capped_confirmation | 沪深300 | 15 | 0.666667 | 51 | 0.196078 | 41 | 8 | 1.5 |
| ml_simple_rule | top | capped_confirmation | 中证500 | 15 | 0.6 | 51 | 0.176471 | 42 | 9 | 2 |
| ml_simple_rule | top | capped_confirmation | 微盘股 | 15 | 0.6 | 51 | 0.176471 | 42 | 12 | -2 |
| ml_simple_rule | top | capped_confirmation | 上证指数 | 18 | 0.611111 | 51 | 0.215686 | 40 | 9 | -3 |
| ml_simple_rule | top | onset | 全A | 14 | 0.5 | 51 | 0.137255 | 44 | 5 | 0 |
| ml_simple_rule | top | onset | 国证2000 | 15 | 0.466667 | 51 | 0.137255 | 44 | 7 | 4 |
| ml_simple_rule | top | onset | 中证1000 | 13 | 0.538462 | 51 | 0.137255 | 44 | 4 | -9 |
| ml_simple_rule | top | onset | 沪深300 | 15 | 0.6 | 51 | 0.176471 | 42 | 7 | 1 |
| ml_simple_rule | top | onset | 中证500 | 15 | 0.6 | 51 | 0.176471 | 42 | 8 | 1 |
| ml_simple_rule | top | onset | 微盘股 | 15 | 0.6 | 51 | 0.176471 | 42 | 11 | 1 |
| ml_simple_rule | top | onset | 上证指数 | 18 | 0.611111 | 51 | 0.215686 | 40 | 9 | 0 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 510 |
| false_alarm | 1554 |
| matched | 652 |
| missed_region | 632 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `ml_elastic_net/bottom/all_a_ml_walk_forward_v3/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 29 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 12/16/53。
- `ml_elastic_net/bottom/all_a_ml_walk_forward_v3/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 26 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 13/14/53。
- `ml_elastic_net/top/all_a_ml_walk_forward_v3/capped_confirmation`：存在 26 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 21/21/44。
- `ml_elastic_net/top/all_a_ml_walk_forward_v3/onset`：存在 29 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 21/21/41。
- `ml_shallow_xgboost/bottom/all_a_ml_walk_forward_v3/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 61 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 29/32/61。
- `ml_shallow_xgboost/bottom/all_a_ml_walk_forward_v3/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 61 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 18/22/60。
- `ml_shallow_xgboost/top/all_a_ml_walk_forward_v3/capped_confirmation`：存在 22 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 34/37/50。
- `ml_shallow_xgboost/top/all_a_ml_walk_forward_v3/onset`：存在 22 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 29/30/50。
- `ml_simple_rule/bottom/all_a_ml_walk_forward_v3/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 63 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 15/17/60。
- `ml_simple_rule/bottom/all_a_ml_walk_forward_v3/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 63 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 12/13/60。
- `ml_simple_rule/top/all_a_ml_walk_forward_v3/capped_confirmation`：存在 57 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 16/18/61。
- `ml_simple_rule/top/all_a_ml_walk_forward_v3/onset`：存在 51 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 21/22/59。
