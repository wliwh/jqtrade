# 顶底区域定位评测

- 评测版本：`all_a_ml_walk_forward_v2_20190102_20260814__stage_d_v1`
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
| ml_elastic_net | bottom | all_a_ml_walk_forward_v2 | capped_confirmation | loose | 109 | 17 | 0.155963 | 203 | 17 | 0.0837438 | 186 | 53 | 0 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v2 | capped_confirmation | strict | 109 | 15 | 0.137615 | 203 | 15 | 0.0738916 | 188 | 53 | 1 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v2 | capped_confirmation | window | 109 | 42 | 0.385321 | 203 | 42 | 0.206897 | 161 | 53 | -2 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v2 | onset | loose | 109 | 18 | 0.165138 | 203 | 18 | 0.08867 | 185 | 50 | -0.5 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v2 | onset | strict | 109 | 18 | 0.165138 | 203 | 18 | 0.08867 | 185 | 50 | -0.5 |
| ml_elastic_net | bottom | all_a_ml_walk_forward_v2 | onset | window | 109 | 42 | 0.385321 | 203 | 42 | 0.206897 | 161 | 50 | -3 |
| ml_elastic_net | top | all_a_ml_walk_forward_v2 | capped_confirmation | loose | 105 | 17 | 0.161905 | 210 | 17 | 0.0809524 | 193 | 38 | -2 |
| ml_elastic_net | top | all_a_ml_walk_forward_v2 | capped_confirmation | strict | 105 | 17 | 0.161905 | 210 | 17 | 0.0809524 | 193 | 38 | -2 |
| ml_elastic_net | top | all_a_ml_walk_forward_v2 | capped_confirmation | window | 105 | 41 | 0.390476 | 210 | 41 | 0.195238 | 169 | 38 | -3 |
| ml_elastic_net | top | all_a_ml_walk_forward_v2 | onset | loose | 105 | 16 | 0.152381 | 210 | 16 | 0.0761905 | 194 | 41 | -3 |
| ml_elastic_net | top | all_a_ml_walk_forward_v2 | onset | strict | 105 | 16 | 0.152381 | 210 | 16 | 0.0761905 | 194 | 41 | -3 |
| ml_elastic_net | top | all_a_ml_walk_forward_v2 | onset | window | 105 | 38 | 0.361905 | 210 | 38 | 0.180952 | 172 | 41 | -4 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v2 | capped_confirmation | loose | 109 | 21 | 0.192661 | 217 | 21 | 0.0967742 | 196 | 48 | 0 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v2 | capped_confirmation | strict | 109 | 18 | 0.165138 | 217 | 18 | 0.0829493 | 199 | 48 | 0 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v2 | capped_confirmation | window | 109 | 62 | 0.568807 | 217 | 62 | 0.285714 | 155 | 48 | -1 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v2 | onset | loose | 109 | 15 | 0.137615 | 217 | 15 | 0.0691244 | 202 | 48 | -1 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v2 | onset | strict | 109 | 11 | 0.100917 | 217 | 11 | 0.0506912 | 206 | 48 | 0 |
| ml_shallow_xgboost | bottom | all_a_ml_walk_forward_v2 | onset | window | 109 | 61 | 0.559633 | 217 | 61 | 0.281106 | 156 | 48 | -2 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v2 | capped_confirmation | loose | 105 | 33 | 0.314286 | 217 | 33 | 0.152074 | 184 | 50 | 0 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v2 | capped_confirmation | strict | 105 | 32 | 0.304762 | 217 | 32 | 0.147465 | 185 | 50 | 0 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v2 | capped_confirmation | window | 105 | 62 | 0.590476 | 217 | 62 | 0.285714 | 155 | 50 | -2 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v2 | onset | loose | 105 | 25 | 0.238095 | 217 | 25 | 0.115207 | 192 | 53 | -1 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v2 | onset | strict | 105 | 24 | 0.228571 | 217 | 24 | 0.110599 | 193 | 53 | -1 |
| ml_shallow_xgboost | top | all_a_ml_walk_forward_v2 | onset | window | 105 | 61 | 0.580952 | 217 | 61 | 0.281106 | 156 | 53 | -1 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v2 | capped_confirmation | loose | 109 | 19 | 0.174312 | 287 | 19 | 0.0662021 | 268 | 63 | 0 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v2 | capped_confirmation | strict | 109 | 17 | 0.155963 | 287 | 17 | 0.0592334 | 270 | 63 | 0 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v2 | capped_confirmation | window | 109 | 61 | 0.559633 | 287 | 61 | 0.212544 | 226 | 63 | 3 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v2 | onset | loose | 109 | 14 | 0.12844 | 287 | 14 | 0.0487805 | 273 | 63 | 0 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v2 | onset | strict | 109 | 13 | 0.119266 | 287 | 13 | 0.0452962 | 274 | 63 | 0 |
| ml_simple_rule | bottom | all_a_ml_walk_forward_v2 | onset | window | 109 | 61 | 0.559633 | 287 | 61 | 0.212544 | 226 | 63 | 2 |
| ml_simple_rule | top | all_a_ml_walk_forward_v2 | capped_confirmation | loose | 105 | 11 | 0.104762 | 287 | 11 | 0.0383275 | 276 | 25 | -2 |
| ml_simple_rule | top | all_a_ml_walk_forward_v2 | capped_confirmation | strict | 105 | 10 | 0.0952381 | 287 | 10 | 0.0348432 | 277 | 25 | -2 |
| ml_simple_rule | top | all_a_ml_walk_forward_v2 | capped_confirmation | window | 105 | 46 | 0.438095 | 287 | 46 | 0.160279 | 241 | 25 | -4 |
| ml_simple_rule | top | all_a_ml_walk_forward_v2 | onset | loose | 105 | 13 | 0.12381 | 287 | 13 | 0.0452962 | 274 | 25 | -3 |
| ml_simple_rule | top | all_a_ml_walk_forward_v2 | onset | strict | 105 | 12 | 0.114286 | 287 | 12 | 0.0418118 | 275 | 25 | -2.5 |
| ml_simple_rule | top | all_a_ml_walk_forward_v2 | onset | window | 105 | 44 | 0.419048 | 287 | 44 | 0.15331 | 243 | 25 | -5 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_elastic_net | bottom | capped_confirmation | 全A | 15 | 0.4 | 29 | 0.206897 | 23 | 6 | -2.5 |
| ml_elastic_net | bottom | capped_confirmation | 国证2000 | 16 | 0.3125 | 29 | 0.172414 | 24 | 7 | -2 |
| ml_elastic_net | bottom | capped_confirmation | 中证1000 | 14 | 0.357143 | 29 | 0.172414 | 24 | 7 | -2 |
| ml_elastic_net | bottom | capped_confirmation | 沪深300 | 15 | 0.466667 | 29 | 0.241379 | 22 | 9 | 0 |
| ml_elastic_net | bottom | capped_confirmation | 中证500 | 16 | 0.375 | 29 | 0.206897 | 23 | 7 | -2 |
| ml_elastic_net | bottom | capped_confirmation | 微盘股 | 15 | 0.4 | 29 | 0.206897 | 23 | 9 | -1.5 |
| ml_elastic_net | bottom | capped_confirmation | 上证指数 | 18 | 0.388889 | 29 | 0.241379 | 22 | 8 | -3 |
| ml_elastic_net | bottom | onset | 全A | 15 | 0.4 | 29 | 0.206897 | 23 | 5 | -3.5 |
| ml_elastic_net | bottom | onset | 国证2000 | 16 | 0.3125 | 29 | 0.172414 | 24 | 7 | -3 |
| ml_elastic_net | bottom | onset | 中证1000 | 14 | 0.357143 | 29 | 0.172414 | 24 | 7 | -3 |
| ml_elastic_net | bottom | onset | 沪深300 | 15 | 0.466667 | 29 | 0.241379 | 22 | 8 | -1 |
| ml_elastic_net | bottom | onset | 中证500 | 16 | 0.375 | 29 | 0.206897 | 23 | 7 | -3 |
| ml_elastic_net | bottom | onset | 微盘股 | 15 | 0.4 | 29 | 0.206897 | 23 | 9 | -2.5 |
| ml_elastic_net | bottom | onset | 上证指数 | 18 | 0.388889 | 29 | 0.241379 | 22 | 7 | -4 |
| ml_elastic_net | top | capped_confirmation | 全A | 14 | 0.428571 | 30 | 0.2 | 24 | 6 | -2.5 |
| ml_elastic_net | top | capped_confirmation | 国证2000 | 15 | 0.333333 | 30 | 0.166667 | 25 | 7 | -2 |
| ml_elastic_net | top | capped_confirmation | 中证1000 | 13 | 0.384615 | 30 | 0.166667 | 25 | 3 | -6 |
| ml_elastic_net | top | capped_confirmation | 沪深300 | 15 | 0.4 | 30 | 0.2 | 24 | 6 | 1 |
| ml_elastic_net | top | capped_confirmation | 中证500 | 15 | 0.4 | 30 | 0.2 | 24 | 4 | -6.5 |
| ml_elastic_net | top | capped_confirmation | 微盘股 | 15 | 0.466667 | 30 | 0.233333 | 23 | 6 | -2 |
| ml_elastic_net | top | capped_confirmation | 上证指数 | 18 | 0.333333 | 30 | 0.2 | 24 | 6 | -2.5 |
| ml_elastic_net | top | onset | 全A | 14 | 0.357143 | 30 | 0.166667 | 25 | 7 | -3 |
| ml_elastic_net | top | onset | 国证2000 | 15 | 0.333333 | 30 | 0.166667 | 25 | 7 | -3 |
| ml_elastic_net | top | onset | 中证1000 | 13 | 0.307692 | 30 | 0.133333 | 26 | 4 | -5.5 |
| ml_elastic_net | top | onset | 沪深300 | 15 | 0.4 | 30 | 0.2 | 24 | 6 | 0 |
| ml_elastic_net | top | onset | 中证500 | 15 | 0.333333 | 30 | 0.166667 | 25 | 5 | -6 |
| ml_elastic_net | top | onset | 微盘股 | 15 | 0.466667 | 30 | 0.233333 | 23 | 6 | -3 |
| ml_elastic_net | top | onset | 上证指数 | 18 | 0.333333 | 30 | 0.2 | 24 | 6 | -3.5 |
| ml_shallow_xgboost | bottom | capped_confirmation | 全A | 15 | 0.6 | 31 | 0.290323 | 22 | 7 | -1 |
| ml_shallow_xgboost | bottom | capped_confirmation | 国证2000 | 16 | 0.5 | 31 | 0.258065 | 23 | 6 | -0.5 |
| ml_shallow_xgboost | bottom | capped_confirmation | 中证1000 | 14 | 0.642857 | 31 | 0.290323 | 22 | 6 | -1 |
| ml_shallow_xgboost | bottom | capped_confirmation | 沪深300 | 15 | 0.533333 | 31 | 0.258065 | 23 | 6 | -1 |
| ml_shallow_xgboost | bottom | capped_confirmation | 中证500 | 16 | 0.6875 | 31 | 0.354839 | 20 | 11 | -1 |
| ml_shallow_xgboost | bottom | capped_confirmation | 微盘股 | 15 | 0.533333 | 31 | 0.258065 | 23 | 6 | -0.5 |
| ml_shallow_xgboost | bottom | capped_confirmation | 上证指数 | 18 | 0.5 | 31 | 0.290323 | 22 | 6 | 0 |
| ml_shallow_xgboost | bottom | onset | 全A | 15 | 0.6 | 31 | 0.290323 | 22 | 7 | -2 |
| ml_shallow_xgboost | bottom | onset | 国证2000 | 16 | 0.5 | 31 | 0.258065 | 23 | 6 | -1.5 |
| ml_shallow_xgboost | bottom | onset | 中证1000 | 14 | 0.642857 | 31 | 0.290323 | 22 | 6 | -2 |
| ml_shallow_xgboost | bottom | onset | 沪深300 | 15 | 0.533333 | 31 | 0.258065 | 23 | 6 | -2 |
| ml_shallow_xgboost | bottom | onset | 中证500 | 16 | 0.625 | 31 | 0.322581 | 21 | 11 | -1.5 |
| ml_shallow_xgboost | bottom | onset | 微盘股 | 15 | 0.533333 | 31 | 0.258065 | 23 | 6 | -1.5 |
| ml_shallow_xgboost | bottom | onset | 上证指数 | 18 | 0.5 | 31 | 0.290323 | 22 | 6 | -1 |
| ml_shallow_xgboost | top | capped_confirmation | 全A | 14 | 0.714286 | 31 | 0.322581 | 21 | 8 | -3 |
| ml_shallow_xgboost | top | capped_confirmation | 国证2000 | 15 | 0.6 | 31 | 0.290323 | 22 | 8 | 0 |
| ml_shallow_xgboost | top | capped_confirmation | 中证1000 | 13 | 0.538462 | 31 | 0.225806 | 24 | 7 | -4 |
| ml_shallow_xgboost | top | capped_confirmation | 沪深300 | 15 | 0.6 | 31 | 0.290323 | 22 | 5 | -2 |
| ml_shallow_xgboost | top | capped_confirmation | 中证500 | 15 | 0.666667 | 31 | 0.322581 | 21 | 7 | -5 |
| ml_shallow_xgboost | top | capped_confirmation | 微盘股 | 15 | 0.466667 | 31 | 0.225806 | 24 | 7 | 0 |
| ml_shallow_xgboost | top | capped_confirmation | 上证指数 | 18 | 0.555556 | 31 | 0.322581 | 21 | 8 | -2 |
| ml_shallow_xgboost | top | onset | 全A | 14 | 0.714286 | 31 | 0.322581 | 21 | 8 | -2 |
| ml_shallow_xgboost | top | onset | 国证2000 | 15 | 0.6 | 31 | 0.290323 | 22 | 9 | -1 |
| ml_shallow_xgboost | top | onset | 中证1000 | 13 | 0.538462 | 31 | 0.225806 | 24 | 7 | -1 |
| ml_shallow_xgboost | top | onset | 沪深300 | 15 | 0.6 | 31 | 0.290323 | 22 | 5 | -3 |
| ml_shallow_xgboost | top | onset | 中证500 | 15 | 0.6 | 31 | 0.290323 | 22 | 8 | -1 |
| ml_shallow_xgboost | top | onset | 微盘股 | 15 | 0.466667 | 31 | 0.225806 | 24 | 8 | -1 |
| ml_shallow_xgboost | top | onset | 上证指数 | 18 | 0.555556 | 31 | 0.322581 | 21 | 8 | -2 |
| ml_simple_rule | bottom | capped_confirmation | 全A | 15 | 0.533333 | 41 | 0.195122 | 33 | 9 | 2 |
| ml_simple_rule | bottom | capped_confirmation | 国证2000 | 16 | 0.5625 | 41 | 0.219512 | 32 | 8 | 1 |
| ml_simple_rule | bottom | capped_confirmation | 中证1000 | 14 | 0.571429 | 41 | 0.195122 | 33 | 8 | 3 |
| ml_simple_rule | bottom | capped_confirmation | 沪深300 | 15 | 0.533333 | 41 | 0.195122 | 33 | 8 | 4 |
| ml_simple_rule | bottom | capped_confirmation | 中证500 | 16 | 0.5625 | 41 | 0.219512 | 32 | 10 | 3 |
| ml_simple_rule | bottom | capped_confirmation | 微盘股 | 15 | 0.6 | 41 | 0.219512 | 32 | 9 | 1 |
| ml_simple_rule | bottom | capped_confirmation | 上证指数 | 18 | 0.555556 | 41 | 0.243902 | 31 | 11 | 4 |
| ml_simple_rule | bottom | onset | 全A | 15 | 0.533333 | 41 | 0.195122 | 33 | 9 | 1 |
| ml_simple_rule | bottom | onset | 国证2000 | 16 | 0.5625 | 41 | 0.219512 | 32 | 8 | 0 |
| ml_simple_rule | bottom | onset | 中证1000 | 14 | 0.571429 | 41 | 0.195122 | 33 | 8 | 2 |
| ml_simple_rule | bottom | onset | 沪深300 | 15 | 0.533333 | 41 | 0.195122 | 33 | 8 | 4 |
| ml_simple_rule | bottom | onset | 中证500 | 16 | 0.5625 | 41 | 0.219512 | 32 | 10 | 2 |
| ml_simple_rule | bottom | onset | 微盘股 | 15 | 0.6 | 41 | 0.219512 | 32 | 9 | 0 |
| ml_simple_rule | bottom | onset | 上证指数 | 18 | 0.555556 | 41 | 0.243902 | 31 | 11 | 3 |
| ml_simple_rule | top | capped_confirmation | 全A | 14 | 0.428571 | 41 | 0.146341 | 35 | 2 | -5 |
| ml_simple_rule | top | capped_confirmation | 国证2000 | 15 | 0.333333 | 41 | 0.121951 | 36 | 3 | -14 |
| ml_simple_rule | top | capped_confirmation | 中证1000 | 13 | 0.461538 | 41 | 0.146341 | 35 | 2 | -9.5 |
| ml_simple_rule | top | capped_confirmation | 沪深300 | 15 | 0.533333 | 41 | 0.195122 | 33 | 4 | -1.5 |
| ml_simple_rule | top | capped_confirmation | 中证500 | 15 | 0.466667 | 41 | 0.170732 | 34 | 4 | 2 |
| ml_simple_rule | top | capped_confirmation | 微盘股 | 15 | 0.4 | 41 | 0.146341 | 35 | 5 | -2 |
| ml_simple_rule | top | capped_confirmation | 上证指数 | 18 | 0.444444 | 41 | 0.195122 | 33 | 5 | -3.5 |
| ml_simple_rule | top | onset | 全A | 14 | 0.357143 | 41 | 0.121951 | 36 | 2 | -5 |
| ml_simple_rule | top | onset | 国证2000 | 15 | 0.333333 | 41 | 0.121951 | 36 | 2 | -15 |
| ml_simple_rule | top | onset | 中证1000 | 13 | 0.461538 | 41 | 0.146341 | 35 | 2 | -10.5 |
| ml_simple_rule | top | onset | 沪深300 | 15 | 0.466667 | 41 | 0.170732 | 34 | 4 | 0 |
| ml_simple_rule | top | onset | 中证500 | 15 | 0.466667 | 41 | 0.170732 | 34 | 4 | 1 |
| ml_simple_rule | top | onset | 微盘股 | 15 | 0.4 | 41 | 0.146341 | 35 | 5 | -4 |
| ml_simple_rule | top | onset | 上证指数 | 18 | 0.444444 | 41 | 0.195122 | 33 | 6 | -4.5 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 557 |
| false_alarm | 1664 |
| matched | 621 |
| missed_region | 663 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `ml_elastic_net/bottom/all_a_ml_walk_forward_v2/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 53 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 15/17/42。
- `ml_elastic_net/bottom/all_a_ml_walk_forward_v2/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 50 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 18/18/42。
- `ml_elastic_net/top/all_a_ml_walk_forward_v2/capped_confirmation`：存在 38 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 17/17/41。
- `ml_elastic_net/top/all_a_ml_walk_forward_v2/onset`：存在 41 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 16/16/38。
- `ml_shallow_xgboost/bottom/all_a_ml_walk_forward_v2/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 48 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 18/21/62。
- `ml_shallow_xgboost/bottom/all_a_ml_walk_forward_v2/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 48 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 11/15/61。
- `ml_shallow_xgboost/top/all_a_ml_walk_forward_v2/capped_confirmation`：存在 50 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 32/33/62。
- `ml_shallow_xgboost/top/all_a_ml_walk_forward_v2/onset`：存在 53 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 24/25/61。
- `ml_simple_rule/bottom/all_a_ml_walk_forward_v2/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 63 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 17/19/61。
- `ml_simple_rule/bottom/all_a_ml_walk_forward_v2/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 63 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 13/14/61。
- `ml_simple_rule/top/all_a_ml_walk_forward_v2/capped_confirmation`：存在 25 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 10/11/46。
- `ml_simple_rule/top/all_a_ml_walk_forward_v2/onset`：存在 25 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 12/13/44。
