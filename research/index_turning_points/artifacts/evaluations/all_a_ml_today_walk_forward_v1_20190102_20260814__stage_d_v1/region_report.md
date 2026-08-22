# 顶底区域定位评测

- 评测版本：`all_a_ml_today_walk_forward_v1_stage_d_v1`
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
| ml_today_elastic_net | bottom | all_a_ml_today_walk_forward_v1 | capped_confirmation | loose | 109 | 32 | 0.293578 | 189 | 32 | 0.169312 | 157 | 44 | 0 |
| ml_today_elastic_net | bottom | all_a_ml_today_walk_forward_v1 | capped_confirmation | strict | 109 | 29 | 0.266055 | 189 | 29 | 0.153439 | 160 | 44 | 0 |
| ml_today_elastic_net | bottom | all_a_ml_today_walk_forward_v1 | capped_confirmation | window | 109 | 62 | 0.568807 | 189 | 62 | 0.328042 | 127 | 44 | -1 |
| ml_today_elastic_net | bottom | all_a_ml_today_walk_forward_v1 | onset | loose | 109 | 25 | 0.229358 | 189 | 25 | 0.132275 | 164 | 44 | 0 |
| ml_today_elastic_net | bottom | all_a_ml_today_walk_forward_v1 | onset | strict | 109 | 23 | 0.211009 | 189 | 23 | 0.121693 | 166 | 44 | 0 |
| ml_today_elastic_net | bottom | all_a_ml_today_walk_forward_v1 | onset | window | 109 | 62 | 0.568807 | 189 | 62 | 0.328042 | 127 | 44 | -2 |
| ml_today_elastic_net | top | all_a_ml_today_walk_forward_v1 | capped_confirmation | loose | 105 | 24 | 0.228571 | 259 | 24 | 0.0926641 | 235 | 44 | -0.5 |
| ml_today_elastic_net | top | all_a_ml_today_walk_forward_v1 | capped_confirmation | strict | 105 | 22 | 0.209524 | 259 | 22 | 0.0849421 | 237 | 44 | 0 |
| ml_today_elastic_net | top | all_a_ml_today_walk_forward_v1 | capped_confirmation | window | 105 | 59 | 0.561905 | 259 | 59 | 0.227799 | 200 | 44 | -3 |
| ml_today_elastic_net | top | all_a_ml_today_walk_forward_v1 | onset | loose | 105 | 21 | 0.2 | 259 | 21 | 0.0810811 | 238 | 43 | -1 |
| ml_today_elastic_net | top | all_a_ml_today_walk_forward_v1 | onset | strict | 105 | 19 | 0.180952 | 259 | 19 | 0.0733591 | 240 | 43 | -1 |
| ml_today_elastic_net | top | all_a_ml_today_walk_forward_v1 | onset | window | 105 | 53 | 0.504762 | 259 | 53 | 0.204633 | 206 | 43 | -4 |
| ml_today_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v1 | capped_confirmation | loose | 109 | 35 | 0.321101 | 343 | 35 | 0.102041 | 308 | 90 | 0 |
| ml_today_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v1 | capped_confirmation | strict | 109 | 32 | 0.293578 | 343 | 32 | 0.0932945 | 311 | 90 | 0 |
| ml_today_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v1 | capped_confirmation | window | 109 | 66 | 0.605505 | 343 | 66 | 0.19242 | 277 | 90 | 1 |
| ml_today_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v1 | onset | loose | 109 | 34 | 0.311927 | 343 | 34 | 0.0991254 | 309 | 90 | 0 |
| ml_today_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v1 | onset | strict | 109 | 33 | 0.302752 | 343 | 33 | 0.0962099 | 310 | 90 | 0 |
| ml_today_shallow_gbdt | bottom | all_a_ml_today_walk_forward_v1 | onset | window | 109 | 65 | 0.59633 | 343 | 65 | 0.189504 | 278 | 90 | 0 |
| ml_today_shallow_gbdt | top | all_a_ml_today_walk_forward_v1 | capped_confirmation | loose | 105 | 43 | 0.409524 | 308 | 43 | 0.13961 | 265 | 65 | -1 |
| ml_today_shallow_gbdt | top | all_a_ml_today_walk_forward_v1 | capped_confirmation | strict | 105 | 43 | 0.409524 | 308 | 43 | 0.13961 | 265 | 65 | -1 |
| ml_today_shallow_gbdt | top | all_a_ml_today_walk_forward_v1 | capped_confirmation | window | 105 | 77 | 0.733333 | 308 | 77 | 0.25 | 231 | 65 | -1 |
| ml_today_shallow_gbdt | top | all_a_ml_today_walk_forward_v1 | onset | loose | 105 | 44 | 0.419048 | 308 | 44 | 0.142857 | 264 | 65 | -1.5 |
| ml_today_shallow_gbdt | top | all_a_ml_today_walk_forward_v1 | onset | strict | 105 | 44 | 0.419048 | 308 | 44 | 0.142857 | 264 | 65 | -1.5 |
| ml_today_shallow_gbdt | top | all_a_ml_today_walk_forward_v1 | onset | window | 105 | 77 | 0.733333 | 308 | 77 | 0.25 | 231 | 65 | -1 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 15 | 0.533333 | 27 | 0.296296 | 19 | 6 | -1.5 |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 16 | 0.625 | 27 | 0.37037 | 17 | 7 | -1 |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 14 | 0.642857 | 27 | 0.333333 | 18 | 7 | -1 |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 15 | 0.466667 | 27 | 0.259259 | 20 | 5 | 1 |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 16 | 0.6875 | 27 | 0.407407 | 16 | 7 | -3 |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 15 | 0.533333 | 27 | 0.296296 | 19 | 7 | -1 |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 18 | 0.5 | 27 | 0.333333 | 18 | 5 | -1 |
| ml_today_elastic_net | bottom | onset | 全A | 15 | 0.533333 | 27 | 0.296296 | 19 | 6 | -2.5 |
| ml_today_elastic_net | bottom | onset | 国证2000 | 16 | 0.625 | 27 | 0.37037 | 17 | 7 | -2 |
| ml_today_elastic_net | bottom | onset | 中证1000 | 14 | 0.642857 | 27 | 0.333333 | 18 | 7 | -2 |
| ml_today_elastic_net | bottom | onset | 沪深300 | 15 | 0.466667 | 27 | 0.259259 | 20 | 5 | 0 |
| ml_today_elastic_net | bottom | onset | 中证500 | 16 | 0.6875 | 27 | 0.407407 | 16 | 7 | -4 |
| ml_today_elastic_net | bottom | onset | 微盘股 | 15 | 0.533333 | 27 | 0.296296 | 19 | 7 | -2 |
| ml_today_elastic_net | bottom | onset | 上证指数 | 18 | 0.5 | 27 | 0.333333 | 18 | 5 | -2 |
| ml_today_elastic_net | top | capped_confirmation | 全A | 14 | 0.571429 | 37 | 0.216216 | 29 | 6 | -3.5 |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 15 | 0.666667 | 37 | 0.27027 | 27 | 6 | -3.5 |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 13 | 0.538462 | 37 | 0.189189 | 30 | 6 | 1 |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 15 | 0.6 | 37 | 0.243243 | 28 | 10 | -1 |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 15 | 0.4 | 37 | 0.162162 | 31 | 4 | -3.5 |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 15 | 0.666667 | 37 | 0.27027 | 27 | 6 | -9 |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 18 | 0.5 | 37 | 0.243243 | 28 | 6 | -3 |
| ml_today_elastic_net | top | onset | 全A | 14 | 0.5 | 37 | 0.189189 | 30 | 6 | -4 |
| ml_today_elastic_net | top | onset | 国证2000 | 15 | 0.6 | 37 | 0.243243 | 28 | 6 | -4 |
| ml_today_elastic_net | top | onset | 中证1000 | 13 | 0.538462 | 37 | 0.189189 | 30 | 6 | 0 |
| ml_today_elastic_net | top | onset | 沪深300 | 15 | 0.533333 | 37 | 0.216216 | 29 | 10 | -1.5 |
| ml_today_elastic_net | top | onset | 中证500 | 15 | 0.333333 | 37 | 0.135135 | 32 | 4 | -4 |
| ml_today_elastic_net | top | onset | 微盘股 | 15 | 0.6 | 37 | 0.243243 | 28 | 5 | -8 |
| ml_today_elastic_net | top | onset | 上证指数 | 18 | 0.444444 | 37 | 0.216216 | 29 | 6 | -2.5 |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 15 | 0.6 | 49 | 0.183673 | 40 | 11 | 0 |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 16 | 0.625 | 49 | 0.204082 | 39 | 13 | 1 |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 14 | 0.642857 | 49 | 0.183673 | 40 | 13 | 1 |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 15 | 0.6 | 49 | 0.183673 | 40 | 12 | 1 |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 16 | 0.6875 | 49 | 0.22449 | 38 | 13 | 0 |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 15 | 0.533333 | 49 | 0.163265 | 41 | 15 | 0.5 |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 18 | 0.555556 | 49 | 0.204082 | 39 | 13 | 1 |
| ml_today_shallow_gbdt | bottom | onset | 全A | 15 | 0.6 | 49 | 0.183673 | 40 | 11 | -1 |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 16 | 0.5625 | 49 | 0.183673 | 40 | 13 | 0 |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 14 | 0.642857 | 49 | 0.183673 | 40 | 13 | 0 |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 15 | 0.6 | 49 | 0.183673 | 40 | 12 | 0 |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 16 | 0.6875 | 49 | 0.22449 | 38 | 13 | -1 |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 15 | 0.533333 | 49 | 0.163265 | 41 | 15 | -0.5 |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 18 | 0.555556 | 49 | 0.204082 | 39 | 13 | 0 |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 14 | 0.785714 | 44 | 0.25 | 33 | 9 | -1 |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 15 | 0.733333 | 44 | 0.25 | 33 | 11 | -1 |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 13 | 0.846154 | 44 | 0.25 | 33 | 8 | -1 |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 15 | 0.666667 | 44 | 0.227273 | 34 | 10 | 0 |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 15 | 0.733333 | 44 | 0.25 | 33 | 8 | -1 |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 15 | 0.666667 | 44 | 0.227273 | 34 | 10 | 0 |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 18 | 0.722222 | 44 | 0.295455 | 31 | 9 | 0 |
| ml_today_shallow_gbdt | top | onset | 全A | 14 | 0.785714 | 44 | 0.25 | 33 | 9 | -2 |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 15 | 0.733333 | 44 | 0.25 | 33 | 11 | -2 |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 13 | 0.846154 | 44 | 0.25 | 33 | 8 | -2 |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 15 | 0.666667 | 44 | 0.227273 | 34 | 10 | -1 |
| ml_today_shallow_gbdt | top | onset | 中证500 | 15 | 0.733333 | 44 | 0.25 | 33 | 8 | 0 |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 15 | 0.666667 | 44 | 0.227273 | 34 | 10 | -1 |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 18 | 0.722222 | 44 | 0.295455 | 31 | 9 | -1 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 485 |
| false_alarm | 1192 |
| matched | 521 |
| missed_region | 335 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `ml_today_elastic_net/bottom/all_a_ml_today_walk_forward_v1/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 44 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 29/32/62。
- `ml_today_elastic_net/bottom/all_a_ml_today_walk_forward_v1/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 44 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 23/25/62。
- `ml_today_elastic_net/top/all_a_ml_today_walk_forward_v1/capped_confirmation`：存在 44 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 22/24/59。
- `ml_today_elastic_net/top/all_a_ml_today_walk_forward_v1/onset`：存在 43 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 19/21/53。
- `ml_today_shallow_gbdt/bottom/all_a_ml_today_walk_forward_v1/capped_confirmation`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 90 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 32/35/66。
- `ml_today_shallow_gbdt/bottom/all_a_ml_today_walk_forward_v1/onset`：区域窗口不完整：预测 5 个、确认 5 个；对应时点召回切片已从分母排除。 存在 90 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 33/34/65。
- `ml_today_shallow_gbdt/top/all_a_ml_today_walk_forward_v1/capped_confirmation`：存在 65 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 43/43/77。
- `ml_today_shallow_gbdt/top/all_a_ml_today_walk_forward_v1/onset`：存在 65 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 44/44/77。
