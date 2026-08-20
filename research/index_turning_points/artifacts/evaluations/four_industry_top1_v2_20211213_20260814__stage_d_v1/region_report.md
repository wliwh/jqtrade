# 顶底区域定位评测

- 评测版本：`four_industry_top1_v2_20211213_20260814__stage_d_v1`
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
| four_industry_top1 | top | four_industry_top1_v2_20211213_20260814 | capped_confirmation | loose | 68 | 41 | 0.602941 | 735 | 41 | 0.0557823 | 694 | 183 | -1 |
| four_industry_top1 | top | four_industry_top1_v2_20211213_20260814 | capped_confirmation | strict | 68 | 40 | 0.588235 | 735 | 40 | 0.0544218 | 695 | 183 | -0.5 |
| four_industry_top1 | top | four_industry_top1_v2_20211213_20260814 | capped_confirmation | window | 68 | 64 | 0.941176 | 735 | 64 | 0.0870748 | 671 | 183 | -2 |
| four_industry_top1 | top | four_industry_top1_v2_20211213_20260814 | onset | loose | 68 | 41 | 0.602941 | 735 | 41 | 0.0557823 | 694 | 177 | 0 |
| four_industry_top1 | top | four_industry_top1_v2_20211213_20260814 | onset | strict | 68 | 38 | 0.558824 | 735 | 38 | 0.0517007 | 697 | 177 | 0.5 |
| four_industry_top1 | top | four_industry_top1_v2_20211213_20260814 | onset | window | 68 | 64 | 0.941176 | 735 | 64 | 0.0870748 | 671 | 177 | -1.5 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| four_industry_top1 | top | capped_confirmation | 全A | 9 | 0.888889 | 105 | 0.0761905 | 97 | 24 | -2 |
| four_industry_top1 | top | capped_confirmation | 国证2000 | 11 | 1 | 105 | 0.104762 | 94 | 38 | -2 |
| four_industry_top1 | top | capped_confirmation | 中证1000 | 8 | 1 | 105 | 0.0761905 | 97 | 27 | -2.5 |
| four_industry_top1 | top | capped_confirmation | 沪深300 | 9 | 0.888889 | 105 | 0.0761905 | 97 | 19 | 0 |
| four_industry_top1 | top | capped_confirmation | 中证500 | 9 | 0.888889 | 105 | 0.0761905 | 97 | 17 | -2 |
| four_industry_top1 | top | capped_confirmation | 微盘股 | 12 | 1 | 105 | 0.114286 | 93 | 35 | -1 |
| four_industry_top1 | top | capped_confirmation | 上证指数 | 10 | 0.9 | 105 | 0.0857143 | 96 | 23 | -2 |
| four_industry_top1 | top | onset | 全A | 9 | 0.888889 | 105 | 0.0761905 | 97 | 22 | -3 |
| four_industry_top1 | top | onset | 国证2000 | 11 | 1 | 105 | 0.104762 | 94 | 36 | 3 |
| four_industry_top1 | top | onset | 中证1000 | 8 | 1 | 105 | 0.0761905 | 97 | 25 | -3 |
| four_industry_top1 | top | onset | 沪深300 | 9 | 0.888889 | 105 | 0.0761905 | 97 | 20 | -1 |
| four_industry_top1 | top | onset | 中证500 | 9 | 0.888889 | 105 | 0.0761905 | 97 | 17 | 1 |
| four_industry_top1 | top | onset | 微盘股 | 12 | 1 | 105 | 0.114286 | 93 | 36 | -1.5 |
| four_industry_top1 | top | onset | 上证指数 | 10 | 0.9 | 105 | 0.0857143 | 96 | 21 | -3 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 360 |
| false_alarm | 982 |
| matched | 128 |
| missed_region | 8 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `four_industry_top1/top/four_industry_top1_v2_20211213_20260814/capped_confirmation`：区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 183 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 40/41/64。
- `four_industry_top1/top/four_industry_top1_v2_20211213_20260814/onset`：区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 177 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 38/41/64。
