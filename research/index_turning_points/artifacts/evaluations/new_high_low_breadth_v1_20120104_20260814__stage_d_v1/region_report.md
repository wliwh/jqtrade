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
| new_high_low_breadth_bottom | bottom | new_high_low_breadth_v1_20120104_20260814 | capped_confirmation | loose | 260 | 43 | 0.165385 | 343 | 43 | 0.125364 | 300 | 73 | 0 |
| new_high_low_breadth_bottom | bottom | new_high_low_breadth_v1_20120104_20260814 | capped_confirmation | strict | 260 | 33 | 0.126923 | 343 | 33 | 0.0962099 | 310 | 73 | 0 |
| new_high_low_breadth_bottom | bottom | new_high_low_breadth_v1_20120104_20260814 | capped_confirmation | window | 260 | 90 | 0.346154 | 343 | 90 | 0.262391 | 253 | 73 | 0 |
| new_high_low_breadth_bottom | bottom | new_high_low_breadth_v1_20120104_20260814 | onset | loose | 260 | 42 | 0.161538 | 343 | 42 | 0.122449 | 301 | 73 | 0 |
| new_high_low_breadth_bottom | bottom | new_high_low_breadth_v1_20120104_20260814 | onset | strict | 260 | 38 | 0.146154 | 343 | 38 | 0.110787 | 305 | 73 | 0 |
| new_high_low_breadth_bottom | bottom | new_high_low_breadth_v1_20120104_20260814 | onset | window | 260 | 90 | 0.346154 | 343 | 90 | 0.262391 | 253 | 73 | -1 |
| new_high_low_breadth_top | top | new_high_low_breadth_v1_20120104_20260814 | capped_confirmation | loose | 255 | 49 | 0.192157 | 364 | 49 | 0.134615 | 315 | 71 | 0 |
| new_high_low_breadth_top | top | new_high_low_breadth_v1_20120104_20260814 | capped_confirmation | strict | 255 | 40 | 0.156863 | 364 | 40 | 0.10989 | 324 | 71 | 0 |
| new_high_low_breadth_top | top | new_high_low_breadth_v1_20120104_20260814 | capped_confirmation | window | 255 | 83 | 0.32549 | 364 | 83 | 0.228022 | 281 | 71 | 0 |
| new_high_low_breadth_top | top | new_high_low_breadth_v1_20120104_20260814 | onset | loose | 255 | 46 | 0.180392 | 364 | 46 | 0.126374 | 318 | 71 | 0.5 |
| new_high_low_breadth_top | top | new_high_low_breadth_v1_20120104_20260814 | onset | strict | 255 | 45 | 0.176471 | 364 | 45 | 0.123626 | 319 | 71 | 0 |
| new_high_low_breadth_top | top | new_high_low_breadth_v1_20120104_20260814 | onset | window | 255 | 83 | 0.32549 | 364 | 83 | 0.228022 | 281 | 71 | 1 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 37 | 0.378378 | 49 | 0.285714 | 35 | 11 | 0.5 |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 32 | 0.40625 | 49 | 0.265306 | 36 | 10 | 0 |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 34 | 0.352941 | 49 | 0.244898 | 37 | 10 | 0 |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 39 | 0.282051 | 49 | 0.22449 | 38 | 8 | 0 |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 38 | 0.342105 | 49 | 0.265306 | 36 | 12 | -2 |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 33 | 0.393939 | 49 | 0.265306 | 36 | 10 | 0 |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 47 | 0.297872 | 49 | 0.285714 | 35 | 12 | -1 |
| new_high_low_breadth_bottom | bottom | onset | 全A | 37 | 0.378378 | 49 | 0.285714 | 35 | 11 | 0 |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 32 | 0.40625 | 49 | 0.265306 | 36 | 10 | -1 |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 34 | 0.352941 | 49 | 0.244898 | 37 | 10 | -0.5 |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 39 | 0.282051 | 49 | 0.22449 | 38 | 8 | -1 |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 38 | 0.342105 | 49 | 0.265306 | 36 | 12 | -1 |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 33 | 0.393939 | 49 | 0.265306 | 36 | 10 | -1 |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 47 | 0.297872 | 49 | 0.285714 | 35 | 12 | -0.5 |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 36 | 0.333333 | 52 | 0.230769 | 40 | 12 | 0.5 |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 31 | 0.322581 | 52 | 0.192308 | 42 | 8 | -1.5 |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 33 | 0.30303 | 52 | 0.192308 | 42 | 11 | 0.5 |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 39 | 0.307692 | 52 | 0.230769 | 40 | 11 | 0.5 |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 37 | 0.351351 | 52 | 0.25 | 39 | 10 | 0 |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 32 | 0.28125 | 52 | 0.173077 | 43 | 6 | -2 |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 47 | 0.361702 | 52 | 0.326923 | 35 | 13 | 1 |
| new_high_low_breadth_top | top | onset | 全A | 36 | 0.333333 | 52 | 0.230769 | 40 | 12 | 0.5 |
| new_high_low_breadth_top | top | onset | 国证2000 | 31 | 0.322581 | 52 | 0.192308 | 42 | 7 | -0.5 |
| new_high_low_breadth_top | top | onset | 中证1000 | 33 | 0.30303 | 52 | 0.192308 | 42 | 11 | 1 |
| new_high_low_breadth_top | top | onset | 沪深300 | 39 | 0.307692 | 52 | 0.230769 | 40 | 10 | -0.5 |
| new_high_low_breadth_top | top | onset | 中证500 | 37 | 0.351351 | 52 | 0.25 | 39 | 11 | 1 |
| new_high_low_breadth_top | top | onset | 微盘股 | 32 | 0.28125 | 52 | 0.173077 | 43 | 6 | -1 |
| new_high_low_breadth_top | top | onset | 上证指数 | 47 | 0.361702 | 52 | 0.326923 | 35 | 14 | 0 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 288 |
| false_alarm | 780 |
| matched | 346 |
| missed_region | 684 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `new_high_low_breadth_bottom/bottom/new_high_low_breadth_v1_20120104_20260814/capped_confirmation`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 73 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 33/43/90。
- `new_high_low_breadth_bottom/bottom/new_high_low_breadth_v1_20120104_20260814/onset`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 73 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 38/42/90。
- `new_high_low_breadth_top/top/new_high_low_breadth_v1_20120104_20260814/capped_confirmation`：存在 71 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 40/49/83。
- `new_high_low_breadth_top/top/new_high_low_breadth_v1_20120104_20260814/onset`：存在 71 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 45/46/83。
