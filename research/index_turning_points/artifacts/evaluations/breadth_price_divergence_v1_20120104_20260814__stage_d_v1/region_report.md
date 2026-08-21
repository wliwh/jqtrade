# 顶底区域定位评测

- 评测版本：`breadth_price_divergence_v1_20120104_20260814__stage_d_v1`
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
| all_a_ma20_breadth_price_divergence_top | top | breadth_price_divergence_v1_20120104_20260814 | capped_confirmation | loose | 255 | 102 | 0.4 | 980 | 102 | 0.104082 | 878 | 240 | 0 |
| all_a_ma20_breadth_price_divergence_top | top | breadth_price_divergence_v1_20120104_20260814 | capped_confirmation | strict | 255 | 94 | 0.368627 | 980 | 94 | 0.0959184 | 886 | 240 | 0 |
| all_a_ma20_breadth_price_divergence_top | top | breadth_price_divergence_v1_20120104_20260814 | capped_confirmation | window | 255 | 140 | 0.54902 | 980 | 140 | 0.142857 | 840 | 240 | -1 |
| all_a_ma20_breadth_price_divergence_top | top | breadth_price_divergence_v1_20120104_20260814 | onset | loose | 255 | 101 | 0.396078 | 980 | 101 | 0.103061 | 879 | 243 | -1 |
| all_a_ma20_breadth_price_divergence_top | top | breadth_price_divergence_v1_20120104_20260814 | onset | strict | 255 | 95 | 0.372549 | 980 | 95 | 0.0969388 | 885 | 243 | -1 |
| all_a_ma20_breadth_price_divergence_top | top | breadth_price_divergence_v1_20120104_20260814 | onset | window | 255 | 138 | 0.541176 | 980 | 138 | 0.140816 | 842 | 243 | -1 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 36 | 0.555556 | 140 | 0.142857 | 120 | 36 | 0 |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 31 | 0.548387 | 140 | 0.121429 | 123 | 36 | -1 |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 33 | 0.575758 | 140 | 0.135714 | 121 | 32 | -1 |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 39 | 0.564103 | 140 | 0.157143 | 118 | 36 | -1.5 |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 37 | 0.567568 | 140 | 0.15 | 119 | 37 | 0 |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 32 | 0.53125 | 140 | 0.121429 | 123 | 26 | 0 |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 47 | 0.510638 | 140 | 0.171429 | 116 | 37 | -1 |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 36 | 0.555556 | 140 | 0.142857 | 120 | 36 | 0.5 |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 31 | 0.516129 | 140 | 0.114286 | 124 | 36 | 0.5 |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 33 | 0.575758 | 140 | 0.135714 | 121 | 33 | 0 |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 39 | 0.538462 | 140 | 0.15 | 119 | 38 | -2 |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 37 | 0.567568 | 140 | 0.15 | 119 | 35 | 0 |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 32 | 0.53125 | 140 | 0.121429 | 123 | 27 | -1 |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 47 | 0.510638 | 140 | 0.171429 | 116 | 38 | -1 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 483 |
| false_alarm | 1199 |
| matched | 278 |
| missed_region | 232 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `all_a_ma20_breadth_price_divergence_top/top/breadth_price_divergence_v1_20120104_20260814/capped_confirmation`：存在 240 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 94/102/140。
- `all_a_ma20_breadth_price_divergence_top/top/breadth_price_divergence_v1_20120104_20260814/onset`：存在 243 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 95/101/138。
