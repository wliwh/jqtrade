# 顶底区域定位评测

- 评测版本：`limit_up_down_breadth_v1_20120705_20260814__stage_d_v1`
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
| limit_up_down_breadth_bottom | bottom | limit_up_down_breadth_v1_20120705_20260814 | capped_confirmation | loose | 247 | 10 | 0.0404858 | 161 | 10 | 0.0621118 | 151 | 10 | 0 |
| limit_up_down_breadth_bottom | bottom | limit_up_down_breadth_v1_20120705_20260814 | capped_confirmation | strict | 247 | 7 | 0.0283401 | 161 | 7 | 0.0434783 | 154 | 10 | -4 |
| limit_up_down_breadth_bottom | bottom | limit_up_down_breadth_v1_20120705_20260814 | capped_confirmation | window | 247 | 46 | 0.186235 | 161 | 46 | 0.285714 | 115 | 10 | 2 |
| limit_up_down_breadth_bottom | bottom | limit_up_down_breadth_v1_20120705_20260814 | onset | loose | 247 | 11 | 0.0445344 | 161 | 11 | 0.068323 | 150 | 10 | 0 |
| limit_up_down_breadth_bottom | bottom | limit_up_down_breadth_v1_20120705_20260814 | onset | strict | 247 | 7 | 0.0283401 | 161 | 7 | 0.0434783 | 154 | 10 | -2 |
| limit_up_down_breadth_bottom | bottom | limit_up_down_breadth_v1_20120705_20260814 | onset | window | 247 | 46 | 0.186235 | 161 | 46 | 0.285714 | 115 | 10 | 1 |
| limit_up_down_breadth_top | top | limit_up_down_breadth_v1_20120705_20260814 | capped_confirmation | loose | 242 | 48 | 0.198347 | 364 | 48 | 0.131868 | 316 | 50 | 1 |
| limit_up_down_breadth_top | top | limit_up_down_breadth_v1_20120705_20260814 | capped_confirmation | strict | 242 | 42 | 0.173554 | 364 | 42 | 0.115385 | 322 | 50 | 1 |
| limit_up_down_breadth_top | top | limit_up_down_breadth_v1_20120705_20260814 | capped_confirmation | window | 242 | 113 | 0.466942 | 364 | 113 | 0.31044 | 251 | 50 | 0 |
| limit_up_down_breadth_top | top | limit_up_down_breadth_v1_20120705_20260814 | onset | loose | 242 | 53 | 0.219008 | 364 | 53 | 0.145604 | 311 | 56 | 0 |
| limit_up_down_breadth_top | top | limit_up_down_breadth_v1_20120705_20260814 | onset | strict | 242 | 47 | 0.194215 | 364 | 47 | 0.129121 | 317 | 56 | 0 |
| limit_up_down_breadth_top | top | limit_up_down_breadth_v1_20120705_20260814 | onset | window | 242 | 110 | 0.454545 | 364 | 110 | 0.302198 | 254 | 56 | 0 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 35 | 0.2 | 23 | 0.304348 | 16 | 1 | -3 |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 31 | 0.193548 | 23 | 0.26087 | 17 | 1 | 2.5 |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 32 | 0.1875 | 23 | 0.26087 | 17 | 1 | 2.5 |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 37 | 0.108108 | 23 | 0.173913 | 19 | 1 | 2.5 |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 36 | 0.194444 | 23 | 0.304348 | 16 | 2 | 2 |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 31 | 0.322581 | 23 | 0.434783 | 13 | 3 | -1 |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 45 | 0.133333 | 23 | 0.26087 | 17 | 1 | -1 |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 35 | 0.2 | 23 | 0.304348 | 16 | 1 | -4 |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 31 | 0.193548 | 23 | 0.26087 | 17 | 1 | 1.5 |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 32 | 0.1875 | 23 | 0.26087 | 17 | 1 | 1.5 |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 37 | 0.108108 | 23 | 0.173913 | 19 | 1 | 1.5 |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 36 | 0.194444 | 23 | 0.304348 | 16 | 2 | 1 |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 31 | 0.322581 | 23 | 0.434783 | 13 | 3 | -2 |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 45 | 0.133333 | 23 | 0.26087 | 17 | 1 | -2 |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 34 | 0.470588 | 52 | 0.307692 | 36 | 8 | -1.5 |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 30 | 0.5 | 52 | 0.288462 | 37 | 6 | -2 |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 31 | 0.483871 | 52 | 0.288462 | 37 | 10 | 1 |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 37 | 0.459459 | 52 | 0.326923 | 35 | 7 | 1 |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 35 | 0.457143 | 52 | 0.307692 | 36 | 6 | 0.5 |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 30 | 0.533333 | 52 | 0.307692 | 36 | 6 | -2 |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 45 | 0.4 | 52 | 0.346154 | 34 | 7 | 0 |
| limit_up_down_breadth_top | top | onset | 全A | 34 | 0.441176 | 52 | 0.288462 | 37 | 8 | -1 |
| limit_up_down_breadth_top | top | onset | 国证2000 | 30 | 0.466667 | 52 | 0.269231 | 38 | 7 | -0.5 |
| limit_up_down_breadth_top | top | onset | 中证1000 | 31 | 0.451613 | 52 | 0.269231 | 38 | 11 | 0.5 |
| limit_up_down_breadth_top | top | onset | 沪深300 | 37 | 0.432432 | 52 | 0.307692 | 36 | 7 | 0.5 |
| limit_up_down_breadth_top | top | onset | 中证500 | 35 | 0.457143 | 52 | 0.307692 | 36 | 8 | 0 |
| limit_up_down_breadth_top | top | onset | 微盘股 | 30 | 0.533333 | 52 | 0.307692 | 36 | 7 | -3 |
| limit_up_down_breadth_top | top | onset | 上证指数 | 45 | 0.422222 | 52 | 0.365385 | 33 | 8 | 0 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 126 |
| false_alarm | 609 |
| matched | 315 |
| missed_region | 663 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `limit_up_down_breadth_bottom/bottom/limit_up_down_breadth_v1_20120705_20260814/capped_confirmation`：区域窗口不完整：预测 0 个、确认 5 个；对应时点召回切片已从分母排除。 存在 10 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 7/10/46。
- `limit_up_down_breadth_bottom/bottom/limit_up_down_breadth_v1_20120705_20260814/onset`：区域窗口不完整：预测 0 个、确认 5 个；对应时点召回切片已从分母排除。 存在 10 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 7/11/46。
- `limit_up_down_breadth_top/top/limit_up_down_breadth_v1_20120705_20260814/capped_confirmation`：存在 50 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 42/48/113。
- `limit_up_down_breadth_top/top/limit_up_down_breadth_v1_20120705_20260814/onset`：存在 56 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 47/53/110。
