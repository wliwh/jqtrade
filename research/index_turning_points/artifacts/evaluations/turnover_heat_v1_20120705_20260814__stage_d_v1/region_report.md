# 顶底区域定位评测

- 评测版本：`turnover_heat_v1_20120705_20260814__stage_d_v1`
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
| all_a_turnover_heat_top | top | turnover_heat_v1_20120705_20260814 | capped_confirmation | loose | 242 | 47 | 0.194215 | 357 | 47 | 0.131653 | 310 | 67 | -1 |
| all_a_turnover_heat_top | top | turnover_heat_v1_20120705_20260814 | capped_confirmation | strict | 242 | 32 | 0.132231 | 357 | 32 | 0.0896359 | 325 | 67 | 0 |
| all_a_turnover_heat_top | top | turnover_heat_v1_20120705_20260814 | capped_confirmation | window | 242 | 115 | 0.475207 | 357 | 115 | 0.322129 | 242 | 67 | 3 |
| all_a_turnover_heat_top | top | turnover_heat_v1_20120705_20260814 | onset | loose | 242 | 47 | 0.194215 | 357 | 47 | 0.131653 | 310 | 67 | 0 |
| all_a_turnover_heat_top | top | turnover_heat_v1_20120705_20260814 | onset | strict | 242 | 30 | 0.123967 | 357 | 30 | 0.0840336 | 327 | 67 | 0.5 |
| all_a_turnover_heat_top | top | turnover_heat_v1_20120705_20260814 | onset | window | 242 | 115 | 0.475207 | 357 | 115 | 0.322129 | 242 | 67 | 2 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 34 | 0.470588 | 51 | 0.313725 | 35 | 11 | 2 |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 30 | 0.466667 | 51 | 0.27451 | 37 | 10 | 4 |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 31 | 0.483871 | 51 | 0.294118 | 36 | 9 | 4 |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 37 | 0.486486 | 51 | 0.352941 | 33 | 12 | 4 |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 35 | 0.514286 | 51 | 0.352941 | 33 | 7 | -0.5 |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 30 | 0.433333 | 51 | 0.254902 | 38 | 8 | -2 |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 45 | 0.466667 | 51 | 0.411765 | 30 | 10 | 4 |
| all_a_turnover_heat_top | top | onset | 全A | 34 | 0.470588 | 51 | 0.313725 | 35 | 11 | 1 |
| all_a_turnover_heat_top | top | onset | 国证2000 | 30 | 0.466667 | 51 | 0.27451 | 37 | 10 | 3 |
| all_a_turnover_heat_top | top | onset | 中证1000 | 31 | 0.483871 | 51 | 0.294118 | 36 | 9 | 3 |
| all_a_turnover_heat_top | top | onset | 沪深300 | 37 | 0.486486 | 51 | 0.352941 | 33 | 12 | 3 |
| all_a_turnover_heat_top | top | onset | 中证500 | 35 | 0.514286 | 51 | 0.352941 | 33 | 7 | 0 |
| all_a_turnover_heat_top | top | onset | 微盘股 | 30 | 0.433333 | 51 | 0.254902 | 38 | 7 | -3 |
| all_a_turnover_heat_top | top | onset | 上证指数 | 45 | 0.466667 | 51 | 0.411765 | 30 | 11 | 3 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 134 |
| false_alarm | 350 |
| matched | 230 |
| missed_region | 254 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `all_a_turnover_heat_top/top/turnover_heat_v1_20120705_20260814/capped_confirmation`：存在 67 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 32/47/115。
- `all_a_turnover_heat_top/top/turnover_heat_v1_20120705_20260814/onset`：存在 67 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 30/47/115。
