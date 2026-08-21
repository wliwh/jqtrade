# 顶底区域定位评测

- 评测版本：`multi_period_ma_breadth_v1_20120104_20260814__stage_d_v1`
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
| multi_period_ma_breadth_bottom | bottom | multi_period_ma_breadth_v1_20120104_20260814 | capped_confirmation | loose | 260 | 27 | 0.103846 | 441 | 27 | 0.0612245 | 414 | 113 | 4 |
| multi_period_ma_breadth_bottom | bottom | multi_period_ma_breadth_v1_20120104_20260814 | capped_confirmation | strict | 260 | 14 | 0.0538462 | 441 | 14 | 0.031746 | 427 | 113 | 6 |
| multi_period_ma_breadth_bottom | bottom | multi_period_ma_breadth_v1_20120104_20260814 | capped_confirmation | window | 260 | 174 | 0.669231 | 441 | 174 | 0.394558 | 267 | 113 | 4 |
| multi_period_ma_breadth_bottom | bottom | multi_period_ma_breadth_v1_20120104_20260814 | onset | loose | 260 | 40 | 0.153846 | 441 | 40 | 0.0907029 | 401 | 108 | 3.5 |
| multi_period_ma_breadth_bottom | bottom | multi_period_ma_breadth_v1_20120104_20260814 | onset | strict | 260 | 24 | 0.0923077 | 441 | 24 | 0.0544218 | 417 | 108 | 2 |
| multi_period_ma_breadth_bottom | bottom | multi_period_ma_breadth_v1_20120104_20260814 | onset | window | 260 | 175 | 0.673077 | 441 | 175 | 0.396825 | 266 | 108 | 3 |
| multi_period_ma_breadth_top | top | multi_period_ma_breadth_v1_20120104_20260814 | capped_confirmation | loose | 255 | 37 | 0.145098 | 301 | 37 | 0.122924 | 264 | 79 | 3 |
| multi_period_ma_breadth_top | top | multi_period_ma_breadth_v1_20120104_20260814 | capped_confirmation | strict | 255 | 22 | 0.0862745 | 301 | 22 | 0.0730897 | 279 | 79 | 3 |
| multi_period_ma_breadth_top | top | multi_period_ma_breadth_v1_20120104_20260814 | capped_confirmation | window | 255 | 82 | 0.321569 | 301 | 82 | 0.272425 | 219 | 79 | 3 |
| multi_period_ma_breadth_top | top | multi_period_ma_breadth_v1_20120104_20260814 | onset | loose | 255 | 42 | 0.164706 | 301 | 42 | 0.139535 | 259 | 76 | 2 |
| multi_period_ma_breadth_top | top | multi_period_ma_breadth_v1_20120104_20260814 | onset | strict | 255 | 36 | 0.141176 | 301 | 36 | 0.119601 | 265 | 76 | 2 |
| multi_period_ma_breadth_top | top | multi_period_ma_breadth_v1_20120104_20260814 | onset | window | 255 | 82 | 0.321569 | 301 | 82 | 0.272425 | 219 | 76 | 2 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 37 | 0.675676 | 63 | 0.396825 | 38 | 16 | 4 |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 32 | 0.8125 | 63 | 0.412698 | 37 | 17 | 4 |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 34 | 0.735294 | 63 | 0.396825 | 38 | 17 | 4 |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 39 | 0.538462 | 63 | 0.333333 | 42 | 16 | 4 |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 38 | 0.684211 | 63 | 0.412698 | 37 | 16 | 4 |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 33 | 0.787879 | 63 | 0.412698 | 37 | 15 | 4 |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 47 | 0.531915 | 63 | 0.396825 | 38 | 16 | 4 |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 37 | 0.675676 | 63 | 0.396825 | 38 | 16 | 3 |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 32 | 0.8125 | 63 | 0.412698 | 37 | 16 | 3 |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 34 | 0.735294 | 63 | 0.396825 | 38 | 16 | 3 |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 39 | 0.564103 | 63 | 0.349206 | 41 | 14 | 3 |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 38 | 0.684211 | 63 | 0.412698 | 37 | 15 | 3 |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 33 | 0.787879 | 63 | 0.412698 | 37 | 15 | 3 |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 47 | 0.531915 | 63 | 0.396825 | 38 | 16 | 3 |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 36 | 0.333333 | 43 | 0.27907 | 31 | 12 | 3 |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 31 | 0.290323 | 43 | 0.209302 | 34 | 11 | 3 |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 33 | 0.30303 | 43 | 0.232558 | 33 | 12 | 3 |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 39 | 0.333333 | 43 | 0.302326 | 30 | 10 | 5 |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 37 | 0.297297 | 43 | 0.255814 | 32 | 10 | 3 |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 32 | 0.34375 | 43 | 0.255814 | 32 | 11 | 3 |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 47 | 0.340426 | 43 | 0.372093 | 27 | 13 | 4 |
| multi_period_ma_breadth_top | top | onset | 全A | 36 | 0.333333 | 43 | 0.27907 | 31 | 11 | 2 |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 31 | 0.290323 | 43 | 0.209302 | 34 | 11 | 2 |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 33 | 0.30303 | 43 | 0.232558 | 33 | 12 | 2 |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 39 | 0.333333 | 43 | 0.302326 | 30 | 10 | 5 |
| multi_period_ma_breadth_top | top | onset | 中证500 | 37 | 0.297297 | 43 | 0.255814 | 32 | 9 | 2 |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 32 | 0.34375 | 43 | 0.255814 | 32 | 10 | 2 |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 47 | 0.340426 | 43 | 0.372093 | 27 | 13 | 3 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 376 |
| false_alarm | 595 |
| matched | 513 |
| missed_region | 517 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `multi_period_ma_breadth_bottom/bottom/multi_period_ma_breadth_v1_20120104_20260814/capped_confirmation`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 113 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 14/27/174。
- `multi_period_ma_breadth_bottom/bottom/multi_period_ma_breadth_v1_20120104_20260814/onset`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 108 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 24/40/175。
- `multi_period_ma_breadth_top/top/multi_period_ma_breadth_v1_20120104_20260814/capped_confirmation`：存在 79 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 22/37/82。
- `multi_period_ma_breadth_top/top/multi_period_ma_breadth_v1_20120104_20260814/onset`：存在 76 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 36/42/82。
