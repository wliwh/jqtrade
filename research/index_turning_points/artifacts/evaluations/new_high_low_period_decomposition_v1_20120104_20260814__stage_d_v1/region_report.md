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
| new_high_low_120_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 260 | 37 | 0.142308 | 301 | 37 | 0.122924 | 264 | 76 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 260 | 23 | 0.0884615 | 301 | 23 | 0.076412 | 278 | 76 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | window | 260 | 76 | 0.292308 | 301 | 76 | 0.252492 | 225 | 76 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | onset | loose | 260 | 36 | 0.138462 | 301 | 36 | 0.119601 | 265 | 76 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | onset | strict | 260 | 31 | 0.119231 | 301 | 31 | 0.10299 | 270 | 76 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | onset | window | 260 | 76 | 0.292308 | 301 | 76 | 0.252492 | 225 | 76 | -1 |
| new_high_low_120_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 255 | 49 | 0.192157 | 336 | 49 | 0.145833 | 287 | 76 | 0 |
| new_high_low_120_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 255 | 40 | 0.156863 | 336 | 40 | 0.119048 | 296 | 76 | 0 |
| new_high_low_120_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | window | 255 | 85 | 0.333333 | 336 | 85 | 0.252976 | 251 | 76 | 0 |
| new_high_low_120_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | onset | loose | 255 | 49 | 0.192157 | 336 | 49 | 0.145833 | 287 | 74 | 0 |
| new_high_low_120_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | onset | strict | 255 | 49 | 0.192157 | 336 | 49 | 0.145833 | 287 | 74 | 0 |
| new_high_low_120_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | onset | window | 255 | 87 | 0.341176 | 336 | 87 | 0.258929 | 249 | 74 | 1 |
| new_high_low_250_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 260 | 24 | 0.0923077 | 189 | 24 | 0.126984 | 165 | 47 | 0 |
| new_high_low_250_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 260 | 19 | 0.0730769 | 189 | 19 | 0.100529 | 170 | 47 | 0 |
| new_high_low_250_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | window | 260 | 51 | 0.196154 | 189 | 51 | 0.269841 | 138 | 47 | 0 |
| new_high_low_250_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | onset | loose | 260 | 26 | 0.1 | 189 | 26 | 0.137566 | 163 | 47 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | onset | strict | 260 | 25 | 0.0961538 | 189 | 25 | 0.132275 | 164 | 47 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | onset | window | 260 | 51 | 0.196154 | 189 | 51 | 0.269841 | 138 | 47 | -1 |
| new_high_low_250_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 255 | 22 | 0.0862745 | 168 | 22 | 0.130952 | 146 | 15 | 0 |
| new_high_low_250_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 255 | 21 | 0.0823529 | 168 | 21 | 0.125 | 147 | 15 | 0 |
| new_high_low_250_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | window | 255 | 45 | 0.176471 | 168 | 45 | 0.267857 | 123 | 15 | 0 |
| new_high_low_250_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | onset | loose | 255 | 22 | 0.0862745 | 168 | 22 | 0.130952 | 146 | 15 | -0.5 |
| new_high_low_250_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | onset | strict | 255 | 22 | 0.0862745 | 168 | 22 | 0.130952 | 146 | 15 | -0.5 |
| new_high_low_250_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | onset | window | 255 | 46 | 0.180392 | 168 | 46 | 0.27381 | 122 | 15 | 0.5 |
| new_high_low_60_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 260 | 61 | 0.234615 | 511 | 61 | 0.119374 | 450 | 101 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 260 | 44 | 0.169231 | 511 | 44 | 0.0861057 | 467 | 101 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | window | 260 | 135 | 0.519231 | 511 | 135 | 0.264188 | 376 | 101 | 1 |
| new_high_low_60_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | onset | loose | 260 | 70 | 0.269231 | 511 | 70 | 0.136986 | 441 | 101 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | onset | strict | 260 | 59 | 0.226923 | 511 | 59 | 0.11546 | 452 | 101 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | new_high_low_period_decomposition_v1_20120104_20260814 | onset | window | 260 | 135 | 0.519231 | 511 | 135 | 0.264188 | 376 | 101 | 0 |
| new_high_low_60_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 255 | 102 | 0.4 | 686 | 102 | 0.148688 | 584 | 196 | 0 |
| new_high_low_60_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 255 | 91 | 0.356863 | 686 | 91 | 0.132653 | 595 | 196 | 0 |
| new_high_low_60_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | capped_confirmation | window | 255 | 142 | 0.556863 | 686 | 142 | 0.206997 | 544 | 196 | 0 |
| new_high_low_60_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | onset | loose | 255 | 104 | 0.407843 | 686 | 104 | 0.151603 | 582 | 193 | 1 |
| new_high_low_60_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | onset | strict | 255 | 98 | 0.384314 | 686 | 98 | 0.142857 | 588 | 193 | 1 |
| new_high_low_60_breadth_reversal_top | top | new_high_low_period_decomposition_v1_20120104_20260814 | onset | window | 255 | 147 | 0.576471 | 686 | 147 | 0.214286 | 539 | 193 | 1 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 37 | 0.297297 | 43 | 0.255814 | 32 | 11 | 1 |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 32 | 0.375 | 43 | 0.27907 | 31 | 11 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 34 | 0.323529 | 43 | 0.255814 | 32 | 11 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 39 | 0.230769 | 43 | 0.209302 | 34 | 8 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 38 | 0.315789 | 43 | 0.27907 | 31 | 13 | -1 |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 33 | 0.30303 | 43 | 0.232558 | 33 | 10 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 47 | 0.234043 | 43 | 0.255814 | 32 | 12 | 0 |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 37 | 0.297297 | 43 | 0.255814 | 32 | 11 | 1 |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 32 | 0.375 | 43 | 0.27907 | 31 | 11 | -1 |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 34 | 0.323529 | 43 | 0.255814 | 32 | 11 | -1 |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 39 | 0.230769 | 43 | 0.209302 | 34 | 8 | -1 |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 38 | 0.315789 | 43 | 0.27907 | 31 | 13 | -1 |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 33 | 0.30303 | 43 | 0.232558 | 33 | 10 | -1 |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 47 | 0.234043 | 43 | 0.255814 | 32 | 12 | 0 |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 36 | 0.333333 | 48 | 0.25 | 36 | 14 | 0.5 |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 31 | 0.322581 | 48 | 0.208333 | 38 | 7 | -2 |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 33 | 0.30303 | 48 | 0.208333 | 38 | 11 | 2 |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 39 | 0.333333 | 48 | 0.270833 | 35 | 12 | 0 |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 37 | 0.351351 | 48 | 0.270833 | 35 | 11 | 2 |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 32 | 0.28125 | 48 | 0.1875 | 39 | 6 | -2 |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 47 | 0.382979 | 48 | 0.375 | 30 | 15 | 0 |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 36 | 0.361111 | 48 | 0.270833 | 35 | 13 | 1 |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 31 | 0.322581 | 48 | 0.208333 | 38 | 6 | 0.5 |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 33 | 0.333333 | 48 | 0.229167 | 37 | 10 | 1 |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 39 | 0.333333 | 48 | 0.270833 | 35 | 12 | -1 |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 37 | 0.378378 | 48 | 0.291667 | 34 | 11 | 1 |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 32 | 0.25 | 48 | 0.166667 | 40 | 6 | 1 |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 47 | 0.382979 | 48 | 0.375 | 30 | 16 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 37 | 0.189189 | 27 | 0.259259 | 20 | 7 | 0 |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 32 | 0.21875 | 27 | 0.259259 | 20 | 6 | 0 |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 34 | 0.205882 | 27 | 0.259259 | 20 | 6 | 0 |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 39 | 0.153846 | 27 | 0.222222 | 21 | 6 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 38 | 0.210526 | 27 | 0.296296 | 19 | 8 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 33 | 0.242424 | 27 | 0.296296 | 19 | 5 | 0 |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 47 | 0.170213 | 27 | 0.296296 | 19 | 9 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 37 | 0.189189 | 27 | 0.259259 | 20 | 7 | 0 |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 32 | 0.21875 | 27 | 0.259259 | 20 | 6 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 34 | 0.205882 | 27 | 0.259259 | 20 | 6 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 39 | 0.153846 | 27 | 0.222222 | 21 | 6 | -2 |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 38 | 0.210526 | 27 | 0.296296 | 19 | 8 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 33 | 0.242424 | 27 | 0.296296 | 19 | 5 | -1 |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 47 | 0.170213 | 27 | 0.296296 | 19 | 9 | -0.5 |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 36 | 0.166667 | 24 | 0.25 | 18 | 1 | 0.5 |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 31 | 0.16129 | 24 | 0.208333 | 19 | 2 | -2 |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 33 | 0.151515 | 24 | 0.208333 | 19 | 2 | -2 |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 39 | 0.179487 | 24 | 0.291667 | 17 | 4 | 1 |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 37 | 0.189189 | 24 | 0.291667 | 17 | 1 | 0 |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 32 | 0.1875 | 24 | 0.25 | 18 | 1 | -2 |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 47 | 0.191489 | 24 | 0.375 | 15 | 4 | 1 |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 36 | 0.166667 | 24 | 0.25 | 18 | 1 | 0.5 |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 31 | 0.16129 | 24 | 0.208333 | 19 | 2 | -3 |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 33 | 0.151515 | 24 | 0.208333 | 19 | 2 | 1 |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 39 | 0.205128 | 24 | 0.333333 | 16 | 3 | 1 |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 37 | 0.189189 | 24 | 0.291667 | 17 | 2 | 1 |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 32 | 0.15625 | 24 | 0.208333 | 19 | 1 | -1 |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 47 | 0.212766 | 24 | 0.416667 | 14 | 4 | 0.5 |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 37 | 0.567568 | 73 | 0.287671 | 52 | 15 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 32 | 0.5625 | 73 | 0.246575 | 55 | 16 | 0.5 |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 34 | 0.558824 | 73 | 0.260274 | 54 | 13 | 1 |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 39 | 0.435897 | 73 | 0.232877 | 56 | 9 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 38 | 0.5 | 73 | 0.260274 | 54 | 17 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 33 | 0.606061 | 73 | 0.273973 | 53 | 15 | 2.5 |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 47 | 0.446809 | 73 | 0.287671 | 52 | 16 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 37 | 0.567568 | 73 | 0.287671 | 52 | 15 | -1 |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 32 | 0.5625 | 73 | 0.246575 | 55 | 16 | -0.5 |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 34 | 0.558824 | 73 | 0.260274 | 54 | 13 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 39 | 0.435897 | 73 | 0.232877 | 56 | 9 | 0 |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 38 | 0.5 | 73 | 0.260274 | 54 | 17 | -1 |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 33 | 0.606061 | 73 | 0.273973 | 53 | 15 | 1 |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 47 | 0.446809 | 73 | 0.287671 | 52 | 16 | -1 |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 36 | 0.555556 | 98 | 0.204082 | 78 | 30 | 0 |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 31 | 0.580645 | 98 | 0.183673 | 80 | 25 | 0 |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 33 | 0.545455 | 98 | 0.183673 | 80 | 30 | -0.5 |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 39 | 0.564103 | 98 | 0.22449 | 76 | 28 | 0.5 |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 37 | 0.540541 | 98 | 0.204082 | 78 | 28 | 2 |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 32 | 0.53125 | 98 | 0.173469 | 81 | 23 | -2 |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 47 | 0.574468 | 98 | 0.27551 | 71 | 32 | 1 |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 36 | 0.611111 | 98 | 0.22449 | 76 | 29 | 1 |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 31 | 0.580645 | 98 | 0.183673 | 80 | 23 | 0.5 |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 33 | 0.575758 | 98 | 0.193878 | 79 | 29 | 1 |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 39 | 0.589744 | 98 | 0.234694 | 75 | 27 | 0 |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 37 | 0.567568 | 98 | 0.214286 | 77 | 28 | 1 |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 32 | 0.53125 | 98 | 0.173469 | 81 | 24 | 0 |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 47 | 0.574468 | 98 | 0.27551 | 71 | 33 | 0 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 1017 |
| false_alarm | 2289 |
| matched | 1076 |
| missed_region | 2014 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `new_high_low_120_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 76 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 23/37/76。
- `new_high_low_120_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/onset`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 76 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 31/36/76。
- `new_high_low_120_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：存在 76 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 40/49/85。
- `new_high_low_120_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/onset`：存在 74 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 49/49/87。
- `new_high_low_250_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 47 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 19/24/51。
- `new_high_low_250_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/onset`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 47 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 25/26/51。
- `new_high_low_250_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：存在 15 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 21/22/45。
- `new_high_low_250_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/onset`：存在 15 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 22/22/46。
- `new_high_low_60_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 101 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 44/61/135。
- `new_high_low_60_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/onset`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 101 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 59/70/135。
- `new_high_low_60_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：存在 196 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 91/102/142。
- `new_high_low_60_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/onset`：存在 193 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 98/104/147。
