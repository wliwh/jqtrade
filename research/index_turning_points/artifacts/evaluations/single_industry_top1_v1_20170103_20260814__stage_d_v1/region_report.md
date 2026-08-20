# 顶底区域定位评测

- 评测版本：`single_industry_top1_v1_20170103_20260814__stage_d_v1`
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
| single_industry_top1_801010 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 17 | 0.129771 | 196 | 17 | 0.0867347 | 179 | 35 | 0 |
| single_industry_top1_801010 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 14 | 0.10687 | 196 | 14 | 0.0714286 | 182 | 35 | -0.5 |
| single_industry_top1_801010 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 52 | 0.396947 | 196 | 52 | 0.265306 | 144 | 35 | 9 |
| single_industry_top1_801010 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 16 | 0.122137 | 196 | 16 | 0.0816327 | 180 | 39 | 0 |
| single_industry_top1_801010 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 13 | 0.0992366 | 196 | 13 | 0.0663265 | 183 | 39 | 0 |
| single_industry_top1_801010 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 52 | 0.396947 | 196 | 52 | 0.265306 | 144 | 39 | 8 |
| single_industry_top1_801020 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 63 | 14 | 0.222222 | 175 | 14 | 0.08 | 161 | 39 | 1 |
| single_industry_top1_801020 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 63 | 14 | 0.222222 | 175 | 14 | 0.08 | 161 | 39 | 1 |
| single_industry_top1_801020 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 63 | 21 | 0.333333 | 175 | 21 | 0.12 | 154 | 39 | 1 |
| single_industry_top1_801020 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 63 | 13 | 0.206349 | 175 | 13 | 0.0742857 | 162 | 38 | 0 |
| single_industry_top1_801020 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 63 | 13 | 0.206349 | 175 | 13 | 0.0742857 | 162 | 38 | 0 |
| single_industry_top1_801020 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 63 | 20 | 0.31746 | 175 | 20 | 0.114286 | 155 | 38 | 0 |
| single_industry_top1_801030 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 7 | 0.0534351 | 35 | 7 | 0.2 | 28 | 7 | 0 |
| single_industry_top1_801030 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 7 | 0.0534351 | 35 | 7 | 0.2 | 28 | 7 | 0 |
| single_industry_top1_801030 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 10 | 0.0763359 | 35 | 10 | 0.285714 | 25 | 7 | 0 |
| single_industry_top1_801030 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 0 | 0 | 35 | 0 | 0 | 35 | 7 |  |
| single_industry_top1_801030 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 0 | 0 | 35 | 0 | 0 | 35 | 7 |  |
| single_industry_top1_801030 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 10 | 0.0763359 | 35 | 10 | 0.285714 | 25 | 7 | -1 |
| single_industry_top1_801040 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 33 | 0.251908 | 441 | 33 | 0.0748299 | 408 | 50 | 0 |
| single_industry_top1_801040 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 33 | 0.251908 | 441 | 33 | 0.0748299 | 408 | 50 | 0 |
| single_industry_top1_801040 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 74 | 0.564885 | 441 | 74 | 0.1678 | 367 | 50 | -2 |
| single_industry_top1_801040 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 28 | 0.21374 | 441 | 28 | 0.0634921 | 413 | 48 | -1 |
| single_industry_top1_801040 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 24 | 0.183206 | 441 | 24 | 0.0544218 | 417 | 48 | -1 |
| single_industry_top1_801040 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 72 | 0.549618 | 441 | 72 | 0.163265 | 369 | 48 | -3 |
| single_industry_top1_801050 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 23 | 0.175573 | 343 | 23 | 0.0670554 | 320 | 66 | -3 |
| single_industry_top1_801050 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 22 | 0.167939 | 343 | 22 | 0.0641399 | 321 | 66 | -2.5 |
| single_industry_top1_801050 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 57 | 0.435115 | 343 | 57 | 0.166181 | 286 | 66 | -8 |
| single_industry_top1_801050 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 22 | 0.167939 | 343 | 22 | 0.0641399 | 321 | 50 | -3.5 |
| single_industry_top1_801050 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 22 | 0.167939 | 343 | 22 | 0.0641399 | 321 | 50 | -3.5 |
| single_industry_top1_801050 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 56 | 0.427481 | 343 | 56 | 0.163265 | 287 | 50 | -8.5 |
| single_industry_top1_801080 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 21 | 0.160305 | 385 | 21 | 0.0545455 | 364 | 45 | -2 |
| single_industry_top1_801080 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 20 | 0.152672 | 385 | 20 | 0.0519481 | 365 | 45 | -2 |
| single_industry_top1_801080 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 58 | 0.442748 | 385 | 58 | 0.150649 | 327 | 45 | 1 |
| single_industry_top1_801080 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 25 | 0.19084 | 385 | 25 | 0.0649351 | 360 | 45 | 0 |
| single_industry_top1_801080 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 25 | 0.19084 | 385 | 25 | 0.0649351 | 360 | 45 | 0 |
| single_industry_top1_801080 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 60 | 0.458015 | 385 | 60 | 0.155844 | 325 | 45 | 0 |
| single_industry_top1_801110 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 19 | 0.145038 | 126 | 19 | 0.150794 | 107 | 10 | -3 |
| single_industry_top1_801110 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 13 | 0.0992366 | 126 | 13 | 0.103175 | 113 | 10 | 0 |
| single_industry_top1_801110 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 35 | 0.267176 | 126 | 35 | 0.277778 | 91 | 10 | -3 |
| single_industry_top1_801110 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 11 | 0.0839695 | 126 | 11 | 0.0873016 | 115 | 12 | -10 |
| single_industry_top1_801110 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 10 | 0.0763359 | 126 | 10 | 0.0793651 | 116 | 12 | -11.5 |
| single_industry_top1_801110 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 32 | 0.244275 | 126 | 32 | 0.253968 | 94 | 12 | -2 |
| single_industry_top1_801120 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 23 | 0.175573 | 350 | 23 | 0.0657143 | 327 | 50 | -2 |
| single_industry_top1_801120 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 14 | 0.10687 | 350 | 14 | 0.04 | 336 | 50 | 0.5 |
| single_industry_top1_801120 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 64 | 0.48855 | 350 | 64 | 0.182857 | 286 | 50 | 2 |
| single_industry_top1_801120 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 25 | 0.19084 | 350 | 25 | 0.0714286 | 325 | 53 | 0 |
| single_industry_top1_801120 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 15 | 0.114504 | 350 | 15 | 0.0428571 | 335 | 53 | 1 |
| single_industry_top1_801120 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 68 | 0.519084 | 350 | 68 | 0.194286 | 282 | 53 | 1 |
| single_industry_top1_801130 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 9 | 0.0687023 | 105 | 9 | 0.0857143 | 96 | 16 | 2 |
| single_industry_top1_801130 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 9 | 0.0687023 | 105 | 9 | 0.0857143 | 96 | 16 | 2 |
| single_industry_top1_801130 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 34 | 0.259542 | 105 | 34 | 0.32381 | 71 | 16 | 2 |
| single_industry_top1_801130 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 15 | 0.114504 | 105 | 15 | 0.142857 | 90 | 17 | 1 |
| single_industry_top1_801130 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 15 | 0.114504 | 105 | 15 | 0.142857 | 90 | 17 | 1 |
| single_industry_top1_801130 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 34 | 0.259542 | 105 | 34 | 0.32381 | 71 | 17 | 1 |
| single_industry_top1_801140 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 8 | 0.0610687 | 49 | 8 | 0.163265 | 41 | 0 | 0 |
| single_industry_top1_801140 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 7 | 0.0534351 | 49 | 7 | 0.142857 | 42 | 0 | 0 |
| single_industry_top1_801140 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 25 | 0.19084 | 49 | 25 | 0.510204 | 24 | 0 | 0 |
| single_industry_top1_801140 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 1 | 0.00763359 | 49 | 1 | 0.0204082 | 48 | 0 | -5 |
| single_industry_top1_801140 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 0 | 0 | 49 | 0 | 0 | 49 | 0 |  |
| single_industry_top1_801140 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 22 | 0.167939 | 49 | 22 | 0.44898 | 27 | 0 | -1 |
| single_industry_top1_801150 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 29 | 0.221374 | 252 | 29 | 0.115079 | 223 | 17 | 0 |
| single_industry_top1_801150 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 28 | 0.21374 | 252 | 28 | 0.111111 | 224 | 17 | 0 |
| single_industry_top1_801150 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 59 | 0.450382 | 252 | 59 | 0.234127 | 193 | 17 | 0 |
| single_industry_top1_801150 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 24 | 0.183206 | 252 | 24 | 0.0952381 | 228 | 17 | -0.5 |
| single_industry_top1_801150 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 20 | 0.152672 | 252 | 20 | 0.0793651 | 232 | 17 | 0.5 |
| single_industry_top1_801150 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 60 | 0.458015 | 252 | 60 | 0.238095 | 192 | 17 | -1 |
| single_industry_top1_801160 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 26 | 0.198473 | 196 | 26 | 0.132653 | 170 | 28 | 0 |
| single_industry_top1_801160 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 24 | 0.183206 | 196 | 24 | 0.122449 | 172 | 28 | 0 |
| single_industry_top1_801160 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 57 | 0.435115 | 196 | 57 | 0.290816 | 139 | 28 | 6 |
| single_industry_top1_801160 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 18 | 0.137405 | 196 | 18 | 0.0918367 | 178 | 34 | 4.5 |
| single_industry_top1_801160 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 10 | 0.0763359 | 196 | 10 | 0.0510204 | 186 | 34 | 4.5 |
| single_industry_top1_801160 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 55 | 0.419847 | 196 | 55 | 0.280612 | 141 | 34 | 5 |
| single_industry_top1_801170 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 6 | 0.0458015 | 56 | 6 | 0.107143 | 50 | 2 | -10.5 |
| single_industry_top1_801170 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 2 | 0.0152672 | 56 | 2 | 0.0357143 | 54 | 2 | -15 |
| single_industry_top1_801170 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 21 | 0.160305 | 56 | 21 | 0.375 | 35 | 2 | 7 |
| single_industry_top1_801170 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 7 | 0.0534351 | 56 | 7 | 0.125 | 49 | 2 | 7 |
| single_industry_top1_801170 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 5 | 0.0381679 | 56 | 5 | 0.0892857 | 51 | 2 | 8 |
| single_industry_top1_801170 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 21 | 0.160305 | 56 | 21 | 0.375 | 35 | 2 | 7 |
| single_industry_top1_801180 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 36 | 0.274809 | 168 | 36 | 0.214286 | 132 | 20 | -1 |
| single_industry_top1_801180 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 33 | 0.251908 | 168 | 33 | 0.196429 | 135 | 20 | -1 |
| single_industry_top1_801180 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 54 | 0.412214 | 168 | 54 | 0.321429 | 114 | 20 | -1 |
| single_industry_top1_801180 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 30 | 0.229008 | 168 | 30 | 0.178571 | 138 | 20 | 1 |
| single_industry_top1_801180 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 19 | 0.145038 | 168 | 19 | 0.113095 | 149 | 20 | 1 |
| single_industry_top1_801180 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 54 | 0.412214 | 168 | 54 | 0.321429 | 114 | 20 | -2 |
| single_industry_top1_801200 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 9 | 0.0687023 | 84 | 9 | 0.107143 | 75 | 16 | 0 |
| single_industry_top1_801200 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 9 | 0.0687023 | 84 | 9 | 0.107143 | 75 | 16 | 0 |
| single_industry_top1_801200 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 19 | 0.145038 | 84 | 19 | 0.22619 | 65 | 16 | 0 |
| single_industry_top1_801200 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 2 | 0.0152672 | 84 | 2 | 0.0238095 | 82 | 16 | -1 |
| single_industry_top1_801200 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 2 | 0.0152672 | 84 | 2 | 0.0238095 | 82 | 16 | -1 |
| single_industry_top1_801200 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 19 | 0.145038 | 84 | 19 | 0.22619 | 65 | 16 | -1 |
| single_industry_top1_801210 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 34 | 0.259542 | 301 | 34 | 0.112957 | 267 | 57 | -1.5 |
| single_industry_top1_801210 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 33 | 0.251908 | 301 | 33 | 0.109635 | 268 | 57 | -2 |
| single_industry_top1_801210 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 67 | 0.51145 | 301 | 67 | 0.222591 | 234 | 57 | -2 |
| single_industry_top1_801210 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 35 | 0.267176 | 301 | 35 | 0.116279 | 266 | 57 | -1 |
| single_industry_top1_801210 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 35 | 0.267176 | 301 | 35 | 0.116279 | 266 | 57 | -1 |
| single_industry_top1_801210 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 62 | 0.473282 | 301 | 62 | 0.20598 | 239 | 57 | -2 |
| single_industry_top1_801230 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 24 | 0.183206 | 210 | 24 | 0.114286 | 186 | 39 | 0 |
| single_industry_top1_801230 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 24 | 0.183206 | 210 | 24 | 0.114286 | 186 | 39 | 0 |
| single_industry_top1_801230 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 58 | 0.442748 | 210 | 58 | 0.27619 | 152 | 39 | -4 |
| single_industry_top1_801230 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 19 | 0.145038 | 210 | 19 | 0.0904762 | 191 | 41 | -1 |
| single_industry_top1_801230 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 19 | 0.145038 | 210 | 19 | 0.0904762 | 191 | 41 | -1 |
| single_industry_top1_801230 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 58 | 0.442748 | 210 | 58 | 0.27619 | 152 | 41 | -5 |
| single_industry_top1_801710 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 11 | 0.0839695 | 77 | 11 | 0.142857 | 66 | 28 | -2 |
| single_industry_top1_801710 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 5 | 0.0381679 | 77 | 5 | 0.0649351 | 72 | 28 | -2 |
| single_industry_top1_801710 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 25 | 0.19084 | 77 | 25 | 0.324675 | 52 | 28 | -1 |
| single_industry_top1_801710 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 9 | 0.0687023 | 77 | 9 | 0.116883 | 68 | 28 | -3 |
| single_industry_top1_801710 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 3 | 0.0229008 | 77 | 3 | 0.038961 | 74 | 28 | -1 |
| single_industry_top1_801710 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 25 | 0.19084 | 77 | 25 | 0.324675 | 52 | 28 | -2 |
| single_industry_top1_801720 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 11 | 0.0839695 | 91 | 11 | 0.120879 | 80 | 6 | 0 |
| single_industry_top1_801720 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 11 | 0.0839695 | 91 | 11 | 0.120879 | 80 | 6 | 0 |
| single_industry_top1_801720 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 30 | 0.229008 | 91 | 30 | 0.32967 | 61 | 6 | 1.5 |
| single_industry_top1_801720 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 4 | 0.0305344 | 91 | 4 | 0.043956 | 87 | 11 | 2 |
| single_industry_top1_801720 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 4 | 0.0305344 | 91 | 4 | 0.043956 | 87 | 11 | 2 |
| single_industry_top1_801720 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 27 | 0.206107 | 91 | 27 | 0.296703 | 64 | 11 | 4 |
| single_industry_top1_801730 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 14 | 0.10687 | 168 | 14 | 0.0833333 | 154 | 36 | 0.5 |
| single_industry_top1_801730 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 14 | 0.10687 | 168 | 14 | 0.0833333 | 154 | 36 | 0.5 |
| single_industry_top1_801730 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 33 | 0.251908 | 168 | 33 | 0.196429 | 135 | 36 | 3 |
| single_industry_top1_801730 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 7 | 0.0534351 | 168 | 7 | 0.0416667 | 161 | 36 | 2 |
| single_industry_top1_801730 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 7 | 0.0534351 | 168 | 7 | 0.0416667 | 161 | 36 | 2 |
| single_industry_top1_801730 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 33 | 0.251908 | 168 | 33 | 0.196429 | 135 | 36 | 2 |
| single_industry_top1_801740 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 43 | 0.328244 | 581 | 43 | 0.0740103 | 538 | 70 | -2 |
| single_industry_top1_801740 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 38 | 0.290076 | 581 | 38 | 0.0654045 | 543 | 70 | -2 |
| single_industry_top1_801740 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 94 | 0.717557 | 581 | 94 | 0.16179 | 487 | 70 | -3 |
| single_industry_top1_801740 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 42 | 0.320611 | 581 | 42 | 0.0722892 | 539 | 71 | -3 |
| single_industry_top1_801740 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 37 | 0.282443 | 581 | 37 | 0.0636833 | 544 | 71 | -3 |
| single_industry_top1_801740 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 91 | 0.694656 | 581 | 91 | 0.156627 | 490 | 71 | -4 |
| single_industry_top1_801750 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 46 | 0.351145 | 301 | 46 | 0.152824 | 255 | 61 | -1 |
| single_industry_top1_801750 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 42 | 0.320611 | 301 | 42 | 0.139535 | 259 | 61 | -0.5 |
| single_industry_top1_801750 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 66 | 0.503817 | 301 | 66 | 0.219269 | 235 | 61 | -1.5 |
| single_industry_top1_801750 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 38 | 0.290076 | 301 | 38 | 0.126246 | 263 | 62 | -2 |
| single_industry_top1_801750 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 36 | 0.274809 | 301 | 36 | 0.119601 | 265 | 62 | -1.5 |
| single_industry_top1_801750 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 66 | 0.503817 | 301 | 66 | 0.219269 | 235 | 62 | -2.5 |
| single_industry_top1_801760 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 27 | 0.206107 | 245 | 27 | 0.110204 | 218 | 41 | 1 |
| single_industry_top1_801760 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 22 | 0.167939 | 245 | 22 | 0.0897959 | 223 | 41 | 0 |
| single_industry_top1_801760 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 50 | 0.381679 | 245 | 50 | 0.204082 | 195 | 41 | 0 |
| single_industry_top1_801760 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 20 | 0.152672 | 245 | 20 | 0.0816327 | 225 | 43 | 0.5 |
| single_industry_top1_801760 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 17 | 0.129771 | 245 | 17 | 0.0693878 | 228 | 43 | 0 |
| single_industry_top1_801760 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 50 | 0.381679 | 245 | 50 | 0.204082 | 195 | 43 | 0 |
| single_industry_top1_801770 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 30 | 0.229008 | 343 | 30 | 0.0874636 | 313 | 73 | 1.5 |
| single_industry_top1_801770 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 23 | 0.175573 | 343 | 23 | 0.0670554 | 320 | 73 | 1 |
| single_industry_top1_801770 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 86 | 0.656489 | 343 | 86 | 0.250729 | 257 | 73 | -2 |
| single_industry_top1_801770 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 27 | 0.206107 | 343 | 27 | 0.0787172 | 316 | 74 | 0 |
| single_industry_top1_801770 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 23 | 0.175573 | 343 | 23 | 0.0670554 | 320 | 74 | 0 |
| single_industry_top1_801770 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 85 | 0.648855 | 343 | 85 | 0.247813 | 258 | 74 | -2 |
| single_industry_top1_801780 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 59 | 0.450382 | 861 | 59 | 0.068525 | 802 | 176 | 0 |
| single_industry_top1_801780 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 53 | 0.40458 | 861 | 53 | 0.0615563 | 808 | 176 | 0 |
| single_industry_top1_801780 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 116 | 0.885496 | 861 | 116 | 0.134727 | 745 | 176 | 0 |
| single_industry_top1_801780 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 60 | 0.458015 | 861 | 60 | 0.0696864 | 801 | 179 | 0 |
| single_industry_top1_801780 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 53 | 0.40458 | 861 | 53 | 0.0615563 | 808 | 179 | 1 |
| single_industry_top1_801780 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 116 | 0.885496 | 861 | 116 | 0.134727 | 745 | 179 | -1 |
| single_industry_top1_801790 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 26 | 0.198473 | 448 | 26 | 0.0580357 | 422 | 52 | -1 |
| single_industry_top1_801790 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 26 | 0.198473 | 448 | 26 | 0.0580357 | 422 | 52 | -1 |
| single_industry_top1_801790 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 70 | 0.534351 | 448 | 70 | 0.15625 | 378 | 52 | -4 |
| single_industry_top1_801790 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 23 | 0.175573 | 448 | 23 | 0.0513393 | 425 | 48 | -1 |
| single_industry_top1_801790 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 22 | 0.167939 | 448 | 22 | 0.0491071 | 426 | 48 | -1 |
| single_industry_top1_801790 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 70 | 0.534351 | 448 | 70 | 0.15625 | 378 | 48 | -5 |
| single_industry_top1_801880 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 9 | 0.0687023 | 119 | 9 | 0.0756303 | 110 | 2 | 0 |
| single_industry_top1_801880 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 9 | 0.0687023 | 119 | 9 | 0.0756303 | 110 | 2 | 0 |
| single_industry_top1_801880 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 19 | 0.145038 | 119 | 19 | 0.159664 | 100 | 2 | -7 |
| single_industry_top1_801880 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 2 | 0.0152672 | 119 | 2 | 0.0168067 | 117 | 6 | 1 |
| single_industry_top1_801880 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 2 | 0.0152672 | 119 | 2 | 0.0168067 | 117 | 6 | 1 |
| single_industry_top1_801880 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 19 | 0.145038 | 119 | 19 | 0.159664 | 100 | 6 | -8 |
| single_industry_top1_801890 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 131 | 9 | 0.0687023 | 56 | 9 | 0.160714 | 47 | 0 | 0 |
| single_industry_top1_801890 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 131 | 9 | 0.0687023 | 56 | 9 | 0.160714 | 47 | 0 | 0 |
| single_industry_top1_801890 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 131 | 17 | 0.129771 | 56 | 17 | 0.303571 | 39 | 0 | 0 |
| single_industry_top1_801890 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 131 | 2 | 0.0152672 | 56 | 2 | 0.0357143 | 54 | 0 | 5 |
| single_industry_top1_801890 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 131 | 2 | 0.0152672 | 56 | 2 | 0.0357143 | 54 | 0 | 5 |
| single_industry_top1_801890 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 131 | 17 | 0.129771 | 56 | 17 | 0.303571 | 39 | 0 | -1 |
| single_industry_top1_801950 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 68 | 28 | 0.411765 | 357 | 28 | 0.0784314 | 329 | 90 | -2 |
| single_industry_top1_801950 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 68 | 27 | 0.397059 | 357 | 27 | 0.0756303 | 330 | 90 | -2 |
| single_industry_top1_801950 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 68 | 58 | 0.852941 | 357 | 58 | 0.162465 | 299 | 90 | -3 |
| single_industry_top1_801950 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 68 | 26 | 0.382353 | 357 | 26 | 0.0728291 | 331 | 91 | 1 |
| single_industry_top1_801950 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 68 | 25 | 0.367647 | 357 | 25 | 0.070028 | 332 | 91 | 1 |
| single_industry_top1_801950 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 68 | 57 | 0.838235 | 357 | 57 | 0.159664 | 300 | 91 | -4 |
| single_industry_top1_801960 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 68 | 13 | 0.191176 | 91 | 13 | 0.142857 | 78 | 18 | 0 |
| single_industry_top1_801960 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 68 | 11 | 0.161765 | 91 | 11 | 0.120879 | 80 | 18 | 0 |
| single_industry_top1_801960 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 68 | 23 | 0.338235 | 91 | 23 | 0.252747 | 68 | 18 | -6 |
| single_industry_top1_801960 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 68 | 6 | 0.0882353 | 91 | 6 | 0.0659341 | 85 | 19 | -5 |
| single_industry_top1_801960 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 68 | 5 | 0.0735294 | 91 | 5 | 0.0549451 | 86 | 19 | -5 |
| single_industry_top1_801960 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 68 | 21 | 0.308824 | 91 | 21 | 0.230769 | 70 | 19 | -5 |
| single_industry_top1_801970 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 68 | 6 | 0.0882353 | 98 | 6 | 0.0612245 | 92 | 20 | -1 |
| single_industry_top1_801970 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 68 | 6 | 0.0882353 | 98 | 6 | 0.0612245 | 92 | 20 | -1 |
| single_industry_top1_801970 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 68 | 28 | 0.411765 | 98 | 28 | 0.285714 | 70 | 20 | -1 |
| single_industry_top1_801970 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 68 | 6 | 0.0882353 | 98 | 6 | 0.0612245 | 92 | 20 | -2 |
| single_industry_top1_801970 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 68 | 6 | 0.0882353 | 98 | 6 | 0.0612245 | 92 | 20 | -2 |
| single_industry_top1_801970 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 68 | 28 | 0.411765 | 98 | 28 | 0.285714 | 70 | 20 | -2 |
| single_industry_top1_801980 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | loose | 68 | 18 | 0.264706 | 98 | 18 | 0.183673 | 80 | 12 | -5 |
| single_industry_top1_801980 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | strict | 68 | 11 | 0.161765 | 98 | 11 | 0.112245 | 87 | 12 | 1 |
| single_industry_top1_801980 | top | single_industry_top1_v1_20170103_20260814 | capped_confirmation | window | 68 | 37 | 0.544118 | 98 | 37 | 0.377551 | 61 | 12 | -5 |
| single_industry_top1_801980 | top | single_industry_top1_v1_20170103_20260814 | onset | loose | 68 | 18 | 0.264706 | 98 | 18 | 0.183673 | 80 | 12 | -6 |
| single_industry_top1_801980 | top | single_industry_top1_v1_20170103_20260814 | onset | strict | 68 | 11 | 0.161765 | 98 | 11 | 0.112245 | 87 | 12 | 0 |
| single_industry_top1_801980 | top | single_industry_top1_v1_20170103_20260814 | onset | window | 68 | 37 | 0.544118 | 98 | 37 | 0.377551 | 61 | 12 | -3 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_industry_top1_801010 | top | capped_confirmation | 全A | 17 | 0.411765 | 28 | 0.25 | 21 | 4 | 11 |
| single_industry_top1_801010 | top | capped_confirmation | 国证2000 | 19 | 0.473684 | 28 | 0.321429 | 19 | 6 | 7 |
| single_industry_top1_801010 | top | capped_confirmation | 中证1000 | 17 | 0.470588 | 28 | 0.285714 | 20 | 6 | 10 |
| single_industry_top1_801010 | top | capped_confirmation | 沪深300 | 19 | 0.315789 | 28 | 0.214286 | 22 | 4 | 7 |
| single_industry_top1_801010 | top | capped_confirmation | 中证500 | 19 | 0.368421 | 28 | 0.25 | 21 | 6 | 9 |
| single_industry_top1_801010 | top | capped_confirmation | 微盘股 | 18 | 0.333333 | 28 | 0.214286 | 22 | 2 | 2.5 |
| single_industry_top1_801010 | top | capped_confirmation | 上证指数 | 22 | 0.409091 | 28 | 0.321429 | 19 | 7 | 12 |
| single_industry_top1_801010 | top | onset | 全A | 17 | 0.411765 | 28 | 0.25 | 21 | 5 | 13 |
| single_industry_top1_801010 | top | onset | 国证2000 | 19 | 0.473684 | 28 | 0.321429 | 19 | 6 | 6 |
| single_industry_top1_801010 | top | onset | 中证1000 | 17 | 0.470588 | 28 | 0.285714 | 20 | 7 | 9 |
| single_industry_top1_801010 | top | onset | 沪深300 | 19 | 0.315789 | 28 | 0.214286 | 22 | 4 | 6 |
| single_industry_top1_801010 | top | onset | 中证500 | 19 | 0.368421 | 28 | 0.25 | 21 | 7 | 8 |
| single_industry_top1_801010 | top | onset | 微盘股 | 18 | 0.333333 | 28 | 0.214286 | 22 | 2 | 1.5 |
| single_industry_top1_801010 | top | onset | 上证指数 | 22 | 0.409091 | 28 | 0.321429 | 19 | 8 | 11 |
| single_industry_top1_801020 | top | capped_confirmation | 全A | 8 | 0.25 | 25 | 0.08 | 23 | 4 | 0.5 |
| single_industry_top1_801020 | top | capped_confirmation | 国证2000 | 8 | 0.125 | 25 | 0.04 | 24 | 4 | 1 |
| single_industry_top1_801020 | top | capped_confirmation | 中证1000 | 9 | 0.111111 | 25 | 0.04 | 24 | 4 | 0 |
| single_industry_top1_801020 | top | capped_confirmation | 沪深300 | 10 | 0.6 | 25 | 0.24 | 19 | 6 | 1 |
| single_industry_top1_801020 | top | capped_confirmation | 中证500 | 10 | 0.3 | 25 | 0.12 | 22 | 7 | 0 |
| single_industry_top1_801020 | top | capped_confirmation | 微盘股 | 6 | 0.333333 | 25 | 0.08 | 23 | 4 | 7 |
| single_industry_top1_801020 | top | capped_confirmation | 上证指数 | 12 | 0.5 | 25 | 0.24 | 19 | 10 | 1 |
| single_industry_top1_801020 | top | onset | 全A | 8 | 0.25 | 25 | 0.08 | 23 | 4 | -0.5 |
| single_industry_top1_801020 | top | onset | 国证2000 | 8 | 0.125 | 25 | 0.04 | 24 | 3 | 0 |
| single_industry_top1_801020 | top | onset | 中证1000 | 9 | 0.111111 | 25 | 0.04 | 24 | 4 | -1 |
| single_industry_top1_801020 | top | onset | 沪深300 | 10 | 0.5 | 25 | 0.2 | 20 | 6 | 0 |
| single_industry_top1_801020 | top | onset | 中证500 | 10 | 0.3 | 25 | 0.12 | 22 | 7 | -1 |
| single_industry_top1_801020 | top | onset | 微盘股 | 6 | 0.333333 | 25 | 0.08 | 23 | 4 | 6 |
| single_industry_top1_801020 | top | onset | 上证指数 | 12 | 0.5 | 25 | 0.24 | 19 | 10 | 0 |
| single_industry_top1_801030 | top | capped_confirmation | 全A | 17 | 0.117647 | 5 | 0.4 | 3 | 1 | 6 |
| single_industry_top1_801030 | top | capped_confirmation | 国证2000 | 19 | 0.0526316 | 5 | 0.2 | 4 | 1 | 0 |
| single_industry_top1_801030 | top | capped_confirmation | 中证1000 | 17 | 0.0588235 | 5 | 0.2 | 4 | 1 | 0 |
| single_industry_top1_801030 | top | capped_confirmation | 沪深300 | 19 | 0.105263 | 5 | 0.4 | 3 | 1 | 6 |
| single_industry_top1_801030 | top | capped_confirmation | 中证500 | 19 | 0.0526316 | 5 | 0.2 | 4 | 1 | 0 |
| single_industry_top1_801030 | top | capped_confirmation | 微盘股 | 18 | 0.0555556 | 5 | 0.2 | 4 | 1 | 0 |
| single_industry_top1_801030 | top | capped_confirmation | 上证指数 | 22 | 0.0909091 | 5 | 0.4 | 3 | 1 | 6 |
| single_industry_top1_801030 | top | onset | 全A | 17 | 0.117647 | 5 | 0.4 | 3 | 1 | 5 |
| single_industry_top1_801030 | top | onset | 国证2000 | 19 | 0.0526316 | 5 | 0.2 | 4 | 1 | -1 |
| single_industry_top1_801030 | top | onset | 中证1000 | 17 | 0.0588235 | 5 | 0.2 | 4 | 1 | -1 |
| single_industry_top1_801030 | top | onset | 沪深300 | 19 | 0.105263 | 5 | 0.4 | 3 | 1 | 5 |
| single_industry_top1_801030 | top | onset | 中证500 | 19 | 0.0526316 | 5 | 0.2 | 4 | 1 | -1 |
| single_industry_top1_801030 | top | onset | 微盘股 | 18 | 0.0555556 | 5 | 0.2 | 4 | 1 | -1 |
| single_industry_top1_801030 | top | onset | 上证指数 | 22 | 0.0909091 | 5 | 0.4 | 3 | 1 | 5 |
| single_industry_top1_801040 | top | capped_confirmation | 全A | 17 | 0.588235 | 63 | 0.15873 | 53 | 9 | -2.5 |
| single_industry_top1_801040 | top | capped_confirmation | 国证2000 | 19 | 0.578947 | 63 | 0.174603 | 52 | 6 | -9 |
| single_industry_top1_801040 | top | capped_confirmation | 中证1000 | 17 | 0.588235 | 63 | 0.15873 | 53 | 7 | -6 |
| single_industry_top1_801040 | top | capped_confirmation | 沪深300 | 19 | 0.631579 | 63 | 0.190476 | 51 | 5 | -1 |
| single_industry_top1_801040 | top | capped_confirmation | 中证500 | 19 | 0.473684 | 63 | 0.142857 | 54 | 9 | -3 |
| single_industry_top1_801040 | top | capped_confirmation | 微盘股 | 18 | 0.5 | 63 | 0.142857 | 54 | 5 | 0 |
| single_industry_top1_801040 | top | capped_confirmation | 上证指数 | 22 | 0.590909 | 63 | 0.206349 | 50 | 9 | -1 |
| single_industry_top1_801040 | top | onset | 全A | 17 | 0.588235 | 63 | 0.15873 | 53 | 9 | -3.5 |
| single_industry_top1_801040 | top | onset | 国证2000 | 19 | 0.526316 | 63 | 0.15873 | 53 | 6 | -7 |
| single_industry_top1_801040 | top | onset | 中证1000 | 17 | 0.529412 | 63 | 0.142857 | 54 | 7 | -7 |
| single_industry_top1_801040 | top | onset | 沪深300 | 19 | 0.631579 | 63 | 0.190476 | 51 | 5 | -2 |
| single_industry_top1_801040 | top | onset | 中证500 | 19 | 0.473684 | 63 | 0.142857 | 54 | 8 | -4 |
| single_industry_top1_801040 | top | onset | 微盘股 | 18 | 0.5 | 63 | 0.142857 | 54 | 5 | -1 |
| single_industry_top1_801040 | top | onset | 上证指数 | 22 | 0.590909 | 63 | 0.206349 | 50 | 8 | -2 |
| single_industry_top1_801050 | top | capped_confirmation | 全A | 17 | 0.529412 | 49 | 0.183673 | 40 | 10 | -10 |
| single_industry_top1_801050 | top | capped_confirmation | 国证2000 | 19 | 0.368421 | 49 | 0.142857 | 42 | 10 | -6 |
| single_industry_top1_801050 | top | capped_confirmation | 中证1000 | 17 | 0.588235 | 49 | 0.204082 | 39 | 9 | -8 |
| single_industry_top1_801050 | top | capped_confirmation | 沪深300 | 19 | 0.421053 | 49 | 0.163265 | 41 | 11 | -7 |
| single_industry_top1_801050 | top | capped_confirmation | 中证500 | 19 | 0.368421 | 49 | 0.142857 | 42 | 7 | -6 |
| single_industry_top1_801050 | top | capped_confirmation | 微盘股 | 18 | 0.444444 | 49 | 0.163265 | 41 | 10 | -5 |
| single_industry_top1_801050 | top | capped_confirmation | 上证指数 | 22 | 0.363636 | 49 | 0.163265 | 41 | 9 | -9 |
| single_industry_top1_801050 | top | onset | 全A | 17 | 0.529412 | 49 | 0.183673 | 40 | 7 | -11 |
| single_industry_top1_801050 | top | onset | 国证2000 | 19 | 0.368421 | 49 | 0.142857 | 42 | 7 | -7 |
| single_industry_top1_801050 | top | onset | 中证1000 | 17 | 0.588235 | 49 | 0.204082 | 39 | 6 | -9 |
| single_industry_top1_801050 | top | onset | 沪深300 | 19 | 0.421053 | 49 | 0.163265 | 41 | 10 | -8 |
| single_industry_top1_801050 | top | onset | 中证500 | 19 | 0.368421 | 49 | 0.142857 | 42 | 5 | -7 |
| single_industry_top1_801050 | top | onset | 微盘股 | 18 | 0.388889 | 49 | 0.142857 | 42 | 9 | -3 |
| single_industry_top1_801050 | top | onset | 上证指数 | 22 | 0.363636 | 49 | 0.163265 | 41 | 6 | -10 |
| single_industry_top1_801080 | top | capped_confirmation | 全A | 17 | 0.470588 | 55 | 0.145455 | 47 | 7 | 1.5 |
| single_industry_top1_801080 | top | capped_confirmation | 国证2000 | 19 | 0.421053 | 55 | 0.145455 | 47 | 7 | -4 |
| single_industry_top1_801080 | top | capped_confirmation | 中证1000 | 17 | 0.470588 | 55 | 0.145455 | 47 | 6 | -2.5 |
| single_industry_top1_801080 | top | capped_confirmation | 沪深300 | 19 | 0.473684 | 55 | 0.163636 | 46 | 6 | 2 |
| single_industry_top1_801080 | top | capped_confirmation | 中证500 | 19 | 0.421053 | 55 | 0.145455 | 47 | 6 | -1 |
| single_industry_top1_801080 | top | capped_confirmation | 微盘股 | 18 | 0.388889 | 55 | 0.127273 | 48 | 5 | 1 |
| single_industry_top1_801080 | top | capped_confirmation | 上证指数 | 22 | 0.454545 | 55 | 0.181818 | 45 | 8 | 9.5 |
| single_industry_top1_801080 | top | onset | 全A | 17 | 0.470588 | 55 | 0.145455 | 47 | 7 | 1.5 |
| single_industry_top1_801080 | top | onset | 国证2000 | 19 | 0.473684 | 55 | 0.163636 | 46 | 7 | -5 |
| single_industry_top1_801080 | top | onset | 中证1000 | 17 | 0.529412 | 55 | 0.163636 | 46 | 6 | -2 |
| single_industry_top1_801080 | top | onset | 沪深300 | 19 | 0.473684 | 55 | 0.163636 | 46 | 6 | 3 |
| single_industry_top1_801080 | top | onset | 中证500 | 19 | 0.421053 | 55 | 0.145455 | 47 | 6 | -2 |
| single_industry_top1_801080 | top | onset | 微盘股 | 18 | 0.388889 | 55 | 0.127273 | 48 | 5 | 0 |
| single_industry_top1_801080 | top | onset | 上证指数 | 22 | 0.454545 | 55 | 0.181818 | 45 | 8 | 9 |
| single_industry_top1_801110 | top | capped_confirmation | 全A | 17 | 0.294118 | 18 | 0.277778 | 13 | 1 | -9 |
| single_industry_top1_801110 | top | capped_confirmation | 国证2000 | 19 | 0.263158 | 18 | 0.277778 | 13 | 2 | 0 |
| single_industry_top1_801110 | top | capped_confirmation | 中证1000 | 17 | 0.294118 | 18 | 0.277778 | 13 | 2 | 0 |
| single_industry_top1_801110 | top | capped_confirmation | 沪深300 | 19 | 0.157895 | 18 | 0.166667 | 15 | 0 | -3 |
| single_industry_top1_801110 | top | capped_confirmation | 中证500 | 19 | 0.263158 | 18 | 0.277778 | 13 | 2 | -3 |
| single_industry_top1_801110 | top | capped_confirmation | 微盘股 | 18 | 0.333333 | 18 | 0.333333 | 12 | 2 | 0 |
| single_industry_top1_801110 | top | capped_confirmation | 上证指数 | 22 | 0.272727 | 18 | 0.333333 | 12 | 1 | -8 |
| single_industry_top1_801110 | top | onset | 全A | 17 | 0.294118 | 18 | 0.277778 | 13 | 1 | -10 |
| single_industry_top1_801110 | top | onset | 国证2000 | 19 | 0.210526 | 18 | 0.222222 | 14 | 2 | 2 |
| single_industry_top1_801110 | top | onset | 中证1000 | 17 | 0.294118 | 18 | 0.277778 | 13 | 2 | -1 |
| single_industry_top1_801110 | top | onset | 沪深300 | 19 | 0.105263 | 18 | 0.111111 | 16 | 1 | -2.5 |
| single_industry_top1_801110 | top | onset | 中证500 | 19 | 0.263158 | 18 | 0.277778 | 13 | 2 | -4 |
| single_industry_top1_801110 | top | onset | 微盘股 | 18 | 0.333333 | 18 | 0.333333 | 12 | 2 | -1 |
| single_industry_top1_801110 | top | onset | 上证指数 | 22 | 0.227273 | 18 | 0.277778 | 13 | 2 | -4 |
| single_industry_top1_801120 | top | capped_confirmation | 全A | 17 | 0.588235 | 50 | 0.2 | 40 | 9 | 0.5 |
| single_industry_top1_801120 | top | capped_confirmation | 国证2000 | 19 | 0.526316 | 50 | 0.2 | 40 | 7 | 3 |
| single_industry_top1_801120 | top | capped_confirmation | 中证1000 | 17 | 0.470588 | 50 | 0.16 | 42 | 7 | 2.5 |
| single_industry_top1_801120 | top | capped_confirmation | 沪深300 | 19 | 0.368421 | 50 | 0.14 | 43 | 5 | 5 |
| single_industry_top1_801120 | top | capped_confirmation | 中证500 | 19 | 0.473684 | 50 | 0.18 | 41 | 6 | -1 |
| single_industry_top1_801120 | top | capped_confirmation | 微盘股 | 18 | 0.5 | 50 | 0.18 | 41 | 6 | -4 |
| single_industry_top1_801120 | top | capped_confirmation | 上证指数 | 22 | 0.5 | 50 | 0.22 | 39 | 10 | 2 |
| single_industry_top1_801120 | top | onset | 全A | 17 | 0.588235 | 50 | 0.2 | 40 | 9 | -0.5 |
| single_industry_top1_801120 | top | onset | 国证2000 | 19 | 0.526316 | 50 | 0.2 | 40 | 7 | 2 |
| single_industry_top1_801120 | top | onset | 中证1000 | 17 | 0.529412 | 50 | 0.18 | 41 | 8 | 2 |
| single_industry_top1_801120 | top | onset | 沪深300 | 19 | 0.421053 | 50 | 0.16 | 42 | 5 | 4.5 |
| single_industry_top1_801120 | top | onset | 中证500 | 19 | 0.578947 | 50 | 0.22 | 39 | 7 | 2 |
| single_industry_top1_801120 | top | onset | 微盘股 | 18 | 0.5 | 50 | 0.18 | 41 | 6 | -5 |
| single_industry_top1_801120 | top | onset | 上证指数 | 22 | 0.5 | 50 | 0.22 | 39 | 11 | 1 |
| single_industry_top1_801130 | top | capped_confirmation | 全A | 17 | 0.235294 | 15 | 0.266667 | 11 | 2 | 2 |
| single_industry_top1_801130 | top | capped_confirmation | 国证2000 | 19 | 0.263158 | 15 | 0.333333 | 10 | 3 | 1 |
| single_industry_top1_801130 | top | capped_confirmation | 中证1000 | 17 | 0.235294 | 15 | 0.266667 | 11 | 2 | 4 |
| single_industry_top1_801130 | top | capped_confirmation | 沪深300 | 19 | 0.210526 | 15 | 0.266667 | 11 | 2 | 2 |
| single_industry_top1_801130 | top | capped_confirmation | 中证500 | 19 | 0.263158 | 15 | 0.333333 | 10 | 2 | 2 |
| single_industry_top1_801130 | top | capped_confirmation | 微盘股 | 18 | 0.222222 | 15 | 0.266667 | 11 | 3 | 1.5 |
| single_industry_top1_801130 | top | capped_confirmation | 上证指数 | 22 | 0.363636 | 15 | 0.533333 | 7 | 2 | 3 |
| single_industry_top1_801130 | top | onset | 全A | 17 | 0.235294 | 15 | 0.266667 | 11 | 2 | 1 |
| single_industry_top1_801130 | top | onset | 国证2000 | 19 | 0.263158 | 15 | 0.333333 | 10 | 3 | 0 |
| single_industry_top1_801130 | top | onset | 中证1000 | 17 | 0.235294 | 15 | 0.266667 | 11 | 2 | 3 |
| single_industry_top1_801130 | top | onset | 沪深300 | 19 | 0.210526 | 15 | 0.266667 | 11 | 2 | 1 |
| single_industry_top1_801130 | top | onset | 中证500 | 19 | 0.263158 | 15 | 0.333333 | 10 | 2 | 1 |
| single_industry_top1_801130 | top | onset | 微盘股 | 18 | 0.222222 | 15 | 0.266667 | 11 | 4 | 0.5 |
| single_industry_top1_801130 | top | onset | 上证指数 | 22 | 0.363636 | 15 | 0.533333 | 7 | 2 | 2 |
| single_industry_top1_801140 | top | capped_confirmation | 全A | 17 | 0.176471 | 7 | 0.428571 | 4 | 0 | 0 |
| single_industry_top1_801140 | top | capped_confirmation | 国证2000 | 19 | 0.210526 | 7 | 0.571429 | 3 | 0 | -6 |
| single_industry_top1_801140 | top | capped_confirmation | 中证1000 | 17 | 0.235294 | 7 | 0.571429 | 3 | 0 | 2 |
| single_industry_top1_801140 | top | capped_confirmation | 沪深300 | 19 | 0.210526 | 7 | 0.571429 | 3 | 0 | -0.5 |
| single_industry_top1_801140 | top | capped_confirmation | 中证500 | 19 | 0.210526 | 7 | 0.571429 | 3 | 0 | -7 |
| single_industry_top1_801140 | top | capped_confirmation | 微盘股 | 18 | 0.166667 | 7 | 0.428571 | 4 | 0 | -4 |
| single_industry_top1_801140 | top | capped_confirmation | 上证指数 | 22 | 0.136364 | 7 | 0.428571 | 4 | 0 | -10 |
| single_industry_top1_801140 | top | onset | 全A | 17 | 0.176471 | 7 | 0.428571 | 4 | 0 | -1 |
| single_industry_top1_801140 | top | onset | 国证2000 | 19 | 0.157895 | 7 | 0.428571 | 4 | 0 | -1 |
| single_industry_top1_801140 | top | onset | 中证1000 | 17 | 0.176471 | 7 | 0.428571 | 4 | 0 | 3 |
| single_industry_top1_801140 | top | onset | 沪深300 | 19 | 0.210526 | 7 | 0.571429 | 3 | 0 | -1.5 |
| single_industry_top1_801140 | top | onset | 中证500 | 19 | 0.157895 | 7 | 0.428571 | 4 | 0 | -1 |
| single_industry_top1_801140 | top | onset | 微盘股 | 18 | 0.166667 | 7 | 0.428571 | 4 | 0 | -5 |
| single_industry_top1_801140 | top | onset | 上证指数 | 22 | 0.136364 | 7 | 0.428571 | 4 | 0 | -1 |
| single_industry_top1_801150 | top | capped_confirmation | 全A | 17 | 0.529412 | 36 | 0.25 | 27 | 1 | 6 |
| single_industry_top1_801150 | top | capped_confirmation | 国证2000 | 19 | 0.421053 | 36 | 0.222222 | 28 | 4 | 0 |
| single_industry_top1_801150 | top | capped_confirmation | 中证1000 | 17 | 0.294118 | 36 | 0.138889 | 31 | 2 | 2 |
| single_industry_top1_801150 | top | capped_confirmation | 沪深300 | 19 | 0.578947 | 36 | 0.305556 | 25 | 2 | 0 |
| single_industry_top1_801150 | top | capped_confirmation | 中证500 | 19 | 0.473684 | 36 | 0.25 | 27 | 4 | 0 |
| single_industry_top1_801150 | top | capped_confirmation | 微盘股 | 18 | 0.388889 | 36 | 0.194444 | 29 | 2 | -7 |
| single_industry_top1_801150 | top | capped_confirmation | 上证指数 | 22 | 0.454545 | 36 | 0.277778 | 26 | 2 | -3.5 |
| single_industry_top1_801150 | top | onset | 全A | 17 | 0.529412 | 36 | 0.25 | 27 | 0 | 5 |
| single_industry_top1_801150 | top | onset | 国证2000 | 19 | 0.421053 | 36 | 0.222222 | 28 | 4 | -1 |
| single_industry_top1_801150 | top | onset | 中证1000 | 17 | 0.294118 | 36 | 0.138889 | 31 | 2 | 1 |
| single_industry_top1_801150 | top | onset | 沪深300 | 19 | 0.578947 | 36 | 0.305556 | 25 | 2 | -1 |
| single_industry_top1_801150 | top | onset | 中证500 | 19 | 0.473684 | 36 | 0.25 | 27 | 4 | -1 |
| single_industry_top1_801150 | top | onset | 微盘股 | 18 | 0.388889 | 36 | 0.194444 | 29 | 3 | -8 |
| single_industry_top1_801150 | top | onset | 上证指数 | 22 | 0.5 | 36 | 0.305556 | 25 | 2 | -1 |
| single_industry_top1_801160 | top | capped_confirmation | 全A | 17 | 0.411765 | 28 | 0.25 | 21 | 3 | 8 |
| single_industry_top1_801160 | top | capped_confirmation | 国证2000 | 19 | 0.368421 | 28 | 0.25 | 21 | 3 | 8 |
| single_industry_top1_801160 | top | capped_confirmation | 中证1000 | 17 | 0.411765 | 28 | 0.25 | 21 | 5 | 11 |
| single_industry_top1_801160 | top | capped_confirmation | 沪深300 | 19 | 0.368421 | 28 | 0.25 | 21 | 1 | -1 |
| single_industry_top1_801160 | top | capped_confirmation | 中证500 | 19 | 0.526316 | 28 | 0.357143 | 18 | 7 | 5 |
| single_industry_top1_801160 | top | capped_confirmation | 微盘股 | 18 | 0.333333 | 28 | 0.214286 | 22 | 1 | 9.5 |
| single_industry_top1_801160 | top | capped_confirmation | 上证指数 | 22 | 0.590909 | 28 | 0.464286 | 15 | 8 | 6 |
| single_industry_top1_801160 | top | onset | 全A | 17 | 0.411765 | 28 | 0.25 | 21 | 4 | 7 |
| single_industry_top1_801160 | top | onset | 国证2000 | 19 | 0.368421 | 28 | 0.25 | 21 | 4 | 7 |
| single_industry_top1_801160 | top | onset | 中证1000 | 17 | 0.411765 | 28 | 0.25 | 21 | 6 | 10 |
| single_industry_top1_801160 | top | onset | 沪深300 | 19 | 0.315789 | 28 | 0.214286 | 22 | 2 | -1.5 |
| single_industry_top1_801160 | top | onset | 中证500 | 19 | 0.526316 | 28 | 0.357143 | 18 | 7 | 4 |
| single_industry_top1_801160 | top | onset | 微盘股 | 18 | 0.333333 | 28 | 0.214286 | 22 | 2 | 8.5 |
| single_industry_top1_801160 | top | onset | 上证指数 | 22 | 0.545455 | 28 | 0.428571 | 16 | 9 | 5 |
| single_industry_top1_801170 | top | capped_confirmation | 全A | 17 | 0.176471 | 8 | 0.375 | 5 | 0 | 7 |
| single_industry_top1_801170 | top | capped_confirmation | 国证2000 | 19 | 0.263158 | 8 | 0.625 | 3 | 1 | 8 |
| single_industry_top1_801170 | top | capped_confirmation | 中证1000 | 17 | 0.235294 | 8 | 0.5 | 4 | 1 | 7.5 |
| single_industry_top1_801170 | top | capped_confirmation | 沪深300 | 19 | 0.0526316 | 8 | 0.125 | 7 | 0 | 9 |
| single_industry_top1_801170 | top | capped_confirmation | 中证500 | 19 | 0.157895 | 8 | 0.375 | 5 | 0 | -6 |
| single_industry_top1_801170 | top | capped_confirmation | 微盘股 | 18 | 0.222222 | 8 | 0.5 | 4 | 0 | 4 |
| single_industry_top1_801170 | top | capped_confirmation | 上证指数 | 22 | 0.0454545 | 8 | 0.125 | 7 | 0 | 9 |
| single_industry_top1_801170 | top | onset | 全A | 17 | 0.176471 | 8 | 0.375 | 5 | 0 | 6 |
| single_industry_top1_801170 | top | onset | 国证2000 | 19 | 0.263158 | 8 | 0.625 | 3 | 1 | 8 |
| single_industry_top1_801170 | top | onset | 中证1000 | 17 | 0.235294 | 8 | 0.5 | 4 | 1 | 7.5 |
| single_industry_top1_801170 | top | onset | 沪深300 | 19 | 0.0526316 | 8 | 0.125 | 7 | 0 | 8 |
| single_industry_top1_801170 | top | onset | 中证500 | 19 | 0.157895 | 8 | 0.375 | 5 | 0 | -7 |
| single_industry_top1_801170 | top | onset | 微盘股 | 18 | 0.222222 | 8 | 0.5 | 4 | 0 | 3 |
| single_industry_top1_801170 | top | onset | 上证指数 | 22 | 0.0454545 | 8 | 0.125 | 7 | 0 | 8 |
| single_industry_top1_801180 | top | capped_confirmation | 全A | 17 | 0.411765 | 24 | 0.291667 | 17 | 3 | -1 |
| single_industry_top1_801180 | top | capped_confirmation | 国证2000 | 19 | 0.473684 | 24 | 0.375 | 15 | 3 | -1 |
| single_industry_top1_801180 | top | capped_confirmation | 中证1000 | 17 | 0.529412 | 24 | 0.375 | 15 | 2 | -1 |
| single_industry_top1_801180 | top | capped_confirmation | 沪深300 | 19 | 0.368421 | 24 | 0.291667 | 17 | 3 | -1 |
| single_industry_top1_801180 | top | capped_confirmation | 中证500 | 19 | 0.421053 | 24 | 0.333333 | 16 | 3 | 1.5 |
| single_industry_top1_801180 | top | capped_confirmation | 微盘股 | 18 | 0.333333 | 24 | 0.25 | 18 | 3 | -5.5 |
| single_industry_top1_801180 | top | capped_confirmation | 上证指数 | 22 | 0.363636 | 24 | 0.333333 | 16 | 3 | -1 |
| single_industry_top1_801180 | top | onset | 全A | 17 | 0.411765 | 24 | 0.291667 | 17 | 3 | -1 |
| single_industry_top1_801180 | top | onset | 国证2000 | 19 | 0.473684 | 24 | 0.375 | 15 | 3 | -2 |
| single_industry_top1_801180 | top | onset | 中证1000 | 17 | 0.529412 | 24 | 0.375 | 15 | 2 | -2 |
| single_industry_top1_801180 | top | onset | 沪深300 | 19 | 0.368421 | 24 | 0.291667 | 17 | 3 | -1 |
| single_industry_top1_801180 | top | onset | 中证500 | 19 | 0.421053 | 24 | 0.333333 | 16 | 3 | 0.5 |
| single_industry_top1_801180 | top | onset | 微盘股 | 18 | 0.333333 | 24 | 0.25 | 18 | 3 | -6.5 |
| single_industry_top1_801180 | top | onset | 上证指数 | 22 | 0.363636 | 24 | 0.333333 | 16 | 3 | -2 |
| single_industry_top1_801200 | top | capped_confirmation | 全A | 17 | 0.117647 | 12 | 0.166667 | 10 | 2 | 4.5 |
| single_industry_top1_801200 | top | capped_confirmation | 国证2000 | 19 | 0.105263 | 12 | 0.166667 | 10 | 3 | -6.5 |
| single_industry_top1_801200 | top | capped_confirmation | 中证1000 | 17 | 0.176471 | 12 | 0.25 | 9 | 2 | 0 |
| single_industry_top1_801200 | top | capped_confirmation | 沪深300 | 19 | 0.157895 | 12 | 0.25 | 9 | 2 | -11 |
| single_industry_top1_801200 | top | capped_confirmation | 中证500 | 19 | 0.157895 | 12 | 0.25 | 9 | 2 | 0 |
| single_industry_top1_801200 | top | capped_confirmation | 微盘股 | 18 | 0.111111 | 12 | 0.166667 | 10 | 3 | -6.5 |
| single_industry_top1_801200 | top | capped_confirmation | 上证指数 | 22 | 0.181818 | 12 | 0.333333 | 8 | 2 | -5.5 |
| single_industry_top1_801200 | top | onset | 全A | 17 | 0.117647 | 12 | 0.166667 | 10 | 2 | 3.5 |
| single_industry_top1_801200 | top | onset | 国证2000 | 19 | 0.105263 | 12 | 0.166667 | 10 | 3 | -7.5 |
| single_industry_top1_801200 | top | onset | 中证1000 | 17 | 0.176471 | 12 | 0.25 | 9 | 2 | -1 |
| single_industry_top1_801200 | top | onset | 沪深300 | 19 | 0.157895 | 12 | 0.25 | 9 | 2 | -1 |
| single_industry_top1_801200 | top | onset | 中证500 | 19 | 0.157895 | 12 | 0.25 | 9 | 2 | -1 |
| single_industry_top1_801200 | top | onset | 微盘股 | 18 | 0.111111 | 12 | 0.166667 | 10 | 3 | -7.5 |
| single_industry_top1_801200 | top | onset | 上证指数 | 22 | 0.181818 | 12 | 0.333333 | 8 | 2 | 4.5 |
| single_industry_top1_801210 | top | capped_confirmation | 全A | 17 | 0.411765 | 43 | 0.162791 | 36 | 7 | -2 |
| single_industry_top1_801210 | top | capped_confirmation | 国证2000 | 19 | 0.631579 | 43 | 0.27907 | 31 | 12 | 1 |
| single_industry_top1_801210 | top | capped_confirmation | 中证1000 | 17 | 0.647059 | 43 | 0.255814 | 32 | 9 | -1 |
| single_industry_top1_801210 | top | capped_confirmation | 沪深300 | 19 | 0.421053 | 43 | 0.186047 | 35 | 5 | -2 |
| single_industry_top1_801210 | top | capped_confirmation | 中证500 | 19 | 0.526316 | 43 | 0.232558 | 33 | 10 | -3 |
| single_industry_top1_801210 | top | capped_confirmation | 微盘股 | 18 | 0.5 | 43 | 0.209302 | 34 | 5 | -1 |
| single_industry_top1_801210 | top | capped_confirmation | 上证指数 | 22 | 0.454545 | 43 | 0.232558 | 33 | 9 | -2 |
| single_industry_top1_801210 | top | onset | 全A | 17 | 0.411765 | 43 | 0.162791 | 36 | 7 | -3 |
| single_industry_top1_801210 | top | onset | 国证2000 | 19 | 0.578947 | 43 | 0.255814 | 32 | 12 | 0 |
| single_industry_top1_801210 | top | onset | 中证1000 | 17 | 0.647059 | 43 | 0.255814 | 32 | 9 | -2 |
| single_industry_top1_801210 | top | onset | 沪深300 | 19 | 0.368421 | 43 | 0.162791 | 36 | 6 | -3 |
| single_industry_top1_801210 | top | onset | 中证500 | 19 | 0.526316 | 43 | 0.232558 | 33 | 9 | -4 |
| single_industry_top1_801210 | top | onset | 微盘股 | 18 | 0.388889 | 43 | 0.162791 | 36 | 5 | -2 |
| single_industry_top1_801210 | top | onset | 上证指数 | 22 | 0.409091 | 43 | 0.209302 | 34 | 9 | -3 |
| single_industry_top1_801230 | top | capped_confirmation | 全A | 17 | 0.470588 | 30 | 0.266667 | 22 | 7 | -4.5 |
| single_industry_top1_801230 | top | capped_confirmation | 国证2000 | 19 | 0.526316 | 30 | 0.333333 | 20 | 6 | -4.5 |
| single_industry_top1_801230 | top | capped_confirmation | 中证1000 | 17 | 0.411765 | 30 | 0.233333 | 23 | 8 | -4 |
| single_industry_top1_801230 | top | capped_confirmation | 沪深300 | 19 | 0.368421 | 30 | 0.233333 | 23 | 5 | 0 |
| single_industry_top1_801230 | top | capped_confirmation | 中证500 | 19 | 0.368421 | 30 | 0.233333 | 23 | 6 | -4 |
| single_industry_top1_801230 | top | capped_confirmation | 微盘股 | 18 | 0.5 | 30 | 0.3 | 21 | 2 | -6 |
| single_industry_top1_801230 | top | capped_confirmation | 上证指数 | 22 | 0.454545 | 30 | 0.333333 | 20 | 5 | -4 |
| single_industry_top1_801230 | top | onset | 全A | 17 | 0.470588 | 30 | 0.266667 | 22 | 7 | -5.5 |
| single_industry_top1_801230 | top | onset | 国证2000 | 19 | 0.526316 | 30 | 0.333333 | 20 | 7 | -5.5 |
| single_industry_top1_801230 | top | onset | 中证1000 | 17 | 0.411765 | 30 | 0.233333 | 23 | 8 | -5 |
| single_industry_top1_801230 | top | onset | 沪深300 | 19 | 0.368421 | 30 | 0.233333 | 23 | 5 | -1 |
| single_industry_top1_801230 | top | onset | 中证500 | 19 | 0.368421 | 30 | 0.233333 | 23 | 6 | -5 |
| single_industry_top1_801230 | top | onset | 微盘股 | 18 | 0.5 | 30 | 0.3 | 21 | 3 | -7 |
| single_industry_top1_801230 | top | onset | 上证指数 | 22 | 0.454545 | 30 | 0.333333 | 20 | 5 | -5 |
| single_industry_top1_801710 | top | capped_confirmation | 全A | 17 | 0.235294 | 11 | 0.363636 | 7 | 4 | -2.5 |
| single_industry_top1_801710 | top | capped_confirmation | 国证2000 | 19 | 0.105263 | 11 | 0.181818 | 9 | 4 | -2.5 |
| single_industry_top1_801710 | top | capped_confirmation | 中证1000 | 17 | 0.235294 | 11 | 0.363636 | 7 | 4 | -1.5 |
| single_industry_top1_801710 | top | capped_confirmation | 沪深300 | 19 | 0.210526 | 11 | 0.363636 | 7 | 4 | -0.5 |
| single_industry_top1_801710 | top | capped_confirmation | 中证500 | 19 | 0.210526 | 11 | 0.363636 | 7 | 4 | -1.5 |
| single_industry_top1_801710 | top | capped_confirmation | 微盘股 | 18 | 0.111111 | 11 | 0.181818 | 9 | 4 | 6.5 |
| single_industry_top1_801710 | top | capped_confirmation | 上证指数 | 22 | 0.227273 | 11 | 0.454545 | 6 | 4 | -1 |
| single_industry_top1_801710 | top | onset | 全A | 17 | 0.235294 | 11 | 0.363636 | 7 | 4 | -3.5 |
| single_industry_top1_801710 | top | onset | 国证2000 | 19 | 0.105263 | 11 | 0.181818 | 9 | 4 | -3.5 |
| single_industry_top1_801710 | top | onset | 中证1000 | 17 | 0.235294 | 11 | 0.363636 | 7 | 4 | -2.5 |
| single_industry_top1_801710 | top | onset | 沪深300 | 19 | 0.210526 | 11 | 0.363636 | 7 | 4 | -1.5 |
| single_industry_top1_801710 | top | onset | 中证500 | 19 | 0.210526 | 11 | 0.363636 | 7 | 4 | -2.5 |
| single_industry_top1_801710 | top | onset | 微盘股 | 18 | 0.111111 | 11 | 0.181818 | 9 | 4 | 5.5 |
| single_industry_top1_801710 | top | onset | 上证指数 | 22 | 0.227273 | 11 | 0.454545 | 6 | 4 | -2 |
| single_industry_top1_801720 | top | capped_confirmation | 全A | 17 | 0.294118 | 13 | 0.384615 | 8 | 1 | 7 |
| single_industry_top1_801720 | top | capped_confirmation | 国证2000 | 19 | 0.263158 | 13 | 0.384615 | 8 | 1 | 3 |
| single_industry_top1_801720 | top | capped_confirmation | 中证1000 | 17 | 0.294118 | 13 | 0.384615 | 8 | 0 | 3 |
| single_industry_top1_801720 | top | capped_confirmation | 沪深300 | 19 | 0.157895 | 13 | 0.230769 | 10 | 1 | 0 |
| single_industry_top1_801720 | top | capped_confirmation | 中证500 | 19 | 0.315789 | 13 | 0.461538 | 7 | 1 | -6 |
| single_industry_top1_801720 | top | capped_confirmation | 微盘股 | 18 | 0.111111 | 13 | 0.153846 | 11 | 1 | 3 |
| single_industry_top1_801720 | top | capped_confirmation | 上证指数 | 22 | 0.181818 | 13 | 0.307692 | 9 | 1 | 3.5 |
| single_industry_top1_801720 | top | onset | 全A | 17 | 0.235294 | 13 | 0.307692 | 9 | 3 | 10.5 |
| single_industry_top1_801720 | top | onset | 国证2000 | 19 | 0.263158 | 13 | 0.384615 | 8 | 1 | 2 |
| single_industry_top1_801720 | top | onset | 中证1000 | 17 | 0.235294 | 13 | 0.307692 | 9 | 1 | 8.5 |
| single_industry_top1_801720 | top | onset | 沪深300 | 19 | 0.157895 | 13 | 0.230769 | 10 | 1 | -1 |
| single_industry_top1_801720 | top | onset | 中证500 | 19 | 0.210526 | 13 | 0.307692 | 9 | 2 | 1.5 |
| single_industry_top1_801720 | top | onset | 微盘股 | 18 | 0.111111 | 13 | 0.153846 | 11 | 1 | 2 |
| single_industry_top1_801720 | top | onset | 上证指数 | 22 | 0.227273 | 13 | 0.384615 | 8 | 2 | 6 |
| single_industry_top1_801730 | top | capped_confirmation | 全A | 17 | 0.235294 | 24 | 0.166667 | 20 | 6 | 5.5 |
| single_industry_top1_801730 | top | capped_confirmation | 国证2000 | 19 | 0.263158 | 24 | 0.208333 | 19 | 4 | 7 |
| single_industry_top1_801730 | top | capped_confirmation | 中证1000 | 17 | 0.294118 | 24 | 0.208333 | 19 | 4 | 0 |
| single_industry_top1_801730 | top | capped_confirmation | 沪深300 | 19 | 0.210526 | 24 | 0.166667 | 20 | 6 | 1.5 |
| single_industry_top1_801730 | top | capped_confirmation | 中证500 | 19 | 0.263158 | 24 | 0.208333 | 19 | 6 | 3 |
| single_industry_top1_801730 | top | capped_confirmation | 微盘股 | 18 | 0.277778 | 24 | 0.208333 | 19 | 4 | 3 |
| single_industry_top1_801730 | top | capped_confirmation | 上证指数 | 22 | 0.227273 | 24 | 0.208333 | 19 | 6 | 3 |
| single_industry_top1_801730 | top | onset | 全A | 17 | 0.235294 | 24 | 0.166667 | 20 | 6 | 4.5 |
| single_industry_top1_801730 | top | onset | 国证2000 | 19 | 0.263158 | 24 | 0.208333 | 19 | 4 | 6 |
| single_industry_top1_801730 | top | onset | 中证1000 | 17 | 0.294118 | 24 | 0.208333 | 19 | 4 | -1 |
| single_industry_top1_801730 | top | onset | 沪深300 | 19 | 0.210526 | 24 | 0.166667 | 20 | 6 | 0.5 |
| single_industry_top1_801730 | top | onset | 中证500 | 19 | 0.263158 | 24 | 0.208333 | 19 | 6 | 2 |
| single_industry_top1_801730 | top | onset | 微盘股 | 18 | 0.277778 | 24 | 0.208333 | 19 | 4 | 2 |
| single_industry_top1_801730 | top | onset | 上证指数 | 22 | 0.227273 | 24 | 0.208333 | 19 | 6 | 2 |
| single_industry_top1_801740 | top | capped_confirmation | 全A | 17 | 0.764706 | 83 | 0.156627 | 70 | 7 | -5 |
| single_industry_top1_801740 | top | capped_confirmation | 国证2000 | 19 | 0.736842 | 83 | 0.168675 | 69 | 11 | -2 |
| single_industry_top1_801740 | top | capped_confirmation | 中证1000 | 17 | 0.882353 | 83 | 0.180723 | 68 | 14 | -2 |
| single_industry_top1_801740 | top | capped_confirmation | 沪深300 | 19 | 0.684211 | 83 | 0.156627 | 70 | 9 | -5 |
| single_industry_top1_801740 | top | capped_confirmation | 中证500 | 19 | 0.631579 | 83 | 0.144578 | 71 | 11 | -3 |
| single_industry_top1_801740 | top | capped_confirmation | 微盘股 | 18 | 0.777778 | 83 | 0.168675 | 69 | 10 | -5 |
| single_industry_top1_801740 | top | capped_confirmation | 上证指数 | 22 | 0.590909 | 83 | 0.156627 | 70 | 8 | -3 |
| single_industry_top1_801740 | top | onset | 全A | 17 | 0.705882 | 83 | 0.144578 | 71 | 7 | -5 |
| single_industry_top1_801740 | top | onset | 国证2000 | 19 | 0.736842 | 83 | 0.168675 | 69 | 11 | -5 |
| single_industry_top1_801740 | top | onset | 中证1000 | 17 | 0.882353 | 83 | 0.180723 | 68 | 14 | -3 |
| single_industry_top1_801740 | top | onset | 沪深300 | 19 | 0.631579 | 83 | 0.144578 | 71 | 9 | -5 |
| single_industry_top1_801740 | top | onset | 中证500 | 19 | 0.631579 | 83 | 0.144578 | 71 | 11 | -1 |
| single_industry_top1_801740 | top | onset | 微盘股 | 18 | 0.722222 | 83 | 0.156627 | 70 | 10 | -4 |
| single_industry_top1_801740 | top | onset | 上证指数 | 22 | 0.590909 | 83 | 0.156627 | 70 | 9 | -3 |
| single_industry_top1_801750 | top | capped_confirmation | 全A | 17 | 0.411765 | 43 | 0.162791 | 36 | 9 | -2 |
| single_industry_top1_801750 | top | capped_confirmation | 国证2000 | 19 | 0.631579 | 43 | 0.27907 | 31 | 9 | -5.5 |
| single_industry_top1_801750 | top | capped_confirmation | 中证1000 | 17 | 0.647059 | 43 | 0.255814 | 32 | 10 | -7 |
| single_industry_top1_801750 | top | capped_confirmation | 沪深300 | 19 | 0.473684 | 43 | 0.209302 | 34 | 7 | 0 |
| single_industry_top1_801750 | top | capped_confirmation | 中证500 | 19 | 0.473684 | 43 | 0.209302 | 34 | 12 | 0 |
| single_industry_top1_801750 | top | capped_confirmation | 微盘股 | 18 | 0.444444 | 43 | 0.186047 | 35 | 6 | -5 |
| single_industry_top1_801750 | top | capped_confirmation | 上证指数 | 22 | 0.454545 | 43 | 0.232558 | 33 | 8 | -1.5 |
| single_industry_top1_801750 | top | onset | 全A | 17 | 0.411765 | 43 | 0.162791 | 36 | 10 | -3 |
| single_industry_top1_801750 | top | onset | 国证2000 | 19 | 0.631579 | 43 | 0.27907 | 31 | 10 | -6.5 |
| single_industry_top1_801750 | top | onset | 中证1000 | 17 | 0.647059 | 43 | 0.255814 | 32 | 10 | -8 |
| single_industry_top1_801750 | top | onset | 沪深300 | 19 | 0.473684 | 43 | 0.209302 | 34 | 7 | 1 |
| single_industry_top1_801750 | top | onset | 中证500 | 19 | 0.473684 | 43 | 0.209302 | 34 | 11 | -1 |
| single_industry_top1_801750 | top | onset | 微盘股 | 18 | 0.444444 | 43 | 0.186047 | 35 | 6 | -5.5 |
| single_industry_top1_801750 | top | onset | 上证指数 | 22 | 0.454545 | 43 | 0.232558 | 33 | 8 | -0.5 |
| single_industry_top1_801760 | top | capped_confirmation | 全A | 17 | 0.352941 | 35 | 0.171429 | 29 | 6 | 0.5 |
| single_industry_top1_801760 | top | capped_confirmation | 国证2000 | 19 | 0.421053 | 35 | 0.228571 | 27 | 13 | -0.5 |
| single_industry_top1_801760 | top | capped_confirmation | 中证1000 | 17 | 0.470588 | 35 | 0.228571 | 27 | 4 | 1 |
| single_industry_top1_801760 | top | capped_confirmation | 沪深300 | 19 | 0.368421 | 35 | 0.2 | 28 | 6 | 1 |
| single_industry_top1_801760 | top | capped_confirmation | 中证500 | 19 | 0.421053 | 35 | 0.228571 | 27 | 3 | 1.5 |
| single_industry_top1_801760 | top | capped_confirmation | 微盘股 | 18 | 0.277778 | 35 | 0.142857 | 30 | 6 | 0 |
| single_industry_top1_801760 | top | capped_confirmation | 上证指数 | 22 | 0.363636 | 35 | 0.228571 | 27 | 3 | 0.5 |
| single_industry_top1_801760 | top | onset | 全A | 17 | 0.352941 | 35 | 0.171429 | 29 | 7 | -0.5 |
| single_industry_top1_801760 | top | onset | 国证2000 | 19 | 0.421053 | 35 | 0.228571 | 27 | 13 | 0 |
| single_industry_top1_801760 | top | onset | 中证1000 | 17 | 0.470588 | 35 | 0.228571 | 27 | 5 | 0 |
| single_industry_top1_801760 | top | onset | 沪深300 | 19 | 0.368421 | 35 | 0.2 | 28 | 6 | 0 |
| single_industry_top1_801760 | top | onset | 中证500 | 19 | 0.421053 | 35 | 0.228571 | 27 | 4 | 0.5 |
| single_industry_top1_801760 | top | onset | 微盘股 | 18 | 0.277778 | 35 | 0.142857 | 30 | 5 | -1 |
| single_industry_top1_801760 | top | onset | 上证指数 | 22 | 0.363636 | 35 | 0.228571 | 27 | 3 | -0.5 |
| single_industry_top1_801770 | top | capped_confirmation | 全A | 17 | 0.705882 | 49 | 0.244898 | 37 | 12 | -5.5 |
| single_industry_top1_801770 | top | capped_confirmation | 国证2000 | 19 | 0.578947 | 49 | 0.22449 | 38 | 12 | -5 |
| single_industry_top1_801770 | top | capped_confirmation | 中证1000 | 17 | 0.764706 | 49 | 0.265306 | 36 | 13 | -1 |
| single_industry_top1_801770 | top | capped_confirmation | 沪深300 | 19 | 0.578947 | 49 | 0.22449 | 38 | 10 | -1 |
| single_industry_top1_801770 | top | capped_confirmation | 中证500 | 19 | 0.631579 | 49 | 0.244898 | 37 | 8 | 1 |
| single_industry_top1_801770 | top | capped_confirmation | 微盘股 | 18 | 0.666667 | 49 | 0.244898 | 37 | 8 | -6 |
| single_industry_top1_801770 | top | capped_confirmation | 上证指数 | 22 | 0.681818 | 49 | 0.306122 | 34 | 10 | -1 |
| single_industry_top1_801770 | top | onset | 全A | 17 | 0.705882 | 49 | 0.244898 | 37 | 12 | -6.5 |
| single_industry_top1_801770 | top | onset | 国证2000 | 19 | 0.578947 | 49 | 0.22449 | 38 | 12 | -6 |
| single_industry_top1_801770 | top | onset | 中证1000 | 17 | 0.764706 | 49 | 0.265306 | 36 | 12 | -2 |
| single_industry_top1_801770 | top | onset | 沪深300 | 19 | 0.578947 | 49 | 0.22449 | 38 | 10 | -2 |
| single_industry_top1_801770 | top | onset | 中证500 | 19 | 0.631579 | 49 | 0.244898 | 37 | 8 | 0 |
| single_industry_top1_801770 | top | onset | 微盘股 | 18 | 0.611111 | 49 | 0.22449 | 38 | 10 | -8 |
| single_industry_top1_801770 | top | onset | 上证指数 | 22 | 0.681818 | 49 | 0.306122 | 34 | 10 | -2 |
| single_industry_top1_801780 | top | capped_confirmation | 全A | 17 | 0.882353 | 123 | 0.121951 | 108 | 31 | 0 |
| single_industry_top1_801780 | top | capped_confirmation | 国证2000 | 19 | 0.842105 | 123 | 0.130081 | 107 | 29 | 1.5 |
| single_industry_top1_801780 | top | capped_confirmation | 中证1000 | 17 | 0.882353 | 123 | 0.121951 | 108 | 20 | 5 |
| single_industry_top1_801780 | top | capped_confirmation | 沪深300 | 19 | 0.842105 | 123 | 0.130081 | 107 | 23 | -1 |
| single_industry_top1_801780 | top | capped_confirmation | 中证500 | 19 | 0.894737 | 123 | 0.138211 | 106 | 20 | -3 |
| single_industry_top1_801780 | top | capped_confirmation | 微盘股 | 18 | 1 | 123 | 0.146341 | 105 | 22 | -0.5 |
| single_industry_top1_801780 | top | capped_confirmation | 上证指数 | 22 | 0.863636 | 123 | 0.154472 | 104 | 31 | -3 |
| single_industry_top1_801780 | top | onset | 全A | 17 | 0.882353 | 123 | 0.121951 | 108 | 29 | -1 |
| single_industry_top1_801780 | top | onset | 国证2000 | 19 | 0.842105 | 123 | 0.130081 | 107 | 29 | 1.5 |
| single_industry_top1_801780 | top | onset | 中证1000 | 17 | 0.882353 | 123 | 0.121951 | 108 | 20 | 4 |
| single_industry_top1_801780 | top | onset | 沪深300 | 19 | 0.842105 | 123 | 0.130081 | 107 | 23 | -1.5 |
| single_industry_top1_801780 | top | onset | 中证500 | 19 | 0.894737 | 123 | 0.138211 | 106 | 23 | -1 |
| single_industry_top1_801780 | top | onset | 微盘股 | 18 | 1 | 123 | 0.146341 | 105 | 25 | -0.5 |
| single_industry_top1_801780 | top | onset | 上证指数 | 22 | 0.863636 | 123 | 0.154472 | 104 | 30 | -3 |
| single_industry_top1_801790 | top | capped_confirmation | 全A | 17 | 0.588235 | 64 | 0.15625 | 54 | 7 | -4 |
| single_industry_top1_801790 | top | capped_confirmation | 国证2000 | 19 | 0.473684 | 64 | 0.140625 | 55 | 6 | -9 |
| single_industry_top1_801790 | top | capped_confirmation | 中证1000 | 17 | 0.470588 | 64 | 0.125 | 56 | 3 | -6.5 |
| single_industry_top1_801790 | top | capped_confirmation | 沪深300 | 19 | 0.684211 | 64 | 0.203125 | 51 | 12 | -2 |
| single_industry_top1_801790 | top | capped_confirmation | 中证500 | 19 | 0.631579 | 64 | 0.1875 | 52 | 9 | -4 |
| single_industry_top1_801790 | top | capped_confirmation | 微盘股 | 18 | 0.333333 | 64 | 0.09375 | 58 | 5 | -1 |
| single_industry_top1_801790 | top | capped_confirmation | 上证指数 | 22 | 0.545455 | 64 | 0.1875 | 52 | 10 | -5 |
| single_industry_top1_801790 | top | onset | 全A | 17 | 0.588235 | 64 | 0.15625 | 54 | 6 | -5 |
| single_industry_top1_801790 | top | onset | 国证2000 | 19 | 0.473684 | 64 | 0.140625 | 55 | 6 | -10 |
| single_industry_top1_801790 | top | onset | 中证1000 | 17 | 0.470588 | 64 | 0.125 | 56 | 2 | -7.5 |
| single_industry_top1_801790 | top | onset | 沪深300 | 19 | 0.684211 | 64 | 0.203125 | 51 | 11 | -3 |
| single_industry_top1_801790 | top | onset | 中证500 | 19 | 0.631579 | 64 | 0.1875 | 52 | 9 | -5 |
| single_industry_top1_801790 | top | onset | 微盘股 | 18 | 0.333333 | 64 | 0.09375 | 58 | 5 | -2 |
| single_industry_top1_801790 | top | onset | 上证指数 | 22 | 0.545455 | 64 | 0.1875 | 52 | 9 | -6 |
| single_industry_top1_801880 | top | capped_confirmation | 全A | 17 | 0.117647 | 17 | 0.117647 | 15 | 0 | -3.5 |
| single_industry_top1_801880 | top | capped_confirmation | 国证2000 | 19 | 0.157895 | 17 | 0.176471 | 14 | 0 | -12 |
| single_industry_top1_801880 | top | capped_confirmation | 中证1000 | 17 | 0.235294 | 17 | 0.235294 | 13 | 0 | -13.5 |
| single_industry_top1_801880 | top | capped_confirmation | 沪深300 | 19 | 0.105263 | 17 | 0.117647 | 15 | 0 | -3.5 |
| single_industry_top1_801880 | top | capped_confirmation | 中证500 | 19 | 0.157895 | 17 | 0.176471 | 14 | 1 | 0 |
| single_industry_top1_801880 | top | capped_confirmation | 微盘股 | 18 | 0.111111 | 17 | 0.117647 | 15 | 0 | -6 |
| single_industry_top1_801880 | top | capped_confirmation | 上证指数 | 22 | 0.136364 | 17 | 0.176471 | 14 | 1 | 0 |
| single_industry_top1_801880 | top | onset | 全A | 17 | 0.117647 | 17 | 0.117647 | 15 | 1 | -4.5 |
| single_industry_top1_801880 | top | onset | 国证2000 | 19 | 0.157895 | 17 | 0.176471 | 14 | 0 | -13 |
| single_industry_top1_801880 | top | onset | 中证1000 | 17 | 0.235294 | 17 | 0.235294 | 13 | 0 | -14.5 |
| single_industry_top1_801880 | top | onset | 沪深300 | 19 | 0.105263 | 17 | 0.117647 | 15 | 1 | -4.5 |
| single_industry_top1_801880 | top | onset | 中证500 | 19 | 0.157895 | 17 | 0.176471 | 14 | 2 | -1 |
| single_industry_top1_801880 | top | onset | 微盘股 | 18 | 0.111111 | 17 | 0.117647 | 15 | 0 | -7 |
| single_industry_top1_801880 | top | onset | 上证指数 | 22 | 0.136364 | 17 | 0.176471 | 14 | 2 | -1 |
| single_industry_top1_801890 | top | capped_confirmation | 全A | 17 | 0.117647 | 8 | 0.25 | 6 | 0 | 8.5 |
| single_industry_top1_801890 | top | capped_confirmation | 国证2000 | 19 | 0.157895 | 8 | 0.375 | 5 | 0 | 0 |
| single_industry_top1_801890 | top | capped_confirmation | 中证1000 | 17 | 0.117647 | 8 | 0.25 | 6 | 0 | -8 |
| single_industry_top1_801890 | top | capped_confirmation | 沪深300 | 19 | 0.105263 | 8 | 0.25 | 6 | 0 | 8.5 |
| single_industry_top1_801890 | top | capped_confirmation | 中证500 | 19 | 0.105263 | 8 | 0.25 | 6 | 0 | 8.5 |
| single_industry_top1_801890 | top | capped_confirmation | 微盘股 | 18 | 0.166667 | 8 | 0.375 | 5 | 0 | -2 |
| single_industry_top1_801890 | top | capped_confirmation | 上证指数 | 22 | 0.136364 | 8 | 0.375 | 5 | 0 | 12 |
| single_industry_top1_801890 | top | onset | 全A | 17 | 0.117647 | 8 | 0.25 | 6 | 0 | 7.5 |
| single_industry_top1_801890 | top | onset | 国证2000 | 19 | 0.157895 | 8 | 0.375 | 5 | 0 | -1 |
| single_industry_top1_801890 | top | onset | 中证1000 | 17 | 0.117647 | 8 | 0.25 | 6 | 0 | -9 |
| single_industry_top1_801890 | top | onset | 沪深300 | 19 | 0.105263 | 8 | 0.25 | 6 | 0 | 7.5 |
| single_industry_top1_801890 | top | onset | 中证500 | 19 | 0.105263 | 8 | 0.25 | 6 | 0 | 7.5 |
| single_industry_top1_801890 | top | onset | 微盘股 | 18 | 0.166667 | 8 | 0.375 | 5 | 0 | -3 |
| single_industry_top1_801890 | top | onset | 上证指数 | 22 | 0.136364 | 8 | 0.375 | 5 | 0 | 11 |
| single_industry_top1_801950 | top | capped_confirmation | 全A | 9 | 0.888889 | 51 | 0.156863 | 43 | 14 | -2 |
| single_industry_top1_801950 | top | capped_confirmation | 国证2000 | 11 | 0.909091 | 51 | 0.196078 | 41 | 18 | -2 |
| single_industry_top1_801950 | top | capped_confirmation | 中证1000 | 8 | 1 | 51 | 0.156863 | 43 | 13 | -2.5 |
| single_industry_top1_801950 | top | capped_confirmation | 沪深300 | 9 | 0.888889 | 51 | 0.156863 | 43 | 8 | -1.5 |
| single_industry_top1_801950 | top | capped_confirmation | 中证500 | 9 | 0.777778 | 51 | 0.137255 | 44 | 10 | -4 |
| single_industry_top1_801950 | top | capped_confirmation | 微盘股 | 12 | 0.833333 | 51 | 0.196078 | 41 | 16 | -1.5 |
| single_industry_top1_801950 | top | capped_confirmation | 上证指数 | 10 | 0.7 | 51 | 0.137255 | 44 | 11 | -4 |
| single_industry_top1_801950 | top | onset | 全A | 9 | 0.888889 | 51 | 0.156863 | 43 | 14 | -0.5 |
| single_industry_top1_801950 | top | onset | 国证2000 | 11 | 0.909091 | 51 | 0.196078 | 41 | 18 | 2.5 |
| single_industry_top1_801950 | top | onset | 中证1000 | 8 | 1 | 51 | 0.156863 | 43 | 13 | -1 |
| single_industry_top1_801950 | top | onset | 沪深300 | 9 | 0.888889 | 51 | 0.156863 | 43 | 8 | -1.5 |
| single_industry_top1_801950 | top | onset | 中证500 | 9 | 0.666667 | 51 | 0.117647 | 45 | 11 | -4.5 |
| single_industry_top1_801950 | top | onset | 微盘股 | 12 | 0.833333 | 51 | 0.196078 | 41 | 16 | -2.5 |
| single_industry_top1_801950 | top | onset | 上证指数 | 10 | 0.7 | 51 | 0.137255 | 44 | 11 | -5 |
| single_industry_top1_801960 | top | capped_confirmation | 全A | 9 | 0.333333 | 13 | 0.230769 | 10 | 3 | -8 |
| single_industry_top1_801960 | top | capped_confirmation | 国证2000 | 11 | 0.272727 | 13 | 0.230769 | 10 | 3 | -4 |
| single_industry_top1_801960 | top | capped_confirmation | 中证1000 | 8 | 0.375 | 13 | 0.230769 | 10 | 3 | -4 |
| single_industry_top1_801960 | top | capped_confirmation | 沪深300 | 9 | 0.444444 | 13 | 0.307692 | 9 | 2 | -9.5 |
| single_industry_top1_801960 | top | capped_confirmation | 中证500 | 9 | 0.333333 | 13 | 0.230769 | 10 | 3 | -9 |
| single_industry_top1_801960 | top | capped_confirmation | 微盘股 | 12 | 0.25 | 13 | 0.230769 | 10 | 3 | -4 |
| single_industry_top1_801960 | top | capped_confirmation | 上证指数 | 10 | 0.4 | 13 | 0.307692 | 9 | 1 | -14.5 |
| single_industry_top1_801960 | top | onset | 全A | 9 | 0.333333 | 13 | 0.230769 | 10 | 3 | -20 |
| single_industry_top1_801960 | top | onset | 国证2000 | 11 | 0.272727 | 13 | 0.230769 | 10 | 3 | -5 |
| single_industry_top1_801960 | top | onset | 中证1000 | 8 | 0.375 | 13 | 0.230769 | 10 | 3 | -5 |
| single_industry_top1_801960 | top | onset | 沪深300 | 9 | 0.333333 | 13 | 0.230769 | 10 | 3 | -1 |
| single_industry_top1_801960 | top | onset | 中证500 | 9 | 0.333333 | 13 | 0.230769 | 10 | 2 | -10 |
| single_industry_top1_801960 | top | onset | 微盘股 | 12 | 0.25 | 13 | 0.230769 | 10 | 3 | -5 |
| single_industry_top1_801960 | top | onset | 上证指数 | 10 | 0.3 | 13 | 0.230769 | 10 | 2 | -11 |
| single_industry_top1_801970 | top | capped_confirmation | 全A | 9 | 0.444444 | 14 | 0.285714 | 10 | 3 | -1 |
| single_industry_top1_801970 | top | capped_confirmation | 国证2000 | 11 | 0.363636 | 14 | 0.285714 | 10 | 3 | -1 |
| single_industry_top1_801970 | top | capped_confirmation | 中证1000 | 8 | 0.625 | 14 | 0.357143 | 9 | 2 | -1 |
| single_industry_top1_801970 | top | capped_confirmation | 沪深300 | 9 | 0.444444 | 14 | 0.285714 | 10 | 3 | 4.5 |
| single_industry_top1_801970 | top | capped_confirmation | 中证500 | 9 | 0.333333 | 14 | 0.214286 | 11 | 3 | -1 |
| single_industry_top1_801970 | top | capped_confirmation | 微盘股 | 12 | 0.416667 | 14 | 0.357143 | 9 | 3 | -1 |
| single_industry_top1_801970 | top | capped_confirmation | 上证指数 | 10 | 0.3 | 14 | 0.214286 | 11 | 3 | -1 |
| single_industry_top1_801970 | top | onset | 全A | 9 | 0.444444 | 14 | 0.285714 | 10 | 3 | -2 |
| single_industry_top1_801970 | top | onset | 国证2000 | 11 | 0.363636 | 14 | 0.285714 | 10 | 3 | -2 |
| single_industry_top1_801970 | top | onset | 中证1000 | 8 | 0.625 | 14 | 0.357143 | 9 | 2 | -2 |
| single_industry_top1_801970 | top | onset | 沪深300 | 9 | 0.444444 | 14 | 0.285714 | 10 | 3 | 3.5 |
| single_industry_top1_801970 | top | onset | 中证500 | 9 | 0.333333 | 14 | 0.214286 | 11 | 3 | -2 |
| single_industry_top1_801970 | top | onset | 微盘股 | 12 | 0.416667 | 14 | 0.357143 | 9 | 3 | -2 |
| single_industry_top1_801970 | top | onset | 上证指数 | 10 | 0.3 | 14 | 0.214286 | 11 | 3 | -2 |
| single_industry_top1_801980 | top | capped_confirmation | 全A | 9 | 0.555556 | 14 | 0.357143 | 9 | 1 | -5 |
| single_industry_top1_801980 | top | capped_confirmation | 国证2000 | 11 | 0.545455 | 14 | 0.428571 | 8 | 2 | -0.5 |
| single_industry_top1_801980 | top | capped_confirmation | 中证1000 | 8 | 0.625 | 14 | 0.357143 | 9 | 1 | -7 |
| single_industry_top1_801980 | top | capped_confirmation | 沪深300 | 9 | 0.444444 | 14 | 0.285714 | 10 | 2 | -8.5 |
| single_industry_top1_801980 | top | capped_confirmation | 中证500 | 9 | 0.444444 | 14 | 0.285714 | 10 | 1 | -6 |
| single_industry_top1_801980 | top | capped_confirmation | 微盘股 | 12 | 0.583333 | 14 | 0.5 | 7 | 3 | -2 |
| single_industry_top1_801980 | top | capped_confirmation | 上证指数 | 10 | 0.6 | 14 | 0.428571 | 8 | 2 | -3.5 |
| single_industry_top1_801980 | top | onset | 全A | 9 | 0.555556 | 14 | 0.357143 | 9 | 1 | -6 |
| single_industry_top1_801980 | top | onset | 国证2000 | 11 | 0.545455 | 14 | 0.428571 | 8 | 2 | -1.5 |
| single_industry_top1_801980 | top | onset | 中证1000 | 8 | 0.625 | 14 | 0.357143 | 9 | 1 | -8 |
| single_industry_top1_801980 | top | onset | 沪深300 | 9 | 0.444444 | 14 | 0.285714 | 10 | 2 | -4.5 |
| single_industry_top1_801980 | top | onset | 中证500 | 9 | 0.444444 | 14 | 0.285714 | 10 | 1 | -7 |
| single_industry_top1_801980 | top | onset | 微盘股 | 12 | 0.583333 | 14 | 0.5 | 7 | 3 | -3 |
| single_industry_top1_801980 | top | onset | 上证指数 | 10 | 0.6 | 14 | 0.428571 | 8 | 2 | -4.5 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 2378 |
| false_alarm | 9420 |
| matched | 3014 |
| missed_region | 4730 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `single_industry_top1_801010/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 35 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 14/17/52。
- `single_industry_top1_801010/top/single_industry_top1_v1_20170103_20260814/onset`：存在 39 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 13/16/52。
- `single_industry_top1_801020/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：区域窗口不完整：预测 0 个、确认 1 个；对应时点召回切片已从分母排除。 存在 39 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 14/14/21。
- `single_industry_top1_801020/top/single_industry_top1_v1_20170103_20260814/onset`：区域窗口不完整：预测 0 个、确认 1 个；对应时点召回切片已从分母排除。 存在 38 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 13/13/20。
- `single_industry_top1_801030/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 7 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 7/7/10。
- `single_industry_top1_801030/top/single_industry_top1_v1_20170103_20260814/onset`：存在 7 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 0/0/10。
- `single_industry_top1_801040/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 50 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 33/33/74。
- `single_industry_top1_801040/top/single_industry_top1_v1_20170103_20260814/onset`：存在 48 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 24/28/72。
- `single_industry_top1_801050/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 66 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 22/23/57。
- `single_industry_top1_801050/top/single_industry_top1_v1_20170103_20260814/onset`：存在 50 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 22/22/56。
- `single_industry_top1_801080/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 45 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 20/21/58。
- `single_industry_top1_801080/top/single_industry_top1_v1_20170103_20260814/onset`：存在 45 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 25/25/60。
- `single_industry_top1_801110/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 10 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 13/19/35。
- `single_industry_top1_801110/top/single_industry_top1_v1_20170103_20260814/onset`：存在 12 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 10/11/32。
- `single_industry_top1_801120/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 50 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 14/23/64。
- `single_industry_top1_801120/top/single_industry_top1_v1_20170103_20260814/onset`：存在 53 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 15/25/68。
- `single_industry_top1_801130/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 16 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 9/9/34。
- `single_industry_top1_801130/top/single_industry_top1_v1_20170103_20260814/onset`：存在 17 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 15/15/34。
- `single_industry_top1_801140/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：区域命中对口径敏感：strict/loose/window 分别为 7/8/25。
- `single_industry_top1_801140/top/single_industry_top1_v1_20170103_20260814/onset`：区域命中对口径敏感：strict/loose/window 分别为 0/1/22。
- `single_industry_top1_801150/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 17 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 28/29/59。
- `single_industry_top1_801150/top/single_industry_top1_v1_20170103_20260814/onset`：存在 17 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 20/24/60。
- `single_industry_top1_801160/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 28 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 24/26/57。
- `single_industry_top1_801160/top/single_industry_top1_v1_20170103_20260814/onset`：存在 34 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 10/18/55。
- `single_industry_top1_801170/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 2 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 2/6/21。
- `single_industry_top1_801170/top/single_industry_top1_v1_20170103_20260814/onset`：存在 2 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 5/7/21。
- `single_industry_top1_801180/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 20 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 33/36/54。
- `single_industry_top1_801180/top/single_industry_top1_v1_20170103_20260814/onset`：存在 20 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 19/30/54。
- `single_industry_top1_801200/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 16 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 9/9/19。
- `single_industry_top1_801200/top/single_industry_top1_v1_20170103_20260814/onset`：存在 16 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 2/2/19。
- `single_industry_top1_801210/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 57 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 33/34/67。
- `single_industry_top1_801210/top/single_industry_top1_v1_20170103_20260814/onset`：存在 57 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 35/35/62。
- `single_industry_top1_801230/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 39 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 24/24/58。
- `single_industry_top1_801230/top/single_industry_top1_v1_20170103_20260814/onset`：存在 41 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 19/19/58。
- `single_industry_top1_801710/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 28 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 5/11/25。
- `single_industry_top1_801710/top/single_industry_top1_v1_20170103_20260814/onset`：存在 28 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 3/9/25。
- `single_industry_top1_801720/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 6 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 11/11/30。
- `single_industry_top1_801720/top/single_industry_top1_v1_20170103_20260814/onset`：存在 11 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 4/4/27。
- `single_industry_top1_801730/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 36 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 14/14/33。
- `single_industry_top1_801730/top/single_industry_top1_v1_20170103_20260814/onset`：存在 36 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 7/7/33。
- `single_industry_top1_801740/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 70 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 38/43/94。
- `single_industry_top1_801740/top/single_industry_top1_v1_20170103_20260814/onset`：存在 71 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 37/42/91。
- `single_industry_top1_801750/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 61 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 42/46/66。
- `single_industry_top1_801750/top/single_industry_top1_v1_20170103_20260814/onset`：存在 62 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 36/38/66。
- `single_industry_top1_801760/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 41 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 22/27/50。
- `single_industry_top1_801760/top/single_industry_top1_v1_20170103_20260814/onset`：存在 43 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 17/20/50。
- `single_industry_top1_801770/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 73 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 23/30/86。
- `single_industry_top1_801770/top/single_industry_top1_v1_20170103_20260814/onset`：存在 74 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 23/27/85。
- `single_industry_top1_801780/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：window 汇总呈现高区域召回但低 episode 精确率：88.5%/13.5%；需同时关注报警密度，不能只读取召回率。 存在 176 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 53/59/116。
- `single_industry_top1_801780/top/single_industry_top1_v1_20170103_20260814/onset`：window 汇总呈现高区域召回但低 episode 精确率：88.5%/13.5%；需同时关注报警密度，不能只读取召回率。 存在 179 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 53/60/116。
- `single_industry_top1_801790/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 52 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 26/26/70。
- `single_industry_top1_801790/top/single_industry_top1_v1_20170103_20260814/onset`：存在 48 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 22/23/70。
- `single_industry_top1_801880/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：存在 2 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 9/9/19。
- `single_industry_top1_801880/top/single_industry_top1_v1_20170103_20260814/onset`：存在 6 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 2/2/19。
- `single_industry_top1_801890/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：区域命中对口径敏感：strict/loose/window 分别为 9/9/17。
- `single_industry_top1_801890/top/single_industry_top1_v1_20170103_20260814/onset`：区域命中对口径敏感：strict/loose/window 分别为 2/2/17。
- `single_industry_top1_801950/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：window 汇总呈现高区域召回但低 episode 精确率：85.3%/16.2%；需同时关注报警密度，不能只读取召回率。 区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 90 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 27/28/58。
- `single_industry_top1_801950/top/single_industry_top1_v1_20170103_20260814/onset`：window 汇总呈现高区域召回但低 episode 精确率：83.8%/16.0%；需同时关注报警密度，不能只读取召回率。 区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 91 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 25/26/57。
- `single_industry_top1_801960/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 18 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 11/13/23。
- `single_industry_top1_801960/top/single_industry_top1_v1_20170103_20260814/onset`：区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 19 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 5/6/21。
- `single_industry_top1_801970/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 20 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 6/6/28。
- `single_industry_top1_801970/top/single_industry_top1_v1_20170103_20260814/onset`：区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 20 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 6/6/28。
- `single_industry_top1_801980/top/single_industry_top1_v1_20170103_20260814/capped_confirmation`：区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 12 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 11/18/37。
- `single_industry_top1_801980/top/single_industry_top1_v1_20170103_20260814/onset`：区域窗口不完整：预测 3 个、确认 0 个；对应时点召回切片已从分母排除。 存在 12 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 11/18/37。
