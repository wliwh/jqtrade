# 顶底区域定位评测

- 评测版本：`ma_period_breadth_decomposition_v1_20120104_20260814__stage_d_v1`
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
| ma120_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 260 | 5 | 0.0192308 | 385 | 5 | 0.012987 | 380 | 67 | 0 |
| ma120_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 260 | 4 | 0.0153846 | 385 | 4 | 0.0103896 | 381 | 67 | 0 |
| ma120_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | window | 260 | 149 | 0.573077 | 385 | 149 | 0.387013 | 236 | 67 | 6 |
| ma120_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | loose | 260 | 12 | 0.0461538 | 385 | 12 | 0.0311688 | 373 | 71 | 4 |
| ma120_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | strict | 260 | 11 | 0.0423077 | 385 | 11 | 0.0285714 | 374 | 71 | 4 |
| ma120_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | window | 260 | 149 | 0.573077 | 385 | 149 | 0.387013 | 236 | 71 | 5 |
| ma120_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 255 | 27 | 0.105882 | 252 | 27 | 0.107143 | 225 | 47 | 2 |
| ma120_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 255 | 23 | 0.0901961 | 252 | 23 | 0.0912698 | 229 | 47 | 1 |
| ma120_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | window | 255 | 81 | 0.317647 | 252 | 81 | 0.321429 | 171 | 47 | 3 |
| ma120_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | loose | 255 | 30 | 0.117647 | 252 | 30 | 0.119048 | 222 | 46 | 2 |
| ma120_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | strict | 255 | 27 | 0.105882 | 252 | 27 | 0.107143 | 225 | 46 | 2 |
| ma120_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | window | 255 | 79 | 0.309804 | 252 | 79 | 0.313492 | 173 | 46 | 2 |
| ma20_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 260 | 55 | 0.211538 | 581 | 55 | 0.0946644 | 526 | 104 | 4 |
| ma20_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 260 | 42 | 0.161538 | 581 | 42 | 0.0722892 | 539 | 104 | 4 |
| ma20_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | window | 260 | 198 | 0.761538 | 581 | 198 | 0.340792 | 383 | 104 | 2 |
| ma20_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | loose | 260 | 77 | 0.296154 | 581 | 77 | 0.13253 | 504 | 101 | 1 |
| ma20_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | strict | 260 | 55 | 0.211538 | 581 | 55 | 0.0946644 | 526 | 101 | 0 |
| ma20_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | window | 260 | 196 | 0.753846 | 581 | 196 | 0.337349 | 385 | 101 | 2 |
| ma20_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 255 | 89 | 0.34902 | 574 | 89 | 0.155052 | 485 | 112 | 0 |
| ma20_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 255 | 74 | 0.290196 | 574 | 74 | 0.12892 | 500 | 112 | 0 |
| ma20_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | window | 255 | 174 | 0.682353 | 574 | 174 | 0.303136 | 400 | 112 | 0 |
| ma20_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | loose | 255 | 96 | 0.376471 | 581 | 96 | 0.165232 | 485 | 112 | 1 |
| ma20_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | strict | 255 | 94 | 0.368627 | 581 | 94 | 0.16179 | 487 | 112 | 1 |
| ma20_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | window | 255 | 172 | 0.67451 | 581 | 172 | 0.296041 | 409 | 112 | 0 |
| ma60_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 260 | 25 | 0.0961538 | 539 | 25 | 0.0463822 | 514 | 191 | 5 |
| ma60_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 260 | 12 | 0.0461538 | 539 | 12 | 0.0222635 | 527 | 191 | 10 |
| ma60_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | window | 260 | 184 | 0.707692 | 539 | 184 | 0.341373 | 355 | 191 | 5 |
| ma60_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | loose | 260 | 32 | 0.123077 | 539 | 32 | 0.0593692 | 507 | 188 | 1 |
| ma60_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | strict | 260 | 14 | 0.0538462 | 539 | 14 | 0.025974 | 525 | 188 | 1 |
| ma60_breadth_reversal_bottom | bottom | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | window | 260 | 186 | 0.715385 | 539 | 186 | 0.345083 | 353 | 188 | 4 |
| ma60_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | loose | 255 | 27 | 0.105882 | 287 | 27 | 0.0940767 | 260 | 69 | 3 |
| ma60_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | strict | 255 | 21 | 0.0823529 | 287 | 21 | 0.0731707 | 266 | 69 | 3 |
| ma60_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | capped_confirmation | window | 255 | 86 | 0.337255 | 287 | 86 | 0.299652 | 201 | 69 | 3 |
| ma60_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | loose | 255 | 36 | 0.141176 | 287 | 36 | 0.125436 | 251 | 68 | 2 |
| ma60_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | strict | 255 | 35 | 0.137255 | 287 | 35 | 0.121951 | 252 | 68 | 2 |
| ma60_breadth_reversal_top | top | ma_period_breadth_decomposition_v1_20120104_20260814 | onset | window | 255 | 84 | 0.329412 | 287 | 84 | 0.292683 | 203 | 68 | 2.5 |

## 各指数 window 口径

| signal_id | direction | event_kind | index_name | region_count | region_recall | episode_count | episode_precision | false_alarm_count | duplicate_alarm_count | median_lead_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 37 | 0.567568 | 55 | 0.381818 | 34 | 9 | 6 |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 32 | 0.65625 | 55 | 0.381818 | 34 | 11 | 5 |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 34 | 0.676471 | 55 | 0.418182 | 32 | 9 | 5 |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 39 | 0.487179 | 55 | 0.345455 | 36 | 9 | 6 |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 38 | 0.578947 | 55 | 0.4 | 33 | 10 | 6 |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 33 | 0.636364 | 55 | 0.381818 | 34 | 10 | 5 |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 47 | 0.468085 | 55 | 0.4 | 33 | 9 | 6 |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 37 | 0.567568 | 55 | 0.381818 | 34 | 10 | 5 |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 32 | 0.65625 | 55 | 0.381818 | 34 | 11 | 4 |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 34 | 0.676471 | 55 | 0.418182 | 32 | 10 | 4 |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 39 | 0.487179 | 55 | 0.345455 | 36 | 9 | 5 |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 38 | 0.578947 | 55 | 0.4 | 33 | 11 | 5 |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 33 | 0.636364 | 55 | 0.381818 | 34 | 10 | 4 |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 47 | 0.468085 | 55 | 0.4 | 33 | 10 | 5 |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 36 | 0.305556 | 36 | 0.305556 | 25 | 7 | 3 |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 31 | 0.354839 | 36 | 0.305556 | 25 | 8 | 3 |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 33 | 0.333333 | 36 | 0.305556 | 25 | 5 | 3 |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 39 | 0.307692 | 36 | 0.333333 | 24 | 7 | 2.5 |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 37 | 0.27027 | 36 | 0.277778 | 26 | 4 | 3 |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 32 | 0.34375 | 36 | 0.305556 | 25 | 5 | 3 |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 47 | 0.319149 | 36 | 0.416667 | 21 | 11 | 3 |
| ma120_breadth_reversal_top | top | onset | 全A | 36 | 0.305556 | 36 | 0.305556 | 25 | 6 | 2 |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 31 | 0.354839 | 36 | 0.305556 | 25 | 8 | 4 |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 33 | 0.30303 | 36 | 0.277778 | 26 | 6 | 4 |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 39 | 0.307692 | 36 | 0.333333 | 24 | 7 | 2.5 |
| ma120_breadth_reversal_top | top | onset | 中证500 | 37 | 0.27027 | 36 | 0.277778 | 26 | 4 | 2 |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 32 | 0.3125 | 36 | 0.277778 | 26 | 5 | 2.5 |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 47 | 0.319149 | 36 | 0.416667 | 21 | 10 | 2 |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 37 | 0.783784 | 83 | 0.349398 | 54 | 14 | 2 |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 32 | 0.84375 | 83 | 0.325301 | 56 | 15 | 2 |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 34 | 0.794118 | 83 | 0.325301 | 56 | 15 | 3 |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 39 | 0.692308 | 83 | 0.325301 | 56 | 12 | 2 |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 38 | 0.763158 | 83 | 0.349398 | 54 | 16 | 3 |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 33 | 0.818182 | 83 | 0.325301 | 56 | 15 | 4 |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 47 | 0.680851 | 83 | 0.385542 | 51 | 17 | 2 |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 37 | 0.783784 | 83 | 0.349398 | 54 | 14 | 1 |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 32 | 0.8125 | 83 | 0.313253 | 57 | 15 | 2 |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 34 | 0.794118 | 83 | 0.325301 | 56 | 14 | 2 |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 39 | 0.692308 | 83 | 0.325301 | 56 | 11 | 1 |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 38 | 0.763158 | 83 | 0.349398 | 54 | 15 | 2 |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 33 | 0.787879 | 83 | 0.313253 | 57 | 16 | 2.5 |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 47 | 0.680851 | 83 | 0.385542 | 51 | 16 | 1 |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 36 | 0.75 | 82 | 0.329268 | 55 | 19 | 0 |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 31 | 0.774194 | 82 | 0.292683 | 58 | 12 | -1.5 |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 33 | 0.727273 | 82 | 0.292683 | 58 | 17 | -2 |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 39 | 0.615385 | 82 | 0.292683 | 58 | 16 | 3 |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 37 | 0.675676 | 82 | 0.304878 | 57 | 15 | 0 |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 32 | 0.65625 | 82 | 0.256098 | 61 | 12 | -2 |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 47 | 0.617021 | 82 | 0.353659 | 53 | 21 | 2 |
| ma20_breadth_reversal_top | top | onset | 全A | 36 | 0.75 | 83 | 0.325301 | 56 | 18 | -1 |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 31 | 0.741935 | 83 | 0.277108 | 60 | 12 | 0 |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 33 | 0.727273 | 83 | 0.289157 | 59 | 17 | 0 |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 39 | 0.615385 | 83 | 0.289157 | 59 | 17 | 2 |
| ma20_breadth_reversal_top | top | onset | 中证500 | 37 | 0.675676 | 83 | 0.301205 | 58 | 15 | 0 |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 32 | 0.625 | 83 | 0.240964 | 63 | 12 | -1.5 |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 47 | 0.617021 | 83 | 0.349398 | 54 | 21 | 1 |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 37 | 0.702703 | 77 | 0.337662 | 51 | 29 | 4 |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 32 | 0.875 | 77 | 0.363636 | 49 | 29 | 5 |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 34 | 0.794118 | 77 | 0.350649 | 50 | 30 | 5 |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 39 | 0.538462 | 77 | 0.272727 | 56 | 26 | 4 |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 38 | 0.710526 | 77 | 0.350649 | 50 | 30 | 4 |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 33 | 0.878788 | 77 | 0.376623 | 48 | 20 | 6 |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 47 | 0.553191 | 77 | 0.337662 | 51 | 27 | 4 |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 37 | 0.72973 | 77 | 0.350649 | 50 | 28 | 4 |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 32 | 0.875 | 77 | 0.363636 | 49 | 29 | 4 |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 34 | 0.794118 | 77 | 0.350649 | 50 | 30 | 4 |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 39 | 0.564103 | 77 | 0.285714 | 55 | 25 | 4 |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 38 | 0.710526 | 77 | 0.350649 | 50 | 30 | 4 |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 33 | 0.848485 | 77 | 0.363636 | 49 | 20 | 5 |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 47 | 0.574468 | 77 | 0.350649 | 50 | 26 | 4 |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 36 | 0.361111 | 41 | 0.317073 | 28 | 12 | 3 |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 31 | 0.322581 | 41 | 0.243902 | 31 | 9 | 3.5 |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 33 | 0.333333 | 41 | 0.268293 | 30 | 8 | 2 |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 39 | 0.358974 | 41 | 0.341463 | 27 | 10 | 3.5 |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 37 | 0.324324 | 41 | 0.292683 | 29 | 9 | 3.5 |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 32 | 0.34375 | 41 | 0.268293 | 30 | 10 | 2 |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 47 | 0.319149 | 41 | 0.365854 | 26 | 11 | 5 |
| ma60_breadth_reversal_top | top | onset | 全A | 36 | 0.361111 | 41 | 0.317073 | 28 | 12 | 2 |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 31 | 0.322581 | 41 | 0.243902 | 31 | 8 | 2 |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 33 | 0.333333 | 41 | 0.268293 | 30 | 8 | 1 |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 39 | 0.333333 | 41 | 0.317073 | 28 | 11 | 3 |
| ma60_breadth_reversal_top | top | onset | 中证500 | 37 | 0.324324 | 41 | 0.292683 | 29 | 9 | 2.5 |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 32 | 0.34375 | 41 | 0.268293 | 30 | 9 | 1 |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 47 | 0.297872 | 41 | 0.341463 | 27 | 11 | 4.5 |

## 明细状态计数

| match_status | rows |
| --- | --- |
| duplicate_alarm | 1176 |
| false_alarm | 2329 |
| matched | 1738 |
| missed_region | 1352 |

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

- `ma120_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 67 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 4/5/149。
- `ma120_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 71 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 11/12/149。
- `ma120_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：存在 47 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 23/27/81。
- `ma120_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：存在 46 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 27/30/79。
- `ma20_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 104 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 42/55/198。
- `ma20_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 101 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 55/77/196。
- `ma20_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：存在 112 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 74/89/174。
- `ma20_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：存在 112 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 94/96/172。
- `ma60_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 191 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 12/25/184。
- `ma60_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：区域窗口不完整：预测 7 个、确认 5 个；对应时点召回切片已从分母排除。 存在 188 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 14/32/186。
- `ma60_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：存在 69 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 21/27/86。
- `ma60_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：存在 68 个指数×episode 重复报警，未计为主匹配。 区域命中对口径敏感：strict/loose/window 分别为 35/36/84。
