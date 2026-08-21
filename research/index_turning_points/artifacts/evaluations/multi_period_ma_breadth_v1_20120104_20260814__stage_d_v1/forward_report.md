# 信号后 OHLC 结果评测

- 评测版本：`multi_period_ma_breadth_v1_20120104_20260814__stage_d_v1`
- 本报告与区域定位报告相互独立，不生成综合总分，也不构成交易回测。

## 口径

对事件日 `t` 和未来 `h` 个指数交易日：

```text
terminal_return_h = close[t+h] / close[t] - 1
max_up_h          = max(high[t+1:t+h]) / close[t] - 1
max_down_h        = min(low[t+1:t+h]) / close[t] - 1
```

- 分别评价 onset 与 capped confirmation 的 5/10/20 日路径；事件日缺失和尾部窗口不完整会保留在明细中，但不进入统计。
- 基线是同一信号覆盖期、同一指数、同一期限的完整非事件日。
- 均值差使用 `结果 ~ 常数 + 事件指示变量`，Newey–West 滞后阶数等于期限；95% 区间使用正态近似。
- 推断至少需要 20 个完整事件和 30 个完整基线日。样本不足时保留分布描述，显著性和 FDR 留空。
- 局部 FDR 家族为同一 signal/direction/version/event kind 下的全部指数、期限和结果；全局 FDR 覆盖 bundle 中全部合格检验。
- `close[t]` 只是统一参考价，不代表信号能在该收盘价成交；结果不能解释成含成本交易收益。

## 可用性

| index_name | event_kind | horizon | events | event_dates_available | complete_windows |
| --- | --- | --- | --- | --- | --- |
| 上证指数 | capped_confirmation | 5 | 106 | 106 | 106 |
| 上证指数 | capped_confirmation | 10 | 106 | 106 | 105 |
| 上证指数 | capped_confirmation | 20 | 106 | 106 | 104 |
| 上证指数 | onset | 5 | 106 | 106 | 106 |
| 上证指数 | onset | 10 | 106 | 106 | 106 |
| 上证指数 | onset | 20 | 106 | 106 | 104 |
| 中证1000 | capped_confirmation | 5 | 106 | 106 | 106 |
| 中证1000 | capped_confirmation | 10 | 106 | 106 | 105 |
| 中证1000 | capped_confirmation | 20 | 106 | 106 | 104 |
| 中证1000 | onset | 5 | 106 | 106 | 106 |
| 中证1000 | onset | 10 | 106 | 106 | 106 |
| 中证1000 | onset | 20 | 106 | 106 | 104 |
| 中证500 | capped_confirmation | 5 | 106 | 106 | 106 |
| 中证500 | capped_confirmation | 10 | 106 | 106 | 105 |
| 中证500 | capped_confirmation | 20 | 106 | 106 | 104 |
| 中证500 | onset | 5 | 106 | 106 | 106 |
| 中证500 | onset | 10 | 106 | 106 | 106 |
| 中证500 | onset | 20 | 106 | 106 | 104 |
| 全A | capped_confirmation | 5 | 106 | 106 | 106 |
| 全A | capped_confirmation | 10 | 106 | 106 | 105 |
| 全A | capped_confirmation | 20 | 106 | 106 | 104 |
| 全A | onset | 5 | 106 | 106 | 106 |
| 全A | onset | 10 | 106 | 106 | 106 |
| 全A | onset | 20 | 106 | 106 | 104 |
| 国证2000 | capped_confirmation | 5 | 106 | 106 | 106 |
| 国证2000 | capped_confirmation | 10 | 106 | 106 | 105 |
| 国证2000 | capped_confirmation | 20 | 106 | 106 | 104 |
| 国证2000 | onset | 5 | 106 | 106 | 106 |
| 国证2000 | onset | 10 | 106 | 106 | 106 |
| 国证2000 | onset | 20 | 106 | 106 | 104 |
| 微盘股 | capped_confirmation | 5 | 106 | 106 | 106 |
| 微盘股 | capped_confirmation | 10 | 106 | 106 | 105 |
| 微盘股 | capped_confirmation | 20 | 106 | 106 | 104 |
| 微盘股 | onset | 5 | 106 | 106 | 106 |
| 微盘股 | onset | 10 | 106 | 106 | 106 |
| 微盘股 | onset | 20 | 106 | 106 | 104 |
| 沪深300 | capped_confirmation | 5 | 106 | 106 | 106 |
| 沪深300 | capped_confirmation | 10 | 106 | 106 | 105 |
| 沪深300 | capped_confirmation | 20 | 106 | 106 | 104 |
| 沪深300 | onset | 5 | 106 | 106 | 106 |
| 沪深300 | onset | 10 | 106 | 106 | 106 |
| 沪深300 | onset | 20 | 106 | 106 | 104 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 5 | max_down | 63 | -0.0241497 | 3479 | -0.0219911 | -0.00215861 | -0.00812664 | 0.00380943 | 0.478373 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 5 | max_up | 63 | 0.0211355 | 3479 | 0.0195966 | 0.00153895 | -0.00361303 | 0.00669094 | 0.55823 | 0.770058 | 0.805012 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 5 | terminal_return | 63 | 0.000678635 | 3479 | 0.00172578 | -0.00104715 | -0.00873405 | 0.00663975 | 0.789468 | 0.857526 | 0.888152 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 10 | max_down | 62 | -0.0341961 | 3475 | -0.0317201 | -0.00247604 | -0.0108501 | 0.00589804 | 0.562231 | 0.770058 | 0.805012 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 10 | max_up | 62 | 0.033459 | 3475 | 0.0292364 | 0.00422261 | -0.0028098 | 0.011255 | 0.239242 | 0.770058 | 0.705362 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 10 | terminal_return | 62 | 0.00681172 | 3475 | 0.00331386 | 0.00349786 | -0.00779996 | 0.0147957 | 0.543967 | 0.770058 | 0.796975 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 20 | max_down | 61 | -0.0510575 | 3466 | -0.0451105 | -0.00594704 | -0.0178716 | 0.00597749 | 0.328323 | 0.770058 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 20 | max_up | 61 | 0.0589529 | 3466 | 0.044094 | 0.0148589 | -0.00587128 | 0.035589 | 0.160057 | 0.770058 | 0.584555 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 全A | 20 | terminal_return | 61 | 0.0149359 | 3466 | 0.00650596 | 0.0084299 | -0.0143336 | 0.0311934 | 0.467938 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_down | 63 | -0.0301273 | 3481 | -0.0266772 | -0.00345013 | -0.0121103 | 0.00521001 | 0.434891 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_up | 63 | 0.0272766 | 3481 | 0.0234047 | 0.0038719 | -0.00359077 | 0.0113346 | 0.309192 | 0.770058 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 63 | 0.00133984 | 3481 | 0.00267751 | -0.00133767 | -0.0124832 | 0.00980784 | 0.814025 | 0.869214 | 0.903676 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_down | 62 | -0.0442724 | 3477 | -0.0388999 | -0.00537247 | -0.0171338 | 0.00638886 | 0.370621 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_up | 62 | 0.0427991 | 3477 | 0.035817 | 0.00698218 | -0.00340099 | 0.0173654 | 0.187502 | 0.770058 | 0.613642 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 62 | 0.00770568 | 3477 | 0.00525123 | 0.00245445 | -0.0129558 | 0.0178647 | 0.754907 | 0.84927 | 0.866093 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_down | 61 | -0.06717 | 3468 | -0.0558104 | -0.0113596 | -0.027788 | 0.00506884 | 0.175335 | 0.770058 | 0.595199 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_up | 61 | 0.0741953 | 3468 | 0.054727 | 0.0194682 | -0.00461249 | 0.043549 | 0.113063 | 0.770058 | 0.474863 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 61 | 0.0169463 | 3468 | 0.0104465 | 0.00649986 | -0.0221492 | 0.0351489 | 0.656549 | 0.770058 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_down | 63 | -0.0299425 | 3481 | -0.027103 | -0.00283949 | -0.0108571 | 0.00517817 | 0.487593 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_up | 63 | 0.0272337 | 3481 | 0.0234955 | 0.00373815 | -0.00352841 | 0.0110047 | 0.313317 | 0.770058 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 63 | 0.00139887 | 3481 | 0.00202905 | -0.000630185 | -0.0110957 | 0.0098353 | 0.90605 | 0.924012 | 0.949895 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_down | 62 | -0.0443516 | 3477 | -0.0395659 | -0.0047857 | -0.0162771 | 0.0067057 | 0.414351 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_up | 62 | 0.0431546 | 3477 | 0.0354784 | 0.00767626 | -0.00266557 | 0.0180181 | 0.14572 | 0.770058 | 0.564945 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 62 | 0.00730662 | 3477 | 0.00394394 | 0.00336268 | -0.0121379 | 0.0188633 | 0.670691 | 0.770058 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_down | 61 | -0.0678381 | 3468 | -0.0568104 | -0.0110277 | -0.0270314 | 0.00497603 | 0.176831 | 0.770058 | 0.595199 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_up | 61 | 0.0738196 | 3468 | 0.0537384 | 0.0200812 | -0.00395533 | 0.0441178 | 0.101532 | 0.770058 | 0.473815 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 61 | 0.0143614 | 3468 | 0.00782338 | 0.00653803 | -0.0224708 | 0.0355469 | 0.658673 | 0.770058 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_down | 63 | -0.0214072 | 3481 | -0.0202626 | -0.00114463 | -0.00629764 | 0.00400837 | 0.663291 | 0.770058 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_up | 63 | 0.0193452 | 3481 | 0.0195875 | -0.000242285 | -0.00441283 | 0.00392826 | 0.909345 | 0.924012 | 0.949895 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 63 | -0.000151438 | 3481 | 0.00147424 | -0.00162567 | -0.00847306 | 0.00522171 | 0.641692 | 0.770058 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_down | 62 | -0.0307298 | 3477 | -0.0290339 | -0.00169593 | -0.00955356 | 0.00616171 | 0.672273 | 0.770058 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_up | 62 | 0.030816 | 3477 | 0.0291561 | 0.00165988 | -0.00421822 | 0.00753798 | 0.57994 | 0.770058 | 0.813491 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 62 | 0.00671253 | 3477 | 0.0027567 | 0.00395584 | -0.00609685 | 0.0140085 | 0.440541 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_down | 61 | -0.0435117 | 3468 | -0.0409107 | -0.00260105 | -0.0139386 | 0.00873654 | 0.652956 | 0.770058 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_up | 61 | 0.0550582 | 3468 | 0.0437997 | 0.0112585 | -0.0078883 | 0.0304053 | 0.249116 | 0.770058 | 0.705362 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 61 | 0.0157417 | 3468 | 0.00540874 | 0.0103329 | -0.00985663 | 0.0305225 | 0.315803 | 0.770058 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 5 | max_down | 63 | -0.0267344 | 3481 | -0.0244783 | -0.0022561 | -0.00916142 | 0.00464923 | 0.521933 | 0.770058 | 0.7829 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 5 | max_up | 63 | 0.0241204 | 3481 | 0.0219396 | 0.00218084 | -0.0037508 | 0.00811248 | 0.471144 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 63 | 0.00164277 | 3481 | 0.00198463 | -0.000341863 | -0.00905826 | 0.00837453 | 0.938725 | 0.938725 | 0.969503 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 10 | max_down | 62 | -0.0379002 | 3477 | -0.0354232 | -0.002477 | -0.012503 | 0.00754897 | 0.628219 | 0.770058 | 0.842081 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 10 | max_up | 62 | 0.0391345 | 3477 | 0.0327911 | 0.00634333 | -0.00203181 | 0.0147185 | 0.137675 | 0.770058 | 0.542097 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 62 | 0.00844914 | 3477 | 0.00382625 | 0.0046229 | -0.00816185 | 0.0174076 | 0.478495 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 20 | max_down | 61 | -0.0576662 | 3468 | -0.0503827 | -0.00728348 | -0.0208834 | 0.00631649 | 0.293865 | 0.770058 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 20 | max_up | 61 | 0.067696 | 3468 | 0.0496179 | 0.018078 | -0.00410894 | 0.040265 | 0.110262 | 0.770058 | 0.474863 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 61 | 0.0159395 | 3468 | 0.00759135 | 0.00834816 | -0.017177 | 0.0338733 | 0.521503 | 0.770058 | 0.7829 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_down | 63 | -0.035166 | 3481 | -0.0275835 | -0.00758252 | -0.0195708 | 0.00440574 | 0.21509 | 0.770058 | 0.686109 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_up | 63 | 0.0318768 | 3481 | 0.0260877 | 0.00578917 | -0.00325487 | 0.0148332 | 0.209619 | 0.770058 | 0.677231 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 63 | 0.00230646 | 3481 | 0.00445745 | -0.00215099 | -0.0172011 | 0.0128991 | 0.77938 | 0.857526 | 0.880734 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_down | 62 | -0.0481375 | 3477 | -0.0402487 | -0.00788887 | -0.022811 | 0.00703326 | 0.300113 | 0.770058 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_up | 62 | 0.0508362 | 3477 | 0.0400767 | 0.0107595 | -0.00170146 | 0.0232204 | 0.0905744 | 0.770058 | 0.438938 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 62 | 0.0133899 | 3477 | 0.00869656 | 0.00469334 | -0.0126968 | 0.0220835 | 0.596822 | 0.770058 | 0.813491 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_down | 61 | -0.0667953 | 3468 | -0.0578643 | -0.00893096 | -0.0310483 | 0.0131864 | 0.428683 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_up | 61 | 0.0864158 | 3468 | 0.0614782 | 0.0249376 | 0.000227344 | 0.0496478 | 0.047925 | 0.770058 | 0.340774 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 61 | 0.0347489 | 3468 | 0.0169129 | 0.017836 | -0.0107834 | 0.0464553 | 0.221897 | 0.770058 | 0.690346 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_down | 63 | -0.0207772 | 3481 | -0.0189043 | -0.00187294 | -0.00675976 | 0.00301388 | 0.452535 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_up | 63 | 0.0167289 | 3481 | 0.0171541 | -0.000425221 | -0.00442032 | 0.00356988 | 0.83475 | 0.876487 | 0.914595 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 63 | -0.000472647 | 3481 | 0.00125703 | -0.00172967 | -0.00802247 | 0.00456312 | 0.590069 | 0.770058 | 0.813491 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_down | 62 | -0.0293511 | 3477 | -0.027116 | -0.00223512 | -0.00954643 | 0.0050762 | 0.549049 | 0.770058 | 0.799771 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_up | 62 | 0.0268277 | 3477 | 0.0256368 | 0.00119097 | -0.00423479 | 0.00661674 | 0.667031 | 0.770058 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 62 | 0.00444012 | 3477 | 0.00235439 | 0.00208573 | -0.00720166 | 0.0113731 | 0.659814 | 0.770058 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_down | 61 | -0.0413155 | 3468 | -0.0384517 | -0.00286386 | -0.0134864 | 0.00775863 | 0.597205 | 0.770058 | 0.813491 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_up | 61 | 0.0475061 | 3468 | 0.0385603 | 0.00894581 | -0.00774472 | 0.0256363 | 0.293478 | 0.770058 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 61 | 0.0120991 | 3468 | 0.00453779 | 0.00756127 | -0.00978187 | 0.0249044 | 0.392816 | 0.770058 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 5 | max_down | 63 | -0.0238291 | 3479 | -0.0219969 | -0.00183225 | -0.00799038 | 0.00432588 | 0.559782 | 0.823375 | 0.805012 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 5 | max_up | 63 | 0.0227813 | 3479 | 0.0195668 | 0.00321451 | -0.00190498 | 0.008334 | 0.218443 | 0.823375 | 0.688096 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 5 | terminal_return | 63 | 0.0022242 | 3479 | 0.0016978 | 0.0005264 | -0.0078025 | 0.0088553 | 0.901414 | 0.915953 | 0.949895 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 10 | max_down | 63 | -0.0341575 | 3474 | -0.0317201 | -0.00243738 | -0.0107705 | 0.00589575 | 0.566451 | 0.823375 | 0.806473 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 10 | max_up | 63 | 0.0342948 | 3474 | 0.02922 | 0.00507481 | -0.00229522 | 0.0124448 | 0.177143 | 0.743999 | 0.595199 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 10 | terminal_return | 63 | 0.00741598 | 3474 | 0.0033019 | 0.00411408 | -0.0076603 | 0.0158885 | 0.493443 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 20 | max_down | 61 | -0.050191 | 3466 | -0.0451257 | -0.0050653 | -0.0172065 | 0.00707591 | 0.413523 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 20 | max_up | 61 | 0.0582596 | 3466 | 0.0441062 | 0.0141534 | -0.0042932 | 0.0325999 | 0.132624 | 0.642715 | 0.530495 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 全A | 20 | terminal_return | 61 | 0.0156539 | 3466 | 0.00649332 | 0.00916054 | -0.0124273 | 0.0307484 | 0.405577 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 5 | max_down | 63 | -0.0284605 | 3481 | -0.0267074 | -0.0017532 | -0.0104489 | 0.00694252 | 0.692719 | 0.823375 | 0.847404 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 5 | max_up | 63 | 0.0309967 | 3481 | 0.0233373 | 0.00765934 | 0.000228464 | 0.0150902 | 0.0433564 | 0.373592 | 0.340774 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 5 | terminal_return | 63 | 0.00457863 | 3481 | 0.00261889 | 0.00195974 | -0.0105084 | 0.0144279 | 0.758028 | 0.823375 | 0.866093 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 10 | max_down | 63 | -0.0414393 | 3476 | -0.0389497 | -0.00248959 | -0.0138321 | 0.0088529 | 0.667047 | 0.823375 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 10 | max_up | 63 | 0.0471779 | 3476 | 0.0357356 | 0.0114423 | 0.00014586 | 0.0227388 | 0.0471103 | 0.373592 | 0.340774 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 10 | terminal_return | 63 | 0.0118254 | 3476 | 0.00517586 | 0.00664955 | -0.00977281 | 0.0230719 | 0.427417 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 20 | max_down | 61 | -0.063213 | 3468 | -0.05588 | -0.00733298 | -0.0236423 | 0.00897638 | 0.378182 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 20 | max_up | 61 | 0.07566 | 3468 | 0.0547013 | 0.0209587 | -0.000822811 | 0.0427402 | 0.0593004 | 0.373592 | 0.347528 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 国证2000 | 20 | terminal_return | 61 | 0.0199359 | 3468 | 0.0103939 | 0.00954198 | -0.0177607 | 0.0368446 | 0.493345 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 5 | max_down | 63 | -0.0285341 | 3481 | -0.0271285 | -0.00140562 | -0.00969099 | 0.00687975 | 0.7395 | 0.823375 | 0.865475 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 5 | max_up | 63 | 0.030924 | 3481 | 0.0234287 | 0.00749526 | -5.94981e-05 | 0.01505 | 0.0518275 | 0.373592 | 0.340774 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 5 | terminal_return | 63 | 0.00445199 | 3481 | 0.0019738 | 0.00247819 | -0.00957428 | 0.0145307 | 0.686942 | 0.823375 | 0.844436 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 10 | max_down | 63 | -0.0418898 | 3476 | -0.0396091 | -0.00228069 | -0.013283 | 0.00872165 | 0.684528 | 0.823375 | 0.844436 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 10 | max_up | 63 | 0.0464701 | 3476 | 0.0354161 | 0.011054 | -0.000184063 | 0.0222921 | 0.0538678 | 0.373592 | 0.340774 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 10 | terminal_return | 63 | 0.0108971 | 3476 | 0.0038779 | 0.00701917 | -0.0091557 | 0.023194 | 0.395018 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 20 | max_down | 61 | -0.0644299 | 3468 | -0.0568703 | -0.00755953 | -0.0232762 | 0.00815713 | 0.345815 | 0.823375 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 20 | max_up | 61 | 0.0748901 | 3468 | 0.0537196 | 0.0211705 | -0.000612597 | 0.0429536 | 0.0567957 | 0.373592 | 0.340774 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证1000 | 20 | terminal_return | 61 | 0.0171078 | 3468 | 0.00777507 | 0.00933274 | -0.0183607 | 0.0370262 | 0.508918 | 0.823375 | 0.772575 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 5 | max_down | 63 | -0.0222511 | 3481 | -0.0202473 | -0.00200384 | -0.00711667 | 0.003109 | 0.442388 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 5 | max_up | 63 | 0.0195127 | 3481 | 0.0195844 | -7.17475e-05 | -0.00438256 | 0.00423907 | 0.973976 | 0.973976 | 0.985198 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 5 | terminal_return | 63 | 0.000397409 | 3481 | 0.0014643 | -0.00106689 | -0.00764491 | 0.00551112 | 0.750565 | 0.823375 | 0.866093 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 10 | max_down | 63 | -0.0308154 | 3476 | -0.0290318 | -0.00178362 | -0.00914604 | 0.00557881 | 0.634909 | 0.823375 | 0.842404 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 10 | max_up | 63 | 0.0298004 | 3476 | 0.029174 | 0.000626363 | -0.00542934 | 0.00668207 | 0.839346 | 0.866866 | 0.915651 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 10 | terminal_return | 63 | 0.00482738 | 3476 | 0.00278973 | 0.00203765 | -0.00860348 | 0.0126788 | 0.707425 | 0.823375 | 0.852972 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 20 | max_down | 61 | -0.0441585 | 3468 | -0.0408993 | -0.00325922 | -0.015332 | 0.00881361 | 0.596717 | 0.823375 | 0.813491 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 20 | max_up | 61 | 0.0523716 | 3468 | 0.043847 | 0.00852468 | -0.00860159 | 0.025651 | 0.329263 | 0.823375 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 沪深300 | 20 | terminal_return | 61 | 0.0151917 | 3468 | 0.00541842 | 0.00977326 | -0.0095687 | 0.0291152 | 0.321996 | 0.823375 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 5 | max_down | 63 | -0.0256664 | 3481 | -0.0244976 | -0.00116878 | -0.00817657 | 0.00583901 | 0.743747 | 0.823375 | 0.865475 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 5 | max_up | 63 | 0.0269797 | 3481 | 0.0218879 | 0.00509183 | -0.00070669 | 0.0108903 | 0.0852279 | 0.447447 | 0.437685 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 5 | terminal_return | 63 | 0.00357352 | 3481 | 0.00194969 | 0.00162383 | -0.00769227 | 0.0109399 | 0.732625 | 0.823375 | 0.865475 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 10 | max_down | 63 | -0.0372789 | 3476 | -0.0354337 | -0.00184522 | -0.0117357 | 0.00804526 | 0.714613 | 0.823375 | 0.853472 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 10 | max_up | 63 | 0.0410279 | 3476 | 0.032755 | 0.00827287 | -0.000170205 | 0.0167159 | 0.0547963 | 0.373592 | 0.340774 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 10 | terminal_return | 63 | 0.00985265 | 3476 | 0.00379948 | 0.00605317 | -0.00735251 | 0.0194588 | 0.376149 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 20 | max_down | 61 | -0.0548015 | 3468 | -0.0504331 | -0.00436837 | -0.0178522 | 0.00911544 | 0.525439 | 0.823375 | 0.783495 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 20 | max_up | 61 | 0.0676153 | 3468 | 0.0496193 | 0.017996 | -0.00175769 | 0.0377496 | 0.0741643 | 0.424759 | 0.406291 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 中证500 | 20 | terminal_return | 61 | 0.0176621 | 3468 | 0.00756105 | 0.010101 | -0.0143181 | 0.0345201 | 0.417503 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 5 | max_down | 63 | -0.0317066 | 3481 | -0.0276461 | -0.00406057 | -0.0148063 | 0.00668517 | 0.458912 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 5 | max_up | 63 | 0.0368983 | 3481 | 0.0259968 | 0.0109015 | 0.00121819 | 0.0205848 | 0.0273438 | 0.373592 | 0.28711 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 5 | terminal_return | 63 | 0.00876734 | 3481 | 0.00434052 | 0.00442682 | -0.0113028 | 0.0201564 | 0.581218 | 0.823375 | 0.813491 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 10 | max_down | 63 | -0.0456299 | 3476 | -0.0402919 | -0.005338 | -0.0200585 | 0.0093825 | 0.477244 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 10 | max_up | 63 | 0.0572678 | 3476 | 0.039957 | 0.0173108 | 0.00258198 | 0.0320397 | 0.0212457 | 0.373592 | 0.281271 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 10 | terminal_return | 63 | 0.0184288 | 3476 | 0.00860389 | 0.00982494 | -0.00970289 | 0.0293528 | 0.324072 | 0.823375 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 20 | max_down | 61 | -0.0649965 | 3468 | -0.057896 | -0.00710051 | -0.0293297 | 0.0151287 | 0.53127 | 0.823375 | 0.78753 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 20 | max_up | 61 | 0.0887725 | 3468 | 0.0614368 | 0.0273357 | 0.00382369 | 0.0508477 | 0.022682 | 0.373592 | 0.281271 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 微盘股 | 20 | terminal_return | 61 | 0.0372344 | 3468 | 0.0168692 | 0.0203653 | -0.00792187 | 0.0486524 | 0.158216 | 0.711971 | 0.584555 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 5 | max_down | 63 | -0.0207468 | 3481 | -0.0189048 | -0.00184199 | -0.00674092 | 0.00305694 | 0.461148 | 0.823375 | 0.753622 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 5 | max_up | 63 | 0.017571 | 3481 | 0.0171389 | 0.000432189 | -0.00359689 | 0.00446126 | 0.833477 | 0.866866 | 0.914595 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 5 | terminal_return | 63 | 0.00045993 | 3481 | 0.00124015 | -0.000780219 | -0.00727885 | 0.00571841 | 0.813964 | 0.866866 | 0.903676 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 10 | max_down | 63 | -0.0290894 | 3476 | -0.0271201 | -0.00196929 | -0.0089455 | 0.00500693 | 0.580071 | 0.823375 | 0.813491 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 10 | max_up | 63 | 0.0267068 | 3476 | 0.0256386 | 0.0010682 | -0.00449265 | 0.00662904 | 0.706545 | 0.823375 | 0.852972 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 10 | terminal_return | 63 | 0.00400829 | 3476 | 0.00236161 | 0.00164668 | -0.0079841 | 0.0112775 | 0.737533 | 0.823375 | 0.865475 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 20 | max_down | 61 | -0.0413102 | 3468 | -0.0384518 | -0.00285844 | -0.0139182 | 0.00820128 | 0.612455 | 0.823375 | 0.829777 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 20 | max_up | 61 | 0.0455327 | 3468 | 0.038595 | 0.00693769 | -0.00748067 | 0.021356 | 0.345633 | 0.823375 | 0.751254 | true |
| multi_period_ma_breadth_bottom | bottom | onset | 上证指数 | 20 | terminal_return | 61 | 0.0123706 | 3468 | 0.00453301 | 0.00783758 | -0.00860898 | 0.0242841 | 0.350286 | 0.823375 | 0.752556 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 5 | max_down | 43 | -0.0293117 | 3499 | -0.02194 | -0.00737177 | -0.0179072 | 0.00316363 | 0.170238 | 0.466304 | 0.595199 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 5 | max_up | 43 | 0.0286623 | 3499 | 0.0195129 | 0.00914945 | 0.00251186 | 0.015787 | 0.00689816 | 0.210717 | 0.281271 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 5 | terminal_return | 43 | 0.00382351 | 3499 | 0.00168115 | 0.00214236 | -0.0115767 | 0.0158615 | 0.75955 | 0.989364 | 0.866093 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 10 | max_down | 43 | -0.0392111 | 3494 | -0.0316719 | -0.00753924 | -0.0229395 | 0.00786101 | 0.337295 | 0.607131 | 0.751254 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 10 | max_up | 43 | 0.0390512 | 3494 | 0.0291905 | 0.00986061 | 0.00138079 | 0.0183404 | 0.0226581 | 0.210717 | 0.281271 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 10 | terminal_return | 43 | 0.00391896 | 3494 | 0.00336848 | 0.000550473 | -0.0184735 | 0.0195744 | 0.954773 | 0.989364 | 0.97713 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 20 | max_down | 43 | -0.0568129 | 3484 | -0.0450702 | -0.0117427 | -0.035673 | 0.0121875 | 0.336158 | 0.607131 | 0.751254 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 20 | max_up | 43 | 0.057695 | 3484 | 0.0441863 | 0.0135087 | -0.00203031 | 0.0290477 | 0.0883982 | 0.35066 | 0.437685 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 全A | 20 | terminal_return | 43 | 0.00584434 | 3484 | 0.00666172 | -0.000817382 | -0.0282789 | 0.0266442 | 0.953479 | 0.989364 | 0.97713 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 5 | max_down | 43 | -0.0337771 | 3501 | -0.0266521 | -0.00712499 | -0.01914 | 0.00488999 | 0.245115 | 0.571934 | 0.705362 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 5 | max_up | 43 | 0.0326871 | 3501 | 0.0233603 | 0.00932675 | 0.00126019 | 0.0173933 | 0.0234393 | 0.210717 | 0.281271 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 43 | 0.00472977 | 3501 | 0.00262823 | 0.00210154 | -0.0137788 | 0.0179818 | 0.795343 | 0.989364 | 0.890784 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 10 | max_down | 43 | -0.0460996 | 3496 | -0.0389066 | -0.00719301 | -0.0254151 | 0.0110291 | 0.439112 | 0.709335 | 0.753622 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 10 | max_up | 43 | 0.0475877 | 3496 | 0.035796 | 0.0117917 | 0.000723212 | 0.0228602 | 0.0367919 | 0.210717 | 0.319709 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 43 | 0.00646398 | 3496 | 0.00527984 | 0.00118414 | -0.0222936 | 0.0246619 | 0.921253 | 0.989364 | 0.955373 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 20 | max_down | 43 | -0.0662746 | 3486 | -0.0558801 | -0.0103945 | -0.0378835 | 0.0170946 | 0.45861 | 0.72231 | 0.753622 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 20 | max_up | 43 | 0.0778986 | 3486 | 0.0547819 | 0.0231167 | -0.00383665 | 0.0500701 | 0.0927618 | 0.35066 | 0.441056 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 43 | 0.0176309 | 3486 | 0.0104716 | 0.00715931 | -0.0321247 | 0.0464433 | 0.720942 | 0.987377 | 0.856969 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 5 | max_down | 43 | -0.0343122 | 3501 | -0.0270656 | -0.0072466 | -0.0193531 | 0.00485988 | 0.240715 | 0.571934 | 0.705362 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 5 | max_up | 43 | 0.0324239 | 3501 | 0.0234531 | 0.00897076 | 0.00102306 | 0.0169185 | 0.026946 | 0.210717 | 0.28711 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 43 | 0.0032939 | 3501 | 0.00200218 | 0.00129172 | -0.0144333 | 0.0170167 | 0.872091 | 0.989364 | 0.931216 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 10 | max_down | 43 | -0.0473027 | 3496 | -0.0395556 | -0.00774708 | -0.0262458 | 0.0107516 | 0.411744 | 0.682628 | 0.753622 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 10 | max_up | 43 | 0.0464764 | 3496 | 0.0354792 | 0.0109972 | 0.000202245 | 0.0217921 | 0.0458555 | 0.222223 | 0.340774 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 43 | 0.00337123 | 3496 | 0.00401062 | -0.000639385 | -0.0242907 | 0.0230119 | 0.957743 | 0.989364 | 0.97713 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 20 | max_down | 43 | -0.0688607 | 3486 | -0.0568547 | -0.0120059 | -0.0401726 | 0.0161607 | 0.403467 | 0.682628 | 0.753622 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 20 | max_up | 43 | 0.0759094 | 3486 | 0.0538163 | 0.0220931 | -0.0050512 | 0.0492374 | 0.110652 | 0.35066 | 0.474863 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 43 | 0.0125474 | 3486 | 0.00787952 | 0.00466791 | -0.0352902 | 0.044626 | 0.818894 | 0.989364 | 0.905094 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 5 | max_down | 43 | -0.0266641 | 3501 | -0.0202046 | -0.00645953 | -0.015992 | 0.00307297 | 0.184126 | 0.48333 | 0.610522 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 5 | max_up | 43 | 0.027661 | 3501 | 0.0194839 | 0.00817702 | 0.001801 | 0.014553 | 0.0119495 | 0.210717 | 0.281271 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 43 | 0.00389522 | 3501 | 0.00141525 | 0.00247997 | -0.00998424 | 0.0149442 | 0.696554 | 0.987377 | 0.847979 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 10 | max_down | 43 | -0.0348651 | 3496 | -0.0289922 | -0.00587287 | -0.018741 | 0.00699528 | 0.371043 | 0.649325 | 0.753622 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 10 | max_up | 43 | 0.0358733 | 3496 | 0.0291029 | 0.00677039 | -0.0013863 | 0.0149271 | 0.103762 | 0.35066 | 0.474863 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 43 | 0.00312031 | 3496 | 0.00282238 | 0.000297927 | -0.0154653 | 0.0160612 | 0.97045 | 0.989364 | 0.985198 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 20 | max_down | 43 | -0.0513325 | 3486 | -0.0408276 | -0.0105048 | -0.0307827 | 0.00977304 | 0.309931 | 0.607131 | 0.751254 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 20 | max_up | 43 | 0.0499727 | 3486 | 0.0439206 | 0.00605216 | -0.00532693 | 0.0174312 | 0.297199 | 0.607131 | 0.751254 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 43 | 0.00116285 | 3486 | 0.00564193 | -0.00447907 | -0.0258071 | 0.0168489 | 0.68062 | 0.987377 | 0.844436 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 5 | max_down | 43 | -0.032609 | 3501 | -0.024419 | -0.00818998 | -0.0197439 | 0.0033639 | 0.164727 | 0.466304 | 0.593017 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 5 | max_up | 43 | 0.0304321 | 3501 | 0.0218745 | 0.00855752 | 0.00109131 | 0.0160237 | 0.0246729 | 0.210717 | 0.282617 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 43 | 0.0028187 | 3501 | 0.00196823 | 0.000850464 | -0.0142671 | 0.015968 | 0.912201 | 0.989364 | 0.949895 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 10 | max_down | 43 | -0.0439465 | 3496 | -0.0353623 | -0.00858417 | -0.0256698 | 0.00850143 | 0.324749 | 0.607131 | 0.751254 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 10 | max_up | 43 | 0.0431369 | 3496 | 0.0327764 | 0.0103605 | 0.00070204 | 0.020019 | 0.0355126 | 0.210717 | 0.319709 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 43 | 0.00376231 | 3496 | 0.00390902 | -0.000146711 | -0.0217167 | 0.0214233 | 0.989364 | 0.989364 | 0.989364 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 20 | max_down | 43 | -0.0638505 | 3486 | -0.0503441 | -0.0135064 | -0.0397954 | 0.0127826 | 0.313943 | 0.607131 | 0.751254 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 20 | max_up | 43 | 0.0655907 | 3486 | 0.0497372 | 0.0158534 | -0.00356695 | 0.0352738 | 0.109597 | 0.35066 | 0.474863 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 43 | 0.0072777 | 3486 | 0.0077413 | -0.000463597 | -0.0325093 | 0.0315821 | 0.977379 | 0.989364 | 0.985198 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 5 | max_down | 43 | -0.0344213 | 3501 | -0.0276359 | -0.00678537 | -0.0192225 | 0.00565172 | 0.284922 | 0.607131 | 0.751254 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 5 | max_up | 43 | 0.0344368 | 3501 | 0.0260893 | 0.00834753 | 0.000632058 | 0.016063 | 0.0339584 | 0.210717 | 0.319709 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 43 | 0.0068038 | 3501 | 0.00438993 | 0.00241388 | -0.0140592 | 0.0188869 | 0.773953 | 0.989364 | 0.878541 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 10 | max_down | 43 | -0.0469397 | 3496 | -0.0403063 | -0.00663347 | -0.0251758 | 0.0119088 | 0.483187 | 0.742458 | 0.753622 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 10 | max_up | 43 | 0.0526117 | 3496 | 0.0401133 | 0.0124984 | 0.000308781 | 0.0246879 | 0.0444683 | 0.222223 | 0.340774 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 43 | 0.0109705 | 3496 | 0.00875183 | 0.00221862 | -0.0220302 | 0.0264675 | 0.85768 | 0.989364 | 0.919725 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 20 | max_down | 43 | -0.0667932 | 3486 | -0.0579105 | -0.0088827 | -0.03699 | 0.0192246 | 0.535643 | 0.803465 | 0.789369 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 20 | max_up | 43 | 0.0806233 | 3486 | 0.0616785 | 0.0189448 | -0.00437504 | 0.0422647 | 0.111321 | 0.35066 | 0.474863 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 43 | 0.0205038 | 3486 | 0.0171807 | 0.00332305 | -0.0328036 | 0.0394497 | 0.856927 | 0.989364 | 0.919725 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 5 | max_down | 43 | -0.026812 | 3501 | -0.0188409 | -0.00797115 | -0.0179674 | 0.00202508 | 0.118068 | 0.354204 | 0.487757 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 5 | max_up | 43 | 0.0256536 | 3501 | 0.017042 | 0.00861157 | 0.00237537 | 0.0148478 | 0.00679831 | 0.210717 | 0.281271 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 43 | 0.00329398 | 3501 | 0.00120088 | 0.0020931 | -0.0105345 | 0.0147207 | 0.74527 | 0.989364 | 0.865475 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 10 | max_down | 43 | -0.0347524 | 3496 | -0.0270617 | -0.00769073 | -0.0214281 | 0.00604666 | 0.272517 | 0.607131 | 0.751254 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 10 | max_up | 43 | 0.0342317 | 3496 | 0.0255522 | 0.00867952 | 0.000613373 | 0.0167457 | 0.0349408 | 0.210717 | 0.319709 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 43 | 0.00223603 | 3496 | 0.00239283 | -0.000156805 | -0.015976 | 0.0156624 | 0.9845 | 0.989364 | 0.988422 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 20 | max_down | 43 | -0.050869 | 3486 | -0.0383486 | -0.0125204 | -0.033271 | 0.00823019 | 0.236961 | 0.571934 | 0.705362 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 20 | max_up | 43 | 0.0483169 | 3486 | 0.0385965 | 0.00972041 | -0.0013981 | 0.0208389 | 0.0866134 | 0.35066 | 0.437685 | true |
| multi_period_ma_breadth_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 43 | 0.000769837 | 3486 | 0.00471658 | -0.00394674 | -0.0249017 | 0.0170082 | 0.712012 | 0.987377 | 0.853472 | true |
| multi_period_ma_breadth_top | top | onset | 全A | 5 | max_down | 43 | -0.0252242 | 3499 | -0.0219902 | -0.00323398 | -0.0112964 | 0.00482844 | 0.431756 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 全A | 5 | max_up | 43 | 0.0310315 | 3499 | 0.0194837 | 0.0115478 | 0.00449344 | 0.0186022 | 0.00133445 | 0.0613449 | 0.245379 | true |
| multi_period_ma_breadth_top | top | onset | 全A | 5 | terminal_return | 43 | 0.0114803 | 3499 | 0.00158705 | 0.00989326 | -0.001446 | 0.0212325 | 0.0872552 | 0.24263 | 0.437685 | true |
| multi_period_ma_breadth_top | top | onset | 全A | 10 | max_down | 43 | -0.0384823 | 3494 | -0.0316808 | -0.00680148 | -0.0209262 | 0.00732328 | 0.345273 | 0.555953 | 0.751254 | true |
| multi_period_ma_breadth_top | top | onset | 全A | 10 | max_up | 43 | 0.0424834 | 3494 | 0.0291483 | 0.0133351 | 0.003695 | 0.0229752 | 0.00670289 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 全A | 10 | terminal_return | 43 | 0.0118801 | 3494 | 0.00327051 | 0.00860962 | -0.00975906 | 0.0269783 | 0.358265 | 0.555953 | 0.752556 | true |
| multi_period_ma_breadth_top | top | onset | 全A | 20 | max_down | 43 | -0.0551643 | 3484 | -0.0450905 | -0.0100738 | -0.0310056 | 0.010858 | 0.345537 | 0.555953 | 0.751254 | true |
| multi_period_ma_breadth_top | top | onset | 全A | 20 | max_up | 43 | 0.061065 | 3484 | 0.0441447 | 0.0169203 | 0.000921075 | 0.0329195 | 0.0381874 | 0.150363 | 0.320774 | true |
| multi_period_ma_breadth_top | top | onset | 全A | 20 | terminal_return | 43 | 0.00945092 | 3484 | 0.00661721 | 0.00283372 | -0.0268774 | 0.0325448 | 0.851711 | 0.884708 | 0.919725 | true |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 5 | max_down | 43 | -0.0312353 | 3501 | -0.0266833 | -0.00455199 | -0.0139648 | 0.00486084 | 0.343208 | 0.555953 | 0.751254 | true |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 5 | max_up | 43 | 0.0341744 | 3501 | 0.0233421 | 0.0108323 | 0.00210754 | 0.0195571 | 0.0149556 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 5 | terminal_return | 43 | 0.0126007 | 3501 | 0.00253156 | 0.0100691 | -0.00387773 | 0.024016 | 0.157054 | 0.380554 | 0.584555 | true |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 10 | max_down | 43 | -0.046351 | 3496 | -0.0389035 | -0.00744747 | -0.0241617 | 0.00926675 | 0.382483 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 10 | max_up | 43 | 0.0508946 | 3496 | 0.0357553 | 0.0151393 | 0.00223914 | 0.0280394 | 0.0214365 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 10 | terminal_return | 43 | 0.0149585 | 3496 | 0.00517536 | 0.00978316 | -0.0128707 | 0.032437 | 0.397311 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 20 | max_down | 43 | -0.0643899 | 3486 | -0.0559034 | -0.0084865 | -0.0325409 | 0.0155679 | 0.489253 | 0.570795 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 20 | max_up | 43 | 0.0804227 | 3486 | 0.0547507 | 0.0256719 | -0.000697339 | 0.0520412 | 0.0563696 | 0.18691 | 0.340774 | true |
| multi_period_ma_breadth_top | top | onset | 国证2000 | 20 | terminal_return | 43 | 0.0218262 | 3486 | 0.0104199 | 0.0114063 | -0.0298524 | 0.052665 | 0.587916 | 0.673431 | 0.813491 | true |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 5 | max_down | 43 | -0.031228 | 3501 | -0.0271034 | -0.00412461 | -0.0137182 | 0.00546902 | 0.399415 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 5 | max_up | 43 | 0.0341487 | 3501 | 0.0234319 | 0.0107168 | 0.00210918 | 0.0193243 | 0.0146761 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 5 | terminal_return | 43 | 0.0119073 | 3501 | 0.00189639 | 0.010011 | -0.00367577 | 0.0236977 | 0.151682 | 0.380554 | 0.579151 | true |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 10 | max_down | 43 | -0.0470214 | 3496 | -0.0395591 | -0.00746234 | -0.0244957 | 0.00957099 | 0.390517 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 10 | max_up | 43 | 0.0500434 | 3496 | 0.0354353 | 0.0146081 | 0.00201174 | 0.0272044 | 0.0230244 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 10 | terminal_return | 43 | 0.0121675 | 3496 | 0.00390242 | 0.0082651 | -0.0144794 | 0.0310096 | 0.476315 | 0.566186 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 20 | max_down | 43 | -0.0662672 | 3486 | -0.0568867 | -0.00938053 | -0.0340607 | 0.0152997 | 0.456294 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 20 | max_up | 43 | 0.0788227 | 3486 | 0.0537804 | 0.0250423 | -0.00189731 | 0.051982 | 0.068461 | 0.205383 | 0.383381 | true |
| multi_period_ma_breadth_top | top | onset | 中证1000 | 20 | terminal_return | 43 | 0.0168604 | 3486 | 0.00782632 | 0.00903407 | -0.0332316 | 0.0512998 | 0.67526 | 0.72104 | 0.842404 | true |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 5 | max_down | 43 | -0.0220537 | 3501 | -0.0202612 | -0.0017925 | -0.00887782 | 0.00529282 | 0.619996 | 0.697495 | 0.835503 | true |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 5 | max_up | 43 | 0.0303573 | 3501 | 0.0194508 | 0.0109065 | 0.00394236 | 0.0178706 | 0.00214385 | 0.0613449 | 0.245379 | true |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 5 | terminal_return | 43 | 0.0111287 | 3501 | 0.0013264 | 0.00980233 | -1.36417e-05 | 0.0196183 | 0.050315 | 0.176103 | 0.340774 | true |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 10 | max_down | 43 | -0.0335433 | 3496 | -0.0290085 | -0.0045348 | -0.0165347 | 0.00746511 | 0.458882 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 10 | max_up | 43 | 0.0395565 | 3496 | 0.0290576 | 0.0104989 | 0.00158168 | 0.0194161 | 0.0210185 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 10 | terminal_return | 43 | 0.0106266 | 3496 | 0.00273006 | 0.00789654 | -0.00755968 | 0.0233528 | 0.316654 | 0.555953 | 0.751254 | true |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 20 | max_down | 43 | -0.0494119 | 3486 | -0.0408513 | -0.00856059 | -0.0268283 | 0.00970708 | 0.35836 | 0.555953 | 0.752556 | true |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 20 | max_up | 43 | 0.0545803 | 3486 | 0.0438637 | 0.0107166 | -0.00161768 | 0.0230509 | 0.0885792 | 0.24263 | 0.437685 | true |
| multi_period_ma_breadth_top | top | onset | 沪深300 | 20 | terminal_return | 43 | 0.00429277 | 3486 | 0.00560332 | -0.00131055 | -0.0245342 | 0.0219131 | 0.911929 | 0.911929 | 0.949895 | true |
| multi_period_ma_breadth_top | top | onset | 中证500 | 5 | max_down | 43 | -0.0286523 | 3501 | -0.0244676 | -0.0041847 | -0.0135128 | 0.00514336 | 0.379247 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 中证500 | 5 | max_up | 43 | 0.0323281 | 3501 | 0.0218513 | 0.0104768 | 0.00263142 | 0.0183222 | 0.00886017 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 中证500 | 5 | terminal_return | 43 | 0.0107561 | 3501 | 0.00187074 | 0.00888537 | -0.00393079 | 0.0217015 | 0.174192 | 0.406448 | 0.595199 | true |
| multi_period_ma_breadth_top | top | onset | 中证500 | 10 | max_down | 43 | -0.0438514 | 3496 | -0.0353634 | -0.00848796 | -0.0243 | 0.00732404 | 0.292737 | 0.555953 | 0.751254 | true |
| multi_period_ma_breadth_top | top | onset | 中证500 | 10 | max_up | 43 | 0.0462773 | 3496 | 0.0327378 | 0.0135395 | 0.0025483 | 0.0245308 | 0.0157602 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 中证500 | 10 | terminal_return | 43 | 0.0116865 | 3496 | 0.00381155 | 0.00787493 | -0.0128289 | 0.0285788 | 0.455965 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 中证500 | 20 | max_down | 43 | -0.0619541 | 3486 | -0.0503674 | -0.0115866 | -0.034385 | 0.0112117 | 0.319193 | 0.555953 | 0.751254 | true |
| multi_period_ma_breadth_top | top | onset | 中证500 | 20 | max_up | 43 | 0.0681719 | 3486 | 0.0497054 | 0.0184665 | -0.00137781 | 0.0383108 | 0.0681648 | 0.205383 | 0.383381 | true |
| multi_period_ma_breadth_top | top | onset | 中证500 | 20 | terminal_return | 43 | 0.0108937 | 3486 | 0.0076967 | 0.00319698 | -0.0314844 | 0.0378783 | 0.856622 | 0.884708 | 0.919725 | true |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 5 | max_down | 43 | -0.0319942 | 3501 | -0.0276657 | -0.00432849 | -0.0147577 | 0.00610073 | 0.41595 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 5 | max_up | 43 | 0.0358554 | 3501 | 0.0260719 | 0.00978352 | 0.00159943 | 0.0179676 | 0.0191271 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 5 | terminal_return | 43 | 0.013315 | 3501 | 0.00430995 | 0.00900502 | -0.0060598 | 0.0240698 | 0.24136 | 0.521433 | 0.705362 | true |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 10 | max_down | 43 | -0.0472934 | 3496 | -0.0403019 | -0.00699145 | -0.0247212 | 0.0107383 | 0.439585 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 10 | max_up | 43 | 0.0566058 | 3496 | 0.0400642 | 0.0165416 | 0.00238448 | 0.0306988 | 0.0220143 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 10 | terminal_return | 43 | 0.0200962 | 3496 | 0.00863958 | 0.0114567 | -0.0128173 | 0.0357307 | 0.354932 | 0.555953 | 0.752556 | true |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 20 | max_down | 43 | -0.0639206 | 3486 | -0.0579459 | -0.00597473 | -0.0314245 | 0.0194751 | 0.645416 | 0.713354 | 0.842404 | true |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 20 | max_up | 43 | 0.0858114 | 3486 | 0.0616145 | 0.0241969 | -4.85783e-07 | 0.0483943 | 0.0500004 | 0.176103 | 0.340774 | true |
| multi_period_ma_breadth_top | top | onset | 微盘股 | 20 | terminal_return | 43 | 0.0257464 | 3486 | 0.017116 | 0.00863032 | -0.030601 | 0.0478617 | 0.666344 | 0.72104 | 0.842404 | true |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 5 | max_down | 43 | -0.0229909 | 3501 | -0.0188878 | -0.00410305 | -0.0116275 | 0.00342141 | 0.28517 | 0.555953 | 0.751254 | true |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 5 | max_up | 43 | 0.0266041 | 3501 | 0.0170304 | 0.00957372 | 0.00326826 | 0.0158792 | 0.00292118 | 0.0613449 | 0.245379 | true |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 5 | terminal_return | 43 | 0.00881711 | 3501 | 0.00113305 | 0.00768406 | -0.00221175 | 0.0175799 | 0.128026 | 0.336068 | 0.520363 | true |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 10 | max_down | 43 | -0.0343898 | 3496 | -0.0270661 | -0.00732364 | -0.0197572 | 0.0051099 | 0.248301 | 0.521433 | 0.705362 | true |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 10 | max_up | 43 | 0.0364292 | 3496 | 0.0255251 | 0.010904 | 0.00183055 | 0.0199775 | 0.0185016 | 0.10361 | 0.281271 | true |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 10 | terminal_return | 43 | 0.00826797 | 3496 | 0.00231864 | 0.00594933 | -0.00971507 | 0.0216137 | 0.45663 | 0.555953 | 0.753622 | true |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 20 | max_down | 43 | -0.0495612 | 3486 | -0.0383647 | -0.0111965 | -0.0297328 | 0.00733988 | 0.236455 | 0.521433 | 0.705362 | true |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 20 | max_up | 43 | 0.0514281 | 3486 | 0.0385581 | 0.01287 | 0.000846582 | 0.0248933 | 0.0359046 | 0.150363 | 0.319709 | true |
| multi_period_ma_breadth_top | top | onset | 上证指数 | 20 | terminal_return | 43 | 0.00300279 | 3486 | 0.00468904 | -0.00168625 | -0.0241331 | 0.0207606 | 0.882943 | 0.897184 | 0.938826 | true |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `multi_period_ma_breadth_bottom/bottom/multi_period_ma_breadth_v1_20120104_20260814/capped_confirmation`：数据可用性——10日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；20日：事件日缺失 0、窗口不完整 14（涉及 7 个指数）。 1 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `multi_period_ma_breadth_bottom/bottom/multi_period_ma_breadth_v1_20120104_20260814/onset`：数据可用性——20日：事件日缺失 0、窗口不完整 14（涉及 7 个指数）。 5 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `multi_period_ma_breadth_top/top/multi_period_ma_breadth_v1_20120104_20260814/capped_confirmation`：13 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `multi_period_ma_breadth_top/top/multi_period_ma_breadth_v1_20120104_20260814/onset`：16 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
