# 信号后 OHLC 结果评测

- 评测版本：`stage_d_v1`
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
| 上证指数 | capped_confirmation | 5 | 313 | 313 | 313 |
| 上证指数 | capped_confirmation | 10 | 313 | 313 | 313 |
| 上证指数 | capped_confirmation | 20 | 313 | 313 | 311 |
| 上证指数 | onset | 5 | 313 | 313 | 313 |
| 上证指数 | onset | 10 | 313 | 313 | 313 |
| 上证指数 | onset | 20 | 313 | 313 | 311 |
| 中证1000 | capped_confirmation | 5 | 313 | 313 | 313 |
| 中证1000 | capped_confirmation | 10 | 313 | 313 | 313 |
| 中证1000 | capped_confirmation | 20 | 313 | 313 | 311 |
| 中证1000 | onset | 5 | 313 | 313 | 313 |
| 中证1000 | onset | 10 | 313 | 313 | 313 |
| 中证1000 | onset | 20 | 313 | 313 | 311 |
| 中证500 | capped_confirmation | 5 | 313 | 313 | 313 |
| 中证500 | capped_confirmation | 10 | 313 | 313 | 313 |
| 中证500 | capped_confirmation | 20 | 313 | 313 | 311 |
| 中证500 | onset | 5 | 313 | 313 | 313 |
| 中证500 | onset | 10 | 313 | 313 | 313 |
| 中证500 | onset | 20 | 313 | 313 | 311 |
| 全A | capped_confirmation | 5 | 313 | 313 | 313 |
| 全A | capped_confirmation | 10 | 313 | 313 | 313 |
| 全A | capped_confirmation | 20 | 313 | 313 | 311 |
| 全A | onset | 5 | 313 | 313 | 313 |
| 全A | onset | 10 | 313 | 313 | 313 |
| 全A | onset | 20 | 313 | 313 | 311 |
| 国证2000 | capped_confirmation | 5 | 313 | 313 | 313 |
| 国证2000 | capped_confirmation | 10 | 313 | 313 | 313 |
| 国证2000 | capped_confirmation | 20 | 313 | 313 | 311 |
| 国证2000 | onset | 5 | 313 | 313 | 313 |
| 国证2000 | onset | 10 | 313 | 313 | 313 |
| 国证2000 | onset | 20 | 313 | 313 | 311 |
| 微盘股 | capped_confirmation | 5 | 313 | 313 | 313 |
| 微盘股 | capped_confirmation | 10 | 313 | 313 | 313 |
| 微盘股 | capped_confirmation | 20 | 313 | 313 | 311 |
| 微盘股 | onset | 5 | 313 | 313 | 313 |
| 微盘股 | onset | 10 | 313 | 313 | 313 |
| 微盘股 | onset | 20 | 313 | 313 | 311 |
| 沪深300 | capped_confirmation | 5 | 313 | 313 | 313 |
| 沪深300 | capped_confirmation | 10 | 313 | 313 | 313 |
| 沪深300 | capped_confirmation | 20 | 313 | 313 | 311 |
| 沪深300 | onset | 5 | 313 | 313 | 313 |
| 沪深300 | onset | 10 | 313 | 313 | 313 |
| 沪深300 | onset | 20 | 313 | 313 | 311 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_down | 43 | -0.0240746 | 3499 | -0.0220043 | -0.00207023 | -0.00839045 | 0.00424998 | 0.520865 | 0.958884 | 0.873541 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_up | 43 | 0.0217 | 3499 | 0.0195984 | 0.00210156 | -0.00408175 | 0.00828487 | 0.50531 | 0.958884 | 0.873541 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | terminal_return | 43 | 0.000457541 | 3499 | 0.00172251 | -0.00126497 | -0.0103593 | 0.00782936 | 0.785141 | 0.958884 | 0.93136 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_down | 43 | -0.0335366 | 3494 | -0.0317417 | -0.00179488 | -0.0092721 | 0.00568233 | 0.638004 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_up | 43 | 0.0296515 | 3494 | 0.0293062 | 0.000345299 | -0.00744012 | 0.00813072 | 0.930727 | 0.958884 | 0.980349 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | terminal_return | 43 | 0.00447806 | 3494 | 0.0033616 | 0.00111645 | -0.00977658 | 0.0120095 | 0.840788 | 0.958884 | 0.957283 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_down | 42 | -0.0476622 | 3485 | -0.0451838 | -0.00247834 | -0.0146593 | 0.00970262 | 0.690054 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_up | 42 | 0.0538316 | 3485 | 0.0442367 | 0.00959483 | -0.00766369 | 0.0268534 | 0.275864 | 0.958884 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | terminal_return | 42 | 0.0208072 | 3485 | 0.00648116 | 0.0143261 | -0.0081899 | 0.0368421 | 0.21237 | 0.958884 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_down | 43 | -0.0301601 | 3501 | -0.0266965 | -0.00346365 | -0.0141145 | 0.00718723 | 0.523871 | 0.958884 | 0.876065 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_up | 43 | 0.0274831 | 3501 | 0.0234243 | 0.00405884 | -0.00549915 | 0.0136168 | 0.405227 | 0.958884 | 0.834968 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 43 | 0.000348255 | 3501 | 0.00268205 | -0.00233379 | -0.0163776 | 0.01171 | 0.744643 | 0.958884 | 0.913005 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_down | 43 | -0.0432688 | 3496 | -0.0389414 | -0.00432736 | -0.015819 | 0.00716425 | 0.460471 | 0.958884 | 0.855311 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_up | 43 | 0.0374229 | 3496 | 0.035921 | 0.00150191 | -0.0100052 | 0.013009 | 0.79809 | 0.958884 | 0.939807 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 43 | 0.00678816 | 3496 | 0.00527586 | 0.0015123 | -0.0152118 | 0.0182364 | 0.859323 | 0.958884 | 0.961777 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_down | 42 | -0.0627779 | 3487 | -0.0559252 | -0.00685263 | -0.0250963 | 0.011391 | 0.461602 | 0.958884 | 0.855311 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_up | 42 | 0.0646324 | 3487 | 0.0549483 | 0.00968414 | -0.0124381 | 0.0318064 | 0.390891 | 0.958884 | 0.829314 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 42 | 0.0240339 | 3487 | 0.0103965 | 0.0136374 | -0.0156838 | 0.0429586 | 0.361977 | 0.958884 | 0.811114 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_down | 43 | -0.0295229 | 3501 | -0.0271244 | -0.00239848 | -0.0123047 | 0.00750774 | 0.635106 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_up | 43 | 0.0267909 | 3501 | 0.0235223 | 0.0032686 | -0.00543822 | 0.0119754 | 0.461854 | 0.958884 | 0.855311 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 43 | 0.000792721 | 3501 | 0.0020329 | -0.00124018 | -0.0143421 | 0.0118617 | 0.852816 | 0.958884 | 0.961777 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_down | 43 | -0.0429428 | 3496 | -0.0396092 | -0.00333351 | -0.0143997 | 0.0077327 | 0.554911 | 0.958884 | 0.888798 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_up | 43 | 0.0374898 | 3496 | 0.0355898 | 0.0019 | -0.00877473 | 0.0125747 | 0.727194 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 43 | 0.0062872 | 3496 | 0.00397475 | 0.00231245 | -0.0134275 | 0.0180524 | 0.773382 | 0.958884 | 0.922203 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_down | 42 | -0.0636827 | 3487 | -0.0569205 | -0.00676223 | -0.0234997 | 0.00997527 | 0.428435 | 0.958884 | 0.839111 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_up | 42 | 0.0641191 | 3487 | 0.0539647 | 0.0101544 | -0.0110634 | 0.0313723 | 0.348235 | 0.958884 | 0.807564 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 42 | 0.0217575 | 3487 | 0.00776992 | 0.0139876 | -0.0145451 | 0.0425202 | 0.336627 | 0.958884 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_down | 43 | -0.0218045 | 3501 | -0.0202642 | -0.00154025 | -0.00721915 | 0.00413864 | 0.595003 | 0.958884 | 0.896134 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_up | 43 | 0.022468 | 3501 | 0.0195477 | 0.00292027 | -0.00358123 | 0.00942176 | 0.378659 | 0.958884 | 0.824975 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 43 | 0.00112361 | 3501 | 0.00144929 | -0.000325678 | -0.00908386 | 0.0084325 | 0.941899 | 0.958884 | 0.98489 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_down | 43 | -0.0292479 | 3496 | -0.0290613 | -0.000186562 | -0.0072793 | 0.00690618 | 0.958884 | 0.958884 | 0.990581 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_up | 43 | 0.0306266 | 3496 | 0.0291674 | 0.00145917 | -0.00640396 | 0.00932231 | 0.716067 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 43 | 0.0045178 | 3496 | 0.00280519 | 0.0017126 | -0.00845253 | 0.0118777 | 0.741236 | 0.958884 | 0.912661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_down | 42 | -0.0401239 | 3487 | -0.0409657 | 0.000841776 | -0.0113435 | 0.013027 | 0.892296 | 0.958884 | 0.963679 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_up | 42 | 0.0558452 | 3487 | 0.0438516 | 0.0119936 | -0.00406405 | 0.0280512 | 0.14321 | 0.958884 | 0.762443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 42 | 0.0208615 | 3487 | 0.00540338 | 0.0154581 | -0.00646884 | 0.0373851 | 0.167044 | 0.958884 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_down | 43 | -0.0262739 | 3501 | -0.0244968 | -0.00177704 | -0.00984554 | 0.00629146 | 0.665975 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_up | 43 | 0.0269068 | 3501 | 0.0219178 | 0.00498891 | -0.00277941 | 0.0127572 | 0.208125 | 0.958884 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 43 | 0.00266273 | 3501 | 0.00197015 | 0.000692578 | -0.0108028 | 0.0121879 | 0.905998 | 0.958884 | 0.971504 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_down | 43 | -0.03814 | 3496 | -0.0354337 | -0.00270628 | -0.0123149 | 0.00690238 | 0.580925 | 0.958884 | 0.891848 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_up | 43 | 0.0359 | 3496 | 0.0328654 | 0.00303463 | -0.00667197 | 0.0127412 | 0.540031 | 0.958884 | 0.881341 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 43 | 0.00486265 | 3496 | 0.00389548 | 0.000967166 | -0.0118102 | 0.0137445 | 0.882059 | 0.958884 | 0.963092 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_down | 42 | -0.0541986 | 3487 | -0.0504642 | -0.00373439 | -0.017557 | 0.0100883 | 0.596443 | 0.958884 | 0.896134 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_up | 42 | 0.0626525 | 3487 | 0.0497772 | 0.0128753 | -0.00588391 | 0.0316346 | 0.178548 | 0.958884 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 42 | 0.0250221 | 3487 | 0.00752744 | 0.0174946 | -0.00690104 | 0.0418903 | 0.159856 | 0.958884 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_down | 43 | -0.0379907 | 3501 | -0.0275921 | -0.0103986 | -0.0269976 | 0.00620046 | 0.219501 | 0.958884 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_up | 43 | 0.0380422 | 3501 | 0.026045 | 0.0119971 | -0.00328407 | 0.0272784 | 0.123858 | 0.958884 | 0.757571 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 43 | 0.000149879 | 3501 | 0.00447165 | -0.00432177 | -0.0294062 | 0.0207626 | 0.735599 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_down | 43 | -0.0543781 | 3496 | -0.0402148 | -0.0141633 | -0.0317902 | 0.00346368 | 0.115288 | 0.958884 | 0.738627 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_up | 43 | 0.0499577 | 3496 | 0.040146 | 0.00981176 | -0.0099716 | 0.0295951 | 0.33101 | 0.958884 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 43 | 0.0104474 | 3496 | 0.00875826 | 0.00168914 | -0.0240005 | 0.0273788 | 0.897458 | 0.958884 | 0.966493 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_down | 42 | -0.0779904 | 3487 | -0.0577782 | -0.0202123 | -0.0496178 | 0.00919327 | 0.177906 | 0.958884 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_up | 42 | 0.0835727 | 3487 | 0.0616484 | 0.0219243 | -0.00922584 | 0.0530744 | 0.167741 | 0.958884 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 42 | 0.0319407 | 3487 | 0.0170439 | 0.0148968 | -0.0281109 | 0.0579045 | 0.497205 | 0.958884 | 0.87322 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_down | 43 | -0.0200021 | 3501 | -0.0189245 | -0.00107759 | -0.00611111 | 0.00395594 | 0.674777 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_up | 43 | 0.0183335 | 3501 | 0.017132 | 0.00120156 | -0.00431999 | 0.0067231 | 0.669729 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 43 | 0.000969319 | 3501 | 0.00122944 | -0.000260117 | -0.007533 | 0.00701277 | 0.944114 | 0.958884 | 0.985843 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_down | 43 | -0.0262442 | 3496 | -0.0271663 | 0.000922116 | -0.005022 | 0.00686623 | 0.761085 | 0.958884 | 0.917671 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_up | 43 | 0.0250577 | 3496 | 0.025665 | -0.000607296 | -0.00737401 | 0.00615942 | 0.860369 | 0.958884 | 0.961777 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 43 | 0.00484672 | 3496 | 0.00236072 | 0.002486 | -0.00615629 | 0.0111283 | 0.572887 | 0.958884 | 0.890044 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_down | 42 | -0.0365722 | 3487 | -0.0385244 | 0.00195216 | -0.00736897 | 0.0112733 | 0.681446 | 0.958884 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_up | 42 | 0.0460352 | 3487 | 0.0386268 | 0.00740839 | -0.00696794 | 0.0217847 | 0.312484 | 0.958884 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 42 | 0.0179672 | 3487 | 0.00450831 | 0.0134589 | -0.00418244 | 0.0311003 | 0.134831 | 0.958884 | 0.760687 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_down | 43 | -0.0248982 | 3499 | -0.0219942 | -0.00290402 | -0.00997586 | 0.00416782 | 0.420897 | 0.814249 | 0.839111 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_up | 43 | 0.0230379 | 3499 | 0.019582 | 0.00345588 | -0.00383184 | 0.0107436 | 0.352659 | 0.814249 | 0.811114 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 5 | terminal_return | 43 | 0.00481292 | 3499 | 0.00166899 | 0.00314393 | -0.00714331 | 0.0134312 | 0.54917 | 0.910467 | 0.888798 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_down | 43 | -0.034279 | 3494 | -0.0317326 | -0.00254642 | -0.0106879 | 0.00559507 | 0.539856 | 0.910467 | 0.881341 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_up | 43 | 0.0300849 | 3494 | 0.0293009 | 0.00078398 | -0.00770044 | 0.0092684 | 0.856282 | 0.982025 | 0.961777 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 10 | terminal_return | 43 | 0.00390397 | 3494 | 0.00336867 | 0.000535299 | -0.00987515 | 0.0109457 | 0.919724 | 0.98996 | 0.977934 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_down | 42 | -0.0481321 | 3485 | -0.0451782 | -0.00295392 | -0.0147088 | 0.00880099 | 0.622342 | 0.964035 | 0.906533 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_up | 42 | 0.0503119 | 3485 | 0.0442792 | 0.00603274 | -0.00768438 | 0.0197499 | 0.388687 | 0.814249 | 0.829314 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 全A | 20 | terminal_return | 42 | 0.0192633 | 3485 | 0.00649977 | 0.0127635 | -0.00634143 | 0.0318684 | 0.190391 | 0.814249 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_down | 43 | -0.0345079 | 3501 | -0.0266431 | -0.0078648 | -0.0198226 | 0.00409302 | 0.197358 | 0.814249 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_up | 43 | 0.0277021 | 3501 | 0.0234216 | 0.0042805 | -0.00558292 | 0.0141439 | 0.394994 | 0.814249 | 0.829314 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | terminal_return | 43 | 0.00239349 | 3501 | 0.00265693 | -0.000263436 | -0.0163313 | 0.0158044 | 0.974365 | 0.994232 | 0.991753 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_down | 43 | -0.0470848 | 3496 | -0.0388945 | -0.00819027 | -0.0206006 | 0.00422006 | 0.195834 | 0.814249 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_up | 43 | 0.0370533 | 3496 | 0.0359256 | 0.00112774 | -0.0120673 | 0.0143227 | 0.866964 | 0.982025 | 0.961777 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | terminal_return | 43 | 0.00487098 | 3496 | 0.00529944 | -0.000428456 | -0.0165977 | 0.0157408 | 0.958579 | 0.994232 | 0.990581 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_down | 42 | -0.0658792 | 3487 | -0.0558879 | -0.00999129 | -0.0286359 | 0.00865333 | 0.293568 | 0.814249 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_up | 42 | 0.0584361 | 3487 | 0.0550229 | 0.00341314 | -0.0161649 | 0.0229912 | 0.732579 | 0.964035 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | terminal_return | 42 | 0.0189396 | 3487 | 0.0104579 | 0.00848168 | -0.0183053 | 0.0352687 | 0.53486 | 0.910467 | 0.879031 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_down | 43 | -0.0336088 | 3501 | -0.0270742 | -0.00653462 | -0.0175553 | 0.00448609 | 0.245169 | 0.814249 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_up | 43 | 0.026997 | 3501 | 0.0235198 | 0.00347725 | -0.0050937 | 0.0120482 | 0.426512 | 0.814249 | 0.839111 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | terminal_return | 43 | 0.0018578 | 3501 | 0.00201982 | -0.000162013 | -0.0147201 | 0.0143961 | 0.982598 | 0.994232 | 0.992059 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_down | 43 | -0.0458914 | 3496 | -0.039573 | -0.00631847 | -0.0184419 | 0.005805 | 0.307014 | 0.814249 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_up | 43 | 0.0361457 | 3496 | 0.0356063 | 0.00053944 | -0.0110174 | 0.0120963 | 0.927106 | 0.98996 | 0.980268 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | terminal_return | 43 | 0.00405611 | 3496 | 0.00400219 | 5.39181e-05 | -0.0145633 | 0.0146712 | 0.994232 | 0.994232 | 0.995548 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_down | 42 | -0.0655699 | 3487 | -0.0568978 | -0.00867206 | -0.0259688 | 0.00862463 | 0.325761 | 0.814249 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_up | 42 | 0.0576592 | 3487 | 0.0540425 | 0.00361674 | -0.0143993 | 0.0216327 | 0.69397 | 0.964035 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | terminal_return | 42 | 0.0170673 | 3487 | 0.00782641 | 0.0092409 | -0.0161102 | 0.034592 | 0.474947 | 0.880048 | 0.8636 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_down | 43 | -0.0214452 | 3501 | -0.0202687 | -0.00117656 | -0.00627235 | 0.00391923 | 0.650879 | 0.964035 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_up | 43 | 0.0247496 | 3501 | 0.0195197 | 0.00522995 | -0.00225819 | 0.0127181 | 0.171023 | 0.814249 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | terminal_return | 43 | 0.00723982 | 3501 | 0.00137417 | 0.00586565 | -0.00334243 | 0.0150737 | 0.211833 | 0.814249 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_down | 43 | -0.0281989 | 3496 | -0.0290742 | 0.000875286 | -0.00559596 | 0.00734653 | 0.790928 | 0.977029 | 0.932827 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_up | 43 | 0.0328681 | 3496 | 0.0291399 | 0.00372819 | -0.00466 | 0.0121164 | 0.38368 | 0.814249 | 0.825782 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | terminal_return | 43 | 0.00478918 | 3496 | 0.00280185 | 0.00198732 | -0.00803417 | 0.0120088 | 0.697513 | 0.964035 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_down | 42 | -0.0391175 | 3487 | -0.0409778 | 0.00186031 | -0.00889096 | 0.0126116 | 0.734503 | 0.964035 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_up | 42 | 0.0536779 | 3487 | 0.0438777 | 0.00980025 | -0.00280979 | 0.0224103 | 0.127691 | 0.814249 | 0.757571 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | terminal_return | 42 | 0.0213989 | 3487 | 0.00539691 | 0.016002 | -0.00225938 | 0.0342633 | 0.085888 | 0.814249 | 0.636582 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_down | 43 | -0.029343 | 3501 | -0.0244591 | -0.00488389 | -0.013689 | 0.00392122 | 0.276973 | 0.814249 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_up | 43 | 0.0256191 | 3501 | 0.0219337 | 0.00368543 | -0.00416272 | 0.0115336 | 0.357363 | 0.814249 | 0.811114 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | terminal_return | 43 | 0.00415188 | 3501 | 0.00195186 | 0.00220002 | -0.0100621 | 0.0144622 | 0.725098 | 0.964035 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_down | 43 | -0.0403884 | 3496 | -0.035406 | -0.00498236 | -0.0156364 | 0.00567168 | 0.359356 | 0.814249 | 0.811114 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_up | 43 | 0.0342978 | 3496 | 0.0328851 | 0.00141268 | -0.00857777 | 0.0114031 | 0.781666 | 0.977029 | 0.930613 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | terminal_return | 43 | 0.00333328 | 3496 | 0.0039143 | -0.00058102 | -0.0126408 | 0.0114787 | 0.924768 | 0.98996 | 0.980268 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_down | 42 | -0.0550968 | 3487 | -0.0504534 | -0.00464347 | -0.0188282 | 0.00954127 | 0.52112 | 0.910467 | 0.873541 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_up | 42 | 0.0572892 | 3487 | 0.0498418 | 0.00744746 | -0.00773512 | 0.02263 | 0.336335 | 0.814249 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | terminal_return | 42 | 0.0222028 | 3487 | 0.0075614 | 0.0146414 | -0.00685944 | 0.0361423 | 0.181975 | 0.814249 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_down | 43 | -0.0420226 | 3501 | -0.0275426 | -0.0144801 | -0.031459 | 0.00249894 | 0.0946169 | 0.814249 | 0.656242 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_up | 43 | 0.0399739 | 3501 | 0.0260213 | 0.0139526 | -0.00248573 | 0.0303909 | 0.0961889 | 0.814249 | 0.66108 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | terminal_return | 43 | 0.00699093 | 3501 | 0.00438763 | 0.00260331 | -0.0232668 | 0.0284735 | 0.843644 | 0.982025 | 0.95909 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_down | 43 | -0.0590984 | 3496 | -0.0401567 | -0.0189417 | -0.0380688 | 0.000185368 | 0.0522577 | 0.814249 | 0.539768 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_up | 43 | 0.0541446 | 3496 | 0.0400945 | 0.0140501 | -0.0105073 | 0.0386075 | 0.262126 | 0.814249 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | terminal_return | 43 | 0.0123444 | 3496 | 0.00873493 | 0.00360944 | -0.0223083 | 0.0295272 | 0.784884 | 0.977029 | 0.93136 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_down | 42 | -0.0844297 | 3487 | -0.0577006 | -0.0267291 | -0.0577575 | 0.00429924 | 0.0913296 | 0.814249 | 0.65137 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_up | 42 | 0.0813397 | 3487 | 0.0616753 | 0.0196645 | -0.0113972 | 0.0507262 | 0.214667 | 0.814249 | 0.795806 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | terminal_return | 42 | 0.0279395 | 3487 | 0.0170921 | 0.0108474 | -0.0314045 | 0.0530992 | 0.614828 | 0.964035 | 0.9043 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_down | 43 | -0.0193533 | 3501 | -0.0189325 | -0.000420774 | -0.0055765 | 0.00473495 | 0.872911 | 0.982025 | 0.961777 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_up | 43 | 0.0198711 | 3501 | 0.0171131 | 0.00275802 | -0.0037183 | 0.00923434 | 0.403893 | 0.814249 | 0.834968 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | terminal_return | 43 | 0.00599444 | 3501 | 0.00116772 | 0.00482672 | -0.00337179 | 0.0130252 | 0.248535 | 0.814249 | 0.806443 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_down | 43 | -0.025867 | 3496 | -0.027171 | 0.00130399 | -0.00462401 | 0.00723198 | 0.666364 | 0.964035 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_up | 43 | 0.0263536 | 3496 | 0.0256491 | 0.000704489 | -0.00665937 | 0.00806834 | 0.851261 | 0.982025 | 0.961777 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | terminal_return | 43 | 0.00615538 | 3496 | 0.00234462 | 0.00381076 | -0.00434121 | 0.0119627 | 0.359546 | 0.814249 | 0.811114 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_down | 42 | -0.0364944 | 3487 | -0.0385253 | 0.0020309 | -0.00664933 | 0.0107111 | 0.646538 | 0.964035 | 0.911661 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_up | 42 | 0.0435904 | 3487 | 0.0386562 | 0.00493415 | -0.00628089 | 0.0161492 | 0.388512 | 0.814249 | 0.829314 | true |
| new_high_low_120_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | terminal_return | 42 | 0.0177852 | 3487 | 0.0045105 | 0.0132747 | -0.0017992 | 0.0283486 | 0.0843372 | 0.814249 | 0.636582 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_down | 48 | -0.0315686 | 3494 | -0.0218984 | -0.0096702 | -0.0182873 | -0.00105313 | 0.0278396 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_up | 48 | 0.0225629 | 3494 | 0.0195836 | 0.00297939 | -0.00338993 | 0.0093487 | 0.359231 | 0.619296 | 0.811114 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 5 | terminal_return | 48 | 0.00183033 | 3494 | 0.00170547 | 0.000124867 | -0.0104849 | 0.0107347 | 0.981597 | 0.995584 | 0.992059 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_down | 48 | -0.0472216 | 3489 | -0.0315508 | -0.0156707 | -0.0311099 | -0.000231584 | 0.0466571 | 0.226108 | 0.528042 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_up | 48 | 0.0358615 | 3489 | 0.0292203 | 0.00664123 | -0.00385103 | 0.0171335 | 0.214749 | 0.483186 | 0.795806 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 10 | terminal_return | 48 | -7.15398e-06 | 3489 | 0.00342171 | -0.00342886 | -0.0244907 | 0.017633 | 0.74966 | 0.913005 | 0.913005 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_down | 48 | -0.0706362 | 3479 | -0.0448626 | -0.0257736 | -0.0588255 | 0.00727825 | 0.126415 | 0.351337 | 0.757571 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_up | 48 | 0.0636664 | 3479 | 0.0440845 | 0.0195819 | -0.00305964 | 0.0422234 | 0.0900494 | 0.308236 | 0.648934 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 全A | 20 | terminal_return | 48 | 0.0059694 | 3479 | 0.00666117 | -0.000691769 | -0.0432114 | 0.0418279 | 0.974561 | 0.995584 | 0.991753 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_down | 48 | -0.0375584 | 3496 | -0.02659 | -0.0109685 | -0.0207374 | -0.00119952 | 0.0277597 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_up | 48 | 0.0269973 | 3496 | 0.0234251 | 0.00357214 | -0.00354542 | 0.0106897 | 0.325273 | 0.602712 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 48 | 0.00206898 | 3496 | 0.00266176 | -0.00059278 | -0.0126066 | 0.0114211 | 0.922957 | 0.995584 | 0.979994 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_down | 48 | -0.055818 | 3491 | -0.0387627 | -0.0170553 | -0.0343196 | 0.000209007 | 0.0528344 | 0.234556 | 0.539768 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_up | 48 | 0.0425592 | 3491 | 0.0358483 | 0.00671091 | -0.00549262 | 0.0189144 | 0.281107 | 0.570348 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 48 | -0.000976014 | 3491 | 0.00538045 | -0.00635646 | -0.0312468 | 0.0185339 | 0.616694 | 0.863372 | 0.904334 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_down | 48 | -0.0802412 | 3481 | -0.0556726 | -0.0245686 | -0.0615004 | 0.0123632 | 0.192277 | 0.448646 | 0.795806 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_up | 48 | 0.0702844 | 3481 | 0.0548537 | 0.0154308 | -0.0107779 | 0.0416395 | 0.248508 | 0.539861 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 48 | 0.00457948 | 3481 | 0.0106413 | -0.0060618 | -0.0548006 | 0.042677 | 0.807408 | 0.913946 | 0.943432 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_down | 48 | -0.0381497 | 3496 | -0.0270025 | -0.0111472 | -0.0208357 | -0.00145873 | 0.0241268 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_up | 48 | 0.0262778 | 3496 | 0.0235247 | 0.0027531 | -0.00434775 | 0.00985394 | 0.447303 | 0.684436 | 0.84965 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 48 | 0.000605662 | 3496 | 0.00203724 | -0.00143158 | -0.0132541 | 0.0103909 | 0.812397 | 0.913946 | 0.945937 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_down | 48 | -0.0570883 | 3491 | -0.03941 | -0.0176783 | -0.0348747 | -0.000481932 | 0.0439121 | 0.226108 | 0.528042 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_up | 48 | 0.0410602 | 3491 | 0.0355379 | 0.00552225 | -0.00664602 | 0.0176905 | 0.373737 | 0.619617 | 0.816605 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 48 | -0.00394837 | 3491 | 0.00411218 | -0.00806055 | -0.0325836 | 0.0164625 | 0.519421 | 0.743717 | 0.873541 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_down | 48 | -0.0829649 | 3481 | -0.056643 | -0.026322 | -0.0638151 | 0.0111712 | 0.168817 | 0.443146 | 0.795806 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_up | 48 | 0.067373 | 3481 | 0.0539023 | 0.0134707 | -0.0114661 | 0.0384075 | 0.289701 | 0.570348 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 48 | -0.000974564 | 3481 | 0.00805927 | -0.00903383 | -0.0568097 | 0.038742 | 0.710926 | 0.913005 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_down | 48 | -0.0301972 | 3496 | -0.0201468 | -0.0100504 | -0.0181302 | -0.0019707 | 0.0147664 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_up | 48 | 0.0243514 | 3496 | 0.0195177 | 0.00483373 | -0.00222919 | 0.0118966 | 0.179795 | 0.448646 | 0.795806 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 48 | 0.00256144 | 3496 | 0.00143001 | 0.00113143 | -0.00984826 | 0.0121111 | 0.839938 | 0.928352 | 0.957283 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_down | 48 | -0.0436474 | 3491 | -0.028863 | -0.0147843 | -0.0287448 | -0.00082385 | 0.0379249 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_up | 48 | 0.0382942 | 3491 | 0.0290599 | 0.0092343 | -0.00266545 | 0.0211341 | 0.128266 | 0.351337 | 0.757571 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 48 | 0.00288472 | 3491 | 0.00282519 | 5.95313e-05 | -0.0210209 | 0.02114 | 0.995584 | 0.995584 | 0.995584 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_down | 48 | -0.0656715 | 3481 | -0.0406148 | -0.0250566 | -0.0542896 | 0.00417635 | 0.0929601 | 0.308236 | 0.656242 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_up | 48 | 0.0708369 | 3481 | 0.0436242 | 0.0272127 | 0.00055814 | 0.0538673 | 0.0453879 | 0.226108 | 0.528042 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 48 | 0.0128867 | 3481 | 0.0054867 | 0.00740004 | -0.0338482 | 0.0486483 | 0.725117 | 0.913005 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_down | 48 | -0.0369738 | 3496 | -0.0243474 | -0.0126264 | -0.0222018 | -0.00305098 | 0.00975182 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_up | 48 | 0.0245861 | 3496 | 0.0219426 | 0.00264356 | -0.00411527 | 0.00940239 | 0.443314 | 0.684436 | 0.84965 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 48 | 0.00026313 | 3496 | 0.0020021 | -0.00173897 | -0.013341 | 0.00986307 | 0.76893 | 0.913946 | 0.918343 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_down | 48 | -0.0548453 | 3491 | -0.0352001 | -0.0196452 | -0.0369242 | -0.00236614 | 0.0258541 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_up | 48 | 0.0379899 | 3491 | 0.0328323 | 0.0051576 | -0.00597173 | 0.0162869 | 0.363714 | 0.619296 | 0.811114 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 48 | -0.00480818 | 3491 | 0.00402707 | -0.00883525 | -0.0320806 | 0.0144101 | 0.45629 | 0.684436 | 0.855311 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_down | 48 | -0.0797261 | 3481 | -0.0501057 | -0.0296204 | -0.0658235 | 0.00658282 | 0.108799 | 0.326398 | 0.715237 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_up | 48 | 0.0632105 | 3481 | 0.0497473 | 0.0134632 | -0.0104179 | 0.0373442 | 0.269173 | 0.565263 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 48 | -0.00140206 | 3481 | 0.00786165 | -0.00926371 | -0.0542743 | 0.0357469 | 0.68666 | 0.913005 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_down | 48 | -0.036939 | 3496 | -0.0275917 | -0.00934736 | -0.0201375 | 0.00144273 | 0.0895209 | 0.308236 | 0.648934 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_up | 48 | 0.0288936 | 3496 | 0.0261535 | 0.00274014 | -0.0043664 | 0.00984668 | 0.449807 | 0.684436 | 0.851158 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 48 | 0.00455252 | 3496 | 0.00441738 | 0.000135135 | -0.0128316 | 0.0131019 | 0.983703 | 0.995584 | 0.992059 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_down | 48 | -0.0523903 | 3491 | -0.0402218 | -0.0121684 | -0.0301957 | 0.0058588 | 0.185834 | 0.448646 | 0.795806 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_up | 48 | 0.0445144 | 3491 | 0.0402067 | 0.00430764 | -0.0074188 | 0.0160341 | 0.471528 | 0.690843 | 0.8636 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 48 | 0.00452115 | 3491 | 0.00883733 | -0.00431618 | -0.0289263 | 0.020294 | 0.731035 | 0.913005 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_down | 48 | -0.0769701 | 3481 | -0.0577574 | -0.0192127 | -0.0565751 | 0.0181498 | 0.313512 | 0.598523 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_up | 48 | 0.0738595 | 3481 | 0.0617445 | 0.0121149 | -0.0131503 | 0.0373802 | 0.347299 | 0.619296 | 0.807564 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 48 | 0.0111169 | 3481 | 0.0173054 | -0.00618846 | -0.0546337 | 0.0422568 | 0.802299 | 0.913946 | 0.943294 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_down | 48 | -0.0286141 | 3496 | -0.0188047 | -0.00980939 | -0.0179848 | -0.00163397 | 0.0186863 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_up | 48 | 0.0235527 | 3496 | 0.0170586 | 0.0064941 | -0.000162231 | 0.0131504 | 0.0558467 | 0.234556 | 0.550255 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 48 | 0.00290592 | 3496 | 0.00120322 | 0.0017027 | -0.00892877 | 0.0123342 | 0.753592 | 0.913005 | 0.913005 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_down | 48 | -0.0420497 | 3491 | -0.0269503 | -0.0150994 | -0.0293725 | -0.000826231 | 0.0381296 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_up | 48 | 0.036313 | 3491 | 0.0255111 | 0.0108019 | -0.00122959 | 0.0228333 | 0.07846 | 0.308236 | 0.636582 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 48 | 0.00251531 | 3491 | 0.00238922 | 0.000126091 | -0.0204712 | 0.0207234 | 0.990427 | 0.995584 | 0.993054 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_down | 48 | -0.0621775 | 3481 | -0.0381747 | -0.0240028 | -0.0529361 | 0.00493061 | 0.103951 | 0.326398 | 0.689362 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_up | 48 | 0.0664202 | 3481 | 0.0383329 | 0.0280872 | 0.00255924 | 0.0536153 | 0.0310451 | 0.226108 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 48 | 0.0119666 | 3481 | 0.00456786 | 0.00739872 | -0.0325292 | 0.0473267 | 0.716462 | 0.913005 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 5 | max_down | 48 | -0.0312332 | 3494 | -0.021903 | -0.00933019 | -0.0174546 | -0.00120582 | 0.0243916 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 5 | max_up | 48 | 0.0234905 | 3494 | 0.0195708 | 0.0039197 | -0.00262374 | 0.0104631 | 0.240357 | 0.522155 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 5 | terminal_return | 48 | -0.000213521 | 3494 | 0.00173354 | -0.00194706 | -0.0126418 | 0.00874771 | 0.721217 | 0.84142 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 10 | max_down | 48 | -0.0472854 | 3489 | -0.03155 | -0.0157354 | -0.0302423 | -0.00122853 | 0.0335046 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 10 | max_up | 48 | 0.0360853 | 3489 | 0.0292172 | 0.00686805 | -0.0043373 | 0.0180734 | 0.229621 | 0.516648 | 0.805223 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 10 | terminal_return | 48 | 0.00207221 | 3489 | 0.0033931 | -0.00132089 | -0.0212958 | 0.018654 | 0.896875 | 0.896875 | 0.966493 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 20 | max_down | 48 | -0.0705244 | 3479 | -0.0448641 | -0.0256603 | -0.0564999 | 0.00517942 | 0.102927 | 0.281931 | 0.688611 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 20 | max_up | 48 | 0.0630246 | 3479 | 0.0440933 | 0.0189313 | -0.00296345 | 0.040826 | 0.0901297 | 0.258099 | 0.648934 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 全A | 20 | terminal_return | 48 | 0.00226906 | 3479 | 0.00671222 | -0.00444316 | -0.0471479 | 0.0382616 | 0.838411 | 0.882333 | 0.957283 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 5 | max_down | 48 | -0.036284 | 3496 | -0.0266075 | -0.00967656 | -0.0196635 | 0.000310364 | 0.0575528 | 0.226576 | 0.550255 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 5 | max_up | 48 | 0.0275384 | 3496 | 0.0234177 | 0.00412067 | -0.00386984 | 0.0121112 | 0.31213 | 0.589932 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 5 | terminal_return | 48 | -2.99499e-05 | 3496 | 0.00269058 | -0.00272053 | -0.015464 | 0.0100229 | 0.675633 | 0.835945 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 10 | max_down | 48 | -0.0555835 | 3491 | -0.0387659 | -0.0168176 | -0.0333479 | -0.000287257 | 0.0461455 | 0.221961 | 0.528042 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 10 | max_up | 48 | 0.0427233 | 3491 | 0.035846 | 0.00687725 | -0.00689586 | 0.0206504 | 0.32774 | 0.589932 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 10 | terminal_return | 48 | -0.000144103 | 3491 | 0.00536901 | -0.00551311 | -0.0302887 | 0.0192625 | 0.662733 | 0.835945 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 20 | max_down | 48 | -0.0800873 | 3481 | -0.0556747 | -0.0244126 | -0.0584794 | 0.00965423 | 0.160154 | 0.388066 | 0.795806 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 20 | max_up | 48 | 0.0696814 | 3481 | 0.054862 | 0.0148194 | -0.0118694 | 0.0415082 | 0.276453 | 0.551194 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 国证2000 | 20 | terminal_return | 48 | 0.0010848 | 3481 | 0.0106895 | -0.00960468 | -0.0588628 | 0.0396534 | 0.702332 | 0.835945 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 5 | max_down | 48 | -0.0369977 | 3496 | -0.0270183 | -0.00997937 | -0.0198175 | -0.000141195 | 0.0467974 | 0.221961 | 0.528042 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 5 | max_up | 48 | 0.0272907 | 3496 | 0.0235108 | 0.00377989 | -0.00429043 | 0.0118502 | 0.358617 | 0.61908 | 0.811114 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 5 | terminal_return | 48 | -0.00109323 | 3496 | 0.00206057 | -0.0031538 | -0.0158206 | 0.00951305 | 0.625549 | 0.821033 | 0.907706 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 10 | max_down | 48 | -0.056893 | 3491 | -0.0394127 | -0.0174803 | -0.0340451 | -0.000915523 | 0.0386092 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 10 | max_up | 48 | 0.0412751 | 3491 | 0.035535 | 0.00574008 | -0.007997 | 0.0194772 | 0.412791 | 0.654782 | 0.839111 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 10 | terminal_return | 48 | -0.00267284 | 3491 | 0.00409464 | -0.00676748 | -0.0311319 | 0.0175969 | 0.586159 | 0.809056 | 0.893371 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 20 | max_down | 48 | -0.0828793 | 3481 | -0.0566442 | -0.0262351 | -0.0608968 | 0.00842657 | 0.13794 | 0.347609 | 0.762443 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 20 | max_up | 48 | 0.0668322 | 3481 | 0.0539098 | 0.0129225 | -0.0125868 | 0.0384318 | 0.320761 | 0.589932 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证1000 | 20 | terminal_return | 48 | -0.00475895 | 3481 | 0.00811145 | -0.0128704 | -0.0614513 | 0.0357105 | 0.603581 | 0.809056 | 0.898243 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 5 | max_down | 48 | -0.0293015 | 3496 | -0.0201591 | -0.00914235 | -0.0169151 | -0.00136959 | 0.0211465 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 5 | max_up | 48 | 0.0257697 | 3496 | 0.0194982 | 0.00627149 | -0.00029311 | 0.0128361 | 0.0611394 | 0.226576 | 0.550255 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 5 | terminal_return | 48 | 0.000458256 | 3496 | 0.00145889 | -0.00100063 | -0.0116782 | 0.00967697 | 0.854266 | 0.882333 | 0.961777 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 10 | max_down | 48 | -0.0433619 | 3491 | -0.028867 | -0.0144949 | -0.028009 | -0.00098085 | 0.0355308 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 10 | max_up | 48 | 0.0393258 | 3491 | 0.0290457 | 0.0102801 | -0.00125849 | 0.0218187 | 0.0807714 | 0.256923 | 0.636582 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 10 | terminal_return | 48 | 0.0057807 | 3491 | 0.00278537 | 0.00299532 | -0.0166464 | 0.022637 | 0.765019 | 0.860647 | 0.918343 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 20 | max_down | 48 | -0.0653086 | 3481 | -0.0406198 | -0.0246888 | -0.0524771 | 0.00309948 | 0.0816164 | 0.256923 | 0.636582 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 20 | max_up | 48 | 0.0700769 | 3481 | 0.0436347 | 0.0264423 | 0.00241735 | 0.0504672 | 0.0309891 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 沪深300 | 20 | terminal_return | 48 | 0.00906996 | 3481 | 0.00553933 | 0.00353063 | -0.0382108 | 0.0452721 | 0.868327 | 0.882333 | 0.961777 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 5 | max_down | 48 | -0.036239 | 3496 | -0.0243575 | -0.0118815 | -0.021023 | -0.00273997 | 0.0108508 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 5 | max_up | 48 | 0.0254736 | 3496 | 0.0219304 | 0.00354321 | -0.00410048 | 0.0111869 | 0.363586 | 0.61908 | 0.811114 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 5 | terminal_return | 48 | -0.00212276 | 3496 | 0.00203486 | -0.00415762 | -0.0162576 | 0.00794239 | 0.500652 | 0.769294 | 0.873541 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 10 | max_down | 48 | -0.0544929 | 3491 | -0.035205 | -0.019288 | -0.0358077 | -0.00276826 | 0.0221117 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 10 | max_up | 48 | 0.0380652 | 3491 | 0.0328313 | 0.00523393 | -0.00737108 | 0.0178389 | 0.415734 | 0.654782 | 0.839111 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 10 | terminal_return | 48 | -0.00266148 | 3491 | 0.00399755 | -0.00665904 | -0.029412 | 0.0160939 | 0.56622 | 0.809056 | 0.889647 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 20 | max_down | 48 | -0.0796655 | 3481 | -0.0501066 | -0.0295589 | -0.0632648 | 0.00414698 | 0.0856411 | 0.256923 | 0.636582 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 20 | max_up | 48 | 0.063246 | 3481 | 0.0497468 | 0.0134992 | -0.0109907 | 0.0379891 | 0.279971 | 0.551194 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 中证500 | 20 | terminal_return | 48 | -0.0058616 | 3481 | 0.00792314 | -0.0137847 | -0.0594164 | 0.031847 | 0.55379 | 0.809056 | 0.888798 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 5 | max_down | 48 | -0.0359241 | 3496 | -0.0276056 | -0.00831854 | -0.0190864 | 0.00244932 | 0.129983 | 0.341206 | 0.760687 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 5 | max_up | 48 | 0.0283265 | 3496 | 0.0261613 | 0.00216521 | -0.0060038 | 0.0103342 | 0.603412 | 0.809056 | 0.898243 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 5 | terminal_return | 48 | 0.0015028 | 3496 | 0.00445926 | -0.00295645 | -0.0168759 | 0.010963 | 0.677191 | 0.835945 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 10 | max_down | 48 | -0.0522535 | 3491 | -0.0402237 | -0.0120298 | -0.0293685 | 0.00530898 | 0.173873 | 0.405704 | 0.795806 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 10 | max_up | 48 | 0.0438535 | 3491 | 0.0402158 | 0.00363766 | -0.00971306 | 0.0169884 | 0.593314 | 0.809056 | 0.896134 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 10 | terminal_return | 48 | 0.0048952 | 3491 | 0.00883218 | -0.00393698 | -0.0283861 | 0.0205121 | 0.752296 | 0.860647 | 0.913005 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 20 | max_down | 48 | -0.0769861 | 3481 | -0.0577572 | -0.0192289 | -0.0540052 | 0.0155473 | 0.278477 | 0.551194 | 0.806443 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 20 | max_up | 48 | 0.0729567 | 3481 | 0.061757 | 0.0111997 | -0.0146654 | 0.0370648 | 0.396054 | 0.654782 | 0.829314 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 微盘股 | 20 | terminal_return | 48 | 0.00769314 | 3481 | 0.0173526 | -0.00965945 | -0.0593603 | 0.0400414 | 0.703255 | 0.835945 | 0.911661 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 5 | max_down | 48 | -0.0282358 | 3496 | -0.0188099 | -0.00942585 | -0.0172566 | -0.00159505 | 0.0183128 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 5 | max_up | 48 | 0.0234544 | 3496 | 0.0170599 | 0.00639446 | 1.87816e-05 | 0.0127701 | 0.0493247 | 0.221961 | 0.539768 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 5 | terminal_return | 48 | 0.000304976 | 3496 | 0.00123893 | -0.000933953 | -0.0110862 | 0.00921827 | 0.856909 | 0.882333 | 0.961777 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 10 | max_down | 48 | -0.0418992 | 3491 | -0.0269524 | -0.0149468 | -0.0283722 | -0.00152137 | 0.0291016 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 10 | max_up | 48 | 0.0367188 | 3491 | 0.0255055 | 0.0112132 | -0.000266953 | 0.0226934 | 0.0555661 | 0.226576 | 0.550255 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 10 | terminal_return | 48 | 0.0049859 | 3491 | 0.00235525 | 0.00263065 | -0.0164263 | 0.0216877 | 0.786729 | 0.869542 | 0.93136 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 20 | max_down | 48 | -0.0622006 | 3481 | -0.0381744 | -0.0240262 | -0.0513311 | 0.0032787 | 0.0845908 | 0.256923 | 0.636582 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 20 | max_up | 48 | 0.0653582 | 3481 | 0.0383476 | 0.0270106 | 0.00387294 | 0.0501483 | 0.022133 | 0.221125 | 0.512138 | true |
| new_high_low_120_breadth_reversal_top | top | onset | 上证指数 | 20 | terminal_return | 48 | 0.00815929 | 3481 | 0.00462036 | 0.00353893 | -0.0360124 | 0.0430902 | 0.860785 | 0.882333 | 0.961777 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_down | 27 | -0.024797 | 3515 | -0.0220082 | -0.00278879 | -0.0112364 | 0.00565878 | 0.517597 | 0.923094 | 0.873541 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_up | 27 | 0.0213781 | 3515 | 0.0196105 | 0.00176765 | -0.00613975 | 0.00967505 | 0.66128 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | terminal_return | 27 | -0.00147559 | 3515 | 0.00173161 | -0.00320719 | -0.0147633 | 0.00834896 | 0.586468 | 0.923094 | 0.893371 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_down | 27 | -0.0328398 | 3510 | -0.0317552 | -0.00108455 | -0.00993403 | 0.00776494 | 0.810169 | 0.930609 | 0.945198 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_up | 27 | 0.0303348 | 3510 | 0.0293025 | 0.00103226 | -0.00900611 | 0.0110706 | 0.840269 | 0.930609 | 0.957283 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | terminal_return | 27 | 0.010927 | 3510 | 0.00331709 | 0.00760994 | -0.00537421 | 0.0205941 | 0.250661 | 0.923094 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_down | 27 | -0.044189 | 3500 | -0.0452213 | 0.0010323 | -0.0127822 | 0.0148468 | 0.883556 | 0.930609 | 0.963092 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_up | 27 | 0.0489565 | 3500 | 0.0443155 | 0.00464103 | -0.0117159 | 0.020998 | 0.57813 | 0.923094 | 0.891848 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | terminal_return | 27 | 0.0169815 | 3500 | 0.00657207 | 0.0104094 | -0.0149257 | 0.0357446 | 0.420644 | 0.923094 | 0.839111 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_down | 27 | -0.032317 | 3517 | -0.0266957 | -0.00562128 | -0.0191968 | 0.00795425 | 0.41703 | 0.923094 | 0.839111 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_up | 27 | 0.0258686 | 3517 | 0.0234551 | 0.00241351 | -0.00831899 | 0.013146 | 0.659385 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 27 | -0.00336062 | 3517 | 0.0026999 | -0.00606052 | -0.022687 | 0.010566 | 0.474956 | 0.923094 | 0.8636 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_down | 27 | -0.0434606 | 3512 | -0.0389597 | -0.00450097 | -0.0181754 | 0.00917343 | 0.518837 | 0.923094 | 0.873541 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_up | 27 | 0.0362153 | 3512 | 0.0359372 | 0.000278092 | -0.013414 | 0.0139702 | 0.968246 | 0.968246 | 0.991753 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 27 | 0.0124849 | 3512 | 0.00523895 | 0.00724596 | -0.0120664 | 0.0265583 | 0.462103 | 0.923094 | 0.855311 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_down | 27 | -0.0589651 | 3502 | -0.055984 | -0.00298112 | -0.0227837 | 0.0168214 | 0.767946 | 0.930396 | 0.918343 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_up | 27 | 0.0572825 | 3502 | 0.0550464 | 0.00223609 | -0.0214749 | 0.0259471 | 0.853355 | 0.930609 | 0.961777 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 27 | 0.0185302 | 3502 | 0.0104974 | 0.00803278 | -0.026735 | 0.0428005 | 0.650664 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_down | 27 | -0.0333354 | 3517 | -0.027106 | -0.00622937 | -0.0187631 | 0.00630438 | 0.32999 | 0.923094 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_up | 27 | 0.0246679 | 3517 | 0.0235535 | 0.00111441 | -0.00838278 | 0.0106116 | 0.818101 | 0.930609 | 0.947143 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 27 | -0.00394506 | 3517 | 0.00206363 | -0.00600869 | -0.0214915 | 0.00947408 | 0.446864 | 0.923094 | 0.84965 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_down | 27 | -0.0447823 | 3512 | -0.0396103 | -0.00517199 | -0.0187141 | 0.00837015 | 0.454123 | 0.923094 | 0.855311 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_up | 27 | 0.0360366 | 3512 | 0.0356096 | 0.000427014 | -0.0122768 | 0.0131308 | 0.947472 | 0.968246 | 0.986624 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 27 | 0.0104062 | 3512 | 0.00395362 | 0.00645256 | -0.0125653 | 0.0254704 | 0.506046 | 0.923094 | 0.873541 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_down | 27 | -0.0618871 | 3502 | -0.0569633 | -0.00492373 | -0.0251082 | 0.0152607 | 0.632568 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_up | 27 | 0.0556844 | 3502 | 0.0540732 | 0.00161117 | -0.0204729 | 0.0236953 | 0.886295 | 0.930609 | 0.963092 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 27 | 0.0138485 | 3502 | 0.00789081 | 0.00595772 | -0.0280138 | 0.0399292 | 0.731047 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_down | 27 | -0.0211983 | 3517 | -0.0202759 | -0.000922407 | -0.00819118 | 0.00634637 | 0.803574 | 0.930609 | 0.943326 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_up | 27 | 0.0229288 | 3517 | 0.0195575 | 0.00337131 | -0.00495163 | 0.0116943 | 0.427241 | 0.923094 | 0.839111 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 27 | 0.000603746 | 3517 | 0.0014518 | -0.000848053 | -0.0118154 | 0.0101193 | 0.879536 | 0.930609 | 0.963092 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_down | 27 | -0.0274765 | 3512 | -0.0290758 | 0.00159928 | -0.00613022 | 0.00932877 | 0.685084 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_up | 27 | 0.0327361 | 3512 | 0.0291579 | 0.00357823 | -0.00661188 | 0.0137683 | 0.491296 | 0.923094 | 0.869575 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 27 | 0.0118431 | 3512 | 0.00275668 | 0.00908645 | -0.00249721 | 0.0206701 | 0.12418 | 0.923094 | 0.757571 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_down | 27 | -0.0358165 | 3502 | -0.0409953 | 0.00517872 | -0.00681991 | 0.0171773 | 0.397578 | 0.923094 | 0.829314 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_up | 27 | 0.0519113 | 3502 | 0.0439333 | 0.00797802 | -0.00687815 | 0.0228342 | 0.292546 | 0.923094 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 27 | 0.0196173 | 3502 | 0.00547918 | 0.0141381 | -0.00850651 | 0.0367828 | 0.221057 | 0.923094 | 0.795806 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_down | 27 | -0.0293166 | 3517 | -0.0244815 | -0.00483509 | -0.0151499 | 0.00547976 | 0.358226 | 0.923094 | 0.811114 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_up | 27 | 0.026116 | 3517 | 0.0219466 | 0.00416935 | -0.00565114 | 0.0139898 | 0.405335 | 0.923094 | 0.834968 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 27 | -0.00053484 | 3517 | 0.00199785 | -0.00253269 | -0.017706 | 0.0126406 | 0.743548 | 0.923094 | 0.913005 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_down | 27 | -0.0384429 | 3512 | -0.0354437 | -0.00299925 | -0.0146681 | 0.00866965 | 0.614417 | 0.923094 | 0.9043 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_up | 27 | 0.0359792 | 3512 | 0.0328786 | 0.00310061 | -0.00951735 | 0.0157186 | 0.630068 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 27 | 0.0109406 | 3512 | 0.00385316 | 0.00708748 | -0.00888611 | 0.0230611 | 0.384491 | 0.923094 | 0.825782 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_down | 27 | -0.0538594 | 3502 | -0.0504828 | -0.0033766 | -0.0208163 | 0.0140631 | 0.704327 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_up | 27 | 0.0561718 | 3502 | 0.0498823 | 0.00628949 | -0.0127069 | 0.0252858 | 0.51638 | 0.923094 | 0.873541 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 27 | 0.0178568 | 3502 | 0.00765762 | 0.0101991 | -0.0188528 | 0.0392511 | 0.491397 | 0.923094 | 0.869575 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_down | 27 | -0.0403538 | 3517 | -0.0276213 | -0.0127325 | -0.0356385 | 0.0101735 | 0.27594 | 0.923094 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_up | 27 | 0.0382505 | 3517 | 0.026098 | 0.0121525 | -0.00797532 | 0.0322803 | 0.236658 | 0.923094 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 27 | -0.00328166 | 3517 | 0.00447833 | -0.00775999 | -0.0417761 | 0.0262561 | 0.654782 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_down | 27 | -0.054312 | 3512 | -0.0402798 | -0.0140322 | -0.0368649 | 0.00880052 | 0.228378 | 0.923094 | 0.805223 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_up | 27 | 0.0512486 | 3512 | 0.0401807 | 0.0110679 | -0.0157799 | 0.0379157 | 0.419089 | 0.923094 | 0.839111 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 27 | 0.0172701 | 3512 | 0.00871351 | 0.00855663 | -0.0205709 | 0.0376842 | 0.564766 | 0.923094 | 0.889506 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_down | 27 | -0.0686548 | 3502 | -0.0579367 | -0.0107181 | -0.0371647 | 0.0157285 | 0.427001 | 0.923094 | 0.839111 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_up | 27 | 0.0812433 | 3502 | 0.0617602 | 0.0194831 | -0.0191865 | 0.0581527 | 0.323389 | 0.923094 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 27 | 0.0389043 | 3502 | 0.017054 | 0.0218502 | -0.0273236 | 0.071024 | 0.383797 | 0.923094 | 0.825782 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_down | 27 | -0.0202364 | 3517 | -0.0189276 | -0.00130878 | -0.00781617 | 0.00519861 | 0.693434 | 0.923094 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_up | 27 | 0.019238 | 3517 | 0.0171305 | 0.00210755 | -0.00541194 | 0.00962703 | 0.582769 | 0.923094 | 0.891848 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 27 | 0.00147255 | 3517 | 0.00122439 | 0.000248158 | -0.00952639 | 0.0100227 | 0.960313 | 0.968246 | 0.990581 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_down | 27 | -0.0252566 | 3512 | -0.0271697 | 0.00191314 | -0.00501895 | 0.00884523 | 0.588558 | 0.923094 | 0.893473 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_up | 27 | 0.0271538 | 3512 | 0.0256461 | 0.00150766 | -0.00766239 | 0.0106777 | 0.747266 | 0.923094 | 0.913005 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 27 | 0.0104565 | 3512 | 0.00232892 | 0.00812758 | -0.00247686 | 0.018732 | 0.133044 | 0.923094 | 0.760687 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_down | 27 | -0.0330528 | 3502 | -0.0385432 | 0.00549034 | -0.00469263 | 0.0156733 | 0.290616 | 0.923094 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_up | 27 | 0.0430767 | 3502 | 0.0386813 | 0.00439537 | -0.00947519 | 0.0182659 | 0.534538 | 0.923094 | 0.879031 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 27 | 0.0165696 | 3502 | 0.00457673 | 0.0119928 | -0.00774075 | 0.0317264 | 0.233589 | 0.923094 | 0.805223 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_down | 27 | -0.0256056 | 3515 | -0.022002 | -0.00360358 | -0.0124372 | 0.00523002 | 0.423964 | 0.87349 | 0.839111 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_up | 27 | 0.0229289 | 3515 | 0.0195985 | 0.00333037 | -0.00671956 | 0.0133803 | 0.516009 | 0.87349 | 0.873541 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 5 | terminal_return | 27 | 0.00417808 | 3515 | 0.00168818 | 0.0024899 | -0.0114516 | 0.0164314 | 0.726303 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_down | 27 | -0.0337464 | 3510 | -0.0317483 | -0.00199812 | -0.0117246 | 0.00772838 | 0.687211 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_up | 27 | 0.0309216 | 3510 | 0.029298 | 0.00162358 | -0.0103332 | 0.0135804 | 0.790129 | 0.921818 | 0.932827 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 10 | terminal_return | 27 | 0.010449 | 3510 | 0.00332076 | 0.00712821 | -0.00518048 | 0.0194369 | 0.256344 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_down | 27 | -0.0450056 | 3500 | -0.045215 | 0.000209329 | -0.0127271 | 0.0131458 | 0.974699 | 0.989125 | 0.991753 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_up | 27 | 0.047515 | 3500 | 0.0443266 | 0.0031884 | -0.0132415 | 0.0196183 | 0.703679 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 全A | 20 | terminal_return | 27 | 0.0173382 | 3500 | 0.00656932 | 0.0107689 | -0.0127821 | 0.0343199 | 0.370133 | 0.87349 | 0.813431 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_down | 27 | -0.0349031 | 3517 | -0.0266758 | -0.00822722 | -0.0236736 | 0.00721918 | 0.296506 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_up | 27 | 0.027086 | 3517 | 0.0234458 | 0.00364022 | -0.00894815 | 0.0162286 | 0.570864 | 0.87349 | 0.890044 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | terminal_return | 27 | 0.00240759 | 3517 | 0.00265562 | -0.000248032 | -0.0211534 | 0.0206573 | 0.981447 | 0.989125 | 0.992059 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_down | 27 | -0.0464916 | 3512 | -0.0389364 | -0.00755524 | -0.0229899 | 0.00787939 | 0.337348 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_up | 27 | 0.0363958 | 3512 | 0.0359358 | 0.000460048 | -0.0177207 | 0.0186408 | 0.960444 | 0.989125 | 0.990581 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | terminal_return | 27 | 0.0124514 | 3512 | 0.00523921 | 0.00721217 | -0.0121335 | 0.0265578 | 0.464963 | 0.87349 | 0.855442 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_down | 27 | -0.0625249 | 3502 | -0.0559565 | -0.00656839 | -0.026578 | 0.0134412 | 0.519969 | 0.87349 | 0.873541 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_up | 27 | 0.0555997 | 3502 | 0.0550594 | 0.000540287 | -0.0248564 | 0.025937 | 0.96674 | 0.989125 | 0.991753 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | terminal_return | 27 | 0.0182305 | 3502 | 0.0104997 | 0.00773086 | -0.026909 | 0.0423707 | 0.6618 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_down | 27 | -0.0348651 | 3517 | -0.0270943 | -0.00777082 | -0.0219182 | 0.00637651 | 0.281666 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_up | 27 | 0.0258443 | 3517 | 0.0235445 | 0.00229982 | -0.00849758 | 0.0130972 | 0.676331 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | terminal_return | 27 | 0.000674717 | 3517 | 0.00202816 | -0.00135344 | -0.0203696 | 0.0176627 | 0.889055 | 0.989125 | 0.963092 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_down | 27 | -0.0470844 | 3512 | -0.0395926 | -0.00749179 | -0.0226314 | 0.00764785 | 0.332097 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_up | 27 | 0.0355026 | 3512 | 0.0356137 | -0.000111114 | -0.0160897 | 0.0158675 | 0.989125 | 0.989125 | 0.993054 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | terminal_return | 27 | 0.0100739 | 3512 | 0.00395618 | 0.00611769 | -0.012118 | 0.0243534 | 0.510835 | 0.87349 | 0.873541 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_down | 27 | -0.0643836 | 3502 | -0.0569441 | -0.0074395 | -0.0278639 | 0.0129849 | 0.475276 | 0.87349 | 0.8636 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_up | 27 | 0.0535839 | 3502 | 0.0540894 | -0.000505524 | -0.023527 | 0.0225159 | 0.96567 | 0.989125 | 0.991753 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | terminal_return | 27 | 0.013613 | 3502 | 0.00789263 | 0.00572032 | -0.0273828 | 0.0388235 | 0.73484 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_down | 27 | -0.0221562 | 3517 | -0.0202686 | -0.00188767 | -0.00828309 | 0.00450775 | 0.562917 | 0.87349 | 0.889439 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_up | 27 | 0.0241055 | 3517 | 0.0195484 | 0.00455707 | -0.00543469 | 0.0145488 | 0.371364 | 0.87349 | 0.813771 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | terminal_return | 27 | 0.00636341 | 3517 | 0.00140758 | 0.00495582 | -0.00763282 | 0.0175445 | 0.440351 | 0.87349 | 0.84965 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_down | 27 | -0.0273589 | 3512 | -0.0290767 | 0.00171775 | -0.00563034 | 0.00906584 | 0.646819 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_up | 27 | 0.0336029 | 3512 | 0.0291512 | 0.00445166 | -0.00660663 | 0.01551 | 0.430098 | 0.87349 | 0.839958 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | terminal_return | 27 | 0.0106942 | 3512 | 0.00276551 | 0.0079287 | -0.00322123 | 0.0190786 | 0.163392 | 0.87349 | 0.795806 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_down | 27 | -0.0359094 | 3502 | -0.0409945 | 0.00508514 | -0.00534774 | 0.015518 | 0.339409 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_up | 27 | 0.0511573 | 3502 | 0.0439391 | 0.00721824 | -0.00706414 | 0.0215006 | 0.321894 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | terminal_return | 27 | 0.0202063 | 3502 | 0.00547464 | 0.0147316 | -0.00542991 | 0.0348931 | 0.152106 | 0.87349 | 0.776974 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_down | 27 | -0.0306091 | 3517 | -0.0244716 | -0.00613752 | -0.0171999 | 0.0049249 | 0.276849 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_up | 27 | 0.0249042 | 3517 | 0.0219559 | 0.00294827 | -0.00715125 | 0.0130478 | 0.567209 | 0.87349 | 0.889647 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | terminal_return | 27 | 0.00293671 | 3517 | 0.0019712 | 0.000965515 | -0.0151282 | 0.0170593 | 0.906395 | 0.989125 | 0.971504 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_down | 27 | -0.0407965 | 3512 | -0.0354256 | -0.00537095 | -0.0181857 | 0.00744382 | 0.411374 | 0.87349 | 0.839111 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_up | 27 | 0.0353072 | 3512 | 0.0328838 | 0.00242345 | -0.0114656 | 0.0163125 | 0.732356 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | terminal_return | 27 | 0.0103208 | 3512 | 0.00385793 | 0.00646289 | -0.00832739 | 0.0212532 | 0.391744 | 0.87349 | 0.829314 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_down | 27 | -0.0548838 | 3502 | -0.0504749 | -0.00440892 | -0.0217591 | 0.0129413 | 0.61844 | 0.87349 | 0.904334 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_up | 27 | 0.0531686 | 3502 | 0.0499054 | 0.00326312 | -0.0154267 | 0.0219529 | 0.732198 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | terminal_return | 27 | 0.0171739 | 3502 | 0.00766288 | 0.00951099 | -0.0176734 | 0.0366953 | 0.492874 | 0.87349 | 0.869575 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_down | 27 | -0.0436523 | 3517 | -0.0275959 | -0.0160564 | -0.0396211 | 0.00750833 | 0.181715 | 0.87349 | 0.795806 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_up | 27 | 0.0396298 | 3517 | 0.0260874 | 0.0135424 | -0.00956307 | 0.0366478 | 0.250647 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | terminal_return | 27 | 0.00647363 | 3517 | 0.00440344 | 0.00207019 | -0.0336924 | 0.0378327 | 0.909667 | 0.989125 | 0.971584 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_down | 27 | -0.0595273 | 3512 | -0.0402397 | -0.0192876 | -0.0454088 | 0.00683366 | 0.14783 | 0.87349 | 0.765477 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_up | 27 | 0.0557575 | 3512 | 0.0401461 | 0.0156114 | -0.0203213 | 0.0515442 | 0.394466 | 0.87349 | 0.829314 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | terminal_return | 27 | 0.0190017 | 3512 | 0.00870019 | 0.0103015 | -0.0217437 | 0.0423466 | 0.528645 | 0.87349 | 0.876948 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_down | 27 | -0.0769221 | 3502 | -0.057873 | -0.0190492 | -0.0480742 | 0.00997588 | 0.198321 | 0.87349 | 0.795806 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_up | 27 | 0.0823859 | 3502 | 0.0617514 | 0.0206344 | -0.0222774 | 0.0635463 | 0.345947 | 0.87349 | 0.807564 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | terminal_return | 27 | 0.0381495 | 3502 | 0.0170598 | 0.0210897 | -0.0295674 | 0.0717468 | 0.414505 | 0.87349 | 0.839111 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_down | 27 | -0.0206899 | 3517 | -0.0189241 | -0.00176581 | -0.00804791 | 0.00451629 | 0.581682 | 0.87349 | 0.891848 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_up | 27 | 0.0200609 | 3517 | 0.0171242 | 0.0029367 | -0.00621657 | 0.01209 | 0.529454 | 0.87349 | 0.876948 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | terminal_return | 27 | 0.00597446 | 3517 | 0.00118983 | 0.00478463 | -0.0063947 | 0.015964 | 0.401549 | 0.87349 | 0.833986 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_down | 27 | -0.0257414 | 3512 | -0.027166 | 0.00142462 | -0.00567864 | 0.00852788 | 0.694249 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_up | 27 | 0.027995 | 3512 | 0.0256397 | 0.00235529 | -0.00797123 | 0.0126818 | 0.654846 | 0.87349 | 0.911661 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | terminal_return | 27 | 0.0110121 | 3512 | 0.00232465 | 0.00868745 | -0.00119896 | 0.0185739 | 0.0850145 | 0.87349 | 0.636582 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_down | 27 | -0.0337764 | 3502 | -0.0385376 | 0.00476121 | -0.00473521 | 0.0142576 | 0.325764 | 0.87349 | 0.806443 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_up | 27 | 0.0426186 | 3502 | 0.0386849 | 0.00393377 | -0.00972118 | 0.0175887 | 0.572315 | 0.87349 | 0.890044 | true |
| new_high_low_250_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | terminal_return | 27 | 0.0166718 | 3502 | 0.00457595 | 0.0120958 | -0.00651308 | 0.0307047 | 0.202662 | 0.87349 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_down | 24 | -0.0339542 | 3518 | -0.0219481 | -0.0120061 | -0.0254261 | 0.00141388 | 0.079516 | 0.434535 | 0.636582 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_up | 24 | 0.0235885 | 3518 | 0.0195969 | 0.00399162 | -0.00625123 | 0.0142345 | 0.444981 | 0.737732 | 0.84965 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 5 | terminal_return | 24 | 0.00213309 | 3518 | 0.00170425 | 0.000428836 | -0.0161766 | 0.0170343 | 0.959631 | 0.985498 | 0.990581 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_down | 24 | -0.0481064 | 3513 | -0.0316519 | -0.0164545 | -0.0427496 | 0.00984058 | 0.220012 | 0.570398 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_up | 24 | 0.0416752 | 3513 | 0.029226 | 0.0124493 | -0.00543588 | 0.0303344 | 0.172476 | 0.570398 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 10 | terminal_return | 24 | 0.0103795 | 3513 | 0.00332733 | 0.00705215 | -0.028381 | 0.0424853 | 0.696468 | 0.877798 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_down | 24 | -0.0779022 | 3503 | -0.0449894 | -0.0329128 | -0.0932612 | 0.0274356 | 0.285095 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_up | 24 | 0.0803458 | 3503 | 0.0441044 | 0.0362414 | -0.00470359 | 0.0771864 | 0.0827685 | 0.434535 | 0.636582 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 全A | 20 | terminal_return | 24 | 0.0157135 | 3503 | 0.00658967 | 0.00912384 | -0.0680719 | 0.0863195 | 0.816806 | 0.970921 | 0.947095 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_down | 24 | -0.0409118 | 3520 | -0.0266419 | -0.0142699 | -0.0283934 | -0.000146367 | 0.0476685 | 0.429016 | 0.529961 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_up | 24 | 0.0257659 | 3520 | 0.0234579 | 0.00230804 | -0.00929651 | 0.0139126 | 0.696665 | 0.877798 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 24 | -0.000853141 | 3520 | 0.00267764 | -0.00353078 | -0.0226971 | 0.0156355 | 0.718048 | 0.887001 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_down | 24 | -0.0572525 | 3515 | -0.0388693 | -0.0183831 | -0.0463001 | 0.00953394 | 0.196828 | 0.570398 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_up | 24 | 0.0482488 | 3515 | 0.0358552 | 0.0123935 | -0.0095445 | 0.0343316 | 0.268177 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 24 | 0.00902303 | 3515 | 0.00526877 | 0.00375426 | -0.0383994 | 0.0459079 | 0.861425 | 0.985498 | 0.961777 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_down | 24 | -0.0897588 | 3505 | -0.0557757 | -0.0339832 | -0.100532 | 0.0325657 | 0.316888 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_up | 24 | 0.084994 | 3505 | 0.0548586 | 0.0301354 | -0.0195289 | 0.0797996 | 0.234324 | 0.570398 | 0.805223 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 24 | 0.00822686 | 3505 | 0.0105748 | -0.00234794 | -0.0922695 | 0.0875736 | 0.959184 | 0.985498 | 0.990581 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_down | 24 | -0.0420201 | 3520 | -0.0270521 | -0.014968 | -0.0294532 | -0.000482824 | 0.0428334 | 0.429016 | 0.528042 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_up | 24 | 0.0260123 | 3520 | 0.0235453 | 0.00246702 | -0.00907481 | 0.0140088 | 0.675259 | 0.877798 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 24 | -0.00223986 | 3520 | 0.00204688 | -0.00428674 | -0.0235786 | 0.0150051 | 0.663184 | 0.877798 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_down | 24 | -0.0592052 | 3515 | -0.0395162 | -0.019689 | -0.0483133 | 0.00893542 | 0.177606 | 0.570398 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_up | 24 | 0.0476337 | 3515 | 0.0355308 | 0.0121029 | -0.00993256 | 0.0341384 | 0.281693 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 24 | 0.00588691 | 3515 | 0.00398998 | 0.00189693 | -0.0407644 | 0.0445582 | 0.930551 | 0.985498 | 0.980349 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_down | 24 | -0.0930785 | 3505 | -0.056754 | -0.0363245 | -0.104788 | 0.0321392 | 0.298382 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_up | 24 | 0.0825978 | 3505 | 0.0538903 | 0.0287075 | -0.0185591 | 0.075974 | 0.233884 | 0.570398 | 0.805223 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 24 | 0.00282387 | 3505 | 0.0079714 | -0.00514753 | -0.0944457 | 0.0841506 | 0.910044 | 0.985498 | 0.971584 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_down | 24 | -0.0317832 | 3520 | -0.0202045 | -0.0115787 | -0.0244506 | 0.00129324 | 0.0778871 | 0.434535 | 0.636582 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_up | 24 | 0.0279502 | 3520 | 0.0195261 | 0.00842407 | -0.00283657 | 0.0196847 | 0.142574 | 0.570398 | 0.762443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 24 | 0.0053429 | 3520 | 0.00141876 | 0.00392414 | -0.0126499 | 0.0204982 | 0.642607 | 0.877798 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_down | 24 | -0.0439185 | 3515 | -0.0289621 | -0.0149564 | -0.0388776 | 0.00896484 | 0.220402 | 0.570398 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_up | 24 | 0.044671 | 3515 | 0.0290794 | 0.0155915 | -0.00308006 | 0.0342631 | 0.101698 | 0.492842 | 0.686459 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 24 | 0.0134049 | 3515 | 0.00275377 | 0.0106511 | -0.0236832 | 0.0449854 | 0.54317 | 0.806897 | 0.881341 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_down | 24 | -0.0708855 | 3505 | -0.0407507 | -0.0301348 | -0.0825926 | 0.022323 | 0.260192 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_up | 24 | 0.0911978 | 3505 | 0.0436711 | 0.0475268 | 0.00202271 | 0.0930308 | 0.0406459 | 0.429016 | 0.512138 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 24 | 0.0276249 | 3505 | 0.00543645 | 0.0221884 | -0.0493531 | 0.0937299 | 0.543261 | 0.806897 | 0.881341 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_down | 24 | -0.0405056 | 3520 | -0.0244094 | -0.0160962 | -0.0310254 | -0.00116702 | 0.0345824 | 0.429016 | 0.512138 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_up | 24 | 0.0251555 | 3520 | 0.0219567 | 0.00319877 | -0.00765613 | 0.0140537 | 0.563547 | 0.806897 | 0.889439 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 24 | -0.00114339 | 3520 | 0.00199984 | -0.00314323 | -0.0215497 | 0.0152632 | 0.737848 | 0.893931 | 0.912661 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_down | 24 | -0.0564093 | 3515 | -0.0353236 | -0.0210857 | -0.049587 | 0.00741551 | 0.147046 | 0.570398 | 0.765477 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_up | 24 | 0.0429779 | 3515 | 0.0328335 | 0.0101444 | -0.00925291 | 0.0295417 | 0.305345 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 24 | 0.00537141 | 3515 | 0.00389724 | 0.00147417 | -0.0373598 | 0.0403082 | 0.940689 | 0.985498 | 0.98489 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_down | 24 | -0.0879074 | 3505 | -0.0502525 | -0.0376549 | -0.103105 | 0.0277951 | 0.259475 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_up | 24 | 0.0766513 | 3505 | 0.0497474 | 0.0269038 | -0.017138 | 0.0709456 | 0.231188 | 0.570398 | 0.805223 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 24 | 0.00594896 | 3505 | 0.00774788 | -0.00179892 | -0.0836071 | 0.0800093 | 0.965622 | 0.985498 | 0.991753 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_down | 24 | -0.041042 | 3520 | -0.0276274 | -0.0134146 | -0.0285553 | 0.00172608 | 0.0824659 | 0.434535 | 0.636582 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_up | 24 | 0.026019 | 3520 | 0.0261918 | -0.000172732 | -0.0108444 | 0.0104989 | 0.974692 | 0.985498 | 0.991753 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 24 | -0.000867921 | 3520 | 0.00445526 | -0.00532318 | -0.0242468 | 0.0136005 | 0.581398 | 0.813957 | 0.891848 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_down | 24 | -0.0559706 | 3515 | -0.0402805 | -0.0156902 | -0.0444055 | 0.0130251 | 0.284191 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_up | 24 | 0.0461314 | 3515 | 0.0402251 | 0.00590626 | -0.0139409 | 0.0257534 | 0.559711 | 0.806897 | 0.889439 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 24 | 0.00840732 | 3515 | 0.00878132 | -0.000374008 | -0.0407037 | 0.0399557 | 0.985498 | 0.985498 | 0.992059 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_down | 24 | -0.0897002 | 3505 | -0.0578018 | -0.0318984 | -0.0988254 | 0.0350286 | 0.350219 | 0.612883 | 0.80968 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_up | 24 | 0.0842302 | 3505 | 0.0617565 | 0.0224738 | -0.0212084 | 0.0661559 | 0.313268 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 24 | 0.00824691 | 3505 | 0.0172827 | -0.00903575 | -0.0940492 | 0.0759777 | 0.834979 | 0.974142 | 0.957283 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_down | 24 | -0.0309247 | 3520 | -0.0188559 | -0.0120689 | -0.0253843 | 0.00124651 | 0.0756478 | 0.434535 | 0.636582 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_up | 24 | 0.028672 | 3520 | 0.017068 | 0.011604 | 0.000285636 | 0.0229224 | 0.044488 | 0.429016 | 0.528042 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 24 | 0.00687671 | 3520 | 0.00118775 | 0.00568896 | -0.0111226 | 0.0225005 | 0.507166 | 0.798787 | 0.873541 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_down | 24 | -0.0430864 | 3515 | -0.0270463 | -0.0160401 | -0.041045 | 0.00896481 | 0.208646 | 0.570398 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_up | 24 | 0.045596 | 3515 | 0.0255215 | 0.0200745 | 0.0012612 | 0.0388879 | 0.0364928 | 0.429016 | 0.512138 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 24 | 0.0157051 | 3515 | 0.00230002 | 0.0134051 | -0.0209559 | 0.047766 | 0.444483 | 0.737732 | 0.84965 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_down | 24 | -0.0676285 | 3505 | -0.0383017 | -0.0293268 | -0.0820911 | 0.0234375 | 0.275985 | 0.570398 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_up | 24 | 0.0905076 | 3505 | 0.0383603 | 0.0521473 | 0.0107532 | 0.0935414 | 0.0135431 | 0.429016 | 0.512138 | true |
| new_high_low_250_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 24 | 0.0292092 | 3505 | 0.00450045 | 0.0247088 | -0.0449469 | 0.0943645 | 0.486889 | 0.786513 | 0.868025 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 5 | max_down | 24 | -0.0307444 | 3518 | -0.02197 | -0.00877441 | -0.0204265 | 0.00287769 | 0.139959 | 0.523209 | 0.762443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 5 | max_up | 24 | 0.0271025 | 3518 | 0.0195729 | 0.00752962 | -0.00374011 | 0.0187993 | 0.190355 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 5 | terminal_return | 24 | 0.00219757 | 3518 | 0.00170381 | 0.000493754 | -0.0168384 | 0.0178259 | 0.955473 | 0.970883 | 0.990581 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 10 | max_down | 24 | -0.0471517 | 3513 | -0.0316584 | -0.0154933 | -0.0397396 | 0.00875293 | 0.210411 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 10 | max_up | 24 | 0.0445912 | 3513 | 0.029206 | 0.0153852 | -0.00429012 | 0.0350605 | 0.125367 | 0.523209 | 0.757571 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 10 | terminal_return | 24 | 0.0119249 | 3513 | 0.00331677 | 0.00860815 | -0.0234432 | 0.0406595 | 0.598608 | 0.812305 | 0.896134 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 20 | max_down | 24 | -0.0757365 | 3503 | -0.0450042 | -0.0307322 | -0.0859771 | 0.0245127 | 0.275568 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 20 | max_up | 24 | 0.0810105 | 3503 | 0.0440998 | 0.0369106 | -0.0015601 | 0.0753814 | 0.0600377 | 0.523209 | 0.550255 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 全A | 20 | terminal_return | 24 | 0.0100744 | 3503 | 0.00662831 | 0.0034461 | -0.0746428 | 0.081535 | 0.931072 | 0.961599 | 0.980349 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 5 | max_down | 24 | -0.0359665 | 3520 | -0.0266756 | -0.0092909 | -0.0229687 | 0.00438693 | 0.183069 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 5 | max_up | 24 | 0.030307 | 3520 | 0.0234269 | 0.0068801 | -0.006874 | 0.0206342 | 0.326872 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 5 | terminal_return | 24 | -0.000704263 | 3520 | 0.00267663 | -0.00338089 | -0.0240392 | 0.0172774 | 0.748386 | 0.873116 | 0.913005 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 10 | max_down | 24 | -0.0553893 | 3515 | -0.0388821 | -0.0165073 | -0.0424232 | 0.00940862 | 0.211873 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 10 | max_up | 24 | 0.0518638 | 3515 | 0.0358306 | 0.0160333 | -0.00931025 | 0.0413768 | 0.214986 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 10 | terminal_return | 24 | 0.0101201 | 3515 | 0.00526128 | 0.00485886 | -0.0357464 | 0.0454642 | 0.81457 | 0.917696 | 0.945952 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 20 | max_down | 24 | -0.0858491 | 3505 | -0.0558024 | -0.0300466 | -0.0899613 | 0.029868 | 0.325646 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 20 | max_up | 24 | 0.0872483 | 3505 | 0.0548432 | 0.0324051 | -0.0174316 | 0.0822417 | 0.202506 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 国证2000 | 20 | terminal_return | 24 | 0.00329435 | 3505 | 0.0106086 | -0.00731423 | -0.0977132 | 0.0830848 | 0.873996 | 0.917696 | 0.961777 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 5 | max_down | 24 | -0.0369479 | 3520 | -0.0270867 | -0.00986122 | -0.0237196 | 0.00399713 | 0.163112 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 5 | max_up | 24 | 0.0307063 | 3520 | 0.0235133 | 0.00719305 | -0.00671669 | 0.0211028 | 0.310792 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 5 | terminal_return | 24 | -0.0017301 | 3520 | 0.0020434 | -0.0037735 | -0.0245952 | 0.0170482 | 0.722432 | 0.873116 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 10 | max_down | 24 | -0.0572848 | 3515 | -0.0395293 | -0.0177555 | -0.0444247 | 0.00891381 | 0.191927 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 10 | max_up | 24 | 0.051301 | 3515 | 0.0355057 | 0.0157953 | -0.0096778 | 0.0412683 | 0.224232 | 0.523209 | 0.799107 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 10 | terminal_return | 24 | 0.0073954 | 3515 | 0.00397968 | 0.00341571 | -0.0373947 | 0.0442262 | 0.869695 | 0.917696 | 0.961777 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 20 | max_down | 24 | -0.0892565 | 3505 | -0.0567801 | -0.0324764 | -0.0942289 | 0.0292762 | 0.302641 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 20 | max_up | 24 | 0.0845883 | 3505 | 0.0538767 | 0.0307116 | -0.0169008 | 0.078324 | 0.206134 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证1000 | 20 | terminal_return | 24 | -0.00197115 | 3505 | 0.00800423 | -0.00997538 | -0.100351 | 0.0804003 | 0.828724 | 0.917696 | 0.954346 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 5 | max_down | 24 | -0.0287854 | 3520 | -0.020225 | -0.0085604 | -0.0199487 | 0.00282788 | 0.140669 | 0.523209 | 0.762443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 5 | max_up | 24 | 0.0304115 | 3520 | 0.0195093 | 0.0109022 | -4.12472e-05 | 0.0218456 | 0.0508655 | 0.523209 | 0.539768 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 5 | terminal_return | 24 | 0.00524394 | 3520 | 0.00141944 | 0.0038245 | -0.0132198 | 0.0208688 | 0.660084 | 0.840801 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 10 | max_down | 24 | -0.0433197 | 3515 | -0.0289662 | -0.0143535 | -0.0372302 | 0.00852316 | 0.218786 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 10 | max_up | 24 | 0.0480875 | 3515 | 0.0290561 | 0.0190314 | -0.000217715 | 0.0382805 | 0.0526438 | 0.523209 | 0.539768 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 10 | terminal_return | 24 | 0.015183 | 3515 | 0.00274163 | 0.0124414 | -0.0182826 | 0.0431654 | 0.427379 | 0.641068 | 0.839111 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 20 | max_down | 24 | -0.0695204 | 3505 | -0.04076 | -0.0287604 | -0.0776147 | 0.0200939 | 0.248563 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 20 | max_up | 24 | 0.0906586 | 3505 | 0.0436748 | 0.0469838 | 0.00697839 | 0.0869892 | 0.0213411 | 0.342339 | 0.512138 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 沪深300 | 20 | terminal_return | 24 | 0.0215767 | 3505 | 0.00547787 | 0.0160988 | -0.0573068 | 0.0895044 | 0.667302 | 0.840801 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 5 | max_down | 24 | -0.0365836 | 3520 | -0.0244361 | -0.0121475 | -0.0253269 | 0.00103199 | 0.0708361 | 0.523209 | 0.615541 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 5 | max_up | 24 | 0.028812 | 3520 | 0.0219318 | 0.00688023 | -0.00622262 | 0.0199831 | 0.303393 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 5 | terminal_return | 24 | -0.001339 | 3520 | 0.00200117 | -0.00334017 | -0.0230897 | 0.0164094 | 0.740276 | 0.873116 | 0.912661 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 10 | max_down | 24 | -0.0551952 | 3515 | -0.0353319 | -0.0198633 | -0.0461321 | 0.00640554 | 0.138324 | 0.523209 | 0.762443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 10 | max_up | 24 | 0.0458764 | 3515 | 0.0328137 | 0.0130627 | -0.00971154 | 0.035837 | 0.260926 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 10 | terminal_return | 24 | 0.00687993 | 3515 | 0.00388694 | 0.00299299 | -0.0330322 | 0.0390182 | 0.870646 | 0.917696 | 0.961777 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 20 | max_down | 24 | -0.0851199 | 3505 | -0.0502716 | -0.0348483 | -0.0941686 | 0.0244721 | 0.24956 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 20 | max_up | 24 | 0.0785963 | 3505 | 0.0497341 | 0.0288621 | -0.0153928 | 0.0731171 | 0.201154 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 中证500 | 20 | terminal_return | 24 | -0.000958518 | 3505 | 0.00779518 | -0.0087537 | -0.0918972 | 0.0743898 | 0.836512 | 0.917696 | 0.957283 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 5 | max_down | 24 | -0.0365862 | 3520 | -0.0276578 | -0.00892836 | -0.0230787 | 0.00522199 | 0.216202 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 5 | max_up | 24 | 0.0290944 | 3520 | 0.0261708 | 0.00292357 | -0.0103713 | 0.0162184 | 0.666462 | 0.840801 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 5 | terminal_return | 24 | -0.00102354 | 3520 | 0.00445632 | -0.00547986 | -0.0260829 | 0.0151231 | 0.602152 | 0.812305 | 0.898243 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 10 | max_down | 24 | -0.0550318 | 3515 | -0.0402869 | -0.0147449 | -0.0413439 | 0.0118541 | 0.277253 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 10 | max_up | 24 | 0.0479915 | 3515 | 0.0402124 | 0.00777913 | -0.0155505 | 0.0311087 | 0.513401 | 0.752192 | 0.873541 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 10 | terminal_return | 24 | 0.00924268 | 3515 | 0.00877562 | 0.000467063 | -0.0373951 | 0.0383292 | 0.98071 | 0.98071 | 0.992059 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 20 | max_down | 24 | -0.087392 | 3505 | -0.0578176 | -0.0295744 | -0.0903567 | 0.0312078 | 0.340253 | 0.534504 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 20 | max_up | 24 | 0.0851823 | 3505 | 0.0617499 | 0.0234324 | -0.0207338 | 0.0675986 | 0.298397 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 微盘股 | 20 | terminal_return | 24 | 0.00150766 | 3505 | 0.0173288 | -0.0158211 | -0.102237 | 0.0705949 | 0.719716 | 0.873116 | 0.911661 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 5 | max_down | 24 | -0.0284039 | 3520 | -0.018873 | -0.00953089 | -0.0215205 | 0.00245875 | 0.11922 | 0.523209 | 0.757395 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 5 | max_up | 24 | 0.0302753 | 3520 | 0.017057 | 0.0132182 | 0.00205561 | 0.0243809 | 0.0202903 | 0.342339 | 0.512138 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 5 | terminal_return | 24 | 0.00638029 | 3520 | 0.00119114 | 0.00518915 | -0.0120206 | 0.0223989 | 0.55453 | 0.793986 | 0.888798 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 10 | max_down | 24 | -0.0419684 | 3515 | -0.027054 | -0.0149144 | -0.0382291 | 0.00840023 | 0.209909 | 0.523209 | 0.795806 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 10 | max_up | 24 | 0.0483138 | 3515 | 0.0255029 | 0.0228109 | 0.00332929 | 0.0422925 | 0.0217358 | 0.342339 | 0.512138 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 10 | terminal_return | 24 | 0.017069 | 3515 | 0.00229071 | 0.0147783 | -0.0160765 | 0.0456331 | 0.347852 | 0.534504 | 0.807564 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 20 | max_down | 24 | -0.0661541 | 3505 | -0.0383118 | -0.0278423 | -0.0770566 | 0.021372 | 0.267498 | 0.528024 | 0.806443 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 20 | max_up | 24 | 0.0900679 | 3505 | 0.0383633 | 0.0517046 | 0.0144945 | 0.0889146 | 0.00645973 | 0.342339 | 0.512138 | true |
| new_high_low_250_breadth_reversal_top | top | onset | 上证指数 | 20 | terminal_return | 24 | 0.0230937 | 3505 | 0.00454233 | 0.0185513 | -0.0519445 | 0.0890472 | 0.606005 | 0.812305 | 0.900079 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_down | 73 | -0.022955 | 3469 | -0.02201 | -0.000945055 | -0.00619944 | 0.00430933 | 0.724444 | 0.928818 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_up | 73 | 0.0212295 | 3469 | 0.0195901 | 0.00163937 | -0.00310042 | 0.00637916 | 0.497828 | 0.928818 | 0.87322 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | terminal_return | 73 | -0.000528228 | 3469 | 0.0017542 | -0.00228243 | -0.00932963 | 0.00476478 | 0.52556 | 0.928818 | 0.876065 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_down | 73 | -0.0347511 | 3464 | -0.0317005 | -0.0030506 | -0.0097099 | 0.0036087 | 0.369256 | 0.928818 | 0.813431 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_up | 73 | 0.0279835 | 3464 | 0.0293384 | -0.00135491 | -0.00735367 | 0.00464385 | 0.657986 | 0.928818 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | terminal_return | 73 | 0.00053562 | 3464 | 0.00343502 | -0.0028994 | -0.0117499 | 0.00595111 | 0.520815 | 0.928818 | 0.873541 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_down | 72 | -0.0480537 | 3455 | -0.0451542 | -0.00289957 | -0.0119516 | 0.00615245 | 0.530113 | 0.928818 | 0.876948 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_up | 72 | 0.0461034 | 3455 | 0.0443145 | 0.00178892 | -0.010026 | 0.0136038 | 0.766644 | 0.928818 | 0.918343 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | terminal_return | 72 | 0.0120767 | 3455 | 0.0065387 | 0.00553803 | -0.0105748 | 0.0216509 | 0.500529 | 0.928818 | 0.873541 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_down | 73 | -0.0292265 | 3471 | -0.0266862 | -0.00254029 | -0.0105548 | 0.00547422 | 0.534439 | 0.928818 | 0.879031 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_up | 73 | 0.026015 | 3471 | 0.0234201 | 0.00259492 | -0.00414304 | 0.00933289 | 0.450348 | 0.928818 | 0.851158 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 73 | -0.00101585 | 3471 | 0.00273091 | -0.00374676 | -0.0141956 | 0.00670205 | 0.482167 | 0.928818 | 0.865283 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_down | 73 | -0.0434186 | 3466 | -0.0389008 | -0.00451781 | -0.0136649 | 0.00462927 | 0.333016 | 0.928818 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_up | 73 | 0.0357935 | 3466 | 0.0359424 | -0.000148835 | -0.00893142 | 0.00863375 | 0.973503 | 0.984448 | 0.991753 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 73 | 0.00374215 | 3466 | 0.00532692 | -0.00158477 | -0.0142704 | 0.0111009 | 0.806569 | 0.940997 | 0.943432 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_down | 72 | -0.0599497 | 3457 | -0.0559247 | -0.00402502 | -0.0167771 | 0.00872707 | 0.536149 | 0.928818 | 0.879238 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_up | 72 | 0.0594715 | 3457 | 0.0549717 | 0.00449981 | -0.0111621 | 0.0201617 | 0.573348 | 0.928818 | 0.890044 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 72 | 0.0205304 | 3457 | 0.0103512 | 0.0101793 | -0.0106493 | 0.0310079 | 0.338121 | 0.928818 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_down | 73 | -0.0291708 | 3471 | -0.0271111 | -0.00205975 | -0.00960338 | 0.00548388 | 0.592533 | 0.928818 | 0.896134 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_up | 73 | 0.025854 | 3471 | 0.0235138 | 0.00234021 | -0.00410027 | 0.00878069 | 0.476351 | 0.928818 | 0.8636 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 73 | -0.00113492 | 3471 | 0.00208416 | -0.00321908 | -0.0131713 | 0.00673318 | 0.526103 | 0.928818 | 0.876065 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_down | 73 | -0.0434357 | 3466 | -0.03957 | -0.00386572 | -0.0128197 | 0.00508829 | 0.397445 | 0.928818 | 0.829314 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_up | 73 | 0.0357101 | 3466 | 0.0356108 | 9.93092e-05 | -0.00816847 | 0.00836708 | 0.981217 | 0.984448 | 0.992059 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 73 | 0.00336304 | 3466 | 0.00401632 | -0.000653286 | -0.0128012 | 0.0114946 | 0.916055 | 0.961858 | 0.975405 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_down | 72 | -0.0609813 | 3457 | -0.0569181 | -0.00406324 | -0.0160782 | 0.00795167 | 0.507433 | 0.928818 | 0.873541 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_up | 72 | 0.0589541 | 3457 | 0.0539841 | 0.00497001 | -0.0101683 | 0.0201084 | 0.519913 | 0.928818 | 0.873541 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 72 | 0.0175323 | 3457 | 0.00773654 | 0.00979574 | -0.0106124 | 0.0302039 | 0.346815 | 0.928818 | 0.807564 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_down | 73 | -0.0203254 | 3471 | -0.020282 | -4.33721e-05 | -0.00440443 | 0.00431769 | 0.984448 | 0.984448 | 0.992059 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_up | 73 | 0.0207771 | 3471 | 0.019558 | 0.00121906 | -0.00359744 | 0.00603555 | 0.61984 | 0.928818 | 0.904631 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 73 | 0.000140547 | 3471 | 0.00147278 | -0.00133223 | -0.00747357 | 0.0048091 | 0.670705 | 0.928818 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_down | 73 | -0.0308558 | 3466 | -0.0290258 | -0.00182996 | -0.00800664 | 0.00434671 | 0.56145 | 0.928818 | 0.889439 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_up | 73 | 0.0282652 | 3466 | 0.0292045 | -0.000939358 | -0.00679239 | 0.00491367 | 0.753094 | 0.928818 | 0.913005 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 73 | -0.000718412 | 3466 | 0.00290065 | -0.00361906 | -0.0119814 | 0.00474327 | 0.396298 | 0.928818 | 0.829314 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_down | 72 | -0.0425897 | 3457 | -0.0409216 | -0.00166813 | -0.0111669 | 0.00783063 | 0.730691 | 0.928818 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_up | 72 | 0.0459601 | 3457 | 0.0439534 | 0.00200672 | -0.00880975 | 0.0128232 | 0.716136 | 0.928818 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 72 | 0.00808101 | 3457 | 0.00553541 | 0.00254559 | -0.0131776 | 0.0182687 | 0.750996 | 0.928818 | 0.913005 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_down | 73 | -0.0260108 | 3471 | -0.024487 | -0.00152378 | -0.00784577 | 0.00479821 | 0.63663 | 0.928818 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_up | 73 | 0.024209 | 3471 | 0.0219315 | 0.00227751 | -0.00333547 | 0.0078905 | 0.426447 | 0.928818 | 0.839111 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 73 | -0.000266215 | 3471 | 0.00202576 | -0.00229198 | -0.0107498 | 0.00616583 | 0.595322 | 0.928818 | 0.896134 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_down | 73 | -0.0391257 | 3466 | -0.0353895 | -0.0037362 | -0.011856 | 0.00438362 | 0.367131 | 0.928818 | 0.813431 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_up | 73 | 0.0334604 | 3466 | 0.0328905 | 0.000569874 | -0.00703015 | 0.0081699 | 0.883158 | 0.961858 | 0.963092 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 73 | 0.00220959 | 3466 | 0.00394299 | -0.0017334 | -0.0123854 | 0.00891862 | 0.749764 | 0.928818 | 0.913005 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_down | 72 | -0.053893 | 3457 | -0.0504381 | -0.00345484 | -0.0135547 | 0.00664503 | 0.502569 | 0.928818 | 0.873541 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_up | 72 | 0.0550982 | 3457 | 0.0498228 | 0.00527542 | -0.00871032 | 0.0192612 | 0.459718 | 0.928818 | 0.855311 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 72 | 0.0168151 | 3457 | 0.00754655 | 0.00926854 | -0.00892704 | 0.0274641 | 0.318089 | 0.928818 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_down | 73 | -0.0334265 | 3471 | -0.0275982 | -0.00582831 | -0.0167613 | 0.00510464 | 0.296084 | 0.928818 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_up | 73 | 0.0330945 | 3471 | 0.0260454 | 0.00704915 | -0.00258069 | 0.016679 | 0.151361 | 0.928818 | 0.776974 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 73 | 0.000523873 | 3471 | 0.00450114 | -0.00397727 | -0.0192322 | 0.0112777 | 0.609344 | 0.928818 | 0.902594 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_down | 73 | -0.0502172 | 3466 | -0.0401798 | -0.0100373 | -0.0228356 | 0.00276095 | 0.124251 | 0.928818 | 0.757571 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_up | 73 | 0.0461135 | 3466 | 0.040142 | 0.0059715 | -0.00747186 | 0.0194149 | 0.383959 | 0.928818 | 0.825782 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 73 | 0.00750398 | 3466 | 0.00880564 | -0.00130166 | -0.0196882 | 0.0170849 | 0.889643 | 0.961858 | 0.963092 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_down | 72 | -0.0678903 | 3457 | -0.0578131 | -0.0100772 | -0.029287 | 0.00913267 | 0.303863 | 0.928818 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_up | 72 | 0.0759304 | 3457 | 0.0616173 | 0.0143132 | -0.00642897 | 0.0350553 | 0.176214 | 0.928818 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 72 | 0.031908 | 3457 | 0.0169153 | 0.0149927 | -0.0129306 | 0.0429161 | 0.292629 | 0.928818 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_down | 73 | -0.0185966 | 3471 | -0.0189448 | 0.000348189 | -0.00390543 | 0.00460181 | 0.872534 | 0.961858 | 0.961777 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_up | 73 | 0.0177269 | 3471 | 0.0171343 | 0.000592531 | -0.00371006 | 0.00489512 | 0.787221 | 0.935753 | 0.93136 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 73 | 2.05478e-05 | 3471 | 0.00125164 | -0.00123109 | -0.00682229 | 0.00436011 | 0.666061 | 0.928818 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_down | 73 | -0.0282399 | 3466 | -0.0271323 | -0.00110758 | -0.00681996 | 0.0046048 | 0.703925 | 0.928818 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_up | 73 | 0.0232903 | 3466 | 0.0257075 | -0.0024172 | -0.00763159 | 0.00279718 | 0.363568 | 0.928818 | 0.811114 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 73 | -0.000200105 | 3466 | 0.0024455 | -0.0026456 | -0.0099994 | 0.00470819 | 0.480729 | 0.928818 | 0.865283 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_down | 72 | -0.0380643 | 3457 | -0.0385103 | 0.00044594 | -0.0068636 | 0.00775548 | 0.904819 | 0.961858 | 0.971504 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_up | 72 | 0.0381344 | 3457 | 0.0387271 | -0.000592662 | -0.010563 | 0.0093777 | 0.907251 | 0.961858 | 0.971504 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 72 | 0.00839055 | 3457 | 0.00459097 | 0.00379958 | -0.00895531 | 0.0165545 | 0.559309 | 0.928818 | 0.889439 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_down | 73 | -0.0224214 | 3469 | -0.0220212 | -0.000400209 | -0.00597711 | 0.00517669 | 0.888144 | 0.961138 | 0.963092 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_up | 73 | 0.0238768 | 3469 | 0.0195344 | 0.00434236 | -0.00120211 | 0.00988684 | 0.124772 | 0.94047 | 0.757571 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 5 | terminal_return | 73 | 0.0049014 | 3469 | 0.00163994 | 0.00326146 | -0.00487274 | 0.0113957 | 0.431941 | 0.961138 | 0.839958 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_down | 73 | -0.0334173 | 3464 | -0.0317287 | -0.00168862 | -0.00845614 | 0.0050789 | 0.624803 | 0.961138 | 0.907706 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_up | 73 | 0.030022 | 3464 | 0.0292954 | 0.000726555 | -0.00584289 | 0.00729601 | 0.828389 | 0.961138 | 0.954346 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 10 | terminal_return | 73 | 0.00213337 | 3464 | 0.00340135 | -0.00126798 | -0.00960244 | 0.00706648 | 0.765559 | 0.961138 | 0.918343 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_down | 72 | -0.046535 | 3455 | -0.0451858 | -0.00134924 | -0.0102678 | 0.00756931 | 0.766835 | 0.961138 | 0.918343 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_up | 72 | 0.0460978 | 3455 | 0.0443146 | 0.0017832 | -0.00848601 | 0.0120524 | 0.733596 | 0.961138 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 全A | 20 | terminal_return | 72 | 0.013903 | 3455 | 0.00650064 | 0.00740238 | -0.00726683 | 0.0220716 | 0.322636 | 0.960274 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_down | 73 | -0.030581 | 3471 | -0.0266577 | -0.00392325 | -0.0122093 | 0.00436279 | 0.3534 | 0.961138 | 0.811114 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_up | 73 | 0.0285594 | 3471 | 0.0233665 | 0.00519284 | -0.00217 | 0.0125557 | 0.166866 | 0.94047 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | terminal_return | 73 | 0.00363134 | 3471 | 0.00263317 | 0.000998169 | -0.0108033 | 0.0127996 | 0.868332 | 0.961138 | 0.961777 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_down | 73 | -0.044387 | 3466 | -0.0388804 | -0.00550657 | -0.0148496 | 0.00383645 | 0.248016 | 0.960274 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_up | 73 | 0.038184 | 3466 | 0.035892 | 0.00229201 | -0.00746191 | 0.0120459 | 0.645109 | 0.961138 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | terminal_return | 73 | 0.00427924 | 3466 | 0.00531561 | -0.00103637 | -0.0131598 | 0.0110871 | 0.866937 | 0.961138 | 0.961777 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_down | 72 | -0.060139 | 3457 | -0.0559207 | -0.00421828 | -0.0170573 | 0.00862071 | 0.5196 | 0.961138 | 0.873541 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_up | 72 | 0.0580313 | 3457 | 0.0550017 | 0.00302956 | -0.0115669 | 0.017626 | 0.684149 | 0.961138 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | terminal_return | 72 | 0.0207951 | 3457 | 0.0103456 | 0.0104495 | -0.00932726 | 0.0302263 | 0.300384 | 0.960274 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_down | 73 | -0.0304146 | 3471 | -0.0270849 | -0.00332971 | -0.0113007 | 0.00464126 | 0.41293 | 0.961138 | 0.839111 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_up | 73 | 0.0281863 | 3471 | 0.0234647 | 0.00472157 | -0.00228757 | 0.0117307 | 0.18673 | 0.94047 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | terminal_return | 73 | 0.00297272 | 3471 | 0.00199777 | 0.000974948 | -0.0103276 | 0.0122775 | 0.865744 | 0.961138 | 0.961777 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_down | 73 | -0.0440761 | 3466 | -0.0395565 | -0.00451957 | -0.0137142 | 0.00467507 | 0.335334 | 0.960274 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_up | 73 | 0.037174 | 3466 | 0.03558 | 0.00159405 | -0.00744098 | 0.0106291 | 0.729491 | 0.961138 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | terminal_return | 73 | 0.003581 | 3466 | 0.00401173 | -0.000430734 | -0.0118322 | 0.0109707 | 0.940973 | 0.961138 | 0.98489 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_down | 72 | -0.0603621 | 3457 | -0.056931 | -0.00343108 | -0.0156007 | 0.00873854 | 0.580538 | 0.961138 | 0.891848 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_up | 72 | 0.0572443 | 3457 | 0.0540197 | 0.00322451 | -0.0106607 | 0.0171097 | 0.648991 | 0.961138 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | terminal_return | 72 | 0.018027 | 3457 | 0.00772623 | 0.0103008 | -0.00878517 | 0.0293868 | 0.290136 | 0.960274 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_down | 73 | -0.0190511 | 3471 | -0.0203088 | 0.00125777 | -0.00299696 | 0.0055125 | 0.562312 | 0.961138 | 0.889439 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_up | 73 | 0.0236487 | 3471 | 0.0194976 | 0.00415109 | -0.00124047 | 0.00954266 | 0.131286 | 0.94047 | 0.760687 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | terminal_return | 73 | 0.00593837 | 3471 | 0.00135084 | 0.00458752 | -0.00219359 | 0.0113686 | 0.18485 | 0.94047 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_down | 73 | -0.028463 | 3466 | -0.0290762 | 0.000613217 | -0.00525679 | 0.00648322 | 0.837765 | 0.961138 | 0.957283 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_up | 73 | 0.0308907 | 3466 | 0.0291492 | 0.00174149 | -0.00454757 | 0.00803056 | 0.587309 | 0.961138 | 0.893371 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | terminal_return | 73 | 0.00147466 | 3466 | 0.00285446 | -0.0013798 | -0.00924761 | 0.00648802 | 0.731049 | 0.961138 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_down | 72 | -0.0403351 | 3457 | -0.0409686 | 0.000633428 | -0.00819054 | 0.0094574 | 0.888108 | 0.961138 | 0.963092 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_up | 72 | 0.0467377 | 3457 | 0.0439372 | 0.00280052 | -0.00654198 | 0.012143 | 0.556846 | 0.961138 | 0.889439 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | terminal_return | 72 | 0.0105615 | 3457 | 0.00548375 | 0.00507777 | -0.00913781 | 0.0192934 | 0.48386 | 0.961138 | 0.865283 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_down | 73 | -0.0268418 | 3471 | -0.0244695 | -0.00237225 | -0.00901789 | 0.00427339 | 0.484147 | 0.961138 | 0.865283 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_up | 73 | 0.026424 | 3471 | 0.0218849 | 0.00453908 | -0.00182739 | 0.0109056 | 0.162289 | 0.94047 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | terminal_return | 73 | 0.00404709 | 3471 | 0.00193505 | 0.00211204 | -0.00762324 | 0.0118473 | 0.670679 | 0.961138 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_down | 73 | -0.0391099 | 3466 | -0.0353898 | -0.00372006 | -0.0120559 | 0.00461581 | 0.381742 | 0.961138 | 0.825782 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_up | 73 | 0.0349313 | 3466 | 0.0328595 | 0.00207179 | -0.00596725 | 0.0101108 | 0.613472 | 0.961138 | 0.9043 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | terminal_return | 73 | 0.00311847 | 3466 | 0.00392385 | -0.000805376 | -0.0112047 | 0.00959398 | 0.879351 | 0.961138 | 0.963092 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_down | 72 | -0.0523738 | 3457 | -0.0504698 | -0.001904 | -0.0124002 | 0.00859222 | 0.722185 | 0.961138 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_up | 72 | 0.054242 | 3457 | 0.0498406 | 0.00440136 | -0.00819539 | 0.0169981 | 0.493449 | 0.961138 | 0.869575 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | terminal_return | 72 | 0.0184938 | 3457 | 0.00751159 | 0.0109822 | -0.0058585 | 0.0278229 | 0.201193 | 0.94047 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_down | 73 | -0.0354303 | 3471 | -0.0275561 | -0.00787427 | -0.0179365 | 0.00218792 | 0.125075 | 0.94047 | 0.757571 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_up | 73 | 0.0361993 | 3471 | 0.0259801 | 0.0102192 | -0.000374685 | 0.0208131 | 0.0586672 | 0.94047 | 0.550255 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | terminal_return | 73 | 0.00761788 | 3471 | 0.00435194 | 0.00326594 | -0.01289 | 0.0194219 | 0.691946 | 0.961138 | 0.911661 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_down | 73 | -0.0533076 | 3466 | -0.0401147 | -0.0131929 | -0.0263992 | 1.35001e-05 | 0.0502304 | 0.94047 | 0.539768 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_up | 73 | 0.0498398 | 3466 | 0.0400635 | 0.00977625 | -0.00602101 | 0.0255735 | 0.225145 | 0.945609 | 0.799107 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | terminal_return | 73 | 0.00892259 | 3466 | 0.00877576 | 0.000146836 | -0.0180835 | 0.0183772 | 0.987404 | 0.987404 | 0.992657 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_down | 72 | -0.0710611 | 3457 | -0.0577471 | -0.013314 | -0.0332989 | 0.00667089 | 0.191635 | 0.94047 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_up | 72 | 0.0763584 | 3457 | 0.0616084 | 0.0147501 | -0.00617155 | 0.0356717 | 0.167024 | 0.94047 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | terminal_return | 72 | 0.0320133 | 3457 | 0.0169131 | 0.0151002 | -0.0133717 | 0.0435721 | 0.298574 | 0.960274 | 0.806443 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_down | 73 | -0.0177324 | 3471 | -0.0189629 | 0.00123058 | -0.00311536 | 0.00557651 | 0.578905 | 0.961138 | 0.891848 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_up | 73 | 0.0202256 | 3471 | 0.0170818 | 0.0031438 | -0.00162688 | 0.00791447 | 0.196493 | 0.94047 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | terminal_return | 73 | 0.00526393 | 3471 | 0.00114136 | 0.00412257 | -0.002309 | 0.0105541 | 0.208993 | 0.94047 | 0.795806 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_down | 73 | -0.0267274 | 3466 | -0.0271641 | 0.000436762 | -0.00505867 | 0.00593219 | 0.87621 | 0.961138 | 0.962812 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_up | 73 | 0.0253416 | 3466 | 0.0256643 | -0.000322701 | -0.00599281 | 0.0053474 | 0.911181 | 0.961138 | 0.971584 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | terminal_return | 73 | 0.00216416 | 3466 | 0.0023957 | -0.000231538 | -0.00691717 | 0.00645409 | 0.945882 | 0.961138 | 0.986326 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_down | 72 | -0.0367409 | 3457 | -0.0385378 | 0.00179692 | -0.00526872 | 0.00886257 | 0.618157 | 0.961138 | 0.904334 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_up | 72 | 0.0383916 | 3457 | 0.0387217 | -0.000330065 | -0.00897724 | 0.00831711 | 0.940363 | 0.961138 | 0.98489 | true |
| new_high_low_60_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | terminal_return | 72 | 0.0104808 | 3457 | 0.00454744 | 0.00593336 | -0.00574643 | 0.0176131 | 0.319404 | 0.960274 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_down | 98 | -0.0301444 | 3444 | -0.0217985 | -0.00834581 | -0.0161913 | -0.000500283 | 0.0370709 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_up | 98 | 0.019996 | 3444 | 0.0196133 | 0.000382635 | -0.00388857 | 0.00465384 | 0.860619 | 0.888836 | 0.961777 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 5 | terminal_return | 98 | -0.00232536 | 3444 | 0.0018219 | -0.00414727 | -0.0126714 | 0.00437687 | 0.340285 | 0.522877 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_down | 98 | -0.0479533 | 3439 | -0.0313022 | -0.0166512 | -0.0307234 | -0.00257897 | 0.020384 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_up | 98 | 0.0300338 | 3439 | 0.0292898 | 0.000743988 | -0.00602247 | 0.00751045 | 0.829372 | 0.870841 | 0.954346 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 10 | terminal_return | 98 | -0.00867624 | 3439 | 0.0037186 | -0.0123948 | -0.0285771 | 0.00378739 | 0.133286 | 0.335881 | 0.760687 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_down | 98 | -0.0662615 | 3429 | -0.0446118 | -0.0216497 | -0.0422441 | -0.00105527 | 0.0393573 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_up | 98 | 0.0490361 | 3429 | 0.0442171 | 0.00481901 | -0.00791134 | 0.0175494 | 0.458119 | 0.623381 | 0.855311 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 全A | 20 | terminal_return | 98 | -0.00279074 | 3429 | 0.00692162 | -0.00971236 | -0.0344064 | 0.0149817 | 0.440776 | 0.623381 | 0.84965 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_down | 98 | -0.0352287 | 3446 | -0.0264971 | -0.00873165 | -0.0177542 | 0.000290933 | 0.0578544 | 0.1969 | 0.550255 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_up | 98 | 0.0229276 | 3446 | 0.023489 | -0.000561409 | -0.00522073 | 0.00409791 | 0.813306 | 0.870841 | 0.945937 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 98 | -0.00279483 | 3446 | 0.00280868 | -0.00560351 | -0.0151617 | 0.0039547 | 0.250535 | 0.476563 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_down | 98 | -0.0572776 | 3441 | -0.0384733 | -0.0188043 | -0.0351935 | -0.00241515 | 0.0245233 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_up | 98 | 0.0355892 | 3441 | 0.0359493 | -0.000360062 | -0.00801457 | 0.00729445 | 0.926542 | 0.926542 | 0.980268 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 98 | -0.0101601 | 3441 | 0.00573437 | -0.0158945 | -0.0345212 | 0.00273226 | 0.0944261 | 0.258645 | 0.656242 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_down | 98 | -0.0775649 | 3431 | -0.055391 | -0.0221738 | -0.0455063 | 0.00115857 | 0.0625078 | 0.1969 | 0.555952 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_up | 98 | 0.0578785 | 3431 | 0.0549831 | 0.00289532 | -0.0115315 | 0.0173221 | 0.694059 | 0.799593 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 98 | -0.000303922 | 3431 | 0.0108691 | -0.011173 | -0.0399484 | 0.0176023 | 0.446634 | 0.623381 | 0.84965 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_down | 98 | -0.0364031 | 3446 | -0.0268904 | -0.00951264 | -0.0185806 | -0.000444662 | 0.0397718 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_up | 98 | 0.0225891 | 3446 | 0.0235896 | -0.00100051 | -0.00566766 | 0.00366663 | 0.67436 | 0.799593 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 98 | -0.00422545 | 3446 | 0.0021954 | -0.00642085 | -0.0160611 | 0.00321936 | 0.191738 | 0.424558 | 0.795806 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_down | 98 | -0.0594526 | 3441 | -0.0390858 | -0.0203668 | -0.0367468 | -0.00398688 | 0.0148071 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_up | 98 | 0.0346867 | 3441 | 0.0356392 | -0.000952493 | -0.00852296 | 0.00661797 | 0.805217 | 0.870841 | 0.943432 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 98 | -0.0132901 | 3441 | 0.00449535 | -0.0177855 | -0.0363822 | 0.0008113 | 0.0608625 | 0.1969 | 0.550255 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_down | 98 | -0.0808795 | 3431 | -0.056319 | -0.0245606 | -0.0480632 | -0.00105797 | 0.0405375 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_up | 98 | 0.0555631 | 3431 | 0.0540433 | 0.00151975 | -0.0120791 | 0.0151186 | 0.826618 | 0.870841 | 0.954346 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 98 | -0.00578758 | 3431 | 0.00832839 | -0.014116 | -0.0428142 | 0.0145823 | 0.335007 | 0.522877 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_down | 98 | -0.0282157 | 3446 | -0.0200573 | -0.0081584 | -0.015096 | -0.00122078 | 0.0211726 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_up | 98 | 0.0215454 | 3446 | 0.0195273 | 0.00201808 | -0.0031151 | 0.00715125 | 0.440966 | 0.623381 | 0.84965 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 98 | -0.000926751 | 3446 | 0.0015128 | -0.00243955 | -0.0108718 | 0.00599269 | 0.570679 | 0.73373 | 0.890044 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_down | 98 | -0.0431428 | 3441 | -0.0286626 | -0.0144802 | -0.0267003 | -0.00226021 | 0.020205 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_up | 98 | 0.0312684 | 3441 | 0.0291258 | 0.00214254 | -0.00582142 | 0.0101065 | 0.597987 | 0.753463 | 0.896134 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 98 | -0.0055078 | 3441 | 0.00306335 | -0.00857114 | -0.0238749 | 0.00673264 | 0.272322 | 0.476563 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_down | 98 | -0.0605524 | 3431 | -0.0403959 | -0.0201565 | -0.0381871 | -0.00212588 | 0.0284451 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_up | 98 | 0.0516641 | 3431 | 0.0437752 | 0.00788883 | -0.00762426 | 0.0234019 | 0.318904 | 0.522877 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 98 | 3.63313e-05 | 3431 | 0.0057459 | -0.00570957 | -0.0291088 | 0.0176897 | 0.63247 | 0.781287 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_down | 98 | -0.034308 | 3446 | -0.02424 | -0.0100681 | -0.0185405 | -0.00159567 | 0.0198516 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_up | 98 | 0.0211473 | 3446 | 0.022002 | -0.000854759 | -0.00517324 | 0.00346372 | 0.698058 | 0.799593 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 98 | -0.00380975 | 3446 | 0.00214316 | -0.00595291 | -0.0149088 | 0.00300297 | 0.192644 | 0.424558 | 0.795806 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_down | 98 | -0.054563 | 3441 | -0.0349227 | -0.0196403 | -0.0344931 | -0.00478754 | 0.00954833 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_up | 98 | 0.0316902 | 3441 | 0.0329368 | -0.00124659 | -0.00817258 | 0.00567939 | 0.724256 | 0.814788 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 98 | -0.0120726 | 3441 | 0.00436234 | -0.0164349 | -0.0333406 | 0.000470724 | 0.0567248 | 0.1969 | 0.550255 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_down | 98 | -0.0742121 | 3431 | -0.0498316 | -0.0243805 | -0.0465939 | -0.0021671 | 0.0314593 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_up | 98 | 0.050826 | 3431 | 0.0499048 | 0.000921172 | -0.0121907 | 0.014033 | 0.890478 | 0.90484 | 0.963092 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 98 | -0.00688537 | 3431 | 0.00815327 | -0.0150386 | -0.0417974 | 0.0117201 | 0.270664 | 0.476563 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_down | 98 | -0.0353041 | 3446 | -0.0275025 | -0.00780159 | -0.0174429 | 0.00183974 | 0.11274 | 0.295943 | 0.734754 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_up | 98 | 0.0239245 | 3446 | 0.026255 | -0.00233054 | -0.00704525 | 0.00238416 | 0.332617 | 0.522877 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 98 | -0.00210567 | 3446 | 0.00460477 | -0.00671045 | -0.0172025 | 0.00378164 | 0.210001 | 0.441003 | 0.795806 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_down | 98 | -0.0564464 | 3441 | -0.0399295 | -0.0165169 | -0.0346514 | 0.00161764 | 0.0742348 | 0.222705 | 0.636582 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_up | 98 | 0.0362568 | 3441 | 0.0403793 | -0.00412249 | -0.011747 | 0.00350199 | 0.289256 | 0.492517 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 98 | -0.00836208 | 3441 | 0.00926696 | -0.017629 | -0.037495 | 0.00223687 | 0.0819807 | 0.234763 | 0.636582 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_down | 98 | -0.0763161 | 3431 | -0.0574961 | -0.01882 | -0.044279 | 0.00663903 | 0.14737 | 0.357088 | 0.765477 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_up | 98 | 0.0589336 | 3431 | 0.0619943 | -0.00306066 | -0.0177535 | 0.0116321 | 0.683063 | 0.799593 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 98 | 0.000923896 | 3431 | 0.0176867 | -0.0167628 | -0.0455417 | 0.012016 | 0.253603 | 0.476563 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_down | 98 | -0.0270765 | 3446 | -0.0187061 | -0.00837042 | -0.0154038 | -0.00133707 | 0.0196689 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_up | 98 | 0.0196827 | 3446 | 0.0170744 | 0.00260829 | -0.00192462 | 0.0071412 | 0.259401 | 0.476563 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 98 | -0.00168079 | 3446 | 0.00130895 | -0.00298974 | -0.0110111 | 0.0050316 | 0.465062 | 0.623381 | 0.855442 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_down | 98 | -0.0423667 | 3441 | -0.0267219 | -0.0156449 | -0.0282781 | -0.00301161 | 0.0152142 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_up | 98 | 0.0285572 | 3441 | 0.025575 | 0.00298214 | -0.00445963 | 0.0104239 | 0.432201 | 0.623381 | 0.839958 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 98 | -0.00732043 | 3441 | 0.00266751 | -0.00998793 | -0.0251086 | 0.0051327 | 0.195431 | 0.424558 | 0.795806 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_down | 98 | -0.0583963 | 3431 | -0.0379329 | -0.0204634 | -0.0389138 | -0.002013 | 0.029717 | 0.159617 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_up | 98 | 0.046913 | 3431 | 0.0384808 | 0.00843224 | -0.00608884 | 0.0229533 | 0.255057 | 0.476563 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 98 | -0.00210053 | 3431 | 0.00486183 | -0.00696237 | -0.0299775 | 0.0160528 | 0.553232 | 0.726117 | 0.888798 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 5 | max_down | 98 | -0.029017 | 3444 | -0.0218306 | -0.00718641 | -0.0137991 | -0.00057371 | 0.0331677 | 0.149255 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 5 | max_up | 98 | 0.0206821 | 3444 | 0.0195938 | 0.00108831 | -0.00309464 | 0.00527127 | 0.610087 | 0.739144 | 0.902594 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 5 | terminal_return | 98 | -0.00244342 | 3444 | 0.00182526 | -0.00426869 | -0.0121798 | 0.00364241 | 0.290246 | 0.529049 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 10 | max_down | 98 | -0.0473732 | 3439 | -0.0313187 | -0.0160545 | -0.0289602 | -0.00314883 | 0.01476 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 10 | max_up | 98 | 0.0307001 | 3439 | 0.0292708 | 0.00142925 | -0.00533705 | 0.00819555 | 0.678866 | 0.763724 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 10 | terminal_return | 98 | -0.00496856 | 3439 | 0.00361295 | -0.00858151 | -0.023012 | 0.00584895 | 0.243787 | 0.524975 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 20 | max_down | 98 | -0.0657166 | 3429 | -0.0446274 | -0.0210893 | -0.0402872 | -0.00189129 | 0.0313116 | 0.149255 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 20 | max_up | 98 | 0.0485974 | 3429 | 0.0442296 | 0.0043678 | -0.00781817 | 0.0165538 | 0.482355 | 0.627399 | 0.865283 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 全A | 20 | terminal_return | 98 | -0.00326469 | 3429 | 0.00693516 | -0.0101998 | -0.0351315 | 0.0147318 | 0.422633 | 0.607158 | 0.839111 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 5 | max_down | 98 | -0.032253 | 3446 | -0.0265817 | -0.00567126 | -0.0132412 | 0.0018987 | 0.141998 | 0.399699 | 0.762443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 5 | max_up | 98 | 0.024516 | 3446 | 0.0234439 | 0.00107217 | -0.00382715 | 0.00597149 | 0.667976 | 0.763724 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 5 | terminal_return | 98 | -0.000943554 | 3446 | 0.00275603 | -0.00369959 | -0.0127702 | 0.00537098 | 0.424047 | 0.607158 | 0.839111 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 10 | max_down | 98 | -0.0558021 | 3441 | -0.0385153 | -0.0172868 | -0.0324177 | -0.00215583 | 0.0251392 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 10 | max_up | 98 | 0.0377777 | 3441 | 0.0358869 | 0.0018908 | -0.00631161 | 0.0100932 | 0.651402 | 0.759969 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 10 | terminal_return | 98 | -0.00559665 | 3441 | 0.0056044 | -0.0112011 | -0.0282559 | 0.00585377 | 0.198002 | 0.519755 | 0.795806 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 20 | max_down | 98 | -0.0763026 | 3431 | -0.0554271 | -0.0208755 | -0.0425931 | 0.000842096 | 0.0595648 | 0.234536 | 0.550255 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 20 | max_up | 98 | 0.0584526 | 3431 | 0.0549667 | 0.00348582 | -0.0108283 | 0.0178 | 0.633146 | 0.752607 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 国证2000 | 20 | terminal_return | 98 | 0.000503442 | 3431 | 0.010846 | -0.0103426 | -0.0395721 | 0.0188869 | 0.487977 | 0.627399 | 0.868025 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 5 | max_down | 98 | -0.0333666 | 3446 | -0.0269768 | -0.00638982 | -0.013959 | 0.0011794 | 0.0980051 | 0.343018 | 0.667494 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 5 | max_up | 98 | 0.02431 | 3446 | 0.0235407 | 0.000769306 | -0.00406163 | 0.00560024 | 0.754948 | 0.796266 | 0.913185 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 5 | terminal_return | 98 | -0.00221274 | 3446 | 0.00213816 | -0.0043509 | -0.0134005 | 0.00469872 | 0.346021 | 0.573667 | 0.807564 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 10 | max_down | 98 | -0.057865 | 3441 | -0.039131 | -0.018734 | -0.0338874 | -0.00358061 | 0.0153874 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 10 | max_up | 98 | 0.0369455 | 3441 | 0.0355749 | 0.00137058 | -0.00669423 | 0.0094354 | 0.739063 | 0.796266 | 0.912661 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 10 | terminal_return | 98 | -0.00823247 | 3441 | 0.00435131 | -0.0125838 | -0.0295458 | 0.00437819 | 0.145922 | 0.399699 | 0.765477 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 20 | max_down | 98 | -0.0793897 | 3431 | -0.0563615 | -0.0230282 | -0.0448674 | -0.001189 | 0.0387616 | 0.162799 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 20 | max_up | 98 | 0.0561388 | 3431 | 0.0540269 | 0.00211191 | -0.0113428 | 0.0155666 | 0.758348 | 0.796266 | 0.915833 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证1000 | 20 | terminal_return | 98 | -0.00513356 | 3431 | 0.00830971 | -0.0134433 | -0.0426287 | 0.0157421 | 0.366628 | 0.580164 | 0.813431 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 5 | max_down | 98 | -0.02813 | 3446 | -0.0200598 | -0.00807027 | -0.014063 | -0.00207755 | 0.00830311 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 5 | max_up | 98 | 0.0217622 | 3446 | 0.0195212 | 0.00224107 | -0.00264191 | 0.00712405 | 0.368358 | 0.580164 | 0.813431 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 5 | terminal_return | 98 | -0.00254434 | 3446 | 0.0015588 | -0.00410314 | -0.0117655 | 0.00355919 | 0.293916 | 0.529049 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 10 | max_down | 98 | -0.0430339 | 3441 | -0.0286657 | -0.0143682 | -0.0255041 | -0.00323231 | 0.0114417 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 10 | max_up | 98 | 0.0314518 | 3441 | 0.0291206 | 0.00233115 | -0.00518454 | 0.00984685 | 0.54323 | 0.683994 | 0.881341 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 10 | terminal_return | 98 | -0.00264816 | 3441 | 0.00298191 | -0.00563006 | -0.0191812 | 0.00792111 | 0.415465 | 0.607158 | 0.839111 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 20 | max_down | 98 | -0.0605618 | 3431 | -0.0403956 | -0.0201662 | -0.0370823 | -0.00324997 | 0.0194621 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 20 | max_up | 98 | 0.0506648 | 3431 | 0.0438038 | 0.00686106 | -0.00735544 | 0.0210776 | 0.344189 | 0.573667 | 0.807564 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 沪深300 | 20 | terminal_return | 98 | -0.00135319 | 3431 | 0.00578559 | -0.00713879 | -0.0307655 | 0.016488 | 0.553709 | 0.683994 | 0.888798 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 5 | max_down | 98 | -0.0322648 | 3446 | -0.0242981 | -0.00796668 | -0.0149777 | -0.000955664 | 0.0259362 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 5 | max_up | 98 | 0.0220577 | 3446 | 0.0219761 | 8.1577e-05 | -0.00434785 | 0.00451101 | 0.971205 | 0.971205 | 0.991753 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 5 | terminal_return | 98 | -0.0032831 | 3446 | 0.00212819 | -0.00541129 | -0.013832 | 0.00300941 | 0.207839 | 0.523753 | 0.795806 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 10 | max_down | 98 | -0.0536215 | 3441 | -0.0349495 | -0.018672 | -0.0324322 | -0.00491189 | 0.00782215 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 10 | max_up | 98 | 0.0332085 | 3441 | 0.0328935 | 0.000315003 | -0.0069875 | 0.00761751 | 0.932621 | 0.947664 | 0.980615 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 10 | terminal_return | 98 | -0.00751582 | 3441 | 0.00423257 | -0.0117484 | -0.0271211 | 0.00362436 | 0.134159 | 0.399699 | 0.760687 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 20 | max_down | 98 | -0.0732919 | 3431 | -0.0498579 | -0.023434 | -0.0440083 | -0.00285977 | 0.0255862 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 20 | max_up | 98 | 0.0510521 | 3431 | 0.0498984 | 0.00115372 | -0.0117961 | 0.0141036 | 0.861379 | 0.889621 | 0.961777 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 中证500 | 20 | terminal_return | 98 | -0.00712823 | 3431 | 0.00816021 | -0.0152884 | -0.0422648 | 0.0116879 | 0.266654 | 0.524975 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 5 | max_down | 98 | -0.032551 | 3446 | -0.0275808 | -0.0049702 | -0.0129578 | 0.00301744 | 0.222623 | 0.524975 | 0.797644 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 5 | max_up | 98 | 0.0243814 | 3446 | 0.026242 | -0.00186069 | -0.00682684 | 0.00310546 | 0.462728 | 0.627399 | 0.855311 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 5 | terminal_return | 98 | -0.00117487 | 3446 | 0.0045783 | -0.00575318 | -0.0154949 | 0.00398858 | 0.247062 | 0.524975 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 10 | max_down | 98 | -0.0556336 | 3441 | -0.0399527 | -0.015681 | -0.0325883 | 0.00122634 | 0.0690894 | 0.256037 | 0.607344 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 10 | max_up | 98 | 0.0374043 | 3441 | 0.0403466 | -0.00294233 | -0.0109567 | 0.00507207 | 0.471786 | 0.627399 | 0.8636 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 10 | terminal_return | 98 | -0.00557355 | 3441 | 0.00918754 | -0.0147611 | -0.0330873 | 0.00356507 | 0.114402 | 0.379333 | 0.738627 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 20 | max_down | 98 | -0.0755783 | 3431 | -0.0575172 | -0.0180611 | -0.0420549 | 0.00593264 | 0.140112 | 0.399699 | 0.762443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 20 | max_up | 98 | 0.0595101 | 3431 | 0.0619778 | -0.00246776 | -0.0167648 | 0.0118292 | 0.73513 | 0.796266 | 0.911661 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 微盘股 | 20 | terminal_return | 98 | 0.000719903 | 3431 | 0.0176925 | -0.0169726 | -0.0460104 | 0.0120652 | 0.251952 | 0.524975 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 5 | max_down | 98 | -0.0267086 | 3446 | -0.0187166 | -0.00799201 | -0.0139904 | -0.00199362 | 0.00901664 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 5 | max_up | 98 | 0.0193987 | 3446 | 0.0170825 | 0.00231621 | -0.00185271 | 0.00648513 | 0.276174 | 0.527241 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 5 | terminal_return | 98 | -0.00307059 | 3446 | 0.00134848 | -0.00441907 | -0.0117015 | 0.00286334 | 0.2343 | 0.524975 | 0.805223 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 10 | max_down | 98 | -0.0423175 | 3441 | -0.0267233 | -0.0155942 | -0.0270296 | -0.00415881 | 0.0075219 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 10 | max_up | 98 | 0.0285629 | 3441 | 0.0255749 | 0.00298801 | -0.0039441 | 0.00992012 | 0.398202 | 0.607158 | 0.829314 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 10 | terminal_return | 98 | -0.00440647 | 3441 | 0.00258452 | -0.00699099 | -0.0202875 | 0.0063055 | 0.302765 | 0.529838 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 20 | max_down | 98 | -0.0585698 | 3431 | -0.037928 | -0.0206418 | -0.0378655 | -0.00341818 | 0.018825 | 0.136165 | 0.512138 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 20 | max_up | 98 | 0.0460292 | 3431 | 0.038506 | 0.00752315 | -0.00562319 | 0.0206695 | 0.262018 | 0.524975 | 0.806443 | true |
| new_high_low_60_breadth_reversal_top | top | onset | 上证指数 | 20 | terminal_return | 98 | -0.00330787 | 3431 | 0.00489632 | -0.00820419 | -0.0309674 | 0.014559 | 0.479932 | 0.627399 | 0.865283 | true |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `new_high_low_120_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `new_high_low_120_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/onset`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `new_high_low_120_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：13 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `new_high_low_120_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/onset`：14 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `new_high_low_250_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `new_high_low_250_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/onset`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `new_high_low_250_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：7 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `new_high_low_250_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/onset`：4 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `new_high_low_60_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `new_high_low_60_breadth_reversal_bottom/bottom/new_high_low_period_decomposition_v1_20120104_20260814/onset`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `new_high_low_60_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/capped_confirmation`：16 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为负；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `new_high_low_60_breadth_reversal_top/top/new_high_low_period_decomposition_v1_20120104_20260814/onset`：15 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为负；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
