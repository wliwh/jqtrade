# 信号后 OHLC 结果评测

- 评测版本：`turnover_heat_v1_20120705_20260814__stage_d_v1`
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
| 上证指数 | capped_confirmation | 5 | 51 | 51 | 51 |
| 上证指数 | capped_confirmation | 10 | 51 | 51 | 51 |
| 上证指数 | capped_confirmation | 20 | 51 | 51 | 51 |
| 上证指数 | onset | 5 | 51 | 51 | 51 |
| 上证指数 | onset | 10 | 51 | 51 | 51 |
| 上证指数 | onset | 20 | 51 | 51 | 51 |
| 中证1000 | capped_confirmation | 5 | 51 | 51 | 51 |
| 中证1000 | capped_confirmation | 10 | 51 | 51 | 51 |
| 中证1000 | capped_confirmation | 20 | 51 | 51 | 51 |
| 中证1000 | onset | 5 | 51 | 51 | 51 |
| 中证1000 | onset | 10 | 51 | 51 | 51 |
| 中证1000 | onset | 20 | 51 | 51 | 51 |
| 中证500 | capped_confirmation | 5 | 51 | 51 | 51 |
| 中证500 | capped_confirmation | 10 | 51 | 51 | 51 |
| 中证500 | capped_confirmation | 20 | 51 | 51 | 51 |
| 中证500 | onset | 5 | 51 | 51 | 51 |
| 中证500 | onset | 10 | 51 | 51 | 51 |
| 中证500 | onset | 20 | 51 | 51 | 51 |
| 全A | capped_confirmation | 5 | 51 | 51 | 51 |
| 全A | capped_confirmation | 10 | 51 | 51 | 51 |
| 全A | capped_confirmation | 20 | 51 | 51 | 51 |
| 全A | onset | 5 | 51 | 51 | 51 |
| 全A | onset | 10 | 51 | 51 | 51 |
| 全A | onset | 20 | 51 | 51 | 51 |
| 国证2000 | capped_confirmation | 5 | 51 | 51 | 51 |
| 国证2000 | capped_confirmation | 10 | 51 | 51 | 51 |
| 国证2000 | capped_confirmation | 20 | 51 | 51 | 51 |
| 国证2000 | onset | 5 | 51 | 51 | 51 |
| 国证2000 | onset | 10 | 51 | 51 | 51 |
| 国证2000 | onset | 20 | 51 | 51 | 51 |
| 微盘股 | capped_confirmation | 5 | 51 | 51 | 51 |
| 微盘股 | capped_confirmation | 10 | 51 | 51 | 51 |
| 微盘股 | capped_confirmation | 20 | 51 | 51 | 51 |
| 微盘股 | onset | 5 | 51 | 51 | 51 |
| 微盘股 | onset | 10 | 51 | 51 | 51 |
| 微盘股 | onset | 20 | 51 | 51 | 51 |
| 沪深300 | capped_confirmation | 5 | 51 | 51 | 51 |
| 沪深300 | capped_confirmation | 10 | 51 | 51 | 51 |
| 沪深300 | capped_confirmation | 20 | 51 | 51 | 51 |
| 沪深300 | onset | 5 | 51 | 51 | 51 |
| 沪深300 | onset | 10 | 51 | 51 | 51 |
| 沪深300 | onset | 20 | 51 | 51 | 51 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 5 | max_down | 51 | -0.0261405 | 3371 | -0.0219756 | -0.00416491 | -0.0110008 | 0.00267099 | 0.232412 | 0.522926 | 0.59763 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 5 | max_up | 51 | 0.0229572 | 3371 | 0.0195447 | 0.00341257 | -0.00113154 | 0.00795668 | 0.141038 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 5 | terminal_return | 51 | 0.00517882 | 3371 | 0.00162347 | 0.00355535 | -0.0045767 | 0.0116874 | 0.391491 | 0.649051 | 0.703177 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 10 | max_down | 51 | -0.0383827 | 3366 | -0.0317439 | -0.00663874 | -0.0159648 | 0.00268732 | 0.162949 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 10 | max_up | 51 | 0.0330769 | 3366 | 0.0292674 | 0.00380957 | -0.00346084 | 0.01108 | 0.304417 | 0.581161 | 0.62451 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 10 | terminal_return | 51 | 0.00122314 | 3366 | 0.00339985 | -0.00217671 | -0.0151508 | 0.0107974 | 0.74228 | 0.812582 | 0.864766 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 20 | max_down | 51 | -0.0582438 | 3356 | -0.0451023 | -0.0131415 | -0.0322351 | 0.00595223 | 0.177339 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 20 | max_up | 51 | 0.0490785 | 3356 | 0.0443624 | 0.00471613 | -0.0074585 | 0.0168908 | 0.447702 | 0.692309 | 0.747048 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 全A | 20 | terminal_return | 51 | -0.00300114 | 3356 | 0.00694659 | -0.00994773 | -0.0307243 | 0.0108288 | 0.348018 | 0.592571 | 0.644857 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 5 | max_down | 51 | -0.0317278 | 3373 | -0.0266673 | -0.00506049 | -0.0129033 | 0.00278226 | 0.205987 | 0.506543 | 0.59763 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 5 | max_up | 51 | 0.0276253 | 3373 | 0.0233909 | 0.00423439 | -0.00221335 | 0.0106821 | 0.198032 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 51 | 0.00802496 | 3373 | 0.00254556 | 0.0054794 | -0.00483484 | 0.0157936 | 0.297763 | 0.581161 | 0.62451 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 10 | max_down | 51 | -0.0422754 | 3368 | -0.0390324 | -0.00324301 | -0.0140989 | 0.00761285 | 0.5582 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 10 | max_up | 51 | 0.0445605 | 3368 | 0.0358171 | 0.0087434 | -0.00131505 | 0.0188019 | 0.0884283 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 51 | 0.00875257 | 3368 | 0.00524577 | 0.00350679 | -0.0123141 | 0.0193277 | 0.663965 | 0.760542 | 0.82148 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 20 | max_down | 51 | -0.0625046 | 3358 | -0.0560965 | -0.00640812 | -0.0279818 | 0.0151656 | 0.560441 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 20 | max_up | 51 | 0.0700614 | 3358 | 0.0549885 | 0.0150729 | -0.00384357 | 0.0339894 | 0.118346 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 51 | 0.0125976 | 3358 | 0.0106239 | 0.00197368 | -0.0253052 | 0.0292526 | 0.88723 | 0.947381 | 0.972095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 5 | max_down | 51 | -0.0324724 | 3373 | -0.0270478 | -0.00542457 | -0.0134216 | 0.00257243 | 0.183677 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 5 | max_up | 51 | 0.0273587 | 3373 | 0.023466 | 0.00389262 | -0.00263386 | 0.0104191 | 0.242399 | 0.526591 | 0.610845 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 51 | 0.00703831 | 3373 | 0.00189298 | 0.00514533 | -0.00534709 | 0.0156378 | 0.336474 | 0.592571 | 0.642 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 10 | max_down | 51 | -0.0434833 | 3368 | -0.0396337 | -0.00384961 | -0.0149059 | 0.00720667 | 0.494961 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 10 | max_up | 51 | 0.0440659 | 3368 | 0.0354662 | 0.0085997 | -0.0017656 | 0.018965 | 0.10392 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 51 | 0.00682371 | 3368 | 0.00392967 | 0.00289403 | -0.013133 | 0.018921 | 0.723397 | 0.812582 | 0.859887 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 20 | max_down | 51 | -0.0643008 | 3358 | -0.0569765 | -0.00732425 | -0.0290384 | 0.0143899 | 0.508539 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 20 | max_up | 51 | 0.0678098 | 3358 | 0.0539512 | 0.0138585 | -0.00506255 | 0.0327797 | 0.151122 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 51 | 0.00879576 | 3358 | 0.00798853 | 0.000807228 | -0.0266662 | 0.0282807 | 0.954076 | 0.969464 | 0.983592 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 5 | max_down | 51 | -0.0246339 | 3373 | -0.0202377 | -0.0043962 | -0.0109643 | 0.00217191 | 0.189562 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 5 | max_up | 51 | 0.0240168 | 3373 | 0.0195196 | 0.00449723 | -0.000776288 | 0.00977074 | 0.0946273 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 51 | 0.00406337 | 3373 | 0.00138744 | 0.00267592 | -0.0058044 | 0.0111562 | 0.536267 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 10 | max_down | 51 | -0.0384168 | 3368 | -0.0290003 | -0.00941655 | -0.0189067 | 7.36016e-05 | 0.0517992 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 10 | max_up | 51 | 0.0322205 | 3368 | 0.0292083 | 0.00301225 | -0.00506873 | 0.0110932 | 0.46502 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 51 | -0.00269346 | 3368 | 0.00292365 | -0.00561711 | -0.0190318 | 0.0077976 | 0.411814 | 0.665237 | 0.718713 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 20 | max_down | 51 | -0.0581995 | 3358 | -0.0407511 | -0.0174485 | -0.0356464 | 0.000749439 | 0.060206 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 20 | max_up | 51 | 0.0437669 | 3358 | 0.0442056 | -0.000438697 | -0.0118061 | 0.0109287 | 0.939704 | 0.969464 | 0.983101 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 51 | -0.0107383 | 3358 | 0.00605429 | -0.0167926 | -0.035471 | 0.00188575 | 0.078049 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 5 | max_down | 51 | -0.0303263 | 3373 | -0.0243788 | -0.0059475 | -0.0137266 | 0.00183155 | 0.133997 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 5 | max_up | 51 | 0.0245915 | 3373 | 0.021882 | 0.00270946 | -0.00236254 | 0.00778145 | 0.295086 | 0.581161 | 0.62451 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 51 | 0.00491227 | 3373 | 0.00188281 | 0.00302946 | -0.00617362 | 0.0122325 | 0.518804 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 10 | max_down | 51 | -0.0425252 | 3368 | -0.0353641 | -0.0071611 | -0.0179716 | 0.00364937 | 0.194168 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 10 | max_up | 51 | 0.0369551 | 3368 | 0.0327948 | 0.00416033 | -0.00407968 | 0.0124003 | 0.322374 | 0.592571 | 0.634673 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 51 | 0.00157309 | 3368 | 0.00389714 | -0.00232406 | -0.0165076 | 0.0118595 | 0.748091 | 0.812582 | 0.864766 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 20 | max_down | 51 | -0.0626161 | 3358 | -0.0503255 | -0.0122906 | -0.033288 | 0.00870675 | 0.251271 | 0.527668 | 0.620786 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 20 | max_up | 51 | 0.0569749 | 3358 | 0.0498149 | 0.00715992 | -0.00758977 | 0.0219096 | 0.341381 | 0.592571 | 0.642 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 51 | 0.00138004 | 3358 | 0.00789392 | -0.00651388 | -0.0305542 | 0.0175264 | 0.595366 | 0.70683 | 0.773362 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 5 | max_down | 51 | -0.0305993 | 3373 | -0.0277047 | -0.0028946 | -0.0112514 | 0.0054622 | 0.497202 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 5 | max_up | 51 | 0.031354 | 3373 | 0.0261649 | 0.00518914 | -0.00135892 | 0.0117372 | 0.120366 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 51 | 0.0109295 | 3373 | 0.00432011 | 0.00660936 | -0.0037031 | 0.0169218 | 0.209049 | 0.506543 | 0.59763 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 10 | max_down | 51 | -0.0400865 | 3368 | -0.0405283 | 0.000441742 | -0.0109261 | 0.0118095 | 0.939289 | 0.969464 | 0.983101 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 10 | max_up | 51 | 0.0484198 | 3368 | 0.0402938 | 0.00812599 | -0.0017573 | 0.0180093 | 0.10707 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 51 | 0.01303 | 3368 | 0.00877345 | 0.00425658 | -0.0119118 | 0.020425 | 0.605854 | 0.70683 | 0.778956 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 20 | max_down | 51 | -0.0642067 | 3358 | -0.0581531 | -0.00605367 | -0.0282775 | 0.0161702 | 0.593414 | 0.70683 | 0.773362 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 20 | max_up | 51 | 0.075026 | 3358 | 0.0620355 | 0.0129904 | -0.00438123 | 0.0303621 | 0.142736 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 51 | 0.0178884 | 3358 | 0.0174848 | 0.000403556 | -0.0256557 | 0.0264628 | 0.975786 | 0.975786 | 0.983592 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 5 | max_down | 51 | -0.0234419 | 3373 | -0.0189029 | -0.004539 | -0.0107605 | 0.0016825 | 0.152731 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 5 | max_up | 51 | 0.0219571 | 3373 | 0.0171358 | 0.00482129 | -0.000204587 | 0.00984716 | 0.0600788 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 51 | 0.00412672 | 3373 | 0.00122544 | 0.00290128 | -0.00507273 | 0.0108753 | 0.475765 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 10 | max_down | 51 | -0.0359911 | 3368 | -0.0270791 | -0.008912 | -0.0175877 | -0.000236266 | 0.044075 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 10 | max_up | 51 | 0.0305424 | 3368 | 0.0257296 | 0.00481274 | -0.00296882 | 0.0125943 | 0.225428 | 0.522926 | 0.59763 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 51 | -0.00157577 | 3368 | 0.00258765 | -0.00416342 | -0.0166613 | 0.00833445 | 0.513798 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 20 | max_down | 51 | -0.0540469 | 3358 | -0.0382591 | -0.0157878 | -0.0329806 | 0.001405 | 0.0718877 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 20 | max_up | 51 | 0.0423918 | 3358 | 0.0389712 | 0.00342064 | -0.00791833 | 0.0147596 | 0.554336 | 0.692309 | 0.74872 | true |
| all_a_turnover_heat_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 51 | -0.00726688 | 3358 | 0.00526512 | -0.012532 | -0.0304345 | 0.0053705 | 0.170055 | 0.506543 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 全A | 5 | max_down | 51 | -0.0271888 | 3371 | -0.0219598 | -0.00522901 | -0.0130069 | 0.00254888 | 0.187607 | 0.66758 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 全A | 5 | max_up | 51 | 0.0223174 | 3371 | 0.0195544 | 0.00276303 | -0.00214837 | 0.00767444 | 0.27018 | 0.66758 | 0.62451 | true |
| all_a_turnover_heat_top | top | onset | 全A | 5 | terminal_return | 51 | 0.00187108 | 3371 | 0.00167351 | 0.000197573 | -0.00960751 | 0.0100027 | 0.968496 | 0.984117 | 0.983592 | true |
| all_a_turnover_heat_top | top | onset | 全A | 10 | max_down | 51 | -0.0376408 | 3366 | -0.0317552 | -0.00588568 | -0.0155017 | 0.00373036 | 0.230273 | 0.66758 | 0.59763 | true |
| all_a_turnover_heat_top | top | onset | 全A | 10 | max_up | 51 | 0.033169 | 3366 | 0.029266 | 0.00390306 | -0.00378321 | 0.0115893 | 0.319599 | 0.667746 | 0.634673 | true |
| all_a_turnover_heat_top | top | onset | 全A | 10 | terminal_return | 51 | 0.00230701 | 3366 | 0.00338343 | -0.00107642 | -0.0139698 | 0.0118169 | 0.87002 | 0.983869 | 0.966609 | true |
| all_a_turnover_heat_top | top | onset | 全A | 20 | max_down | 51 | -0.0589813 | 3356 | -0.0450911 | -0.0138902 | -0.0336731 | 0.00589268 | 0.168766 | 0.664516 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 全A | 20 | max_up | 51 | 0.0486313 | 3356 | 0.0443692 | 0.00426215 | -0.00821345 | 0.0167378 | 0.503105 | 0.793403 | 0.74872 | true |
| all_a_turnover_heat_top | top | onset | 全A | 20 | terminal_return | 51 | -0.00508536 | 3356 | 0.00697826 | -0.0120636 | -0.0348272 | 0.0106999 | 0.29894 | 0.66758 | 0.62451 | true |
| all_a_turnover_heat_top | top | onset | 国证2000 | 5 | max_down | 51 | -0.0304376 | 3373 | -0.0266868 | -0.00375078 | -0.0124163 | 0.00491475 | 0.396235 | 0.756448 | 0.703177 | true |
| all_a_turnover_heat_top | top | onset | 国证2000 | 5 | max_up | 51 | 0.0278003 | 3373 | 0.0233883 | 0.00441202 | -0.00258699 | 0.011411 | 0.216629 | 0.66758 | 0.59763 | true |
| all_a_turnover_heat_top | top | onset | 国证2000 | 5 | terminal_return | 51 | 0.00446272 | 3373 | 0.00259942 | 0.0018633 | -0.00948506 | 0.0132117 | 0.747593 | 0.918359 | 0.864766 | true |
| all_a_turnover_heat_top | top | onset | 国证2000 | 10 | max_down | 51 | -0.0413318 | 3368 | -0.0390467 | -0.00228513 | -0.0130376 | 0.00846729 | 0.677012 | 0.872949 | 0.822587 | true |
| all_a_turnover_heat_top | top | onset | 国证2000 | 10 | max_up | 51 | 0.0454307 | 3368 | 0.0358039 | 0.00962684 | -0.00167207 | 0.0209258 | 0.0949295 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 国证2000 | 10 | terminal_return | 51 | 0.0106377 | 3368 | 0.00521723 | 0.00542049 | -0.0104696 | 0.0213106 | 0.503748 | 0.793403 | 0.74872 | true |
| all_a_turnover_heat_top | top | onset | 国证2000 | 20 | max_down | 51 | -0.0628088 | 3358 | -0.0560919 | -0.00671692 | -0.028635 | 0.0152011 | 0.54807 | 0.802986 | 0.74872 | true |
| all_a_turnover_heat_top | top | onset | 国证2000 | 20 | max_up | 51 | 0.0708217 | 3358 | 0.054977 | 0.0158447 | -0.00365165 | 0.035341 | 0.111184 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 国证2000 | 20 | terminal_return | 51 | 0.0117316 | 3358 | 0.0106371 | 0.00109455 | -0.0276391 | 0.0298282 | 0.940484 | 0.984117 | 0.983101 | true |
| all_a_turnover_heat_top | top | onset | 中证1000 | 5 | max_down | 51 | -0.0316029 | 3373 | -0.027061 | -0.00454194 | -0.0132618 | 0.00417796 | 0.307299 | 0.66758 | 0.62451 | true |
| all_a_turnover_heat_top | top | onset | 中证1000 | 5 | max_up | 51 | 0.0275863 | 3373 | 0.0234626 | 0.00412372 | -0.00304931 | 0.0112968 | 0.259832 | 0.66758 | 0.62451 | true |
| all_a_turnover_heat_top | top | onset | 中证1000 | 5 | terminal_return | 51 | 0.0031331 | 3373 | 0.00195203 | 0.00118107 | -0.0104188 | 0.0127809 | 0.841822 | 0.982126 | 0.94705 | true |
| all_a_turnover_heat_top | top | onset | 中证1000 | 10 | max_down | 51 | -0.0428798 | 3368 | -0.0396428 | -0.00323693 | -0.0142486 | 0.00777469 | 0.564511 | 0.808277 | 0.74872 | true |
| all_a_turnover_heat_top | top | onset | 中证1000 | 10 | max_up | 51 | 0.0450403 | 3368 | 0.0354514 | 0.00958884 | -0.00228408 | 0.0214618 | 0.113435 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 中证1000 | 10 | terminal_return | 51 | 0.00888551 | 3368 | 0.00389845 | 0.00498706 | -0.0112131 | 0.0211872 | 0.546264 | 0.802986 | 0.74872 | true |
| all_a_turnover_heat_top | top | onset | 中证1000 | 20 | max_down | 51 | -0.06482 | 3358 | -0.0569686 | -0.00785139 | -0.0297324 | 0.0140296 | 0.481875 | 0.793403 | 0.74872 | true |
| all_a_turnover_heat_top | top | onset | 中证1000 | 20 | max_up | 51 | 0.068575 | 3358 | 0.0539396 | 0.0146354 | -0.00493305 | 0.0342038 | 0.142676 | 0.631352 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 中证1000 | 20 | terminal_return | 51 | 0.00798146 | 3358 | 0.0080009 | -1.94432e-05 | -0.0288013 | 0.0287624 | 0.998944 | 0.998944 | 0.998944 | true |
| all_a_turnover_heat_top | top | onset | 沪深300 | 5 | max_down | 51 | -0.0267327 | 3373 | -0.020206 | -0.00652678 | -0.0141775 | 0.00112395 | 0.0945121 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 沪深300 | 5 | max_up | 51 | 0.0217765 | 3373 | 0.0195534 | 0.00222305 | -0.00344307 | 0.00788917 | 0.4419 | 0.777334 | 0.747048 | true |
| all_a_turnover_heat_top | top | onset | 沪深300 | 5 | terminal_return | 51 | 0.000621304 | 3373 | 0.00143949 | -0.000818181 | -0.0109754 | 0.00933906 | 0.874551 | 0.983869 | 0.966609 | true |
| all_a_turnover_heat_top | top | onset | 沪深300 | 10 | max_down | 51 | -0.0375999 | 3368 | -0.0290127 | -0.00858725 | -0.0183514 | 0.00117686 | 0.0847506 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 沪深300 | 10 | max_up | 51 | 0.0312417 | 3368 | 0.0292231 | 0.00201859 | -0.00650832 | 0.0105455 | 0.642652 | 0.872949 | 0.817921 | true |
| all_a_turnover_heat_top | top | onset | 沪深300 | 10 | terminal_return | 51 | -0.0023176 | 3368 | 0.00291796 | -0.00523556 | -0.0190175 | 0.00854642 | 0.45653 | 0.777334 | 0.747048 | true |
| all_a_turnover_heat_top | top | onset | 沪深300 | 20 | max_down | 51 | -0.059343 | 3358 | -0.0407337 | -0.0186093 | -0.0376429 | 0.000424337 | 0.0553265 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 沪深300 | 20 | max_up | 51 | 0.0420504 | 3358 | 0.0442317 | -0.00218126 | -0.0137319 | 0.00936938 | 0.711284 | 0.896218 | 0.853541 | true |
| all_a_turnover_heat_top | top | onset | 沪深300 | 20 | terminal_return | 51 | -0.0139991 | 3358 | 0.00610382 | -0.0201029 | -0.0410025 | 0.000796746 | 0.0593921 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 中证500 | 5 | max_down | 51 | -0.0299311 | 3373 | -0.0243848 | -0.00554633 | -0.0139886 | 0.00289595 | 0.197863 | 0.66758 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 中证500 | 5 | max_up | 51 | 0.0244673 | 3373 | 0.0218839 | 0.00258338 | -0.0030432 | 0.00820996 | 0.368167 | 0.724829 | 0.672305 | true |
| all_a_turnover_heat_top | top | onset | 中证500 | 5 | terminal_return | 51 | 0.0014727 | 3373 | 0.00193482 | -0.000462117 | -0.0110337 | 0.0101095 | 0.931723 | 0.984117 | 0.983101 | true |
| all_a_turnover_heat_top | top | onset | 中证500 | 10 | max_down | 51 | -0.0412909 | 3368 | -0.0353828 | -0.00590812 | -0.0166775 | 0.0048613 | 0.282259 | 0.66758 | 0.62451 | true |
| all_a_turnover_heat_top | top | onset | 中证500 | 10 | max_up | 51 | 0.0376599 | 3368 | 0.0327841 | 0.00487574 | -0.00422565 | 0.0139771 | 0.293719 | 0.66758 | 0.62451 | true |
| all_a_turnover_heat_top | top | onset | 中证500 | 10 | terminal_return | 51 | 0.00301499 | 3368 | 0.00387531 | -0.000860317 | -0.0149151 | 0.0131945 | 0.904503 | 0.984117 | 0.982478 | true |
| all_a_turnover_heat_top | top | onset | 中证500 | 20 | max_down | 51 | -0.0635193 | 3358 | -0.0503117 | -0.0132076 | -0.0346115 | 0.00819639 | 0.226493 | 0.66758 | 0.59763 | true |
| all_a_turnover_heat_top | top | onset | 中证500 | 20 | max_up | 51 | 0.0574134 | 3358 | 0.0498083 | 0.00760511 | -0.00765195 | 0.0228622 | 0.328574 | 0.667746 | 0.636927 | true |
| all_a_turnover_heat_top | top | onset | 中证500 | 20 | terminal_return | 51 | -7.94823e-05 | 3358 | 0.00791609 | -0.00799557 | -0.0335082 | 0.017517 | 0.539045 | 0.802986 | 0.74872 | true |
| all_a_turnover_heat_top | top | onset | 微盘股 | 5 | max_down | 51 | -0.0289214 | 3373 | -0.0277301 | -0.00119131 | -0.00992083 | 0.00753822 | 0.789101 | 0.937988 | 0.895736 | true |
| all_a_turnover_heat_top | top | onset | 微盘股 | 5 | max_up | 51 | 0.0318679 | 3373 | 0.0261571 | 0.00571075 | -0.00207088 | 0.0134924 | 0.150322 | 0.631352 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 微盘股 | 5 | terminal_return | 51 | 0.00916177 | 3373 | 0.00434683 | 0.00481493 | -0.00679749 | 0.0164274 | 0.416397 | 0.77156 | 0.718713 | true |
| all_a_turnover_heat_top | top | onset | 微盘股 | 10 | max_down | 51 | -0.0380328 | 3368 | -0.0405594 | 0.00252659 | -0.00890993 | 0.0139631 | 0.665008 | 0.872949 | 0.82148 | true |
| all_a_turnover_heat_top | top | onset | 微盘股 | 10 | max_up | 51 | 0.0494334 | 3368 | 0.0402784 | 0.00915498 | -0.00201063 | 0.0203206 | 0.108042 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 微盘股 | 10 | terminal_return | 51 | 0.0146643 | 3368 | 0.0087487 | 0.00591559 | -0.00953927 | 0.0213704 | 0.453122 | 0.777334 | 0.747048 | true |
| all_a_turnover_heat_top | top | onset | 微盘股 | 20 | max_down | 51 | -0.0630678 | 3358 | -0.0581704 | -0.00489745 | -0.0280901 | 0.0182952 | 0.67896 | 0.872949 | 0.822587 | true |
| all_a_turnover_heat_top | top | onset | 微盘股 | 20 | max_up | 51 | 0.0768528 | 3358 | 0.0620078 | 0.014845 | -0.00377248 | 0.0334626 | 0.118089 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 微盘股 | 20 | terminal_return | 51 | 0.0181421 | 3358 | 0.0174809 | 0.000661162 | -0.0291627 | 0.030485 | 0.965342 | 0.984117 | 0.983592 | true |
| all_a_turnover_heat_top | top | onset | 上证指数 | 5 | max_down | 51 | -0.0248798 | 3373 | -0.0188812 | -0.00599858 | -0.0132389 | 0.00124177 | 0.104409 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 上证指数 | 5 | max_up | 51 | 0.0201968 | 3373 | 0.0171624 | 0.00303438 | -0.00227713 | 0.00834589 | 0.262835 | 0.66758 | 0.62451 | true |
| all_a_turnover_heat_top | top | onset | 上证指数 | 5 | terminal_return | 51 | 0.000931998 | 3373 | 0.00127375 | -0.000341748 | -0.0098928 | 0.0092093 | 0.944089 | 0.984117 | 0.983101 | true |
| all_a_turnover_heat_top | top | onset | 上证指数 | 10 | max_down | 51 | -0.0346356 | 3368 | -0.0270996 | -0.00753592 | -0.0167835 | 0.00171165 | 0.110217 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 上证指数 | 10 | max_up | 51 | 0.0300614 | 3368 | 0.0257369 | 0.00432446 | -0.00384356 | 0.0124925 | 0.29941 | 0.66758 | 0.62451 | true |
| all_a_turnover_heat_top | top | onset | 上证指数 | 10 | terminal_return | 51 | -0.000322964 | 3368 | 0.00256868 | -0.00289164 | -0.0155124 | 0.00972914 | 0.653381 | 0.872949 | 0.82148 | true |
| all_a_turnover_heat_top | top | onset | 上证指数 | 20 | max_down | 51 | -0.0546364 | 3358 | -0.0382502 | -0.0163862 | -0.0344352 | 0.00166284 | 0.0751697 | 0.619969 | 0.594095 | true |
| all_a_turnover_heat_top | top | onset | 上证指数 | 20 | max_up | 51 | 0.0408044 | 3358 | 0.0389953 | 0.00180914 | -0.00970005 | 0.0133183 | 0.758011 | 0.918359 | 0.868267 | true |
| all_a_turnover_heat_top | top | onset | 上证指数 | 20 | terminal_return | 51 | -0.0101285 | 3358 | 0.00530858 | -0.0154371 | -0.0354738 | 0.00459961 | 0.131027 | 0.631352 | 0.594095 | true |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `all_a_turnover_heat_top/top/turnover_heat_v1_20120705_20260814/capped_confirmation`：1 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `all_a_turnover_heat_top/top/turnover_heat_v1_20120705_20260814/onset`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
