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
| 上证指数 | capped_confirmation | 5 | 101 | 101 | 101 |
| 上证指数 | capped_confirmation | 10 | 101 | 101 | 101 |
| 上证指数 | capped_confirmation | 20 | 101 | 101 | 100 |
| 上证指数 | onset | 5 | 101 | 101 | 101 |
| 上证指数 | onset | 10 | 101 | 101 | 101 |
| 上证指数 | onset | 20 | 101 | 101 | 100 |
| 中证1000 | capped_confirmation | 5 | 101 | 101 | 101 |
| 中证1000 | capped_confirmation | 10 | 101 | 101 | 101 |
| 中证1000 | capped_confirmation | 20 | 101 | 101 | 100 |
| 中证1000 | onset | 5 | 101 | 101 | 101 |
| 中证1000 | onset | 10 | 101 | 101 | 101 |
| 中证1000 | onset | 20 | 101 | 101 | 100 |
| 中证500 | capped_confirmation | 5 | 101 | 101 | 101 |
| 中证500 | capped_confirmation | 10 | 101 | 101 | 101 |
| 中证500 | capped_confirmation | 20 | 101 | 101 | 100 |
| 中证500 | onset | 5 | 101 | 101 | 101 |
| 中证500 | onset | 10 | 101 | 101 | 101 |
| 中证500 | onset | 20 | 101 | 101 | 100 |
| 全A | capped_confirmation | 5 | 101 | 101 | 101 |
| 全A | capped_confirmation | 10 | 101 | 101 | 101 |
| 全A | capped_confirmation | 20 | 101 | 101 | 100 |
| 全A | onset | 5 | 101 | 101 | 101 |
| 全A | onset | 10 | 101 | 101 | 101 |
| 全A | onset | 20 | 101 | 101 | 100 |
| 国证2000 | capped_confirmation | 5 | 101 | 101 | 101 |
| 国证2000 | capped_confirmation | 10 | 101 | 101 | 101 |
| 国证2000 | capped_confirmation | 20 | 101 | 101 | 100 |
| 国证2000 | onset | 5 | 101 | 101 | 101 |
| 国证2000 | onset | 10 | 101 | 101 | 101 |
| 国证2000 | onset | 20 | 101 | 101 | 100 |
| 微盘股 | capped_confirmation | 5 | 101 | 101 | 101 |
| 微盘股 | capped_confirmation | 10 | 101 | 101 | 101 |
| 微盘股 | capped_confirmation | 20 | 101 | 101 | 100 |
| 微盘股 | onset | 5 | 101 | 101 | 101 |
| 微盘股 | onset | 10 | 101 | 101 | 101 |
| 微盘股 | onset | 20 | 101 | 101 | 100 |
| 沪深300 | capped_confirmation | 5 | 101 | 101 | 101 |
| 沪深300 | capped_confirmation | 10 | 101 | 101 | 101 |
| 沪深300 | capped_confirmation | 20 | 101 | 101 | 100 |
| 沪深300 | onset | 5 | 101 | 101 | 101 |
| 沪深300 | onset | 10 | 101 | 101 | 101 |
| 沪深300 | onset | 20 | 101 | 101 | 100 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 5 | max_down | 49 | -0.0233436 | 3493 | -0.022011 | -0.00133259 | -0.00718562 | 0.00452044 | 0.655421 | 0.997887 | 0.860241 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 5 | max_up | 49 | 0.0214148 | 3493 | 0.0195988 | 0.00181597 | -0.00381309 | 0.00744503 | 0.527185 | 0.997887 | 0.80078 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 5 | terminal_return | 49 | 0.00103144 | 3493 | 0.00171664 | -0.000685201 | -0.00898363 | 0.00761323 | 0.871434 | 0.997887 | 0.993923 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 10 | max_down | 49 | -0.0335272 | 3488 | -0.0317387 | -0.00178852 | -0.00922909 | 0.00565206 | 0.637546 | 0.997887 | 0.845588 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 10 | max_up | 49 | 0.0293006 | 3488 | 0.0293106 | -9.99878e-06 | -0.00740873 | 0.00738873 | 0.997887 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 10 | terminal_return | 49 | 0.00335304 | 3488 | 0.00337549 | -2.24525e-05 | -0.0108577 | 0.0108128 | 0.996759 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 20 | max_down | 48 | -0.0458929 | 3479 | -0.045204 | -0.000688884 | -0.0114657 | 0.010088 | 0.900295 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 20 | max_up | 48 | 0.0531807 | 3479 | 0.0442292 | 0.00895149 | -0.00700918 | 0.0249121 | 0.271655 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 全A | 20 | terminal_return | 48 | 0.022674 | 3479 | 0.0064307 | 0.0162433 | -0.00333364 | 0.0358202 | 0.103898 | 0.997887 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_down | 49 | -0.0309357 | 3495 | -0.0266797 | -0.00425603 | -0.0136265 | 0.00511449 | 0.373348 | 0.997887 | 0.712777 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_up | 49 | 0.0256217 | 3495 | 0.0234434 | 0.00217834 | -0.00558184 | 0.00993852 | 0.582192 | 0.997887 | 0.819622 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 49 | -0.000733927 | 3495 | 0.00270123 | -0.00343515 | -0.0155884 | 0.00871806 | 0.579578 | 0.997887 | 0.819622 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_down | 49 | -0.0445521 | 3490 | -0.038916 | -0.0056361 | -0.0159586 | 0.00468643 | 0.284547 | 0.997887 | 0.682913 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_up | 49 | 0.0360955 | 3490 | 0.0359371 | 0.000158457 | -0.0102604 | 0.0105773 | 0.976219 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 49 | 0.00407821 | 3490 | 0.0053113 | -0.0012331 | -0.016661 | 0.0141948 | 0.875516 | 0.997887 | 0.993923 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_down | 48 | -0.0632707 | 3481 | -0.0559066 | -0.0073641 | -0.0234352 | 0.008707 | 0.369126 | 0.997887 | 0.712777 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_up | 48 | 0.0644824 | 3481 | 0.0549337 | 0.00954877 | -0.0115765 | 0.030674 | 0.375652 | 0.997887 | 0.712777 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 48 | 0.0265687 | 3481 | 0.0103381 | 0.0162307 | -0.010693 | 0.0431543 | 0.237378 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_down | 49 | -0.0302234 | 3495 | -0.0271104 | -0.003113 | -0.0118223 | 0.00559629 | 0.483571 | 0.997887 | 0.794855 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_up | 49 | 0.0256565 | 3495 | 0.0235326 | 0.00212386 | -0.00536099 | 0.0096087 | 0.578103 | 0.997887 | 0.819622 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 49 | 0.000347943 | 3495 | 0.00204126 | -0.00169332 | -0.0133684 | 0.00998177 | 0.776202 | 0.997887 | 0.922655 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_down | 49 | -0.0439053 | 3490 | -0.03959 | -0.00431534 | -0.0145442 | 0.00591355 | 0.408305 | 0.997887 | 0.746039 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_up | 49 | 0.0367768 | 3490 | 0.0355965 | 0.00118035 | -0.00889115 | 0.0112519 | 0.81832 | 0.997887 | 0.968153 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 49 | 0.0044903 | 3490 | 0.003996 | 0.000494291 | -0.014599 | 0.0155876 | 0.94882 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_down | 48 | -0.0633161 | 3481 | -0.0569139 | -0.00640222 | -0.021779 | 0.00897454 | 0.414466 | 0.997887 | 0.746039 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_up | 48 | 0.0643729 | 3481 | 0.0539437 | 0.0104292 | -0.00988714 | 0.0307455 | 0.314345 | 0.997887 | 0.683599 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 48 | 0.0255107 | 3481 | 0.00769406 | 0.0178166 | -0.0084293 | 0.0440625 | 0.183349 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_down | 49 | -0.0201321 | 3495 | -0.020285 | 0.000152902 | -0.00487403 | 0.00517984 | 0.952461 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_up | 49 | 0.0219109 | 3495 | 0.0195505 | 0.00236039 | -0.00352272 | 0.0082435 | 0.431644 | 0.997887 | 0.760659 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 49 | 0.00200375 | 3495 | 0.00143751 | 0.000566237 | -0.00694265 | 0.00807513 | 0.8825 | 0.997887 | 0.997264 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_down | 49 | -0.0286199 | 3490 | -0.0290698 | 0.000449857 | -0.00590928 | 0.00680899 | 0.889723 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_up | 49 | 0.0295999 | 3490 | 0.0291793 | 0.000420593 | -0.00685696 | 0.00769815 | 0.909813 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 49 | 0.00363717 | 3490 | 0.00281461 | 0.000822554 | -0.00845931 | 0.0101044 | 0.862106 | 0.997887 | 0.993923 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_down | 48 | -0.0377149 | 3481 | -0.0410003 | 0.0032854 | -0.00555983 | 0.0121306 | 0.46661 | 0.997887 | 0.794498 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_up | 48 | 0.0533925 | 3481 | 0.0438647 | 0.00952774 | -0.00508768 | 0.0241432 | 0.201349 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 48 | 0.0211394 | 3481 | 0.0053729 | 0.0157665 | -0.00199387 | 0.0335269 | 0.081866 | 0.997887 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 5 | max_down | 49 | -0.0263014 | 3495 | -0.0244934 | -0.00180802 | -0.00904737 | 0.00543133 | 0.624482 | 0.997887 | 0.840471 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 5 | max_up | 49 | 0.0260815 | 3495 | 0.0219208 | 0.00416068 | -0.00272108 | 0.0110424 | 0.236015 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 49 | 0.00290259 | 3495 | 0.0019656 | 0.000936997 | -0.0095701 | 0.0114441 | 0.861246 | 0.997887 | 0.993923 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 10 | max_down | 49 | -0.0386138 | 3490 | -0.0354224 | -0.00319137 | -0.0124445 | 0.00606173 | 0.499041 | 0.997887 | 0.794855 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 10 | max_up | 49 | 0.0355172 | 3490 | 0.0328656 | 0.00265163 | -0.00674363 | 0.0120469 | 0.580146 | 0.997887 | 0.819622 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 49 | 0.00402672 | 3490 | 0.00390556 | 0.000121162 | -0.0129937 | 0.013236 | 0.985553 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 20 | max_down | 48 | -0.0538438 | 3481 | -0.0504626 | -0.00338113 | -0.0163758 | 0.00961354 | 0.610066 | 0.997887 | 0.835525 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 20 | max_up | 48 | 0.0628454 | 3481 | 0.0497523 | 0.013093 | -0.00516011 | 0.0313462 | 0.15975 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 48 | 0.027834 | 3481 | 0.00745851 | 0.0203755 | -0.00197074 | 0.0427218 | 0.0739139 | 0.997887 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_down | 49 | -0.0380908 | 3495 | -0.0275728 | -0.0105179 | -0.0251185 | 0.00408263 | 0.157967 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_up | 49 | 0.0342909 | 3495 | 0.026077 | 0.00821389 | -0.00399592 | 0.0204237 | 0.18732 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 49 | -0.00242861 | 3495 | 0.00451522 | -0.00694383 | -0.0279905 | 0.0141028 | 0.517856 | 0.997887 | 0.80078 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_down | 49 | -0.0548521 | 3490 | -0.0401838 | -0.0146683 | -0.0302913 | 0.000954769 | 0.0657361 | 0.997887 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_up | 49 | 0.0461729 | 3490 | 0.0401822 | 0.00599073 | -0.0105841 | 0.0225656 | 0.478689 | 0.997887 | 0.794855 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 49 | 0.00515118 | 3490 | 0.00882972 | -0.00367854 | -0.0254526 | 0.0180955 | 0.74055 | 0.997887 | 0.914104 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_down | 48 | -0.0774132 | 3481 | -0.0577513 | -0.0196619 | -0.045176 | 0.00585219 | 0.130933 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_up | 48 | 0.0785859 | 3481 | 0.0616793 | 0.0169066 | -0.0105636 | 0.0443769 | 0.227707 | 0.997887 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 48 | 0.0300316 | 3481 | 0.0170446 | 0.012987 | -0.0239134 | 0.0498875 | 0.490309 | 0.997887 | 0.794855 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_down | 49 | -0.0191599 | 3495 | -0.0189345 | -0.000225394 | -0.00488712 | 0.00443633 | 0.924501 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_up | 49 | 0.0179679 | 3495 | 0.017135 | 0.00083291 | -0.00415842 | 0.00582424 | 0.743617 | 0.997887 | 0.914104 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 49 | 0.00140904 | 3495 | 0.00122372 | 0.00018532 | -0.00644982 | 0.00682045 | 0.956343 | 0.997887 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_down | 49 | -0.0262764 | 3490 | -0.0271675 | 0.000891043 | -0.00515058 | 0.00693266 | 0.772529 | 0.997887 | 0.922655 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_up | 49 | 0.0244041 | 3490 | 0.0256752 | -0.00127114 | -0.00757064 | 0.00502837 | 0.692477 | 0.997887 | 0.887666 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 49 | 0.00322892 | 3490 | 0.00237916 | 0.000849757 | -0.0076647 | 0.00936422 | 0.844914 | 0.997887 | 0.990318 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_down | 48 | -0.0352981 | 3481 | -0.0385453 | 0.00324721 | -0.00508944 | 0.0115839 | 0.445201 | 0.997887 | 0.769386 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_up | 48 | 0.0448609 | 3481 | 0.0386302 | 0.00623065 | -0.00701186 | 0.0194732 | 0.356431 | 0.997887 | 0.712777 | true |
| new_high_low_breadth_bottom | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 48 | 0.0182048 | 3481 | 0.00448184 | 0.013723 | -0.00168163 | 0.0291276 | 0.0808042 | 0.997887 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | onset | 全A | 5 | max_down | 49 | -0.0244647 | 3493 | -0.0219953 | -0.00246941 | -0.00881692 | 0.0038781 | 0.445755 | 0.850987 | 0.769386 | true |
| new_high_low_breadth_bottom | bottom | onset | 全A | 5 | max_up | 49 | 0.0230511 | 3493 | 0.0195759 | 0.00347527 | -0.00332815 | 0.0102787 | 0.316734 | 0.774408 | 0.683599 | true |
| new_high_low_breadth_bottom | bottom | onset | 全A | 5 | terminal_return | 49 | 0.00478648 | 3493 | 0.00166396 | 0.00312252 | -0.00639487 | 0.0126399 | 0.520193 | 0.853136 | 0.80078 | true |
| new_high_low_breadth_bottom | bottom | onset | 全A | 10 | max_down | 49 | -0.0340575 | 3488 | -0.0317313 | -0.00232622 | -0.010298 | 0.00564556 | 0.567362 | 0.861625 | 0.819622 | true |
| new_high_low_breadth_bottom | bottom | onset | 全A | 10 | max_up | 49 | 0.0307465 | 3488 | 0.0292903 | 0.00145624 | -0.00686449 | 0.00977697 | 0.731578 | 0.940601 | 0.912637 | true |
| new_high_low_breadth_bottom | bottom | onset | 全A | 10 | terminal_return | 49 | 0.00394603 | 3488 | 0.00336716 | 0.000578871 | -0.0100254 | 0.0111831 | 0.914794 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 全A | 20 | max_down | 48 | -0.0460695 | 3479 | -0.0452015 | -0.000867939 | -0.0117344 | 0.00999855 | 0.875599 | 0.999087 | 0.993923 | true |
| new_high_low_breadth_bottom | bottom | onset | 全A | 20 | max_up | 48 | 0.0505984 | 3479 | 0.0442648 | 0.00633358 | -0.00727327 | 0.0199404 | 0.3616 | 0.785544 | 0.712777 | true |
| new_high_low_breadth_bottom | bottom | onset | 全A | 20 | terminal_return | 48 | 0.0217882 | 3479 | 0.00644292 | 0.0153453 | -0.00243182 | 0.0331223 | 0.0906685 | 0.774408 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 5 | max_down | 49 | -0.0338731 | 3495 | -0.0266385 | -0.00723466 | -0.0179397 | 0.00347043 | 0.185305 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 5 | max_up | 49 | 0.0270172 | 3495 | 0.0234238 | 0.00359335 | -0.00501837 | 0.0122051 | 0.413452 | 0.828035 | 0.746039 | true |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 5 | terminal_return | 49 | 0.00182545 | 3495 | 0.00266534 | -0.000839889 | -0.0150543 | 0.0133745 | 0.907802 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 10 | max_down | 49 | -0.0473196 | 3490 | -0.0388771 | -0.00844249 | -0.0198108 | 0.00292579 | 0.145513 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 10 | max_up | 49 | 0.037704 | 3490 | 0.0359145 | 0.00178945 | -0.0104795 | 0.0140584 | 0.774977 | 0.957325 | 0.922655 | true |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 10 | terminal_return | 49 | 0.0040182 | 3490 | 0.00531215 | -0.00129395 | -0.0167339 | 0.014146 | 0.869528 | 0.999087 | 0.993923 | true |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 20 | max_down | 48 | -0.0652963 | 3481 | -0.0558787 | -0.0094176 | -0.0261442 | 0.00730899 | 0.269792 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 20 | max_up | 48 | 0.0603847 | 3481 | 0.0549902 | 0.00539458 | -0.0143469 | 0.0251361 | 0.59224 | 0.861625 | 0.824556 | true |
| new_high_low_breadth_bottom | bottom | onset | 国证2000 | 20 | terminal_return | 48 | 0.0230409 | 3481 | 0.0103867 | 0.0126542 | -0.0135656 | 0.038874 | 0.344181 | 0.774408 | 0.699226 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 5 | max_down | 49 | -0.0330903 | 3495 | -0.0270702 | -0.00602006 | -0.0159011 | 0.00386095 | 0.232423 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 5 | max_up | 49 | 0.0268799 | 3495 | 0.0235155 | 0.00336441 | -0.00453418 | 0.011263 | 0.403795 | 0.828035 | 0.746039 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 5 | terminal_return | 49 | 0.00192217 | 3495 | 0.00201919 | -9.70251e-05 | -0.0132365 | 0.0130424 | 0.988452 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 10 | max_down | 49 | -0.0463157 | 3490 | -0.0395562 | -0.00675954 | -0.0178958 | 0.00437668 | 0.234166 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 10 | max_up | 49 | 0.0374107 | 3490 | 0.0355876 | 0.00182315 | -0.00950286 | 0.0131492 | 0.752381 | 0.948 | 0.91531 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 10 | terminal_return | 49 | 0.00399445 | 3490 | 0.00400297 | -8.51615e-06 | -0.0145938 | 0.0145768 | 0.999087 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 20 | max_down | 48 | -0.0646298 | 3481 | -0.0568958 | -0.00773397 | -0.0237292 | 0.00826122 | 0.343284 | 0.774408 | 0.699226 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 20 | max_up | 48 | 0.0599637 | 3481 | 0.0540045 | 0.00595926 | -0.0125554 | 0.0244739 | 0.528132 | 0.853136 | 0.80078 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证1000 | 20 | terminal_return | 48 | 0.0222943 | 3481 | 0.00773841 | 0.0145559 | -0.0105332 | 0.039645 | 0.255485 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 5 | max_down | 49 | -0.0205049 | 3495 | -0.0202798 | -0.000225033 | -0.00497637 | 0.0045263 | 0.926039 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 5 | max_up | 49 | 0.0239432 | 3495 | 0.019522 | 0.00442117 | -0.00234335 | 0.0111857 | 0.200186 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 5 | terminal_return | 49 | 0.00690599 | 3495 | 0.00136878 | 0.00553721 | -0.00297464 | 0.0140491 | 0.202296 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 10 | max_down | 49 | -0.0275307 | 3490 | -0.0290851 | 0.00155438 | -0.00477643 | 0.00788518 | 0.630353 | 0.882494 | 0.840471 | true |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 10 | max_up | 49 | 0.0317674 | 3490 | 0.0291489 | 0.00261852 | -0.00523749 | 0.0104745 | 0.513566 | 0.853136 | 0.80078 | true |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 10 | terminal_return | 49 | 0.00463035 | 3490 | 0.00280067 | 0.00182969 | -0.00743851 | 0.0110979 | 0.698805 | 0.928058 | 0.887666 | true |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 20 | max_down | 48 | -0.0362267 | 3481 | -0.0410208 | 0.00479413 | -0.00365987 | 0.0132481 | 0.26636 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 20 | max_up | 48 | 0.0515301 | 3481 | 0.0438904 | 0.00763971 | -0.00449328 | 0.0197727 | 0.21715 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 沪深300 | 20 | terminal_return | 48 | 0.0217354 | 3481 | 0.00536468 | 0.0163707 | 0.000816015 | 0.0319254 | 0.0391294 | 0.774408 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 5 | max_down | 49 | -0.0288489 | 3495 | -0.0244577 | -0.00439127 | -0.0123561 | 0.00357361 | 0.279873 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 5 | max_up | 49 | 0.0255461 | 3495 | 0.0219284 | 0.00361778 | -0.00374024 | 0.0109758 | 0.335202 | 0.774408 | 0.692384 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 5 | terminal_return | 49 | 0.00415573 | 3495 | 0.00194803 | 0.00220771 | -0.00903771 | 0.0134531 | 0.700394 | 0.928058 | 0.887666 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 10 | max_down | 49 | -0.0406449 | 3490 | -0.0353939 | -0.00525098 | -0.0153132 | 0.00481122 | 0.306388 | 0.774408 | 0.683599 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 10 | max_up | 49 | 0.0355682 | 3490 | 0.0328648 | 0.00270336 | -0.00744996 | 0.0128567 | 0.60177 | 0.861625 | 0.828742 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 10 | terminal_return | 49 | 0.00385443 | 3490 | 0.00390798 | -5.35462e-05 | -0.0129831 | 0.012876 | 0.993524 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 20 | max_down | 48 | -0.0544658 | 3481 | -0.0504541 | -0.00401177 | -0.0175862 | 0.00956264 | 0.562416 | 0.861625 | 0.819622 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 20 | max_up | 48 | 0.0589542 | 3481 | 0.049806 | 0.00914821 | -0.00708297 | 0.0253794 | 0.269292 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 中证500 | 20 | terminal_return | 48 | 0.0258392 | 3481 | 0.00748602 | 0.0183532 | -0.00303054 | 0.0397369 | 0.0925251 | 0.774408 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 5 | max_down | 49 | -0.0402127 | 3495 | -0.0275431 | -0.0126696 | -0.0276428 | 0.00230358 | 0.0972246 | 0.774408 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 5 | max_up | 49 | 0.0369099 | 3495 | 0.0260403 | 0.0108696 | -0.002959 | 0.0246982 | 0.123413 | 0.774408 | 0.676087 | true |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 5 | terminal_return | 49 | 0.005104 | 3495 | 0.00440961 | 0.000694388 | -0.0215087 | 0.0228974 | 0.951122 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 10 | max_down | 49 | -0.0580962 | 3490 | -0.0401382 | -0.017958 | -0.0349848 | -0.000931118 | 0.0387169 | 0.774408 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 10 | max_up | 49 | 0.0515588 | 3490 | 0.0401066 | 0.0114522 | -0.00974529 | 0.0326498 | 0.289638 | 0.774408 | 0.683599 | true |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 10 | terminal_return | 49 | 0.0087509 | 3490 | 0.00877918 | -2.82735e-05 | -0.0224549 | 0.0223983 | 0.998028 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 20 | max_down | 48 | -0.0818857 | 3481 | -0.0576896 | -0.0241961 | -0.0514121 | 0.00301998 | 0.0814188 | 0.774408 | 0.609584 | true |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 20 | max_up | 48 | 0.0778269 | 3481 | 0.0616898 | 0.0161371 | -0.0114325 | 0.0437068 | 0.251286 | 0.774408 | 0.678154 | true |
| new_high_low_breadth_bottom | bottom | onset | 微盘股 | 20 | terminal_return | 48 | 0.0275327 | 3481 | 0.017079 | 0.0104537 | -0.0265485 | 0.0474559 | 0.579763 | 0.861625 | 0.819622 | true |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 5 | max_down | 49 | -0.0191693 | 3495 | -0.0189343 | -0.000235005 | -0.00479201 | 0.004322 | 0.919489 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 5 | max_up | 49 | 0.0192341 | 3495 | 0.0171173 | 0.00211686 | -0.00373731 | 0.00797102 | 0.47849 | 0.853136 | 0.794855 | true |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 5 | terminal_return | 49 | 0.00503161 | 3495 | 0.00117293 | 0.00385868 | -0.00355946 | 0.0112768 | 0.307952 | 0.774408 | 0.683599 | true |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 10 | max_down | 49 | -0.0260148 | 3490 | -0.0271711 | 0.00115638 | -0.00487532 | 0.00718807 | 0.707092 | 0.928058 | 0.890936 | true |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 10 | max_up | 49 | 0.0257806 | 3490 | 0.0256559 | 0.000124729 | -0.00677055 | 0.00702001 | 0.971717 | 0.999087 | 0.999087 | true |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 10 | terminal_return | 49 | 0.00512204 | 3490 | 0.00235258 | 0.00276946 | -0.00530684 | 0.0108458 | 0.501516 | 0.853136 | 0.794855 | true |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 20 | max_down | 48 | -0.0352172 | 3481 | -0.0385465 | 0.0033293 | -0.0047728 | 0.0114314 | 0.420589 | 0.828035 | 0.746398 | true |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 20 | max_up | 48 | 0.0425522 | 3481 | 0.0386621 | 0.00389013 | -0.00710489 | 0.0148851 | 0.488019 | 0.853136 | 0.794855 | true |
| new_high_low_breadth_bottom | bottom | onset | 上证指数 | 20 | terminal_return | 48 | 0.0177196 | 3481 | 0.00448853 | 0.013231 | -0.000651609 | 0.0271137 | 0.0617616 | 0.774408 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 5 | max_down | 52 | -0.027471 | 3490 | -0.0219484 | -0.00552264 | -0.0137 | 0.00265475 | 0.185604 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 5 | max_up | 52 | 0.0237546 | 3490 | 0.0195624 | 0.00419224 | -0.00229032 | 0.0106748 | 0.204968 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 5 | terminal_return | 52 | 0.00659722 | 3490 | 0.0016343 | 0.00496293 | -0.00509503 | 0.0150209 | 0.33348 | 0.525231 | 0.692384 | true |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 10 | max_down | 52 | -0.0392449 | 3485 | -0.0316519 | -0.00759303 | -0.0219998 | 0.00681375 | 0.3016 | 0.521463 | 0.683599 | true |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 10 | max_up | 52 | 0.0384988 | 3485 | 0.0291733 | 0.00932549 | -0.000465282 | 0.0191163 | 0.0619217 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 10 | terminal_return | 52 | 0.00634491 | 3485 | 0.00333087 | 0.00301405 | -0.015771 | 0.0217991 | 0.753156 | 0.835019 | 0.91531 | true |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 20 | max_down | 52 | -0.060761 | 3475 | -0.0449807 | -0.0157803 | -0.0469042 | 0.0153435 | 0.320343 | 0.521463 | 0.683599 | true |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 20 | max_up | 52 | 0.0670634 | 3475 | 0.0440111 | 0.0230523 | 0.00266864 | 0.043436 | 0.0266501 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 全A | 20 | terminal_return | 52 | 0.0143174 | 3475 | 0.00653705 | 0.0077804 | -0.0306341 | 0.0461949 | 0.691386 | 0.817803 | 0.887666 | true |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 5 | max_down | 52 | -0.0322183 | 3492 | -0.0266569 | -0.00556141 | -0.0147634 | 0.0036406 | 0.23619 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 5 | max_up | 52 | 0.0282748 | 3492 | 0.023402 | 0.0048728 | -0.00213506 | 0.0118807 | 0.172929 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 52 | 0.00736884 | 3492 | 0.00258352 | 0.00478532 | -0.00642764 | 0.0159983 | 0.402894 | 0.590176 | 0.746039 | true |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 10 | max_down | 52 | -0.0461642 | 3487 | -0.0388871 | -0.00727707 | -0.0231118 | 0.00855767 | 0.367724 | 0.564282 | 0.712777 | true |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 10 | max_up | 52 | 0.0465596 | 3487 | 0.0357809 | 0.0107787 | -0.000358193 | 0.0219155 | 0.0578331 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 52 | 0.00873067 | 3487 | 0.00524299 | 0.00348768 | -0.0184641 | 0.0254394 | 0.755494 | 0.835019 | 0.91531 | true |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 20 | max_down | 52 | -0.0702621 | 3477 | -0.0557936 | -0.0144685 | -0.0490495 | 0.0201125 | 0.412186 | 0.590176 | 0.746039 | true |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 20 | max_up | 52 | 0.0760437 | 3477 | 0.0547498 | 0.0212939 | -0.00250972 | 0.0450976 | 0.0795422 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 52 | 0.0151553 | 3477 | 0.0104901 | 0.0046652 | -0.0408537 | 0.0501841 | 0.840793 | 0.913275 | 0.990093 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 5 | max_down | 52 | -0.0330795 | 3492 | -0.0270652 | -0.00601424 | -0.0152125 | 0.00318405 | 0.200006 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 5 | max_up | 52 | 0.0275704 | 3492 | 0.0235023 | 0.0040681 | -0.00302825 | 0.0111645 | 0.261181 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 52 | 0.00597395 | 3492 | 0.00195894 | 0.00401501 | -0.007158 | 0.015188 | 0.481231 | 0.639966 | 0.794855 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 10 | max_down | 52 | -0.0475596 | 3487 | -0.0395318 | -0.00802777 | -0.023942 | 0.00788651 | 0.322811 | 0.521463 | 0.683599 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 10 | max_up | 52 | 0.0445945 | 3487 | 0.0354789 | 0.00911564 | -0.00197034 | 0.0202016 | 0.107039 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 52 | 0.00522624 | 3487 | 0.0039846 | 0.00124163 | -0.0203875 | 0.0228708 | 0.910415 | 0.971564 | 0.999087 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 20 | max_down | 52 | -0.0726724 | 3477 | -0.0567666 | -0.0159058 | -0.0511345 | 0.0193229 | 0.376188 | 0.564282 | 0.712777 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 20 | max_up | 52 | 0.0725678 | 3477 | 0.0538091 | 0.0187587 | -0.00370095 | 0.0412183 | 0.101625 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 52 | 0.00955029 | 3477 | 0.00791226 | 0.00163804 | -0.0429215 | 0.0461975 | 0.942561 | 0.973465 | 0.999087 | true |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 5 | max_down | 52 | -0.0261863 | 3492 | -0.020195 | -0.00599124 | -0.0136771 | 0.00169462 | 0.12655 | 0.498291 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 5 | max_up | 52 | 0.0248725 | 3492 | 0.0195044 | 0.00536811 | -0.00175334 | 0.0124896 | 0.139558 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 52 | 0.0066676 | 3492 | 0.00136757 | 0.00530002 | -0.00516124 | 0.0157613 | 0.320709 | 0.521463 | 0.683599 | true |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 10 | max_down | 52 | -0.0362681 | 3487 | -0.0289561 | -0.00731194 | -0.0205303 | 0.00590639 | 0.278273 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 10 | max_up | 52 | 0.0386948 | 3487 | 0.0290434 | 0.00965145 | -0.00140112 | 0.020704 | 0.0869828 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 52 | 0.00649716 | 3487 | 0.00277125 | 0.0037259 | -0.0145234 | 0.0219752 | 0.689033 | 0.817803 | 0.887666 | true |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 20 | max_down | 52 | -0.0566551 | 3477 | -0.0407209 | -0.0159342 | -0.0435051 | 0.0116367 | 0.257317 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 20 | max_up | 52 | 0.0708095 | 3477 | 0.0435933 | 0.0272163 | 0.00286042 | 0.0515721 | 0.0285106 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 52 | 0.0189473 | 3477 | 0.00538755 | 0.0135598 | -0.0225061 | 0.0496256 | 0.461179 | 0.639966 | 0.790593 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 5 | max_down | 52 | -0.0323165 | 3492 | -0.0244023 | -0.00791421 | -0.0170993 | 0.00127091 | 0.0912572 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 5 | max_up | 52 | 0.0261043 | 3492 | 0.0219169 | 0.00418736 | -0.00267916 | 0.0110539 | 0.231988 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 52 | 0.00569852 | 3492 | 0.00192316 | 0.00377536 | -0.00713815 | 0.0146889 | 0.497751 | 0.639966 | 0.794855 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 10 | max_down | 52 | -0.0459311 | 3487 | -0.0353105 | -0.0106206 | -0.0265157 | 0.00527446 | 0.190326 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 10 | max_up | 52 | 0.0414141 | 3487 | 0.0327753 | 0.00863874 | -0.00146426 | 0.0187417 | 0.0937517 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 52 | 0.00372396 | 3487 | 0.00390997 | -0.000186014 | -0.0206991 | 0.020327 | 0.98582 | 0.99686 | 0.999087 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 20 | max_down | 52 | -0.0690098 | 3477 | -0.0502319 | -0.0187779 | -0.0526762 | 0.0151204 | 0.277595 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 20 | max_up | 52 | 0.0694651 | 3477 | 0.0496383 | 0.0198269 | -0.00138849 | 0.0410422 | 0.0669926 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 52 | 0.00969181 | 3477 | 0.00770639 | 0.00198542 | -0.0395182 | 0.0434891 | 0.925299 | 0.971564 | 0.999087 | true |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 5 | max_down | 52 | -0.0313272 | 3492 | -0.0276645 | -0.00366264 | -0.0137614 | 0.00643613 | 0.477173 | 0.639966 | 0.794855 | true |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 5 | max_up | 52 | 0.0309182 | 3492 | 0.0261202 | 0.00479805 | -0.00204553 | 0.0116416 | 0.169392 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 52 | 0.0104682 | 3492 | 0.00432914 | 0.00613908 | -0.00576085 | 0.018039 | 0.311945 | 0.521463 | 0.683599 | true |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 10 | max_down | 52 | -0.0433941 | 3487 | -0.040342 | -0.00305202 | -0.019511 | 0.013407 | 0.716272 | 0.820457 | 0.898012 | true |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 10 | max_up | 52 | 0.0479416 | 3487 | 0.0401507 | 0.00779089 | -0.00281004 | 0.0183918 | 0.149739 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 52 | 0.0129146 | 3487 | 0.00871711 | 0.00419747 | -0.0172268 | 0.0256217 | 0.700974 | 0.817803 | 0.887666 | true |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 20 | max_down | 52 | -0.0666043 | 3477 | -0.0578903 | -0.00871404 | -0.0438404 | 0.0264124 | 0.626804 | 0.789773 | 0.840471 | true |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 20 | max_up | 52 | 0.0761048 | 3477 | 0.061697 | 0.0144078 | -0.00761096 | 0.0364266 | 0.199664 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 52 | 0.0173077 | 3477 | 0.0172199 | 8.78317e-05 | -0.0436623 | 0.0438379 | 0.99686 | 0.99686 | 0.999087 | true |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 5 | max_down | 52 | -0.0251939 | 3492 | -0.0188444 | -0.00634945 | -0.0141041 | 0.00140518 | 0.10853 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 5 | max_up | 52 | 0.02372 | 3492 | 0.0170486 | 0.00667139 | 0.000114888 | 0.0132279 | 0.0461144 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 52 | 0.00669868 | 3492 | 0.00114479 | 0.00555389 | -0.00448518 | 0.015593 | 0.27822 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 10 | max_down | 52 | -0.035132 | 3487 | -0.0270362 | -0.00809579 | -0.0214821 | 0.0052905 | 0.23587 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 10 | max_up | 52 | 0.0369129 | 3487 | 0.0254898 | 0.0114232 | 0.000630543 | 0.0222158 | 0.038032 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 52 | 0.00642257 | 3487 | 0.0023308 | 0.00409177 | -0.0135897 | 0.0217733 | 0.650135 | 0.803108 | 0.85777 | true |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 20 | max_down | 52 | -0.0536172 | 3477 | -0.0382751 | -0.0153421 | -0.0425322 | 0.011848 | 0.268754 | 0.515624 | 0.678154 | true |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 20 | max_up | 52 | 0.0659911 | 3477 | 0.038307 | 0.027684 | 0.00510209 | 0.050266 | 0.0162685 | 0.455824 | 0.609584 | true |
| new_high_low_breadth_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 52 | 0.0166042 | 3477 | 0.00448999 | 0.0121142 | -0.0226233 | 0.0468518 | 0.494277 | 0.639966 | 0.794855 | true |
| new_high_low_breadth_top | top | onset | 全A | 5 | max_down | 52 | -0.0273831 | 3490 | -0.0219497 | -0.00543346 | -0.0132258 | 0.00235888 | 0.171728 | 0.491766 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 全A | 5 | max_up | 52 | 0.0253523 | 3490 | 0.0195386 | 0.0058137 | -0.00129314 | 0.0129205 | 0.108854 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 全A | 5 | terminal_return | 52 | 0.00482135 | 3490 | 0.00166076 | 0.0031606 | -0.00768867 | 0.0140099 | 0.56801 | 0.692978 | 0.819622 | true |
| new_high_low_breadth_top | top | onset | 全A | 10 | max_down | 52 | -0.0385633 | 3485 | -0.031662 | -0.00690123 | -0.0205143 | 0.00671181 | 0.3204 | 0.531189 | 0.683599 | true |
| new_high_low_breadth_top | top | onset | 全A | 10 | max_up | 52 | 0.039685 | 3485 | 0.0291556 | 0.0105294 | -0.000235247 | 0.021294 | 0.0552168 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 全A | 10 | terminal_return | 52 | 0.0119361 | 3485 | 0.00324744 | 0.00868866 | -0.00889394 | 0.0262713 | 0.332766 | 0.537545 | 0.692384 | true |
| new_high_low_breadth_top | top | onset | 全A | 20 | max_down | 52 | -0.0598605 | 3475 | -0.0449942 | -0.0148663 | -0.0440821 | 0.0143494 | 0.318599 | 0.531189 | 0.683599 | true |
| new_high_low_breadth_top | top | onset | 全A | 20 | max_up | 52 | 0.0673234 | 3475 | 0.0440072 | 0.0233162 | 0.00358863 | 0.0430437 | 0.0205287 | 0.325788 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 全A | 20 | terminal_return | 52 | 0.0132695 | 3475 | 0.00655273 | 0.0067168 | -0.0322047 | 0.0456383 | 0.73518 | 0.798558 | 0.912637 | true |
| new_high_low_breadth_top | top | onset | 国证2000 | 5 | max_down | 52 | -0.0318026 | 3492 | -0.0266631 | -0.00513953 | -0.0140639 | 0.0037848 | 0.258997 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 国证2000 | 5 | max_up | 52 | 0.0292213 | 3492 | 0.0233879 | 0.00583336 | -0.00214146 | 0.0138082 | 0.151662 | 0.477736 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 国证2000 | 5 | terminal_return | 52 | 0.00604808 | 3492 | 0.00260318 | 0.0034449 | -0.00850268 | 0.0153925 | 0.571982 | 0.692978 | 0.819622 | true |
| new_high_low_breadth_top | top | onset | 国证2000 | 10 | max_down | 52 | -0.0459955 | 3487 | -0.0388896 | -0.00710585 | -0.0219107 | 0.00769896 | 0.346838 | 0.546271 | 0.699226 | true |
| new_high_low_breadth_top | top | onset | 国证2000 | 10 | max_up | 52 | 0.0479014 | 3487 | 0.0357609 | 0.0121405 | -0.000613841 | 0.0248949 | 0.0620876 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 国证2000 | 10 | terminal_return | 52 | 0.0135273 | 3487 | 0.00517146 | 0.00835582 | -0.0126974 | 0.029409 | 0.436624 | 0.625166 | 0.764092 | true |
| new_high_low_breadth_top | top | onset | 国证2000 | 20 | max_down | 52 | -0.0690744 | 3477 | -0.0558114 | -0.013263 | -0.0450866 | 0.0185605 | 0.414005 | 0.614797 | 0.746039 | true |
| new_high_low_breadth_top | top | onset | 国证2000 | 20 | max_up | 52 | 0.0761053 | 3477 | 0.0547489 | 0.0213564 | -0.00273693 | 0.0454498 | 0.0823254 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 国证2000 | 20 | terminal_return | 52 | 0.01495 | 3477 | 0.0104932 | 0.00445687 | -0.0414133 | 0.050327 | 0.848965 | 0.891414 | 0.99046 | true |
| new_high_low_breadth_top | top | onset | 中证1000 | 5 | max_down | 52 | -0.0326494 | 3492 | -0.0270716 | -0.0055778 | -0.0144257 | 0.00327013 | 0.216608 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 中证1000 | 5 | max_up | 52 | 0.0290578 | 3492 | 0.0234801 | 0.00557771 | -0.00257504 | 0.0137305 | 0.179941 | 0.492882 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 中证1000 | 5 | terminal_return | 52 | 0.00494848 | 3492 | 0.00197421 | 0.00297427 | -0.00904933 | 0.0149979 | 0.627787 | 0.706261 | 0.840471 | true |
| new_high_low_breadth_top | top | onset | 中证1000 | 10 | max_down | 52 | -0.0471559 | 3487 | -0.0395378 | -0.00761807 | -0.0226081 | 0.00737199 | 0.319207 | 0.531189 | 0.683599 | true |
| new_high_low_breadth_top | top | onset | 中证1000 | 10 | max_up | 52 | 0.0461073 | 3487 | 0.0354563 | 0.010651 | -0.00210571 | 0.0234076 | 0.101742 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 中证1000 | 10 | terminal_return | 52 | 0.0105837 | 3487 | 0.00390471 | 0.00667899 | -0.0140549 | 0.0274129 | 0.527797 | 0.692978 | 0.80078 | true |
| new_high_low_breadth_top | top | onset | 中证1000 | 20 | max_down | 52 | -0.0716426 | 3477 | -0.056782 | -0.0148606 | -0.0473548 | 0.0176336 | 0.370056 | 0.568622 | 0.712777 | true |
| new_high_low_breadth_top | top | onset | 中证1000 | 20 | max_up | 52 | 0.0727393 | 3477 | 0.0538066 | 0.0189328 | -0.00389412 | 0.0417596 | 0.104027 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 中证1000 | 20 | terminal_return | 52 | 0.00904805 | 3477 | 0.00791977 | 0.00112828 | -0.0440635 | 0.04632 | 0.960971 | 0.985571 | 0.999087 | true |
| new_high_low_breadth_top | top | onset | 沪深300 | 5 | max_down | 52 | -0.0263397 | 3492 | -0.0201927 | -0.00614695 | -0.0136479 | 0.00135404 | 0.108232 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 沪深300 | 5 | max_up | 52 | 0.0266094 | 3492 | 0.0194785 | 0.00713084 | 8.41756e-05 | 0.0141775 | 0.0473213 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 沪深300 | 5 | terminal_return | 52 | 0.004424 | 3492 | 0.00140098 | 0.00302301 | -0.00784776 | 0.0138938 | 0.58572 | 0.696233 | 0.820008 | true |
| new_high_low_breadth_top | top | onset | 沪深300 | 10 | max_down | 52 | -0.0357132 | 3487 | -0.0289644 | -0.00674884 | -0.0196606 | 0.00616296 | 0.305614 | 0.531189 | 0.683599 | true |
| new_high_low_breadth_top | top | onset | 沪深300 | 10 | max_up | 52 | 0.0403549 | 3487 | 0.0290186 | 0.0113363 | 0.000296893 | 0.0223756 | 0.0441449 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 沪深300 | 10 | terminal_return | 52 | 0.0123423 | 3487 | 0.00268409 | 0.00965817 | -0.00740642 | 0.0267228 | 0.267294 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 沪深300 | 20 | max_down | 52 | -0.0559946 | 3477 | -0.0407307 | -0.0152638 | -0.0415979 | 0.0110702 | 0.25593 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 沪深300 | 20 | max_up | 52 | 0.0709727 | 3477 | 0.0435908 | 0.0273818 | 0.00557424 | 0.0491894 | 0.0138552 | 0.325788 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 沪深300 | 20 | terminal_return | 52 | 0.0171961 | 3477 | 0.00541374 | 0.0117824 | -0.0250508 | 0.0486156 | 0.530676 | 0.692978 | 0.80078 | true |
| new_high_low_breadth_top | top | onset | 中证500 | 5 | max_down | 52 | -0.0321885 | 3492 | -0.0244042 | -0.0077843 | -0.0163579 | 0.00078933 | 0.0751491 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 中证500 | 5 | max_up | 52 | 0.0274815 | 3492 | 0.0218964 | 0.0055851 | -0.00238418 | 0.0135544 | 0.169559 | 0.491766 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 中证500 | 5 | terminal_return | 52 | 0.0038248 | 3492 | 0.00195106 | 0.00187374 | -0.0101202 | 0.0138677 | 0.759453 | 0.810942 | 0.915705 | true |
| new_high_low_breadth_top | top | onset | 中证500 | 10 | max_down | 52 | -0.0453186 | 3487 | -0.0353197 | -0.00999892 | -0.0249739 | 0.00497608 | 0.190634 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 中证500 | 10 | max_up | 52 | 0.0427275 | 3487 | 0.0327557 | 0.00997173 | -0.00182809 | 0.0217716 | 0.0976516 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 中证500 | 10 | terminal_return | 52 | 0.00964095 | 3487 | 0.00382173 | 0.00581921 | -0.0136942 | 0.0253326 | 0.558881 | 0.692978 | 0.819622 | true |
| new_high_low_breadth_top | top | onset | 中证500 | 20 | max_down | 52 | -0.0681006 | 3477 | -0.0502455 | -0.0178551 | -0.0493808 | 0.0136707 | 0.266967 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 中证500 | 20 | max_up | 52 | 0.0697145 | 3477 | 0.0496345 | 0.02008 | -0.0016477 | 0.0418076 | 0.0700844 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 中证500 | 20 | terminal_return | 52 | 0.00825239 | 3477 | 0.00772792 | 0.000524466 | -0.041606 | 0.0426549 | 0.980534 | 0.985571 | 0.999087 | true |
| new_high_low_breadth_top | top | onset | 微盘股 | 5 | max_down | 52 | -0.0309465 | 3492 | -0.0276702 | -0.00327634 | -0.0130548 | 0.00650209 | 0.511365 | 0.692978 | 0.80078 | true |
| new_high_low_breadth_top | top | onset | 微盘股 | 5 | max_up | 52 | 0.0308419 | 3492 | 0.0261213 | 0.00472061 | -0.00310352 | 0.0125447 | 0.236989 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 微盘股 | 5 | terminal_return | 52 | 0.00832012 | 3492 | 0.00436112 | 0.00395899 | -0.00898053 | 0.0168985 | 0.548717 | 0.692978 | 0.819622 | true |
| new_high_low_breadth_top | top | onset | 微盘股 | 10 | max_down | 52 | -0.0434515 | 3487 | -0.0403412 | -0.00311037 | -0.0187044 | 0.0124837 | 0.695843 | 0.769089 | 0.887666 | true |
| new_high_low_breadth_top | top | onset | 微盘股 | 10 | max_up | 52 | 0.0493881 | 3487 | 0.0401291 | 0.00925897 | -0.00297237 | 0.0214903 | 0.13789 | 0.457213 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 微盘股 | 10 | terminal_return | 52 | 0.0172266 | 3487 | 0.00865281 | 0.00857377 | -0.0122478 | 0.0293954 | 0.419623 | 0.614797 | 0.746398 | true |
| new_high_low_breadth_top | top | onset | 微盘股 | 20 | max_down | 52 | -0.0660343 | 3477 | -0.0578988 | -0.00813542 | -0.040817 | 0.0245461 | 0.625618 | 0.706261 | 0.840471 | true |
| new_high_low_breadth_top | top | onset | 微盘股 | 20 | max_up | 52 | 0.0766039 | 3477 | 0.0616895 | 0.0149144 | -0.00784019 | 0.037669 | 0.198907 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 微盘股 | 20 | terminal_return | 52 | 0.0168114 | 3477 | 0.0172273 | -0.000415933 | -0.0454936 | 0.0446618 | 0.985571 | 0.985571 | 0.999087 | true |
| new_high_low_breadth_top | top | onset | 上证指数 | 5 | max_down | 52 | -0.025522 | 3492 | -0.0188395 | -0.00668248 | -0.0142476 | 0.000882657 | 0.083395 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 上证指数 | 5 | max_up | 52 | 0.0240561 | 3492 | 0.0170436 | 0.00701242 | 0.000472257 | 0.0135526 | 0.0355945 | 0.38099 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 上证指数 | 5 | terminal_return | 52 | 0.00393566 | 3492 | 0.00118593 | 0.00274972 | -0.00757928 | 0.0130787 | 0.601824 | 0.702128 | 0.828742 | true |
| new_high_low_breadth_top | top | onset | 上证指数 | 10 | max_down | 52 | -0.0347949 | 3487 | -0.0270412 | -0.00775372 | -0.02047 | 0.00496251 | 0.232044 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 上证指数 | 10 | max_up | 52 | 0.0379461 | 3487 | 0.0254744 | 0.0124717 | 0.00190652 | 0.023037 | 0.0206849 | 0.325788 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 上证指数 | 10 | terminal_return | 52 | 0.0115922 | 3487 | 0.00225371 | 0.00933852 | -0.0070101 | 0.0256871 | 0.262895 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 上证指数 | 20 | max_down | 52 | -0.0533909 | 3477 | -0.0382785 | -0.0151125 | -0.040902 | 0.0106771 | 0.250745 | 0.495281 | 0.678154 | true |
| new_high_low_breadth_top | top | onset | 上证指数 | 20 | max_up | 52 | 0.0657125 | 3477 | 0.0383112 | 0.0274013 | 0.00719292 | 0.0476097 | 0.00786908 | 0.325788 | 0.609584 | true |
| new_high_low_breadth_top | top | onset | 上证指数 | 20 | terminal_return | 52 | 0.0148442 | 3477 | 0.00451631 | 0.0103279 | -0.0243533 | 0.0450091 | 0.559435 | 0.692978 | 0.819622 | true |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `new_high_low_breadth_bottom/bottom/new_high_low_breadth_v1_20120104_20260814/capped_confirmation`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `new_high_low_breadth_bottom/bottom/new_high_low_breadth_v1_20120104_20260814/onset`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 2 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `new_high_low_breadth_top/top/new_high_low_breadth_v1_20120104_20260814/capped_confirmation`：5 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `new_high_low_breadth_top/top/new_high_low_breadth_v1_20120104_20260814/onset`：7 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
