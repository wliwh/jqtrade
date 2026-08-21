# 信号后 OHLC 结果评测

- 评测版本：`limit_up_down_breadth_v1_20120705_20260814__stage_d_v1`
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
| 上证指数 | capped_confirmation | 5 | 75 | 75 | 74 |
| 上证指数 | capped_confirmation | 10 | 75 | 75 | 74 |
| 上证指数 | capped_confirmation | 20 | 75 | 75 | 74 |
| 上证指数 | onset | 5 | 75 | 75 | 74 |
| 上证指数 | onset | 10 | 75 | 75 | 74 |
| 上证指数 | onset | 20 | 75 | 75 | 74 |
| 中证1000 | capped_confirmation | 5 | 75 | 75 | 74 |
| 中证1000 | capped_confirmation | 10 | 75 | 75 | 74 |
| 中证1000 | capped_confirmation | 20 | 75 | 75 | 74 |
| 中证1000 | onset | 5 | 75 | 75 | 74 |
| 中证1000 | onset | 10 | 75 | 75 | 74 |
| 中证1000 | onset | 20 | 75 | 75 | 74 |
| 中证500 | capped_confirmation | 5 | 75 | 75 | 74 |
| 中证500 | capped_confirmation | 10 | 75 | 75 | 74 |
| 中证500 | capped_confirmation | 20 | 75 | 75 | 74 |
| 中证500 | onset | 5 | 75 | 75 | 74 |
| 中证500 | onset | 10 | 75 | 75 | 74 |
| 中证500 | onset | 20 | 75 | 75 | 74 |
| 全A | capped_confirmation | 5 | 75 | 75 | 74 |
| 全A | capped_confirmation | 10 | 75 | 75 | 74 |
| 全A | capped_confirmation | 20 | 75 | 75 | 74 |
| 全A | onset | 5 | 75 | 75 | 74 |
| 全A | onset | 10 | 75 | 75 | 74 |
| 全A | onset | 20 | 75 | 75 | 74 |
| 国证2000 | capped_confirmation | 5 | 75 | 75 | 74 |
| 国证2000 | capped_confirmation | 10 | 75 | 75 | 74 |
| 国证2000 | capped_confirmation | 20 | 75 | 75 | 74 |
| 国证2000 | onset | 5 | 75 | 75 | 74 |
| 国证2000 | onset | 10 | 75 | 75 | 74 |
| 国证2000 | onset | 20 | 75 | 75 | 74 |
| 微盘股 | capped_confirmation | 5 | 75 | 75 | 74 |
| 微盘股 | capped_confirmation | 10 | 75 | 75 | 74 |
| 微盘股 | capped_confirmation | 20 | 75 | 75 | 74 |
| 微盘股 | onset | 5 | 75 | 75 | 74 |
| 微盘股 | onset | 10 | 75 | 75 | 74 |
| 微盘股 | onset | 20 | 75 | 75 | 74 |
| 沪深300 | capped_confirmation | 5 | 75 | 75 | 74 |
| 沪深300 | capped_confirmation | 10 | 75 | 75 | 74 |
| 沪深300 | capped_confirmation | 20 | 75 | 75 | 74 |
| 沪深300 | onset | 5 | 75 | 75 | 74 |
| 沪深300 | onset | 10 | 75 | 75 | 74 |
| 沪深300 | onset | 20 | 75 | 75 | 74 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 5 | max_down | 23 | -0.0261499 | 3399 | -0.0220099 | -0.00414008 | -0.015699 | 0.00741884 | 0.482669 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 5 | max_up | 23 | 0.0127296 | 3399 | 0.019642 | -0.00691242 | -0.0128351 | -0.000989777 | 0.0221637 | 0.118859 | 0.370672 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 5 | terminal_return | 23 | -0.00881482 | 3399 | 0.00174745 | -0.0105623 | -0.0240999 | 0.00297541 | 0.126211 | 0.397564 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 10 | max_down | 23 | -0.0337532 | 3394 | -0.0318301 | -0.00192317 | -0.0142988 | 0.0104525 | 0.760683 | 0.844146 | 0.901037 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 10 | max_up | 23 | 0.0185516 | 3394 | 0.0293972 | -0.0108456 | -0.0192502 | -0.00244102 | 0.0114304 | 0.102873 | 0.32005 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 10 | terminal_return | 23 | -0.0035206 | 3394 | 0.00341404 | -0.00693464 | -0.0210889 | 0.00721964 | 0.33692 | 0.624294 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 20 | max_down | 23 | -0.0425644 | 3384 | -0.0453176 | 0.00275318 | -0.0110425 | 0.0165488 | 0.695683 | 0.796873 | 0.896178 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 20 | max_up | 23 | 0.0264483 | 3384 | 0.0445552 | -0.0181069 | -0.0291741 | -0.00703968 | 0.00134255 | 0.0281935 | 0.112774 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 全A | 20 | terminal_return | 23 | -0.00402437 | 3384 | 0.00687124 | -0.0108956 | -0.0277308 | 0.00593964 | 0.204622 | 0.439267 | 0.8057 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_down | 23 | -0.0330622 | 3401 | -0.0266999 | -0.00636233 | -0.0241846 | 0.0114599 | 0.484116 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_up | 23 | 0.0179043 | 3401 | 0.0234915 | -0.00558718 | -0.0137064 | 0.00253205 | 0.177415 | 0.399349 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 23 | -0.0111895 | 3401 | 0.00272061 | -0.0139101 | -0.033473 | 0.00565284 | 0.163425 | 0.399349 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_down | 23 | -0.0441923 | 3396 | -0.0390461 | -0.00514619 | -0.0238713 | 0.0135789 | 0.590119 | 0.722027 | 0.83547 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_up | 23 | 0.0241559 | 3396 | 0.0360273 | -0.0118714 | -0.0217202 | -0.00202269 | 0.0181503 | 0.118859 | 0.370672 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 23 | -0.00494843 | 3396 | 0.00536748 | -0.0103159 | -0.0292456 | 0.00861382 | 0.285467 | 0.580143 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_down | 23 | -0.0549501 | 3386 | -0.0562008 | 0.00125077 | -0.018196 | 0.0206975 | 0.899682 | 0.92918 | 0.955138 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_up | 23 | 0.0388388 | 3386 | 0.0553253 | -0.0164864 | -0.0336192 | 0.000646328 | 0.0592869 | 0.249005 | 0.619013 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 23 | 0.000295856 | 3386 | 0.0107238 | -0.010428 | -0.0349962 | 0.0141403 | 0.405454 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_down | 23 | -0.0328958 | 3401 | -0.0270896 | -0.00580623 | -0.0229175 | 0.0113051 | 0.506006 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_up | 23 | 0.0175 | 3401 | 0.0235648 | -0.0060648 | -0.0142921 | 0.0021625 | 0.148507 | 0.399349 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 23 | -0.0115832 | 3401 | 0.00206127 | -0.0136445 | -0.0328794 | 0.00559043 | 0.164424 | 0.399349 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_down | 23 | -0.0443824 | 3396 | -0.0396593 | -0.00472303 | -0.0227414 | 0.0132954 | 0.60742 | 0.722027 | 0.841043 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_up | 23 | 0.0238925 | 3396 | 0.0356737 | -0.0117812 | -0.0217362 | -0.00182624 | 0.0203646 | 0.118859 | 0.370672 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 23 | -0.00511954 | 3396 | 0.00403442 | -0.00915397 | -0.0275461 | 0.00923818 | 0.329306 | 0.624294 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_down | 23 | -0.0555957 | 3386 | -0.0570962 | 0.00150056 | -0.0171207 | 0.0201218 | 0.874502 | 0.918227 | 0.937763 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_up | 23 | 0.0377817 | 3386 | 0.0542698 | -0.0164881 | -0.0332339 | 0.00025777 | 0.053628 | 0.241326 | 0.614284 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 23 | -0.00182125 | 3386 | 0.00806733 | -0.00988858 | -0.0344418 | 0.0146647 | 0.429895 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_down | 23 | -0.0232296 | 3401 | -0.0202834 | -0.00294619 | -0.0111218 | 0.00522942 | 0.479994 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_up | 23 | 0.0117164 | 3401 | 0.0196398 | -0.00792341 | -0.0135293 | -0.00231756 | 0.00560046 | 0.0717027 | 0.239009 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 23 | -0.00719986 | 3401 | 0.00148564 | -0.0086855 | -0.0193319 | 0.00196092 | 0.109822 | 0.364145 | 0.72829 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_down | 23 | -0.0299072 | 3396 | -0.0291356 | -0.000771644 | -0.0102726 | 0.00872928 | 0.873522 | 0.918227 | 0.937763 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_up | 23 | 0.0194655 | 3396 | 0.0293195 | -0.00985399 | -0.0187612 | -0.00094675 | 0.0301337 | 0.146033 | 0.446688 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 23 | -0.00183059 | 3396 | 0.0028715 | -0.00470209 | -0.0176012 | 0.00819698 | 0.474932 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_down | 23 | -0.0394667 | 3386 | -0.0410226 | 0.00155594 | -0.0110013 | 0.0141132 | 0.808114 | 0.877779 | 0.905088 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_up | 23 | 0.0262487 | 3386 | 0.044321 | -0.0180723 | -0.028473 | -0.00767164 | 0.000659902 | 0.0224248 | 0.089699 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 23 | -0.00622872 | 3386 | 0.0058848 | -0.0121135 | -0.0279449 | 0.00371783 | 0.133689 | 0.399349 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 5 | max_down | 23 | -0.0281955 | 3401 | -0.0244422 | -0.00375335 | -0.0176583 | 0.0101516 | 0.596764 | 0.722027 | 0.83547 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 5 | max_up | 23 | 0.0156768 | 3401 | 0.0219646 | -0.00628777 | -0.0132012 | 0.000625692 | 0.0746493 | 0.293931 | 0.627054 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 23 | -0.0081983 | 3401 | 0.00199642 | -0.0101947 | -0.0261057 | 0.0057163 | 0.209175 | 0.439267 | 0.810955 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 10 | max_down | 23 | -0.0377416 | 3396 | -0.0354555 | -0.00228612 | -0.0171943 | 0.0126221 | 0.763751 | 0.844146 | 0.901037 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 10 | max_up | 23 | 0.0222145 | 3396 | 0.0329289 | -0.0107144 | -0.0198776 | -0.00155127 | 0.0219162 | 0.118859 | 0.370672 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 23 | -0.00387258 | 3396 | 0.00391486 | -0.00778745 | -0.0244336 | 0.00885875 | 0.359179 | 0.637658 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 20 | max_down | 23 | -0.0461607 | 3386 | -0.0505389 | 0.00437815 | -0.0117345 | 0.0204908 | 0.59433 | 0.722027 | 0.83547 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 20 | max_up | 23 | 0.0329594 | 3386 | 0.0500373 | -0.0170779 | -0.0317623 | -0.00239341 | 0.0226398 | 0.118859 | 0.370672 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 23 | -0.000278617 | 3386 | 0.00785132 | -0.00812994 | -0.0292047 | 0.0129448 | 0.449588 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_down | 23 | -0.034147 | 3401 | -0.0277046 | -0.00644242 | -0.0252825 | 0.0123977 | 0.502712 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_up | 23 | 0.0206817 | 3401 | 0.0262798 | -0.0055981 | -0.0137346 | 0.00253838 | 0.177489 | 0.399349 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 23 | -0.010327 | 3401 | 0.00451827 | -0.0148453 | -0.036049 | 0.00635852 | 0.169988 | 0.399349 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_down | 23 | -0.0477249 | 3396 | -0.0404729 | -0.007252 | -0.0287357 | 0.0142317 | 0.508219 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_up | 23 | 0.0303969 | 3396 | 0.0404828 | -0.0100859 | -0.021375 | 0.00120325 | 0.0799298 | 0.296211 | 0.649752 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 23 | -0.000412224 | 3396 | 0.00889958 | -0.00931181 | -0.031023 | 0.0123994 | 0.400554 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_down | 23 | -0.0588876 | 3386 | -0.0582392 | -0.000648375 | -0.0235657 | 0.0222689 | 0.955778 | 0.955778 | 0.983086 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_up | 23 | 0.0474492 | 3386 | 0.0623303 | -0.0148811 | -0.0331119 | 0.00334976 | 0.109628 | 0.364145 | 0.72829 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 23 | 0.0094166 | 3386 | 0.0175457 | -0.00812908 | -0.0354377 | 0.0191796 | 0.559596 | 0.719481 | 0.83547 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_down | 23 | -0.0210824 | 3401 | -0.0189562 | -0.00212619 | -0.0109322 | 0.00667982 | 0.636044 | 0.742051 | 0.852569 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_up | 23 | 0.0109727 | 3401 | 0.0172498 | -0.00627703 | -0.0107264 | -0.00182764 | 0.00569069 | 0.0717027 | 0.239009 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 23 | -0.00649659 | 3401 | 0.00132117 | -0.00781776 | -0.0185709 | 0.00293538 | 0.154168 | 0.399349 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_down | 23 | -0.0276437 | 3396 | -0.0272091 | -0.000434613 | -0.0104099 | 0.00954065 | 0.931947 | 0.946978 | 0.967803 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_up | 23 | 0.0167926 | 3396 | 0.0258624 | -0.00906983 | -0.0160721 | -0.00206752 | 0.0111261 | 0.102873 | 0.32005 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 23 | -0.00218165 | 3396 | 0.00255742 | -0.00473907 | -0.0161905 | 0.0067124 | 0.417294 | 0.667038 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_down | 23 | -0.0330614 | 3386 | -0.0385322 | 0.00547085 | -0.00635072 | 0.0172924 | 0.364376 | 0.637658 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_up | 23 | 0.0234508 | 3386 | 0.0391281 | -0.0156773 | -0.024755 | -0.00659965 | 0.000711897 | 0.0224248 | 0.089699 | true |
| limit_up_down_breadth_bottom | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 23 | -0.00204621 | 3386 | 0.00512602 | -0.00717223 | -0.0213921 | 0.0070476 | 0.322864 | 0.624294 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 5 | max_down | 23 | -0.0203442 | 3399 | -0.0220491 | 0.00170497 | -0.00895572 | 0.0123657 | 0.753928 | 0.914039 | 0.901037 | true |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 5 | max_up | 23 | 0.0168319 | 3399 | 0.0196142 | -0.00278228 | -0.00899565 | 0.00343108 | 0.380123 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 5 | terminal_return | 23 | -0.00298372 | 3399 | 0.00170799 | -0.00469171 | -0.0172343 | 0.00785087 | 0.463459 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 10 | max_down | 23 | -0.0299998 | 3394 | -0.0318555 | 0.0018557 | -0.0113659 | 0.0150773 | 0.783244 | 0.914039 | 0.901114 | true |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 10 | max_up | 23 | 0.0217344 | 3394 | 0.0293756 | -0.00764128 | -0.0164057 | 0.0011232 | 0.0874844 | 0.837589 | 0.68894 | true |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 10 | terminal_return | 23 | -0.00210534 | 3394 | 0.00340445 | -0.00550979 | -0.0202263 | 0.00920672 | 0.463062 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 20 | max_down | 23 | -0.0389397 | 3384 | -0.0453422 | 0.00640254 | -0.00803805 | 0.0208431 | 0.384843 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 20 | max_up | 23 | 0.0298676 | 3384 | 0.044532 | -0.0146644 | -0.0273502 | -0.00197853 | 0.0234704 | 0.494229 | 0.370672 | true |
| limit_up_down_breadth_bottom | bottom | onset | 全A | 20 | terminal_return | 23 | 0.00123864 | 3384 | 0.00683546 | -0.00559682 | -0.023398 | 0.0122043 | 0.537736 | 0.914039 | 0.832613 | true |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 5 | max_down | 23 | -0.0267664 | 3401 | -0.0267425 | -2.39041e-05 | -0.0159097 | 0.0158619 | 0.997647 | 0.997647 | 0.997647 | true |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 5 | max_up | 23 | 0.0215595 | 3401 | 0.0234668 | -0.00190727 | -0.0105189 | 0.00670438 | 0.664222 | 0.914039 | 0.880968 | true |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 5 | terminal_return | 23 | -0.00348757 | 3401 | 0.00266852 | -0.00615609 | -0.0244265 | 0.0121143 | 0.508991 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 10 | max_down | 23 | -0.0413873 | 3396 | -0.0390651 | -0.00232218 | -0.0219732 | 0.0173288 | 0.816836 | 0.914039 | 0.910212 | true |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 10 | max_up | 23 | 0.0276696 | 3396 | 0.0360036 | -0.00833396 | -0.0198801 | 0.00321219 | 0.157151 | 0.837589 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 10 | terminal_return | 23 | -0.00507889 | 3396 | 0.00536836 | -0.0104473 | -0.0304117 | 0.00951721 | 0.305055 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 20 | max_down | 23 | -0.0528085 | 3386 | -0.0562154 | 0.00340686 | -0.0162929 | 0.0231067 | 0.73464 | 0.914039 | 0.901037 | true |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 20 | max_up | 23 | 0.0415447 | 3386 | 0.0553069 | -0.0137622 | -0.0331599 | 0.00563542 | 0.164353 | 0.837589 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | onset | 国证2000 | 20 | terminal_return | 23 | 0.00573489 | 3386 | 0.0106869 | -0.00495198 | -0.0312143 | 0.0213103 | 0.711699 | 0.914039 | 0.896741 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 5 | max_down | 23 | -0.0269828 | 3401 | -0.0271296 | 0.000146809 | -0.0151623 | 0.0154559 | 0.985004 | 0.997647 | 0.996872 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 5 | max_up | 23 | 0.0211552 | 3401 | 0.02354 | -0.00238487 | -0.0106085 | 0.00583881 | 0.569763 | 0.914039 | 0.83547 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 5 | terminal_return | 23 | -0.00375743 | 3401 | 0.00200835 | -0.00576577 | -0.0236482 | 0.0121167 | 0.527415 | 0.914039 | 0.832613 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 10 | max_down | 23 | -0.0418913 | 3396 | -0.0396762 | -0.00221504 | -0.0212853 | 0.0168552 | 0.819913 | 0.914039 | 0.910212 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 10 | max_up | 23 | 0.0275221 | 3396 | 0.0356491 | -0.00812706 | -0.0195888 | 0.00333468 | 0.164603 | 0.837589 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 10 | terminal_return | 23 | -0.00502036 | 3396 | 0.00403375 | -0.00905411 | -0.0290014 | 0.0108932 | 0.373655 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 20 | max_down | 23 | -0.0534744 | 3386 | -0.0571106 | 0.00363626 | -0.0154063 | 0.0226788 | 0.708203 | 0.914039 | 0.896741 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 20 | max_up | 23 | 0.0407966 | 3386 | 0.0542493 | -0.0134527 | -0.0323906 | 0.00548523 | 0.163832 | 0.837589 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证1000 | 20 | terminal_return | 23 | 0.00383299 | 3386 | 0.00802892 | -0.00419593 | -0.0303577 | 0.0219659 | 0.753254 | 0.914039 | 0.901037 | true |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 5 | max_down | 23 | -0.0191122 | 3401 | -0.0203112 | 0.00119908 | -0.00658088 | 0.00897904 | 0.762589 | 0.914039 | 0.901037 | true |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 5 | max_up | 23 | 0.0154433 | 3401 | 0.0196146 | -0.00417127 | -0.0101689 | 0.00182638 | 0.172836 | 0.837589 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 5 | terminal_return | 23 | -0.00283209 | 3401 | 0.0014561 | -0.00428819 | -0.0144362 | 0.00585985 | 0.407543 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 10 | max_down | 23 | -0.0261742 | 3396 | -0.0291608 | 0.00298668 | -0.00659594 | 0.0125693 | 0.541275 | 0.914039 | 0.832613 | true |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 10 | max_up | 23 | 0.0204813 | 3396 | 0.0293126 | -0.0088313 | -0.0175517 | -0.00011094 | 0.0471517 | 0.742639 | 0.594111 | true |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 10 | terminal_return | 23 | -0.000122769 | 3396 | 0.00285993 | -0.0029827 | -0.0158235 | 0.0098581 | 0.648912 | 0.914039 | 0.865215 | true |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 20 | max_down | 23 | -0.0351626 | 3386 | -0.0410518 | 0.00588922 | -0.00690162 | 0.0186801 | 0.366827 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 20 | max_up | 23 | 0.0293252 | 3386 | 0.0443001 | -0.0149749 | -0.0265344 | -0.0034153 | 0.0111143 | 0.494229 | 0.32005 | true |
| limit_up_down_breadth_bottom | bottom | onset | 沪深300 | 20 | terminal_return | 23 | -0.00157467 | 3386 | 0.00585318 | -0.00742785 | -0.0229033 | 0.00804762 | 0.346832 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 5 | max_down | 23 | -0.0226376 | 3401 | -0.0244797 | 0.00184216 | -0.010438 | 0.0141223 | 0.768742 | 0.914039 | 0.901037 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 5 | max_up | 23 | 0.0197107 | 3401 | 0.0219373 | -0.00222667 | -0.0096022 | 0.00514885 | 0.554035 | 0.914039 | 0.83547 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 5 | terminal_return | 23 | -0.00107386 | 3401 | 0.00194824 | -0.0030221 | -0.0172789 | 0.0112347 | 0.677796 | 0.914039 | 0.884998 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 10 | max_down | 23 | -0.0345411 | 3396 | -0.0354772 | 0.000936107 | -0.0145792 | 0.0164514 | 0.905865 | 0.951158 | 0.955138 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 10 | max_up | 23 | 0.0256332 | 3396 | 0.0329058 | -0.0072726 | -0.0174287 | 0.00288346 | 0.16046 | 0.837589 | 0.7582 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 10 | terminal_return | 23 | -0.00220549 | 3396 | 0.00390357 | -0.00610907 | -0.0234824 | 0.0112643 | 0.490694 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 20 | max_down | 23 | -0.0436421 | 3386 | -0.050556 | 0.00691387 | -0.00944782 | 0.0232756 | 0.407542 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 20 | max_up | 23 | 0.0364401 | 3386 | 0.0500136 | -0.0135736 | -0.0300691 | 0.00292194 | 0.106785 | 0.837589 | 0.72829 | true |
| limit_up_down_breadth_bottom | bottom | onset | 中证500 | 20 | terminal_return | 23 | 0.00531368 | 3386 | 0.00781334 | -0.00249966 | -0.0249154 | 0.0199161 | 0.826988 | 0.914039 | 0.914039 | true |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 5 | max_down | 23 | -0.0264298 | 3401 | -0.0277568 | 0.00132699 | -0.013024 | 0.015678 | 0.856183 | 0.91423 | 0.926001 | true |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 5 | max_up | 23 | 0.0252788 | 3401 | 0.0262487 | -0.000969924 | -0.0106291 | 0.0086892 | 0.843973 | 0.91423 | 0.923017 | true |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 5 | terminal_return | 23 | -0.00196034 | 3401 | 0.00446169 | -0.00642203 | -0.0254837 | 0.0126397 | 0.509037 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 10 | max_down | 23 | -0.0445992 | 3396 | -0.0404941 | -0.00410518 | -0.0250297 | 0.0168193 | 0.700584 | 0.914039 | 0.896178 | true |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 10 | max_up | 23 | 0.0350486 | 3396 | 0.0404513 | -0.00540277 | -0.0184579 | 0.00765237 | 0.417291 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 10 | terminal_return | 23 | 0.000259285 | 3396 | 0.00889503 | -0.00863575 | -0.0309167 | 0.0136452 | 0.447454 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 20 | max_down | 23 | -0.0552932 | 3386 | -0.0582637 | 0.00297048 | -0.0185947 | 0.0245357 | 0.787177 | 0.914039 | 0.901114 | true |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 20 | max_up | 23 | 0.0527854 | 3386 | 0.062294 | -0.00950867 | -0.0303639 | 0.0113465 | 0.371516 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 微盘股 | 20 | terminal_return | 23 | 0.017286 | 3386 | 0.0174922 | -0.000206248 | -0.0295252 | 0.0291127 | 0.988999 | 0.997647 | 0.996911 | true |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 5 | max_down | 23 | -0.0166935 | 3401 | -0.0189859 | 0.0022924 | -0.005746 | 0.0103308 | 0.576193 | 0.914039 | 0.83547 | true |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 5 | max_up | 23 | 0.0146928 | 3401 | 0.0172246 | -0.00253181 | -0.00834205 | 0.00327842 | 0.393065 | 0.914039 | 0.821985 | true |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 5 | terminal_return | 23 | -0.0015906 | 3401 | 0.00128799 | -0.00287859 | -0.0131602 | 0.00740301 | 0.583176 | 0.914039 | 0.83547 | true |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 10 | max_down | 23 | -0.0244182 | 3396 | -0.027231 | 0.00281279 | -0.00745797 | 0.0130835 | 0.591425 | 0.914039 | 0.83547 | true |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 10 | max_up | 23 | 0.0195443 | 3396 | 0.0258438 | -0.00629949 | -0.0141669 | 0.00156796 | 0.11656 | 0.837589 | 0.753156 | true |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 10 | terminal_return | 23 | -0.000552192 | 3396 | 0.00254639 | -0.00309858 | -0.0156451 | 0.00944798 | 0.628347 | 0.914039 | 0.85168 | true |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 20 | max_down | 23 | -0.030218 | 3386 | -0.0385515 | 0.00833354 | -0.00417268 | 0.0208398 | 0.191536 | 0.861912 | 0.77904 | true |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 20 | max_up | 23 | 0.0269385 | 3386 | 0.0391044 | -0.012166 | -0.0226953 | -0.00163657 | 0.0235347 | 0.494229 | 0.370672 | true |
| limit_up_down_breadth_bottom | bottom | onset | 上证指数 | 20 | terminal_return | 23 | 0.00216649 | 3386 | 0.00509741 | -0.00293092 | -0.0181809 | 0.012319 | 0.7064 | 0.914039 | 0.896741 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 5 | max_down | 51 | -0.0226945 | 3371 | -0.0220277 | -0.000666753 | -0.00699936 | 0.00566585 | 0.836505 | 0.941068 | 0.920521 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 5 | max_up | 51 | 0.0233707 | 3371 | 0.0195384 | 0.00383227 | -0.00105206 | 0.00871659 | 0.124091 | 0.745359 | 0.7582 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 5 | terminal_return | 51 | 0.00571056 | 3371 | 0.00161542 | 0.00409514 | -0.00352302 | 0.0117133 | 0.292067 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 10 | max_down | 51 | -0.034126 | 3366 | -0.0318084 | -0.00231756 | -0.0136755 | 0.00904039 | 0.689206 | 0.897454 | 0.895257 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 10 | max_up | 51 | 0.0334211 | 3366 | 0.0292621 | 0.004159 | -0.0031225 | 0.0114405 | 0.262927 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 10 | terminal_return | 51 | 0.00250594 | 3366 | 0.00338042 | -0.00087448 | -0.0150533 | 0.0133044 | 0.903784 | 0.976839 | 0.955138 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 20 | max_down | 51 | -0.0545219 | 3356 | -0.0451589 | -0.00936309 | -0.0267271 | 0.00800089 | 0.290566 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 20 | max_up | 51 | 0.049497 | 3356 | 0.044356 | 0.00514101 | -0.00787251 | 0.0181545 | 0.438753 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 全A | 20 | terminal_return | 51 | -0.00298538 | 3356 | 0.00694635 | -0.00993174 | -0.0331645 | 0.0133011 | 0.4021 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 5 | max_down | 51 | -0.0255652 | 3373 | -0.0267604 | 0.00119527 | -0.00656311 | 0.00895364 | 0.762682 | 0.907417 | 0.901037 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 5 | max_up | 51 | 0.0300493 | 3373 | 0.0233543 | 0.00669506 | 0.000396207 | 0.0129939 | 0.0372251 | 0.568571 | 0.493722 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 51 | 0.00904313 | 3373 | 0.00253016 | 0.00651297 | -0.00326402 | 0.01629 | 0.191669 | 0.745359 | 0.77904 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 10 | max_down | 51 | -0.0388715 | 3368 | -0.0390839 | 0.000212479 | -0.0118202 | 0.0122452 | 0.97239 | 0.988074 | 0.988074 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 10 | max_up | 51 | 0.0410639 | 3368 | 0.03587 | 0.00519394 | -0.00368997 | 0.0140778 | 0.251834 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 51 | 0.00468712 | 3368 | 0.00530733 | -0.000620212 | -0.0160192 | 0.0147787 | 0.937079 | 0.983933 | 0.967803 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 20 | max_down | 51 | -0.0669296 | 3358 | -0.0560293 | -0.0109003 | -0.0309709 | 0.00917038 | 0.287118 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 20 | max_up | 51 | 0.0625224 | 3358 | 0.055103 | 0.00741932 | -0.00931168 | 0.0241503 | 0.384761 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 51 | 0.00275796 | 3358 | 0.0107734 | -0.00801542 | -0.0370762 | 0.0210454 | 0.588785 | 0.835467 | 0.83547 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 5 | max_down | 51 | -0.0261252 | 3373 | -0.0271438 | 0.0010186 | -0.006795 | 0.0088322 | 0.79833 | 0.91445 | 0.902148 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 5 | max_up | 51 | 0.0291091 | 3373 | 0.0234396 | 0.00566957 | -0.00045889 | 0.011798 | 0.0697951 | 0.568571 | 0.627054 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 51 | 0.00734938 | 3373 | 0.00188828 | 0.00546111 | -0.00428671 | 0.0152089 | 0.272175 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 10 | max_down | 51 | -0.0403546 | 3368 | -0.0396811 | -0.000673489 | -0.0130145 | 0.0116675 | 0.914818 | 0.976839 | 0.960559 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 10 | max_up | 51 | 0.0394519 | 3368 | 0.035536 | 0.0039159 | -0.00466569 | 0.0124975 | 0.371121 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 51 | 0.00177386 | 3368 | 0.00400614 | -0.00223228 | -0.0177361 | 0.0132716 | 0.777786 | 0.907417 | 0.901114 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 20 | max_down | 51 | -0.0677151 | 3358 | -0.0569247 | -0.0107904 | -0.0309885 | 0.00940774 | 0.29506 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 20 | max_up | 51 | 0.0603494 | 3358 | 0.0540645 | 0.00628491 | -0.0103858 | 0.0229556 | 0.459951 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 51 | -0.00109507 | 3358 | 0.00813875 | -0.00923382 | -0.0381437 | 0.0196761 | 0.5313 | 0.79695 | 0.832613 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 5 | max_down | 51 | -0.0231627 | 3373 | -0.0202599 | -0.00290271 | -0.00862664 | 0.00282122 | 0.320247 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 5 | max_up | 51 | 0.0215276 | 3373 | 0.0195572 | 0.00197045 | -0.00275829 | 0.00669918 | 0.414085 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 51 | 0.00405264 | 3373 | 0.0013876 | 0.00266503 | -0.00476466 | 0.0100947 | 0.482023 | 0.747366 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 10 | max_down | 51 | -0.0329991 | 3368 | -0.0290823 | -0.0039168 | -0.0149456 | 0.00711205 | 0.486381 | 0.747366 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 10 | max_up | 51 | 0.0325311 | 3368 | 0.0292036 | 0.00332755 | -0.00445492 | 0.01111 | 0.40201 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 51 | 0.00249147 | 3368 | 0.00284514 | -0.000353666 | -0.0146248 | 0.0139175 | 0.96126 | 0.988074 | 0.984705 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 20 | max_down | 51 | -0.0491977 | 3358 | -0.0408878 | -0.00830988 | -0.024243 | 0.00762325 | 0.306671 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 20 | max_up | 51 | 0.0465357 | 3358 | 0.0441636 | 0.00237213 | -0.00961096 | 0.0143552 | 0.69802 | 0.897454 | 0.896178 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 51 | -0.00343464 | 3358 | 0.00594337 | -0.00937801 | -0.0292477 | 0.0104916 | 0.354928 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 5 | max_down | 51 | -0.024893 | 3373 | -0.0244609 | -0.000432032 | -0.00734772 | 0.00648366 | 0.902548 | 0.976839 | 0.955138 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 5 | max_up | 51 | 0.0271205 | 3373 | 0.0218438 | 0.00527676 | -0.00035566 | 0.0109092 | 0.0663228 | 0.568571 | 0.619013 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 51 | 0.00629496 | 3373 | 0.00186191 | 0.00443305 | -0.00389996 | 0.0127661 | 0.29709 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 10 | max_down | 51 | -0.0385867 | 3368 | -0.0354237 | -0.00316305 | -0.0150634 | 0.00873732 | 0.602397 | 0.835467 | 0.838696 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 10 | max_up | 51 | 0.0371976 | 3368 | 0.0327911 | 0.0044065 | -0.00367109 | 0.0124841 | 0.284969 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 51 | 0.00124252 | 3368 | 0.00390215 | -0.00265963 | -0.0178055 | 0.0124862 | 0.730712 | 0.907417 | 0.901037 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 20 | max_down | 51 | -0.0627407 | 3358 | -0.0503236 | -0.0124172 | -0.031439 | 0.00660467 | 0.200736 | 0.745359 | 0.802943 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 20 | max_up | 51 | 0.0554671 | 3358 | 0.0498378 | 0.00562923 | -0.00935103 | 0.0206095 | 0.461413 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 51 | -0.00361933 | 3358 | 0.00796985 | -0.0115892 | -0.0376256 | 0.0144472 | 0.382977 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 5 | max_down | 51 | -0.0246867 | 3373 | -0.0277941 | 0.00310741 | -0.00451088 | 0.0107257 | 0.424023 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 5 | max_up | 51 | 0.0357687 | 3373 | 0.0260982 | 0.00967054 | 0.00302953 | 0.0163116 | 0.00431558 | 0.271881 | 0.239009 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 51 | 0.0140503 | 3373 | 0.00427292 | 0.00977742 | -7.68583e-05 | 0.0196317 | 0.0518095 | 0.568571 | 0.614284 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 10 | max_down | 51 | -0.0373634 | 3368 | -0.0405695 | 0.00320609 | -0.00958693 | 0.0159991 | 0.623285 | 0.835467 | 0.85168 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 10 | max_up | 51 | 0.0496067 | 3368 | 0.0402758 | 0.00933093 | -0.000452631 | 0.0191145 | 0.0615781 | 0.568571 | 0.619013 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 51 | 0.0114658 | 3368 | 0.00879713 | 0.00266865 | -0.0141999 | 0.0195372 | 0.756502 | 0.907417 | 0.901037 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 20 | max_down | 51 | -0.0689424 | 3358 | -0.0580811 | -0.0108613 | -0.0321535 | 0.0104309 | 0.317402 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 20 | max_up | 51 | 0.0769811 | 3358 | 0.0620058 | 0.0149753 | -0.000899366 | 0.0308499 | 0.0644639 | 0.568571 | 0.619013 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 51 | 0.0131412 | 3358 | 0.0175569 | -0.00441566 | -0.0338231 | 0.0249918 | 0.768526 | 0.907417 | 0.901037 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 5 | max_down | 51 | -0.0203663 | 3373 | -0.0189494 | -0.0014169 | -0.00696755 | 0.00413374 | 0.616846 | 0.835467 | 0.849427 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 5 | max_up | 51 | 0.0211341 | 3373 | 0.0171482 | 0.00398586 | -0.00035947 | 0.00833119 | 0.0721995 | 0.568571 | 0.627054 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 51 | 0.00528082 | 3373 | 0.00120799 | 0.00407283 | -0.00260808 | 0.0107537 | 0.232142 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 10 | max_down | 51 | -0.0305774 | 3368 | -0.0271611 | -0.00341632 | -0.0144837 | 0.00765101 | 0.545164 | 0.798728 | 0.832613 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 10 | max_up | 51 | 0.0305812 | 3368 | 0.025729 | 0.00485217 | -0.00204601 | 0.0117504 | 0.167999 | 0.745359 | 0.7582 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 51 | 0.00258457 | 3368 | 0.00252465 | 5.99181e-05 | -0.013429 | 0.0135489 | 0.993053 | 0.993053 | 0.99701 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 20 | max_down | 51 | -0.0469819 | 3358 | -0.0383664 | -0.00861544 | -0.0245892 | 0.00735828 | 0.290453 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 20 | max_up | 51 | 0.0438878 | 3358 | 0.0389485 | 0.00493932 | -0.00672758 | 0.0166062 | 0.406658 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 51 | -0.00280391 | 3358 | 0.00519733 | -0.00800125 | -0.0277571 | 0.0117547 | 0.427306 | 0.745359 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 全A | 5 | max_down | 51 | -0.02472 | 3371 | -0.0219971 | -0.00272288 | -0.0105604 | 0.00511464 | 0.495913 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 全A | 5 | max_up | 51 | 0.0225202 | 3371 | 0.0195513 | 0.00296896 | -0.00275542 | 0.00869335 | 0.309363 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 全A | 5 | terminal_return | 51 | 0.00371395 | 3371 | 0.00164563 | 0.00206832 | -0.00763188 | 0.0117685 | 0.676005 | 0.86915 | 0.884998 | true |
| limit_up_down_breadth_top | top | onset | 全A | 10 | max_down | 51 | -0.0354651 | 3366 | -0.0317881 | -0.00367694 | -0.0154199 | 0.008066 | 0.539405 | 0.814053 | 0.832613 | true |
| limit_up_down_breadth_top | top | onset | 全A | 10 | max_up | 51 | 0.0323669 | 3366 | 0.0292781 | 0.00308879 | -0.00456023 | 0.0107378 | 0.428665 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 全A | 10 | terminal_return | 51 | -0.000129298 | 3366 | 0.00342034 | -0.00354964 | -0.0180769 | 0.0109776 | 0.632001 | 0.84715 | 0.85168 | true |
| limit_up_down_breadth_top | top | onset | 全A | 20 | max_down | 51 | -0.0546931 | 3356 | -0.0451563 | -0.00953681 | -0.0269748 | 0.00790114 | 0.283754 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 全A | 20 | max_up | 51 | 0.0471698 | 3356 | 0.0443914 | 0.00277838 | -0.0100303 | 0.015587 | 0.670726 | 0.86915 | 0.884937 | true |
| limit_up_down_breadth_top | top | onset | 全A | 20 | terminal_return | 51 | -0.00592534 | 3356 | 0.00699103 | -0.0129164 | -0.0364294 | 0.0105967 | 0.281623 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 国证2000 | 5 | max_down | 51 | -0.0279345 | 3373 | -0.0267246 | -0.00120992 | -0.0103641 | 0.00794428 | 0.795593 | 0.873806 | 0.902148 | true |
| limit_up_down_breadth_top | top | onset | 国证2000 | 5 | max_up | 51 | 0.0311054 | 3373 | 0.0233383 | 0.00776713 | -0.000331583 | 0.0158658 | 0.0601418 | 0.814053 | 0.619013 | true |
| limit_up_down_breadth_top | top | onset | 国证2000 | 5 | terminal_return | 51 | 0.00958531 | 3373 | 0.00252196 | 0.00706334 | -0.00541535 | 0.019542 | 0.267248 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 国证2000 | 10 | max_down | 51 | -0.0407774 | 3368 | -0.0390551 | -0.00172228 | -0.014237 | 0.0107925 | 0.787364 | 0.873806 | 0.901114 | true |
| limit_up_down_breadth_top | top | onset | 国证2000 | 10 | max_up | 51 | 0.042189 | 3368 | 0.035853 | 0.006336 | -0.00376054 | 0.0164325 | 0.218704 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 国证2000 | 10 | terminal_return | 51 | 0.00306611 | 3368 | 0.00533188 | -0.00226577 | -0.0189628 | 0.0144313 | 0.790263 | 0.873806 | 0.901114 | true |
| limit_up_down_breadth_top | top | onset | 国证2000 | 20 | max_down | 51 | -0.0685287 | 3358 | -0.056005 | -0.0125237 | -0.0331279 | 0.00808051 | 0.233523 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 国证2000 | 20 | max_up | 51 | 0.0612239 | 3358 | 0.0551228 | 0.00610115 | -0.0111296 | 0.0233319 | 0.487678 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 国证2000 | 20 | terminal_return | 51 | -0.000714577 | 3358 | 0.0108261 | -0.0115407 | -0.0409366 | 0.0178552 | 0.441605 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 中证1000 | 5 | max_down | 51 | -0.0284622 | 3373 | -0.0271084 | -0.00135377 | -0.0107053 | 0.00799773 | 0.77661 | 0.873806 | 0.901114 | true |
| limit_up_down_breadth_top | top | onset | 中证1000 | 5 | max_up | 51 | 0.0300894 | 3373 | 0.0234248 | 0.00666464 | -0.00140358 | 0.0147328 | 0.105441 | 0.814053 | 0.72829 | true |
| limit_up_down_breadth_top | top | onset | 中证1000 | 5 | terminal_return | 51 | 0.00758283 | 3373 | 0.00188475 | 0.00569809 | -0.00676507 | 0.0181612 | 0.370199 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 中证1000 | 10 | max_down | 51 | -0.0418274 | 3368 | -0.0396588 | -0.00216861 | -0.0151072 | 0.01077 | 0.742525 | 0.873806 | 0.901037 | true |
| limit_up_down_breadth_top | top | onset | 中证1000 | 10 | max_up | 51 | 0.0404192 | 3368 | 0.0355214 | 0.00489776 | -0.00496224 | 0.0147577 | 0.33026 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 中证1000 | 10 | terminal_return | 51 | -8.3033e-05 | 3368 | 0.00403426 | -0.00411729 | -0.0209028 | 0.0126682 | 0.630683 | 0.84715 | 0.85168 | true |
| limit_up_down_breadth_top | top | onset | 中证1000 | 20 | max_down | 51 | -0.0693164 | 3358 | -0.0569003 | -0.0124161 | -0.0332096 | 0.0083775 | 0.241865 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 中证1000 | 20 | max_up | 51 | 0.0589828 | 3358 | 0.0540853 | 0.00489746 | -0.0123463 | 0.0221412 | 0.577755 | 0.814053 | 0.83547 | true |
| limit_up_down_breadth_top | top | onset | 中证1000 | 20 | terminal_return | 51 | -0.00445548 | 3358 | 0.00818979 | -0.0126453 | -0.041882 | 0.0165915 | 0.396591 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 沪深300 | 5 | max_down | 51 | -0.0250364 | 3373 | -0.0202316 | -0.0048048 | -0.0117932 | 0.00218356 | 0.177792 | 0.814053 | 0.7582 | true |
| limit_up_down_breadth_top | top | onset | 沪深300 | 5 | max_up | 51 | 0.0200496 | 3373 | 0.0195795 | 0.00047004 | -0.00439612 | 0.0053362 | 0.849839 | 0.892331 | 0.923101 | true |
| limit_up_down_breadth_top | top | onset | 沪深300 | 5 | terminal_return | 51 | -9.63521e-05 | 3373 | 0.00145034 | -0.00154669 | -0.0101868 | 0.00709344 | 0.725691 | 0.873806 | 0.900857 | true |
| limit_up_down_breadth_top | top | onset | 沪深300 | 10 | max_down | 51 | -0.0347647 | 3368 | -0.0290556 | -0.00570916 | -0.0174327 | 0.00601443 | 0.33984 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 沪深300 | 10 | max_up | 51 | 0.030678 | 3368 | 0.0292316 | 0.00144636 | -0.00658138 | 0.0094741 | 0.723989 | 0.873806 | 0.900857 | true |
| limit_up_down_breadth_top | top | onset | 沪深300 | 10 | terminal_return | 51 | -0.00119756 | 3368 | 0.002901 | -0.00409856 | -0.0186714 | 0.0104743 | 0.581466 | 0.814053 | 0.83547 | true |
| limit_up_down_breadth_top | top | onset | 沪深300 | 20 | max_down | 51 | -0.0492441 | 3358 | -0.0408871 | -0.00835699 | -0.0240055 | 0.00729152 | 0.295226 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 沪深300 | 20 | max_up | 51 | 0.0446738 | 3358 | 0.0441918 | 0.000481929 | -0.0114307 | 0.0123945 | 0.9368 | 0.95191 | 0.967803 | true |
| limit_up_down_breadth_top | top | onset | 沪深300 | 20 | terminal_return | 51 | -0.00633266 | 3358 | 0.00598738 | -0.01232 | -0.0324765 | 0.00783638 | 0.230919 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 中证500 | 5 | max_down | 51 | -0.0270499 | 3373 | -0.0244283 | -0.00262154 | -0.0113037 | 0.00606061 | 0.553974 | 0.814053 | 0.83547 | true |
| limit_up_down_breadth_top | top | onset | 中证500 | 5 | max_up | 51 | 0.0268968 | 3373 | 0.0218472 | 0.00504966 | -0.00189193 | 0.0119913 | 0.153926 | 0.814053 | 0.7582 | true |
| limit_up_down_breadth_top | top | onset | 中证500 | 5 | terminal_return | 51 | 0.00549155 | 3373 | 0.00187405 | 0.0036175 | -0.00719821 | 0.0144332 | 0.51211 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 中证500 | 10 | max_down | 51 | -0.0394809 | 3368 | -0.0354102 | -0.00407079 | -0.0162294 | 0.00808782 | 0.511681 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 中证500 | 10 | max_up | 51 | 0.0368066 | 3368 | 0.0327971 | 0.0040095 | -0.00457743 | 0.0125964 | 0.360095 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 中证500 | 10 | terminal_return | 51 | -0.000845158 | 3368 | 0.00393376 | -0.00477892 | -0.0201783 | 0.0106204 | 0.543021 | 0.814053 | 0.832613 | true |
| limit_up_down_breadth_top | top | onset | 中证500 | 20 | max_down | 51 | -0.0635788 | 3358 | -0.0503108 | -0.013268 | -0.0326873 | 0.00615131 | 0.180524 | 0.814053 | 0.7582 | true |
| limit_up_down_breadth_top | top | onset | 中证500 | 20 | max_up | 51 | 0.0539311 | 3358 | 0.0498612 | 0.00406992 | -0.0103036 | 0.0184434 | 0.578907 | 0.814053 | 0.83547 | true |
| limit_up_down_breadth_top | top | onset | 中证500 | 20 | terminal_return | 51 | -0.00658796 | 3358 | 0.00801494 | -0.0146029 | -0.0406453 | 0.0114395 | 0.27175 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 微盘股 | 5 | max_down | 51 | -0.028182 | 3373 | -0.0277413 | -0.000440715 | -0.00963745 | 0.00875603 | 0.925169 | 0.95191 | 0.967397 | true |
| limit_up_down_breadth_top | top | onset | 微盘股 | 5 | max_up | 51 | 0.035143 | 3373 | 0.0261076 | 0.00903541 | 0.000651783 | 0.017419 | 0.0346539 | 0.814053 | 0.485154 | true |
| limit_up_down_breadth_top | top | onset | 微盘股 | 5 | terminal_return | 51 | 0.0137006 | 3373 | 0.00427821 | 0.00942237 | -0.00351377 | 0.0223585 | 0.153402 | 0.814053 | 0.7582 | true |
| limit_up_down_breadth_top | top | onset | 微盘股 | 10 | max_down | 51 | -0.040254 | 3368 | -0.0405257 | 0.00027176 | -0.012716 | 0.0132596 | 0.967287 | 0.967287 | 0.986867 | true |
| limit_up_down_breadth_top | top | onset | 微盘股 | 10 | max_up | 51 | 0.0496539 | 3368 | 0.0402751 | 0.00937882 | -0.00180657 | 0.0205642 | 0.100293 | 0.814053 | 0.72829 | true |
| limit_up_down_breadth_top | top | onset | 微盘股 | 10 | terminal_return | 51 | 0.0105323 | 3368 | 0.00881127 | 0.00172101 | -0.0156577 | 0.0190997 | 0.846099 | 0.892331 | 0.923017 | true |
| limit_up_down_breadth_top | top | onset | 微盘股 | 20 | max_down | 51 | -0.0700978 | 3358 | -0.0580636 | -0.0120342 | -0.0336255 | 0.00955704 | 0.274641 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 微盘股 | 20 | max_up | 51 | 0.0758008 | 3358 | 0.0620238 | 0.013777 | -0.0028597 | 0.0304137 | 0.104569 | 0.814053 | 0.72829 | true |
| limit_up_down_breadth_top | top | onset | 微盘股 | 20 | terminal_return | 51 | 0.00879172 | 3358 | 0.017623 | -0.00883124 | -0.0394268 | 0.0217643 | 0.571568 | 0.814053 | 0.83547 | true |
| limit_up_down_breadth_top | top | onset | 上证指数 | 5 | max_down | 51 | -0.0216688 | 3373 | -0.0189297 | -0.0027391 | -0.00955898 | 0.00408079 | 0.431164 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 上证指数 | 5 | max_up | 51 | 0.0197795 | 3373 | 0.0171687 | 0.00261072 | -0.00219152 | 0.00741296 | 0.286628 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 上证指数 | 5 | terminal_return | 51 | 0.0023198 | 3373 | 0.00125276 | 0.00106704 | -0.00738019 | 0.00951427 | 0.804456 | 0.873806 | 0.905013 | true |
| limit_up_down_breadth_top | top | onset | 上证指数 | 10 | max_down | 51 | -0.0316658 | 3368 | -0.0271446 | -0.00452119 | -0.0160827 | 0.00704036 | 0.443399 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 上证指数 | 10 | max_up | 51 | 0.0294011 | 3368 | 0.0257469 | 0.00365425 | -0.00377541 | 0.0110839 | 0.335037 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 上证指数 | 10 | terminal_return | 51 | -4.30556e-06 | 3368 | 0.00256385 | -0.00256816 | -0.0166517 | 0.0115153 | 0.720785 | 0.873806 | 0.900857 | true |
| limit_up_down_breadth_top | top | onset | 上证指数 | 20 | max_down | 51 | -0.0466563 | 3358 | -0.0383714 | -0.00828492 | -0.0242066 | 0.00763672 | 0.307777 | 0.814053 | 0.821985 | true |
| limit_up_down_breadth_top | top | onset | 上证指数 | 20 | max_up | 51 | 0.0426118 | 3358 | 0.0389678 | 0.00364394 | -0.00807817 | 0.0153661 | 0.542335 | 0.814053 | 0.832613 | true |
| limit_up_down_breadth_top | top | onset | 上证指数 | 20 | terminal_return | 51 | -0.00543465 | 3358 | 0.00523729 | -0.0106719 | -0.0311029 | 0.00975905 | 0.305936 | 0.814053 | 0.821985 | true |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `limit_up_down_breadth_bottom/bottom/limit_up_down_breadth_v1_20120705_20260814/capped_confirmation`：13 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `limit_up_down_breadth_bottom/bottom/limit_up_down_breadth_v1_20120705_20260814/onset`：4 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `limit_up_down_breadth_top/top/limit_up_down_breadth_v1_20120705_20260814/capped_confirmation`：数据可用性——5日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；10日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 2 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为负；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `limit_up_down_breadth_top/top/limit_up_down_breadth_v1_20120705_20260814/onset`：数据可用性——5日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；10日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 1 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为负；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
