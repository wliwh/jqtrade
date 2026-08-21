# 信号后 OHLC 结果评测

- 评测版本：`ma_period_breadth_decomposition_v1_20120104_20260814__stage_d_v1`
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
| 上证指数 | capped_confirmation | 5 | 374 | 374 | 374 |
| 上证指数 | capped_confirmation | 10 | 374 | 374 | 372 |
| 上证指数 | capped_confirmation | 20 | 374 | 374 | 371 |
| 上证指数 | onset | 5 | 375 | 375 | 374 |
| 上证指数 | onset | 10 | 375 | 375 | 373 |
| 上证指数 | onset | 20 | 375 | 375 | 371 |
| 中证1000 | capped_confirmation | 5 | 374 | 374 | 374 |
| 中证1000 | capped_confirmation | 10 | 374 | 374 | 372 |
| 中证1000 | capped_confirmation | 20 | 374 | 374 | 371 |
| 中证1000 | onset | 5 | 375 | 375 | 374 |
| 中证1000 | onset | 10 | 375 | 375 | 373 |
| 中证1000 | onset | 20 | 375 | 375 | 371 |
| 中证500 | capped_confirmation | 5 | 374 | 374 | 374 |
| 中证500 | capped_confirmation | 10 | 374 | 374 | 372 |
| 中证500 | capped_confirmation | 20 | 374 | 374 | 371 |
| 中证500 | onset | 5 | 375 | 375 | 374 |
| 中证500 | onset | 10 | 375 | 375 | 373 |
| 中证500 | onset | 20 | 375 | 375 | 371 |
| 全A | capped_confirmation | 5 | 374 | 374 | 374 |
| 全A | capped_confirmation | 10 | 374 | 374 | 372 |
| 全A | capped_confirmation | 20 | 374 | 374 | 371 |
| 全A | onset | 5 | 375 | 374 | 373 |
| 全A | onset | 10 | 375 | 374 | 372 |
| 全A | onset | 20 | 375 | 374 | 370 |
| 国证2000 | capped_confirmation | 5 | 374 | 374 | 374 |
| 国证2000 | capped_confirmation | 10 | 374 | 374 | 372 |
| 国证2000 | capped_confirmation | 20 | 374 | 374 | 371 |
| 国证2000 | onset | 5 | 375 | 375 | 374 |
| 国证2000 | onset | 10 | 375 | 375 | 373 |
| 国证2000 | onset | 20 | 375 | 375 | 371 |
| 微盘股 | capped_confirmation | 5 | 374 | 374 | 374 |
| 微盘股 | capped_confirmation | 10 | 374 | 374 | 372 |
| 微盘股 | capped_confirmation | 20 | 374 | 374 | 371 |
| 微盘股 | onset | 5 | 375 | 375 | 374 |
| 微盘股 | onset | 10 | 375 | 375 | 373 |
| 微盘股 | onset | 20 | 375 | 375 | 371 |
| 沪深300 | capped_confirmation | 5 | 374 | 374 | 374 |
| 沪深300 | capped_confirmation | 10 | 374 | 374 | 372 |
| 沪深300 | capped_confirmation | 20 | 374 | 374 | 371 |
| 沪深300 | onset | 5 | 375 | 375 | 374 |
| 沪深300 | onset | 10 | 375 | 375 | 373 |
| 沪深300 | onset | 20 | 375 | 375 | 371 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_down | 55 | -0.0172238 | 3487 | -0.0221053 | 0.00488144 | 9.19422e-05 | 0.00967094 | 0.0457573 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_up | 55 | 0.027335 | 3487 | 0.0195023 | 0.00783268 | -0.00416224 | 0.0198276 | 0.200588 | 0.526542 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | terminal_return | 55 | 0.0082264 | 3487 | 0.00160433 | 0.00662207 | -0.00332387 | 0.016568 | 0.191899 | 0.526542 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_down | 54 | -0.0310281 | 3483 | -0.0317749 | 0.000746835 | -0.00781355 | 0.00930722 | 0.864226 | 0.968616 | 0.943083 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_up | 54 | 0.0386929 | 3483 | 0.029165 | 0.00952795 | -0.00267006 | 0.021726 | 0.125777 | 0.466114 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | terminal_return | 54 | 0.0040522 | 3483 | 0.00336468 | 0.000687515 | -0.0122423 | 0.0136173 | 0.916996 | 0.968616 | 0.961219 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_down | 54 | -0.0439757 | 3473 | -0.0452326 | 0.00125687 | -0.0113764 | 0.0138901 | 0.845394 | 0.968616 | 0.934383 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_up | 54 | 0.0511651 | 3473 | 0.044245 | 0.00692001 | -0.00734331 | 0.0211833 | 0.341647 | 0.659424 | 0.770068 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | terminal_return | 54 | 0.00925266 | 3473 | 0.00661131 | 0.00264135 | -0.0162804 | 0.0215631 | 0.78439 | 0.968616 | 0.890938 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_down | 55 | -0.0197932 | 3489 | -0.026848 | 0.00705484 | 0.00116729 | 0.0129424 | 0.0188442 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_up | 55 | 0.0356917 | 3489 | 0.0232809 | 0.0124108 | -0.00163769 | 0.0264593 | 0.0833598 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 55 | 0.0130775 | 3489 | 0.00248941 | 0.010588 | -0.0020856 | 0.0232617 | 0.101535 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_down | 54 | -0.0342501 | 3485 | -0.0390675 | 0.0048174 | -0.00465171 | 0.0142865 | 0.318692 | 0.659424 | 0.759524 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_up | 54 | 0.0491166 | 3485 | 0.0357351 | 0.0133815 | -0.00127642 | 0.0280395 | 0.073563 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 54 | 0.0118341 | 3485 | 0.0051929 | 0.00664117 | -0.00843331 | 0.0217157 | 0.387867 | 0.659424 | 0.79262 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_down | 54 | -0.0529236 | 3475 | -0.0560547 | 0.00313109 | -0.0124101 | 0.0186723 | 0.69293 | 0.968616 | 0.87056 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_up | 54 | 0.068444 | 3475 | 0.0548556 | 0.0135884 | -0.00480518 | 0.031982 | 0.147627 | 0.516693 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 54 | 0.019268 | 3475 | 0.0104235 | 0.00884454 | -0.0164015 | 0.0340906 | 0.492301 | 0.756462 | 0.819466 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_down | 55 | -0.0201531 | 3489 | -0.0272638 | 0.00711073 | 0.00115961 | 0.0130618 | 0.0191849 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_up | 55 | 0.0355894 | 3489 | 0.0233724 | 0.012217 | -0.00187802 | 0.026312 | 0.0893476 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 55 | 0.01267 | 3489 | 0.00184993 | 0.0108201 | -0.00218333 | 0.0238235 | 0.10291 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_down | 54 | -0.0350978 | 3485 | -0.0397203 | 0.00462245 | -0.00500483 | 0.0142497 | 0.346666 | 0.659424 | 0.772903 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_up | 54 | 0.0480412 | 3485 | 0.0354203 | 0.0126209 | -0.00204334 | 0.0272852 | 0.0916245 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 54 | 0.0097369 | 3485 | 0.003914 | 0.0058229 | -0.00924787 | 0.0208937 | 0.448878 | 0.706983 | 0.806371 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_down | 54 | -0.0542397 | 3475 | -0.0570439 | 0.00280416 | -0.013011 | 0.0186194 | 0.728198 | 0.968616 | 0.874304 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_up | 54 | 0.0662767 | 3475 | 0.0538961 | 0.0123806 | -0.00551018 | 0.0302714 | 0.174991 | 0.526542 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 54 | 0.0155331 | 3475 | 0.00781834 | 0.0077148 | -0.0172328 | 0.0326624 | 0.544441 | 0.816662 | 0.825499 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_down | 55 | -0.0163501 | 3489 | -0.0203449 | 0.00399482 | -0.000283442 | 0.00827307 | 0.067228 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_up | 55 | 0.024173 | 3489 | 0.0195108 | 0.00466221 | -0.00614359 | 0.015468 | 0.397748 | 0.659424 | 0.79463 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 55 | 0.00521904 | 3489 | 0.00138585 | 0.00383319 | -0.00454047 | 0.0122069 | 0.369599 | 0.659424 | 0.788579 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_down | 54 | -0.0301981 | 3485 | -0.029046 | -0.00115207 | -0.0100613 | 0.0077572 | 0.799921 | 0.968616 | 0.902597 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_up | 54 | 0.0351305 | 3485 | 0.029093 | 0.00603747 | -0.00513407 | 0.017209 | 0.289487 | 0.651345 | 0.759524 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 54 | -0.000812495 | 3485 | 0.00288238 | -0.00369487 | -0.016304 | 0.00891427 | 0.565737 | 0.82887 | 0.825499 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_down | 54 | -0.041167 | 3475 | -0.0409524 | -0.000214603 | -0.0122849 | 0.0118557 | 0.972201 | 0.981983 | 0.981287 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_up | 54 | 0.0451038 | 3475 | 0.0439771 | 0.00112672 | -0.0123196 | 0.014573 | 0.869545 | 0.968616 | 0.944506 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 54 | 0.00432756 | 3475 | 0.00560693 | -0.00127936 | -0.0172479 | 0.0146892 | 0.87522 | 0.968616 | 0.947946 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_down | 55 | -0.0192333 | 3489 | -0.0246017 | 0.00536839 | -0.000193913 | 0.0109307 | 0.0585349 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_up | 55 | 0.0304129 | 3489 | 0.0218454 | 0.00856744 | -0.00439414 | 0.021529 | 0.195135 | 0.526542 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 55 | 0.00903527 | 3489 | 0.00186731 | 0.00716796 | -0.00445943 | 0.0187954 | 0.226937 | 0.549887 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_down | 54 | -0.0335688 | 3485 | -0.035496 | 0.00192722 | -0.0069698 | 0.0108242 | 0.671155 | 0.960971 | 0.863128 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_up | 54 | 0.0418006 | 3485 | 0.0327644 | 0.00903625 | -0.00410889 | 0.0221814 | 0.177869 | 0.526542 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 54 | 0.0046681 | 3485 | 0.00389545 | 0.000772651 | -0.0124277 | 0.013973 | 0.908664 | 0.968616 | 0.959427 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_down | 54 | -0.0503499 | 3475 | -0.0505111 | 0.000161202 | -0.0138295 | 0.0141519 | 0.981983 | 0.981983 | 0.987206 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_up | 54 | 0.055873 | 3475 | 0.0498381 | 0.00603496 | -0.00919466 | 0.0212646 | 0.437349 | 0.706486 | 0.794997 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 54 | 0.00901548 | 3475 | 0.00771576 | 0.00129971 | -0.0196442 | 0.0222436 | 0.903191 | 0.968616 | 0.957661 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_down | 55 | -0.0218072 | 3489 | -0.0278114 | 0.00600423 | -0.00125998 | 0.0132684 | 0.105224 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_up | 55 | 0.0402742 | 3489 | 0.0259686 | 0.0143056 | 0.0001595 | 0.0284518 | 0.0474681 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 55 | 0.0135289 | 3489 | 0.00427561 | 0.00925327 | -0.00358289 | 0.0220894 | 0.157681 | 0.522836 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_down | 54 | -0.0353374 | 3485 | -0.0404651 | 0.00512771 | -0.00586902 | 0.0161244 | 0.36075 | 0.659424 | 0.781453 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_up | 54 | 0.0568097 | 3485 | 0.0400088 | 0.0168009 | 0.00148103 | 0.0321208 | 0.0315964 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 54 | 0.0181982 | 3485 | 0.00863283 | 0.00956539 | -0.00713439 | 0.0262652 | 0.261582 | 0.610358 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_down | 54 | -0.0558963 | 3475 | -0.0580517 | 0.00215534 | -0.015306 | 0.0196167 | 0.808832 | 0.968616 | 0.905892 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_up | 54 | 0.0775671 | 3475 | 0.061666 | 0.0159012 | -0.00326056 | 0.0350629 | 0.103846 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 54 | 0.0284595 | 3475 | 0.0170466 | 0.011413 | -0.0146213 | 0.0374473 | 0.390214 | 0.659424 | 0.79262 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_down | 55 | -0.0155302 | 3489 | -0.0189913 | 0.00346115 | -0.000571575 | 0.00749387 | 0.0925299 | 0.414321 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_up | 55 | 0.0217648 | 3489 | 0.0170737 | 0.00469102 | -0.00495749 | 0.0143395 | 0.340623 | 0.659424 | 0.770068 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 55 | 0.00504359 | 3489 | 0.0011661 | 0.00387749 | -0.00359105 | 0.011346 | 0.308875 | 0.659424 | 0.759524 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_down | 54 | -0.02747 | 3485 | -0.0271502 | -0.000319797 | -0.00836063 | 0.00772104 | 0.937866 | 0.968616 | 0.961935 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_up | 54 | 0.031718 | 3485 | 0.0255637 | 0.00615425 | -0.00372384 | 0.0160323 | 0.222041 | 0.549887 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 54 | 0.000537531 | 3485 | 0.00241964 | -0.00188211 | -0.012941 | 0.00917676 | 0.738702 | 0.968616 | 0.878079 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_down | 54 | -0.0375195 | 3475 | -0.0385164 | 0.000996964 | -0.0102304 | 0.0122244 | 0.861832 | 0.968616 | 0.943083 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_up | 54 | 0.0404531 | 3475 | 0.038688 | 0.00176511 | -0.00978105 | 0.0133113 | 0.764457 | 0.968616 | 0.883684 | true |
| ma120_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 54 | 0.00534458 | 3475 | 0.00465798 | 0.000686601 | -0.0136161 | 0.0149893 | 0.925038 | 0.968616 | 0.961219 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_down | 55 | -0.0176569 | 3487 | -0.0220984 | 0.00444149 | -0.00129439 | 0.0101774 | 0.129091 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_up | 55 | 0.0291675 | 3487 | 0.0194734 | 0.00969405 | -0.00283001 | 0.0222181 | 0.12924 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 5 | terminal_return | 55 | 0.0134811 | 3487 | 0.00152145 | 0.0119597 | -0.0009183 | 0.0248377 | 0.0687229 | 0.279643 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_down | 54 | -0.0279149 | 3483 | -0.0318232 | 0.00390829 | -0.00320402 | 0.0110206 | 0.281462 | 0.422193 | 0.759524 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_up | 54 | 0.0414609 | 3483 | 0.029122 | 0.0123389 | -0.000523305 | 0.0252011 | 0.0600732 | 0.279643 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 10 | terminal_return | 54 | 0.00743902 | 3483 | 0.00331217 | 0.00412685 | -0.0079791 | 0.0162328 | 0.504036 | 0.648047 | 0.819466 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_down | 54 | -0.0422438 | 3473 | -0.0452595 | 0.0030157 | -0.00886204 | 0.0148935 | 0.618742 | 0.696084 | 0.84131 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_up | 54 | 0.0541231 | 3473 | 0.044199 | 0.00992401 | -0.00493518 | 0.0247832 | 0.190526 | 0.353033 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 全A | 20 | terminal_return | 54 | 0.0127551 | 3473 | 0.00655686 | 0.00619821 | -0.0125563 | 0.0249528 | 0.517139 | 0.651595 | 0.824399 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_down | 55 | -0.0202764 | 3489 | -0.0268404 | 0.00656395 | 4.71652e-05 | 0.0130807 | 0.0483607 | 0.253893 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_up | 55 | 0.0376639 | 3489 | 0.0232498 | 0.0144141 | 0.000233509 | 0.0285947 | 0.0463409 | 0.253893 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | terminal_return | 55 | 0.0206437 | 3489 | 0.00237014 | 0.0182736 | 0.00332117 | 0.033226 | 0.0166046 | 0.247199 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_down | 54 | -0.0317283 | 3485 | -0.0391066 | 0.0073783 | -0.00129339 | 0.01605 | 0.0953822 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_up | 54 | 0.0522308 | 3485 | 0.0356868 | 0.0165439 | 0.00164748 | 0.0314404 | 0.029498 | 0.247199 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | terminal_return | 54 | 0.0138391 | 3485 | 0.00516183 | 0.00867724 | -0.00566422 | 0.0230187 | 0.235666 | 0.390709 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_down | 54 | -0.0506099 | 3475 | -0.0560906 | 0.0054808 | -0.00909487 | 0.0200565 | 0.461118 | 0.618095 | 0.810913 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_up | 54 | 0.071415 | 3475 | 0.0548094 | 0.0166055 | -0.00157004 | 0.0347811 | 0.0733428 | 0.279643 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | terminal_return | 54 | 0.0225517 | 3475 | 0.0103725 | 0.0121793 | -0.0118483 | 0.0362068 | 0.320466 | 0.46952 | 0.759524 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_down | 55 | -0.0205064 | 3489 | -0.0272583 | 0.00675188 | 0.000118276 | 0.0133855 | 0.0460487 | 0.253893 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_up | 55 | 0.0379434 | 3489 | 0.0233353 | 0.0146081 | 0.00043581 | 0.0287805 | 0.0433552 | 0.253893 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | terminal_return | 55 | 0.0198681 | 3489 | 0.00173646 | 0.0181317 | 0.00304529 | 0.033218 | 0.0184909 | 0.247199 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_down | 54 | -0.0326063 | 3485 | -0.0397589 | 0.00715255 | -0.00151301 | 0.0158181 | 0.105709 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_up | 54 | 0.0518408 | 3485 | 0.0353614 | 0.0164794 | 0.0014709 | 0.0314878 | 0.0313904 | 0.247199 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | terminal_return | 54 | 0.0122008 | 3485 | 0.00387582 | 0.00832495 | -0.00616216 | 0.0228121 | 0.260036 | 0.399568 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_down | 54 | -0.0519725 | 3475 | -0.0570791 | 0.00510666 | -0.0097004 | 0.0199137 | 0.499063 | 0.648047 | 0.819466 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_up | 54 | 0.0697982 | 3475 | 0.0538414 | 0.0159568 | -0.00190178 | 0.0338155 | 0.079898 | 0.279643 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | terminal_return | 54 | 0.0185984 | 3475 | 0.00777071 | 0.0108277 | -0.0130547 | 0.0347101 | 0.37421 | 0.523894 | 0.792445 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_down | 55 | -0.0166934 | 3489 | -0.0203395 | 0.00364617 | -0.00166927 | 0.0089616 | 0.178794 | 0.34833 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_up | 55 | 0.026151 | 3489 | 0.0194796 | 0.00667134 | -0.00483445 | 0.0181771 | 0.255765 | 0.399568 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | terminal_return | 55 | 0.00862884 | 3489 | 0.0013321 | 0.00729674 | -0.00437188 | 0.0189654 | 0.220331 | 0.384449 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_down | 54 | -0.0270713 | 3485 | -0.0290944 | 0.0020231 | -0.00500697 | 0.00905317 | 0.572724 | 0.668178 | 0.825499 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_up | 54 | 0.0378712 | 3485 | 0.0290506 | 0.00882063 | -0.00295567 | 0.0205969 | 0.142085 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | terminal_return | 54 | 0.00377505 | 3485 | 0.0028113 | 0.00096375 | -0.0104533 | 0.0123808 | 0.868589 | 0.882599 | 0.944506 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_down | 54 | -0.0401937 | 3475 | -0.0409675 | 0.000773798 | -0.0107415 | 0.0122891 | 0.895216 | 0.895216 | 0.955908 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_up | 54 | 0.0480661 | 3475 | 0.043931 | 0.00413511 | -0.00973462 | 0.0180048 | 0.558983 | 0.668178 | 0.825499 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | terminal_return | 54 | 0.00815469 | 3475 | 0.00554746 | 0.00260723 | -0.0138118 | 0.0190263 | 0.755622 | 0.780397 | 0.882922 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_down | 55 | -0.0197449 | 3489 | -0.0245936 | 0.00484869 | -0.00149238 | 0.0111898 | 0.133949 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_up | 55 | 0.0325191 | 3489 | 0.0218122 | 0.0107069 | -0.00262811 | 0.0240419 | 0.115553 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | terminal_return | 55 | 0.0149959 | 3489 | 0.00177335 | 0.0132226 | -0.000677295 | 0.0271224 | 0.0622521 | 0.279643 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_down | 54 | -0.0305422 | 3485 | -0.0355429 | 0.00500071 | -0.0028369 | 0.0128383 | 0.211096 | 0.379972 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_up | 54 | 0.0451816 | 3485 | 0.032712 | 0.0124696 | -0.00137694 | 0.0263161 | 0.0775485 | 0.279643 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | terminal_return | 54 | 0.00773428 | 3485 | 0.00384794 | 0.00388634 | -0.00903622 | 0.0168089 | 0.555558 | 0.668178 | 0.825499 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_down | 54 | -0.0484129 | 3475 | -0.0505412 | 0.00212828 | -0.0109969 | 0.0152534 | 0.750622 | 0.780397 | 0.881165 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_up | 54 | 0.0590073 | 3475 | 0.0497894 | 0.0092179 | -0.00664484 | 0.0250806 | 0.254718 | 0.399568 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | terminal_return | 54 | 0.0121678 | 3475 | 0.00766678 | 0.00450107 | -0.0159361 | 0.0249383 | 0.665982 | 0.711133 | 0.863128 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_down | 55 | -0.0214957 | 3489 | -0.0278164 | 0.00632068 | -0.00159224 | 0.0142336 | 0.11744 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_up | 55 | 0.0432828 | 3489 | 0.0259212 | 0.0173616 | 0.00265956 | 0.0320637 | 0.0206371 | 0.247199 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | terminal_return | 55 | 0.0238738 | 3489 | 0.00411253 | 0.0197613 | 0.00378976 | 0.0357328 | 0.0153053 | 0.247199 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_down | 54 | -0.0326782 | 3485 | -0.0405063 | 0.00782813 | -0.00342892 | 0.0190852 | 0.172889 | 0.34833 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_up | 54 | 0.0615789 | 3485 | 0.0399349 | 0.021644 | 0.00604778 | 0.0372402 | 0.00652751 | 0.247199 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | terminal_return | 54 | 0.0206287 | 3485 | 0.00859517 | 0.0120336 | -0.00414085 | 0.028208 | 0.144781 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_down | 54 | -0.0539777 | 3475 | -0.0580815 | 0.00410382 | -0.0126608 | 0.0208685 | 0.631378 | 0.697839 | 0.848753 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_up | 54 | 0.0822342 | 3475 | 0.0615935 | 0.0206408 | 0.00243025 | 0.0388513 | 0.0263125 | 0.247199 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | terminal_return | 54 | 0.0340693 | 3475 | 0.0169594 | 0.0171099 | -0.00804371 | 0.0422635 | 0.182458 | 0.34833 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_down | 55 | -0.0155205 | 3489 | -0.0189915 | 0.00347092 | -0.00145369 | 0.00839552 | 0.167147 | 0.34833 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_up | 55 | 0.0233507 | 3489 | 0.0170487 | 0.00630198 | -0.00389539 | 0.0164993 | 0.225788 | 0.384449 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | terminal_return | 55 | 0.00871528 | 3489 | 0.00110822 | 0.00760706 | -0.00249186 | 0.017706 | 0.139842 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_down | 54 | -0.0241771 | 3485 | -0.0272013 | 0.00302417 | -0.00306388 | 0.00911223 | 0.330251 | 0.47286 | 0.761458 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_up | 54 | 0.0341979 | 3485 | 0.0255253 | 0.00867263 | -0.00177087 | 0.0191161 | 0.103599 | 0.314525 | 0.749588 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | terminal_return | 54 | 0.00451506 | 3485 | 0.00235801 | 0.00215705 | -0.00743978 | 0.0117539 | 0.659544 | 0.711133 | 0.862656 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_down | 54 | -0.0356789 | 3475 | -0.038545 | 0.00286615 | -0.00740706 | 0.0131394 | 0.584499 | 0.669517 | 0.830942 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_up | 54 | 0.0432177 | 3475 | 0.038645 | 0.00457268 | -0.0074469 | 0.0165923 | 0.455876 | 0.618095 | 0.810913 | true |
| ma120_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | terminal_return | 54 | 0.0088053 | 3475 | 0.00460421 | 0.00420109 | -0.0100549 | 0.0184571 | 0.56354 | 0.668178 | 0.825499 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_down | 36 | -0.0373768 | 3506 | -0.0218719 | -0.0155049 | -0.0332545 | 0.00224458 | 0.0868707 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_up | 36 | 0.0193741 | 3506 | 0.0196265 | -0.000252369 | -0.0060505 | 0.00554576 | 0.932014 | 0.947047 | 0.961219 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 5 | terminal_return | 36 | -0.0132106 | 3506 | 0.00186033 | -0.0150709 | -0.0335604 | 0.00341863 | 0.110131 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_down | 36 | -0.0468872 | 3501 | -0.031608 | -0.0152792 | -0.0365091 | 0.00595077 | 0.15836 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_up | 36 | 0.0297898 | 3501 | 0.0293055 | 0.00048434 | -0.00746701 | 0.00843569 | 0.904967 | 0.947047 | 0.958062 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 10 | terminal_return | 36 | -0.00517672 | 3501 | 0.00346311 | -0.00863984 | -0.0263252 | 0.00904553 | 0.338305 | 0.665555 | 0.770068 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_down | 36 | -0.0531711 | 3491 | -0.0451313 | -0.00803983 | -0.029494 | 0.0134144 | 0.462646 | 0.742857 | 0.810913 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_up | 36 | 0.0502826 | 3491 | 0.0442898 | 0.00599273 | -0.0120565 | 0.024042 | 0.515201 | 0.742857 | 0.82345 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 全A | 20 | terminal_return | 36 | 0.0143891 | 3491 | 0.00657197 | 0.00781716 | -0.0164376 | 0.0320719 | 0.527585 | 0.742857 | 0.825499 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_down | 36 | -0.0439031 | 3508 | -0.0265624 | -0.0173408 | -0.0397064 | 0.00502486 | 0.128599 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_up | 36 | 0.0212663 | 3508 | 0.0234962 | -0.00222988 | -0.0091937 | 0.00473394 | 0.530259 | 0.742857 | 0.825499 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 36 | -0.016229 | 3508 | 0.00284751 | -0.0190765 | -0.0433021 | 0.00514898 | 0.12273 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_down | 36 | -0.0561771 | 3503 | -0.0388174 | -0.0173597 | -0.0436578 | 0.00893848 | 0.195729 | 0.536127 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_up | 36 | 0.031893 | 3503 | 0.0359809 | -0.0040879 | -0.013522 | 0.0053462 | 0.39572 | 0.733246 | 0.79354 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 36 | -0.00739534 | 3503 | 0.00542464 | -0.01282 | -0.0356631 | 0.0100232 | 0.271338 | 0.611271 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_down | 36 | -0.0623117 | 3493 | -0.0559418 | -0.00636991 | -0.032738 | 0.0199982 | 0.635864 | 0.758517 | 0.848753 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_up | 36 | 0.0525428 | 3493 | 0.0550895 | -0.00254674 | -0.0187484 | 0.0136549 | 0.758013 | 0.823358 | 0.883558 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 36 | 0.0151945 | 3493 | 0.0105111 | 0.00468347 | -0.0190342 | 0.0284011 | 0.698729 | 0.772279 | 0.873684 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_down | 36 | -0.0440261 | 3508 | -0.0269803 | -0.0170458 | -0.039668 | 0.00557649 | 0.139716 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_up | 36 | 0.022065 | 3508 | 0.0235773 | -0.0015123 | -0.00874255 | 0.00571794 | 0.681835 | 0.771703 | 0.869284 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 36 | -0.0167374 | 3508 | 0.00221032 | -0.0189477 | -0.04363 | 0.00573468 | 0.132423 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_down | 36 | -0.0565862 | 3503 | -0.0394757 | -0.0171105 | -0.0437331 | 0.00951204 | 0.207775 | 0.54541 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_up | 36 | 0.0329134 | 3503 | 0.0356406 | -0.00272715 | -0.0123566 | 0.00690226 | 0.578831 | 0.742857 | 0.827214 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 36 | -0.0084294 | 3503 | 0.00413061 | -0.01256 | -0.0360687 | 0.0109486 | 0.295019 | 0.640904 | 0.759524 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_down | 36 | -0.0633108 | 3493 | -0.056936 | -0.0063748 | -0.0329402 | 0.0201906 | 0.638117 | 0.758517 | 0.848753 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_up | 36 | 0.0548094 | 3493 | 0.0540781 | 0.000731332 | -0.0159382 | 0.0174008 | 0.931474 | 0.947047 | 0.961219 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 36 | 0.0151056 | 3493 | 0.00786251 | 0.00724311 | -0.0173159 | 0.0318021 | 0.563226 | 0.742857 | 0.825499 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_down | 36 | -0.0345604 | 3508 | -0.0201364 | -0.014424 | -0.028317 | -0.000531094 | 0.0418583 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_up | 36 | 0.0211675 | 3508 | 0.0195669 | 0.00160057 | -0.00427004 | 0.00747118 | 0.593081 | 0.742857 | 0.830942 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 36 | -0.0101946 | 3508 | 0.00156479 | -0.0117594 | -0.0261794 | 0.00266064 | 0.109963 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_down | 36 | -0.0434246 | 3503 | -0.028916 | -0.0145086 | -0.031843 | 0.00282586 | 0.100905 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_up | 36 | 0.031428 | 3503 | 0.0291621 | 0.00226592 | -0.00618061 | 0.0107125 | 0.599025 | 0.742857 | 0.830942 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 36 | -0.00336456 | 3503 | 0.00288962 | -0.00625418 | -0.0219083 | 0.00939997 | 0.43359 | 0.742857 | 0.794997 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_down | 36 | -0.0512275 | 3493 | -0.0408498 | -0.0103778 | -0.0282143 | 0.00745874 | 0.254127 | 0.611271 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_up | 36 | 0.0573868 | 3493 | 0.0438563 | 0.0135305 | -0.0147645 | 0.0418254 | 0.348624 | 0.665555 | 0.772903 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 36 | 0.0144612 | 3493 | 0.00549589 | 0.00896534 | -0.0208597 | 0.0387904 | 0.555746 | 0.742857 | 0.825499 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_down | 36 | -0.0416145 | 3508 | -0.0243429 | -0.0172716 | -0.0382622 | 0.00371901 | 0.106802 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_up | 36 | 0.019927 | 3508 | 0.0219994 | -0.0020724 | -0.00826852 | 0.00412373 | 0.512111 | 0.742857 | 0.82345 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 36 | -0.015748 | 3508 | 0.00216047 | -0.0179084 | -0.0387125 | 0.00289559 | 0.0915651 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_down | 36 | -0.0516265 | 3503 | -0.0353005 | -0.016326 | -0.0400596 | 0.00740764 | 0.177577 | 0.508516 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_up | 36 | 0.0311755 | 3503 | 0.03292 | -0.00174456 | -0.00970901 | 0.0062199 | 0.667687 | 0.771703 | 0.863128 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 36 | -0.00549317 | 3503 | 0.00400384 | -0.00949701 | -0.0279594 | 0.00896541 | 0.31335 | 0.658035 | 0.759524 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_down | 36 | -0.0574915 | 3493 | -0.0504367 | -0.0070548 | -0.0309308 | 0.0168212 | 0.562499 | 0.742857 | 0.825499 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_up | 36 | 0.0511539 | 3493 | 0.0499178 | 0.00123609 | -0.0130519 | 0.0155241 | 0.865352 | 0.92402 | 0.943083 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 36 | 0.0158471 | 3493 | 0.00765205 | 0.00819502 | -0.0140759 | 0.0304659 | 0.470774 | 0.742857 | 0.814428 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_down | 36 | -0.0471919 | 3508 | -0.0275184 | -0.0196734 | -0.0432819 | 0.00393498 | 0.102403 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_up | 36 | 0.0202486 | 3508 | 0.0262516 | -0.00600298 | -0.0125494 | 0.000543467 | 0.0722902 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 36 | -0.0172459 | 3508 | 0.00464155 | -0.0218875 | -0.0465967 | 0.0028217 | 0.0825329 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_down | 36 | -0.0599322 | 3503 | -0.040186 | -0.0197461 | -0.0476947 | 0.00820245 | 0.166122 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_up | 36 | 0.0329532 | 3503 | 0.0403403 | -0.00738712 | -0.0176572 | 0.00288293 | 0.158598 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 36 | -0.00692744 | 3503 | 0.0089402 | -0.0158676 | -0.0383414 | 0.00660617 | 0.166402 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_down | 36 | -0.0654999 | 3493 | -0.0579416 | -0.00755833 | -0.0359141 | 0.0207974 | 0.60136 | 0.742857 | 0.831579 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_up | 36 | 0.0517646 | 3493 | 0.0620138 | -0.0102493 | -0.0277724 | 0.00727377 | 0.251626 | 0.611271 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 36 | 0.0166836 | 3493 | 0.0172267 | -0.000543185 | -0.0252437 | 0.0241573 | 0.96562 | 0.96562 | 0.977254 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_down | 36 | -0.0333327 | 3508 | -0.0187899 | -0.0145428 | -0.0289846 | -0.00010109 | 0.0484136 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_up | 36 | 0.018216 | 3508 | 0.0171356 | 0.00108045 | -0.00415686 | 0.00631777 | 0.685958 | 0.771703 | 0.870108 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 36 | -0.0104755 | 3508 | 0.00134637 | -0.0118219 | -0.0265148 | 0.00287101 | 0.114792 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_down | 36 | -0.0418021 | 3503 | -0.0270046 | -0.0147975 | -0.0327233 | 0.00312824 | 0.105671 | 0.499206 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_up | 36 | 0.0285307 | 3503 | 0.0256281 | 0.00290258 | -0.00540839 | 0.0112135 | 0.493644 | 0.742857 | 0.819466 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 36 | -0.00278854 | 3503 | 0.00244416 | -0.0052327 | -0.020456 | 0.00999056 | 0.500495 | 0.742857 | 0.819466 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_down | 36 | -0.0485088 | 3493 | -0.038398 | -0.0101108 | -0.0281393 | 0.00791774 | 0.271676 | 0.611271 | 0.749588 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_up | 36 | 0.0500606 | 3493 | 0.038598 | 0.0114626 | -0.011812 | 0.0347372 | 0.334402 | 0.665555 | 0.766084 | true |
| ma120_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 36 | 0.0120695 | 3493 | 0.00459221 | 0.0074773 | -0.017575 | 0.0325296 | 0.558549 | 0.742857 | 0.825499 | true |
| ma120_breadth_reversal_top | top | onset | 全A | 5 | max_down | 36 | -0.0358916 | 3506 | -0.0218871 | -0.0140045 | -0.0325913 | 0.00458222 | 0.13973 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 全A | 5 | max_up | 36 | 0.0199108 | 3506 | 0.019621 | 0.00028981 | -0.00578846 | 0.00636808 | 0.925544 | 0.949195 | 0.961219 | true |
| ma120_breadth_reversal_top | top | onset | 全A | 5 | terminal_return | 36 | -0.0113623 | 3506 | 0.00184136 | -0.0132037 | -0.0331241 | 0.00671675 | 0.1939 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 全A | 10 | max_down | 36 | -0.0471648 | 3501 | -0.0316051 | -0.0155596 | -0.0406512 | 0.00953192 | 0.224204 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 全A | 10 | max_up | 36 | 0.0296527 | 3501 | 0.0293069 | 0.000345768 | -0.0079027 | 0.00859424 | 0.934518 | 0.949195 | 0.961219 | true |
| ma120_breadth_reversal_top | top | onset | 全A | 10 | terminal_return | 36 | -0.00528535 | 3501 | 0.00346423 | -0.00874958 | -0.028667 | 0.0111678 | 0.389229 | 0.791014 | 0.79262 | true |
| ma120_breadth_reversal_top | top | onset | 全A | 20 | max_down | 36 | -0.0548777 | 3491 | -0.0451137 | -0.00976404 | -0.0353017 | 0.0157736 | 0.453626 | 0.848762 | 0.808823 | true |
| ma120_breadth_reversal_top | top | onset | 全A | 20 | max_up | 36 | 0.0492627 | 3491 | 0.0443003 | 0.00496231 | -0.0143491 | 0.0242738 | 0.61451 | 0.848762 | 0.838574 | true |
| ma120_breadth_reversal_top | top | onset | 全A | 20 | terminal_return | 36 | 0.011609 | 3491 | 0.00660063 | 0.00500837 | -0.0191227 | 0.0291395 | 0.684159 | 0.848762 | 0.869284 | true |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 5 | max_down | 36 | -0.0425174 | 3508 | -0.0265766 | -0.0159408 | -0.0386272 | 0.00674558 | 0.168447 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 5 | max_up | 36 | 0.022706 | 3508 | 0.0234814 | -0.000775348 | -0.00794242 | 0.00639172 | 0.832078 | 0.909199 | 0.922363 | true |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 5 | terminal_return | 36 | -0.0138632 | 3508 | 0.00282323 | -0.0166864 | -0.0423903 | 0.0090174 | 0.203233 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 10 | max_down | 36 | -0.0563092 | 3503 | -0.0388161 | -0.0174931 | -0.0482507 | 0.0132644 | 0.264964 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 10 | max_up | 36 | 0.0330517 | 3503 | 0.035969 | -0.00291726 | -0.0126874 | 0.00685289 | 0.55839 | 0.848762 | 0.825499 | true |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 10 | terminal_return | 36 | -0.00897831 | 3503 | 0.00544091 | -0.0144192 | -0.0413786 | 0.0125402 | 0.294499 | 0.710655 | 0.759524 | true |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 20 | max_down | 36 | -0.0633744 | 3493 | -0.0559308 | -0.00744359 | -0.038446 | 0.0235589 | 0.637934 | 0.848762 | 0.848753 | true |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 20 | max_up | 36 | 0.050663 | 3493 | 0.0551089 | -0.00444588 | -0.0208746 | 0.0119828 | 0.595829 | 0.848762 | 0.830942 | true |
| ma120_breadth_reversal_top | top | onset | 国证2000 | 20 | terminal_return | 36 | 0.012766 | 3493 | 0.0105361 | 0.00222988 | -0.0234307 | 0.0278904 | 0.864757 | 0.923384 | 0.943083 | true |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 5 | max_down | 36 | -0.0426801 | 3508 | -0.0269941 | -0.015686 | -0.0386873 | 0.00731534 | 0.18134 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 5 | max_up | 36 | 0.0233283 | 3508 | 0.0235644 | -0.000236073 | -0.00749787 | 0.00702572 | 0.949195 | 0.949195 | 0.965803 | true |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 5 | terminal_return | 36 | -0.0140666 | 3508 | 0.00218291 | -0.0162495 | -0.0423412 | 0.00984215 | 0.222215 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 10 | max_down | 36 | -0.0571214 | 3503 | -0.0394702 | -0.0176512 | -0.0487929 | 0.0134905 | 0.266597 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 10 | max_up | 36 | 0.03378 | 3503 | 0.0356317 | -0.00185169 | -0.0118288 | 0.00812547 | 0.716036 | 0.848762 | 0.874304 | true |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 10 | terminal_return | 36 | -0.0100331 | 3503 | 0.00414709 | -0.0141802 | -0.0418895 | 0.0135291 | 0.315847 | 0.710655 | 0.759524 | true |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 20 | max_down | 36 | -0.0646794 | 3493 | -0.0569219 | -0.00775751 | -0.0389061 | 0.0233911 | 0.625454 | 0.848762 | 0.844851 | true |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 20 | max_up | 36 | 0.0523315 | 3493 | 0.0541036 | -0.00177209 | -0.0186589 | 0.0151147 | 0.83704 | 0.909199 | 0.926504 | true |
| ma120_breadth_reversal_top | top | onset | 中证1000 | 20 | terminal_return | 36 | 0.0122307 | 3493 | 0.00789214 | 0.0043386 | -0.0219662 | 0.0306434 | 0.746488 | 0.855068 | 0.879042 | true |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 5 | max_down | 36 | -0.0324848 | 3508 | -0.0201577 | -0.012327 | -0.0269157 | 0.00226167 | 0.0976928 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 5 | max_up | 36 | 0.0209516 | 3508 | 0.0195691 | 0.0013825 | -0.00480645 | 0.00757145 | 0.66151 | 0.848762 | 0.863128 | true |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 5 | terminal_return | 36 | -0.00847141 | 3508 | 0.00154711 | -0.0100185 | -0.0251348 | 0.00509777 | 0.193939 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 10 | max_down | 36 | -0.0427709 | 3503 | -0.0289227 | -0.0138482 | -0.0342988 | 0.00660231 | 0.184433 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 10 | max_up | 36 | 0.0313049 | 3503 | 0.0291634 | 0.00214149 | -0.00748221 | 0.0117652 | 0.662733 | 0.848762 | 0.863128 | true |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 10 | terminal_return | 36 | -0.00207865 | 3503 | 0.00287641 | -0.00495506 | -0.0213131 | 0.011403 | 0.552708 | 0.848762 | 0.825499 | true |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 20 | max_down | 36 | -0.0521634 | 3493 | -0.0408401 | -0.0113233 | -0.0325638 | 0.00991719 | 0.296081 | 0.710655 | 0.759524 | true |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 20 | max_up | 36 | 0.0574858 | 3493 | 0.0438553 | 0.0136306 | -0.0165966 | 0.0438578 | 0.376784 | 0.791014 | 0.79262 | true |
| ma120_breadth_reversal_top | top | onset | 沪深300 | 20 | terminal_return | 36 | 0.0117803 | 3493 | 0.00552352 | 0.00625674 | -0.022583 | 0.0350965 | 0.670677 | 0.848762 | 0.863128 | true |
| ma120_breadth_reversal_top | top | onset | 中证500 | 5 | max_down | 36 | -0.0403562 | 3508 | -0.0243558 | -0.0160004 | -0.0372347 | 0.00523393 | 0.139705 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 中证500 | 5 | max_up | 36 | 0.0208831 | 3508 | 0.0219896 | -0.00110652 | -0.00733079 | 0.00511775 | 0.727511 | 0.848762 | 0.874304 | true |
| ma120_breadth_reversal_top | top | onset | 中证500 | 5 | terminal_return | 36 | -0.0150978 | 3508 | 0.00215379 | -0.0172516 | -0.0407422 | 0.00623891 | 0.150026 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 中证500 | 10 | max_down | 36 | -0.052876 | 3503 | -0.0352877 | -0.0175883 | -0.0455194 | 0.0103428 | 0.217122 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 中证500 | 10 | max_up | 36 | 0.0309978 | 3503 | 0.0329218 | -0.00192401 | -0.0103195 | 0.00647152 | 0.653305 | 0.848762 | 0.858955 | true |
| ma120_breadth_reversal_top | top | onset | 中证500 | 10 | terminal_return | 36 | -0.007234 | 3503 | 0.00402173 | -0.0112557 | -0.0329725 | 0.010461 | 0.309694 | 0.710655 | 0.759524 | true |
| ma120_breadth_reversal_top | top | onset | 中证500 | 20 | max_down | 36 | -0.0600651 | 3493 | -0.0504101 | -0.00965501 | -0.03787 | 0.01856 | 0.502411 | 0.848762 | 0.819466 | true |
| ma120_breadth_reversal_top | top | onset | 中证500 | 20 | max_up | 36 | 0.0477981 | 3493 | 0.0499524 | -0.0021543 | -0.0168539 | 0.0125453 | 0.773923 | 0.870664 | 0.889522 | true |
| ma120_breadth_reversal_top | top | onset | 中证500 | 20 | terminal_return | 36 | 0.0122549 | 3493 | 0.00768907 | 0.00456583 | -0.0180492 | 0.0271809 | 0.692318 | 0.848762 | 0.87056 | true |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 5 | max_down | 36 | -0.0437167 | 3508 | -0.0275541 | -0.0161626 | -0.0395684 | 0.00724311 | 0.175908 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 5 | max_up | 36 | 0.0238771 | 3508 | 0.0262143 | -0.00233723 | -0.00937015 | 0.00469568 | 0.514813 | 0.848762 | 0.82345 | true |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 5 | terminal_return | 36 | -0.0156641 | 3508 | 0.00462531 | -0.0202894 | -0.046387 | 0.0058083 | 0.127563 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 10 | max_down | 36 | -0.0590175 | 3503 | -0.0401954 | -0.0188221 | -0.0510526 | 0.0134083 | 0.25237 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 10 | max_up | 36 | 0.0350483 | 3503 | 0.0403188 | -0.00527048 | -0.0158832 | 0.00534224 | 0.330368 | 0.717696 | 0.761458 | true |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 10 | terminal_return | 36 | -0.00626883 | 3503 | 0.00893343 | -0.0152023 | -0.042072 | 0.0116675 | 0.267464 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 20 | max_down | 36 | -0.0664459 | 3493 | -0.0579319 | -0.00851406 | -0.041202 | 0.0241739 | 0.609693 | 0.848762 | 0.837139 | true |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 20 | max_up | 36 | 0.0520467 | 3493 | 0.0620109 | -0.00996428 | -0.0277271 | 0.00779854 | 0.271556 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 微盘股 | 20 | terminal_return | 36 | 0.0152433 | 3493 | 0.0172416 | -0.00199831 | -0.0282861 | 0.0242895 | 0.88156 | 0.925637 | 0.951772 | true |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 5 | max_down | 36 | -0.0321288 | 3508 | -0.0188022 | -0.0133266 | -0.0280169 | 0.00136374 | 0.0753959 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 5 | max_up | 36 | 0.0181603 | 3508 | 0.0171361 | 0.0010242 | -0.00453693 | 0.00658533 | 0.718119 | 0.848762 | 0.874304 | true |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 5 | terminal_return | 36 | -0.0097597 | 3508 | 0.00133902 | -0.0110987 | -0.0266693 | 0.0044719 | 0.162388 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 10 | max_down | 36 | -0.0425726 | 3503 | -0.0269967 | -0.0155759 | -0.0365278 | 0.00537591 | 0.14509 | 0.710655 | 0.749588 | true |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 10 | max_up | 36 | 0.0277988 | 3503 | 0.0256356 | 0.00216316 | -0.00716118 | 0.0114875 | 0.649323 | 0.848762 | 0.855481 | true |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 10 | terminal_return | 36 | -0.0031 | 3503 | 0.00244736 | -0.00554736 | -0.0219919 | 0.0108972 | 0.508497 | 0.848762 | 0.823177 | true |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 20 | max_down | 36 | -0.0500657 | 3493 | -0.038382 | -0.0116837 | -0.0330721 | 0.00970477 | 0.284317 | 0.710655 | 0.759524 | true |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 20 | max_up | 36 | 0.0489503 | 3493 | 0.0386095 | 0.0103408 | -0.0144146 | 0.0350962 | 0.412941 | 0.812978 | 0.794658 | true |
| ma120_breadth_reversal_top | top | onset | 上证指数 | 20 | terminal_return | 36 | 0.00922964 | 3493 | 0.00462148 | 0.00460816 | -0.0198308 | 0.0290472 | 0.711701 | 0.848762 | 0.874304 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_down | 83 | -0.0207333 | 3459 | -0.0220606 | 0.0013273 | -0.00296231 | 0.00561691 | 0.544205 | 0.874893 | 0.825499 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_up | 83 | 0.0197406 | 3459 | 0.0196211 | 0.000119521 | -0.00346341 | 0.00370245 | 0.947869 | 0.995263 | 0.965803 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | terminal_return | 83 | 0.00207757 | 3459 | 0.00169827 | 0.000379295 | -0.00552688 | 0.00628547 | 0.899834 | 0.995263 | 0.95633 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_down | 83 | -0.0282359 | 3454 | -0.0318483 | 0.00361234 | -0.00226807 | 0.00949276 | 0.228578 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_up | 83 | 0.0303183 | 3454 | 0.0292862 | 0.00103205 | -0.00428045 | 0.00634454 | 0.703378 | 0.904104 | 0.873816 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | terminal_return | 83 | 0.0101376 | 3454 | 0.00321267 | 0.00692495 | -0.00133166 | 0.0151816 | 0.100201 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_down | 82 | -0.0407346 | 3445 | -0.04532 | 0.00458536 | -0.00401494 | 0.0131857 | 0.296023 | 0.714689 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_up | 82 | 0.0496483 | 3445 | 0.0442249 | 0.0054234 | -0.00532099 | 0.0161678 | 0.322496 | 0.714689 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | terminal_return | 82 | 0.0167451 | 3445 | 0.00641151 | 0.0103336 | -0.00386952 | 0.0245368 | 0.153864 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_down | 83 | -0.0252452 | 3461 | -0.0267743 | 0.00152917 | -0.00416335 | 0.00722168 | 0.598534 | 0.874893 | 0.830942 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_up | 83 | 0.0242773 | 3461 | 0.0234542 | 0.000823049 | -0.00405499 | 0.00570109 | 0.74087 | 0.904104 | 0.879042 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 83 | 0.00321725 | 3461 | 0.00264022 | 0.000577037 | -0.00740012 | 0.00855419 | 0.887255 | 0.995263 | 0.952578 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_down | 83 | -0.0351171 | 3456 | -0.0390871 | 0.00397007 | -0.00386743 | 0.0118076 | 0.320791 | 0.714689 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_up | 83 | 0.037982 | 3456 | 0.0358902 | 0.00209181 | -0.0056569 | 0.00984052 | 0.596726 | 0.874893 | 0.830942 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 83 | 0.0140539 | 3456 | 0.00508386 | 0.00897006 | -0.00155973 | 0.0194998 | 0.0949838 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_down | 82 | -0.0518268 | 3447 | -0.0561062 | 0.00427938 | -0.00671792 | 0.0152767 | 0.445645 | 0.825755 | 0.804076 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_up | 82 | 0.0632734 | 3447 | 0.0548682 | 0.00840516 | -0.00575247 | 0.0225628 | 0.244578 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 82 | 0.023247 | 3447 | 0.010257 | 0.01299 | -0.00604328 | 0.0320233 | 0.181002 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_down | 83 | -0.0258482 | 3461 | -0.0271848 | 0.00133661 | -0.00420977 | 0.00688299 | 0.636687 | 0.874893 | 0.848753 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_up | 83 | 0.0248629 | 3461 | 0.0235308 | 0.00133216 | -0.00374104 | 0.00640536 | 0.606782 | 0.874893 | 0.835568 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 83 | 0.00311847 | 3461 | 0.00199146 | 0.00112701 | -0.00696516 | 0.00921918 | 0.784874 | 0.932964 | 0.890938 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_down | 83 | -0.035933 | 3456 | -0.039739 | 0.00380604 | -0.00383596 | 0.011448 | 0.328984 | 0.714689 | 0.761458 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_up | 83 | 0.0385384 | 3456 | 0.0355426 | 0.00299582 | -0.00505529 | 0.0110469 | 0.465808 | 0.834958 | 0.811408 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 83 | 0.0143388 | 3456 | 0.00375462 | 0.0105842 | -0.000173438 | 0.0213418 | 0.0538054 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_down | 82 | -0.0531843 | 3447 | -0.0570918 | 0.00390747 | -0.00716819 | 0.0149831 | 0.489261 | 0.834958 | 0.819466 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_up | 82 | 0.0634918 | 3447 | 0.0538618 | 0.00963005 | -0.00473304 | 0.0239931 | 0.188804 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 82 | 0.021211 | 3447 | 0.00762061 | 0.0135904 | -0.00564128 | 0.0328221 | 0.166032 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_down | 83 | -0.0184419 | 3461 | -0.0203271 | 0.00188523 | -0.00200271 | 0.00577318 | 0.341915 | 0.718022 | 0.770068 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_up | 83 | 0.0185825 | 3461 | 0.0196071 | -0.00102467 | -0.00430521 | 0.00225586 | 0.540402 | 0.874893 | 0.825499 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 83 | 0.00146054 | 3461 | 0.00144497 | 1.55659e-05 | -0.00527736 | 0.0053085 | 0.995401 | 0.995401 | 0.996719 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_down | 83 | -0.0253037 | 3456 | -0.0291539 | 0.00385013 | -0.00163927 | 0.00933953 | 0.169225 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_up | 83 | 0.0284307 | 3456 | 0.0292033 | -0.000772606 | -0.00545225 | 0.00390704 | 0.746245 | 0.904104 | 0.879042 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 83 | 0.00734033 | 3456 | 0.00271758 | 0.00462275 | -0.00336298 | 0.0126085 | 0.256544 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_down | 82 | -0.0360434 | 3447 | -0.0410725 | 0.0050291 | -0.0031819 | 0.0132401 | 0.229958 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_up | 82 | 0.0467481 | 3447 | 0.0439288 | 0.00281934 | -0.00663978 | 0.0122785 | 0.559094 | 0.874893 | 0.825499 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 82 | 0.0132821 | 3447 | 0.0054043 | 0.0078778 | -0.00468143 | 0.020437 | 0.218917 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_down | 83 | -0.0235287 | 3461 | -0.0245421 | 0.00101338 | -0.00382488 | 0.00585164 | 0.681422 | 0.904104 | 0.869284 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_up | 83 | 0.0220401 | 3461 | 0.0219769 | 6.32299e-05 | -0.00405815 | 0.00418461 | 0.976011 | 0.995401 | 0.983819 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 83 | 0.00168014 | 3461 | 0.00198571 | -0.000305567 | -0.00708434 | 0.0064732 | 0.929598 | 0.995263 | 0.961219 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_down | 83 | -0.0316967 | 3456 | -0.0355571 | 0.00386045 | -0.00280369 | 0.0105246 | 0.256207 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_up | 83 | 0.0340479 | 3456 | 0.0328747 | 0.00117317 | -0.00507703 | 0.00742337 | 0.712953 | 0.904104 | 0.874304 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 83 | 0.0123438 | 3456 | 0.00370462 | 0.00863921 | -0.000531157 | 0.0178096 | 0.0648233 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_down | 82 | -0.0471015 | 3447 | -0.0505897 | 0.00348813 | -0.00642425 | 0.0134005 | 0.490372 | 0.834958 | 0.819466 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_up | 82 | 0.0562176 | 3447 | 0.0497809 | 0.00643673 | -0.00602883 | 0.0189023 | 0.311506 | 0.714689 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 82 | 0.0186384 | 3447 | 0.00747629 | 0.0111621 | -0.00559493 | 0.0279192 | 0.191693 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_down | 83 | -0.0281532 | 3461 | -0.0277078 | -0.000445405 | -0.00743972 | 0.00654891 | 0.90067 | 0.995263 | 0.95633 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_up | 83 | 0.0290436 | 3461 | 0.0261222 | 0.00292138 | -0.00378889 | 0.00963165 | 0.39349 | 0.770472 | 0.79262 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 83 | 0.00535216 | 3461 | 0.00439684 | 0.000955318 | -0.00923444 | 0.0111451 | 0.854205 | 0.995263 | 0.938633 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_down | 83 | -0.0399541 | 3456 | -0.0403973 | 0.000443221 | -0.0104543 | 0.0113408 | 0.936463 | 0.995263 | 0.96191 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_up | 83 | 0.0449861 | 3456 | 0.0401518 | 0.00483428 | -0.0054424 | 0.015111 | 0.356524 | 0.724549 | 0.781253 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 83 | 0.0165577 | 3456 | 0.00859197 | 0.00796573 | -0.00544382 | 0.0213753 | 0.2443 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_down | 82 | -0.0514743 | 3447 | -0.0581744 | 0.00670011 | -0.00557704 | 0.0189773 | 0.284778 | 0.714689 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_up | 82 | 0.0716519 | 3447 | 0.0616775 | 0.0099744 | -0.00500078 | 0.0249496 | 0.191729 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 82 | 0.0351088 | 3447 | 0.0167957 | 0.0183131 | -0.000356883 | 0.036983 | 0.0545386 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_down | 83 | -0.0174114 | 3461 | -0.0189742 | 0.00156276 | -0.00210444 | 0.00522996 | 0.403581 | 0.770472 | 0.79463 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_up | 83 | 0.0166034 | 3461 | 0.0171596 | -0.000556153 | -0.0036465 | 0.00253419 | 0.72429 | 0.904104 | 0.874304 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 83 | 0.0011688 | 3461 | 0.00122766 | -5.88597e-05 | -0.00504071 | 0.00492299 | 0.981525 | 0.995401 | 0.987206 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_down | 83 | -0.0235875 | 3456 | -0.0272408 | 0.00365332 | -0.00147744 | 0.00878407 | 0.162834 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_up | 83 | 0.0245719 | 3456 | 0.0256837 | -0.00111178 | -0.00551539 | 0.00329184 | 0.620713 | 0.874893 | 0.842442 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 83 | 0.00619513 | 3456 | 0.00229956 | 0.00389557 | -0.00332428 | 0.0111154 | 0.290263 | 0.714689 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_down | 82 | -0.0334067 | 3447 | -0.0386224 | 0.00521571 | -0.00224677 | 0.0126782 | 0.17072 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_up | 82 | 0.0407696 | 3447 | 0.0386661 | 0.00210351 | -0.00668049 | 0.0108875 | 0.63881 | 0.874893 | 0.848753 | true |
| ma20_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 82 | 0.0118387 | 3447 | 0.00449792 | 0.00734077 | -0.00388263 | 0.0185642 | 0.199858 | 0.714689 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_down | 83 | -0.0208443 | 3459 | -0.0220579 | 0.00121358 | -0.00353565 | 0.00596282 | 0.616482 | 0.732799 | 0.839748 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_up | 83 | 0.0220071 | 3459 | 0.0195667 | 0.00244037 | -0.00157744 | 0.00645819 | 0.233857 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 5 | terminal_return | 83 | 0.00483462 | 3459 | 0.00163211 | 0.0032025 | -0.00313831 | 0.00954332 | 0.322212 | 0.599128 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_down | 83 | -0.0281215 | 3454 | -0.031851 | 0.00372953 | -0.00223965 | 0.0096987 | 0.220724 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_up | 83 | 0.0308828 | 3454 | 0.0292726 | 0.00161012 | -0.0043699 | 0.00759013 | 0.597687 | 0.732799 | 0.830942 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 10 | terminal_return | 83 | 0.0111804 | 3454 | 0.00318762 | 0.00799275 | 0.000132386 | 0.0158531 | 0.0462603 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_down | 82 | -0.0406812 | 3445 | -0.0453212 | 0.00464005 | -0.00353869 | 0.0128188 | 0.266152 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_up | 82 | 0.0484105 | 3445 | 0.0442544 | 0.00415615 | -0.00516371 | 0.013476 | 0.382089 | 0.599128 | 0.79262 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 全A | 20 | terminal_return | 82 | 0.0134324 | 3445 | 0.00649036 | 0.00694202 | -0.00620499 | 0.020089 | 0.300697 | 0.599128 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_down | 83 | -0.0258707 | 3461 | -0.0267593 | 0.000888675 | -0.00532273 | 0.00710008 | 0.779155 | 0.861171 | 0.890938 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_up | 83 | 0.027039 | 3461 | 0.023388 | 0.00365102 | -0.00195923 | 0.00926126 | 0.202125 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | terminal_return | 83 | 0.00723003 | 3461 | 0.00254398 | 0.00468604 | -0.00395239 | 0.0133245 | 0.287677 | 0.599128 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_down | 83 | -0.0358278 | 3456 | -0.0390701 | 0.00324222 | -0.00480536 | 0.0112898 | 0.429733 | 0.599128 | 0.794997 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_up | 83 | 0.0395051 | 3456 | 0.0358536 | 0.00365143 | -0.00551335 | 0.0128162 | 0.43486 | 0.599128 | 0.794997 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | terminal_return | 83 | 0.0155194 | 3456 | 0.00504866 | 0.0104708 | -0.000488229 | 0.0214298 | 0.0611124 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_down | 82 | -0.0517444 | 3447 | -0.0561082 | 0.00436374 | -0.00637807 | 0.0151056 | 0.425899 | 0.599128 | 0.794997 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_up | 82 | 0.0625804 | 3447 | 0.0548847 | 0.00769571 | -0.00540099 | 0.0207924 | 0.24944 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | terminal_return | 82 | 0.0201866 | 3447 | 0.0103298 | 0.00985679 | -0.00808188 | 0.0277955 | 0.281496 | 0.599128 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_down | 83 | -0.026047 | 3461 | -0.02718 | 0.00113299 | -0.004914 | 0.00717997 | 0.713445 | 0.817219 | 0.874304 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_up | 83 | 0.0275244 | 3461 | 0.023467 | 0.00405741 | -0.00151551 | 0.00963034 | 0.153582 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | terminal_return | 83 | 0.00695134 | 3461 | 0.00189954 | 0.0050518 | -0.00342328 | 0.0135269 | 0.242681 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_down | 83 | -0.0363661 | 3456 | -0.0397286 | 0.0033625 | -0.00437281 | 0.0110978 | 0.394213 | 0.599128 | 0.79262 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_up | 83 | 0.0398663 | 3456 | 0.0355107 | 0.00435564 | -0.00474316 | 0.0134544 | 0.34811 | 0.599128 | 0.772903 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | terminal_return | 83 | 0.0155163 | 3456 | 0.00372634 | 0.0117899 | 0.000926064 | 0.0226538 | 0.0334138 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_down | 82 | -0.0531791 | 3447 | -0.0570919 | 0.00391282 | -0.00688882 | 0.0147145 | 0.477706 | 0.631966 | 0.816437 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_up | 82 | 0.062676 | 3447 | 0.0538812 | 0.00879486 | -0.00436669 | 0.0219564 | 0.190291 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | terminal_return | 82 | 0.0176446 | 3447 | 0.00770545 | 0.00993919 | -0.00808248 | 0.0279609 | 0.279712 | 0.599128 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_down | 83 | -0.0183298 | 3461 | -0.0203298 | 0.00199999 | -0.00200891 | 0.00600889 | 0.328163 | 0.599128 | 0.761458 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_up | 83 | 0.0199212 | 3461 | 0.019575 | 0.000346171 | -0.0032874 | 0.00397974 | 0.851873 | 0.879803 | 0.938633 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | terminal_return | 83 | 0.00349363 | 3461 | 0.00139622 | 0.00209741 | -0.00319682 | 0.00739165 | 0.437459 | 0.599128 | 0.794997 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_down | 83 | -0.0249443 | 3456 | -0.0291625 | 0.00421825 | -0.00112247 | 0.00955896 | 0.121607 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_up | 83 | 0.0279363 | 3456 | 0.0292152 | -0.00127885 | -0.00622544 | 0.00366775 | 0.612352 | 0.732799 | 0.837139 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | terminal_return | 83 | 0.00838023 | 3456 | 0.00269261 | 0.00568762 | -0.00141957 | 0.0127948 | 0.116761 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_down | 82 | -0.0358808 | 3447 | -0.0410764 | 0.00519553 | -0.00256229 | 0.0129533 | 0.189304 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_up | 82 | 0.0446104 | 3447 | 0.0439797 | 0.000630789 | -0.00769826 | 0.00895984 | 0.881997 | 0.896223 | 0.951772 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | terminal_return | 82 | 0.0102936 | 3447 | 0.00547539 | 0.00481825 | -0.00701008 | 0.0166466 | 0.424637 | 0.599128 | 0.794997 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_down | 83 | -0.0238657 | 3461 | -0.024534 | 0.000668366 | -0.00487574 | 0.00621247 | 0.81321 | 0.866357 | 0.908105 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_up | 83 | 0.024872 | 3461 | 0.021909 | 0.00296303 | -0.00192505 | 0.00785111 | 0.234793 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | terminal_return | 83 | 0.00497937 | 3461 | 0.00190659 | 0.00307278 | -0.00438052 | 0.0105261 | 0.419061 | 0.599128 | 0.794997 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_down | 83 | -0.0321078 | 3456 | -0.0355472 | 0.0034394 | -0.00344039 | 0.0103192 | 0.327156 | 0.599128 | 0.761458 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_up | 83 | 0.0348182 | 3456 | 0.0328563 | 0.00196197 | -0.00525887 | 0.00918281 | 0.594344 | 0.732799 | 0.830942 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | terminal_return | 83 | 0.0129074 | 3456 | 0.00369109 | 0.00921631 | 4.0557e-05 | 0.0183921 | 0.0489918 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_down | 82 | -0.0471183 | 3447 | -0.0505893 | 0.00347098 | -0.00619398 | 0.0131359 | 0.481498 | 0.631966 | 0.816437 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_up | 82 | 0.054252 | 3447 | 0.0498276 | 0.00442437 | -0.00644938 | 0.0152981 | 0.425163 | 0.599128 | 0.794997 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | terminal_return | 82 | 0.0145093 | 3447 | 0.00757451 | 0.00693483 | -0.00859695 | 0.0224666 | 0.381505 | 0.599128 | 0.79262 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_down | 83 | -0.0284814 | 3461 | -0.0277 | -0.000781427 | -0.0077121 | 0.00614925 | 0.825102 | 0.866357 | 0.918444 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_up | 83 | 0.0312094 | 3461 | 0.0260702 | 0.00513919 | -0.00318224 | 0.0134606 | 0.2261 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | terminal_return | 83 | 0.010229 | 3461 | 0.00427989 | 0.00594907 | -0.00533307 | 0.0172312 | 0.301367 | 0.599128 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_down | 83 | -0.041604 | 3456 | -0.0403577 | -0.00124632 | -0.0122137 | 0.00972104 | 0.823744 | 0.866357 | 0.918444 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_up | 83 | 0.0474285 | 3456 | 0.0400931 | 0.00733536 | -0.00623698 | 0.0209077 | 0.289459 | 0.599128 | 0.759524 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | terminal_return | 83 | 0.0184102 | 3456 | 0.00854748 | 0.0098627 | -0.0053026 | 0.025028 | 0.202423 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_down | 82 | -0.0524708 | 3447 | -0.0581507 | 0.00567992 | -0.00675603 | 0.0181159 | 0.370681 | 0.599128 | 0.788579 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_up | 82 | 0.0724639 | 3447 | 0.0616582 | 0.0108057 | -0.00518912 | 0.0268005 | 0.185461 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | terminal_return | 82 | 0.0335596 | 3447 | 0.0168325 | 0.0167271 | -0.00220506 | 0.0356592 | 0.0833242 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_down | 83 | -0.0172662 | 3461 | -0.0189777 | 0.00171152 | -0.00217938 | 0.00560241 | 0.388599 | 0.599128 | 0.79262 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_up | 83 | 0.0178878 | 3461 | 0.0171288 | 0.000759021 | -0.00251196 | 0.00403 | 0.649244 | 0.757452 | 0.855481 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | terminal_return | 83 | 0.00286919 | 3461 | 0.00118688 | 0.00168231 | -0.00343152 | 0.00679614 | 0.519066 | 0.667371 | 0.824399 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_down | 83 | -0.0234668 | 3456 | -0.0272437 | 0.0037769 | -0.00138278 | 0.00893658 | 0.151365 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_up | 83 | 0.0249891 | 3456 | 0.0256737 | -0.000684575 | -0.00536675 | 0.0039976 | 0.774442 | 0.861171 | 0.889522 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | terminal_return | 83 | 0.00745704 | 3456 | 0.00226926 | 0.00518778 | -0.00134392 | 0.0117195 | 0.119536 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_down | 82 | -0.033163 | 3447 | -0.0386282 | 0.00546515 | -0.00159825 | 0.0125286 | 0.129391 | 0.599128 | 0.749588 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_up | 82 | 0.0391634 | 3447 | 0.0387043 | 0.000459123 | -0.00716995 | 0.0080882 | 0.906104 | 0.906104 | 0.958062 | true |
| ma20_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | terminal_return | 82 | 0.00948158 | 3447 | 0.00455399 | 0.00492758 | -0.00553804 | 0.0153932 | 0.356092 | 0.599128 | 0.781253 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_down | 82 | -0.0235352 | 3460 | -0.0219938 | -0.0015414 | -0.00661334 | 0.00353054 | 0.551403 | 0.868459 | 0.825499 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_up | 82 | 0.0222231 | 3460 | 0.0195623 | 0.00266076 | -0.00164899 | 0.00697051 | 0.226253 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 5 | terminal_return | 82 | 0.00463632 | 3460 | 0.00163774 | 0.00299858 | -0.00352923 | 0.00952639 | 0.367942 | 0.828149 | 0.787999 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_down | 82 | -0.0335659 | 3455 | -0.0317207 | -0.00184514 | -0.0107077 | 0.00701746 | 0.68323 | 0.871765 | 0.869284 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_up | 82 | 0.0333831 | 3455 | 0.0292138 | 0.00416936 | -0.00238197 | 0.0107207 | 0.212261 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 10 | terminal_return | 82 | 0.00746442 | 3455 | 0.00327812 | 0.0041863 | -0.00700866 | 0.0153813 | 0.4636 | 0.828149 | 0.810913 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_down | 82 | -0.0521854 | 3445 | -0.0450474 | -0.00713804 | -0.0250911 | 0.0108151 | 0.435813 | 0.828149 | 0.794997 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_up | 82 | 0.0516899 | 3445 | 0.0441763 | 0.00751361 | -0.00582743 | 0.0208546 | 0.269653 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 全A | 20 | terminal_return | 82 | 0.0055864 | 3445 | 0.00667711 | -0.00109071 | -0.0244553 | 0.0222739 | 0.927098 | 0.942051 | 0.961219 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_down | 82 | -0.0286111 | 3462 | -0.0266942 | -0.00191691 | -0.0076158 | 0.00378198 | 0.509718 | 0.831393 | 0.823391 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_up | 82 | 0.0255822 | 3462 | 0.0234236 | 0.00215867 | -0.0029064 | 0.00722374 | 0.403534 | 0.828149 | 0.79463 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 82 | 0.00534858 | 3462 | 0.0025899 | 0.00275868 | -0.00477999 | 0.0102974 | 0.473228 | 0.828149 | 0.816437 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_down | 82 | -0.0392963 | 3457 | -0.0389868 | -0.000309425 | -0.00962049 | 0.00900164 | 0.948067 | 0.948067 | 0.965803 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_up | 82 | 0.0407222 | 3457 | 0.0358258 | 0.00489641 | -0.00356245 | 0.0133553 | 0.256565 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 82 | 0.0126306 | 3457 | 0.00512021 | 0.00751043 | -0.00545324 | 0.0204741 | 0.256159 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_down | 82 | -0.0605675 | 3447 | -0.0558983 | -0.00466917 | -0.025125 | 0.0157867 | 0.654599 | 0.871765 | 0.859115 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_up | 82 | 0.06942 | 3447 | 0.054722 | 0.0146979 | -0.00727165 | 0.0366675 | 0.189768 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 82 | 0.0173182 | 3447 | 0.010398 | 0.00692013 | -0.0262477 | 0.040088 | 0.682588 | 0.871765 | 0.869284 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_down | 82 | -0.0295271 | 3462 | -0.0270973 | -0.00242982 | -0.00831849 | 0.00345885 | 0.418661 | 0.828149 | 0.794997 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_up | 82 | 0.0254696 | 3462 | 0.0235168 | 0.00195284 | -0.00311889 | 0.00702457 | 0.450436 | 0.828149 | 0.806371 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 82 | 0.00416983 | 3462 | 0.00196688 | 0.00220295 | -0.00548014 | 0.00988604 | 0.574126 | 0.87162 | 0.825499 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_down | 82 | -0.0412071 | 3457 | -0.0396128 | -0.00159431 | -0.0112574 | 0.00806882 | 0.746409 | 0.873283 | 0.879042 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_up | 82 | 0.0396253 | 3457 | 0.0355177 | 0.00410762 | -0.0043688 | 0.012584 | 0.342212 | 0.828149 | 0.770068 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 82 | 0.00999157 | 3457 | 0.0038608 | 0.00613078 | -0.00720542 | 0.019467 | 0.367572 | 0.828149 | 0.787999 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_down | 82 | -0.0625476 | 3447 | -0.0568691 | -0.0056785 | -0.02661 | 0.015253 | 0.594915 | 0.87162 | 0.830942 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_up | 82 | 0.0666499 | 3447 | 0.0537866 | 0.0128633 | -0.00902832 | 0.0347549 | 0.249454 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 82 | 0.0123517 | 3447 | 0.00783136 | 0.00452034 | -0.028767 | 0.0378077 | 0.790113 | 0.873283 | 0.895541 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_down | 82 | -0.0208711 | 3462 | -0.020269 | -0.000602132 | -0.00541491 | 0.00421065 | 0.806288 | 0.875796 | 0.904382 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_up | 82 | 0.0226355 | 3462 | 0.0195108 | 0.00312462 | -0.00164521 | 0.00789444 | 0.199157 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 82 | 0.00473403 | 3462 | 0.00136744 | 0.00336658 | -0.00323709 | 0.00997026 | 0.31769 | 0.828149 | 0.759524 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_down | 82 | -0.0311137 | 3457 | -0.0290149 | -0.00209878 | -0.0106488 | 0.00645122 | 0.630428 | 0.871765 | 0.848753 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_up | 82 | 0.0321625 | 3457 | 0.0291145 | 0.00304797 | -0.00415902 | 0.010255 | 0.40715 | 0.828149 | 0.794658 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 82 | 0.0048198 | 3457 | 0.00277871 | 0.00204109 | -0.00897185 | 0.013054 | 0.716412 | 0.871765 | 0.874304 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_down | 82 | -0.0482111 | 3447 | -0.040783 | -0.00742807 | -0.023107 | 0.00825085 | 0.353111 | 0.828149 | 0.778818 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_up | 82 | 0.0458803 | 3447 | 0.0439494 | 0.0019309 | -0.00860937 | 0.0124712 | 0.719552 | 0.871765 | 0.874304 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 82 | 0.00086093 | 3447 | 0.00569979 | -0.00483886 | -0.0225301 | 0.0128524 | 0.591895 | 0.87162 | 0.830942 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_down | 82 | -0.0277994 | 3462 | -0.0244407 | -0.00335874 | -0.00923448 | 0.00251699 | 0.262546 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_up | 82 | 0.023823 | 3462 | 0.0219347 | 0.0018883 | -0.00259715 | 0.00637375 | 0.409299 | 0.828149 | 0.794658 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 82 | 0.00296973 | 3462 | 0.00195508 | 0.00101466 | -0.00601438 | 0.0080437 | 0.77723 | 0.873283 | 0.890282 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_down | 82 | -0.039083 | 3457 | -0.0353808 | -0.0037022 | -0.0133311 | 0.00592671 | 0.451091 | 0.828149 | 0.806371 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_up | 82 | 0.0354049 | 3457 | 0.0328429 | 0.00256195 | -0.00425344 | 0.00937735 | 0.461258 | 0.828149 | 0.810913 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 82 | 0.00643024 | 3457 | 0.00384739 | 0.00258285 | -0.00927967 | 0.0144454 | 0.669558 | 0.871765 | 0.863128 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_down | 82 | -0.0584194 | 3447 | -0.0503204 | -0.00809897 | -0.0277312 | 0.0115333 | 0.418764 | 0.828149 | 0.794997 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_up | 82 | 0.0568818 | 3447 | 0.0497651 | 0.00711672 | -0.00871307 | 0.0229465 | 0.378225 | 0.828149 | 0.79262 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 82 | 0.00592783 | 3447 | 0.00777866 | -0.00185082 | -0.0285756 | 0.024874 | 0.892027 | 0.942051 | 0.955202 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_down | 82 | -0.0279978 | 3462 | -0.0277116 | -0.000286194 | -0.00637622 | 0.00580383 | 0.926612 | 0.942051 | 0.961219 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_up | 82 | 0.0269089 | 3462 | 0.0261736 | 0.000735283 | -0.00382287 | 0.00529344 | 0.751873 | 0.873283 | 0.881265 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 82 | 0.0068297 | 3462 | 0.00436212 | 0.00246758 | -0.00476129 | 0.00969644 | 0.503466 | 0.831393 | 0.819466 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_down | 82 | -0.0388607 | 3457 | -0.0404231 | 0.00156241 | -0.00860097 | 0.0117258 | 0.763179 | 0.873283 | 0.883558 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_up | 82 | 0.0431694 | 3457 | 0.0401963 | 0.00297309 | -0.0043661 | 0.0103123 | 0.4272 | 0.828149 | 0.794997 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 82 | 0.0148898 | 3457 | 0.00863383 | 0.00625593 | -0.00665478 | 0.0191666 | 0.342252 | 0.828149 | 0.770068 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_down | 82 | -0.0651146 | 3447 | -0.0578499 | -0.00726474 | -0.0291176 | 0.0145881 | 0.514672 | 0.831393 | 0.82345 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_up | 82 | 0.0716263 | 3447 | 0.0616781 | 0.00994814 | -0.00658304 | 0.0264793 | 0.238203 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 82 | 0.0158276 | 3447 | 0.0172544 | -0.00142679 | -0.0311933 | 0.0283397 | 0.92515 | 0.942051 | 0.961219 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_down | 82 | -0.0208842 | 3462 | -0.0188915 | -0.00199277 | -0.00697206 | 0.00298652 | 0.432797 | 0.828149 | 0.794997 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_up | 82 | 0.0202461 | 3462 | 0.0170731 | 0.003173 | -0.00128997 | 0.00763597 | 0.163473 | 0.828149 | 0.749588 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 82 | 0.00350095 | 3462 | 0.0011724 | 0.00232855 | -0.00396388 | 0.00862098 | 0.468262 | 0.828149 | 0.813538 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_down | 82 | -0.0304666 | 3457 | -0.0270766 | -0.00338999 | -0.0119244 | 0.00514438 | 0.436248 | 0.828149 | 0.794997 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_up | 82 | 0.0289787 | 3457 | 0.0255789 | 0.00339989 | -0.00332225 | 0.010122 | 0.32153 | 0.828149 | 0.759524 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 82 | 0.00441789 | 3457 | 0.00234285 | 0.00207504 | -0.00823483 | 0.0123849 | 0.693223 | 0.871765 | 0.87056 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_down | 82 | -0.0466168 | 3447 | -0.0383081 | -0.00830871 | -0.0240199 | 0.00740253 | 0.299958 | 0.828149 | 0.759524 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_up | 82 | 0.0425747 | 3447 | 0.0386231 | 0.00395159 | -0.0061416 | 0.0140448 | 0.442867 | 0.828149 | 0.800974 | true |
| ma20_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 82 | 0.000698867 | 3447 | 0.00476292 | -0.00406406 | -0.0214713 | 0.0133432 | 0.647241 | 0.871765 | 0.855481 | true |
| ma20_breadth_reversal_top | top | onset | 全A | 5 | max_down | 82 | -0.0237959 | 3460 | -0.0219876 | -0.00180829 | -0.00729616 | 0.00367959 | 0.518389 | 0.956908 | 0.824399 | true |
| ma20_breadth_reversal_top | top | onset | 全A | 5 | max_up | 82 | 0.0207042 | 3460 | 0.0195983 | 0.00110591 | -0.00315265 | 0.00536447 | 0.610757 | 0.956908 | 0.837139 | true |
| ma20_breadth_reversal_top | top | onset | 全A | 5 | terminal_return | 82 | 0.00216149 | 3460 | 0.00169639 | 0.000465102 | -0.00636166 | 0.00729186 | 0.893772 | 0.956908 | 0.955716 | true |
| ma20_breadth_reversal_top | top | onset | 全A | 10 | max_down | 82 | -0.0343229 | 3455 | -0.0317028 | -0.00262015 | -0.0117899 | 0.00654959 | 0.575447 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 全A | 10 | max_up | 82 | 0.0325095 | 3455 | 0.0292345 | 0.00327505 | -0.0033132 | 0.0098633 | 0.329896 | 0.956908 | 0.761458 | true |
| ma20_breadth_reversal_top | top | onset | 全A | 10 | terminal_return | 82 | 0.00733654 | 3455 | 0.00328116 | 0.00405538 | -0.00588399 | 0.0139948 | 0.423882 | 0.956908 | 0.794997 | true |
| ma20_breadth_reversal_top | top | onset | 全A | 20 | max_down | 82 | -0.0521806 | 3445 | -0.0450475 | -0.00713312 | -0.0243625 | 0.0100963 | 0.417104 | 0.956908 | 0.794997 | true |
| ma20_breadth_reversal_top | top | onset | 全A | 20 | max_up | 82 | 0.0509047 | 3445 | 0.044195 | 0.00670969 | -0.00603554 | 0.0194549 | 0.302149 | 0.956908 | 0.759524 | true |
| ma20_breadth_reversal_top | top | onset | 全A | 20 | terminal_return | 82 | 0.00265 | 3445 | 0.00674701 | -0.004097 | -0.026781 | 0.018587 | 0.723339 | 0.956908 | 0.874304 | true |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 5 | max_down | 82 | -0.0277348 | 3462 | -0.0267149 | -0.00101984 | -0.00696447 | 0.00492479 | 0.736682 | 0.956908 | 0.878079 | true |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 5 | max_up | 82 | 0.0250362 | 3462 | 0.0234365 | 0.00159972 | -0.00354137 | 0.00674081 | 0.541941 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 5 | terminal_return | 82 | 0.00366952 | 3462 | 0.00262967 | 0.00103984 | -0.00678989 | 0.00886958 | 0.794631 | 0.956908 | 0.897968 | true |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 10 | max_down | 82 | -0.039714 | 3457 | -0.0389769 | -0.000737078 | -0.0105552 | 0.00908106 | 0.883019 | 0.956908 | 0.951772 | true |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 10 | max_up | 82 | 0.0410691 | 3457 | 0.0358176 | 0.00525154 | -0.00316413 | 0.0136672 | 0.2213 | 0.956908 | 0.749588 | true |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 10 | terminal_return | 82 | 0.0135324 | 3457 | 0.00509882 | 0.00843362 | -0.00325437 | 0.0201216 | 0.157284 | 0.956908 | 0.749588 | true |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 20 | max_down | 82 | -0.0595214 | 3447 | -0.0559232 | -0.00359819 | -0.0228663 | 0.0156699 | 0.714353 | 0.956908 | 0.874304 | true |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 20 | max_up | 82 | 0.0682056 | 3447 | 0.0547509 | 0.0134547 | -0.00670205 | 0.0336114 | 0.190769 | 0.956908 | 0.749588 | true |
| ma20_breadth_reversal_top | top | onset | 国证2000 | 20 | terminal_return | 82 | 0.0147895 | 3447 | 0.0104582 | 0.00433128 | -0.0266974 | 0.03536 | 0.784396 | 0.956908 | 0.890938 | true |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 5 | max_down | 82 | -0.0288561 | 3462 | -0.0271131 | -0.00174294 | -0.0078154 | 0.00432952 | 0.573729 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 5 | max_up | 82 | 0.0250347 | 3462 | 0.0235271 | 0.00150758 | -0.00361914 | 0.0066343 | 0.564369 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 5 | terminal_return | 82 | 0.00236402 | 3462 | 0.00200965 | 0.000354366 | -0.00753796 | 0.00824669 | 0.929873 | 0.956908 | 0.961219 | true |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 10 | max_down | 82 | -0.0415588 | 3457 | -0.0396045 | -0.00195432 | -0.0119877 | 0.00807909 | 0.702631 | 0.956908 | 0.873816 | true |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 10 | max_up | 82 | 0.0397759 | 3457 | 0.0355141 | 0.00426184 | -0.00405617 | 0.0125799 | 0.315267 | 0.956908 | 0.759524 | true |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 10 | terminal_return | 82 | 0.0106109 | 3457 | 0.00384611 | 0.00676482 | -0.00518213 | 0.0187118 | 0.267074 | 0.956908 | 0.749588 | true |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 20 | max_down | 82 | -0.0614939 | 3447 | -0.0568941 | -0.00459974 | -0.024253 | 0.0150535 | 0.64643 | 0.956908 | 0.855481 | true |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 20 | max_up | 82 | 0.0654851 | 3447 | 0.0538143 | 0.0116707 | -0.00835712 | 0.0316985 | 0.253396 | 0.956908 | 0.749588 | true |
| ma20_breadth_reversal_top | top | onset | 中证1000 | 20 | terminal_return | 82 | 0.00961148 | 3447 | 0.00789655 | 0.00171493 | -0.0294905 | 0.0329204 | 0.914223 | 0.956908 | 0.961219 | true |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 5 | max_down | 82 | -0.0218356 | 3462 | -0.0202462 | -0.00158945 | -0.00692949 | 0.00375058 | 0.55963 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 5 | max_up | 82 | 0.0201129 | 3462 | 0.0195706 | 0.000542324 | -0.00375334 | 0.00483799 | 0.804561 | 0.956908 | 0.904058 | true |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 5 | terminal_return | 82 | 0.00165161 | 3462 | 0.00144045 | 0.000211159 | -0.00657537 | 0.00699769 | 0.951372 | 0.956908 | 0.966716 | true |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 10 | max_down | 82 | -0.0324 | 3457 | -0.0289844 | -0.00341553 | -0.0123007 | 0.0054696 | 0.451184 | 0.956908 | 0.806371 | true |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 10 | max_up | 82 | 0.0304978 | 3457 | 0.029154 | 0.00134373 | -0.00589463 | 0.00858209 | 0.715967 | 0.956908 | 0.874304 | true |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 10 | terminal_return | 82 | 0.00376562 | 3457 | 0.00280371 | 0.000961906 | -0.00926204 | 0.0111858 | 0.853697 | 0.956908 | 0.938633 | true |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 20 | max_down | 82 | -0.0490336 | 3447 | -0.0407635 | -0.00827016 | -0.0236195 | 0.00707913 | 0.290948 | 0.956908 | 0.759524 | true |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 20 | max_up | 82 | 0.0451601 | 3447 | 0.0439666 | 0.00119351 | -0.00945469 | 0.0118417 | 0.826114 | 0.956908 | 0.918444 | true |
| ma20_breadth_reversal_top | top | onset | 沪深300 | 20 | terminal_return | 82 | -0.00247794 | 3447 | 0.00577921 | -0.00825716 | -0.0265651 | 0.0100508 | 0.376702 | 0.956908 | 0.79262 | true |
| ma20_breadth_reversal_top | top | onset | 中证500 | 5 | max_down | 82 | -0.0275091 | 3462 | -0.0244475 | -0.00306157 | -0.00919368 | 0.00307054 | 0.327794 | 0.956908 | 0.761458 | true |
| ma20_breadth_reversal_top | top | onset | 中证500 | 5 | max_up | 82 | 0.0225997 | 3462 | 0.0219637 | 0.000636057 | -0.00409873 | 0.00537084 | 0.792319 | 0.956908 | 0.896696 | true |
| ma20_breadth_reversal_top | top | onset | 中证500 | 5 | terminal_return | 82 | 0.00060542 | 3462 | 0.00201108 | -0.00140566 | -0.00893327 | 0.00612196 | 0.714367 | 0.956908 | 0.874304 | true |
| ma20_breadth_reversal_top | top | onset | 中证500 | 10 | max_down | 82 | -0.0395377 | 3457 | -0.03537 | -0.00416767 | -0.0141482 | 0.00581285 | 0.413096 | 0.956908 | 0.794658 | true |
| ma20_breadth_reversal_top | top | onset | 中证500 | 10 | max_up | 82 | 0.0349699 | 3457 | 0.0328532 | 0.0021167 | -0.00491158 | 0.00914497 | 0.554996 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 中证500 | 10 | terminal_return | 82 | 0.00703951 | 3457 | 0.00383294 | 0.00320658 | -0.00745773 | 0.0138709 | 0.555634 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 中证500 | 20 | max_down | 82 | -0.0578289 | 3447 | -0.0503345 | -0.00749442 | -0.0261355 | 0.0111466 | 0.4307 | 0.956908 | 0.794997 | true |
| ma20_breadth_reversal_top | top | onset | 中证500 | 20 | max_up | 82 | 0.0559853 | 3447 | 0.0497864 | 0.00619888 | -0.00848182 | 0.0208796 | 0.407895 | 0.956908 | 0.794658 | true |
| ma20_breadth_reversal_top | top | onset | 中证500 | 20 | terminal_return | 82 | 0.00290692 | 3447 | 0.00785052 | -0.0049436 | -0.0303137 | 0.0204265 | 0.702517 | 0.956908 | 0.873816 | true |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 5 | max_down | 82 | -0.0273676 | 3462 | -0.0277266 | 0.00035895 | -0.00595892 | 0.00667682 | 0.911333 | 0.956908 | 0.960903 | true |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 5 | max_up | 82 | 0.0263285 | 3462 | 0.0261873 | 0.000141206 | -0.0049808 | 0.00526321 | 0.956908 | 0.956908 | 0.971037 | true |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 5 | terminal_return | 82 | 0.00509916 | 3462 | 0.00440311 | 0.000696047 | -0.00722691 | 0.008619 | 0.863288 | 0.956908 | 0.943083 | true |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 10 | max_down | 82 | -0.0388263 | 3457 | -0.0404239 | 0.00159756 | -0.00895006 | 0.0121452 | 0.76657 | 0.956908 | 0.884773 | true |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 10 | max_up | 82 | 0.0428899 | 3457 | 0.0402029 | 0.00268695 | -0.00483457 | 0.0102085 | 0.483814 | 0.956908 | 0.816437 | true |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 10 | terminal_return | 82 | 0.0163688 | 3457 | 0.00859875 | 0.00777007 | -0.00397968 | 0.0195198 | 0.194927 | 0.956908 | 0.749588 | true |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 20 | max_down | 82 | -0.0642802 | 3447 | -0.0578698 | -0.00641043 | -0.027356 | 0.0145352 | 0.548599 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 20 | max_up | 82 | 0.0716395 | 3447 | 0.0616778 | 0.00996166 | -0.00528184 | 0.0252052 | 0.200241 | 0.956908 | 0.749588 | true |
| ma20_breadth_reversal_top | top | onset | 微盘股 | 20 | terminal_return | 82 | 0.0133224 | 3447 | 0.017314 | -0.00399159 | -0.0324583 | 0.0244751 | 0.783446 | 0.956908 | 0.890938 | true |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 5 | max_down | 82 | -0.0219053 | 3462 | -0.0188673 | -0.00303797 | -0.00877314 | 0.00269721 | 0.299164 | 0.956908 | 0.759524 | true |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 5 | max_up | 82 | 0.0182649 | 3462 | 0.01712 | 0.0011449 | -0.0029508 | 0.00524061 | 0.583765 | 0.956908 | 0.830942 | true |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 5 | terminal_return | 82 | 0.000501683 | 3462 | 0.00124344 | -0.000741759 | -0.00756102 | 0.00607751 | 0.831173 | 0.956908 | 0.922363 | true |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 10 | max_down | 82 | -0.0316865 | 3457 | -0.0270476 | -0.00463884 | -0.0136421 | 0.00436442 | 0.312557 | 0.956908 | 0.759524 | true |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 10 | max_up | 82 | 0.0276135 | 3457 | 0.0256112 | 0.0020023 | -0.00488827 | 0.00889286 | 0.568985 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 10 | terminal_return | 82 | 0.00376741 | 3457 | 0.00235828 | 0.00140913 | -0.00827047 | 0.0110887 | 0.77539 | 0.956908 | 0.889522 | true |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 20 | max_down | 82 | -0.047314 | 3447 | -0.0382915 | -0.00902243 | -0.0244639 | 0.00641901 | 0.252115 | 0.956908 | 0.749588 | true |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 20 | max_up | 82 | 0.0418307 | 3447 | 0.0386408 | 0.00318983 | -0.00664803 | 0.0130277 | 0.525095 | 0.956908 | 0.825499 | true |
| ma20_breadth_reversal_top | top | onset | 上证指数 | 20 | terminal_return | 82 | -0.00272766 | 3447 | 0.00484444 | -0.0075721 | -0.0253425 | 0.0101983 | 0.403622 | 0.956908 | 0.79463 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_down | 77 | -0.0231921 | 3465 | -0.0220036 | -0.00118845 | -0.00721638 | 0.00483947 | 0.699179 | 0.975334 | 0.873684 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | max_up | 77 | 0.0189636 | 3465 | 0.0196386 | -0.00067503 | -0.00435956 | 0.0030095 | 0.719531 | 0.975334 | 0.874304 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 5 | terminal_return | 77 | 0.000172637 | 3465 | 0.00174126 | -0.00156862 | -0.00897078 | 0.00583354 | 0.677885 | 0.975334 | 0.869284 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_down | 76 | -0.0332629 | 3461 | -0.0317306 | -0.0015323 | -0.0113375 | 0.00827291 | 0.759379 | 0.975334 | 0.883558 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | max_up | 76 | 0.0303589 | 3461 | 0.0292874 | 0.00107154 | -0.00585159 | 0.00799466 | 0.761614 | 0.975334 | 0.883558 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 10 | terminal_return | 76 | 0.00281215 | 3461 | 0.00338754 | -0.000575393 | -0.0134347 | 0.0122839 | 0.930115 | 0.975334 | 0.961219 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_down | 76 | -0.0430934 | 3451 | -0.04526 | 0.00216664 | -0.00999772 | 0.014331 | 0.727012 | 0.975334 | 0.874304 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | max_up | 76 | 0.0606599 | 3451 | 0.0439918 | 0.016668 | -0.00688008 | 0.0402162 | 0.165337 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 全A | 20 | terminal_return | 76 | 0.0215338 | 3451 | 0.00632401 | 0.0152098 | -0.00704969 | 0.0374693 | 0.180486 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_down | 77 | -0.0262123 | 3467 | -0.0267502 | 0.000537895 | -0.0060799 | 0.00715569 | 0.873426 | 0.975334 | 0.94736 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | max_up | 77 | 0.0245447 | 3467 | 0.0234497 | 0.00109503 | -0.0039629 | 0.00615295 | 0.671322 | 0.975334 | 0.863128 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 77 | 0.00415768 | 3467 | 0.00262033 | 0.00153735 | -0.00749361 | 0.0105683 | 0.738641 | 0.975334 | 0.878079 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_down | 76 | -0.0390726 | 3463 | -0.0389923 | -8.03016e-05 | -0.0115068 | 0.0113462 | 0.98901 | 0.98901 | 0.99295 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | max_up | 76 | 0.039794 | 3463 | 0.0358547 | 0.00393929 | -0.00548206 | 0.0133606 | 0.412488 | 0.975334 | 0.794658 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 76 | 0.00810502 | 3463 | 0.00523255 | 0.00287247 | -0.0131452 | 0.0188902 | 0.725221 | 0.975334 | 0.874304 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_down | 76 | -0.0509124 | 3453 | -0.0561189 | 0.0052065 | -0.00925417 | 0.0196672 | 0.480381 | 0.975334 | 0.816437 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | max_up | 76 | 0.0765345 | 3453 | 0.054591 | 0.0219435 | -0.00400055 | 0.0478875 | 0.097364 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 76 | 0.0322298 | 3453 | 0.0100819 | 0.0221479 | -0.00317805 | 0.0474739 | 0.08652 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_down | 77 | -0.0267212 | 3467 | -0.0271631 | 0.000441872 | -0.00637008 | 0.00725382 | 0.89883 | 0.975334 | 0.95633 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | max_up | 77 | 0.024977 | 3467 | 0.0235305 | 0.00144644 | -0.00352486 | 0.00641773 | 0.56849 | 0.975334 | 0.825499 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 77 | 0.00348443 | 3467 | 0.00198528 | 0.00149915 | -0.00767272 | 0.010671 | 0.748693 | 0.975334 | 0.880267 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_down | 76 | -0.0401383 | 3463 | -0.039639 | -0.000499273 | -0.0122146 | 0.011216 | 0.93343 | 0.975334 | 0.961219 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | max_up | 76 | 0.0399498 | 3463 | 0.0355177 | 0.00443219 | -0.00504303 | 0.0139074 | 0.359236 | 0.975334 | 0.781453 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 76 | 0.00652861 | 3463 | 0.00394742 | 0.00258119 | -0.0136585 | 0.0188208 | 0.755398 | 0.975334 | 0.882922 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_down | 76 | -0.0527099 | 3453 | -0.0570954 | 0.00438554 | -0.010459 | 0.0192301 | 0.562559 | 0.975334 | 0.825499 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | max_up | 76 | 0.0761993 | 3453 | 0.0535988 | 0.0226005 | -0.00342477 | 0.0486257 | 0.0887413 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 76 | 0.0296673 | 3453 | 0.0074581 | 0.0222092 | -0.00386136 | 0.0482798 | 0.0949786 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_down | 77 | -0.0218872 | 3467 | -0.0202473 | -0.0016399 | -0.00736934 | 0.00408953 | 0.574798 | 0.975334 | 0.825499 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | max_up | 77 | 0.0177333 | 3467 | 0.0196242 | -0.00189097 | -0.00516465 | 0.00138271 | 0.25757 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 77 | -0.00192742 | 3467 | 0.00152024 | -0.00344766 | -0.0102368 | 0.00334151 | 0.319579 | 0.975334 | 0.759524 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_down | 76 | -0.0317817 | 3463 | -0.0290039 | -0.00277775 | -0.0122285 | 0.00667302 | 0.564562 | 0.975334 | 0.825499 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | max_up | 76 | 0.0272171 | 3463 | 0.0292284 | -0.00201131 | -0.00796171 | 0.00393909 | 0.507649 | 0.975334 | 0.823177 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 76 | -0.000142556 | 3463 | 0.00289115 | -0.00303371 | -0.0145741 | 0.00850666 | 0.606385 | 0.975334 | 0.835568 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_down | 76 | -0.0403706 | 3453 | -0.0409685 | 0.000597894 | -0.0108533 | 0.0120491 | 0.91849 | 0.975334 | 0.961219 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | max_up | 76 | 0.054464 | 3453 | 0.0437639 | 0.0107002 | -0.0115562 | 0.0329565 | 0.346037 | 0.975334 | 0.772903 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 76 | 0.0157545 | 3453 | 0.00536357 | 0.0103909 | -0.0105988 | 0.0313806 | 0.331901 | 0.975334 | 0.762666 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_down | 77 | -0.0249919 | 3467 | -0.0245079 | -0.000484023 | -0.00697439 | 0.00600635 | 0.883789 | 0.975334 | 0.951772 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | max_up | 77 | 0.0212641 | 3467 | 0.0219942 | -0.000730187 | -0.00498451 | 0.00352413 | 0.736567 | 0.975334 | 0.878079 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 77 | 0.00167438 | 3467 | 0.00198531 | -0.000310926 | -0.0082785 | 0.00765665 | 0.939032 | 0.975334 | 0.961935 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_down | 76 | -0.0367969 | 3463 | -0.0354374 | -0.00135952 | -0.0120463 | 0.00932725 | 0.803097 | 0.975334 | 0.904058 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | max_up | 76 | 0.0341108 | 3463 | 0.0328757 | 0.00123506 | -0.00673894 | 0.00920906 | 0.76145 | 0.975334 | 0.883558 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 76 | 0.00363683 | 3463 | 0.00391317 | -0.00027634 | -0.0146053 | 0.0140527 | 0.969848 | 0.98549 | 0.98022 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_down | 76 | -0.0486517 | 3453 | -0.0505495 | 0.00189782 | -0.011578 | 0.0153737 | 0.782525 | 0.975334 | 0.890938 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | max_up | 76 | 0.0671238 | 3453 | 0.049552 | 0.0175719 | -0.0072797 | 0.0424234 | 0.165789 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 76 | 0.0233631 | 3453 | 0.00739169 | 0.0159714 | -0.00801273 | 0.0399556 | 0.191827 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_down | 77 | -0.027471 | 3467 | -0.0277238 | 0.000252781 | -0.00684767 | 0.00735323 | 0.944371 | 0.975334 | 0.96479 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | max_up | 77 | 0.027192 | 3467 | 0.0261683 | 0.00102366 | -0.00583759 | 0.00788492 | 0.769965 | 0.975334 | 0.887337 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 77 | 0.00750414 | 3467 | 0.0043507 | 0.00315345 | -0.00646779 | 0.0127747 | 0.520609 | 0.975334 | 0.825116 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_down | 76 | -0.0399463 | 3463 | -0.0403966 | 0.00045023 | -0.0115264 | 0.0124268 | 0.941264 | 0.975334 | 0.962917 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | max_up | 76 | 0.0467821 | 3463 | 0.0401221 | 0.00665999 | -0.00489994 | 0.0182199 | 0.25881 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 76 | 0.016089 | 3463 | 0.00861835 | 0.00747069 | -0.00957442 | 0.0245158 | 0.390315 | 0.975334 | 0.79262 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_down | 76 | -0.0491967 | 3453 | -0.0582129 | 0.00901622 | -0.00563031 | 0.0236627 | 0.227605 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | max_up | 76 | 0.0863177 | 3453 | 0.0613721 | 0.0249456 | -0.00219912 | 0.0520904 | 0.0716695 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 76 | 0.0449571 | 3453 | 0.0166107 | 0.0283464 | 0.00277435 | 0.0539184 | 0.0298072 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_down | 77 | -0.0204231 | 3467 | -0.0189046 | -0.00151855 | -0.00678543 | 0.00374833 | 0.571999 | 0.975334 | 0.825499 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | max_up | 77 | 0.015389 | 3467 | 0.0171856 | -0.00179661 | -0.00464505 | 0.00105184 | 0.216372 | 0.975334 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 77 | -0.001133 | 3467 | 0.00127868 | -0.00241168 | -0.00853854 | 0.00371518 | 0.440409 | 0.975334 | 0.798439 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_down | 76 | -0.0295323 | 3463 | -0.0271029 | -0.00242935 | -0.0113249 | 0.0064662 | 0.592462 | 0.975334 | 0.830942 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | max_up | 76 | 0.0240322 | 3463 | 0.0256933 | -0.00166112 | -0.00705181 | 0.00372957 | 0.545866 | 0.975334 | 0.825499 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 76 | -0.000311729 | 3463 | 0.00245024 | -0.00276197 | -0.013736 | 0.00821205 | 0.621802 | 0.975334 | 0.842442 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_down | 76 | -0.037729 | 3453 | -0.0385182 | 0.000789131 | -0.0100942 | 0.0116725 | 0.886989 | 0.975334 | 0.952578 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | max_up | 76 | 0.04829 | 3453 | 0.0385042 | 0.00978583 | -0.0095709 | 0.0291426 | 0.321743 | 0.975334 | 0.759524 | true |
| ma60_breadth_reversal_bottom | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 76 | 0.0136125 | 3453 | 0.00447163 | 0.00914089 | -0.0088792 | 0.027161 | 0.32011 | 0.975334 | 0.759524 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_down | 76 | -0.0212019 | 3466 | -0.0220476 | 0.000845709 | -0.0039524 | 0.00564381 | 0.729743 | 0.86743 | 0.874304 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 5 | max_up | 76 | 0.0190417 | 3466 | 0.0196367 | -0.000595004 | -0.00419302 | 0.00300301 | 0.745843 | 0.87015 | 0.879042 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 5 | terminal_return | 76 | 0.00206462 | 3466 | 0.00169932 | 0.0003653 | -0.00537088 | 0.00610148 | 0.900666 | 0.957807 | 0.95633 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_down | 76 | -0.0317804 | 3461 | -0.0317631 | -1.7251e-05 | -0.00826109 | 0.00822659 | 0.996727 | 0.996727 | 0.996727 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 10 | max_up | 76 | 0.0314677 | 3461 | 0.0292631 | 0.0022046 | -0.00390995 | 0.00831914 | 0.479767 | 0.719651 | 0.816437 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 10 | terminal_return | 76 | 0.00647009 | 3461 | 0.00330722 | 0.00316287 | -0.0083193 | 0.014645 | 0.589266 | 0.8025 | 0.830942 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_down | 75 | -0.0428237 | 3452 | -0.0452653 | 0.0024416 | -0.00911027 | 0.0139935 | 0.67868 | 0.855137 | 0.869284 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 20 | max_up | 75 | 0.0597645 | 3452 | 0.0440161 | 0.0157484 | -0.0057094 | 0.0372063 | 0.150294 | 0.469876 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 全A | 20 | terminal_return | 75 | 0.0245778 | 3452 | 0.00626228 | 0.0183156 | -0.00205969 | 0.0386908 | 0.0780917 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_down | 77 | -0.0225443 | 3467 | -0.0268317 | 0.0042874 | -0.00101756 | 0.00959237 | 0.113183 | 0.419444 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | max_up | 77 | 0.0263811 | 3467 | 0.0234089 | 0.00297218 | -0.00209916 | 0.00804352 | 0.250678 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 5 | terminal_return | 77 | 0.00720651 | 3467 | 0.00255262 | 0.00465389 | -0.00288718 | 0.012195 | 0.226435 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_down | 77 | -0.0351399 | 3462 | -0.0390797 | 0.00393986 | -0.00586781 | 0.0137475 | 0.431074 | 0.69635 | 0.794997 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | max_up | 77 | 0.0438579 | 3462 | 0.0357632 | 0.00809476 | -0.000882146 | 0.0170717 | 0.0771623 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 10 | terminal_return | 77 | 0.015085 | 3462 | 0.00507647 | 0.0100085 | -0.0046336 | 0.0246506 | 0.180328 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_down | 76 | -0.0479577 | 3453 | -0.0561839 | 0.00822621 | -0.00561599 | 0.0220684 | 0.244101 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | max_up | 76 | 0.0774739 | 3453 | 0.0545703 | 0.0229036 | -0.00082062 | 0.0466279 | 0.0584638 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 国证2000 | 20 | terminal_return | 76 | 0.0374079 | 3453 | 0.00996789 | 0.02744 | 0.00357096 | 0.051309 | 0.0242447 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_down | 77 | -0.0233773 | 3467 | -0.0272373 | 0.00386001 | -0.00152867 | 0.00924869 | 0.160324 | 0.469876 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | max_up | 77 | 0.0264255 | 3467 | 0.0234984 | 0.00292713 | -0.0020118 | 0.00786605 | 0.245389 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 5 | terminal_return | 77 | 0.00627594 | 3467 | 0.00192328 | 0.00435266 | -0.00315257 | 0.0118579 | 0.255662 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_down | 77 | -0.0366217 | 3462 | -0.0397171 | 0.00309536 | -0.00689423 | 0.0130849 | 0.543637 | 0.77839 | 0.825499 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | max_up | 77 | 0.0432967 | 3462 | 0.0354419 | 0.00785476 | -0.00103669 | 0.0167462 | 0.0833668 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 10 | terminal_return | 77 | 0.0128976 | 3462 | 0.00380502 | 0.00909256 | -0.0056482 | 0.0238333 | 0.226667 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_down | 76 | -0.0503641 | 3453 | -0.0571471 | 0.00678294 | -0.00729844 | 0.0208643 | 0.345107 | 0.621192 | 0.772903 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | max_up | 76 | 0.0764674 | 3453 | 0.0535929 | 0.0228745 | -0.000790785 | 0.0465397 | 0.0581579 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证1000 | 20 | terminal_return | 76 | 0.0340363 | 3453 | 0.00736194 | 0.0266744 | 0.00222581 | 0.0511229 | 0.0324811 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_down | 77 | -0.0211749 | 3467 | -0.0202631 | -0.000911731 | -0.00576781 | 0.00394435 | 0.71288 | 0.86743 | 0.874304 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | max_up | 77 | 0.0163387 | 3467 | 0.0196552 | -0.00331646 | -0.00658444 | -4.8485e-05 | 0.0466924 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 5 | terminal_return | 77 | -0.00123974 | 3467 | 0.00150497 | -0.00274471 | -0.0080363 | 0.00254689 | 0.309326 | 0.573163 | 0.759524 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_down | 77 | -0.0310229 | 3462 | -0.02902 | -0.00200289 | -0.00974224 | 0.00573646 | 0.61199 | 0.803237 | 0.837139 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | max_up | 77 | 0.0262233 | 3462 | 0.029251 | -0.00302777 | -0.0082618 | 0.00220627 | 0.256873 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 10 | terminal_return | 77 | 0.00194938 | 3462 | 0.0028455 | -0.000896122 | -0.0112846 | 0.00949237 | 0.86574 | 0.957807 | 0.943083 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_down | 76 | -0.0414529 | 3453 | -0.0409447 | -0.000508221 | -0.0114406 | 0.0104241 | 0.9274 | 0.957807 | 0.961219 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | max_up | 76 | 0.0512471 | 3453 | 0.0438347 | 0.00741238 | -0.0126596 | 0.0274844 | 0.469184 | 0.719651 | 0.813538 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 沪深300 | 20 | terminal_return | 76 | 0.0165797 | 3453 | 0.00534541 | 0.0112343 | -0.0073766 | 0.0298452 | 0.236755 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_down | 77 | -0.0224133 | 3467 | -0.0245651 | 0.00215186 | -0.0030286 | 0.00733231 | 0.415562 | 0.688958 | 0.794997 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | max_up | 77 | 0.0220781 | 3467 | 0.0219762 | 0.000101965 | -0.00396267 | 0.0041666 | 0.960785 | 0.976282 | 0.973664 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 5 | terminal_return | 77 | 0.00371528 | 3467 | 0.00193998 | 0.0017753 | -0.00442221 | 0.00797281 | 0.574491 | 0.8025 | 0.825499 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_down | 77 | -0.0339415 | 3462 | -0.0355005 | 0.00155895 | -0.00728033 | 0.0103982 | 0.729585 | 0.86743 | 0.874304 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | max_up | 77 | 0.0358249 | 3462 | 0.0328373 | 0.00298767 | -0.00398138 | 0.00995673 | 0.400761 | 0.682377 | 0.79463 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 10 | terminal_return | 77 | 0.00847663 | 3462 | 0.00380561 | 0.00467103 | -0.00782932 | 0.0171714 | 0.463927 | 0.719651 | 0.810913 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_down | 76 | -0.0471677 | 3453 | -0.0505822 | 0.00341451 | -0.00930188 | 0.0161309 | 0.598691 | 0.8025 | 0.830942 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | max_up | 76 | 0.0666746 | 3453 | 0.0495619 | 0.0171127 | -0.00540642 | 0.0396318 | 0.136371 | 0.469876 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 中证500 | 20 | terminal_return | 76 | 0.0268479 | 3453 | 0.00731499 | 0.019533 | -0.00289021 | 0.0419561 | 0.0877537 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_down | 77 | -0.0224052 | 3467 | -0.0278363 | 0.00543102 | -0.000275973 | 0.011138 | 0.0621505 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | max_up | 77 | 0.0317334 | 3467 | 0.0260675 | 0.00566596 | -0.00201635 | 0.0133483 | 0.148299 | 0.469876 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 5 | terminal_return | 77 | 0.0128792 | 3467 | 0.00423132 | 0.00864784 | -0.000732278 | 0.018028 | 0.0707641 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_down | 77 | -0.0339988 | 3462 | -0.040529 | 0.00653018 | -0.00369572 | 0.0167561 | 0.210701 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | max_up | 77 | 0.0526732 | 3462 | 0.0399892 | 0.012684 | -0.000135341 | 0.0255034 | 0.0524639 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 10 | terminal_return | 77 | 0.0243546 | 3462 | 0.00843236 | 0.0159222 | -0.000359088 | 0.0322035 | 0.055267 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_down | 76 | -0.0452705 | 3453 | -0.0582993 | 0.0130288 | -0.0012018 | 0.0272594 | 0.0727375 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | max_up | 76 | 0.0901924 | 3453 | 0.0612868 | 0.0289057 | 0.00260508 | 0.0552062 | 0.0312293 | 0.34553 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 微盘股 | 20 | terminal_return | 76 | 0.0537138 | 3453 | 0.016418 | 0.0372958 | 0.0116006 | 0.062991 | 0.00444282 | 0.279897 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_down | 77 | -0.0191624 | 3467 | -0.0189326 | -0.000229807 | -0.00457367 | 0.00411405 | 0.917414 | 0.957807 | 0.961219 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | max_up | 77 | 0.0151664 | 3467 | 0.0171905 | -0.00202412 | -0.00512528 | 0.00107704 | 0.200796 | 0.490394 | 0.749588 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 5 | terminal_return | 77 | 5.52448e-05 | 3467 | 0.00125229 | -0.00119704 | -0.00611812 | 0.00372403 | 0.633528 | 0.814537 | 0.848753 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_down | 77 | -0.027997 | 3462 | -0.0271364 | -0.000860578 | -0.00798909 | 0.00626794 | 0.812953 | 0.931201 | 0.908105 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | max_up | 77 | 0.02416 | 3462 | 0.0256909 | -0.00153093 | -0.00625839 | 0.00319652 | 0.525608 | 0.770077 | 0.825499 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 10 | terminal_return | 77 | 0.00288014 | 3462 | 0.00238005 | 0.000500092 | -0.00897234 | 0.00997252 | 0.917584 | 0.957807 | 0.961219 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_down | 76 | -0.0377816 | 3453 | -0.038517 | 0.00073545 | -0.00952903 | 0.0109999 | 0.888317 | 0.957807 | 0.952578 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | max_up | 76 | 0.0463219 | 3453 | 0.0385475 | 0.00777441 | -0.00927087 | 0.0248197 | 0.371342 | 0.649848 | 0.788579 | true |
| ma60_breadth_reversal_bottom | bottom | onset | 上证指数 | 20 | terminal_return | 76 | 0.0155975 | 3453 | 0.00442795 | 0.0111695 | -0.00456369 | 0.0269027 | 0.164084 | 0.469876 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_down | 41 | -0.0303005 | 3501 | -0.0219326 | -0.0083679 | -0.0196497 | 0.00291391 | 0.146012 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 5 | max_up | 41 | 0.0241036 | 3501 | 0.0195715 | 0.00453215 | -0.00148499 | 0.0105493 | 0.139867 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 5 | terminal_return | 41 | -0.00598513 | 3501 | 0.00179724 | -0.00778237 | -0.0213777 | 0.00581301 | 0.26188 | 0.629271 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_down | 41 | -0.0445843 | 3496 | -0.0316132 | -0.0129711 | -0.0303728 | 0.00443051 | 0.144021 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 10 | max_up | 41 | 0.0333021 | 3496 | 0.0292636 | 0.00403853 | -0.00590948 | 0.0139865 | 0.426213 | 0.662246 | 0.794997 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 10 | terminal_return | 41 | -0.0037649 | 3496 | 0.00345891 | -0.00722381 | -0.0281563 | 0.0137087 | 0.498789 | 0.662246 | 0.819466 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_down | 41 | -0.0579466 | 3486 | -0.0450636 | -0.0128831 | -0.0331836 | 0.0074175 | 0.213556 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 20 | max_up | 41 | 0.0477971 | 3486 | 0.0443105 | 0.00348665 | -0.0135176 | 0.0204909 | 0.687765 | 0.741049 | 0.87056 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 全A | 20 | terminal_return | 41 | -0.00202215 | 3486 | 0.00675377 | -0.00877592 | -0.0336994 | 0.0161475 | 0.490103 | 0.662246 | 0.819466 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_down | 41 | -0.0350536 | 3503 | -0.0266412 | -0.00841243 | -0.0216664 | 0.00484157 | 0.213489 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | max_up | 41 | 0.0280496 | 3503 | 0.0234199 | 0.00462961 | -0.0024012 | 0.0116604 | 0.196839 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 41 | -0.0051596 | 3503 | 0.00274518 | -0.00790478 | -0.0245983 | 0.00878876 | 0.353353 | 0.662173 | 0.778818 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_down | 41 | -0.053652 | 3498 | -0.0388222 | -0.0148298 | -0.0359631 | 0.00630351 | 0.169013 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | max_up | 41 | 0.0421145 | 3498 | 0.0358669 | 0.00624763 | -0.00709504 | 0.0195903 | 0.358745 | 0.662173 | 0.781453 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 41 | -0.00197629 | 3498 | 0.00537945 | -0.00735574 | -0.0340567 | 0.0193452 | 0.589229 | 0.676895 | 0.830942 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_down | 41 | -0.0670409 | 3488 | -0.0558771 | -0.0111638 | -0.0352462 | 0.0129186 | 0.363565 | 0.662173 | 0.785301 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | max_up | 41 | 0.0591609 | 3488 | 0.0550154 | 0.00414549 | -0.017375 | 0.0256659 | 0.705761 | 0.741049 | 0.874304 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 41 | 0.00129556 | 3488 | 0.0106677 | -0.00937216 | -0.0404369 | 0.0216926 | 0.554301 | 0.662246 | 0.825499 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_down | 41 | -0.0362939 | 3503 | -0.0270465 | -0.00924744 | -0.0226115 | 0.00411661 | 0.17502 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | max_up | 41 | 0.0275362 | 3503 | 0.0235155 | 0.00402076 | -0.00296325 | 0.0110048 | 0.259155 | 0.629271 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 41 | -0.00650414 | 3503 | 0.00211759 | -0.00862173 | -0.0252546 | 0.00801115 | 0.309641 | 0.629271 | 0.759524 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_down | 41 | -0.055573 | 3498 | -0.0394631 | -0.0161099 | -0.03762 | 0.00540014 | 0.142121 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | max_up | 41 | 0.041139 | 3498 | 0.0355481 | 0.00559097 | -0.00747785 | 0.0186598 | 0.401746 | 0.662246 | 0.79463 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 41 | -0.00508487 | 3498 | 0.00410937 | -0.00919424 | -0.0361159 | 0.0177274 | 0.503256 | 0.662246 | 0.819466 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_down | 41 | -0.0698679 | 3488 | -0.0568498 | -0.0130181 | -0.0376928 | 0.0116565 | 0.301099 | 0.629271 | 0.759524 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | max_up | 41 | 0.0576133 | 3488 | 0.0540441 | 0.00356921 | -0.0172309 | 0.0243694 | 0.736624 | 0.760776 | 0.878079 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 41 | -0.00124728 | 3488 | 0.00804434 | -0.00929162 | -0.0403105 | 0.0217273 | 0.557128 | 0.662246 | 0.825499 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_down | 41 | -0.0277858 | 3503 | -0.0201951 | -0.00759064 | -0.0174567 | 0.0022754 | 0.131563 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | max_up | 41 | 0.0238917 | 3503 | 0.0195327 | 0.00435895 | -0.00203152 | 0.0107494 | 0.181249 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 41 | -0.00552898 | 3503 | 0.00152697 | -0.00705595 | -0.0190162 | 0.00490433 | 0.247559 | 0.629271 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_down | 41 | -0.0388587 | 3498 | -0.0289488 | -0.00990996 | -0.0242685 | 0.0044486 | 0.176136 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | max_up | 41 | 0.0315449 | 3498 | 0.0291575 | 0.00238741 | -0.00657715 | 0.011352 | 0.601684 | 0.676895 | 0.831579 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 41 | -0.0027466 | 3498 | 0.00289132 | -0.00563792 | -0.0225889 | 0.011313 | 0.514466 | 0.662246 | 0.82345 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_down | 41 | -0.0520826 | 3488 | -0.0408248 | -0.0112577 | -0.0283621 | 0.00584666 | 0.197041 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | max_up | 41 | 0.0473548 | 3488 | 0.0439548 | 0.00340003 | -0.0141346 | 0.0209347 | 0.703907 | 0.741049 | 0.873816 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 41 | -0.00164143 | 3488 | 0.00567232 | -0.00731375 | -0.0297459 | 0.0151184 | 0.5228 | 0.662246 | 0.825499 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_down | 41 | -0.0343272 | 3503 | -0.0244036 | -0.0099236 | -0.0224323 | 0.00258508 | 0.11996 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | max_up | 41 | 0.0254023 | 3503 | 0.0219383 | 0.00346397 | -0.00288262 | 0.00981056 | 0.284724 | 0.629271 | 0.759524 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 41 | -0.00769202 | 3503 | 0.00209174 | -0.00978376 | -0.0249886 | 0.00542107 | 0.207241 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_down | 41 | -0.0517186 | 3498 | -0.0352761 | -0.0164425 | -0.0365386 | 0.00365352 | 0.108788 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | max_up | 41 | 0.0376661 | 3498 | 0.0328464 | 0.00481965 | -0.00666331 | 0.0163026 | 0.410704 | 0.662246 | 0.794658 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 41 | -0.00467401 | 3498 | 0.00400782 | -0.00868183 | -0.0329289 | 0.0155652 | 0.482809 | 0.662246 | 0.816437 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_down | 41 | -0.0654579 | 3488 | -0.0503329 | -0.015125 | -0.0379307 | 0.00768073 | 0.193638 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | max_up | 41 | 0.0528669 | 3488 | 0.0498959 | 0.00297103 | -0.0163184 | 0.0222605 | 0.762739 | 0.775041 | 0.883558 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 41 | -0.000154918 | 3488 | 0.0078284 | -0.00798332 | -0.0371037 | 0.021137 | 0.591038 | 0.676895 | 0.830942 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_down | 41 | -0.0346893 | 3503 | -0.0276367 | -0.00705263 | -0.0205468 | 0.00644157 | 0.305657 | 0.629271 | 0.759524 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | max_up | 41 | 0.0288475 | 3503 | 0.0261595 | 0.00268799 | -0.00372098 | 0.00909695 | 0.411052 | 0.662246 | 0.794658 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 41 | -0.00451063 | 3503 | 0.00452373 | -0.00903436 | -0.0256346 | 0.00756584 | 0.28611 | 0.629271 | 0.759524 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_down | 41 | -0.0554942 | 3498 | -0.0402098 | -0.0152844 | -0.0370632 | 0.00649437 | 0.168966 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | max_up | 41 | 0.043893 | 3498 | 0.0402226 | 0.00367031 | -0.00818814 | 0.0155288 | 0.54409 | 0.662246 | 0.825499 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 41 | -0.00110099 | 3498 | 0.00889459 | -0.00999558 | -0.0365621 | 0.016571 | 0.460853 | 0.662246 | 0.810913 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_down | 41 | -0.0690206 | 3488 | -0.0578894 | -0.0111312 | -0.0366617 | 0.0143993 | 0.392799 | 0.662246 | 0.79262 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | max_up | 41 | 0.0600992 | 3488 | 0.0619306 | -0.00183133 | -0.0210044 | 0.0173417 | 0.851495 | 0.851495 | 0.938633 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 41 | 0.0035364 | 3488 | 0.0173821 | -0.0138457 | -0.0439829 | 0.0162916 | 0.367874 | 0.662173 | 0.787999 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_down | 41 | -0.0261415 | 3503 | -0.0188533 | -0.00728819 | -0.0170222 | 0.00244587 | 0.142237 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | max_up | 41 | 0.0215131 | 3503 | 0.0170954 | 0.00441763 | -0.00145546 | 0.0102907 | 0.140408 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 41 | -0.00472813 | 3503 | 0.00129597 | -0.0060241 | -0.0174235 | 0.00537527 | 0.300305 | 0.629271 | 0.759524 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_down | 41 | -0.037202 | 3498 | -0.0270374 | -0.0101646 | -0.0243578 | 0.00402858 | 0.160415 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | max_up | 41 | 0.0286516 | 3498 | 0.0256225 | 0.00302907 | -0.00544014 | 0.0114983 | 0.483298 | 0.662246 | 0.816437 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 41 | -0.00332486 | 3498 | 0.00245792 | -0.00578278 | -0.0225361 | 0.0109705 | 0.498698 | 0.662246 | 0.819466 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_down | 41 | -0.0492713 | 3488 | -0.0383746 | -0.0108967 | -0.0282629 | 0.00646944 | 0.218757 | 0.626441 | 0.749588 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | max_up | 41 | 0.041819 | 3488 | 0.0386785 | 0.00314053 | -0.0122326 | 0.0185136 | 0.68886 | 0.741049 | 0.87056 | true |
| ma60_breadth_reversal_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 41 | -0.00192471 | 3488 | 0.00474599 | -0.0066707 | -0.027667 | 0.0143256 | 0.533477 | 0.662246 | 0.825499 | true |
| ma60_breadth_reversal_top | top | onset | 全A | 5 | max_down | 41 | -0.0267634 | 3501 | -0.021974 | -0.00478942 | -0.0138691 | 0.00429025 | 0.301193 | 0.695078 | 0.759524 | true |
| ma60_breadth_reversal_top | top | onset | 全A | 5 | max_up | 41 | 0.0244298 | 3501 | 0.0195676 | 0.00486214 | -0.00173898 | 0.0114632 | 0.148834 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 全A | 5 | terminal_return | 41 | -0.00100975 | 3501 | 0.00173898 | -0.00274872 | -0.0153716 | 0.00987413 | 0.669521 | 0.739997 | 0.863128 | true |
| ma60_breadth_reversal_top | top | onset | 全A | 10 | max_down | 41 | -0.0451074 | 3496 | -0.031607 | -0.0135004 | -0.030324 | 0.00332334 | 0.11576 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 全A | 10 | max_up | 41 | 0.0341883 | 3496 | 0.0292532 | 0.00493508 | -0.00563532 | 0.0155055 | 0.36015 | 0.695078 | 0.781453 | true |
| ma60_breadth_reversal_top | top | onset | 全A | 10 | terminal_return | 41 | -0.00550965 | 3496 | 0.00347938 | -0.00898903 | -0.0302287 | 0.0122506 | 0.406816 | 0.695078 | 0.794658 | true |
| ma60_breadth_reversal_top | top | onset | 全A | 20 | max_down | 41 | -0.058557 | 3486 | -0.0450564 | -0.0135006 | -0.0335473 | 0.00654605 | 0.186842 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 全A | 20 | max_up | 41 | 0.0485243 | 3486 | 0.0443019 | 0.0042224 | -0.0127495 | 0.0211943 | 0.625816 | 0.739997 | 0.844851 | true |
| ma60_breadth_reversal_top | top | onset | 全A | 20 | terminal_return | 41 | -0.00262591 | 3486 | 0.00676087 | -0.00938678 | -0.0362759 | 0.0175024 | 0.493836 | 0.695078 | 0.819466 | true |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 5 | max_down | 41 | -0.0327186 | 3503 | -0.0266685 | -0.00605004 | -0.0167109 | 0.00461087 | 0.266011 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 5 | max_up | 41 | 0.0271462 | 3503 | 0.0234305 | 0.00371572 | -0.00478723 | 0.0122187 | 0.391719 | 0.695078 | 0.79262 | true |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 5 | terminal_return | 41 | -0.000882129 | 3503 | 0.00269511 | -0.00357724 | -0.0190068 | 0.0118523 | 0.649532 | 0.739997 | 0.855481 | true |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 10 | max_down | 41 | -0.0549227 | 3498 | -0.0388073 | -0.0161154 | -0.0359299 | 0.00369911 | 0.110915 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 10 | max_up | 41 | 0.0411891 | 3498 | 0.0358777 | 0.0053114 | -0.0094549 | 0.0200777 | 0.480806 | 0.695078 | 0.816437 | true |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 10 | terminal_return | 41 | -0.00315571 | 3498 | 0.00539327 | -0.00854899 | -0.0350478 | 0.0179498 | 0.52717 | 0.695078 | 0.825499 | true |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 20 | max_down | 41 | -0.068514 | 3488 | -0.0558598 | -0.0126542 | -0.0358991 | 0.0105906 | 0.285971 | 0.695078 | 0.759524 | true |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 20 | max_up | 41 | 0.0591727 | 3488 | 0.0550152 | 0.0041575 | -0.0193021 | 0.0276171 | 0.728328 | 0.764744 | 0.874304 | true |
| ma60_breadth_reversal_top | top | onset | 国证2000 | 20 | terminal_return | 41 | 0.000170875 | 3488 | 0.0106809 | -0.0105101 | -0.0437582 | 0.0227381 | 0.535537 | 0.695078 | 0.825499 | true |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 5 | max_down | 41 | -0.0333007 | 3503 | -0.0270815 | -0.00621914 | -0.0170515 | 0.00461318 | 0.260465 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 5 | max_up | 41 | 0.027168 | 3503 | 0.0235198 | 0.00364826 | -0.00488308 | 0.0121796 | 0.401944 | 0.695078 | 0.79463 | true |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 5 | terminal_return | 41 | -0.0014408 | 3503 | 0.00205833 | -0.00349913 | -0.0188814 | 0.0118831 | 0.6557 | 0.739997 | 0.859115 | true |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 10 | max_down | 41 | -0.0560936 | 3498 | -0.039457 | -0.0166366 | -0.0367697 | 0.00349654 | 0.105317 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 10 | max_up | 41 | 0.0406179 | 3498 | 0.0355542 | 0.00506371 | -0.00963441 | 0.0197618 | 0.499518 | 0.695078 | 0.819466 | true |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 10 | terminal_return | 41 | -0.00547245 | 3498 | 0.00411391 | -0.00958636 | -0.0363344 | 0.0171617 | 0.482397 | 0.695078 | 0.816437 | true |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 20 | max_down | 41 | -0.0704159 | 3488 | -0.0568433 | -0.0135726 | -0.0372464 | 0.0101012 | 0.261139 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 20 | max_up | 41 | 0.0577945 | 3488 | 0.0540419 | 0.00375259 | -0.0189446 | 0.0264497 | 0.745898 | 0.770354 | 0.879042 | true |
| ma60_breadth_reversal_top | top | onset | 中证1000 | 20 | terminal_return | 41 | -0.00258332 | 3488 | 0.00806005 | -0.0106434 | -0.0438839 | 0.0225972 | 0.530281 | 0.695078 | 0.825499 | true |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 5 | max_down | 41 | -0.0231618 | 3503 | -0.0202492 | -0.00291254 | -0.0107161 | 0.004891 | 0.464451 | 0.695078 | 0.810913 | true |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 5 | max_up | 41 | 0.0245866 | 3503 | 0.0195246 | 0.00506205 | -0.00104322 | 0.0111673 | 0.104143 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 5 | terminal_return | 41 | 6.63608e-05 | 3503 | 0.00146148 | -0.00139512 | -0.0124596 | 0.00966933 | 0.804803 | 0.817783 | 0.904058 | true |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 10 | max_down | 41 | -0.0387628 | 3498 | -0.0289499 | -0.00981292 | -0.0239287 | 0.0043029 | 0.173029 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 10 | max_up | 41 | 0.0325051 | 3498 | 0.0291463 | 0.0033588 | -0.00551465 | 0.0122323 | 0.458145 | 0.695078 | 0.810913 | true |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 10 | terminal_return | 41 | -0.00499755 | 3498 | 0.0029177 | -0.00791525 | -0.0257777 | 0.00994719 | 0.38511 | 0.695078 | 0.79262 | true |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 20 | max_down | 41 | -0.0517122 | 3488 | -0.0408292 | -0.010883 | -0.0283532 | 0.00658709 | 0.222092 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 20 | max_up | 41 | 0.0477612 | 3488 | 0.04395 | 0.0038112 | -0.0120946 | 0.019717 | 0.638613 | 0.739997 | 0.848753 | true |
| ma60_breadth_reversal_top | top | onset | 沪深300 | 20 | terminal_return | 41 | -0.00132206 | 3488 | 0.00566857 | -0.00699063 | -0.0316735 | 0.0176923 | 0.578822 | 0.715015 | 0.827214 | true |
| ma60_breadth_reversal_top | top | onset | 中证500 | 5 | max_down | 41 | -0.0303672 | 3503 | -0.0244499 | -0.00591726 | -0.0163972 | 0.00456273 | 0.268439 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 中证500 | 5 | max_up | 41 | 0.026229 | 3503 | 0.0219286 | 0.00430041 | -0.00319307 | 0.0117939 | 0.260666 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 中证500 | 5 | terminal_return | 41 | -0.00243238 | 3503 | 0.00203018 | -0.00446256 | -0.0186389 | 0.00971378 | 0.537243 | 0.695078 | 0.825499 | true |
| ma60_breadth_reversal_top | top | onset | 中证500 | 10 | max_down | 41 | -0.0520042 | 3498 | -0.0352727 | -0.0167315 | -0.0358637 | 0.00240066 | 0.0865171 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 中证500 | 10 | max_up | 41 | 0.0378322 | 3498 | 0.0328445 | 0.00498772 | -0.00732935 | 0.0173048 | 0.427376 | 0.695078 | 0.794997 | true |
| ma60_breadth_reversal_top | top | onset | 中证500 | 10 | terminal_return | 41 | -0.00661156 | 3498 | 0.00403053 | -0.0106421 | -0.035117 | 0.0138328 | 0.394081 | 0.695078 | 0.79262 | true |
| ma60_breadth_reversal_top | top | onset | 中证500 | 20 | max_down | 41 | -0.0662275 | 3488 | -0.0503239 | -0.0159036 | -0.0382317 | 0.00642444 | 0.162699 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 中证500 | 20 | max_up | 41 | 0.0539842 | 3488 | 0.0498828 | 0.00410145 | -0.0162175 | 0.0244204 | 0.692376 | 0.746483 | 0.87056 | true |
| ma60_breadth_reversal_top | top | onset | 中证500 | 20 | terminal_return | 41 | -0.00185805 | 3488 | 0.00784842 | -0.00970647 | -0.0408141 | 0.0214011 | 0.540818 | 0.695078 | 0.825499 | true |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 5 | max_down | 41 | -0.0328064 | 3503 | -0.0276587 | -0.0051477 | -0.0168568 | 0.00656137 | 0.388863 | 0.695078 | 0.79262 | true |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 5 | max_up | 41 | 0.0289902 | 3503 | 0.0261578 | 0.00283241 | -0.00498399 | 0.0106488 | 0.477555 | 0.695078 | 0.816437 | true |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 5 | terminal_return | 41 | -0.00112094 | 3503 | 0.00448406 | -0.005605 | -0.021645 | 0.010435 | 0.493408 | 0.695078 | 0.819466 | true |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 10 | max_down | 41 | -0.0553655 | 3498 | -0.0402113 | -0.0151542 | -0.0365352 | 0.00622687 | 0.164777 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 10 | max_up | 41 | 0.044096 | 3498 | 0.0402203 | 0.00387577 | -0.00888525 | 0.0166368 | 0.55165 | 0.695078 | 0.825499 | true |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 10 | terminal_return | 41 | -0.00296353 | 3498 | 0.00891642 | -0.0118799 | -0.0385122 | 0.0147523 | 0.381952 | 0.695078 | 0.79262 | true |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 20 | max_down | 41 | -0.0691289 | 3488 | -0.0578881 | -0.0112408 | -0.0366678 | 0.0141863 | 0.38623 | 0.695078 | 0.79262 | true |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 20 | max_up | 41 | 0.0620128 | 3488 | 0.0619081 | 0.000104719 | -0.0200527 | 0.0202621 | 0.991876 | 0.991876 | 0.994507 | true |
| ma60_breadth_reversal_top | top | onset | 微盘股 | 20 | terminal_return | 41 | 0.00418571 | 3488 | 0.0173744 | -0.0131887 | -0.0456445 | 0.0192671 | 0.425762 | 0.695078 | 0.794997 | true |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 5 | max_down | 41 | -0.0232146 | 3503 | -0.0188875 | -0.00432712 | -0.0121205 | 0.00346629 | 0.276486 | 0.695078 | 0.759524 | true |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 5 | max_up | 41 | 0.0212982 | 3503 | 0.0170979 | 0.0042003 | -0.00159287 | 0.00999346 | 0.155292 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 5 | terminal_return | 41 | -0.000896448 | 3503 | 0.00125112 | -0.00214757 | -0.0130367 | 0.0087416 | 0.699087 | 0.746483 | 0.873684 | true |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 10 | max_down | 41 | -0.038377 | 3498 | -0.0270236 | -0.0113534 | -0.025673 | 0.00296621 | 0.120185 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 10 | max_up | 41 | 0.0284871 | 3498 | 0.0256245 | 0.00286264 | -0.00548548 | 0.0112108 | 0.501519 | 0.695078 | 0.819466 | true |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 10 | terminal_return | 41 | -0.00662529 | 3498 | 0.00249661 | -0.00912189 | -0.0267574 | 0.00851361 | 0.310677 | 0.695078 | 0.759524 | true |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 20 | max_down | 41 | -0.049705 | 3488 | -0.0383695 | -0.0113355 | -0.0289342 | 0.00626316 | 0.206784 | 0.695078 | 0.749588 | true |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 20 | max_up | 41 | 0.0419895 | 3488 | 0.0386765 | 0.003313 | -0.0116757 | 0.0183017 | 0.664851 | 0.739997 | 0.863128 | true |
| ma60_breadth_reversal_top | top | onset | 上证指数 | 20 | terminal_return | 41 | -0.0024047 | 3488 | 0.00475163 | -0.00715634 | -0.0304366 | 0.016124 | 0.54684 | 0.695078 | 0.825499 | true |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `ma120_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：数据可用性——10日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 5 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `ma120_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：数据可用性——10日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 12 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `ma120_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：2 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `ma120_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
- `ma20_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `ma20_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 3 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `ma20_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
- `ma20_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：数据可用性——5日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；10日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
- `ma60_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：数据可用性——10日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）；20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 1 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `ma60_breadth_reversal_bottom/bottom/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：数据可用性——5日：事件日缺失 1、窗口不完整 1（涉及 1 个指数）；10日：事件日缺失 1、窗口不完整 1（涉及 1 个指数）；20日：事件日缺失 1、窗口不完整 8（涉及 7 个指数）。 5 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `ma60_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/capped_confirmation`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为负；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `ma60_breadth_reversal_top/top/ma_period_breadth_decomposition_v1_20120104_20260814/onset`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。 最长 20 日 terminal 均值差在 7/7 个指数均为负；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
