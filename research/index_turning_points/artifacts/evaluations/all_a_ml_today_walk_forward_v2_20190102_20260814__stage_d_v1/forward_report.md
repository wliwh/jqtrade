# 信号后 OHLC 结果评测

- 评测版本：`all_a_ml_today_walk_forward_v2_stage_d_v1`
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
| 上证指数 | capped_confirmation | 5 | 17 | 17 | 17 |
| 上证指数 | capped_confirmation | 10 | 17 | 17 | 17 |
| 上证指数 | capped_confirmation | 20 | 17 | 17 | 17 |
| 上证指数 | onset | 5 | 17 | 17 | 17 |
| 上证指数 | onset | 10 | 17 | 17 | 17 |
| 上证指数 | onset | 20 | 17 | 17 | 17 |
| 中证1000 | capped_confirmation | 5 | 17 | 17 | 17 |
| 中证1000 | capped_confirmation | 10 | 17 | 17 | 17 |
| 中证1000 | capped_confirmation | 20 | 17 | 17 | 17 |
| 中证1000 | onset | 5 | 17 | 17 | 17 |
| 中证1000 | onset | 10 | 17 | 17 | 17 |
| 中证1000 | onset | 20 | 17 | 17 | 17 |
| 中证500 | capped_confirmation | 5 | 17 | 17 | 17 |
| 中证500 | capped_confirmation | 10 | 17 | 17 | 17 |
| 中证500 | capped_confirmation | 20 | 17 | 17 | 17 |
| 中证500 | onset | 5 | 17 | 17 | 17 |
| 中证500 | onset | 10 | 17 | 17 | 17 |
| 中证500 | onset | 20 | 17 | 17 | 17 |
| 全A | capped_confirmation | 5 | 17 | 17 | 17 |
| 全A | capped_confirmation | 10 | 17 | 17 | 17 |
| 全A | capped_confirmation | 20 | 17 | 17 | 17 |
| 全A | onset | 5 | 17 | 17 | 17 |
| 全A | onset | 10 | 17 | 17 | 17 |
| 全A | onset | 20 | 17 | 17 | 17 |
| 国证2000 | capped_confirmation | 5 | 17 | 17 | 17 |
| 国证2000 | capped_confirmation | 10 | 17 | 17 | 17 |
| 国证2000 | capped_confirmation | 20 | 17 | 17 | 17 |
| 国证2000 | onset | 5 | 17 | 17 | 17 |
| 国证2000 | onset | 10 | 17 | 17 | 17 |
| 国证2000 | onset | 20 | 17 | 17 | 17 |
| 微盘股 | capped_confirmation | 5 | 17 | 17 | 17 |
| 微盘股 | capped_confirmation | 10 | 17 | 17 | 17 |
| 微盘股 | capped_confirmation | 20 | 17 | 17 | 17 |
| 微盘股 | onset | 5 | 17 | 17 | 17 |
| 微盘股 | onset | 10 | 17 | 17 | 17 |
| 微盘股 | onset | 20 | 17 | 17 | 17 |
| 沪深300 | capped_confirmation | 5 | 17 | 17 | 17 |
| 沪深300 | capped_confirmation | 10 | 17 | 17 | 17 |
| 沪深300 | capped_confirmation | 20 | 17 | 17 | 17 |
| 沪深300 | onset | 5 | 17 | 17 | 17 |
| 沪深300 | onset | 10 | 17 | 17 | 17 |
| 沪深300 | onset | 20 | 17 | 17 | 17 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 5 | max_down | 3 | -0.0477376 | 1840 | -0.0198087 | -0.0279289 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 5 | max_up | 3 | 0.0236562 | 1840 | 0.0186228 | 0.00503344 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 5 | terminal_return | 3 | 0.00923423 | 1840 | 0.00188395 | 0.00735029 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 10 | max_down | 3 | -0.0606016 | 1835 | -0.0287475 | -0.0318541 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 10 | max_up | 3 | 0.0296328 | 1835 | 0.0278049 | 0.00182784 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 10 | terminal_return | 3 | -0.00196793 | 1835 | 0.00365177 | -0.00561971 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 20 | max_down | 3 | -0.0625072 | 1825 | -0.0412345 | -0.0212727 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 20 | max_up | 3 | 0.0357425 | 1825 | 0.0417977 | -0.0060552 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 全A | 20 | terminal_return | 3 | -0.00882986 | 1825 | 0.00703791 | -0.0158678 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 5 | max_down | 3 | -0.0611802 | 1840 | -0.0252174 | -0.0359628 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 5 | max_up | 3 | 0.0286654 | 1840 | 0.0228674 | 0.00579793 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 3 | 0.00732289 | 1840 | 0.00255337 | 0.00476952 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 10 | max_down | 3 | -0.0770316 | 1835 | -0.0370595 | -0.0399721 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 10 | max_up | 3 | 0.0454754 | 1835 | 0.0344587 | 0.0110167 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 3 | 0.00684814 | 1835 | 0.00485918 | 0.00198896 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 20 | max_down | 3 | -0.0770316 | 1825 | -0.0540434 | -0.0229882 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 20 | max_up | 3 | 0.0589329 | 1825 | 0.0514699 | 0.00746302 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 3 | 0.0154591 | 1825 | 0.00932392 | 0.00613514 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 5 | max_down | 3 | -0.0568888 | 1840 | -0.0248578 | -0.032031 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 5 | max_up | 3 | 0.0338504 | 1840 | 0.022659 | 0.0111915 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 3 | 0.0147301 | 1840 | 0.00212391 | 0.0126062 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 10 | max_down | 3 | -0.0741296 | 1835 | -0.0364946 | -0.037635 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 10 | max_up | 3 | 0.05002 | 1835 | 0.0338576 | 0.0161624 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 3 | 0.0106424 | 1835 | 0.00403795 | 0.00660445 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 20 | max_down | 3 | -0.0767952 | 1825 | -0.0529779 | -0.0238172 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 20 | max_up | 3 | 0.0621293 | 1825 | 0.0505811 | 0.0115482 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 3 | 0.0123824 | 1825 | 0.00776604 | 0.00461638 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 5 | max_down | 3 | -0.0411194 | 1840 | -0.0180829 | -0.0230365 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 5 | max_up | 3 | 0.0207266 | 1840 | 0.0182869 | 0.00243971 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 3 | 0.00934705 | 1840 | 0.00155377 | 0.00779328 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 10 | max_down | 3 | -0.0514486 | 1835 | -0.0259845 | -0.0254641 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 10 | max_up | 3 | 0.0207266 | 1835 | 0.027412 | -0.0066854 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 3 | -0.00804517 | 1835 | 0.00302824 | -0.0110734 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 20 | max_down | 3 | -0.0608871 | 1825 | -0.0367211 | -0.024166 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 20 | max_up | 3 | 0.0243326 | 1825 | 0.0413746 | -0.0170419 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 3 | -0.0252877 | 1825 | 0.00585097 | -0.0311387 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 5 | max_down | 3 | -0.052754 | 1840 | -0.0216003 | -0.0311537 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 5 | max_up | 3 | 0.0293905 | 1840 | 0.0210162 | 0.00837431 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 3 | 0.00987744 | 1840 | 0.00226547 | 0.00761198 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 10 | max_down | 3 | -0.0634556 | 1835 | -0.0312991 | -0.0321565 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 10 | max_up | 3 | 0.0373096 | 1835 | 0.03137 | 0.00593959 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 3 | 0.00418169 | 1835 | 0.00435567 | -0.000173976 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 20 | max_down | 3 | -0.0641656 | 1825 | -0.0449566 | -0.019209 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 20 | max_up | 3 | 0.0476179 | 1825 | 0.0475327 | 8.52123e-05 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 3 | 0.0086759 | 1825 | 0.0085181 | 0.000157799 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 5 | max_down | 3 | -0.0630599 | 1840 | -0.0261111 | -0.0369488 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 5 | max_up | 3 | 0.0143403 | 1840 | 0.0278343 | -0.0134939 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 3 | -0.0080193 | 1840 | 0.00691914 | -0.0149384 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 10 | max_down | 3 | -0.076408 | 1835 | -0.038141 | -0.038267 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 10 | max_up | 3 | 0.0377902 | 1835 | 0.0431803 | -0.00539014 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 3 | -0.000816492 | 1835 | 0.0135563 | -0.0143728 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 20 | max_down | 3 | -0.076408 | 1825 | -0.0553261 | -0.0210819 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 20 | max_up | 3 | 0.0656646 | 1825 | 0.0666004 | -0.000935796 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 3 | 0.0272663 | 1825 | 0.0265293 | 0.000736998 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 5 | max_down | 3 | -0.042874 | 1840 | -0.0162895 | -0.0265845 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 5 | max_up | 3 | 0.0202671 | 1840 | 0.0156248 | 0.00464229 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 3 | 0.00460549 | 1840 | 0.00152293 | 0.00308255 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 10 | max_down | 3 | -0.0530018 | 1835 | -0.0234182 | -0.0295836 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 10 | max_up | 3 | 0.0223922 | 1835 | 0.0233758 | -0.000983639 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 3 | -0.0054638 | 1835 | 0.00293663 | -0.00840044 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 20 | max_down | 3 | -0.055074 | 1825 | -0.0334821 | -0.0215919 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 20 | max_up | 3 | 0.0283372 | 1825 | 0.0350557 | -0.00671847 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 3 | -0.00649179 | 1825 | 0.00558954 | -0.0120813 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 5 | max_down | 3 | -0.0595886 | 1840 | -0.0197894 | -0.0397993 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 5 | max_up | 3 | 0.00876186 | 1840 | 0.0186471 | -0.00988521 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 5 | terminal_return | 3 | -0.0179038 | 1840 | 0.00192819 | -0.019832 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 10 | max_down | 3 | -0.0714695 | 1835 | -0.0287297 | -0.0427398 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 10 | max_up | 3 | 0.0184758 | 1835 | 0.0278232 | -0.00934743 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 10 | terminal_return | 3 | -0.0140788 | 1835 | 0.00367157 | -0.0177504 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 20 | max_down | 3 | -0.074461 | 1825 | -0.0412148 | -0.0332462 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 20 | max_up | 3 | 0.0242151 | 1825 | 0.0418166 | -0.0176015 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 全A | 20 | terminal_return | 3 | -0.0132238 | 1825 | 0.00704513 | -0.0202689 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 5 | max_down | 3 | -0.0739741 | 1840 | -0.0251965 | -0.0487776 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 5 | max_up | 3 | 0.0142066 | 1840 | 0.022891 | -0.00868446 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 5 | terminal_return | 3 | -0.0279387 | 1840 | 0.00261086 | -0.0305496 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 10 | max_down | 3 | -0.0890202 | 1835 | -0.0370399 | -0.0519803 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 10 | max_up | 3 | 0.0291489 | 1835 | 0.0344854 | -0.00533649 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 10 | terminal_return | 3 | -0.0113676 | 1835 | 0.00488896 | -0.0162566 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 20 | max_down | 3 | -0.0900928 | 1825 | -0.0540219 | -0.0360709 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 20 | max_up | 3 | 0.0416451 | 1825 | 0.0514983 | -0.00985321 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 国证2000 | 20 | terminal_return | 3 | 0.00674119 | 1825 | 0.00933825 | -0.00259706 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 5 | max_down | 3 | -0.0705068 | 1840 | -0.0248356 | -0.0456712 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 5 | max_up | 3 | 0.0123567 | 1840 | 0.022694 | -0.0103373 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 5 | terminal_return | 3 | -0.0200087 | 1840 | 0.00218055 | -0.0221892 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 10 | max_down | 3 | -0.0866386 | 1835 | -0.0364742 | -0.0501644 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 10 | max_up | 3 | 0.0333848 | 1835 | 0.0338848 | -0.000500009 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 10 | terminal_return | 3 | -0.00695571 | 1835 | 0.00406672 | -0.0110224 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 20 | max_down | 3 | -0.0890657 | 1825 | -0.0529577 | -0.036108 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 20 | max_up | 3 | 0.0445485 | 1825 | 0.05061 | -0.00606146 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证1000 | 20 | terminal_return | 3 | 0.00483151 | 1825 | 0.00777845 | -0.00294695 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 5 | max_down | 3 | -0.0521254 | 1840 | -0.0180649 | -0.0340605 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 5 | max_up | 3 | 0.0114769 | 1840 | 0.018302 | -0.00682506 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 5 | terminal_return | 3 | -0.0130434 | 1840 | 0.00159027 | -0.0146336 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 10 | max_down | 3 | -0.0616502 | 1835 | -0.0259678 | -0.0356824 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 10 | max_up | 3 | 0.0133266 | 1835 | 0.0274241 | -0.0140975 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 10 | terminal_return | 3 | -0.0166734 | 1835 | 0.00304235 | -0.0197157 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 20 | max_down | 3 | -0.0697731 | 1825 | -0.0367065 | -0.0330666 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 20 | max_up | 3 | 0.0167544 | 1825 | 0.041387 | -0.0246326 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 沪深300 | 20 | terminal_return | 3 | -0.0271515 | 1825 | 0.00585404 | -0.0330055 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 5 | max_down | 3 | -0.0657385 | 1840 | -0.0215791 | -0.0441593 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 5 | max_up | 3 | 0.0106164 | 1840 | 0.0210468 | -0.0104305 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 5 | terminal_return | 3 | -0.0204824 | 1840 | 0.00231497 | -0.0227974 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 10 | max_down | 3 | -0.075856 | 1835 | -0.0312788 | -0.0445772 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 10 | max_up | 3 | 0.0222172 | 1835 | 0.0313947 | -0.00917753 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 10 | terminal_return | 3 | -0.0109339 | 1835 | 0.00438038 | -0.0153143 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 20 | max_down | 3 | -0.0772371 | 1825 | -0.0449351 | -0.032302 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 20 | max_up | 3 | 0.0320985 | 1825 | 0.0475582 | -0.0154597 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 中证500 | 20 | terminal_return | 3 | 0.000772231 | 1825 | 0.00853109 | -0.00775886 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 5 | max_down | 3 | -0.0754386 | 1840 | -0.0260909 | -0.0493476 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 5 | max_up | 3 | 0.0209534 | 1840 | 0.0278235 | -0.00687003 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 5 | terminal_return | 3 | -0.0395325 | 1840 | 0.00697052 | -0.046503 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 10 | max_down | 3 | -0.0855274 | 1835 | -0.0381261 | -0.0474013 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 10 | max_up | 3 | 0.029054 | 1835 | 0.0431946 | -0.0141406 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 10 | terminal_return | 3 | -0.0173097 | 1835 | 0.0135832 | -0.0308929 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 20 | max_down | 3 | -0.0891008 | 1825 | -0.0553052 | -0.0337955 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 20 | max_up | 3 | 0.0488506 | 1825 | 0.0666281 | -0.0177775 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 微盘股 | 20 | terminal_return | 3 | 0.0177861 | 1825 | 0.0265449 | -0.0087588 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 5 | max_down | 3 | -0.0551469 | 1840 | -0.0162695 | -0.0388774 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 5 | max_up | 3 | 0.00564881 | 1840 | 0.0156487 | -0.00999985 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 5 | terminal_return | 3 | -0.0199615 | 1840 | 0.00156299 | -0.0215245 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 10 | max_down | 3 | -0.0637306 | 1835 | -0.0234007 | -0.0403299 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 10 | max_up | 3 | 0.0113082 | 1835 | 0.023394 | -0.0120858 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 10 | terminal_return | 3 | -0.0166822 | 1835 | 0.00295498 | -0.0196372 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 20 | max_down | 3 | -0.0673885 | 1825 | -0.0334618 | -0.0339267 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 20 | max_up | 3 | 0.0169481 | 1825 | 0.0350744 | -0.0181263 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | bottom | onset | 上证指数 | 20 | terminal_return | 3 | -0.0130155 | 1825 | 0.00560027 | -0.0186157 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 5 | max_down | 8 | -0.021403 | 1835 | -0.0198474 | -0.00155564 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 5 | max_up | 8 | 0.033119 | 1835 | 0.0185678 | 0.0145512 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 5 | terminal_return | 8 | 0.0050276 | 1835 | 0.00188226 | 0.00314534 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 10 | max_down | 8 | -0.0275096 | 1830 | -0.0288051 | 0.00129545 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 10 | max_up | 8 | 0.0433646 | 1830 | 0.0277399 | 0.0156247 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 10 | terminal_return | 8 | 0.00756983 | 1830 | 0.00362543 | 0.0039444 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 20 | max_down | 8 | -0.0444895 | 1820 | -0.0412553 | -0.00323429 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 20 | max_up | 8 | 0.046932 | 1820 | 0.0417651 | 0.0051669 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 全A | 20 | terminal_return | 8 | -0.0121598 | 1820 | 0.00709614 | -0.019256 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 5 | max_down | 8 | -0.0275619 | 1835 | -0.025266 | -0.00229592 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 5 | max_up | 8 | 0.0394991 | 1835 | 0.0228044 | 0.0166947 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 5 | terminal_return | 8 | 0.00391806 | 1835 | 0.00255522 | 0.00136285 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 10 | max_down | 8 | -0.0368451 | 1830 | -0.0371259 | 0.000280895 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 10 | max_up | 8 | 0.0603279 | 1830 | 0.0343637 | 0.0259643 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 10 | terminal_return | 8 | 0.0174721 | 1830 | 0.0048073 | 0.0126648 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 20 | max_down | 8 | -0.059948 | 1820 | -0.0540553 | -0.00589274 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 20 | max_up | 8 | 0.0630865 | 1820 | 0.0514311 | 0.0116554 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 国证2000 | 20 | terminal_return | 8 | -0.00818457 | 1820 | 0.00941099 | -0.0175956 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 5 | max_down | 8 | -0.0252824 | 1835 | -0.0249084 | -0.000374003 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 5 | max_up | 8 | 0.0409293 | 1835 | 0.0225976 | 0.0183317 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 5 | terminal_return | 8 | 0.00659557 | 1835 | 0.00212502 | 0.00447055 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 10 | max_down | 8 | -0.0351823 | 1830 | -0.036562 | 0.00137969 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 10 | max_up | 8 | 0.0604573 | 1830 | 0.0337678 | 0.0266896 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 10 | terminal_return | 8 | 0.018486 | 1830 | 0.00398561 | 0.0145004 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 20 | max_down | 8 | -0.057056 | 1820 | -0.0529992 | -0.00405675 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 20 | max_up | 8 | 0.0621396 | 1820 | 0.0505493 | 0.0115903 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证1000 | 20 | terminal_return | 8 | -0.00786428 | 1820 | 0.00784235 | -0.0157066 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 5 | max_down | 8 | -0.0200501 | 1835 | -0.018112 | -0.00193809 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 5 | max_up | 8 | 0.0303826 | 1835 | 0.0182382 | 0.0121444 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 5 | terminal_return | 8 | 0.00387818 | 1835 | 0.00155638 | 0.0023218 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 10 | max_down | 8 | -0.025287 | 1830 | -0.0260293 | 0.000742304 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 10 | max_up | 8 | 0.0361486 | 1830 | 0.0273629 | 0.00878572 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 10 | terminal_return | 8 | -0.00140537 | 1830 | 0.00302947 | -0.00443484 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 20 | max_down | 8 | -0.0374007 | 1820 | -0.036758 | -0.000642693 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 20 | max_up | 8 | 0.0368487 | 1820 | 0.0413664 | -0.00451761 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 沪深300 | 20 | terminal_return | 8 | -0.0163483 | 1820 | 0.00589722 | -0.0222455 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 5 | max_down | 8 | -0.0208998 | 1835 | -0.0216543 | 0.000754511 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 5 | max_up | 8 | 0.0387533 | 1835 | 0.0209526 | 0.0178007 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 5 | terminal_return | 8 | 0.00870211 | 1835 | 0.00224985 | 0.00645225 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 10 | max_down | 8 | -0.0299193 | 1830 | -0.0313579 | 0.0014385 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 10 | max_up | 8 | 0.0564313 | 1830 | 0.0312702 | 0.025161 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 10 | terminal_return | 8 | 0.0169256 | 1830 | 0.00430043 | 0.0126252 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 20 | max_down | 8 | -0.0492829 | 1820 | -0.0449692 | -0.00431364 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 20 | max_up | 8 | 0.0618103 | 1820 | 0.0474701 | 0.0143403 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 中证500 | 20 | terminal_return | 8 | -0.00459732 | 1820 | 0.00857601 | -0.0131733 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 5 | max_down | 8 | -0.0332155 | 1835 | -0.0261406 | -0.00707491 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 5 | max_up | 8 | 0.0461478 | 1835 | 0.0277323 | 0.0184154 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 5 | terminal_return | 8 | 0.0115349 | 1835 | 0.0068746 | 0.00466032 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 10 | max_down | 8 | -0.0424024 | 1830 | -0.0381851 | -0.00421727 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 10 | max_up | 8 | 0.0732007 | 1830 | 0.0430403 | 0.0301604 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 10 | terminal_return | 8 | 0.0227789 | 1830 | 0.0134924 | 0.00928653 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 20 | max_down | 8 | -0.0738427 | 1820 | -0.0552794 | -0.0185633 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 20 | max_up | 8 | 0.0839214 | 1820 | 0.0665227 | 0.0173986 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 微盘股 | 20 | terminal_return | 8 | 0.0120644 | 1820 | 0.0265941 | -0.0145297 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 5 | max_down | 8 | -0.0184206 | 1835 | -0.0163237 | -0.00209688 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 5 | max_up | 8 | 0.0280316 | 1835 | 0.0155783 | 0.0124533 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 5 | terminal_return | 8 | 0.00359594 | 1835 | 0.00151894 | 0.002077 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 10 | max_down | 8 | -0.0227868 | 1830 | -0.0234695 | 0.000682723 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 10 | max_up | 8 | 0.0350364 | 1830 | 0.0233233 | 0.0117132 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 10 | terminal_return | 8 | 0.00103532 | 1830 | 0.00293118 | -0.00189585 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 20 | max_down | 8 | -0.0349704 | 1820 | -0.0335111 | -0.00145925 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 20 | max_up | 8 | 0.0354487 | 1820 | 0.0350429 | 0.000405866 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | capped_confirmation | 上证指数 | 20 | terminal_return | 8 | -0.013464 | 1820 | 0.00565338 | -0.0191174 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 全A | 5 | max_down | 8 | -0.0127008 | 1835 | -0.0198853 | 0.00718449 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 全A | 5 | max_up | 8 | 0.0424662 | 1835 | 0.0185271 | 0.0239392 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 全A | 5 | terminal_return | 8 | 0.00825025 | 1835 | 0.00186821 | 0.00638205 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 全A | 10 | max_down | 8 | -0.0184729 | 1830 | -0.0288446 | 0.0103717 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 全A | 10 | max_up | 8 | 0.0537889 | 1830 | 0.0276944 | 0.0260946 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 全A | 10 | terminal_return | 8 | 0.0153729 | 1830 | 0.00359132 | 0.0117816 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 全A | 20 | max_down | 8 | -0.0365071 | 1820 | -0.0412903 | 0.00478319 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 全A | 20 | max_up | 8 | 0.0582326 | 1820 | 0.0417154 | 0.0165172 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 全A | 20 | terminal_return | 8 | -0.00130062 | 1820 | 0.00704841 | -0.00834903 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 5 | max_down | 8 | -0.0162883 | 1835 | -0.0253151 | 0.00902687 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 5 | max_up | 8 | 0.0497514 | 1835 | 0.0227597 | 0.0269917 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 5 | terminal_return | 8 | 0.00839899 | 1835 | 0.00253568 | 0.00586331 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 10 | max_down | 8 | -0.0272248 | 1830 | -0.037168 | 0.00994318 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 10 | max_up | 8 | 0.0700334 | 1830 | 0.0343212 | 0.0357121 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 10 | terminal_return | 8 | 0.0251937 | 1830 | 0.00477354 | 0.0204202 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 20 | max_down | 8 | -0.0523571 | 1820 | -0.0540887 | 0.00173156 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 20 | max_up | 8 | 0.0761591 | 1820 | 0.0513736 | 0.0247855 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 国证2000 | 20 | terminal_return | 8 | 0.00582848 | 1820 | 0.0093494 | -0.00352091 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 5 | max_down | 8 | -0.0138646 | 1835 | -0.0249581 | 0.0110935 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 5 | max_up | 8 | 0.049548 | 1835 | 0.02256 | 0.026988 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 5 | terminal_return | 8 | 0.00953515 | 1835 | 0.00211221 | 0.00742294 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 10 | max_down | 8 | -0.0239912 | 1830 | -0.036611 | 0.0126197 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 10 | max_up | 8 | 0.0694573 | 1830 | 0.0337284 | 0.0357288 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 10 | terminal_return | 8 | 0.0265609 | 1830 | 0.00395031 | 0.0226106 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 20 | max_down | 8 | -0.0470451 | 1820 | -0.0530433 | 0.00599818 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 20 | max_up | 8 | 0.0746413 | 1820 | 0.0504944 | 0.0241469 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证1000 | 20 | terminal_return | 8 | 0.00659821 | 1820 | 0.00777878 | -0.00118058 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 5 | max_down | 8 | -0.0134908 | 1835 | -0.0181406 | 0.00464977 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 5 | max_up | 8 | 0.0393066 | 1835 | 0.0181992 | 0.0211073 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 5 | terminal_return | 8 | 0.00577044 | 1835 | 0.00154813 | 0.00422231 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 10 | max_down | 8 | -0.0168064 | 1830 | -0.0260664 | 0.00925995 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 10 | max_up | 8 | 0.0476014 | 1830 | 0.0273128 | 0.0202886 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 10 | terminal_return | 8 | 0.00651889 | 1830 | 0.00299483 | 0.00352407 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 20 | max_down | 8 | -0.0290758 | 1820 | -0.0367945 | 0.00771871 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 20 | max_up | 8 | 0.048299 | 1820 | 0.041316 | 0.00698297 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 沪深300 | 20 | terminal_return | 8 | -0.00806229 | 1820 | 0.0058608 | -0.0139231 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 5 | max_down | 8 | -0.0113105 | 1835 | -0.0216961 | 0.0103856 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 5 | max_up | 8 | 0.0465529 | 1835 | 0.0209186 | 0.0256343 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 5 | terminal_return | 8 | 0.0119973 | 1835 | 0.00223549 | 0.00976182 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 10 | max_down | 8 | -0.0188923 | 1830 | -0.0314061 | 0.0125138 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 10 | max_up | 8 | 0.0661559 | 1830 | 0.0312277 | 0.0349282 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 10 | terminal_return | 8 | 0.0244269 | 1830 | 0.00426764 | 0.0201593 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 20 | max_down | 8 | -0.0380441 | 1820 | -0.0450186 | 0.00697457 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 20 | max_up | 8 | 0.0745476 | 1820 | 0.0474141 | 0.0271335 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 中证500 | 20 | terminal_return | 8 | 0.00947248 | 1820 | 0.00851416 | 0.000958314 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 5 | max_down | 8 | -0.021628 | 1835 | -0.0261911 | 0.00456304 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 5 | max_up | 8 | 0.0572821 | 1835 | 0.0276838 | 0.0295983 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 5 | terminal_return | 8 | 0.0201557 | 1835 | 0.00683701 | 0.0133187 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 10 | max_down | 8 | -0.0328687 | 1830 | -0.0382268 | 0.00535807 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 10 | max_up | 8 | 0.0843999 | 1830 | 0.0429913 | 0.0414086 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 10 | terminal_return | 8 | 0.0337282 | 1830 | 0.0134445 | 0.0202837 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 20 | max_down | 8 | -0.0655798 | 1820 | -0.0553158 | -0.0102641 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 20 | max_up | 8 | 0.0986138 | 1820 | 0.0664582 | 0.0321556 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 微盘股 | 20 | terminal_return | 8 | 0.0231697 | 1820 | 0.0265453 | -0.00337557 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 5 | max_down | 8 | -0.0123836 | 1835 | -0.01635 | 0.00396643 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 5 | max_up | 8 | 0.0352721 | 1835 | 0.0155468 | 0.0197253 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 5 | terminal_return | 8 | 0.00463229 | 1835 | 0.00151442 | 0.00311787 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 10 | max_down | 8 | -0.01497 | 1830 | -0.0235037 | 0.00853365 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 10 | max_up | 8 | 0.0439307 | 1830 | 0.0232844 | 0.0206463 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 10 | terminal_return | 8 | 0.00747305 | 1830 | 0.00290303 | 0.00457001 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 20 | max_down | 8 | -0.0278984 | 1820 | -0.0335422 | 0.00564375 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 20 | max_up | 8 | 0.0442963 | 1820 | 0.035004 | 0.00929236 |  |  |  |  |  | false |
| ml_today_calibrated_elastic_net | top | onset | 上证指数 | 20 | terminal_return | 8 | -0.00652436 | 1820 | 0.00562288 | -0.0121472 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 5 | max_down | 5 | -0.0331542 | 1838 | -0.019818 | -0.0133362 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 5 | max_up | 5 | 0.0166944 | 1838 | 0.0186362 | -0.00194187 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 5 | terminal_return | 5 | 0.00348961 | 1838 | 0.00189158 | 0.00159803 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 10 | max_down | 5 | -0.0618982 | 1833 | -0.0287092 | -0.033189 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 10 | max_up | 5 | 0.0194957 | 1833 | 0.0278306 | -0.00833495 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 10 | terminal_return | 5 | -0.0284164 | 1833 | 0.00373005 | -0.0321465 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 20 | max_down | 5 | -0.0745253 | 1823 | -0.0411782 | -0.0333471 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 20 | max_up | 5 | 0.0208028 | 1823 | 0.0418453 | -0.0210425 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 全A | 20 | terminal_return | 5 | -0.0232756 | 1823 | 0.00709494 | -0.0303705 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 5 | max_down | 5 | -0.0380489 | 1838 | -0.0252412 | -0.0128077 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 5 | max_up | 5 | 0.026734 | 1838 | 0.0228664 | 0.00386761 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 5 | 0.0103089 | 1838 | 0.00254006 | 0.00776888 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 10 | max_down | 5 | -0.0780644 | 1833 | -0.037013 | -0.0410513 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 10 | max_up | 5 | 0.0319356 | 1833 | 0.0344836 | -0.00254802 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 5 | -0.0284887 | 1833 | 0.0049534 | -0.0334421 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 20 | max_down | 5 | -0.0929085 | 1823 | -0.0539746 | -0.0389339 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 20 | max_up | 5 | 0.0424306 | 1823 | 0.0515069 | -0.00907636 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 5 | -0.0124794 | 1823 | 0.00939382 | -0.0218732 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 5 | max_down | 5 | -0.0379974 | 1838 | -0.0248744 | -0.013123 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 5 | max_up | 5 | 0.0230376 | 1838 | 0.0226762 | 0.000361405 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 5 | 0.00918478 | 1838 | 0.00212528 | 0.0070595 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 10 | max_down | 5 | -0.0775312 | 1833 | -0.0364443 | -0.0410869 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 10 | max_up | 5 | 0.0273607 | 1833 | 0.0339017 | -0.006541 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 5 | -0.0315107 | 1833 | 0.00414572 | -0.0356564 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 20 | max_down | 5 | -0.0920253 | 1823 | -0.05291 | -0.0391153 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 20 | max_up | 5 | 0.0400592 | 1823 | 0.0506289 | -0.0105697 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 5 | -0.0146867 | 1823 | 0.00783522 | -0.0225219 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 5 | max_down | 5 | -0.0341282 | 1838 | -0.0180768 | -0.0160513 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 5 | max_up | 5 | 0.0135981 | 1838 | 0.0183036 | -0.00470552 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 5 | -0.00257517 | 1838 | 0.00157772 | -0.00415289 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 10 | max_down | 5 | -0.0540692 | 1833 | -0.0259496 | -0.0281196 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 10 | max_up | 5 | 0.015412 | 1833 | 0.0274338 | -0.0120218 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 5 | -0.0285807 | 1833 | 0.00309634 | -0.0316771 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 20 | max_down | 5 | -0.0746926 | 1823 | -0.0366567 | -0.0380358 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 20 | max_up | 5 | 0.015412 | 1823 | 0.0414177 | -0.0260057 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 5 | -0.0316497 | 1823 | 0.00590258 | -0.0375523 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 5 | max_down | 5 | -0.0333053 | 1838 | -0.0216193 | -0.011686 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 5 | max_up | 5 | 0.0242598 | 1838 | 0.0210211 | 0.0032387 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 5 | 0.0104469 | 1838 | 0.00225564 | 0.00819125 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 10 | max_down | 5 | -0.0656702 | 1833 | -0.031258 | -0.0344122 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 10 | max_up | 5 | 0.0281868 | 1833 | 0.0313885 | -0.00320162 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 5 | -0.0257099 | 1833 | 0.00443739 | -0.0301473 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 20 | max_down | 5 | -0.0763595 | 1823 | -0.0449021 | -0.0314574 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 20 | max_up | 5 | 0.0320884 | 1823 | 0.0475752 | -0.0154868 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 5 | -0.0126564 | 1823 | 0.00857643 | -0.0212329 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 5 | max_down | 5 | -0.0379173 | 1838 | -0.0261393 | -0.011778 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 5 | max_up | 5 | 0.029683 | 1838 | 0.0278072 | 0.00187576 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 5 | 0.012542 | 1838 | 0.00687947 | 0.00566252 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 10 | max_down | 5 | -0.0744805 | 1833 | -0.0381045 | -0.036376 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 10 | max_up | 5 | 0.0443408 | 1833 | 0.0431684 | 0.00117245 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 5 | -0.0116283 | 1833 | 0.0136014 | -0.0252298 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 20 | max_down | 5 | -0.088473 | 1823 | -0.0552699 | -0.0332031 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 20 | max_up | 5 | 0.0617479 | 1823 | 0.0666122 | -0.00486426 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 5 | 0.00994716 | 1823 | 0.026576 | -0.0166288 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 5 | max_down | 5 | -0.0298146 | 1838 | -0.0162961 | -0.0135184 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 5 | max_up | 5 | 0.0166168 | 1838 | 0.0156297 | 0.000987061 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 5 | 0.00283262 | 1838 | 0.0015244 | 0.00130822 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 10 | max_down | 5 | -0.0522297 | 1833 | -0.0233881 | -0.0288417 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 10 | max_up | 5 | 0.0192949 | 1833 | 0.0233854 | -0.00409051 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 5 | -0.0223466 | 1833 | 0.00299185 | -0.0253385 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 20 | max_down | 5 | -0.0647234 | 1823 | -0.0334319 | -0.0312915 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 20 | max_up | 5 | 0.0192949 | 1823 | 0.0350879 | -0.015793 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 5 | -0.0169183 | 1823 | 0.0056314 | -0.0225497 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 5 | max_down | 5 | -0.029244 | 1838 | -0.0198286 | -0.00941543 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 5 | max_up | 5 | 0.0147322 | 1838 | 0.0186416 | -0.00390942 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 5 | terminal_return | 5 | -0.0110714 | 1838 | 0.00193119 | -0.0130026 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 10 | max_down | 5 | -0.0552398 | 1833 | -0.0287273 | -0.0265125 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 10 | max_up | 5 | 0.0212344 | 1833 | 0.0278259 | -0.0065915 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 10 | terminal_return | 5 | -0.0252982 | 1833 | 0.00372154 | -0.0290198 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 20 | max_down | 5 | -0.0695897 | 1823 | -0.0411917 | -0.0283979 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 20 | max_up | 5 | 0.0218177 | 1823 | 0.0418425 | -0.0200247 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 全A | 20 | terminal_return | 5 | -0.0246094 | 1823 | 0.0070986 | -0.0317079 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 5 | max_down | 5 | -0.0348454 | 1838 | -0.0252499 | -0.00959548 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 5 | max_up | 5 | 0.0238637 | 1838 | 0.0228742 | 0.000989544 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 5 | terminal_return | 5 | -0.00347235 | 1838 | 0.00257755 | -0.0060499 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 10 | max_down | 5 | -0.0669093 | 1833 | -0.0370435 | -0.0298659 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 10 | max_up | 5 | 0.0347572 | 1833 | 0.0344759 | 0.000281326 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 10 | terminal_return | 5 | -0.0228418 | 1833 | 0.004938 | -0.0277798 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 20 | max_down | 5 | -0.0875612 | 1823 | -0.0539893 | -0.033572 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 20 | max_up | 5 | 0.0414506 | 1823 | 0.0515096 | -0.0100591 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 国证2000 | 20 | terminal_return | 5 | -0.0120246 | 1823 | 0.00939257 | -0.0214172 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 5 | max_down | 5 | -0.034997 | 1838 | -0.0248825 | -0.0101145 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 5 | max_up | 5 | 0.020381 | 1838 | 0.0226834 | -0.00230242 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 5 | terminal_return | 5 | -0.0070003 | 1838 | 0.00216931 | -0.0091696 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 10 | max_down | 5 | -0.0687242 | 1833 | -0.0364683 | -0.0322559 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 10 | max_up | 5 | 0.0280965 | 1833 | 0.0338997 | -0.00580324 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 10 | terminal_return | 5 | -0.0276147 | 1833 | 0.0041351 | -0.0317497 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 20 | max_down | 5 | -0.0877614 | 1823 | -0.0529217 | -0.0348397 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 20 | max_up | 5 | 0.0360212 | 1823 | 0.05064 | -0.0146188 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证1000 | 20 | terminal_return | 5 | -0.0149532 | 1823 | 0.00783595 | -0.0227891 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 5 | max_down | 5 | -0.0327264 | 1838 | -0.0180807 | -0.0146458 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 5 | max_up | 5 | 0.0130446 | 1838 | 0.0183051 | -0.00526052 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 5 | terminal_return | 5 | -0.0171386 | 1838 | 0.00161734 | -0.0187559 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 10 | max_down | 5 | -0.0498486 | 1833 | -0.0259611 | -0.0238875 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 10 | max_up | 5 | 0.0171593 | 1833 | 0.027429 | -0.0102697 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 10 | terminal_return | 5 | -0.026999 | 1833 | 0.00309202 | -0.030091 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 20 | max_down | 5 | -0.0692739 | 1823 | -0.0366716 | -0.0326023 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 20 | max_up | 5 | 0.0171593 | 1823 | 0.0414129 | -0.0242536 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 沪深300 | 20 | terminal_return | 5 | -0.0332255 | 1823 | 0.0059069 | -0.0391324 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 5 | max_down | 5 | -0.0300148 | 1838 | -0.0216282 | -0.00838657 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 5 | max_up | 5 | 0.0174846 | 1838 | 0.0210395 | -0.00355487 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 5 | terminal_return | 5 | -0.00626456 | 1838 | 0.0023011 | -0.00856566 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 10 | max_down | 5 | -0.0594422 | 1833 | -0.031275 | -0.0281672 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 10 | max_up | 5 | 0.0272337 | 1833 | 0.0313911 | -0.0041574 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 10 | terminal_return | 5 | -0.0231363 | 1833 | 0.00443037 | -0.0275667 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 20 | max_down | 5 | -0.0742974 | 1823 | -0.0449077 | -0.0293897 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 20 | max_up | 5 | 0.0282337 | 1823 | 0.0475857 | -0.0193521 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 中证500 | 20 | terminal_return | 5 | -0.0178149 | 1823 | 0.00859058 | -0.0264055 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 5 | max_down | 5 | -0.0317643 | 1838 | -0.0261561 | -0.00560822 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 5 | max_up | 5 | 0.0326338 | 1838 | 0.0277992 | 0.00483468 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 5 | terminal_return | 5 | 0.00667147 | 1838 | 0.00689544 | -0.000223969 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 10 | max_down | 5 | -0.0547309 | 1833 | -0.0381584 | -0.0165725 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 10 | max_up | 5 | 0.0501275 | 1833 | 0.0431526 | 0.00697488 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 10 | terminal_return | 5 | 0.00142341 | 1833 | 0.0135658 | -0.0121424 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 20 | max_down | 5 | -0.0785832 | 1823 | -0.055297 | -0.0232862 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 20 | max_up | 5 | 0.0648491 | 1823 | 0.0666037 | -0.00175457 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 微盘股 | 20 | terminal_return | 5 | 0.0148787 | 1823 | 0.0265625 | -0.0116837 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 5 | max_down | 5 | -0.0279096 | 1838 | -0.0163013 | -0.0116083 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 5 | max_up | 5 | 0.0135031 | 1838 | 0.0156382 | -0.00213508 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 5 | terminal_return | 5 | -0.0094627 | 1838 | 0.00155785 | -0.0110206 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 10 | max_down | 5 | -0.0466574 | 1833 | -0.0234033 | -0.0232541 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 10 | max_up | 5 | 0.0202305 | 1833 | 0.0233828 | -0.00315226 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 10 | terminal_return | 5 | -0.020437 | 1833 | 0.00298664 | -0.0234237 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 20 | max_down | 5 | -0.0605358 | 1823 | -0.0334434 | -0.0270924 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 20 | max_up | 5 | 0.0202305 | 1823 | 0.0350853 | -0.0148547 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | bottom | onset | 上证指数 | 20 | terminal_return | 5 | -0.0202212 | 1823 | 0.00564045 | -0.0258616 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 5 | max_down | 1 | 0.00174323 | 1842 | -0.0198659 | 0.0216091 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 5 | max_up | 1 | 0.0243289 | 1842 | 0.0186279 | 0.00570098 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 5 | terminal_return | 1 | 0.0216218 | 1842 | 0.0018852 | 0.0197366 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 10 | max_down | 1 | 0.00174323 | 1837 | -0.0288161 | 0.0305593 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 10 | max_up | 1 | 0.0331367 | 1837 | 0.027805 | 0.00533171 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 10 | terminal_return | 1 | 0.00990913 | 1837 | 0.00363919 | 0.00626994 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 20 | max_down | 1 | 0.00174323 | 1827 | -0.041293 | 0.0430362 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 20 | max_up | 1 | 0.060139 | 1827 | 0.0417777 | 0.0183613 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 全A | 20 | terminal_return | 1 | 0.0590094 | 1827 | 0.00698341 | 0.052026 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 5 | max_down | 1 | 0.000504671 | 1842 | -0.0252899 | 0.0257946 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 5 | max_up | 1 | 0.021709 | 1842 | 0.0228775 | -0.00116852 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 5 | terminal_return | 1 | 0.0211626 | 1842 | 0.00255103 | 0.0186115 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 10 | max_down | 1 | 0.000504671 | 1837 | -0.0371452 | 0.0376499 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 10 | max_up | 1 | 0.0328326 | 1837 | 0.0344776 | -0.00164492 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 10 | terminal_return | 1 | 0.0192494 | 1837 | 0.00485459 | 0.0143949 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 20 | max_down | 1 | 0.000504671 | 1827 | -0.054111 | 0.0546156 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 20 | max_up | 1 | 0.087597 | 1827 | 0.0514624 | 0.0361347 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 国证2000 | 20 | terminal_return | 1 | 0.087597 | 1827 | 0.00929115 | 0.0783059 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 5 | max_down | 1 | 0.000453292 | 1842 | -0.0249238 | 0.025377 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 5 | max_up | 1 | 0.0251264 | 1842 | 0.0226759 | 0.00245055 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 5 | terminal_return | 1 | 0.0235864 | 1842 | 0.00213279 | 0.0214537 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 10 | max_down | 1 | 0.000453292 | 1837 | -0.0365762 | 0.0370295 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 10 | max_up | 1 | 0.0338534 | 1837 | 0.033884 | -3.05409e-05 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 10 | terminal_return | 1 | 0.0180706 | 1837 | 0.00404109 | 0.0140295 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 20 | max_down | 1 | 0.000453292 | 1827 | -0.0530463 | 0.0534996 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 20 | max_up | 1 | 0.0862994 | 1827 | 0.0505805 | 0.0357189 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证1000 | 20 | terminal_return | 1 | 0.0862979 | 1827 | 0.00773064 | 0.0785673 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 5 | max_down | 1 | 0.00120486 | 1842 | -0.0181309 | 0.0193357 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 5 | max_up | 1 | 0.0239322 | 1842 | 0.0182878 | 0.00564439 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 5 | terminal_return | 1 | 0.0169051 | 1842 | 0.00155813 | 0.0153469 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 10 | max_down | 1 | -0.00445726 | 1837 | -0.0260378 | 0.0215806 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 10 | max_up | 1 | 0.0312082 | 1837 | 0.027399 | 0.00380916 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 10 | terminal_return | 1 | -0.000891944 | 1837 | 0.00301229 | -0.00390423 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 20 | max_down | 1 | -0.00507817 | 1827 | -0.0367781 | 0.0316999 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 20 | max_up | 1 | 0.0398566 | 1827 | 0.0413474 | -0.0014908 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 沪深300 | 20 | terminal_return | 1 | 0.0354314 | 1827 | 0.00578365 | 0.0296477 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 5 | max_down | 1 | 0.00186569 | 1842 | -0.0216638 | 0.0235295 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 5 | max_up | 1 | 0.0346842 | 1842 | 0.0210225 | 0.0136617 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 5 | terminal_return | 1 | 0.0327857 | 1842 | 0.0022613 | 0.0305244 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 10 | max_down | 1 | 0.00186569 | 1837 | -0.0313697 | 0.0332354 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 10 | max_up | 1 | 0.0440454 | 1837 | 0.0313728 | 0.0126726 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 10 | terminal_return | 1 | 0.0186225 | 1837 | 0.00434762 | 0.0142749 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 20 | max_down | 1 | 0.00186569 | 1827 | -0.0450138 | 0.0468795 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 20 | max_up | 1 | 0.0772 | 1827 | 0.0475166 | 0.0296834 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 中证500 | 20 | terminal_return | 1 | 0.0768836 | 1827 | 0.00848094 | 0.0684027 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 5 | max_down | 1 | 0.00399247 | 1842 | -0.0261877 | 0.0301801 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 5 | max_up | 1 | 0.0360655 | 1842 | 0.0278078 | 0.0082577 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 5 | terminal_return | 1 | 0.0355416 | 1842 | 0.00687928 | 0.0286623 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 10 | max_down | 1 | 0.00399247 | 1837 | -0.0382264 | 0.0422189 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 10 | max_up | 1 | 0.049204 | 1837 | 0.0431683 | 0.00603571 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 10 | terminal_return | 1 | 0.0461097 | 1837 | 0.0135151 | 0.0325946 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 20 | max_down | 1 | 0.00399247 | 1827 | -0.0553932 | 0.0593856 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 20 | max_up | 1 | 0.114301 | 1827 | 0.0665728 | 0.0477284 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 微盘股 | 20 | terminal_return | 1 | 0.085536 | 1827 | 0.0264982 | 0.0590378 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 5 | max_down | 1 | 0.00228888 | 1842 | -0.0163429 | 0.0186318 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 5 | max_up | 1 | 0.0222211 | 1842 | 0.0156288 | 0.00659228 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 5 | terminal_return | 1 | 0.0167436 | 1842 | 0.00151969 | 0.0152239 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 10 | max_down | 1 | 0.00228888 | 1837 | -0.0234805 | 0.0257694 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 10 | max_up | 1 | 0.0287709 | 1837 | 0.0233713 | 0.00539955 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 10 | terminal_return | 1 | 0.00720615 | 1837 | 0.00292059 | 0.00428556 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 20 | max_down | 1 | 0.00228888 | 1827 | -0.0335371 | 0.035826 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 20 | max_up | 1 | 0.0481796 | 1827 | 0.0350375 | 0.0131422 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | capped_confirmation | 上证指数 | 20 | terminal_return | 1 | 0.0459162 | 1827 | 0.00554763 | 0.0403686 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 5 | max_down | 1 | -0.000269998 | 1842 | -0.0198648 | 0.0195948 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 5 | max_up | 1 | 0.0271639 | 1842 | 0.0186263 | 0.00853759 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 5 | terminal_return | 1 | 0.0271639 | 1842 | 0.00188219 | 0.0252817 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 10 | max_down | 1 | -0.000269998 | 1837 | -0.028815 | 0.028545 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 10 | max_up | 1 | 0.0371215 | 1837 | 0.0278029 | 0.00931868 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 10 | terminal_return | 1 | 0.0153081 | 1837 | 0.00363625 | 0.0116718 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 20 | max_down | 1 | -0.000269998 | 1827 | -0.0412918 | 0.0410219 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 20 | max_up | 1 | 0.061669 | 1827 | 0.0417768 | 0.0198922 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 全A | 20 | terminal_return | 1 | 0.0483253 | 1827 | 0.00698926 | 0.041336 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 5 | max_down | 1 | -0.00349426 | 1842 | -0.0252878 | 0.0217935 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 5 | max_up | 1 | 0.0202163 | 1842 | 0.0228783 | -0.00266203 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 5 | terminal_return | 1 | 0.0202163 | 1842 | 0.00255155 | 0.0176648 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 10 | max_down | 1 | -0.00349426 | 1837 | -0.037143 | 0.0336488 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 10 | max_up | 1 | 0.0341608 | 1837 | 0.0344768 | -0.00031602 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 10 | terminal_return | 1 | 0.0182914 | 1837 | 0.00485511 | 0.0134363 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 20 | max_down | 1 | -0.00349426 | 1827 | -0.0541088 | 0.0506145 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 20 | max_up | 1 | 0.0846185 | 1827 | 0.051464 | 0.0331545 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 国证2000 | 20 | terminal_return | 1 | 0.0663189 | 1827 | 0.0093028 | 0.0570161 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 5 | max_down | 1 | -0.0021681 | 1842 | -0.0249223 | 0.0227542 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 5 | max_up | 1 | 0.0253149 | 1842 | 0.0226758 | 0.00263917 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 5 | terminal_return | 1 | 0.0253149 | 1842 | 0.00213185 | 0.0231831 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 10 | max_down | 1 | -0.0021681 | 1837 | -0.0365748 | 0.0344067 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 10 | max_up | 1 | 0.0364477 | 1837 | 0.0338826 | 0.00256513 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 10 | terminal_return | 1 | 0.0192054 | 1837 | 0.00404048 | 0.0151649 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 20 | max_down | 1 | -0.0021681 | 1827 | -0.0530448 | 0.0508767 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 20 | max_up | 1 | 0.0829509 | 1827 | 0.0505823 | 0.0323686 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证1000 | 20 | terminal_return | 1 | 0.0674483 | 1827 | 0.00774095 | 0.0597074 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 5 | max_down | 1 | 0.00119718 | 1842 | -0.0181309 | 0.0193281 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 5 | max_up | 1 | 0.0300385 | 1842 | 0.0182845 | 0.011754 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 5 | terminal_return | 1 | 0.0283927 | 1842 | 0.00155189 | 0.0268408 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 10 | max_down | 1 | 0.00119718 | 1837 | -0.0260409 | 0.0272381 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 10 | max_up | 1 | 0.0373579 | 1837 | 0.0273957 | 0.0099622 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 10 | terminal_return | 1 | 0.0101872 | 1837 | 0.00300626 | 0.0071809 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 20 | max_down | 1 | 0.000855127 | 1827 | -0.0367814 | 0.0376365 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 20 | max_up | 1 | 0.0460579 | 1827 | 0.041344 | 0.00471386 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 沪深300 | 20 | terminal_return | 1 | 0.0344083 | 1827 | 0.00578421 | 0.0286241 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 5 | max_down | 1 | -0.00145007 | 1842 | -0.021662 | 0.0202119 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 5 | max_up | 1 | 0.0347129 | 1842 | 0.0210224 | 0.0136905 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 5 | terminal_return | 1 | 0.0347129 | 1842 | 0.00226025 | 0.0324527 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 10 | max_down | 1 | -0.00145007 | 1837 | -0.0313679 | 0.0299178 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 10 | max_up | 1 | 0.0469892 | 1837 | 0.0313712 | 0.015618 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 10 | terminal_return | 1 | 0.0236549 | 1837 | 0.00434488 | 0.01931 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 20 | max_down | 1 | -0.00145007 | 1827 | -0.0450119 | 0.0435619 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 20 | max_up | 1 | 0.0719002 | 1827 | 0.0475195 | 0.0243807 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 中证500 | 20 | terminal_return | 1 | 0.0571134 | 1827 | 0.00849176 | 0.0486216 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 5 | max_down | 1 | -0.00383246 | 1842 | -0.0261834 | 0.0223509 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 5 | max_up | 1 | 0.0334616 | 1842 | 0.0278092 | 0.00565235 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 5 | terminal_return | 1 | 0.0334616 | 1842 | 0.0068804 | 0.0265812 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 10 | max_down | 1 | -0.00383246 | 1837 | -0.0382222 | 0.0343897 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 10 | max_up | 1 | 0.0538965 | 1837 | 0.0431657 | 0.0107307 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 10 | terminal_return | 1 | 0.0329335 | 1837 | 0.0135222 | 0.0194112 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 20 | max_down | 1 | -0.00383246 | 1827 | -0.0553889 | 0.0515564 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 20 | max_up | 1 | 0.119285 | 1827 | 0.0665701 | 0.0527147 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 微盘股 | 20 | terminal_return | 1 | 0.0810592 | 1827 | 0.0265007 | 0.0545586 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 5 | max_down | 1 | 0.000398086 | 1842 | -0.0163419 | 0.01674 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 5 | max_up | 1 | 0.0273513 | 1842 | 0.015626 | 0.0117253 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 5 | terminal_return | 1 | 0.0252784 | 1842 | 0.00151506 | 0.0237634 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 10 | max_down | 1 | 0.000398086 | 1837 | -0.0234795 | 0.0238776 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 10 | max_up | 1 | 0.033934 | 1837 | 0.0233685 | 0.0105655 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 10 | terminal_return | 1 | 0.0160315 | 1837 | 0.00291579 | 0.0131157 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 20 | max_down | 1 | 0.000398086 | 1827 | -0.0335361 | 0.0339341 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 20 | max_up | 1 | 0.0534402 | 1827 | 0.0350346 | 0.0184056 |  |  |  |  |  | false |
| ml_today_calibrated_shallow_gbdt | top | onset | 上证指数 | 20 | terminal_return | 1 | 0.0425412 | 1827 | 0.00554948 | 0.0369917 |  |  |  |  |  | false |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `ml_today_calibrated_elastic_net/bottom/all_a_ml_today_walk_forward_v2/capped_confirmation`：63/63 项检验未达到样本门槛，仅可读取描述统计。
- `ml_today_calibrated_elastic_net/bottom/all_a_ml_today_walk_forward_v2/onset`：63/63 项检验未达到样本门槛，仅可读取描述统计。
- `ml_today_calibrated_elastic_net/top/all_a_ml_today_walk_forward_v2/capped_confirmation`：63/63 项检验未达到样本门槛，仅可读取描述统计。
- `ml_today_calibrated_elastic_net/top/all_a_ml_today_walk_forward_v2/onset`：63/63 项检验未达到样本门槛，仅可读取描述统计。
- `ml_today_calibrated_shallow_gbdt/bottom/all_a_ml_today_walk_forward_v2/capped_confirmation`：63/63 项检验未达到样本门槛，仅可读取描述统计。
- `ml_today_calibrated_shallow_gbdt/bottom/all_a_ml_today_walk_forward_v2/onset`：63/63 项检验未达到样本门槛，仅可读取描述统计。
- `ml_today_calibrated_shallow_gbdt/top/all_a_ml_today_walk_forward_v2/capped_confirmation`：63/63 项检验未达到样本门槛，仅可读取描述统计。
- `ml_today_calibrated_shallow_gbdt/top/all_a_ml_today_walk_forward_v2/onset`：63/63 项检验未达到样本门槛，仅可读取描述统计。
