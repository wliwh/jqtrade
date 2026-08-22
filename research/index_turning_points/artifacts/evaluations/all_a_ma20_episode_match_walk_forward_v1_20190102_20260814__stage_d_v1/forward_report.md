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
| 上证指数 | capped_confirmation | 5 | 22 | 22 | 22 |
| 上证指数 | capped_confirmation | 10 | 22 | 22 | 22 |
| 上证指数 | capped_confirmation | 20 | 22 | 22 | 21 |
| 上证指数 | onset | 5 | 22 | 22 | 22 |
| 上证指数 | onset | 10 | 22 | 22 | 22 |
| 上证指数 | onset | 20 | 22 | 22 | 21 |
| 中证1000 | capped_confirmation | 5 | 22 | 22 | 22 |
| 中证1000 | capped_confirmation | 10 | 22 | 22 | 22 |
| 中证1000 | capped_confirmation | 20 | 22 | 22 | 21 |
| 中证1000 | onset | 5 | 22 | 22 | 22 |
| 中证1000 | onset | 10 | 22 | 22 | 22 |
| 中证1000 | onset | 20 | 22 | 22 | 21 |
| 中证500 | capped_confirmation | 5 | 22 | 22 | 22 |
| 中证500 | capped_confirmation | 10 | 22 | 22 | 22 |
| 中证500 | capped_confirmation | 20 | 22 | 22 | 21 |
| 中证500 | onset | 5 | 22 | 22 | 22 |
| 中证500 | onset | 10 | 22 | 22 | 22 |
| 中证500 | onset | 20 | 22 | 22 | 21 |
| 全A | capped_confirmation | 5 | 22 | 22 | 22 |
| 全A | capped_confirmation | 10 | 22 | 22 | 22 |
| 全A | capped_confirmation | 20 | 22 | 22 | 21 |
| 全A | onset | 5 | 22 | 22 | 22 |
| 全A | onset | 10 | 22 | 22 | 22 |
| 全A | onset | 20 | 22 | 22 | 21 |
| 国证2000 | capped_confirmation | 5 | 22 | 22 | 22 |
| 国证2000 | capped_confirmation | 10 | 22 | 22 | 22 |
| 国证2000 | capped_confirmation | 20 | 22 | 22 | 21 |
| 国证2000 | onset | 5 | 22 | 22 | 22 |
| 国证2000 | onset | 10 | 22 | 22 | 22 |
| 国证2000 | onset | 20 | 22 | 22 | 21 |
| 微盘股 | capped_confirmation | 5 | 22 | 22 | 22 |
| 微盘股 | capped_confirmation | 10 | 22 | 22 | 22 |
| 微盘股 | capped_confirmation | 20 | 22 | 22 | 21 |
| 微盘股 | onset | 5 | 22 | 22 | 22 |
| 微盘股 | onset | 10 | 22 | 22 | 22 |
| 微盘股 | onset | 20 | 22 | 22 | 21 |
| 沪深300 | capped_confirmation | 5 | 22 | 22 | 22 |
| 沪深300 | capped_confirmation | 10 | 22 | 22 | 22 |
| 沪深300 | capped_confirmation | 20 | 22 | 22 | 21 |
| 沪深300 | onset | 5 | 22 | 22 | 22 |
| 沪深300 | onset | 10 | 22 | 22 | 22 |
| 沪深300 | onset | 20 | 22 | 22 | 21 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 5 | max_down | 20 | -0.0238288 | 1823 | -0.0198105 | -0.00401829 | -0.0145419 | 0.00650533 | 0.454221 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 5 | max_up | 20 | 0.0164613 | 1823 | 0.0186548 | -0.00219344 | -0.00927175 | 0.00488486 | 0.543606 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 5 | terminal_return | 20 | -0.00195036 | 1823 | 0.00193811 | -0.00388847 | -0.0180708 | 0.0102939 | 0.591001 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 10 | max_down | 20 | -0.0305387 | 1818 | -0.0287803 | -0.0017584 | -0.0139679 | 0.0104511 | 0.777732 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 10 | max_up | 20 | 0.0263831 | 1818 | 0.0278236 | -0.00144054 | -0.0123878 | 0.0095067 | 0.796473 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 10 | terminal_return | 20 | 0.0108645 | 1818 | 0.00356315 | 0.00730134 | -0.00695304 | 0.0215557 | 0.315405 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 20 | max_down | 19 | -0.0453469 | 1809 | -0.0412266 | -0.00412029 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 20 | max_up | 19 | 0.0564437 | 1809 | 0.0416338 | 0.0148099 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 全A | 20 | terminal_return | 19 | 0.0183966 | 1809 | 0.0068923 | 0.0115043 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 5 | max_down | 20 | -0.0324575 | 1823 | -0.0251972 | -0.00726036 | -0.023062 | 0.00854125 | 0.367822 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 5 | max_up | 20 | 0.0207145 | 1823 | 0.0229006 | -0.0021861 | -0.0120721 | 0.00769994 | 0.664714 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 20 | -0.00401648 | 1823 | 0.0026333 | -0.00664977 | -0.0269667 | 0.0136671 | 0.52119 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 10 | max_down | 20 | -0.0441774 | 1818 | -0.0370471 | -0.00713024 | -0.0273318 | 0.0130713 | 0.489068 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 10 | max_up | 20 | 0.0345131 | 1818 | 0.0344763 | 3.68132e-05 | -0.0159337 | 0.0160073 | 0.996395 | 0.996395 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 20 | 0.00971894 | 1818 | 0.004809 | 0.00490994 | -0.0156584 | 0.0254783 | 0.639872 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 20 | max_down | 19 | -0.0644879 | 1809 | -0.0539718 | -0.0105161 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 20 | max_up | 19 | 0.0702884 | 1809 | 0.0512846 | 0.0190038 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 19 | 0.021873 | 1809 | 0.00920229 | 0.0126707 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 5 | max_down | 20 | -0.031788 | 1823 | -0.0248345 | -0.00695351 | -0.0212395 | 0.00733245 | 0.340081 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 5 | max_up | 20 | 0.0196213 | 1823 | 0.0227107 | -0.0030894 | -0.0121315 | 0.00595273 | 0.503069 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 20 | -0.00483518 | 1823 | 0.002221 | -0.00705618 | -0.0259879 | 0.0118755 | 0.465068 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 10 | max_down | 20 | -0.0427977 | 1818 | -0.0364874 | -0.00631033 | -0.0242125 | 0.0115919 | 0.489643 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 10 | max_up | 20 | 0.033823 | 1818 | 0.0338846 | -6.16498e-05 | -0.0151371 | 0.0150138 | 0.993605 | 0.996395 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 20 | 0.0115307 | 1818 | 0.00396642 | 0.00756432 | -0.0122626 | 0.0273912 | 0.454596 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 20 | max_down | 19 | -0.0614554 | 1809 | -0.0529284 | -0.00852705 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 20 | max_up | 19 | 0.0685736 | 1809 | 0.0504113 | 0.0181623 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 19 | 0.0182417 | 1809 | 0.00766367 | 0.010578 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 5 | max_down | 20 | -0.0187829 | 1823 | -0.0181131 | -0.000669752 | -0.00751186 | 0.00617235 | 0.847853 | 0.943235 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 5 | max_up | 20 | 0.0164187 | 1823 | 0.0183114 | -0.00189273 | -0.00847377 | 0.00468832 | 0.572957 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 20 | 0.000144224 | 1823 | 0.00158206 | -0.00143783 | -0.0121516 | 0.0092759 | 0.792519 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 10 | max_down | 20 | -0.0228407 | 1818 | -0.0260611 | 0.00322044 | -0.00472864 | 0.0111695 | 0.427159 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 10 | max_up | 20 | 0.0266589 | 1818 | 0.0274093 | -0.000750351 | -0.0101647 | 0.00866399 | 0.875861 | 0.943235 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 20 | 0.011736 | 1818 | 0.00291417 | 0.00882184 | -0.00409482 | 0.0217385 | 0.180687 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 20 | max_down | 19 | -0.0358965 | 1809 | -0.0367698 | 0.000873311 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 20 | max_up | 19 | 0.0565628 | 1809 | 0.0411868 | 0.015376 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 19 | 0.0177716 | 1809 | 0.00567413 | 0.0120975 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 5 | max_down | 20 | -0.0270589 | 1823 | -0.0215917 | -0.00546722 | -0.016318 | 0.00538359 | 0.323371 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 5 | max_up | 20 | 0.0170587 | 1823 | 0.0210734 | -0.00401473 | -0.0113654 | 0.00333591 | 0.284394 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 20 | -0.00448463 | 1823 | 0.00235205 | -0.00683668 | -0.0219459 | 0.00827254 | 0.37515 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 10 | max_down | 20 | -0.0349116 | 1818 | -0.0313124 | -0.00359914 | -0.0172576 | 0.0100593 | 0.60552 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 10 | max_up | 20 | 0.0307679 | 1818 | 0.0313865 | -0.000618554 | -0.0132008 | 0.0119637 | 0.923239 | 0.9694 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 20 | 0.0116001 | 1818 | 0.00427568 | 0.00732446 | -0.00863901 | 0.0232879 | 0.368493 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 20 | max_down | 19 | -0.0515287 | 1809 | -0.0449194 | -0.00660932 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 20 | max_up | 19 | 0.0608896 | 1809 | 0.0473925 | 0.0134971 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 19 | 0.0150542 | 1809 | 0.00844971 | 0.00660447 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 5 | max_down | 20 | -0.0416322 | 1823 | -0.0260017 | -0.0156305 | -0.0376187 | 0.00635764 | 0.163533 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 5 | max_up | 20 | 0.0334836 | 1823 | 0.0277501 | 0.00573357 | -0.0171571 | 0.0286242 | 0.623474 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 20 | 0.00115001 | 1823 | 0.00695785 | -0.00580784 | -0.0400085 | 0.0283928 | 0.739255 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 10 | max_down | 20 | -0.0599568 | 1818 | -0.0379641 | -0.0219927 | -0.0587967 | 0.0148113 | 0.24151 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 10 | max_up | 20 | 0.0528893 | 1818 | 0.0430646 | 0.00982462 | -0.0237289 | 0.0433781 | 0.566037 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 20 | 0.00791056 | 1818 | 0.0135947 | -0.00568409 | -0.0454682 | 0.0341001 | 0.779453 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 20 | max_down | 19 | -0.0717306 | 1809 | -0.0551887 | -0.0165419 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 20 | max_up | 19 | 0.0902905 | 1809 | 0.0663501 | 0.0239405 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 19 | 0.0458739 | 1809 | 0.0263274 | 0.0195465 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 5 | max_down | 20 | -0.0177064 | 1823 | -0.0163177 | -0.00138866 | -0.00931661 | 0.00653929 | 0.731362 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 5 | max_up | 20 | 0.0139087 | 1823 | 0.0156513 | -0.00174262 | -0.00746346 | 0.00397821 | 0.550483 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 20 | -0.00055586 | 1823 | 0.00155081 | -0.00210667 | -0.012837 | 0.00862369 | 0.700383 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 10 | max_down | 20 | -0.0227122 | 1818 | -0.0234748 | 0.000762611 | -0.00811608 | 0.0096413 | 0.866309 | 0.943235 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 10 | max_up | 20 | 0.0192714 | 1818 | 0.0234194 | -0.00414793 | -0.0123617 | 0.00406581 | 0.322273 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 20 | 0.00660319 | 1818 | 0.00288244 | 0.00372075 | -0.00680465 | 0.0142462 | 0.488395 | 0.929219 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 20 | max_down | 19 | -0.0361716 | 1809 | -0.0334896 | -0.00268196 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 20 | max_up | 19 | 0.043892 | 1809 | 0.0349517 | 0.0089403 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 19 | 0.0115429 | 1809 | 0.00550698 | 0.00603591 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 5 | max_down | 20 | -0.0190549 | 1823 | -0.0198629 | 0.000808041 | -0.00759804 | 0.00921412 | 0.850558 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 5 | max_up | 20 | 0.0216067 | 1823 | 0.0185983 | 0.0030084 | -0.00560513 | 0.0116219 | 0.493623 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 5 | terminal_return | 20 | 0.00302594 | 1823 | 0.00188351 | 0.00114242 | -0.0126546 | 0.0149395 | 0.871076 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 10 | max_down | 20 | -0.0278357 | 1818 | -0.0288101 | 0.000974391 | -0.0117588 | 0.0137075 | 0.880775 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 10 | max_up | 20 | 0.0313847 | 1818 | 0.0277686 | 0.00361614 | -0.00912736 | 0.0163596 | 0.57809 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 10 | terminal_return | 20 | 0.0114049 | 1818 | 0.00355721 | 0.00784766 | -0.00851971 | 0.024215 | 0.34734 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 20 | max_down | 19 | -0.0413382 | 1809 | -0.0412687 | -6.94887e-05 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 20 | max_up | 19 | 0.0534657 | 1809 | 0.0416651 | 0.0118006 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 全A | 20 | terminal_return | 19 | 0.0175116 | 1809 | 0.00690159 | 0.01061 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 5 | max_down | 20 | -0.027442 | 1823 | -0.0252522 | -0.00218978 | -0.0153204 | 0.0109408 | 0.743767 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 5 | max_up | 20 | 0.0282495 | 1823 | 0.0228179 | 0.00543156 | -0.00749622 | 0.0183593 | 0.410231 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 5 | terminal_return | 20 | 0.00409711 | 1823 | 0.00254428 | 0.00155283 | -0.0195486 | 0.0226542 | 0.885316 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 10 | max_down | 20 | -0.0425004 | 1818 | -0.0370656 | -0.00543477 | -0.0267615 | 0.0158919 | 0.617445 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 10 | max_up | 20 | 0.0433352 | 1818 | 0.0343792 | 0.00895596 | -0.0116783 | 0.0295902 | 0.394933 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 10 | terminal_return | 20 | 0.0139136 | 1818 | 0.00476285 | 0.00915073 | -0.0161591 | 0.0344605 | 0.47855 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 20 | max_down | 19 | -0.0594474 | 1809 | -0.0540247 | -0.00542266 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 20 | max_up | 19 | 0.0698247 | 1809 | 0.0512895 | 0.0185353 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 国证2000 | 20 | terminal_return | 19 | 0.0235004 | 1809 | 0.0091852 | 0.0143152 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 5 | max_down | 20 | -0.0258254 | 1823 | -0.0248999 | -0.00092548 | -0.0129234 | 0.0110724 | 0.879827 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 5 | max_up | 20 | 0.0265503 | 1823 | 0.0226347 | 0.00391561 | -0.00661801 | 0.0144492 | 0.466258 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 5 | terminal_return | 20 | 0.0022214 | 1823 | 0.00214358 | 7.78107e-05 | -0.0181665 | 0.0183221 | 0.99333 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 10 | max_down | 20 | -0.0399565 | 1818 | -0.0365186 | -0.00343792 | -0.0225401 | 0.0156643 | 0.724275 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 10 | max_up | 20 | 0.0407013 | 1818 | 0.0338089 | 0.00689233 | -0.0101365 | 0.0239211 | 0.427602 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 10 | terminal_return | 20 | 0.0143809 | 1818 | 0.00393506 | 0.0104459 | -0.0120923 | 0.0329841 | 0.363662 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 20 | max_down | 19 | -0.0562263 | 1809 | -0.0529833 | -0.00324304 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 20 | max_up | 19 | 0.0668152 | 1809 | 0.0504297 | 0.0163855 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证1000 | 20 | terminal_return | 19 | 0.0182865 | 1809 | 0.0076632 | 0.0106233 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 5 | max_down | 20 | -0.0148176 | 1823 | -0.0181566 | 0.00333898 | -0.00188717 | 0.00856514 | 0.210482 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 5 | max_up | 20 | 0.0195813 | 1823 | 0.0182767 | 0.00130461 | -0.00585628 | 0.00846549 | 0.721029 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 5 | terminal_return | 20 | 0.00346035 | 1823 | 0.00154568 | 0.00191467 | -0.00786815 | 0.0116975 | 0.70127 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 10 | max_down | 20 | -0.020589 | 1818 | -0.0260859 | 0.00549688 | -0.00128054 | 0.0122743 | 0.111908 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 10 | max_up | 20 | 0.0278843 | 1818 | 0.0273958 | 0.000488519 | -0.00927299 | 0.01025 | 0.921862 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 10 | terminal_return | 20 | 0.0105588 | 1818 | 0.00292712 | 0.00763165 | -0.00512576 | 0.0203891 | 0.240997 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 20 | max_down | 19 | -0.0324825 | 1809 | -0.0368057 | 0.00432322 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 20 | max_up | 19 | 0.0519738 | 1809 | 0.041235 | 0.0107388 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 沪深300 | 20 | terminal_return | 19 | 0.0155746 | 1809 | 0.0056972 | 0.00987742 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 5 | max_down | 20 | -0.0214618 | 1823 | -0.0216531 | 0.000191276 | -0.00887048 | 0.00925304 | 0.966999 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 5 | max_up | 20 | 0.023456 | 1823 | 0.0210032 | 0.00245271 | -0.00695221 | 0.0118576 | 0.609247 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 5 | terminal_return | 20 | 0.000703802 | 1823 | 0.00229513 | -0.00159133 | -0.0156354 | 0.0124528 | 0.824247 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 10 | max_down | 20 | -0.031868 | 1818 | -0.0313459 | -0.000522104 | -0.0147654 | 0.0137212 | 0.942724 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 10 | max_up | 20 | 0.0354474 | 1818 | 0.031335 | 0.00411245 | -0.00869196 | 0.0169169 | 0.52902 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 10 | terminal_return | 20 | 0.0124562 | 1818 | 0.00426626 | 0.00818993 | -0.0090194 | 0.0253993 | 0.350941 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 20 | max_down | 19 | -0.0469471 | 1809 | -0.0449675 | -0.00197952 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 20 | max_up | 19 | 0.0562624 | 1809 | 0.0474411 | 0.00882132 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 中证500 | 20 | terminal_return | 19 | 0.0139443 | 1809 | 0.00846137 | 0.00548292 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 5 | max_down | 20 | -0.0384079 | 1823 | -0.026037 | -0.0123709 | -0.0296624 | 0.0049207 | 0.160844 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 5 | max_up | 20 | 0.0378035 | 1823 | 0.0277027 | 0.0101008 | -0.0184737 | 0.0386753 | 0.488408 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 5 | terminal_return | 20 | 0.00911683 | 1823 | 0.00687045 | 0.00224638 | -0.034794 | 0.0392868 | 0.90538 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 10 | max_down | 20 | -0.0630356 | 1818 | -0.0379303 | -0.0251053 | -0.0618561 | 0.0116455 | 0.180597 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 10 | max_up | 20 | 0.062893 | 1818 | 0.0429546 | 0.0199384 | -0.0256879 | 0.0655648 | 0.391717 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 10 | terminal_return | 20 | 0.0126321 | 1818 | 0.0135427 | -0.000910582 | -0.0498022 | 0.047981 | 0.97088 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 20 | max_down | 19 | -0.0711927 | 1809 | -0.0551944 | -0.0159983 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 20 | max_up | 19 | 0.0944639 | 1809 | 0.0663062 | 0.0281577 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 微盘股 | 20 | terminal_return | 19 | 0.0503517 | 1809 | 0.0262803 | 0.0240714 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 5 | max_down | 20 | -0.0142707 | 1823 | -0.0163554 | 0.00208473 | -0.00332122 | 0.00749069 | 0.449741 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 5 | max_up | 20 | 0.0168859 | 1823 | 0.0156186 | 0.00126725 | -0.00497494 | 0.00750944 | 0.690699 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 5 | terminal_return | 20 | 0.00207232 | 1823 | 0.00152198 | 0.000550344 | -0.009561 | 0.0106617 | 0.915043 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 10 | max_down | 20 | -0.0210047 | 1818 | -0.0234936 | 0.00248892 | -0.00624462 | 0.0112225 | 0.576456 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 10 | max_up | 20 | 0.023128 | 1818 | 0.0233769 | -0.000248955 | -0.00924719 | 0.00874928 | 0.956754 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 10 | terminal_return | 20 | 0.0068759 | 1818 | 0.00287944 | 0.00399647 | -0.00789355 | 0.0158865 | 0.510028 | 0.99333 | 0.996395 | true |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 20 | max_down | 19 | -0.0326373 | 1809 | -0.0335267 | 0.000889474 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 20 | max_up | 19 | 0.0407921 | 1809 | 0.0349843 | 0.00580776 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | bottom | onset | 上证指数 | 20 | terminal_return | 19 | 0.0109521 | 1809 | 0.00551319 | 0.00543896 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 5 | max_down | 2 | -0.0325431 | 1841 | -0.0198404 | -0.0127028 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 5 | max_up | 2 | 0.0131139 | 1841 | 0.018637 | -0.00552304 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 5 | terminal_return | 2 | -0.0268461 | 1841 | 0.00192714 | -0.0287732 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 10 | max_down | 2 | -0.0880031 | 1836 | -0.028735 | -0.0592681 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 10 | max_up | 2 | 0.0131139 | 1836 | 0.0278239 | -0.01471 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 10 | terminal_return | 2 | -0.0264658 | 1836 | 0.0036754 | -0.0301412 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 20 | max_down | 2 | -0.0880031 | 1826 | -0.0412182 | -0.0467848 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 20 | max_up | 2 | 0.0414617 | 1826 | 0.0417881 | -0.000326383 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 全A | 20 | terminal_return | 2 | 0.022166 | 1826 | 0.00699527 | 0.0151707 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 5 | max_down | 2 | -0.0390692 | 1841 | -0.025261 | -0.0138083 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 5 | max_up | 2 | 0.0196419 | 1841 | 0.0228804 | -0.00323849 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 5 | terminal_return | 2 | -0.0228329 | 1841 | 0.00258872 | -0.0254216 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 10 | max_down | 2 | -0.105522 | 1836 | -0.0370502 | -0.0684714 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 10 | max_up | 2 | 0.0196419 | 1836 | 0.0344928 | -0.0148509 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 10 | terminal_return | 2 | -0.0185593 | 1836 | 0.00488794 | -0.0234472 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 20 | max_down | 2 | -0.105522 | 1826 | -0.0540247 | -0.0514969 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 20 | max_up | 2 | 0.0640791 | 1826 | 0.0514683 | 0.0126108 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 国证2000 | 20 | terminal_return | 2 | 0.0506107 | 1826 | 0.00928878 | 0.041322 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 5 | max_down | 2 | -0.037261 | 1841 | -0.0248966 | -0.0123645 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 5 | max_up | 2 | 0.0181297 | 1841 | 0.0226821 | -0.00455247 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 5 | terminal_return | 2 | -0.0249634 | 1841 | 0.00217388 | -0.0271373 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 10 | max_down | 2 | -0.104848 | 1836 | -0.0364816 | -0.0683668 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 10 | max_up | 2 | 0.0181297 | 1836 | 0.0339011 | -0.0157714 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 10 | terminal_return | 2 | -0.0217913 | 1836 | 0.00407687 | -0.0258682 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 20 | max_down | 2 | -0.104848 | 1826 | -0.0529602 | -0.0518882 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 20 | max_up | 2 | 0.0622784 | 1826 | 0.0505872 | 0.0116911 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证1000 | 20 | terminal_return | 2 | 0.047649 | 1826 | 0.00772994 | 0.0399191 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 5 | max_down | 2 | -0.0342806 | 1841 | -0.0181028 | -0.0161777 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 5 | max_up | 2 | 0.012667 | 1841 | 0.018297 | -0.00562997 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 5 | terminal_return | 2 | -0.0304833 | 1841 | 0.00160127 | -0.0320846 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 10 | max_down | 2 | -0.0810726 | 1836 | -0.0259661 | -0.0551064 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 10 | max_up | 2 | 0.012667 | 1836 | 0.0274172 | -0.0147501 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 10 | terminal_return | 2 | -0.03661 | 1836 | 0.00305332 | -0.0396633 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 20 | max_down | 2 | -0.0810726 | 1826 | -0.0367122 | -0.0443603 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 20 | max_up | 2 | 0.0207702 | 1826 | 0.0413691 | -0.020599 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 沪深300 | 20 | terminal_return | 2 | -0.00443702 | 1826 | 0.00581108 | -0.0102481 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 5 | max_down | 2 | -0.032862 | 1841 | -0.0216388 | -0.0112232 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 5 | max_up | 2 | 0.0140196 | 1841 | 0.0210375 | -0.00701785 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 5 | terminal_return | 2 | -0.028266 | 1841 | 0.00231104 | -0.0305771 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 10 | max_down | 2 | -0.098646 | 1836 | -0.0312783 | -0.0673677 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 10 | max_up | 2 | 0.0140196 | 1836 | 0.0313987 | -0.017379 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 10 | terminal_return | 2 | -0.0242698 | 1836 | 0.00438656 | -0.0286564 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 20 | max_down | 2 | -0.098646 | 1826 | -0.0449293 | -0.0537166 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 20 | max_up | 2 | 0.0449388 | 1826 | 0.0475357 | -0.0025969 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 中证500 | 20 | terminal_return | 2 | 0.0285741 | 1826 | 0.00849639 | 0.0200777 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 5 | max_down | 2 | -0.0471795 | 1841 | -0.0261485 | -0.0210311 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 5 | max_up | 2 | 0.0208918 | 1841 | 0.0278198 | -0.00692803 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 5 | terminal_return | 2 | -0.0263092 | 1841 | 0.0069309 | -0.0332401 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 10 | max_down | 2 | -0.122419 | 1836 | -0.0381117 | -0.0843071 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 10 | max_up | 2 | 0.0351471 | 1836 | 0.0431803 | -0.00803317 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 10 | terminal_return | 2 | -0.0239214 | 1836 | 0.0135736 | -0.037495 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 20 | max_down | 2 | -0.122419 | 1826 | -0.0552872 | -0.0671316 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 20 | max_up | 2 | 0.0825296 | 1826 | 0.0665814 | 0.0159481 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 微盘股 | 20 | terminal_return | 2 | 0.061414 | 1826 | 0.0264923 | 0.0349217 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 5 | max_down | 2 | -0.0288499 | 1841 | -0.0163192 | -0.0125307 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 5 | max_up | 2 | 0.0120478 | 1841 | 0.0156363 | -0.00358848 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 5 | terminal_return | 2 | -0.025405 | 1841 | 0.00155721 | -0.0269622 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 10 | max_down | 2 | -0.0787577 | 1836 | -0.0234063 | -0.0553514 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 10 | max_up | 2 | 0.0120478 | 1836 | 0.0233866 | -0.0113388 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 10 | terminal_return | 2 | -0.0328653 | 1836 | 0.00296191 | -0.0358272 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 20 | max_down | 2 | -0.0787577 | 1826 | -0.0334679 | -0.0452897 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 20 | max_up | 2 | 0.028256 | 1826 | 0.0350521 | -0.00679613 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | capped_confirmation | 上证指数 | 20 | terminal_return | 2 | 0.00358287 | 1826 | 0.00557189 | -0.00198902 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 5 | max_down | 2 | -0.0342468 | 1841 | -0.0198385 | -0.0144083 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 5 | max_up | 2 | 0.00543264 | 1841 | 0.0186453 | -0.0132127 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 5 | terminal_return | 2 | -0.00508343 | 1841 | 0.00190349 | -0.00698693 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 10 | max_down | 2 | -0.0991313 | 1836 | -0.0287228 | -0.0704084 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 10 | max_up | 2 | 0.00543264 | 1836 | 0.0278323 | -0.0223997 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 10 | terminal_return | 2 | -0.0470888 | 1836 | 0.00369786 | -0.0507867 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 20 | max_down | 2 | -0.0991313 | 1826 | -0.041206 | -0.0579252 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 20 | max_up | 2 | 0.0219668 | 1826 | 0.0418094 | -0.0198426 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 全A | 20 | terminal_return | 2 | 0.0170773 | 1826 | 0.00700084 | 0.0100765 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 5 | max_down | 2 | -0.0395751 | 1841 | -0.0252604 | -0.0143147 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 5 | max_up | 2 | 0.00801278 | 1841 | 0.022893 | -0.0148803 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 5 | terminal_return | 2 | 0.000255029 | 1841 | 0.00256364 | -0.00230861 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 10 | max_down | 2 | -0.117336 | 1836 | -0.0370373 | -0.0802986 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 10 | max_up | 2 | 0.00801278 | 1836 | 0.0345055 | -0.0264927 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 10 | terminal_return | 2 | -0.0460921 | 1836 | 0.00491793 | -0.05101 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 20 | max_down | 2 | -0.117336 | 1826 | -0.0540118 | -0.0633242 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 20 | max_up | 2 | 0.0407893 | 1826 | 0.0514938 | -0.0107046 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 国证2000 | 20 | terminal_return | 2 | 0.040663 | 1826 | 0.00929967 | 0.0313634 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 5 | max_down | 2 | -0.0376068 | 1841 | -0.0248962 | -0.0127107 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 5 | max_up | 2 | 0.00825543 | 1841 | 0.0226929 | -0.0144374 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 5 | terminal_return | 2 | -0.000286331 | 1841 | 0.00214707 | -0.0024334 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 10 | max_down | 2 | -0.116832 | 1836 | -0.0364686 | -0.0803635 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 10 | max_up | 2 | 0.00825543 | 1836 | 0.0339119 | -0.0256564 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 10 | terminal_return | 2 | -0.0484769 | 1836 | 0.00410594 | -0.0525828 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 20 | max_down | 2 | -0.116832 | 1826 | -0.0529471 | -0.063885 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 20 | max_up | 2 | 0.0367483 | 1826 | 0.0506152 | -0.0138669 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证1000 | 20 | terminal_return | 2 | 0.0360501 | 1826 | 0.00774265 | 0.0283075 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 5 | max_down | 2 | -0.0321122 | 1841 | -0.0181052 | -0.014007 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 5 | max_up | 2 | 0.00347723 | 1841 | 0.018307 | -0.0148297 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 5 | terminal_return | 2 | -0.0104042 | 1841 | 0.00157946 | -0.0119837 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 10 | max_down | 2 | -0.0913513 | 1836 | -0.0259549 | -0.0653964 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 10 | max_up | 2 | 0.00347723 | 1836 | 0.0274272 | -0.0239499 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 10 | terminal_return | 2 | -0.0527124 | 1836 | 0.00307087 | -0.0557833 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 20 | max_down | 2 | -0.0913513 | 1826 | -0.036701 | -0.0546503 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 20 | max_up | 2 | 0.00958053 | 1826 | 0.0413814 | -0.0318008 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 沪深300 | 20 | terminal_return | 2 | -0.00402473 | 1826 | 0.00581063 | -0.00983536 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 5 | max_down | 2 | -0.0329735 | 1841 | -0.0216387 | -0.0113348 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 5 | max_up | 2 | 0.00738758 | 1841 | 0.0210447 | -0.0136571 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 5 | terminal_return | 2 | -0.00353806 | 1841 | 0.00228418 | -0.00582224 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 10 | max_down | 2 | -0.110798 | 1836 | -0.031265 | -0.0795333 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 10 | max_up | 2 | 0.00738758 | 1836 | 0.0314059 | -0.0240183 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 10 | terminal_return | 2 | -0.0487268 | 1836 | 0.00441321 | -0.05314 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 20 | max_down | 2 | -0.110798 | 1826 | -0.044916 | -0.0658823 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 20 | max_up | 2 | 0.0216318 | 1826 | 0.0475612 | -0.0259294 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 中证500 | 20 | terminal_return | 2 | 0.0177909 | 1826 | 0.0085082 | 0.00928269 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 5 | max_down | 2 | -0.0458437 | 1841 | -0.0261499 | -0.0196938 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 5 | max_up | 2 | 0.00918629 | 1841 | 0.0278325 | -0.0186462 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 5 | terminal_return | 2 | -0.0035136 | 1841 | 0.00690613 | -0.0104197 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 10 | max_down | 2 | -0.133655 | 1836 | -0.0380995 | -0.0955557 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 10 | max_up | 2 | 0.0197278 | 1836 | 0.0431971 | -0.0234692 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 10 | terminal_return | 2 | -0.0486704 | 1836 | 0.0136006 | -0.0622709 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 20 | max_down | 2 | -0.133655 | 1826 | -0.0552749 | -0.0783803 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 20 | max_up | 2 | 0.0713322 | 1826 | 0.0665937 | 0.00473852 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 微盘股 | 20 | terminal_return | 2 | 0.052477 | 1826 | 0.0265021 | 0.0259749 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 5 | max_down | 2 | -0.0277138 | 1841 | -0.0163204 | -0.0113934 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 5 | max_up | 2 | 0.00272256 | 1841 | 0.0156464 | -0.0129238 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 5 | terminal_return | 2 | -0.00613594 | 1841 | 0.00153628 | -0.00767221 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 10 | max_down | 2 | -0.0880532 | 1836 | -0.0233962 | -0.064657 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 10 | max_up | 2 | 0.00272256 | 1836 | 0.0233967 | -0.0206742 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 10 | terminal_return | 2 | -0.0486168 | 1836 | 0.00297907 | -0.0515958 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 20 | max_down | 2 | -0.0880532 | 1826 | -0.0334578 | -0.0545954 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 20 | max_up | 2 | 0.0180413 | 1826 | 0.0350633 | -0.0170219 |  |  |  |  |  | false |
| ma20_episode_ml_l2_logistic | top | onset | 上证指数 | 20 | terminal_return | 2 | 0.00238964 | 1826 | 0.0055732 | -0.00318356 |  |  |  |  |  | false |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `ma20_episode_ml_l2_logistic/bottom/all_a_ma20_episode_match_walk_forward_v1/capped_confirmation`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 21/63 项检验未达到样本门槛，仅可读取描述统计。 42 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
- `ma20_episode_ml_l2_logistic/bottom/all_a_ma20_episode_match_walk_forward_v1/onset`：数据可用性——20日：事件日缺失 0、窗口不完整 7（涉及 7 个指数）。 21/63 项检验未达到样本门槛，仅可读取描述统计。 42 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
- `ma20_episode_ml_l2_logistic/top/all_a_ma20_episode_match_walk_forward_v1/capped_confirmation`：63/63 项检验未达到样本门槛，仅可读取描述统计。
- `ma20_episode_ml_l2_logistic/top/all_a_ma20_episode_match_walk_forward_v1/onset`：63/63 项检验未达到样本门槛，仅可读取描述统计。
