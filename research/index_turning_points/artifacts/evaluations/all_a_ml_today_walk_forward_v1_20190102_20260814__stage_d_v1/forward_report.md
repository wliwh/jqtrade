# 信号后 OHLC 结果评测

- 评测版本：`all_a_ml_today_walk_forward_v1_stage_d_v1`
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
| 上证指数 | capped_confirmation | 5 | 157 | 157 | 157 |
| 上证指数 | capped_confirmation | 10 | 157 | 157 | 157 |
| 上证指数 | capped_confirmation | 20 | 157 | 157 | 157 |
| 上证指数 | onset | 5 | 157 | 157 | 157 |
| 上证指数 | onset | 10 | 157 | 157 | 157 |
| 上证指数 | onset | 20 | 157 | 157 | 157 |
| 中证1000 | capped_confirmation | 5 | 157 | 157 | 157 |
| 中证1000 | capped_confirmation | 10 | 157 | 157 | 157 |
| 中证1000 | capped_confirmation | 20 | 157 | 157 | 157 |
| 中证1000 | onset | 5 | 157 | 157 | 157 |
| 中证1000 | onset | 10 | 157 | 157 | 157 |
| 中证1000 | onset | 20 | 157 | 157 | 157 |
| 中证500 | capped_confirmation | 5 | 157 | 157 | 157 |
| 中证500 | capped_confirmation | 10 | 157 | 157 | 157 |
| 中证500 | capped_confirmation | 20 | 157 | 157 | 157 |
| 中证500 | onset | 5 | 157 | 157 | 157 |
| 中证500 | onset | 10 | 157 | 157 | 157 |
| 中证500 | onset | 20 | 157 | 157 | 157 |
| 全A | capped_confirmation | 5 | 157 | 157 | 157 |
| 全A | capped_confirmation | 10 | 157 | 157 | 157 |
| 全A | capped_confirmation | 20 | 157 | 157 | 157 |
| 全A | onset | 5 | 157 | 157 | 157 |
| 全A | onset | 10 | 157 | 157 | 157 |
| 全A | onset | 20 | 157 | 157 | 157 |
| 国证2000 | capped_confirmation | 5 | 157 | 157 | 157 |
| 国证2000 | capped_confirmation | 10 | 157 | 157 | 157 |
| 国证2000 | capped_confirmation | 20 | 157 | 157 | 157 |
| 国证2000 | onset | 5 | 157 | 157 | 157 |
| 国证2000 | onset | 10 | 157 | 157 | 157 |
| 国证2000 | onset | 20 | 157 | 157 | 157 |
| 微盘股 | capped_confirmation | 5 | 157 | 157 | 157 |
| 微盘股 | capped_confirmation | 10 | 157 | 157 | 157 |
| 微盘股 | capped_confirmation | 20 | 157 | 157 | 157 |
| 微盘股 | onset | 5 | 157 | 157 | 157 |
| 微盘股 | onset | 10 | 157 | 157 | 157 |
| 微盘股 | onset | 20 | 157 | 157 | 157 |
| 沪深300 | capped_confirmation | 5 | 157 | 157 | 157 |
| 沪深300 | capped_confirmation | 10 | 157 | 157 | 157 |
| 沪深300 | capped_confirmation | 20 | 157 | 157 | 157 |
| 沪深300 | onset | 5 | 157 | 157 | 157 |
| 沪深300 | onset | 10 | 157 | 157 | 157 |
| 沪深300 | onset | 20 | 157 | 157 | 157 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 5 | max_down | 27 | -0.0278173 | 1816 | -0.0197357 | -0.00808157 | -0.0186714 | 0.00250828 | 0.134716 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 5 | max_up | 27 | 0.0262902 | 1816 | 0.0185171 | 0.00777307 | -0.00199858 | 0.0175447 | 0.118967 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 5 | terminal_return | 27 | 0.00656687 | 1816 | 0.00182646 | 0.00474041 | -0.00971262 | 0.0191934 | 0.520318 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 10 | max_down | 27 | -0.0384561 | 1811 | -0.0286555 | -0.00980062 | -0.0272729 | 0.00767166 | 0.271589 | 0.49405 | 0.943604 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 10 | max_up | 27 | 0.0399158 | 1811 | 0.0276274 | 0.0122884 | 0.00012483 | 0.0244519 | 0.0476906 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 10 | terminal_return | 27 | 0.0107929 | 1811 | 0.003536 | 0.00725695 | -0.0147643 | 0.0292782 | 0.518341 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 20 | max_down | 27 | -0.0479619 | 1801 | -0.0411691 | -0.00679283 | -0.0263849 | 0.0127992 | 0.496784 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 20 | max_up | 27 | 0.0552461 | 1801 | 0.041586 | 0.0136601 | -0.00318948 | 0.0305097 | 0.112063 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 全A | 20 | terminal_return | 27 | 0.0192976 | 1801 | 0.00682769 | 0.0124699 | -0.0128684 | 0.0378082 | 0.334751 | 0.540751 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 5 | max_down | 27 | -0.0376761 | 1816 | -0.0250916 | -0.0125845 | -0.0274463 | 0.00227731 | 0.0969819 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 5 | max_up | 27 | 0.032953 | 1816 | 0.0227271 | 0.0102259 | -0.00175181 | 0.0222036 | 0.0942608 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 27 | 0.00643169 | 1816 | 0.00250359 | 0.0039281 | -0.0154297 | 0.0232859 | 0.690833 | 0.699248 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 10 | max_down | 27 | -0.054313 | 1811 | -0.0368685 | -0.0174445 | -0.0432709 | 0.00838186 | 0.18554 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 10 | max_up | 27 | 0.0534692 | 1811 | 0.0341935 | 0.0192757 | 0.0036742 | 0.0348773 | 0.015453 | 0.294006 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 27 | 0.0175444 | 1811 | 0.00467335 | 0.0128711 | -0.0191822 | 0.0449244 | 0.431257 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 20 | max_down | 27 | -0.065489 | 1801 | -0.0539101 | -0.0115789 | -0.0429895 | 0.0198316 | 0.469976 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 20 | max_up | 27 | 0.0784471 | 1801 | 0.0510779 | 0.0273692 | 0.00589787 | 0.0488406 | 0.0124761 | 0.294006 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 27 | 0.037526 | 1801 | 0.00891134 | 0.0286147 | -0.00520242 | 0.0624318 | 0.0972219 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 5 | max_down | 27 | -0.036958 | 1816 | -0.0247309 | -0.0122272 | -0.02645 | 0.00199566 | 0.0919914 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 5 | max_up | 27 | 0.0311294 | 1816 | 0.0225515 | 0.00857789 | -0.00297263 | 0.0201284 | 0.14551 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 27 | 0.00587263 | 1816 | 0.002089 | 0.00378363 | -0.014706 | 0.0222732 | 0.688357 | 0.699248 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 10 | max_down | 27 | -0.0526952 | 1811 | -0.0363154 | -0.0163797 | -0.0410776 | 0.00831808 | 0.193641 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 10 | max_up | 27 | 0.0515924 | 1811 | 0.0336199 | 0.0179725 | 0.00289121 | 0.0330538 | 0.0195044 | 0.294006 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 27 | 0.0157109 | 1811 | 0.00387486 | 0.0118361 | -0.0185536 | 0.0422258 | 0.44524 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 20 | max_down | 27 | -0.0637997 | 1801 | -0.0528553 | -0.0109443 | -0.040578 | 0.0186894 | 0.469147 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 20 | max_up | 27 | 0.0759484 | 1801 | 0.05022 | 0.0257284 | 0.0047186 | 0.0467382 | 0.0163862 | 0.294006 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 27 | 0.032157 | 1801 | 0.00740807 | 0.0247489 | -0.0077869 | 0.0572848 | 0.135986 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 5 | max_down | 27 | -0.0228954 | 1816 | -0.0180494 | -0.00484599 | -0.0135374 | 0.00384541 | 0.274472 | 0.49405 | 0.943604 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 5 | max_up | 27 | 0.0247654 | 1816 | 0.0181946 | 0.00657081 | -0.00260509 | 0.0157467 | 0.160455 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 27 | 0.0066353 | 1816 | 0.00149109 | 0.00514421 | -0.00722756 | 0.017516 | 0.415088 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 10 | max_down | 27 | -0.0311872 | 1811 | -0.0259491 | -0.0052381 | -0.0174488 | 0.00697258 | 0.400463 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 10 | max_up | 27 | 0.0362286 | 1811 | 0.0272695 | 0.00895906 | -0.00332402 | 0.0212421 | 0.152835 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 27 | 0.0086222 | 1811 | 0.0029265 | 0.00569571 | -0.0108318 | 0.0222232 | 0.499388 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 20 | max_down | 27 | -0.0402796 | 1801 | -0.036708 | -0.00357163 | -0.0175743 | 0.010431 | 0.617121 | 0.682081 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 20 | max_up | 27 | 0.0466306 | 1801 | 0.0412674 | 0.00536323 | -0.0104522 | 0.0211787 | 0.506266 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 27 | 0.0104368 | 1801 | 0.00573035 | 0.00470647 | -0.0173114 | 0.0267243 | 0.675243 | 0.699248 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 5 | max_down | 27 | -0.0318919 | 1816 | -0.0214987 | -0.0103932 | -0.0221071 | 0.00132064 | 0.0820303 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 5 | max_up | 27 | 0.0296845 | 1816 | 0.0209012 | 0.00878336 | -0.00147168 | 0.0190384 | 0.0932059 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 27 | 0.00622969 | 1816 | 0.0022191 | 0.00401058 | -0.0115844 | 0.0196056 | 0.614222 | 0.682081 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 10 | max_down | 27 | -0.0442143 | 1811 | -0.0311598 | -0.0130545 | -0.0329656 | 0.00685667 | 0.198776 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 10 | max_up | 27 | 0.046525 | 1811 | 0.0311539 | 0.015371 | 0.00208698 | 0.0286551 | 0.0233338 | 0.294006 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 27 | 0.0133569 | 1811 | 0.00422118 | 0.00913574 | -0.0159699 | 0.0342414 | 0.475705 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 20 | max_down | 27 | -0.0530607 | 1801 | -0.0448671 | -0.00819362 | -0.0316679 | 0.0152807 | 0.493892 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 20 | max_up | 27 | 0.0675146 | 1801 | 0.0472333 | 0.0202813 | 0.000583536 | 0.0399791 | 0.0435846 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 27 | 0.0272158 | 1801 | 0.00823805 | 0.0189777 | -0.0101755 | 0.0481309 | 0.201994 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 5 | max_down | 27 | -0.0404126 | 1816 | -0.0259595 | -0.0144531 | -0.0305248 | 0.00161871 | 0.0779691 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 5 | max_up | 27 | 0.0390868 | 1816 | 0.0276447 | 0.0114421 | -0.0015739 | 0.0244581 | 0.0848891 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 27 | 0.0111865 | 1816 | 0.00683102 | 0.00435545 | -0.0176951 | 0.0264061 | 0.698652 | 0.699248 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 10 | max_down | 27 | -0.0550593 | 1811 | -0.0379522 | -0.0171071 | -0.043312 | 0.00909781 | 0.200711 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 10 | max_up | 27 | 0.0572907 | 1811 | 0.042961 | 0.0143297 | -0.00439704 | 0.0330564 | 0.133668 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 27 | 0.0206618 | 1811 | 0.0134265 | 0.00723528 | -0.0277963 | 0.0422668 | 0.685617 | 0.699248 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 20 | max_down | 27 | -0.0716738 | 1801 | -0.0551161 | -0.0165576 | -0.0483755 | 0.0152602 | 0.307747 | 0.524001 | 0.975002 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 20 | max_up | 27 | 0.090507 | 1801 | 0.0662405 | 0.0242666 | -0.00373402 | 0.0522671 | 0.08939 | 0.382811 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 27 | 0.054407 | 1801 | 0.0261126 | 0.0282944 | -0.0113285 | 0.0679173 | 0.161627 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 5 | max_down | 27 | -0.0222909 | 1816 | -0.0162442 | -0.00604669 | -0.0147442 | 0.00265082 | 0.172998 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 5 | max_up | 27 | 0.020223 | 1816 | 0.0155641 | 0.00465888 | -0.00299509 | 0.0123129 | 0.232858 | 0.444547 | 0.91323 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 27 | 0.00488761 | 1816 | 0.001478 | 0.00340961 | -0.00785361 | 0.0146728 | 0.552959 | 0.633389 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 10 | max_down | 27 | -0.0300613 | 1811 | -0.0233682 | -0.00669314 | -0.0200878 | 0.00670149 | 0.327388 | 0.540751 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 10 | max_up | 27 | 0.030192 | 1811 | 0.0232726 | 0.00691943 | -0.00294727 | 0.0167861 | 0.169277 | 0.397675 | 0.911847 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 27 | 0.00855391 | 1811 | 0.00283897 | 0.00571494 | -0.010146 | 0.0215758 | 0.480052 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 20 | max_down | 27 | -0.0362933 | 1801 | -0.0334759 | -0.00281742 | -0.0171111 | 0.0114762 | 0.699248 | 0.699248 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 20 | max_up | 27 | 0.0405094 | 1801 | 0.0349627 | 0.00554666 | -0.00717413 | 0.0182675 | 0.392761 | 0.607038 | 0.999178 | true |
| ml_today_elastic_net | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 27 | 0.0147443 | 1801 | 0.00543217 | 0.00931216 | -0.00784568 | 0.02647 | 0.287437 | 0.503015 | 0.944803 | true |
| ml_today_elastic_net | bottom | onset | 全A | 5 | max_down | 27 | -0.0294361 | 1816 | -0.0197117 | -0.00972445 | -0.0200475 | 0.000598605 | 0.0648421 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 全A | 5 | max_up | 27 | 0.0265649 | 1816 | 0.018513 | 0.00805184 | -0.00082278 | 0.0169265 | 0.075357 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 全A | 5 | terminal_return | 27 | 0.00354445 | 1816 | 0.0018714 | 0.00167305 | -0.0132652 | 0.0166113 | 0.826249 | 0.897477 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 全A | 10 | max_down | 27 | -0.0413094 | 1811 | -0.0286129 | -0.0126965 | -0.0295551 | 0.00416211 | 0.139915 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 全A | 10 | max_up | 27 | 0.037267 | 1811 | 0.0276669 | 0.00960005 | -0.00333503 | 0.0225351 | 0.145764 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 全A | 10 | terminal_return | 27 | 0.00932827 | 1811 | 0.00355783 | 0.00577044 | -0.0160836 | 0.0276244 | 0.604788 | 0.748754 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 全A | 20 | max_down | 27 | -0.0518328 | 1801 | -0.041111 | -0.0107218 | -0.0305283 | 0.00908472 | 0.28869 | 0.519642 | 0.944803 | true |
| ml_today_elastic_net | bottom | onset | 全A | 20 | max_up | 27 | 0.0517521 | 1801 | 0.0416383 | 0.0101137 | -0.00748234 | 0.0277098 | 0.259931 | 0.481637 | 0.935751 | true |
| ml_today_elastic_net | bottom | onset | 全A | 20 | terminal_return | 27 | 0.0151732 | 1801 | 0.00688952 | 0.00828373 | -0.0188681 | 0.0354356 | 0.549857 | 0.736388 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 国证2000 | 5 | max_down | 27 | -0.0431895 | 1816 | -0.0250096 | -0.0181799 | -0.0319912 | -0.00436858 | 0.00988129 | 0.266641 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 国证2000 | 5 | max_up | 27 | 0.0305901 | 1816 | 0.0227622 | 0.0078279 | -0.00275831 | 0.0184141 | 0.147252 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 国证2000 | 5 | terminal_return | 27 | 0.00166029 | 1816 | 0.00257453 | -0.000914241 | -0.0204082 | 0.0185797 | 0.92676 | 0.973099 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 国证2000 | 10 | max_down | 27 | -0.0610788 | 1811 | -0.0367676 | -0.0243112 | -0.0486759 | 5.34595e-05 | 0.0505006 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 国证2000 | 10 | max_up | 27 | 0.0458956 | 1811 | 0.0343064 | 0.0115892 | -0.00340851 | 0.026587 | 0.129885 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 国证2000 | 10 | terminal_return | 27 | 0.010507 | 1811 | 0.00477827 | 0.00572878 | -0.0248622 | 0.0363198 | 0.713582 | 0.80278 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 国证2000 | 20 | max_down | 27 | -0.0736286 | 1801 | -0.053788 | -0.0198406 | -0.0511346 | 0.0114534 | 0.213996 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 国证2000 | 20 | max_up | 27 | 0.0712323 | 1801 | 0.051186 | 0.0200462 | -0.00246406 | 0.0425565 | 0.0809063 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 国证2000 | 20 | terminal_return | 27 | 0.0282463 | 1801 | 0.00905046 | 0.0191958 | -0.0184058 | 0.0567974 | 0.317025 | 0.525594 | 0.980248 | true |
| ml_today_elastic_net | bottom | onset | 中证1000 | 5 | max_down | 27 | -0.0413612 | 1816 | -0.0246654 | -0.0166958 | -0.0298265 | -0.00356506 | 0.0126972 | 0.266641 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证1000 | 5 | max_up | 27 | 0.0303579 | 1816 | 0.022563 | 0.00779487 | -0.00256423 | 0.018154 | 0.140258 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证1000 | 5 | terminal_return | 27 | 0.00154963 | 1816 | 0.00215327 | -0.000603646 | -0.0193439 | 0.0181366 | 0.94966 | 0.980796 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 中证1000 | 10 | max_down | 27 | -0.0586097 | 1811 | -0.0362272 | -0.0223825 | -0.0457964 | 0.00103141 | 0.0609774 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证1000 | 10 | max_up | 27 | 0.0444006 | 1811 | 0.0337272 | 0.0106735 | -0.00442525 | 0.0257722 | 0.165885 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证1000 | 10 | terminal_return | 27 | 0.0099169 | 1811 | 0.00396124 | 0.00595566 | -0.0232703 | 0.0351816 | 0.689592 | 0.80278 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 中证1000 | 20 | max_down | 27 | -0.0713433 | 1801 | -0.0527423 | -0.0186011 | -0.0484496 | 0.0112475 | 0.221921 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证1000 | 20 | max_up | 27 | 0.0696373 | 1801 | 0.0503146 | 0.0193226 | -0.00299038 | 0.0416356 | 0.0896358 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证1000 | 20 | terminal_return | 27 | 0.0246358 | 1801 | 0.00752082 | 0.017115 | -0.0195566 | 0.0537865 | 0.360322 | 0.567508 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 沪深300 | 5 | max_down | 27 | -0.023623 | 1816 | -0.0180386 | -0.00558443 | -0.0140948 | 0.00292594 | 0.198396 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 沪深300 | 5 | max_up | 27 | 0.0260303 | 1816 | 0.0181758 | 0.00785455 | -0.000180667 | 0.0158898 | 0.0553739 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 沪深300 | 5 | terminal_return | 27 | 0.00495056 | 1816 | 0.00151614 | 0.00343442 | -0.00962117 | 0.01649 | 0.606134 | 0.748754 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 沪深300 | 10 | max_down | 27 | -0.0318738 | 1811 | -0.0259389 | -0.00593488 | -0.0180308 | 0.00616104 | 0.336212 | 0.543112 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 沪深300 | 10 | max_up | 27 | 0.0363161 | 1811 | 0.0272682 | 0.00904793 | -0.00395199 | 0.0220478 | 0.172518 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 沪深300 | 10 | terminal_return | 27 | 0.0106432 | 1811 | 0.00289636 | 0.00774688 | -0.00940297 | 0.0248967 | 0.37596 | 0.575504 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 沪深300 | 20 | max_down | 27 | -0.0415426 | 1801 | -0.0366891 | -0.00485353 | -0.0184331 | 0.00872605 | 0.483596 | 0.677034 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 沪深300 | 20 | max_up | 27 | 0.0460544 | 1801 | 0.041276 | 0.00477839 | -0.011827 | 0.0213838 | 0.572747 | 0.736388 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 沪深300 | 20 | terminal_return | 27 | 0.00994264 | 1801 | 0.00573776 | 0.00420488 | -0.0177771 | 0.0261869 | 0.707718 | 0.80278 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 中证500 | 5 | max_down | 27 | -0.0354245 | 1816 | -0.0214462 | -0.0139783 | -0.0254833 | -0.00247331 | 0.017249 | 0.271672 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证500 | 5 | max_up | 27 | 0.027271 | 1816 | 0.0209371 | 0.00633392 | -0.00360032 | 0.0162682 | 0.211422 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证500 | 5 | terminal_return | 27 | 0.00193008 | 1816 | 0.00228303 | -0.000352951 | -0.0168415 | 0.0161356 | 0.966534 | 0.982124 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 中证500 | 10 | max_down | 27 | -0.0488469 | 1811 | -0.0310908 | -0.0177561 | -0.0367541 | 0.00124188 | 0.0669698 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证500 | 10 | max_up | 27 | 0.039677 | 1811 | 0.031256 | 0.00842095 | -0.00536116 | 0.0222031 | 0.231084 | 0.44116 | 0.91323 | true |
| ml_today_elastic_net | bottom | onset | 中证500 | 10 | terminal_return | 27 | 0.0089516 | 1811 | 0.00428686 | 0.00466474 | -0.0195759 | 0.0289054 | 0.706046 | 0.80278 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 中证500 | 20 | max_down | 27 | -0.0596405 | 1801 | -0.0447684 | -0.014872 | -0.0383482 | 0.00860417 | 0.214366 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证500 | 20 | max_up | 27 | 0.0611559 | 1801 | 0.0473286 | 0.0138273 | -0.00692769 | 0.0345823 | 0.191626 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 中证500 | 20 | terminal_return | 27 | 0.0199984 | 1801 | 0.00834625 | 0.0116522 | -0.020547 | 0.0438514 | 0.47815 | 0.677034 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 微盘股 | 5 | max_down | 27 | -0.0480215 | 1816 | -0.0258464 | -0.0221751 | -0.0366668 | -0.00768336 | 0.00270714 | 0.17055 | 0.820374 | true |
| ml_today_elastic_net | bottom | onset | 微盘股 | 5 | max_up | 27 | 0.0347573 | 1816 | 0.027709 | 0.00704831 | -0.00394847 | 0.0180451 | 0.209026 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 微盘股 | 5 | terminal_return | 27 | 0.00441476 | 1816 | 0.0069317 | -0.00251695 | -0.0237685 | 0.0187346 | 0.816433 | 0.897477 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 微盘股 | 10 | max_down | 27 | -0.0627665 | 1811 | -0.0378372 | -0.0249293 | -0.0473385 | -0.00252011 | 0.0292264 | 0.368253 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 微盘股 | 10 | max_up | 27 | 0.0511629 | 1811 | 0.0430524 | 0.00811052 | -0.00767578 | 0.0238968 | 0.31394 | 0.525594 | 0.976704 | true |
| ml_today_elastic_net | bottom | onset | 微盘股 | 10 | terminal_return | 27 | 0.0135046 | 1811 | 0.0135332 | -2.86109e-05 | -0.0311842 | 0.031127 | 0.998564 | 0.998564 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 微盘股 | 20 | max_down | 27 | -0.0798887 | 1801 | -0.054993 | -0.0248957 | -0.053377 | 0.00358553 | 0.0866658 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 微盘股 | 20 | max_up | 27 | 0.0827361 | 1801 | 0.066357 | 0.0163791 | -0.0103644 | 0.0431225 | 0.229982 | 0.44116 | 0.91323 | true |
| ml_today_elastic_net | bottom | onset | 微盘股 | 20 | terminal_return | 27 | 0.0439877 | 1801 | 0.0262688 | 0.0177189 | -0.0221466 | 0.0575845 | 0.383669 | 0.575504 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 上证指数 | 5 | max_down | 27 | -0.0235646 | 1816 | -0.0162253 | -0.00733937 | -0.0159065 | 0.00122781 | 0.0931322 | 0.391155 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 上证指数 | 5 | max_up | 27 | 0.0198733 | 1816 | 0.0155693 | 0.00430397 | -0.00253314 | 0.0111411 | 0.217269 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 上证指数 | 5 | terminal_return | 27 | 0.00247464 | 1816 | 0.00151388 | 0.000960761 | -0.0105438 | 0.0124653 | 0.869981 | 0.928963 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 上证指数 | 10 | max_down | 27 | -0.0320852 | 1811 | -0.023338 | -0.00874717 | -0.0218015 | 0.00430713 | 0.189076 | 0.44116 | 0.911847 | true |
| ml_today_elastic_net | bottom | onset | 上证指数 | 10 | max_up | 27 | 0.0286662 | 1811 | 0.0232953 | 0.00537085 | -0.00501119 | 0.0157529 | 0.310607 | 0.525594 | 0.975002 | true |
| ml_today_elastic_net | bottom | onset | 上证指数 | 10 | terminal_return | 27 | 0.00819052 | 1811 | 0.00284439 | 0.00534613 | -0.0106866 | 0.0213789 | 0.513394 | 0.703126 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 上证指数 | 20 | max_down | 27 | -0.0392691 | 1801 | -0.0334313 | -0.00583786 | -0.0202064 | 0.00853066 | 0.425835 | 0.623898 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 上证指数 | 20 | max_up | 27 | 0.0381095 | 1801 | 0.0349987 | 0.00311075 | -0.0100001 | 0.0162216 | 0.641903 | 0.77769 | 0.999178 | true |
| ml_today_elastic_net | bottom | onset | 上证指数 | 20 | terminal_return | 27 | 0.0107876 | 1801 | 0.00549149 | 0.00529611 | -0.0127573 | 0.0233495 | 0.565304 | 0.736388 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 全A | 5 | max_down | 37 | -0.0176585 | 1806 | -0.0198991 | 0.0022406 | -0.00433386 | 0.00881506 | 0.504149 | 0.72185 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 全A | 5 | max_up | 37 | 0.0286299 | 1806 | 0.0184261 | 0.0102038 | -0.00406694 | 0.0244745 | 0.161086 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 全A | 5 | terminal_return | 37 | 0.0105819 | 1806 | 0.00171796 | 0.00886396 | -0.00244525 | 0.0201732 | 0.124486 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 全A | 10 | max_down | 37 | -0.0282449 | 1801 | -0.0288108 | 0.000565947 | -0.0133662 | 0.0144981 | 0.936541 | 0.998204 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 全A | 10 | max_up | 37 | 0.0371098 | 1801 | 0.0276168 | 0.00949299 | -0.00528318 | 0.0242692 | 0.207955 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 全A | 10 | terminal_return | 37 | 0.00536823 | 1801 | 0.00360715 | 0.00176108 | -0.01571 | 0.0192321 | 0.843384 | 0.973732 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 全A | 20 | max_down | 37 | -0.0338861 | 1791 | -0.0414219 | 0.00753582 | -0.00889415 | 0.0239658 | 0.368663 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 全A | 20 | max_up | 37 | 0.0506373 | 1791 | 0.0416049 | 0.00903237 | -0.0101699 | 0.0282347 | 0.356557 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 全A | 20 | terminal_return | 37 | 0.0152948 | 1791 | 0.00684075 | 0.00845409 | -0.0118846 | 0.0287928 | 0.415242 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 5 | max_down | 37 | -0.022651 | 1806 | -0.0253297 | 0.00267876 | -0.00510837 | 0.0104659 | 0.50016 | 0.72185 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 5 | max_up | 37 | 0.0340617 | 1806 | 0.0226477 | 0.011414 | -0.00545979 | 0.0282878 | 0.184903 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 5 | terminal_return | 37 | 0.014197 | 1806 | 0.00232275 | 0.0118743 | -0.0016139 | 0.0253624 | 0.084441 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 10 | max_down | 37 | -0.0370273 | 1801 | -0.0371267 | 9.94052e-05 | -0.018416 | 0.0186148 | 0.991604 | 0.998204 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 10 | max_up | 37 | 0.0459664 | 1801 | 0.0342406 | 0.0117257 | -0.00580377 | 0.0292553 | 0.189833 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 10 | terminal_return | 37 | 0.00733584 | 1801 | 0.00481161 | 0.00252423 | -0.0208586 | 0.0259071 | 0.83243 | 0.973732 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 20 | max_down | 37 | -0.045292 | 1791 | -0.0542627 | 0.00897066 | -0.0119817 | 0.029923 | 0.401376 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 20 | max_up | 37 | 0.0653213 | 1791 | 0.0511962 | 0.0141251 | -0.0102547 | 0.0385048 | 0.256133 | 0.646543 | 0.932596 | true |
| ml_today_elastic_net | top | capped_confirmation | 国证2000 | 20 | terminal_return | 37 | 0.0213549 | 1791 | 0.00908565 | 0.0122693 | -0.0144009 | 0.0389395 | 0.367229 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 5 | max_down | 37 | -0.0218992 | 1806 | -0.0249717 | 0.00307247 | -0.00454629 | 0.0106912 | 0.429281 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 5 | max_up | 37 | 0.033171 | 1806 | 0.0224622 | 0.0107088 | -0.00582429 | 0.0272419 | 0.204252 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 5 | terminal_return | 37 | 0.0131598 | 1806 | 0.00191875 | 0.0112411 | -0.00184944 | 0.0243316 | 0.0923578 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 10 | max_down | 37 | -0.0359743 | 1801 | -0.036568 | 0.000593729 | -0.0171801 | 0.0183676 | 0.947797 | 0.998204 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 10 | max_up | 37 | 0.0444976 | 1801 | 0.0336659 | 0.0108317 | -0.00645914 | 0.0281226 | 0.219512 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 10 | terminal_return | 37 | 0.00645865 | 1801 | 0.00399922 | 0.00245944 | -0.019953 | 0.0248719 | 0.829704 | 0.973732 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 20 | max_down | 37 | -0.0447995 | 1791 | -0.0531868 | 0.00838723 | -0.0121196 | 0.028894 | 0.422765 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 20 | max_up | 37 | 0.0638723 | 1791 | 0.0503258 | 0.0135465 | -0.010618 | 0.0377109 | 0.27187 | 0.646543 | 0.943604 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证1000 | 20 | terminal_return | 37 | 0.020289 | 1791 | 0.00751506 | 0.0127739 | -0.0132996 | 0.0388474 | 0.336934 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 5 | max_down | 37 | -0.0157639 | 1806 | -0.0181687 | 0.00240477 | -0.00371915 | 0.00852868 | 0.4415 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 5 | max_up | 37 | 0.0269535 | 1806 | 0.0181134 | 0.00884015 | -0.00402541 | 0.0217057 | 0.178061 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 5 | terminal_return | 37 | 0.00774796 | 1806 | 0.00143981 | 0.00630815 | -0.00408703 | 0.0167033 | 0.234285 | 0.646543 | 0.91323 | true |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 10 | max_down | 37 | -0.0250899 | 1801 | -0.0260453 | 0.000955389 | -0.0106543 | 0.0125651 | 0.871863 | 0.980846 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 10 | max_up | 37 | 0.0348682 | 1801 | 0.0272477 | 0.00762053 | -0.00583501 | 0.0210761 | 0.26698 | 0.646543 | 0.943604 | true |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 10 | terminal_return | 37 | 0.0029342 | 1801 | 0.00301173 | -7.75225e-05 | -0.0144963 | 0.0143413 | 0.991592 | 0.998204 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 20 | max_down | 37 | -0.0310589 | 1791 | -0.0368786 | 0.00581966 | -0.00891433 | 0.0205536 | 0.438834 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 20 | max_up | 37 | 0.0457871 | 1791 | 0.0412548 | 0.00453224 | -0.0114589 | 0.0205234 | 0.578548 | 0.775501 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 沪深300 | 20 | terminal_return | 37 | 0.00855606 | 1791 | 0.00574293 | 0.00281313 | -0.0140628 | 0.019689 | 0.743878 | 0.918908 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 5 | max_down | 37 | -0.019582 | 1806 | -0.0216934 | 0.00211143 | -0.0050718 | 0.00929467 | 0.564533 | 0.773165 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 5 | max_up | 37 | 0.0326019 | 1806 | 0.0207928 | 0.0118091 | -0.00345644 | 0.0270747 | 0.129465 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 5 | terminal_return | 37 | 0.013274 | 1806 | 0.00205258 | 0.0112214 | -0.00109788 | 0.0235407 | 0.0742082 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 10 | max_down | 37 | -0.0304193 | 1801 | -0.0313707 | 0.000951446 | -0.0141793 | 0.0160821 | 0.90191 | 0.996848 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 10 | max_up | 37 | 0.0433924 | 1801 | 0.031133 | 0.0122594 | -0.0038876 | 0.0284064 | 0.136723 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 10 | terminal_return | 37 | 0.00890693 | 1801 | 0.00426187 | 0.00464506 | -0.0152474 | 0.0245375 | 0.647185 | 0.832095 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 20 | max_down | 37 | -0.0365118 | 1791 | -0.0451632 | 0.00865145 | -0.00922454 | 0.0265274 | 0.342833 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 20 | max_up | 37 | 0.0604259 | 1791 | 0.0472665 | 0.0131594 | -0.00875323 | 0.035072 | 0.239172 | 0.646543 | 0.914695 | true |
| ml_today_elastic_net | top | capped_confirmation | 中证500 | 20 | terminal_return | 37 | 0.0212215 | 1791 | 0.00825592 | 0.0129655 | -0.0104156 | 0.0363467 | 0.27709 | 0.646543 | 0.943604 | true |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 5 | max_down | 37 | -0.0219186 | 1806 | -0.0262584 | 0.00433979 | -0.00311362 | 0.0117932 | 0.253778 | 0.646543 | 0.932596 | true |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 5 | max_up | 37 | 0.0383881 | 1806 | 0.0275956 | 0.0107925 | -0.00568657 | 0.0272716 | 0.199266 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 5 | terminal_return | 37 | 0.019736 | 1806 | 0.00663175 | 0.0131042 | -0.000953464 | 0.0271619 | 0.0676903 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 10 | max_down | 37 | -0.0382266 | 1801 | -0.038203 | -2.35722e-05 | -0.0205453 | 0.0204982 | 0.998204 | 0.998204 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 10 | max_up | 37 | 0.0557764 | 1801 | 0.0429126 | 0.0128638 | -0.00497938 | 0.030707 | 0.157645 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 10 | terminal_return | 37 | 0.0185847 | 1801 | 0.013429 | 0.00515564 | -0.0216881 | 0.0319993 | 0.70659 | 0.890303 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 20 | max_down | 37 | -0.0448524 | 1791 | -0.0555778 | 0.0107253 | -0.0123405 | 0.0337912 | 0.362096 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 20 | max_up | 37 | 0.0847115 | 1791 | 0.0662247 | 0.0184867 | -0.00839122 | 0.0453647 | 0.177628 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 微盘股 | 20 | terminal_return | 37 | 0.042133 | 1791 | 0.0262082 | 0.0159248 | -0.0157389 | 0.0475885 | 0.324254 | 0.678403 | 0.996488 | true |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 5 | max_down | 37 | -0.0149522 | 1806 | -0.0163611 | 0.0014089 | -0.00391936 | 0.00673715 | 0.604275 | 0.79311 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 5 | max_up | 37 | 0.0232817 | 1806 | 0.0154757 | 0.00780606 | -0.00345444 | 0.0190666 | 0.174235 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 5 | terminal_return | 37 | 0.00742719 | 1806 | 0.00140709 | 0.0060201 | -0.00279248 | 0.0148327 | 0.180594 | 0.646543 | 0.911847 | true |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 10 | max_down | 37 | -0.0245426 | 1801 | -0.0234444 | -0.00109822 | -0.0124864 | 0.01029 | 0.850083 | 0.973732 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 10 | max_up | 37 | 0.0303938 | 1801 | 0.02323 | 0.00716372 | -0.00444852 | 0.018776 | 0.226607 | 0.646543 | 0.91323 | true |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 10 | terminal_return | 37 | 0.0031303 | 1801 | 0.00291866 | 0.000211637 | -0.0135746 | 0.0139979 | 0.975996 | 0.998204 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 20 | max_down | 37 | -0.0288574 | 1791 | -0.0336138 | 0.00475633 | -0.00898016 | 0.0184928 | 0.497354 | 0.72185 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 20 | max_up | 37 | 0.0412107 | 1791 | 0.0349173 | 0.00629343 | -0.00832811 | 0.020915 | 0.398878 | 0.678403 | 0.999178 | true |
| ml_today_elastic_net | top | capped_confirmation | 上证指数 | 20 | terminal_return | 37 | 0.0101166 | 1791 | 0.00547578 | 0.00464082 | -0.0105735 | 0.0198551 | 0.549933 | 0.769907 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 全A | 5 | max_down | 37 | -0.017357 | 1806 | -0.0199053 | 0.00254835 | -0.00339231 | 0.008489 | 0.400474 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 全A | 5 | max_up | 37 | 0.0279614 | 1806 | 0.0184398 | 0.00952162 | -0.00694967 | 0.0259929 | 0.257204 | 0.650337 | 0.932596 | true |
| ml_today_elastic_net | top | onset | 全A | 5 | terminal_return | 37 | 0.0115547 | 1806 | 0.00169803 | 0.00985662 | -0.00191267 | 0.0216259 | 0.100699 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 全A | 10 | max_down | 37 | -0.0289047 | 1801 | -0.0287973 | -0.000107427 | -0.01255 | 0.0123351 | 0.986499 | 0.999178 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 全A | 10 | max_up | 37 | 0.0378279 | 1801 | 0.0276021 | 0.0102258 | -0.006205 | 0.0266566 | 0.222534 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 全A | 10 | terminal_return | 37 | 0.00731235 | 1801 | 0.00356721 | 0.00374515 | -0.0139901 | 0.0214804 | 0.678953 | 0.838706 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 全A | 20 | max_down | 37 | -0.0349462 | 1791 | -0.0414 | 0.00645388 | -0.00955068 | 0.0224584 | 0.429309 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 全A | 20 | max_up | 37 | 0.0516782 | 1791 | 0.0415834 | 0.0100948 | -0.010632 | 0.0308216 | 0.33978 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 全A | 20 | terminal_return | 37 | 0.0163929 | 1791 | 0.00681807 | 0.00957481 | -0.0130432 | 0.0321929 | 0.406698 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 国证2000 | 5 | max_down | 37 | -0.023012 | 1806 | -0.0253223 | 0.00231034 | -0.00508467 | 0.00970535 | 0.540313 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 国证2000 | 5 | max_up | 37 | 0.0337692 | 1806 | 0.0226537 | 0.0111154 | -0.00759181 | 0.0298227 | 0.244185 | 0.650337 | 0.914695 | true |
| ml_today_elastic_net | top | onset | 国证2000 | 5 | terminal_return | 37 | 0.014319 | 1806 | 0.00232025 | 0.0119987 | -0.00218451 | 0.026182 | 0.0972931 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 国证2000 | 10 | max_down | 37 | -0.03733 | 1801 | -0.0371205 | -0.00020954 | -0.0168345 | 0.0164154 | 0.980291 | 0.999178 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 国证2000 | 10 | max_up | 37 | 0.0471187 | 1801 | 0.0342169 | 0.0129018 | -0.00641055 | 0.0322141 | 0.1904 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 国证2000 | 10 | terminal_return | 37 | 0.00889491 | 1801 | 0.00477958 | 0.00411533 | -0.0185154 | 0.0267461 | 0.721526 | 0.84178 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 国证2000 | 20 | max_down | 37 | -0.0461845 | 1791 | -0.0542442 | 0.00805973 | -0.0124337 | 0.0285532 | 0.440804 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 国证2000 | 20 | max_up | 37 | 0.0667788 | 1791 | 0.0511661 | 0.0156127 | -0.0105162 | 0.0417417 | 0.241538 | 0.650337 | 0.914695 | true |
| ml_today_elastic_net | top | onset | 国证2000 | 20 | terminal_return | 37 | 0.0249141 | 1791 | 0.00901212 | 0.015902 | -0.0140172 | 0.0458212 | 0.297534 | 0.650337 | 0.967168 | true |
| ml_today_elastic_net | top | onset | 中证1000 | 5 | max_down | 37 | -0.0221683 | 1806 | -0.0249662 | 0.00279782 | -0.00453152 | 0.0101272 | 0.454347 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 中证1000 | 5 | max_up | 37 | 0.0334176 | 1806 | 0.0224571 | 0.0109605 | -0.00786839 | 0.0297894 | 0.253896 | 0.650337 | 0.932596 | true |
| ml_today_elastic_net | top | onset | 中证1000 | 5 | terminal_return | 37 | 0.0142743 | 1806 | 0.00189592 | 0.0123784 | -0.00220852 | 0.0269652 | 0.096263 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 中证1000 | 10 | max_down | 37 | -0.036259 | 1801 | -0.0365621 | 0.000303183 | -0.0156292 | 0.0162355 | 0.970248 | 0.999178 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 中证1000 | 10 | max_up | 37 | 0.0455085 | 1801 | 0.0336451 | 0.0118634 | -0.00757719 | 0.0313039 | 0.23167 | 0.650337 | 0.91323 | true |
| ml_today_elastic_net | top | onset | 中证1000 | 10 | terminal_return | 37 | 0.00820159 | 1801 | 0.00396341 | 0.00423818 | -0.0178103 | 0.0262867 | 0.706357 | 0.84178 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 中证1000 | 20 | max_down | 37 | -0.0457165 | 1791 | -0.0531678 | 0.00745131 | -0.0127459 | 0.0276485 | 0.469619 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 中证1000 | 20 | max_up | 37 | 0.0649705 | 1791 | 0.0503032 | 0.0146673 | -0.0116385 | 0.0409731 | 0.274466 | 0.650337 | 0.943604 | true |
| ml_today_elastic_net | top | onset | 中证1000 | 20 | terminal_return | 37 | 0.0238173 | 1791 | 0.00744217 | 0.0163751 | -0.0134197 | 0.0461699 | 0.281386 | 0.650337 | 0.944803 | true |
| ml_today_elastic_net | top | onset | 沪深300 | 5 | max_down | 37 | -0.0159708 | 1806 | -0.0181644 | 0.00219361 | -0.00313302 | 0.00752024 | 0.419571 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 沪深300 | 5 | max_up | 37 | 0.0262466 | 1806 | 0.0181279 | 0.00811868 | -0.00687415 | 0.0231115 | 0.288532 | 0.650337 | 0.944803 | true |
| ml_today_elastic_net | top | onset | 沪深300 | 5 | terminal_return | 37 | 0.00919581 | 1806 | 0.00141015 | 0.00778566 | -0.00271833 | 0.0182896 | 0.146287 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 沪深300 | 10 | max_down | 37 | -0.0260316 | 1801 | -0.026026 | -5.65261e-06 | -0.0107578 | 0.0107465 | 0.999178 | 0.999178 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 沪深300 | 10 | max_up | 37 | 0.0355002 | 1801 | 0.0272347 | 0.00826549 | -0.00698336 | 0.0235143 | 0.288055 | 0.650337 | 0.944803 | true |
| ml_today_elastic_net | top | onset | 沪深300 | 10 | terminal_return | 37 | 0.00461192 | 1801 | 0.00297726 | 0.00163466 | -0.0138587 | 0.017128 | 0.836171 | 0.925682 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 沪深300 | 20 | max_down | 37 | -0.0320582 | 1791 | -0.0368579 | 0.00479976 | -0.00980047 | 0.0194 | 0.519354 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 沪深300 | 20 | max_up | 37 | 0.0468632 | 1791 | 0.0412326 | 0.00563057 | -0.0116343 | 0.0228955 | 0.522686 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 沪深300 | 20 | terminal_return | 37 | 0.00931293 | 1791 | 0.00572729 | 0.00358563 | -0.0152289 | 0.0224002 | 0.708752 | 0.84178 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 中证500 | 5 | max_down | 37 | -0.0194366 | 1806 | -0.0216964 | 0.00225972 | -0.00497207 | 0.0094915 | 0.540245 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 中证500 | 5 | max_up | 37 | 0.0326693 | 1806 | 0.0207914 | 0.0118779 | -0.0060562 | 0.0298119 | 0.194245 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 中证500 | 5 | terminal_return | 37 | 0.0145295 | 1806 | 0.00202686 | 0.0125027 | -0.00139788 | 0.0264032 | 0.0779178 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 中证500 | 10 | max_down | 37 | -0.0309639 | 1801 | -0.0313596 | 0.000395678 | -0.0131904 | 0.0139818 | 0.95448 | 0.999178 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 中证500 | 10 | max_up | 37 | 0.0450266 | 1801 | 0.0310994 | 0.0139272 | -0.00441741 | 0.0322719 | 0.136743 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 中证500 | 10 | terminal_return | 37 | 0.0111758 | 1801 | 0.00421526 | 0.00696056 | -0.012908 | 0.0268291 | 0.492306 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 中证500 | 20 | max_down | 37 | -0.0376013 | 1791 | -0.0451407 | 0.00753945 | -0.00998943 | 0.0250683 | 0.399213 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 中证500 | 20 | max_up | 37 | 0.0622368 | 1791 | 0.047229 | 0.0150077 | -0.00875526 | 0.0387707 | 0.21577 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 中证500 | 20 | terminal_return | 37 | 0.0239701 | 1791 | 0.00819914 | 0.015771 | -0.0105999 | 0.0421419 | 0.24113 | 0.650337 | 0.914695 | true |
| ml_today_elastic_net | top | onset | 微盘股 | 5 | max_down | 37 | -0.0233319 | 1806 | -0.0262294 | 0.00289753 | -0.00502771 | 0.0108228 | 0.473626 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 微盘股 | 5 | max_up | 37 | 0.0377395 | 1806 | 0.0276089 | 0.0101306 | -0.00694893 | 0.0272102 | 0.245008 | 0.650337 | 0.914695 | true |
| ml_today_elastic_net | top | onset | 微盘股 | 5 | terminal_return | 37 | 0.0165276 | 1806 | 0.00669748 | 0.0098301 | -0.00362403 | 0.0232842 | 0.152129 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 微盘股 | 10 | max_down | 37 | -0.0384849 | 1801 | -0.0381977 | -0.00028722 | -0.0185049 | 0.0179304 | 0.975348 | 0.999178 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 微盘股 | 10 | max_up | 37 | 0.0572346 | 1801 | 0.0428826 | 0.014352 | -0.00450558 | 0.0332095 | 0.135777 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 微盘股 | 10 | terminal_return | 37 | 0.0192022 | 1801 | 0.0134163 | 0.00578584 | -0.0193566 | 0.0309283 | 0.651961 | 0.821471 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 微盘股 | 20 | max_down | 37 | -0.0463592 | 1791 | -0.0555466 | 0.00918739 | -0.012907 | 0.0312818 | 0.415064 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 微盘股 | 20 | max_up | 37 | 0.0854916 | 1791 | 0.0662086 | 0.019283 | -0.00803202 | 0.046598 | 0.166463 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 微盘股 | 20 | terminal_return | 37 | 0.0451783 | 1791 | 0.0261453 | 0.019033 | -0.0145419 | 0.0526079 | 0.266531 | 0.650337 | 0.943604 | true |
| ml_today_elastic_net | top | onset | 上证指数 | 5 | max_down | 37 | -0.0146568 | 1806 | -0.0163671 | 0.00171034 | -0.00332868 | 0.00674937 | 0.505883 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 上证指数 | 5 | max_up | 37 | 0.0224006 | 1806 | 0.0154937 | 0.0069069 | -0.0061375 | 0.0199513 | 0.299361 | 0.650337 | 0.967168 | true |
| ml_today_elastic_net | top | onset | 上证指数 | 5 | terminal_return | 37 | 0.0079561 | 1806 | 0.00139626 | 0.00655984 | -0.00184517 | 0.0149648 | 0.126087 | 0.650337 | 0.911847 | true |
| ml_today_elastic_net | top | onset | 上证指数 | 10 | max_down | 37 | -0.024669 | 1801 | -0.0234418 | -0.0012272 | -0.011554 | 0.00909964 | 0.815824 | 0.925682 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 上证指数 | 10 | max_up | 37 | 0.031184 | 1801 | 0.0232138 | 0.00797023 | -0.00519961 | 0.0211401 | 0.235555 | 0.650337 | 0.91323 | true |
| ml_today_elastic_net | top | onset | 上证指数 | 10 | terminal_return | 37 | 0.00436337 | 1801 | 0.00289333 | 0.00147004 | -0.0125805 | 0.0155206 | 0.837521 | 0.925682 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 上证指数 | 20 | max_down | 37 | -0.0299127 | 1791 | -0.033592 | 0.00367927 | -0.00975417 | 0.0171127 | 0.59139 | 0.760359 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 上证指数 | 20 | max_up | 37 | 0.0419536 | 1791 | 0.0349019 | 0.00705166 | -0.00857497 | 0.0226783 | 0.376444 | 0.70916 | 0.999178 | true |
| ml_today_elastic_net | top | onset | 上证指数 | 20 | terminal_return | 37 | 0.0109712 | 1791 | 0.00545813 | 0.00551302 | -0.011139 | 0.0221651 | 0.516403 | 0.70916 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 5 | max_down | 49 | -0.0212138 | 1794 | -0.019817 | -0.00139676 | -0.00828316 | 0.00548963 | 0.690965 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 5 | max_up | 49 | 0.0225655 | 1794 | 0.0185235 | 0.00404203 | -0.00186004 | 0.00994409 | 0.179497 | 0.990227 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 5 | terminal_return | 49 | 0.004632 | 1794 | 0.00182118 | 0.00281082 | -0.00719713 | 0.0128188 | 0.581988 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 10 | max_down | 49 | -0.0312233 | 1789 | -0.0287331 | -0.00249021 | -0.0145382 | 0.00955775 | 0.685393 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 10 | max_up | 49 | 0.031453 | 1789 | 0.0277081 | 0.00374493 | -0.0053821 | 0.012872 | 0.421275 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 10 | terminal_return | 49 | 0.00563611 | 1789 | 0.003588 | 0.00204812 | -0.0143786 | 0.0184748 | 0.806939 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 20 | max_down | 49 | -0.0419587 | 1779 | -0.0412504 | -0.000708315 | -0.0169029 | 0.0154863 | 0.931684 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 20 | max_up | 49 | 0.0414457 | 1779 | 0.0417971 | -0.000351446 | -0.0155777 | 0.0148748 | 0.963916 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 全A | 20 | terminal_return | 49 | 0.0038308 | 1779 | 0.00709949 | -0.00326868 | -0.0254271 | 0.0188898 | 0.772484 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 5 | max_down | 49 | -0.02637 | 1794 | -0.0252461 | -0.00112389 | -0.0104307 | 0.00818288 | 0.812896 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 5 | max_up | 49 | 0.0285459 | 1794 | 0.022722 | 0.00582381 | -0.00166226 | 0.0133099 | 0.127312 | 0.990227 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 5 | terminal_return | 49 | 0.00694526 | 1794 | 0.00244139 | 0.00450387 | -0.00812586 | 0.0171336 | 0.484582 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 10 | max_down | 49 | -0.0385959 | 1789 | -0.0370844 | -0.00151148 | -0.0164717 | 0.0134487 | 0.843025 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 10 | max_up | 49 | 0.039689 | 1789 | 0.0343339 | 0.00535514 | -0.00614916 | 0.0168595 | 0.361579 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 10 | terminal_return | 49 | 0.00976287 | 1789 | 0.0047282 | 0.00503466 | -0.015184 | 0.0252533 | 0.625507 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 20 | max_down | 49 | -0.053372 | 1779 | -0.0541006 | 0.000728595 | -0.0200081 | 0.0214653 | 0.945097 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 20 | max_up | 49 | 0.0531127 | 1779 | 0.0514372 | 0.00167551 | -0.0176405 | 0.0209915 | 0.864999 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 国证2000 | 20 | terminal_return | 49 | 0.00949917 | 1779 | 0.00932944 | 0.000169734 | -0.0269905 | 0.0273299 | 0.990227 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 5 | max_down | 49 | -0.0264948 | 1794 | -0.0248667 | -0.00162809 | -0.0109425 | 0.00768633 | 0.731905 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 5 | max_up | 49 | 0.0270881 | 1794 | 0.0225567 | 0.0045314 | -0.00272082 | 0.0117836 | 0.220701 | 0.990227 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 5 | terminal_return | 49 | 0.00466913 | 1794 | 0.00207547 | 0.00259366 | -0.00995241 | 0.0151397 | 0.685336 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 10 | max_down | 49 | -0.0388086 | 1789 | -0.0364943 | -0.00231422 | -0.0171698 | 0.0125414 | 0.760115 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 10 | max_up | 49 | 0.0374169 | 1789 | 0.0337872 | 0.00362969 | -0.00764386 | 0.0149032 | 0.528006 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 10 | terminal_return | 49 | 0.00584311 | 1789 | 0.00399958 | 0.00184353 | -0.0183358 | 0.0220228 | 0.85789 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 20 | max_down | 49 | -0.0538861 | 1779 | -0.0529931 | -0.000893 | -0.0215092 | 0.0197232 | 0.932342 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 20 | max_up | 49 | 0.0500561 | 1779 | 0.050615 | -0.00055896 | -0.0199287 | 0.0188108 | 0.954895 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证1000 | 20 | terminal_return | 49 | 0.0040727 | 1779 | 0.00787555 | -0.00380285 | -0.0313723 | 0.0237666 | 0.786885 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 5 | max_down | 49 | -0.0192587 | 1794 | -0.0180893 | -0.00116942 | -0.00678442 | 0.00444557 | 0.683123 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 5 | max_up | 49 | 0.0201052 | 1794 | 0.0182413 | 0.00186385 | -0.00339801 | 0.00712571 | 0.487514 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 5 | terminal_return | 49 | 0.0033767 | 1794 | 0.00151701 | 0.00185969 | -0.00685204 | 0.0105714 | 0.675654 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 10 | max_down | 49 | -0.0277608 | 1789 | -0.0259786 | -0.00178221 | -0.0126381 | 0.00907365 | 0.747624 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 10 | max_up | 49 | 0.0294675 | 1789 | 0.0273445 | 0.00212296 | -0.00611138 | 0.0103573 | 0.613332 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 10 | terminal_return | 49 | 0.00384635 | 1789 | 0.00298726 | 0.000859089 | -0.0137862 | 0.0155043 | 0.908466 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 20 | max_down | 49 | -0.0371205 | 1779 | -0.0367509 | -0.000369626 | -0.01436 | 0.0136208 | 0.958702 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 20 | max_up | 49 | 0.0388423 | 1779 | 0.0414156 | -0.00257324 | -0.0157156 | 0.0105691 | 0.701154 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 沪深300 | 20 | terminal_return | 49 | 0.00077127 | 1779 | 0.00593837 | -0.0051671 | -0.0247716 | 0.0144374 | 0.605441 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 5 | max_down | 49 | -0.0231853 | 1794 | -0.0216091 | -0.00157617 | -0.00979274 | 0.00664039 | 0.706929 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 5 | max_up | 49 | 0.024269 | 1794 | 0.0209414 | 0.00332761 | -0.00301399 | 0.0096692 | 0.303731 | 0.990227 | 0.975002 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 5 | terminal_return | 49 | 0.00448345 | 1794 | 0.00221762 | 0.00226583 | -0.00858216 | 0.0131138 | 0.682255 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 10 | max_down | 49 | -0.0341723 | 1789 | -0.0312743 | -0.00289801 | -0.0165568 | 0.0107608 | 0.677515 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 10 | max_up | 49 | 0.0339575 | 1789 | 0.0313091 | 0.00264833 | -0.00743001 | 0.0127267 | 0.606526 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 10 | terminal_return | 49 | 0.00528782 | 1789 | 0.00432984 | 0.000957971 | -0.0172394 | 0.0191554 | 0.917819 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 20 | max_down | 49 | -0.0462591 | 1779 | -0.0449531 | -0.00130598 | -0.0194329 | 0.016821 | 0.887703 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 20 | max_up | 49 | 0.0453348 | 1779 | 0.0475934 | -0.00225859 | -0.0196935 | 0.0151763 | 0.799567 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 中证500 | 20 | terminal_return | 49 | 0.00429429 | 1779 | 0.0086347 | -0.00434041 | -0.0290807 | 0.0203999 | 0.730951 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 5 | max_down | 49 | -0.023382 | 1794 | -0.0262475 | 0.00286551 | -0.00486431 | 0.0105953 | 0.467478 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 5 | max_up | 49 | 0.0349121 | 1794 | 0.0276184 | 0.00729375 | -0.000942613 | 0.0155301 | 0.0826191 | 0.990227 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 5 | terminal_return | 49 | 0.0164761 | 1794 | 0.00663313 | 0.00984302 | -0.00191305 | 0.0215991 | 0.100787 | 0.990227 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 10 | max_down | 49 | -0.0324961 | 1789 | -0.0383598 | 0.0058637 | -0.00549101 | 0.0172184 | 0.311459 | 0.990227 | 0.975002 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 10 | max_up | 49 | 0.0467996 | 1789 | 0.0430722 | 0.00372743 | -0.00796733 | 0.0154222 | 0.532166 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 10 | terminal_return | 49 | 0.0244808 | 1789 | 0.0132329 | 0.0112479 | -0.00596691 | 0.0284627 | 0.200322 | 0.990227 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 20 | max_down | 49 | -0.0480695 | 1779 | -0.0555615 | 0.00749204 | -0.0105413 | 0.0255254 | 0.415479 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 20 | max_up | 49 | 0.0709223 | 1779 | 0.0664798 | 0.00444247 | -0.0152733 | 0.0241583 | 0.658751 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 微盘股 | 20 | terminal_return | 49 | 0.0385826 | 1779 | 0.0261986 | 0.0123841 | -0.0145759 | 0.039344 | 0.367947 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 5 | max_down | 49 | -0.0167014 | 1794 | -0.0163227 | -0.000378689 | -0.00543646 | 0.00467908 | 0.883329 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 5 | max_up | 49 | 0.0186113 | 1794 | 0.015551 | 0.00306029 | -0.00134113 | 0.00746171 | 0.172952 | 0.990227 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 5 | terminal_return | 49 | 0.00401117 | 1794 | 0.00146013 | 0.00255104 | -0.00511453 | 0.0102166 | 0.514226 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 10 | max_down | 49 | -0.0247824 | 1789 | -0.0234305 | -0.00135195 | -0.0114215 | 0.00871758 | 0.792433 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 10 | max_up | 49 | 0.0260192 | 1789 | 0.0233018 | 0.00271739 | -0.00408536 | 0.00952015 | 0.433667 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 10 | terminal_return | 49 | 0.00388769 | 1789 | 0.0028965 | 0.000991194 | -0.0117242 | 0.0137066 | 0.878567 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 20 | max_down | 49 | -0.0336489 | 1779 | -0.0335139 | -0.000135023 | -0.0138145 | 0.0135445 | 0.984565 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 20 | max_up | 49 | 0.0332628 | 1779 | 0.0350937 | -0.00183097 | -0.0127547 | 0.0090928 | 0.742516 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | capped_confirmation | 上证指数 | 20 | terminal_return | 49 | 0.00294598 | 1779 | 0.00564198 | -0.002696 | -0.0193526 | 0.0139606 | 0.75106 | 0.990227 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 全A | 5 | max_down | 49 | -0.019375 | 1794 | -0.0198672 | 0.000492234 | -0.00568534 | 0.00666981 | 0.875896 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 全A | 5 | max_up | 49 | 0.0238735 | 1794 | 0.0184878 | 0.00538569 | -0.000665986 | 0.0114374 | 0.0811064 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 全A | 5 | terminal_return | 49 | 0.00786065 | 1794 | 0.00173299 | 0.00612766 | -0.00366861 | 0.0159239 | 0.220199 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 全A | 10 | max_down | 49 | -0.0295054 | 1789 | -0.0287801 | -0.000725242 | -0.0121275 | 0.010677 | 0.900788 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 全A | 10 | max_up | 49 | 0.0342084 | 1789 | 0.0276326 | 0.00657581 | -0.00289351 | 0.0160451 | 0.173485 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 全A | 10 | terminal_return | 49 | 0.00872832 | 1789 | 0.0035033 | 0.00522501 | -0.0102717 | 0.0207217 | 0.508706 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 全A | 20 | max_down | 49 | -0.0404017 | 1779 | -0.0412933 | 0.000891572 | -0.0154871 | 0.0172702 | 0.915033 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 全A | 20 | max_up | 49 | 0.044941 | 1779 | 0.0417009 | 0.00324012 | -0.0127224 | 0.0192026 | 0.690744 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 全A | 20 | terminal_return | 49 | 0.00938618 | 1779 | 0.00694647 | 0.0024397 | -0.020175 | 0.0250544 | 0.832538 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 5 | max_down | 49 | -0.0242138 | 1794 | -0.025305 | 0.00109121 | -0.00713943 | 0.00932184 | 0.794976 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 5 | max_up | 49 | 0.0313194 | 1794 | 0.0226463 | 0.00867312 | 0.00108455 | 0.0162617 | 0.025083 | 0.526743 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 5 | terminal_return | 49 | 0.0120094 | 1794 | 0.00230307 | 0.00970634 | -0.0032107 | 0.0226234 | 0.140801 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 10 | max_down | 49 | -0.0373641 | 1789 | -0.0371182 | -0.000245903 | -0.0146952 | 0.0142034 | 0.973391 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 10 | max_up | 49 | 0.0428379 | 1789 | 0.0342477 | 0.00859023 | -0.00298046 | 0.0201609 | 0.145634 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 10 | terminal_return | 49 | 0.0121672 | 1789 | 0.00466235 | 0.00750485 | -0.0117981 | 0.0268078 | 0.44604 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 20 | max_down | 49 | -0.0514554 | 1779 | -0.0541534 | 0.002698 | -0.0177496 | 0.0231457 | 0.795932 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 20 | max_up | 49 | 0.0576422 | 1779 | 0.0513124 | 0.00632971 | -0.0136437 | 0.0263031 | 0.534509 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 国证2000 | 20 | terminal_return | 49 | 0.016794 | 1779 | 0.00912851 | 0.00766547 | -0.0197656 | 0.0350966 | 0.58389 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 5 | max_down | 49 | -0.0243096 | 1794 | -0.0249264 | 0.000616827 | -0.00763037 | 0.00886402 | 0.883453 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 5 | max_up | 49 | 0.0297225 | 1794 | 0.0224848 | 0.00723775 | -0.000175645 | 0.0146511 | 0.0556762 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 5 | terminal_return | 49 | 0.00910457 | 1794 | 0.00195433 | 0.00715025 | -0.00573295 | 0.0200334 | 0.276679 | 0.805183 | 0.943604 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 10 | max_down | 49 | -0.0375487 | 1789 | -0.0365288 | -0.00101983 | -0.0155188 | 0.0134792 | 0.890349 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 10 | max_up | 49 | 0.0408299 | 1789 | 0.0336937 | 0.00713616 | -0.00429803 | 0.0185703 | 0.221235 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 10 | terminal_return | 49 | 0.0084991 | 1789 | 0.00392683 | 0.00457227 | -0.014693 | 0.0238376 | 0.64181 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 20 | max_down | 49 | -0.0521166 | 1779 | -0.0530418 | 0.000925248 | -0.0196397 | 0.0214902 | 0.929731 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 20 | max_up | 49 | 0.0545488 | 1779 | 0.0504913 | 0.00405748 | -0.0159994 | 0.0241144 | 0.691732 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证1000 | 20 | terminal_return | 49 | 0.0110933 | 1779 | 0.00768218 | 0.00341111 | -0.0243208 | 0.031143 | 0.809489 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 5 | max_down | 49 | -0.0173636 | 1794 | -0.0181411 | 0.000777411 | -0.004498 | 0.00605282 | 0.772707 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 5 | max_up | 49 | 0.0203528 | 1794 | 0.0182346 | 0.00211827 | -0.00319057 | 0.00742711 | 0.434183 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 5 | terminal_return | 49 | 0.00544296 | 1794 | 0.00146057 | 0.00398238 | -0.00413005 | 0.0120948 | 0.335969 | 0.881917 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 10 | max_down | 49 | -0.0257632 | 1789 | -0.0260333 | 0.000270028 | -0.0100953 | 0.0106354 | 0.959278 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 10 | max_up | 49 | 0.0312195 | 1789 | 0.0272965 | 0.00392298 | -0.00442753 | 0.0122735 | 0.357161 | 0.900046 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 10 | terminal_return | 49 | 0.00720971 | 1789 | 0.00289514 | 0.00431457 | -0.00947869 | 0.0181078 | 0.539815 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 20 | max_down | 49 | -0.0356655 | 1779 | -0.0367909 | 0.00112544 | -0.0133585 | 0.0156093 | 0.878952 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 20 | max_up | 49 | 0.0413892 | 1779 | 0.0413454 | 4.37689e-05 | -0.0136974 | 0.013785 | 0.995019 | 0.995019 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 沪深300 | 20 | terminal_return | 49 | 0.00494376 | 1779 | 0.00582345 | -0.000879687 | -0.0209926 | 0.0192332 | 0.931685 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 5 | max_down | 49 | -0.0212052 | 1794 | -0.0216632 | 0.000457986 | -0.00689736 | 0.00781333 | 0.902867 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 5 | max_up | 49 | 0.0262495 | 1794 | 0.0208873 | 0.00536224 | -0.00129806 | 0.0120225 | 0.114565 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 5 | terminal_return | 49 | 0.0079232 | 1794 | 0.00212367 | 0.00579953 | -0.00530749 | 0.0169066 | 0.306112 | 0.83848 | 0.975002 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 10 | max_down | 49 | -0.0322194 | 1789 | -0.0313278 | -0.000891583 | -0.0137206 | 0.0119374 | 0.891651 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 10 | max_up | 49 | 0.0369976 | 1789 | 0.0312259 | 0.00577176 | -0.00472543 | 0.016269 | 0.281175 | 0.805183 | 0.944803 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 10 | terminal_return | 49 | 0.00859581 | 1789 | 0.00423924 | 0.00435657 | -0.0123229 | 0.021036 | 0.608693 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 20 | max_down | 49 | -0.0445596 | 1779 | -0.0449999 | 0.000440287 | -0.0175859 | 0.0184665 | 0.961818 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 20 | max_up | 49 | 0.0492716 | 1779 | 0.0474849 | 0.00178673 | -0.0165266 | 0.0201001 | 0.848349 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 中证500 | 20 | terminal_return | 49 | 0.0102806 | 1779 | 0.00846982 | 0.00181076 | -0.0229966 | 0.0266181 | 0.886238 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 5 | max_down | 49 | -0.0213007 | 1794 | -0.0263043 | 0.00500361 | -0.00208369 | 0.0120909 | 0.166434 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 5 | max_up | 49 | 0.038192 | 1794 | 0.0275288 | 0.0106632 | 0.00225803 | 0.0190684 | 0.0128988 | 0.406313 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 5 | terminal_return | 49 | 0.0237766 | 1794 | 0.00643373 | 0.0173428 | 0.00579087 | 0.0288948 | 0.00325545 | 0.205093 | 0.820374 | true |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 10 | max_down | 49 | -0.0311736 | 1789 | -0.038396 | 0.00722239 | -0.00356033 | 0.0180051 | 0.18924 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 10 | max_up | 49 | 0.0509207 | 1789 | 0.0429593 | 0.00796141 | -0.00395954 | 0.0198824 | 0.190539 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 10 | terminal_return | 49 | 0.0268439 | 1789 | 0.0131682 | 0.0136757 | -0.00353765 | 0.030889 | 0.119426 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 20 | max_down | 49 | -0.0449081 | 1779 | -0.0556486 | 0.0107405 | -0.00633394 | 0.027815 | 0.217606 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 20 | max_up | 49 | 0.0751584 | 1779 | 0.0663631 | 0.00879526 | -0.0115837 | 0.0291742 | 0.397604 | 0.963426 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 微盘股 | 20 | terminal_return | 49 | 0.0464866 | 1779 | 0.0259809 | 0.0205057 | -0.00707639 | 0.0480878 | 0.145076 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 5 | max_down | 49 | -0.0155716 | 1794 | -0.0163536 | 0.000782017 | -0.00400663 | 0.00557066 | 0.748907 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 5 | max_up | 49 | 0.0186405 | 1794 | 0.0155502 | 0.0030903 | -0.00127996 | 0.00746055 | 0.16576 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 5 | terminal_return | 49 | 0.0062486 | 1794 | 0.00139902 | 0.00484959 | -0.0023628 | 0.012062 | 0.187538 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 10 | max_down | 49 | -0.0230665 | 1789 | -0.0234775 | 0.000410931 | -0.00884952 | 0.00967138 | 0.930692 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 10 | max_up | 49 | 0.0275885 | 1789 | 0.0232588 | 0.00432971 | -0.00252442 | 0.0111838 | 0.215672 | 0.696891 | 0.911847 | true |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 10 | terminal_return | 49 | 0.00652779 | 1789 | 0.00282419 | 0.0037036 | -0.00808127 | 0.0154885 | 0.537918 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 20 | max_down | 49 | -0.0326032 | 1779 | -0.0335427 | 0.00093946 | -0.0129654 | 0.0148443 | 0.894648 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 20 | max_up | 49 | 0.0354698 | 1779 | 0.0350329 | 0.000436895 | -0.0108783 | 0.0117521 | 0.939675 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | bottom | onset | 上证指数 | 20 | terminal_return | 49 | 0.0062935 | 1779 | 0.00554978 | 0.000743723 | -0.0163032 | 0.0177906 | 0.931855 | 0.989091 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 5 | max_down | 44 | -0.0191766 | 1799 | -0.0198707 | 0.000694104 | -0.00449111 | 0.00587932 | 0.793036 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 5 | max_up | 44 | 0.0236499 | 1799 | 0.0185082 | 0.00514168 | -0.00678376 | 0.0170671 | 0.398079 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 5 | terminal_return | 44 | 0.0048329 | 1799 | 0.00182408 | 0.00300882 | -0.00575161 | 0.0117692 | 0.500837 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 10 | max_down | 44 | -0.0285393 | 1794 | -0.0288058 | 0.000266541 | -0.00700146 | 0.00753454 | 0.942698 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 10 | max_up | 44 | 0.0304409 | 1794 | 0.0277434 | 0.00269756 | -0.00933017 | 0.0147253 | 0.660237 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 10 | terminal_return | 44 | 0.00404416 | 1794 | 0.00363275 | 0.000411408 | -0.0113863 | 0.0122091 | 0.945508 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 20 | max_down | 44 | -0.0424834 | 1784 | -0.0412395 | -0.00124392 | -0.0123804 | 0.00989253 | 0.826706 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 20 | max_up | 44 | 0.0446612 | 1784 | 0.0417168 | 0.0029444 | -0.0130975 | 0.0189862 | 0.719036 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 全A | 20 | terminal_return | 44 | 0.00786625 | 1784 | 0.0069908 | 0.00087545 | -0.0169344 | 0.0186853 | 0.923247 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 5 | max_down | 44 | -0.0251804 | 1799 | -0.0252783 | 9.79172e-05 | -0.00664365 | 0.00683949 | 0.977289 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 5 | max_up | 44 | 0.0271137 | 1799 | 0.0227733 | 0.00434046 | -0.00931809 | 0.017999 | 0.53338 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 5 | terminal_return | 44 | 0.00498326 | 1799 | 0.00250189 | 0.00248137 | -0.00777508 | 0.0127378 | 0.635366 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 10 | max_down | 44 | -0.0373605 | 1794 | -0.0371189 | -0.000241598 | -0.00978537 | 0.00930218 | 0.960428 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 10 | max_up | 44 | 0.0383374 | 1794 | 0.034382 | 0.00395538 | -0.0108507 | 0.0187615 | 0.600553 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 10 | terminal_return | 44 | 0.00618447 | 1794 | 0.00483 | 0.00135447 | -0.0140445 | 0.0167535 | 0.863125 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 20 | max_down | 44 | -0.0556684 | 1784 | -0.0540419 | -0.00162648 | -0.016324 | 0.0130711 | 0.828286 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 20 | max_up | 44 | 0.0553722 | 1784 | 0.0513862 | 0.00398601 | -0.016681 | 0.024653 | 0.705414 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 国证2000 | 20 | terminal_return | 44 | 0.0115305 | 1784 | 0.00927981 | 0.00225064 | -0.0220515 | 0.0265528 | 0.855962 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 5 | max_down | 44 | -0.0245697 | 1799 | -0.0249183 | 0.000348642 | -0.00621678 | 0.00691407 | 0.917105 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 5 | max_up | 44 | 0.0273242 | 1799 | 0.0225635 | 0.00476063 | -0.00873309 | 0.0182544 | 0.489254 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 5 | terminal_return | 44 | 0.00532114 | 1799 | 0.00206673 | 0.00325441 | -0.00706933 | 0.0135781 | 0.536667 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 10 | max_down | 44 | -0.0360508 | 1794 | -0.0365684 | 0.000517669 | -0.00832437 | 0.00935971 | 0.908643 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 10 | max_up | 44 | 0.0377468 | 1794 | 0.0337892 | 0.00395759 | -0.010548 | 0.0184631 | 0.592821 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 10 | terminal_return | 44 | 0.00596589 | 1794 | 0.00400171 | 0.00196418 | -0.0131281 | 0.0170564 | 0.798658 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 20 | max_down | 44 | -0.0544779 | 1784 | -0.052981 | -0.00149688 | -0.0152096 | 0.0122158 | 0.830583 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 20 | max_up | 44 | 0.0553247 | 1784 | 0.0504835 | 0.00484121 | -0.0157303 | 0.0254127 | 0.644614 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证1000 | 20 | terminal_return | 44 | 0.0110035 | 1784 | 0.00769396 | 0.00330952 | -0.0208689 | 0.0274879 | 0.788482 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 5 | max_down | 44 | -0.0169621 | 1799 | -0.0181487 | 0.00118659 | -0.00330756 | 0.00568074 | 0.604809 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 5 | max_up | 44 | 0.023071 | 1799 | 0.018174 | 0.00489705 | -0.0060612 | 0.0158553 | 0.38109 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 5 | terminal_return | 44 | 0.00473612 | 1799 | 0.00148893 | 0.00324719 | -0.00494953 | 0.0114439 | 0.437473 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 10 | max_down | 44 | -0.0249222 | 1794 | -0.0260532 | 0.00113095 | -0.00532338 | 0.00758529 | 0.731268 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 10 | max_up | 44 | 0.0292866 | 1794 | 0.0273549 | 0.00193174 | -0.00931151 | 0.013175 | 0.736303 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 10 | terminal_return | 44 | 0.00177167 | 1794 | 0.00304054 | -0.00126887 | -0.0109987 | 0.00846091 | 0.798256 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 20 | max_down | 44 | -0.0374045 | 1784 | -0.0367449 | -0.000659603 | -0.0105075 | 0.00918833 | 0.895555 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 20 | max_up | 44 | 0.0417265 | 1784 | 0.0413372 | 0.000389248 | -0.0131373 | 0.0139158 | 0.955021 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 沪深300 | 20 | terminal_return | 44 | 0.00535241 | 1784 | 0.0058109 | -0.000458497 | -0.0155905 | 0.0146735 | 0.952643 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 5 | max_down | 44 | -0.020912 | 1799 | -0.0216691 | 0.000757041 | -0.00505105 | 0.00656513 | 0.798359 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 5 | max_up | 44 | 0.0256657 | 1799 | 0.0209165 | 0.00474923 | -0.00774698 | 0.0172454 | 0.456329 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 5 | terminal_return | 44 | 0.00537026 | 1799 | 0.00220223 | 0.00316803 | -0.00611603 | 0.0124521 | 0.503612 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 10 | max_down | 44 | -0.0304907 | 1794 | -0.0313727 | 0.000881994 | -0.00679118 | 0.00855517 | 0.821752 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 10 | max_up | 44 | 0.0343689 | 1794 | 0.0313064 | 0.00306248 | -0.00983824 | 0.0159632 | 0.64173 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 10 | terminal_return | 44 | 0.00501572 | 1794 | 0.00433919 | 0.000676533 | -0.0118446 | 0.0131976 | 0.91566 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 20 | max_down | 44 | -0.0465912 | 1784 | -0.0449486 | -0.00164265 | -0.013556 | 0.0102707 | 0.786967 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 20 | max_up | 44 | 0.0496525 | 1784 | 0.0474805 | 0.002172 | -0.0158319 | 0.0201759 | 0.81308 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 中证500 | 20 | terminal_return | 44 | 0.00829894 | 1784 | 0.00852377 | -0.000224828 | -0.0204254 | 0.0199758 | 0.982596 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 5 | max_down | 44 | -0.0282136 | 1799 | -0.0261213 | -0.00209227 | -0.0108909 | 0.00670635 | 0.641159 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 5 | max_up | 44 | 0.0291226 | 1799 | 0.0277802 | 0.00134235 | -0.0116876 | 0.0143723 | 0.83998 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 5 | terminal_return | 44 | 0.00453723 | 1799 | 0.00695249 | -0.00241526 | -0.0144924 | 0.00966186 | 0.695077 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 10 | max_down | 44 | -0.0423931 | 1794 | -0.0381007 | -0.00429242 | -0.0188645 | 0.0102797 | 0.563706 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 10 | max_up | 44 | 0.044149 | 1794 | 0.0431476 | 0.00100147 | -0.0145743 | 0.0165773 | 0.899715 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 10 | terminal_return | 44 | 0.00927119 | 1794 | 0.0136373 | -0.00436614 | -0.0237802 | 0.0150479 | 0.659361 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 20 | max_down | 44 | -0.0590509 | 1784 | -0.0552697 | -0.00378122 | -0.0239267 | 0.0163643 | 0.71296 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 20 | max_up | 44 | 0.0659352 | 1784 | 0.0666153 | -0.000680038 | -0.0231528 | 0.0217928 | 0.952705 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 微盘股 | 20 | terminal_return | 44 | 0.026997 | 1784 | 0.026519 | 0.000478022 | -0.0278367 | 0.0287928 | 0.973603 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 5 | max_down | 44 | -0.0157057 | 1799 | -0.0163481 | 0.000642421 | -0.00344995 | 0.00473479 | 0.758325 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 5 | max_up | 44 | 0.0197288 | 1799 | 0.0155322 | 0.00419663 | -0.00550482 | 0.0138981 | 0.39652 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 5 | terminal_return | 44 | 0.00398903 | 1799 | 0.00146776 | 0.00252127 | -0.00424973 | 0.00929227 | 0.465493 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 10 | max_down | 44 | -0.0229519 | 1794 | -0.0234791 | 0.000527198 | -0.00581226 | 0.00686665 | 0.870521 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 10 | max_up | 44 | 0.0258237 | 1794 | 0.0233142 | 0.0025095 | -0.00748489 | 0.0125039 | 0.622622 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 10 | terminal_return | 44 | 0.00171112 | 1794 | 0.00295264 | -0.00124153 | -0.0104021 | 0.00791903 | 0.790518 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 20 | max_down | 44 | -0.0354384 | 1784 | -0.0334701 | -0.00196832 | -0.0119097 | 0.00797305 | 0.697967 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 20 | max_up | 44 | 0.0366232 | 1784 | 0.0350057 | 0.0016175 | -0.0105575 | 0.0137925 | 0.794561 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | capped_confirmation | 上证指数 | 20 | terminal_return | 44 | 0.00372367 | 1784 | 0.00561525 | -0.00189158 | -0.0156846 | 0.0119014 | 0.788087 | 0.982596 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 全A | 5 | max_down | 44 | -0.0208814 | 1799 | -0.019829 | -0.00105238 | -0.00727275 | 0.005168 | 0.740193 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 全A | 5 | max_up | 44 | 0.0216224 | 1799 | 0.0185578 | 0.0030646 | -0.0101752 | 0.0163044 | 0.650061 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 全A | 5 | terminal_return | 44 | 8.97842e-05 | 1799 | 0.00194009 | -0.0018503 | -0.0118556 | 0.00815503 | 0.717004 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 全A | 10 | max_down | 44 | -0.0292802 | 1794 | -0.0287877 | -0.000492579 | -0.00765984 | 0.00667469 | 0.892846 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 全A | 10 | max_up | 44 | 0.0289113 | 1794 | 0.0277809 | 0.00113042 | -0.0108777 | 0.0131386 | 0.853613 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 全A | 10 | terminal_return | 44 | 0.00256159 | 1794 | 0.00366911 | -0.00110753 | -0.0115133 | 0.00929823 | 0.834752 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 全A | 20 | max_down | 44 | -0.0449421 | 1784 | -0.0411788 | -0.00376325 | -0.0142562 | 0.00672965 | 0.482088 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 全A | 20 | max_up | 44 | 0.0432549 | 1784 | 0.0417515 | 0.00150339 | -0.0133715 | 0.0163782 | 0.84297 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 全A | 20 | terminal_return | 44 | 0.00738743 | 1784 | 0.00700261 | 0.000384819 | -0.0168686 | 0.0176382 | 0.965131 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 5 | max_down | 44 | -0.0248395 | 1799 | -0.0252866 | 0.000447145 | -0.00701771 | 0.007912 | 0.90654 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 5 | max_up | 44 | 0.0276265 | 1799 | 0.0227607 | 0.00486582 | -0.0100347 | 0.0197663 | 0.522142 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 5 | terminal_return | 44 | 0.00168516 | 1799 | 0.00258256 | -0.000897394 | -0.0128707 | 0.011076 | 0.883211 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 10 | max_down | 44 | -0.0357247 | 1794 | -0.0371591 | 0.00143434 | -0.00750466 | 0.0103733 | 0.753143 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 10 | max_up | 44 | 0.0388621 | 1794 | 0.0343691 | 0.00449295 | -0.00979487 | 0.0187808 | 0.537669 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 10 | terminal_return | 44 | 0.00727005 | 1794 | 0.00480338 | 0.00246667 | -0.0115932 | 0.0165266 | 0.73095 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 20 | max_down | 44 | -0.0563062 | 1784 | -0.0540262 | -0.00227996 | -0.0164429 | 0.011883 | 0.752366 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 20 | max_up | 44 | 0.0556327 | 1784 | 0.0513797 | 0.00425296 | -0.0145453 | 0.0230512 | 0.657451 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 国证2000 | 20 | terminal_return | 44 | 0.0130361 | 1784 | 0.00924268 | 0.00379342 | -0.019791 | 0.0273778 | 0.752568 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 5 | max_down | 44 | -0.0245429 | 1799 | -0.024919 | 0.000376107 | -0.00692403 | 0.00767624 | 0.919566 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 5 | max_up | 44 | 0.0273774 | 1799 | 0.0225622 | 0.00481518 | -0.0102154 | 0.0198458 | 0.530068 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 5 | terminal_return | 44 | 0.00170579 | 1799 | 0.00215516 | -0.000449364 | -0.0126727 | 0.011774 | 0.942558 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 10 | max_down | 44 | -0.0350117 | 1794 | -0.0365939 | 0.00158217 | -0.00681604 | 0.00998037 | 0.711941 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 10 | max_up | 44 | 0.0382468 | 1794 | 0.0337769 | 0.00446991 | -0.0100589 | 0.0189987 | 0.546501 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 10 | terminal_return | 44 | 0.00679288 | 1794 | 0.00398142 | 0.00281145 | -0.0111837 | 0.0168066 | 0.693773 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 20 | max_down | 44 | -0.0549302 | 1784 | -0.0529698 | -0.00196034 | -0.0149968 | 0.0110761 | 0.768198 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 20 | max_up | 44 | 0.0550085 | 1784 | 0.0504913 | 0.00451718 | -0.0146604 | 0.0236948 | 0.64432 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证1000 | 20 | terminal_return | 44 | 0.013054 | 1784 | 0.00764338 | 0.00541066 | -0.018411 | 0.0292323 | 0.656191 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 5 | max_down | 44 | -0.0203768 | 1799 | -0.0180652 | -0.00231155 | -0.00785796 | 0.00323486 | 0.414008 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 5 | max_up | 44 | 0.0198 | 1799 | 0.018254 | 0.001546 | -0.010606 | 0.013698 | 0.803087 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 5 | terminal_return | 44 | -0.000899201 | 1799 | 0.00162676 | -0.00252596 | -0.0114992 | 0.00644733 | 0.581129 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 10 | max_down | 44 | -0.0276804 | 1794 | -0.0259855 | -0.00169487 | -0.00860806 | 0.00521833 | 0.630856 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 10 | max_up | 44 | 0.027637 | 1794 | 0.0273953 | 0.00024165 | -0.0112606 | 0.0117439 | 0.967154 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 10 | terminal_return | 44 | 0.00126113 | 1794 | 0.00305306 | -0.00179193 | -0.0120568 | 0.00847298 | 0.732234 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 20 | max_down | 44 | -0.0411687 | 1784 | -0.0366521 | -0.0045166 | -0.0140496 | 0.00501641 | 0.353086 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 20 | max_up | 44 | 0.0398628 | 1784 | 0.0413832 | -0.00152035 | -0.0145027 | 0.011462 | 0.818454 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 沪深300 | 20 | terminal_return | 44 | 0.00393684 | 1784 | 0.00584582 | -0.00190898 | -0.0167255 | 0.0129075 | 0.800632 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证500 | 5 | max_down | 44 | -0.0219562 | 1799 | -0.0216435 | -0.000312717 | -0.00706892 | 0.00644349 | 0.927715 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证500 | 5 | max_up | 44 | 0.0252189 | 1799 | 0.0209274 | 0.00429145 | -0.00996867 | 0.0185516 | 0.555296 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证500 | 5 | terminal_return | 44 | 0.001051 | 1799 | 0.00230787 | -0.00125686 | -0.0125829 | 0.0100692 | 0.827816 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证500 | 10 | max_down | 44 | -0.0304701 | 1794 | -0.0313732 | 0.000903102 | -0.00662927 | 0.00843548 | 0.814212 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证500 | 10 | max_up | 44 | 0.0345361 | 1794 | 0.0313023 | 0.00323376 | -0.010473 | 0.0169405 | 0.643786 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证500 | 10 | terminal_return | 44 | 0.00571679 | 1794 | 0.00432199 | 0.00139479 | -0.010601 | 0.0133906 | 0.819726 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证500 | 20 | max_down | 44 | -0.0478036 | 1784 | -0.0449187 | -0.00288488 | -0.0138333 | 0.00806356 | 0.605537 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证500 | 20 | max_up | 44 | 0.0491391 | 1784 | 0.0474932 | 0.00164591 | -0.0153047 | 0.0185966 | 0.849061 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 中证500 | 20 | terminal_return | 44 | 0.0099605 | 1784 | 0.00848279 | 0.00147771 | -0.0187963 | 0.0217517 | 0.886402 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 5 | max_down | 44 | -0.0272304 | 1799 | -0.0261454 | -0.00108506 | -0.00980521 | 0.00763509 | 0.807319 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 5 | max_up | 44 | 0.0302195 | 1799 | 0.0277534 | 0.00246608 | -0.0113952 | 0.0163273 | 0.72731 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 5 | terminal_return | 44 | 0.00272641 | 1799 | 0.00699678 | -0.00427036 | -0.016444 | 0.0079033 | 0.491741 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 10 | max_down | 44 | -0.0395788 | 1794 | -0.0381697 | -0.00140908 | -0.0140148 | 0.0111966 | 0.82658 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 10 | max_up | 44 | 0.0451782 | 1794 | 0.0431223 | 0.00205589 | -0.0116778 | 0.0157896 | 0.769212 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 10 | terminal_return | 44 | 0.0106609 | 1794 | 0.0136032 | -0.00294232 | -0.0197056 | 0.0138209 | 0.730829 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 20 | max_down | 44 | -0.0597872 | 1784 | -0.0552515 | -0.00453574 | -0.0240161 | 0.0149446 | 0.648131 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 20 | max_up | 44 | 0.0669538 | 1784 | 0.0665901 | 0.000363701 | -0.0196858 | 0.0204132 | 0.971637 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 微盘股 | 20 | terminal_return | 44 | 0.0262006 | 1784 | 0.0265387 | -0.000338092 | -0.0269678 | 0.0262916 | 0.980147 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 5 | max_down | 44 | -0.0179285 | 1799 | -0.0162938 | -0.00163471 | -0.00678823 | 0.00351881 | 0.534127 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 5 | max_up | 44 | 0.0171915 | 1799 | 0.0155943 | 0.00159727 | -0.00897444 | 0.012169 | 0.767127 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 5 | terminal_return | 44 | -0.00106387 | 1799 | 0.00159134 | -0.00265521 | -0.01022 | 0.00490956 | 0.491482 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 10 | max_down | 44 | -0.0244244 | 1794 | -0.023443 | -0.00098133 | -0.00748715 | 0.00552449 | 0.767502 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 10 | max_up | 44 | 0.0242179 | 1794 | 0.0233535 | 0.000864321 | -0.00918595 | 0.0109146 | 0.866143 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 10 | terminal_return | 44 | 0.00171962 | 1794 | 0.00295244 | -0.00123281 | -0.0105685 | 0.00810287 | 0.79577 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 20 | max_down | 44 | -0.0379776 | 1784 | -0.0334075 | -0.00457011 | -0.0139562 | 0.00481602 | 0.33992 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 20 | max_up | 44 | 0.0350927 | 1784 | 0.0350435 | 4.92137e-05 | -0.0113761 | 0.0114745 | 0.993264 | 0.993264 | 0.999178 | true |
| ml_today_shallow_gbdt | top | onset | 上证指数 | 20 | terminal_return | 44 | 0.00295165 | 1784 | 0.00563429 | -0.00268264 | -0.0157783 | 0.010413 | 0.688048 | 0.993264 | 0.999178 | true |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `ml_today_elastic_net/bottom/all_a_ml_today_walk_forward_v1/capped_confirmation`：7 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `ml_today_elastic_net/bottom/all_a_ml_today_walk_forward_v1/onset`：5 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为正；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `ml_today_elastic_net/top/all_a_ml_today_walk_forward_v1/capped_confirmation`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
- `ml_today_elastic_net/top/all_a_ml_today_walk_forward_v1/onset`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
- `ml_today_shallow_gbdt/bottom/all_a_ml_today_walk_forward_v1/capped_confirmation`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
- `ml_today_shallow_gbdt/bottom/all_a_ml_today_walk_forward_v1/onset`：3 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。
- `ml_today_shallow_gbdt/top/all_a_ml_today_walk_forward_v1/capped_confirmation`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
- `ml_today_shallow_gbdt/top/all_a_ml_today_walk_forward_v1/onset`：63 项合格检验均未达到名义 p<0.05，因此也没有全局 FDR 发现。
