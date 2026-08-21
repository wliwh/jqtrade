# 信号后 OHLC 结果评测

- 评测版本：`breadth_price_divergence_v1_20120104_20260814__stage_d_v1`
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
| 上证指数 | capped_confirmation | 5 | 140 | 140 | 140 |
| 上证指数 | capped_confirmation | 10 | 140 | 140 | 140 |
| 上证指数 | capped_confirmation | 20 | 140 | 140 | 140 |
| 上证指数 | onset | 5 | 140 | 140 | 140 |
| 上证指数 | onset | 10 | 140 | 140 | 140 |
| 上证指数 | onset | 20 | 140 | 140 | 140 |
| 中证1000 | capped_confirmation | 5 | 140 | 140 | 140 |
| 中证1000 | capped_confirmation | 10 | 140 | 140 | 140 |
| 中证1000 | capped_confirmation | 20 | 140 | 140 | 140 |
| 中证1000 | onset | 5 | 140 | 140 | 140 |
| 中证1000 | onset | 10 | 140 | 140 | 140 |
| 中证1000 | onset | 20 | 140 | 140 | 140 |
| 中证500 | capped_confirmation | 5 | 140 | 140 | 140 |
| 中证500 | capped_confirmation | 10 | 140 | 140 | 140 |
| 中证500 | capped_confirmation | 20 | 140 | 140 | 140 |
| 中证500 | onset | 5 | 140 | 140 | 140 |
| 中证500 | onset | 10 | 140 | 140 | 140 |
| 中证500 | onset | 20 | 140 | 140 | 140 |
| 全A | capped_confirmation | 5 | 140 | 140 | 140 |
| 全A | capped_confirmation | 10 | 140 | 140 | 140 |
| 全A | capped_confirmation | 20 | 140 | 140 | 140 |
| 全A | onset | 5 | 140 | 140 | 140 |
| 全A | onset | 10 | 140 | 140 | 140 |
| 全A | onset | 20 | 140 | 140 | 140 |
| 国证2000 | capped_confirmation | 5 | 140 | 140 | 140 |
| 国证2000 | capped_confirmation | 10 | 140 | 140 | 140 |
| 国证2000 | capped_confirmation | 20 | 140 | 140 | 140 |
| 国证2000 | onset | 5 | 140 | 140 | 140 |
| 国证2000 | onset | 10 | 140 | 140 | 140 |
| 国证2000 | onset | 20 | 140 | 140 | 140 |
| 微盘股 | capped_confirmation | 5 | 140 | 140 | 140 |
| 微盘股 | capped_confirmation | 10 | 140 | 140 | 140 |
| 微盘股 | capped_confirmation | 20 | 140 | 140 | 140 |
| 微盘股 | onset | 5 | 140 | 140 | 140 |
| 微盘股 | onset | 10 | 140 | 140 | 140 |
| 微盘股 | onset | 20 | 140 | 140 | 140 |
| 沪深300 | capped_confirmation | 5 | 140 | 140 | 140 |
| 沪深300 | capped_confirmation | 10 | 140 | 140 | 140 |
| 沪深300 | capped_confirmation | 20 | 140 | 140 | 140 |
| 沪深300 | onset | 5 | 140 | 140 | 140 |
| 沪深300 | onset | 10 | 140 | 140 | 140 |
| 沪深300 | onset | 20 | 140 | 140 | 140 |

## 描述统计与推断

| signal_id | direction | event_kind | index_name | horizon | outcome_name | event_count | event_mean | baseline_count | baseline_mean | mean_difference | ci95_lower | ci95_upper | hac_p_value | local_fdr_q_value | global_fdr_q_value | inference_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 5 | max_down | 140 | -0.0212219 | 3402 | -0.0220627 | 0.000840773 | -0.00323771 | 0.00491926 | 0.686176 | 0.882226 | 0.864581 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 5 | max_up | 140 | 0.0167777 | 3402 | 0.0197411 | -0.00296335 | -0.00613742 | 0.000210716 | 0.0672676 | 0.353155 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 5 | terminal_return | 140 | 0.000134766 | 3402 | 0.00177187 | -0.0016371 | -0.00693599 | 0.00366179 | 0.544817 | 0.780079 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 10 | max_down | 140 | -0.0319159 | 3397 | -0.0317572 | -0.000158669 | -0.00675056 | 0.00643322 | 0.962372 | 0.996603 | 0.983046 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 10 | max_up | 140 | 0.0257706 | 3397 | 0.0294563 | -0.00368572 | -0.00914351 | 0.00177207 | 0.185631 | 0.508468 | 0.49765 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 10 | terminal_return | 140 | -0.000344311 | 3397 | 0.00352847 | -0.00387278 | -0.0130515 | 0.00530593 | 0.408246 | 0.650697 | 0.703456 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 20 | max_down | 140 | -0.050065 | 3387 | -0.0450128 | -0.00505218 | -0.0155772 | 0.00547288 | 0.346793 | 0.650697 | 0.682644 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 20 | max_up | 140 | 0.0365863 | 3387 | 0.0446719 | -0.00808565 | -0.0172963 | 0.00112503 | 0.0853238 | 0.378085 | 0.362829 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 全A | 20 | terminal_return | 140 | -0.00534042 | 3387 | 0.00714745 | -0.0124879 | -0.0269277 | 0.001952 | 0.090067 | 0.378085 | 0.366079 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 5 | max_down | 140 | -0.0246585 | 3404 | -0.0268241 | 0.00216561 | -0.00265725 | 0.00698847 | 0.378805 | 0.650697 | 0.69173 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 5 | max_up | 140 | 0.0187854 | 3404 | 0.0236663 | -0.00488089 | -0.00841266 | -0.00134911 | 0.00675461 | 0.193133 | 0.30137 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 5 | terminal_return | 140 | -1.11133e-05 | 3404 | 0.00276333 | -0.00277444 | -0.00890786 | 0.00335897 | 0.375292 | 0.650697 | 0.69173 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 10 | max_down | 140 | -0.0388279 | 3399 | -0.0390009 | 0.000172925 | -0.00778428 | 0.00813013 | 0.966025 | 0.996603 | 0.983046 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 10 | max_up | 140 | 0.0292988 | 3399 | 0.0362128 | -0.00691397 | -0.0134636 | -0.00036439 | 0.0385418 | 0.269793 | 0.323751 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 10 | terminal_return | 140 | -0.00123621 | 3399 | 0.00556321 | -0.00679942 | -0.0178564 | 0.00425756 | 0.228092 | 0.556388 | 0.556169 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 20 | max_down | 140 | -0.0620232 | 3389 | -0.0557582 | -0.00626492 | -0.0194652 | 0.00693539 | 0.352255 | 0.650697 | 0.682644 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 20 | max_up | 140 | 0.0447727 | 3389 | 0.0554887 | -0.010716 | -0.0233347 | 0.00190276 | 0.0960215 | 0.378085 | 0.366628 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 国证2000 | 20 | terminal_return | 140 | -0.00669029 | 3389 | 0.0112714 | -0.0179617 | -0.0365723 | 0.000648958 | 0.0585371 | 0.339841 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 5 | max_down | 140 | -0.0252252 | 3404 | -0.0272328 | 0.00200759 | -0.00280055 | 0.00681574 | 0.413141 | 0.650697 | 0.703456 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 5 | max_up | 140 | 0.0187865 | 3404 | 0.0237584 | -0.00497191 | -0.00836764 | -0.00157617 | 0.00410786 | 0.193133 | 0.30137 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 5 | terminal_return | 140 | -0.000870303 | 3404 | 0.00213663 | -0.00300694 | -0.00911408 | 0.0031002 | 0.334528 | 0.650697 | 0.682644 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 10 | max_down | 140 | -0.0396651 | 3399 | -0.0396491 | -1.60275e-05 | -0.00801636 | 0.0079843 | 0.996867 | 0.996867 | 0.996867 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 10 | max_up | 140 | 0.0291126 | 3399 | 0.0358806 | -0.00676802 | -0.0131183 | -0.000417713 | 0.0367144 | 0.269793 | 0.323751 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 10 | terminal_return | 140 | -0.00265902 | 3399 | 0.00427724 | -0.00693626 | -0.01787 | 0.00399745 | 0.213717 | 0.556388 | 0.549559 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 20 | max_down | 140 | -0.0624704 | 3389 | -0.0567751 | -0.00569538 | -0.0188971 | 0.00750637 | 0.397795 | 0.650697 | 0.703456 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 20 | max_up | 140 | 0.0437739 | 3389 | 0.0545115 | -0.0107376 | -0.0229405 | 0.00146534 | 0.0845921 | 0.378085 | 0.362829 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证1000 | 20 | terminal_return | 140 | -0.00899217 | 3389 | 0.00863572 | -0.0176279 | -0.0359505 | 0.000694706 | 0.0593373 | 0.339841 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 5 | max_down | 140 | -0.0200859 | 3404 | -0.020291 | 0.000205106 | -0.00363922 | 0.00404943 | 0.916716 | 0.996603 | 0.980626 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 5 | max_up | 140 | 0.018841 | 3404 | 0.0196137 | -0.000772679 | -0.00449223 | 0.00294687 | 0.683891 | 0.882226 | 0.864581 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 5 | terminal_return | 140 | 0.00105294 | 3404 | 0.00146148 | -0.000408538 | -0.00561228 | 0.00479521 | 0.877707 | 0.996603 | 0.978682 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 10 | max_down | 140 | -0.0288295 | 3399 | -0.0290732 | 0.000243675 | -0.00551736 | 0.00600471 | 0.933929 | 0.996603 | 0.980626 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 10 | max_up | 140 | 0.0283362 | 3399 | 0.0292201 | -0.000883966 | -0.00665474 | 0.00488681 | 0.764 | 0.942028 | 0.925615 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 10 | terminal_return | 140 | 0.0013728 | 3399 | 0.00288586 | -0.00151306 | -0.0101212 | 0.00709504 | 0.730461 | 0.920381 | 0.902334 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 20 | max_down | 140 | -0.0449088 | 3389 | -0.0407923 | -0.00411645 | -0.0129574 | 0.00472446 | 0.361452 | 0.650697 | 0.682644 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 20 | max_up | 140 | 0.0399852 | 3389 | 0.0441599 | -0.00417475 | -0.0131697 | 0.00482022 | 0.362993 | 0.650697 | 0.682644 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 沪深300 | 20 | terminal_return | 140 | -0.00211793 | 3389 | 0.00590566 | -0.00802359 | -0.0211142 | 0.00506702 | 0.22962 | 0.556388 | 0.556169 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 5 | max_down | 140 | -0.0229104 | 3404 | -0.0245845 | 0.00167408 | -0.00281582 | 0.00616398 | 0.464904 | 0.697357 | 0.760753 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 5 | max_up | 140 | 0.0185483 | 3404 | 0.0221194 | -0.00357111 | -0.00688688 | -0.000255331 | 0.0347784 | 0.269793 | 0.323751 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 5 | terminal_return | 140 | 0.000424796 | 3404 | 0.00204246 | -0.00161766 | -0.00736555 | 0.00413023 | 0.581214 | 0.79601 | 0.822842 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 10 | max_down | 140 | -0.0352113 | 3399 | -0.0354771 | 0.000265793 | -0.00692648 | 0.00745807 | 0.942258 | 0.996603 | 0.981194 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 10 | max_up | 140 | 0.0287193 | 3399 | 0.0330746 | -0.00435529 | -0.010342 | 0.00163139 | 0.153899 | 0.503501 | 0.461697 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 10 | terminal_return | 140 | -0.000283404 | 3399 | 0.00407984 | -0.00436325 | -0.0143242 | 0.00559766 | 0.390587 | 0.650697 | 0.703057 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 20 | max_down | 140 | -0.0542382 | 3389 | -0.0503546 | -0.00388366 | -0.0154818 | 0.00771449 | 0.511625 | 0.74959 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 20 | max_up | 140 | 0.0422125 | 3389 | 0.0502492 | -0.00803676 | -0.0187573 | 0.00268374 | 0.141741 | 0.496092 | 0.446483 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 中证500 | 20 | terminal_return | 140 | -0.00518988 | 3389 | 0.00826961 | -0.0134595 | -0.0297003 | 0.00278136 | 0.104304 | 0.386539 | 0.386539 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 5 | max_down | 140 | -0.0243524 | 3404 | -0.0278567 | 0.00350428 | -0.00163699 | 0.00864556 | 0.181572 | 0.508468 | 0.497348 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 5 | max_up | 140 | 0.0215906 | 3404 | 0.0263798 | -0.00478918 | -0.00853716 | -0.00104121 | 0.0122624 | 0.193133 | 0.30137 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 5 | terminal_return | 140 | 0.00193453 | 3404 | 0.0045214 | -0.00258688 | -0.00906735 | 0.0038936 | 0.433984 | 0.666853 | 0.729092 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 10 | max_down | 140 | -0.0399912 | 3399 | -0.0404032 | 0.000411944 | -0.00811444 | 0.00893832 | 0.924557 | 0.996603 | 0.980626 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 10 | max_up | 140 | 0.0333203 | 3399 | 0.0405512 | -0.00723089 | -0.0136838 | -0.000777975 | 0.0280704 | 0.269793 | 0.321534 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 10 | terminal_return | 140 | 0.00101693 | 3399 | 0.00909849 | -0.00808155 | -0.0194095 | 0.0032464 | 0.162024 | 0.503501 | 0.463977 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 20 | max_down | 140 | -0.0634619 | 3389 | -0.0577938 | -0.00566806 | -0.0189321 | 0.00759596 | 0.402278 | 0.650697 | 0.703456 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 20 | max_up | 140 | 0.0486767 | 3389 | 0.0624559 | -0.0137792 | -0.0251714 | -0.00238704 | 0.0177551 | 0.223715 | 0.30137 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 微盘股 | 20 | terminal_return | 140 | -0.0044535 | 3389 | 0.0181166 | -0.0225701 | -0.0398605 | -0.00527969 | 0.0105127 | 0.193133 | 0.30137 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 5 | max_down | 140 | -0.0187744 | 3404 | -0.0189443 | 0.000169914 | -0.00350981 | 0.00384964 | 0.927886 | 0.996603 | 0.980626 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 5 | max_up | 140 | 0.0162475 | 3404 | 0.0171835 | -0.000936031 | -0.00410069 | 0.00222863 | 0.562102 | 0.786943 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 5 | terminal_return | 140 | 0.000679404 | 3404 | 0.00124877 | -0.000569368 | -0.00532989 | 0.00419115 | 0.814659 | 0.968368 | 0.959318 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 10 | max_down | 140 | -0.0270875 | 3399 | -0.0271579 | 7.04504e-05 | -0.00566234 | 0.00580324 | 0.980784 | 0.996603 | 0.98863 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 10 | max_up | 140 | 0.0249483 | 3399 | 0.0256868 | -0.000738594 | -0.00586269 | 0.0043855 | 0.777547 | 0.942028 | 0.933057 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 10 | terminal_return | 140 | 0.000759997 | 3399 | 0.0024581 | -0.0016981 | -0.00968038 | 0.00628417 | 0.676708 | 0.882226 | 0.864581 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 20 | max_down | 140 | -0.0425514 | 3389 | -0.0383339 | -0.00421759 | -0.01313 | 0.00469483 | 0.353655 | 0.650697 | 0.682644 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 20 | max_up | 140 | 0.0345916 | 3389 | 0.0388853 | -0.00429373 | -0.0125553 | 0.00396782 | 0.308364 | 0.650697 | 0.658541 | true |
| all_a_ma20_breadth_price_divergence_top | top | capped_confirmation | 上证指数 | 20 | terminal_return | 140 | -0.00371577 | 3389 | 0.00501484 | -0.00873061 | -0.0211378 | 0.0036766 | 0.167834 | 0.503501 | 0.469934 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 5 | max_down | 140 | -0.0189592 | 3402 | -0.0221558 | 0.00319659 | -0.000457151 | 0.00685032 | 0.086388 | 0.340153 | 0.362829 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 5 | max_up | 140 | 0.0170778 | 3402 | 0.0197287 | -0.00265088 | -0.00575988 | 0.000458122 | 0.0946847 | 0.35089 | 0.366628 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 5 | terminal_return | 140 | 0.000211723 | 3402 | 0.0017687 | -0.00155698 | -0.00680591 | 0.00369196 | 0.560978 | 0.817523 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 10 | max_down | 140 | -0.0306847 | 3397 | -0.031808 | 0.00112329 | -0.00519003 | 0.0074366 | 0.727292 | 0.881142 | 0.902334 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 10 | max_up | 140 | 0.026699 | 3397 | 0.029418 | -0.00271902 | -0.00830016 | 0.00286212 | 0.339642 | 0.629336 | 0.682644 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 10 | terminal_return | 140 | 0.00128933 | 3397 | 0.00346114 | -0.00217181 | -0.0103684 | 0.00602477 | 0.60353 | 0.817523 | 0.833994 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 20 | max_down | 140 | -0.0482422 | 3387 | -0.0450882 | -0.00315406 | -0.0139416 | 0.00763349 | 0.566602 | 0.817523 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 20 | max_up | 140 | 0.0377604 | 3387 | 0.0446234 | -0.00686303 | -0.0164421 | 0.00271605 | 0.160241 | 0.420633 | 0.463977 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 全A | 20 | terminal_return | 140 | -0.00511564 | 3387 | 0.00713815 | -0.0122538 | -0.0276894 | 0.00318183 | 0.119714 | 0.3917 | 0.421612 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 5 | max_down | 140 | -0.0226105 | 3404 | -0.0269083 | 0.0042978 | -0.000127988 | 0.00872358 | 0.0569989 | 0.313354 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 5 | max_up | 140 | 0.0192814 | 3404 | 0.0236459 | -0.00436451 | -0.00808156 | -0.00064745 | 0.021369 | 0.313354 | 0.30137 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 5 | terminal_return | 140 | -0.000269821 | 3404 | 0.00277397 | -0.00304379 | -0.00920734 | 0.00311976 | 0.333084 | 0.629336 | 0.682644 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 10 | max_down | 140 | -0.0381701 | 3399 | -0.039028 | 0.0008579 | -0.00688797 | 0.00860377 | 0.828145 | 0.920413 | 0.966169 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 10 | max_up | 140 | 0.0299888 | 3399 | 0.0361844 | -0.0061956 | -0.0130068 | 0.000615554 | 0.0746081 | 0.313354 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 10 | terminal_return | 140 | -0.000240095 | 3399 | 0.00552218 | -0.00576228 | -0.0158361 | 0.00431153 | 0.262232 | 0.550687 | 0.590022 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 20 | max_down | 140 | -0.0600983 | 3389 | -0.0558378 | -0.00426058 | -0.0176256 | 0.00910449 | 0.532091 | 0.817523 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 20 | max_up | 140 | 0.0456213 | 3389 | 0.0554536 | -0.00983234 | -0.0231948 | 0.00353014 | 0.149246 | 0.408803 | 0.458658 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 国证2000 | 20 | terminal_return | 140 | -0.00672809 | 3389 | 0.011273 | -0.018001 | -0.0374923 | 0.00149022 | 0.0702731 | 0.313354 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 5 | max_down | 140 | -0.0229687 | 3404 | -0.0273256 | 0.00435684 | -8.34402e-05 | 0.00879712 | 0.0544589 | 0.313354 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 5 | max_up | 140 | 0.0191951 | 3404 | 0.0237416 | -0.00454648 | -0.00815896 | -0.000933995 | 0.0136345 | 0.313354 | 0.30137 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 5 | terminal_return | 140 | -0.00104824 | 3404 | 0.00214395 | -0.00319219 | -0.00932993 | 0.00294555 | 0.308022 | 0.606418 | 0.658541 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 10 | max_down | 140 | -0.0386336 | 3399 | -0.0396916 | 0.00105803 | -0.00674594 | 0.00886201 | 0.790448 | 0.920413 | 0.939589 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 10 | max_up | 140 | 0.0297996 | 3399 | 0.0358523 | -0.00605273 | -0.0126412 | 0.000535785 | 0.0717643 | 0.313354 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 10 | terminal_return | 140 | -0.00165384 | 3399 | 0.00423584 | -0.00588968 | -0.0158499 | 0.00407056 | 0.246462 | 0.541674 | 0.57122 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 20 | max_down | 140 | -0.0604156 | 3389 | -0.0568599 | -0.00355565 | -0.0170725 | 0.00996124 | 0.606146 | 0.817523 | 0.833994 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 20 | max_up | 140 | 0.0445369 | 3389 | 0.05448 | -0.00994304 | -0.0228525 | 0.0029664 | 0.13114 | 0.391935 | 0.434834 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证1000 | 20 | terminal_return | 140 | -0.00917211 | 3389 | 0.00864315 | -0.0178153 | -0.037111 | 0.00148045 | 0.0703547 | 0.313354 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 5 | max_down | 140 | -0.0178801 | 3404 | -0.0203818 | 0.0025017 | -0.000794519 | 0.00579792 | 0.136866 | 0.391935 | 0.442183 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 5 | max_up | 140 | 0.0192232 | 3404 | 0.0195979 | -0.00037473 | -0.00408956 | 0.0033401 | 0.84327 | 0.920413 | 0.967404 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 5 | terminal_return | 140 | 0.00134068 | 3404 | 0.00144964 | -0.00010896 | -0.0053412 | 0.00512328 | 0.967442 | 0.967442 | 0.983046 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 10 | max_down | 140 | -0.0274223 | 3399 | -0.0291312 | 0.00170889 | -0.00373048 | 0.00714827 | 0.538043 | 0.817523 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 10 | max_up | 140 | 0.0295053 | 3399 | 0.029172 | 0.000333328 | -0.00556902 | 0.00623567 | 0.911863 | 0.948489 | 0.980626 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 10 | terminal_return | 140 | 0.00351275 | 3399 | 0.00279771 | 0.000715034 | -0.00697116 | 0.00840123 | 0.855319 | 0.920413 | 0.96972 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 20 | max_down | 140 | -0.043114 | 3389 | -0.0408665 | -0.00224753 | -0.0112051 | 0.00671005 | 0.622875 | 0.817523 | 0.834917 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 20 | max_up | 140 | 0.0416831 | 3389 | 0.0440898 | -0.00240665 | -0.0116273 | 0.00681395 | 0.608948 | 0.817523 | 0.833994 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 沪深300 | 20 | terminal_return | 140 | -0.00177491 | 3389 | 0.00589149 | -0.0076664 | -0.0216103 | 0.0062775 | 0.281206 | 0.571483 | 0.621613 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 5 | max_down | 140 | -0.0207924 | 3404 | -0.0246716 | 0.00387917 | -0.000192144 | 0.00795049 | 0.061832 | 0.313354 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 5 | max_up | 140 | 0.0186122 | 3404 | 0.0221168 | -0.00350467 | -0.00677147 | -0.000237871 | 0.0354905 | 0.313354 | 0.323751 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 5 | terminal_return | 140 | -1.14934e-05 | 3404 | 0.0020604 | -0.00207189 | -0.00773562 | 0.00359183 | 0.473372 | 0.817523 | 0.764678 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 10 | max_down | 140 | -0.0340014 | 3399 | -0.0355269 | 0.00152551 | -0.00539123 | 0.00844225 | 0.665534 | 0.822131 | 0.864509 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 10 | max_up | 140 | 0.0293262 | 3399 | 0.0330496 | -0.00372334 | -0.00985455 | 0.00240788 | 0.233944 | 0.541674 | 0.556169 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 10 | terminal_return | 140 | 0.00107527 | 3399 | 0.00402388 | -0.00294861 | -0.0119216 | 0.00602438 | 0.519527 | 0.817523 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 20 | max_down | 140 | -0.0524201 | 3389 | -0.0504297 | -0.00199041 | -0.0139605 | 0.00997967 | 0.74449 | 0.88496 | 0.910736 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 20 | max_up | 140 | 0.0432119 | 3389 | 0.050208 | -0.00699608 | -0.0183113 | 0.00431918 | 0.225573 | 0.541674 | 0.556169 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 中证500 | 20 | terminal_return | 140 | -0.00532864 | 3389 | 0.00827534 | -0.013604 | -0.0309545 | 0.00374655 | 0.124349 | 0.3917 | 0.423459 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 5 | max_down | 140 | -0.0225745 | 3404 | -0.0279298 | 0.00535529 | 0.000631306 | 0.0100793 | 0.0262879 | 0.313354 | 0.321534 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 5 | max_up | 140 | 0.0218743 | 3404 | 0.0263681 | -0.0044938 | -0.00830579 | -0.000681824 | 0.0208565 | 0.313354 | 0.30137 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 5 | terminal_return | 140 | 0.00201113 | 3404 | 0.00451825 | -0.00250712 | -0.00894406 | 0.00392982 | 0.445225 | 0.801405 | 0.738136 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 10 | max_down | 140 | -0.0386013 | 3399 | -0.0404604 | 0.00185908 | -0.00649101 | 0.0102092 | 0.662562 | 0.822131 | 0.864509 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 10 | max_up | 140 | 0.0344394 | 3399 | 0.0405051 | -0.00606572 | -0.0126728 | 0.000541367 | 0.071955 | 0.313354 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 10 | terminal_return | 140 | 0.00236911 | 3399 | 0.00904279 | -0.00667368 | -0.0171449 | 0.00379757 | 0.211602 | 0.533236 | 0.549559 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 20 | max_down | 140 | -0.0610112 | 3389 | -0.0578951 | -0.00311612 | -0.0166662 | 0.010434 | 0.652176 | 0.822131 | 0.864509 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 20 | max_up | 140 | 0.0507024 | 3389 | 0.0623723 | -0.0116699 | -0.0236239 | 0.000284115 | 0.0556948 | 0.313354 | 0.348171 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 微盘股 | 20 | terminal_return | 140 | -0.00320914 | 3389 | 0.0180652 | -0.0212743 | -0.0394146 | -0.00313403 | 0.0215264 | 0.313354 | 0.30137 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 5 | max_down | 140 | -0.0164926 | 3404 | -0.0190381 | 0.00254554 | -0.00066744 | 0.00575851 | 0.12046 | 0.3917 | 0.421612 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 5 | max_up | 140 | 0.0168413 | 3404 | 0.0171591 | -0.000317783 | -0.00349455 | 0.00285898 | 0.844559 | 0.920413 | 0.967404 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 5 | terminal_return | 140 | 0.00142126 | 3404 | 0.00121826 | 0.000202995 | -0.00456047 | 0.00496646 | 0.933434 | 0.948489 | 0.980626 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 10 | max_down | 140 | -0.0256579 | 3399 | -0.0272168 | 0.00155893 | -0.0038125 | 0.00693037 | 0.569461 | 0.817523 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 10 | max_up | 140 | 0.0261008 | 3399 | 0.0256394 | 0.000461402 | -0.00474011 | 0.00566292 | 0.861974 | 0.920413 | 0.96972 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 10 | terminal_return | 140 | 0.00271249 | 3399 | 0.00237768 | 0.000334803 | -0.00663146 | 0.00730106 | 0.924951 | 0.948489 | 0.980626 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 20 | max_down | 140 | -0.0406589 | 3389 | -0.038412 | -0.00224689 | -0.011188 | 0.00669418 | 0.622332 | 0.817523 | 0.834917 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 20 | max_up | 140 | 0.0361306 | 3389 | 0.0388217 | -0.00269114 | -0.0112567 | 0.00587443 | 0.538031 | 0.817523 | 0.815365 | true |
| all_a_ma20_breadth_price_divergence_top | top | onset | 上证指数 | 20 | terminal_return | 140 | -0.00283571 | 3389 | 0.00497849 | -0.0078142 | -0.0211098 | 0.00548138 | 0.249342 | 0.541674 | 0.57122 | true |

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

- `all_a_ma20_breadth_price_divergence_top/top/breadth_price_divergence_v1_20120104_20260814/capped_confirmation`：9 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为负；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
- `all_a_ma20_breadth_price_divergence_top/top/breadth_price_divergence_v1_20120104_20260814/onset`：6 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。 最长 20 日 terminal 均值差在 7/7 个指数均为负；这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。
