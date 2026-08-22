P1 runtime environment
item                                              value
          research_id               momentum_signal_p1_multi_universe_v2
          source_path  research/momentum_signal_validation/p1_jq_sign...
   runtime_logic_hash  800fd99dc6304baf852d2d4e3545b48a8326139a3e3128...
       python_version  3.6.7 |Anaconda, Inc.| (default, Oct 23 2018, ...
       pandas_version                                             0.23.4
        numpy_version                                             1.14.6
      jqdata_imported                                               True
     get_price_source                                    python_builtins
get_industries_source                                     module_globals
             run_time                                2026-08-22 09:28:05
[P1   0% | 00:00:00] started
[P1   4% | 00:00:01] broad data ready: 2725 dates x 10 assets
  sw_l1 prices: 10/38
  sw_l1 prices: 20/38
  sw_l1 prices: 30/38
  sw_l1 prices: 38/38
[P1   8% | 00:00:04] industry_sw_l1 data ready: 2725 dates x 38 assets
[P1  12% | 00:00:04] style data ready: 2725 dates x 12 assets
[P1  16% | 00:00:04] broad coverage checked: 10/10 assets have research history
[P1  20% | 00:00:04] broad signals ready: 32 panels, 5 horizons
[P1  24% | 00:00:13] broad IC grid complete: 160 parameter cells
[P1  28% | 00:11:44] broad quantile diagnostics complete
[P1  32% | 00:12:18] broad Top-K diagnostics complete
[P1  36% | 00:22:26] broad R2 double-sort complete
[P1  40% | 00:22:26] broad protocol checks complete
[P1  44% | 00:22:26] industry_sw_l1 coverage checked: 32/38 assets have research history
[P1  48% | 00:22:28] industry_sw_l1 signals ready: 32 panels, 5 horizons
[P1  52% | 00:22:38] industry_sw_l1 IC grid complete: 160 parameter cells
[P1  56% | 00:46:37] industry_sw_l1 quantile diagnostics complete
[P1  60% | 00:47:11] industry_sw_l1 Top-K diagnostics complete
[P1  64% | 01:09:34] industry_sw_l1 R2 double-sort complete
[P1  68% | 01:09:34] industry_sw_l1 protocol checks complete
[P1  72% | 01:09:34] style coverage checked: 12/12 assets have research history
[P1  76% | 01:09:35] style signals ready: 32 panels, 5 horizons
[P1  80% | 01:09:43] style IC grid complete: 160 parameter cells
[P1  84% | 01:24:25] style quantile diagnostics complete
[P1  88% | 01:24:59] style Top-K diagnostics complete
[P1  92% | 01:38:17] style R2 double-sort complete
[P1  96% | 01:38:17] style protocol checks complete
[P1 100% | 01:38:17] all universe results combined

P1 frozen primary cell
universe_group       period     n      mean  annualized_icir    t_stat   p_value   q_value
         broad  development  1461  0.088112         2.257277  3.152753  0.001617  0.015222
         broad   validation   484  0.020686         0.629817  0.515882  0.605937  0.835775
         broad   locked_oos   633  0.025305         0.711637  0.647749  0.517148  0.954327
         broad          all  2578  0.060032         1.616205  2.993734  0.002756  0.016331
industry_sw_l1  development  1461  0.026167         1.293470  1.732483  0.083188  0.299402
industry_sw_l1   validation   484 -0.021016        -1.129124 -0.939775  0.347333  0.611406
industry_sw_l1   locked_oos   633 -0.003460        -0.163053 -0.147048  0.883094  0.990861
industry_sw_l1          all  2578  0.010034         0.496320  0.897127  0.369651  0.680065
         style  development  1461  0.067227         1.871726  2.585532  0.009723  0.067637
         style   validation   484  0.076077         2.078609  1.721927  0.085083  0.348257
         style   locked_oos   633  0.037294         0.939002  0.858377  0.390685  0.762311
         style          all  2578  0.061539         1.662847  3.072620  0.002122  0.012751

P1 protocol checks
universe_group                                 check     value  is_gate  passed status  gate_overall
         broad          development_rank_ic_positive  0.088112     True    True   pass         False
         broad           validation_rank_ic_positive  0.020686     True    True   pass         False
         broad  validation_neighbor_plateau_positive  0.020686     True    True   pass         False
         broad  validation_top_minus_bottom_positive -0.000924     True   False   fail         False
         broad       validation_top1_excess_positive -0.001699     True   False   fail         False
         broad           locked_oos_rank_ic_positive  0.025305    False    True   pass         False
industry_sw_l1          development_rank_ic_positive  0.026167     True    True   pass         False
industry_sw_l1           validation_rank_ic_positive -0.021016     True   False   fail         False
industry_sw_l1  validation_neighbor_plateau_positive -0.021016     True   False   fail         False
industry_sw_l1  validation_top_minus_bottom_positive -0.001372     True   False   fail         False
industry_sw_l1       validation_top1_excess_positive  0.000941     True    True   pass         False
industry_sw_l1           locked_oos_rank_ic_positive -0.003460    False   False   fail         False
         style          development_rank_ic_positive  0.067227     True    True   pass          True
         style           validation_rank_ic_positive  0.076077     True    True   pass          True
         style  validation_neighbor_plateau_positive  0.076077     True    True   pass          True
         style  validation_top_minus_bottom_positive  0.003649     True    True   pass          True
         style       validation_top1_excess_positive  0.001816     True    True   pass          True
         style           locked_oos_rank_ic_positive  0.037294    False    True   pass          True

All detailed outputs are available in RESULTS by table name.