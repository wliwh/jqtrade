# ETF 旧策略归档

归档日期：2026-08-15

这里保存 ETF 轮动策略整理前除经典 `ETF_wy03.py` 和最新三份策略之外的历史实现。文件没有删除；归档只表示不再作为当前入口，不代表策略无研究价值。

## 目录

- [`strategies/`](strategies/)：旧策略、模块化变体和优化变体。
- [`strategy_notes.md`](strategy_notes.md)：原 `ETF_summ.md`，记录各代策略逻辑与指标研究。

## 已归档策略

```text
ETF_7star.py
ETF_Zopt.py
ETF_acc.py
ETF_atr.py
ETF_atr_modular.py
ETF_equalweight.py
ETF_gao.py
ETF_gao_dynamic.py
ETF_gao_modular.py
ETF_gao_opt.py
ETF_lex.py
ETF_long.py
ETF_long_modular.py
ETF_long_opt.py
ETF_modular.py
ETF_std_score.py
ETF_vol.py
ETF_wy03_opt.py
ETF_yj15.py
ETF_yj15_modular.py
```

与 Gao 策略配套的 YAML 已移到 [`../../../backtest_executor/archive/config/`](../../../backtest_executor/archive/config/)。历史文章仍保留原策略名，以维持研究叙事；需要复现时使用本目录中的实际路径。
