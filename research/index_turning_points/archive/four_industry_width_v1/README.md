# 四行业宽度 V1 归档

状态：只读历史实验；样本 `2021-12-13—2026-08-14`，不代表现行区域评测结论。旧实验使用单点顶底和极值前窗口，未表示 W/M 区域，也未区分预测与确认。

- [`results/four_industry_top1/`](results/four_industry_top1/)：Top1 与旧单点顶底关系；
- [`results/four_industry_forward_returns/`](results/four_industry_forward_returns/)：活跃、首次触发、持续和退出后的收益；
- [`code/`](code/)：复现代码，默认输出到 `/tmp/jqtrade_four_industry_width_v1/`。

全局 FDR 后无显著项，行业方向不一致；数据只有国证2000 `399303` 代理，不能形成交易结论。旧 manifest 路径原样保留。现役代码只能经 [`legacy_four_industry_v1.py`](../../adapters/legacy_four_industry_v1.py) 读取，不得改写 `results/`。
