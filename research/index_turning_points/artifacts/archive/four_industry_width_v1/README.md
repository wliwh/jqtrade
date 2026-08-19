# 四行业宽度 V1 结果归档

归档日期：2026-08-19；原始样本：2021-12-13—2026-08-14。

状态：只读历史实验，不代表现行顶底区域评测结论。

## 内容

- [`four_industry_top1/`](four_industry_top1/)：四行业 MA20 宽度 Top1 与单点 directional-change 顶底的提前窗口、连续区间和市场宽度增量分析；原路径为 `artifacts/four_industry_top1/`。
- [`four_industry_forward_returns/`](four_industry_forward_returns/)：全部活跃日、首次触发、持续期和退出日后的多期限收盘收益分析；原路径为 `artifacts/four_industry_forward_returns/`。

历史结果显示，全指数全局 FDR 校正后没有显著项，行业分项方向也不一致。输入未包含中证2000 `932000`，国证2000 `399303` 只是代理，不能据此形成交易或清仓结论。

## 归档原因与复现

旧顶底评测只使用单点极值及其前置窗口，没有表示 W 底、M 顶和顶底区域，也没有把极值前预测与极值后确认分开。因此这些文件只保留为方法演进和结果追溯证据。

复现代码仍位于：

- [`../../../analyze_breadth.py`](../../../analyze_breadth.py)
- [`../../../analyze_forward_returns.py`](../../../analyze_forward_returns.py)

运行旧脚本会在原 `artifacts/four_industry_*` 路径生成新的结果目录，不应覆盖或改写本归档。
