# 研究产物

产物按事实类型和版本隔离：

- `ground_truth/<bundle>/`：事后标签、区域、峰瓣和未来结果；
- `signals/<signal_version>/`：严格点时的逐日信号与 episode；
- `evaluations/<evaluation_version>/`：区域定位明细/完整切片/报告、信号后 OHLC 明细/统计/报告及统一 manifest；
- `viewers/<viewer_version>/`：可删除并重建的人工审计 HTML。

现有 [`ground_truth/index_ohlc_20260814/`](ground_truth/index_ohlc_20260814/) 是目录重组前已经生成的 bundle。其 CSV、manifest、哈希和 manifest 内的旧逻辑文件路径均保持原样，代表生成时事实；新 bundle 由 `pipelines/build_ground_truth.py` 写入一个全新目录。

评测 bundle 由 `pipelines/evaluate_signal.py` 写入全新目录，拒绝覆盖非空目录，并要求 TDX OHLC 与 ground-truth manifest 的来源哈希一致。区域与信号后结果不合成总分。

当前阶段 E 基线：

- `signals/four_industry_top1_v2_20211213_20260814/`：V2 JQ 输入生成的四行业 Top1 daily、episode 与 manifest；
- `evaluations/four_industry_top1_v2_20211213_20260814__stage_d_v2/`：当前正式固定格式报告。`stage_d_v1` 是同次运行中报告发现规则增强前的不可变中间版本，指标 CSV 口径未变，现行阅读入口为 v2。
- `signals/single_industry_top1_v1_20170103_20260814/`：从 2017 年开始、按历史存续边界生成的 32 条单行业 Top1 序列；
- `evaluations/single_industry_top1_v1_20170103_20260814__stage_d_v1/`：单行业区域定位与事件后 OHLC 固定格式评测。
