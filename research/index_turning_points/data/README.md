# 输入数据

`inputs/<dataset>/<version>/` 保存从 JQ 等外部环境接收的不可变输入快照。每个版本至少应包含 manifest、数据口径说明和 manifest 声明的数据文件；生成程序属于 `adapters/`，不复制进快照。

当前输入：

- [`all_a_breadth_v1_20120101_20260814`](inputs/all_a_breadth/all_a_breadth_v1_20120101_20260814/jq_breadth_export.md)：当前已验收的均线与行业宽度事实源；
- [`all_a_p1_inputs_v1_20120101_20260814`](inputs/all_a_p1_inputs/all_a_p1_inputs_v1_20120101_20260814/)：文件完整性已通过，但 MA 相等比较受浮点路径影响，只保留追溯，不进入后续研究；
- [`all_a_p1_inputs_v2_20120101_20260814`](inputs/all_a_p1_inputs/all_a_p1_inputs_v2_20120101_20260814/)：当前 P1 共用输入，覆盖 2012-01-04 至 2026-08-14；日表 3549 行、行业表 100211 行，文件结构、哈希、字段还原和有效分母已按 manifest 验收。采集与使用边界见 [`p1_jq_input_collection_v2.md`](../docs/signals/p1_jq_input_collection_v2.md)。

V2 相对 V1 只修正 MA 比较容差；后续不得沿用 V1 的 MA 触发结果。改变股票池、日期范围、复权、行业映射、数值容差或字段口径时创建新版本，不覆盖旧目录。本地验收不能替代独立 JQ 短区间重跑。
